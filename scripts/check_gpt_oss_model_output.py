#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

FINAL_CHANNEL_PREFIX = "<|channel|>final<|message|>"
RETURN_TOKEN = "<|return|>"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = name.strip().lower()
    if key not in mapping:
        raise ValueError(f"unsupported dtype: {name}")
    return mapping[key]


def load_validation_questions(cfg: dict[str, Any], limit: int) -> list[str]:
    questions = cfg.get("validation_questions")
    if isinstance(questions, list):
        normalized = [item.strip() for item in questions if isinstance(item, str) and item.strip()]
        if normalized:
            return normalized[:limit]

    prompt_source_path = Path(cfg["prompt_source_path"]).resolve()
    if prompt_source_path.suffix == ".jsonl":
        records = read_jsonl_records(prompt_source_path)
    else:
        records = json.loads(prompt_source_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"prompt source must be a JSON array: {prompt_source_path}")

    prompts: list[str] = []
    for record in records:
        messages = record.get("messages")
        if not isinstance(messages, list):
            continue
        user_message = next(
            (msg.get("content", "").strip() for msg in messages if msg.get("role") == "user"),
            "",
        )
        if user_message:
            prompts.append(user_message)
        if len(prompts) >= limit:
            break

    if not prompts:
        raise ValueError("no validation questions found in config or prompt source")
    return prompts


def build_prompt_inputs(tokenizer, question: str) -> dict[str, torch.Tensor]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_text = rendered + FINAL_CHANNEL_PREFIX
    return tokenizer(prompt_text, return_tensors="pt")


def extract_final_text(decoded_text: str) -> str:
    text = decoded_text
    if FINAL_CHANNEL_PREFIX in text:
        text = text.split(FINAL_CHANNEL_PREFIX, 1)[1]
    if RETURN_TOKEN in text:
        text = text.split(RETURN_TOKEN, 1)[0]
    if "<|start|>" in text:
        text = text.split("<|start|>", 1)[0]
    if "<|channel|>" in text:
        text = text.split("<|channel|>", 1)[0]
    return text.strip()


def load_model(model_source: str, cache_dir: Path, dtype_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        cache_dir=str(cache_dir),
        trust_remote_code=True,
        torch_dtype=resolve_dtype(dtype_name),
        low_cpu_mem_usage=True,
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run direct GPT-OSS generation checks.")
    parser.add_argument(
        "--config",
        default="/home/work/dev_data/fine-tuning/configs/gpt_oss_20b_seed_v2_all.json",
        help="Path to JSON config.",
    )
    parser.add_argument("--model-path", help="Merged model path. If omitted, config.merged_output_dir is used.")
    parser.add_argument("--adapter-path", help="Optional LoRA adapter path for adapter-mode validation.")
    parser.add_argument("--report-path", help="Optional override for validation report path.")
    parser.add_argument("--dtype", default="bfloat16", help="Torch dtype. Default: bfloat16")
    parser.add_argument("--question-limit", type=int, default=5, help="Maximum questions to evaluate.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation cap per question.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = read_json(config_path)

    cache_dir = Path(cfg["cache_dir"]).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    adapter_path = Path(args.adapter_path).resolve() if args.adapter_path else None
    if adapter_path:
        model_path = args.model_path or cfg["model_name"]
    else:
        model_path = args.model_path or cfg.get("merged_output_dir") or cfg["model_name"]
    model_source = str(Path(model_path).resolve()) if Path(model_path).exists() else str(model_path)
    report_path = Path(
        args.report_path or cfg.get("validation_report_path") or "/tmp/gpt_oss_validation_report.json"
    ).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer_source = str(adapter_path) if adapter_path else model_source
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        cache_dir=str(cache_dir),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = load_model(model_source, cache_dir, args.dtype)
    if adapter_path:
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"adapter path not found: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        evaluation_mode = "adapter"
    else:
        model = base_model
        evaluation_mode = "merged_or_base"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    questions = load_validation_questions(cfg, args.question_limit)
    generations: list[dict[str, str]] = []
    for question in questions:
        encoded = build_prompt_inputs(tokenizer, question)
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[0, input_ids.shape[-1] :]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        generations.append(
            {
                "question": question,
                "generation_raw": raw_text.strip(),
                "generation_final": extract_final_text(raw_text),
            }
        )

    report = {
        "status": "success",
        "evaluation_mode": evaluation_mode,
        "model_source": model_source,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "config_path": str(config_path),
        "question_limit": args.question_limit,
        "max_new_tokens": args.max_new_tokens,
        "generations": generations,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
