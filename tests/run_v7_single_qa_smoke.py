#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = "당신은 제로인 펀드평가 방법론에 근거해 답변하는 도메인 어시스턴트입니다."


def load_module(relative_path: str, module_name: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def build_smoke_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    user_messages = [message for message in messages if message.get("role") == "user"]
    return [{"role": "system", "content": SYSTEM_PROMPT}, *user_messages]


def build_prompt_inputs(tokenizer, helper, messages: list[dict[str, str]]) -> dict[str, torch.Tensor]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_text = rendered + helper.FINAL_CHANNEL_PREFIX
    return tokenizer(prompt_text, return_tensors="pt")


def render_markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = [
        "# GPT-OSS Smoke Report",
        "",
        "## Summary",
        f"- status: `{report['status']}`",
        f"- model_name: `{report['model_name']}`",
        f"- adapter_path: `{report.get('adapter_path', '')}`",
        f"- prompt_source_path: `{report.get('prompt_source_path', '')}`",
        f"- max_new_tokens: `{report.get('max_new_tokens', '')}`",
        "",
        "## System Prompt",
        "",
        report.get("system_prompt", ""),
        "",
        "## Sample Inference",
        "",
    ]

    for item in report.get("sample_inference", []):
        sample_id = item.get("id") or "sample"
        lines.extend(
            [
                f"### {sample_id}",
                "",
                "**Question**",
                "",
                item["prompt"],
                "",
                "**Answer**",
                "",
                item["generation_final"],
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v7 adapter smoke inference and write a markdown log.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "tests/smoke_v7_single_qa_config.json"),
        help="Path to smoke config JSON.",
    )
    args = parser.parse_args()

    cfg = read_json(Path(args.config).resolve())
    helper = load_module("scripts/check_gpt_oss_model_output.py", "check_gpt_oss_model_output")

    cache_dir = Path(cfg["cache_dir"]).resolve()
    adapter_path = Path(cfg["adapter_path"]).resolve()
    prompt_source_path = Path(cfg["prompt_source_path"]).resolve()
    report_path = Path(cfg["report_path"]).resolve()
    max_new_tokens = int(cfg.get("max_new_tokens", 384))
    dtype_name = str(cfg.get("dtype", "bfloat16"))

    cache_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path),
        cache_dir=str(cache_dir),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = helper.load_model(cfg["model_name"], cache_dir, dtype_name)
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    prompt_records = read_jsonl_records(prompt_source_path)
    samples: list[dict[str, str]] = []
    for record in prompt_records:
        messages = build_smoke_messages(record["messages"])
        question = next(msg["content"] for msg in messages if msg["role"] == "user")
        encoded = build_prompt_inputs(tokenizer, helper, messages)
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[0, input_ids.shape[-1] :]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        samples.append(
            {
                "id": record.get("id", ""),
                "prompt": question,
                "generation_raw": raw_text.strip(),
                "generation_final": helper.extract_final_text(raw_text),
            }
        )

    report = {
        "status": "success",
        "model_name": cfg["model_name"],
        "adapter_path": str(adapter_path),
        "prompt_source_path": str(prompt_source_path),
        "system_prompt": SYSTEM_PROMPT,
        "max_new_tokens": max_new_tokens,
        "sample_inference": samples,
    }
    markdown = render_markdown_report(report)
    report_path.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
