#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_DO_SAMPLE = False
FINAL_CHANNEL_PREFIX = "<|channel|>final<|message|>"
RETURN_TOKEN = "<|return|>"


def load_contracts_module():
    path = Path(__file__).with_name("v8_full_ft_contracts.py")
    spec = importlib.util.spec_from_file_location("v8_full_ft_contracts_validator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


CONTRACTS = load_contracts_module()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate GPT-OSS full fine-tuning outputs.")
    parser.add_argument(
        "--config",
        default=str((Path(__file__).resolve().parents[1] / "configs/gpt_oss_20b_seed_v8_round1_full_ft.json").resolve()),
        help="Path to JSON config.",
    )
    parser.add_argument("--model-path", help="Optional explicit final-export directory to validate.")
    parser.add_argument("--checkpoint-path", help="Optional explicit checkpoint directory to validate.")
    parser.add_argument("--report-path", help="Optional Markdown report path override.")
    parser.add_argument("--question-limit", type=int, help="Optional prompt count override.")
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config must be a JSON object: {path}")
    return data


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            records.append(row)
    return records


def resolve_prompt_source_path(cfg: Mapping[str, Any], prompt_source_path: str | Path | None = None) -> Path:
    candidate = prompt_source_path or cfg.get("prompt_source_path") or cfg.get("dataset_path")
    if not candidate:
        raise ValueError("prompt_source_path is required")
    return Path(str(candidate)).resolve()


def load_prompts(
    cfg: Mapping[str, Any],
    *,
    prompt_source_path: str | Path | None = None,
    limit: int | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    source_path = resolve_prompt_source_path(cfg, prompt_source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"prompt source path not found: {source_path}")

    if source_path.suffix == ".jsonl":
        records: Any = read_jsonl_records(source_path)
    else:
        records = json.loads(source_path.read_text(encoding="utf-8"))

    if not isinstance(records, list):
        raise ValueError(f"prompt source must be a JSON array or JSONL file: {source_path}")

    max_count = limit if limit is not None else int(cfg.get("sample_prompt_count", 5))
    prompts: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        messages = record.get("messages")
        if not isinstance(messages, list):
            continue
        prompt_messages = [message for message in messages if isinstance(message, dict) and message.get("role") != "assistant"]
        if not prompt_messages:
            continue
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") != "user":
                continue
            content = str(message.get("content", "")).strip()
            if content:
                prompts.append(
                    {
                        "prompt_text": content,
                        "prompt_messages": prompt_messages,
                    }
                )
                break
        if len(prompts) >= max_count:
            break

    if not prompts:
        raise ValueError(f"no prompts found in prompt source: {source_path}")
    return source_path, prompts


def sanitize_run_id(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "validation"


def derive_default_run_id(*, target_source: str, resolved_model_path: Path) -> str:
    identifier_parts = [sanitize_run_id(target_source)]
    model_name = sanitize_run_id(resolved_model_path.name)
    if model_name:
        identifier_parts.append(model_name)
    parent_name = sanitize_run_id(resolved_model_path.parent.name)
    if parent_name and parent_name != model_name:
        identifier_parts.append(parent_name)
    return "-".join(identifier_parts)


def resolve_report_path(
    config_path: Path,
    *,
    target_source: str,
    resolved_model_path: Path,
    report_path: str | Path | None = None,
) -> Path:
    if report_path:
        return Path(str(report_path)).resolve()
    root = config_path.resolve().parents[1]
    run_id = derive_default_run_id(
        target_source=target_source,
        resolved_model_path=resolved_model_path,
    )
    return (root / f"tests/log/v8_round1_full_ft_{run_id}_report.md").resolve()


def build_generation_settings() -> dict[str, Any]:
    return {
        "do_sample": DEFAULT_DO_SAMPLE,
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
    }


def build_prompt_text(tokenizer: Any, prompt_messages: list[dict[str, Any]]) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise RuntimeError("tokenizer does not support apply_chat_template required for GPT-OSS validation")
    rendered = apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return str(rendered) + FINAL_CHANNEL_PREFIX


def extract_final_text(decoded_text: str) -> str:
    text = str(decoded_text)
    if FINAL_CHANNEL_PREFIX in text:
        text = text.split(FINAL_CHANNEL_PREFIX, 1)[1]
    elif "<|channel|>final<|message|>" in text:
        text = text.split("<|channel|>final<|message|>", 1)[1]
    if RETURN_TOKEN in text:
        text = text.split(RETURN_TOKEN, 1)[0]
    if "<|start|>" in text:
        text = text.split("<|start|>", 1)[0]
    if "<|channel|>" in text:
        text = text.split("<|channel|>", 1)[0]
    return text.strip()


class FullFTValidationRuntime:
    def load_tokenizer(self, *, model_source: str, cache_dir: Path):
        from transformers import AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            cache_dir=str(cache_dir),
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_model(self, *, model_source: str, cache_dir: Path):
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM  # type: ignore

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            cache_dir=str(cache_dir),
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        return model

    def generate_final_answer(
        self,
        *,
        model,
        tokenizer,
        prompt_text: str,
        prompt_messages: list[dict[str, Any]] | None = None,
        generation_settings: dict[str, Any],
    ) -> dict[str, str]:
        import torch  # type: ignore

        rendered_prompt = (
            build_prompt_text(tokenizer, prompt_messages)
            if prompt_messages is not None
            else prompt_text
        )
        encoded = tokenizer(rendered_prompt, return_tensors="pt")
        model_device = getattr(model, "device", None)
        if model_device is not None:
            encoded = {key: value.to(model_device) for key, value in encoded.items()}

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=generation_settings["max_new_tokens"],
                do_sample=generation_settings["do_sample"],
                temperature=generation_settings["temperature"],
                top_p=generation_settings["top_p"],
                pad_token_id=getattr(tokenizer, "pad_token_id", None),
                eos_token_id=getattr(tokenizer, "eos_token_id", None),
            )

        input_length = encoded["input_ids"].shape[-1]
        new_tokens = generated[0, input_length:]
        raw_generation = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
        return {
            "raw_generation": raw_generation,
            "final_answer": extract_final_text(raw_generation),
        }


def render_markdown_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# V8 Full FT Validation Report",
        "",
        f"- Run Status: {result['status']}",
        f"- Config Path: {result['config_path']}",
        f"- Resolved Model Path: {result['resolved_model_path']}",
        f"- Target Source: {result['target_source']}",
        f"- Prompt Source Path: {result['prompt_source_path']}",
        "",
        "## Generation Settings",
        f"- do_sample: {result['generation_settings']['do_sample']}",
        f"- max_new_tokens: {result['generation_settings']['max_new_tokens']}",
        f"- temperature: {result['generation_settings']['temperature']}",
        f"- top_p: {result['generation_settings']['top_p']}",
        "",
    ]

    samples = result.get("samples", [])
    if samples:
        lines.append("## Samples")
        lines.append("")
        for index, sample in enumerate(samples, start=1):
            lines.extend(
                [
                    f"### Sample {index}",
                    "",
                    "**Prompt**",
                    "",
                    sample["prompt_text"],
                    "",
                    "**Raw Generation**",
                    "",
                    sample.get("raw_generation", "") or "(empty)",
                    "",
                    "**Generated Final Answer**",
                    "",
                    sample["final_answer"] or "(empty)",
                    "",
                ]
            )

    error = result.get("error")
    if error:
        lines.extend(
            [
                "## Error",
                "",
                str(error),
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def write_report(report_path: Path, result: Mapping[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown_report(result), encoding="utf-8")


def run_validation(
    *,
    config_path: str | Path,
    model_path: str | None = None,
    checkpoint_path: str | None = None,
    report_path: str | Path | None = None,
    prompt_source_path: str | Path | None = None,
    question_limit: int | None = None,
    runtime: Any | None = None,
) -> dict[str, Any]:
    config_path = Path(str(config_path)).resolve()
    cfg = CONTRACTS.validate_full_ft_config(read_json(config_path))
    target = CONTRACTS.resolve_validation_target(
        cfg,
        model_path=model_path,
        checkpoint_path=checkpoint_path,
    )
    resolved_model_path = Path(target["path"]).resolve()
    resolved_report_path = resolve_report_path(
        config_path,
        target_source=str(target["source"]),
        resolved_model_path=resolved_model_path,
        report_path=report_path,
    )
    resolved_prompt_source_path = resolve_prompt_source_path(cfg, prompt_source_path)
    generation_settings = build_generation_settings()
    runtime = runtime or FullFTValidationRuntime()

    result: dict[str, Any] = {
        "status": "failed",
        "exit_code": 1,
        "config_path": str(config_path),
        "resolved_model_path": str(resolved_model_path),
        "target_source": str(target["source"]),
        "prompt_source_path": str(resolved_prompt_source_path),
        "generation_settings": generation_settings,
        "samples": [],
        "report_path": str(resolved_report_path),
    }

    try:
        if not resolved_model_path.exists():
            raise FileNotFoundError(f"resolved model path not found: {resolved_model_path}")

        cache_dir = Path(str(cfg["cache_dir"])).resolve()
        prompt_path, prompts = load_prompts(cfg, prompt_source_path=prompt_source_path, limit=question_limit)
        result["prompt_source_path"] = str(prompt_path)

        tokenizer = runtime.load_tokenizer(model_source=str(resolved_model_path), cache_dir=cache_dir)
        model = runtime.load_model(model_source=str(resolved_model_path), cache_dir=cache_dir)

        generation_signature = inspect.signature(runtime.generate_final_answer)
        supports_prompt_messages = "prompt_messages" in generation_signature.parameters
        for prompt in prompts:
            prompt_text = str(prompt["prompt_text"]).strip()
            generation_kwargs = {
                "model": model,
                "tokenizer": tokenizer,
                "prompt_text": prompt_text,
                "generation_settings": generation_settings,
            }
            if supports_prompt_messages:
                generation_kwargs["prompt_messages"] = list(prompt["prompt_messages"])
            generated_output = runtime.generate_final_answer(**generation_kwargs)
            if isinstance(generated_output, Mapping):
                raw_generation = str(generated_output.get("raw_generation", "")).strip()
                final_answer = str(generated_output.get("final_answer", "")).strip()
            else:
                raw_generation = str(generated_output).strip()
                final_answer = raw_generation
            result["samples"].append(
                {
                    "prompt_text": prompt_text,
                    "raw_generation": raw_generation,
                    "final_answer": final_answer,
                }
            )

        if not any(sample["final_answer"] for sample in result["samples"]):
            raise ValueError("all sampled generations were empty")

        result["status"] = "success"
        result["exit_code"] = 0
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    write_report(resolved_report_path, result)
    return result


def main(argv: list[str] | None = None, *, runtime: Any | None = None) -> int:
    args = parse_args(argv)
    result = run_validation(
        config_path=args.config,
        model_path=args.model_path,
        checkpoint_path=args.checkpoint_path,
        report_path=args.report_path,
        question_limit=args.question_limit,
        runtime=runtime,
    )

    stream = sys.stdout if result["exit_code"] == 0 else sys.stderr
    print(f"validation status: {result['status']}", file=stream)
    print(f"report path: {result['report_path']}", file=stream)
    if result["exit_code"] != 0 and result.get("error"):
        print(result["error"], file=stream)
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
