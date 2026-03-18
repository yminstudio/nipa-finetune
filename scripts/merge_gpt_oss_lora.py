#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def load_base_model(model_name: str, cache_dir: Path, dtype_name: str):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        trust_remote_code=True,
        torch_dtype=resolve_dtype(dtype_name),
        low_cpu_mem_usage=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a GPT-OSS LoRA adapter into the base model.")
    parser.add_argument(
        "--config",
        default="/home/work/dev_data/fine-tuning/configs/gpt_oss_20b_seed_v2_all.json",
        help="Path to JSON config.",
    )
    parser.add_argument("--adapter-path", help="Optional override for LoRA adapter directory.")
    parser.add_argument("--merged-output-dir", help="Optional override for merged model output directory.")
    parser.add_argument("--report-path", help="Optional override for merge report path.")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Torch dtype for loading the base model. Default: bfloat16",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = read_json(config_path)

    cache_dir = Path(cfg["cache_dir"]).resolve()
    adapter_path = Path(args.adapter_path or cfg["output_dir"]).resolve()
    merged_output_dir = Path(
        args.merged_output_dir or cfg.get("merged_output_dir") or f"{cfg['output_dir']}-merged"
    ).resolve()
    report_path = Path(
        args.report_path or cfg.get("merge_report_path") or merged_output_dir / "merge_result.json"
    ).resolve()

    if not adapter_path.is_dir():
        raise FileNotFoundError(f"adapter path not found: {adapter_path}")

    merged_output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["model_name"]
    tokenizer_source = str(adapter_path if (adapter_path / "tokenizer_config.json").exists() else model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        cache_dir=str(cache_dir),
        trust_remote_code=True,
    )
    base_model = load_base_model(model_name, cache_dir, args.dtype)
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    merged_model = peft_model.merge_and_unload()
    merged_model.config.use_cache = True

    merged_model.save_pretrained(
        str(merged_output_dir),
        safe_serialization=True,
        max_shard_size="10GB",
    )
    tokenizer.save_pretrained(str(merged_output_dir))

    report = {
        "status": "success",
        "model_name": model_name,
        "adapter_path": str(adapter_path),
        "merged_output_dir": str(merged_output_dir),
        "dtype": args.dtype,
        "config_path": str(config_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
