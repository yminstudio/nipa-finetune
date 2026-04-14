from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


DATASET_OUTPUT_RELATIVE_PATH = "llm_datasets/seed_v8/seed_v8_round1_full_ft.jsonl"
MODEL_OUTPUT_RELATIVE_DIR = "llm_model_full/gpt-oss-20b-seed-v8-round1-full-ft"
TRAIN_REPORT_RELATIVE_PATH = f"{MODEL_OUTPUT_RELATIVE_DIR}/train_result.json"
FINAL_EXPORT_RELATIVE_DIR = f"{MODEL_OUTPUT_RELATIVE_DIR}/final-export"
DEEPSPEED_CONFIG_RELATIVE_PATH = "configs/deepspeed/gpt_oss_20b_zero3_bf16.json"
REQUIRED_CONFIG_KEYS = {
    "bf16",
    "cache_dir",
    "dataset_path",
    "deepspeed_config_path",
    "gradient_accumulation_steps",
    "gradient_checkpointing",
    "learning_rate",
    "logging_steps",
    "max_length",
    "max_steps",
    "model_name",
    "output_dir",
    "per_device_train_batch_size",
    "prompt_source_path",
    "report_path",
    "resume_mode",
    "sample_prompt_count",
    "save_steps",
    "save_total_limit",
    "seed",
}
ALLOWED_RESUME_MODES = {
    "fail",
    "resume_latest",
    "resume_from_path",
    "overwrite_empty_only",
}


def _stringify(path: Path) -> str:
    return str(path.resolve())


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_positive_int(cfg: Mapping[str, Any], key: str) -> None:
    value = cfg.get(key)
    if not _is_plain_int(value) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")


def _require_int(cfg: Mapping[str, Any], key: str) -> int:
    value = cfg.get(key)
    if not _is_plain_int(value):
        raise ValueError(f"{key} must be an integer")
    return value


def build_default_config(root: Path | str) -> dict[str, Any]:
    base = Path(root)
    return {
        "model_name": "unsloth/gpt-oss-20b-BF16",
        "cache_dir": _stringify(base / ".cache" / "huggingface"),
        "dataset_path": _stringify(base / DATASET_OUTPUT_RELATIVE_PATH),
        "output_dir": _stringify(base / MODEL_OUTPUT_RELATIVE_DIR),
        "report_path": _stringify(base / TRAIN_REPORT_RELATIVE_PATH),
        "max_length": 4096,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_steps": 1000,
        "learning_rate": 1e-5,
        "logging_steps": 10,
        "save_steps": 100,
        "save_total_limit": 2,
        "seed": 42,
        "bf16": True,
        "gradient_checkpointing": True,
        "deepspeed_config_path": _stringify(base / DEEPSPEED_CONFIG_RELATIVE_PATH),
        "prompt_source_path": _stringify(base / DATASET_OUTPUT_RELATIVE_PATH),
        "sample_prompt_count": 5,
        "resume_mode": "fail",
    }


def validate_full_ft_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    missing = sorted(REQUIRED_CONFIG_KEYS - set(cfg.keys()))
    if missing:
        raise ValueError(f"missing required config keys: {', '.join(missing)}")

    resume_mode = cfg["resume_mode"]
    if resume_mode not in ALLOWED_RESUME_MODES:
        raise ValueError(f"unsupported resume_mode: {resume_mode}")
    if resume_mode == "resume_from_path" and not cfg.get("resume_from_checkpoint"):
        raise ValueError("resume_from_checkpoint is required when resume_mode=resume_from_path")

    if cfg.get("validation_model_path") and cfg.get("validation_checkpoint_path"):
        raise ValueError("validation_model_path and validation_checkpoint_path are mutually exclusive")

    _require_positive_int(cfg, "per_device_train_batch_size")
    _require_positive_int(cfg, "gradient_accumulation_steps")
    _require_positive_int(cfg, "sample_prompt_count")

    save_steps = _require_int(cfg, "save_steps")
    max_steps = _require_int(cfg, "max_steps")
    if save_steps > max_steps:
        raise ValueError("save_steps must be less than or equal to max_steps")

    return dict(cfg)


def resolve_validation_target(
    cfg: Mapping[str, Any],
    *,
    model_path: str | None = None,
    checkpoint_path: str | None = None,
    adapter_path: str | None = None,
) -> dict[str, str]:
    if adapter_path:
        raise ValueError("adapter_path is not supported for full fine-tuning validation")
    if model_path and checkpoint_path:
        raise ValueError("model_path and checkpoint_path are mutually exclusive")

    if model_path:
        return {"path": model_path, "source": "model_path"}
    if checkpoint_path:
        return {"path": checkpoint_path, "source": "checkpoint_path"}

    if cfg.get("validation_model_path"):
        return {"path": str(cfg["validation_model_path"]), "source": "validation_model_path"}
    if cfg.get("validation_checkpoint_path"):
        return {"path": str(cfg["validation_checkpoint_path"]), "source": "validation_checkpoint_path"}

    output_dir = Path(str(cfg["output_dir"]))
    return {"path": str((output_dir / "final-export").resolve()), "source": "final_export"}
