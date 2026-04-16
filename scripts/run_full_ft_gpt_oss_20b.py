#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

GIB = 1024**3
TIB = 1024**4
MIN_GPU_COUNT = 2
MIN_GPU_VRAM_GIB = 40.0
MIN_DISK_FREE_BYTES = TIB
REQUIRED_PACKAGES = ("torch", "transformers", "accelerate", "deepspeed")
TRAINER_STATE_FILENAME = "trainer_state.json"
RESUME_ARTIFACT_METADATA_FILENAME = "resume-artifact.json"
POST_TRAINING_PHASE_POLL_SECONDS = 1.0
POST_TRAINING_PHASE_TIMEOUT_SECONDS = 6 * 60 * 60


def load_contracts_module():
    path = Path(__file__).with_name("v8_full_ft_contracts.py")
    spec = importlib.util.spec_from_file_location("v8_full_ft_contracts_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_validation_module():
    path = Path(__file__).with_name("check_gpt_oss_full_ft_output.py")
    spec = importlib.util.spec_from_file_location("check_gpt_oss_full_ft_output_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


CONTRACTS = load_contracts_module()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT-OSS 20B full fine-tuning.")
    parser.add_argument(
        "--config",
        default=str((Path(__file__).resolve().parents[1] / "configs/gpt_oss_20b_seed_v8_round1_full_ft.json").resolve()),
        help="Path to JSON config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, inputs, readiness, and report writing without starting training.",
    )
    parser.add_argument(
        "--orchestrate",
        action="store_true",
        help="Run full pipeline: training subprocess (torchrun) then validation subprocess.",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=None,
        help="GPU count for torchrun in orchestrate mode (auto-detected if omitted).",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="In orchestrate mode, skip the post-training validation subprocess.",
    )
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config must be a JSON object: {path}")
    return data


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_and_validate_config(config_path: Path) -> dict[str, Any]:
    cfg = read_json(config_path)
    return CONTRACTS.validate_full_ft_config(cfg)


def ensure_existing_file(path: Path, *, label: str, require_non_empty: bool = False) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.exists():
        raise ValueError(f"{label} does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{label} must be a file: {resolved}")

    size_bytes = resolved.stat().st_size
    if require_non_empty and size_bytes <= 0:
        raise ValueError(f"{label} is empty: {resolved}")

    return {
        "path": str(resolved),
        "exists": True,
        "size_bytes": size_bytes,
    }


def discover_checkpoints(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    checkpoints = [item for item in output_dir.iterdir() if item.is_dir() and item.name.startswith("checkpoint-")]
    return sorted(checkpoints, key=checkpoint_sort_key)


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match is not None:
        return (int(match.group(1)), path.name)
    return (-1, path.name)


def has_final_export(output_dir: Path) -> bool:
    return (output_dir / "final-export").is_dir()


def checkpoint_contains_model_artifacts(checkpoint_dir: Path) -> bool:
    return any(path.is_file() for path in checkpoint_dir.glob("*.bin")) or any(
        path.is_file() for path in checkpoint_dir.glob("*.safetensors")
    )


def checkpoint_contains_native_resume_artifacts(checkpoint_dir: Path) -> bool:
    trainer_resume_files = (
        checkpoint_dir / "optimizer.pt",
        checkpoint_dir / "scheduler.pt",
    )
    if all(path.is_file() for path in trainer_resume_files):
        return True

    latest_path = checkpoint_dir / "latest"
    zero_to_fp32_path = checkpoint_dir / "zero_to_fp32.py"
    if latest_path.is_file() or zero_to_fp32_path.is_file():
        return True

    for child in checkpoint_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("global_step"):
            continue
        has_model_states = any(path.is_file() for path in child.glob("*model_states.pt"))
        has_optim_states = any(path.is_file() for path in child.glob("*optim_states.pt"))
        if has_model_states and has_optim_states:
            return True

    return False


def resolve_resume_checkpoint_metadata(checkpoint_dir: Path) -> dict[str, Any] | None:
    if not checkpoint_dir.is_dir():
        return None
    checkpoint_step, _ = checkpoint_sort_key(checkpoint_dir)
    if checkpoint_step < 0:
        return None

    trainer_state_path = checkpoint_dir / TRAINER_STATE_FILENAME
    if not trainer_state_path.is_file():
        return None
    if not checkpoint_contains_model_artifacts(checkpoint_dir):
        return None

    trainer_state = read_json(trainer_state_path)
    global_step = trainer_state.get("global_step", checkpoint_step)
    if not isinstance(global_step, int) or global_step <= 0:
        global_step = checkpoint_step

    metadata_path = checkpoint_dir / RESUME_ARTIFACT_METADATA_FILENAME
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        loaded = read_json(metadata_path)
        if isinstance(loaded, dict):
            metadata = loaded

    if metadata and not bool(metadata.get("resume_capable", False)):
        return None
    if metadata.get("checkpoint_kind") == "export_only":
        return None
    if not checkpoint_contains_native_resume_artifacts(checkpoint_dir):
        return None

    return {
        "path": str(checkpoint_dir.resolve()),
        "step": global_step,
        "trainer_state_path": str(trainer_state_path.resolve()),
        "metadata_path": str(metadata_path.resolve()) if metadata_path.is_file() else None,
        "resume_capable": bool(metadata.get("resume_capable", True)),
    }


def discover_resume_checkpoints(output_dir: Path) -> list[dict[str, Any]]:
    checkpoints: list[dict[str, Any]] = []
    for checkpoint_dir in discover_checkpoints(output_dir):
        metadata = resolve_resume_checkpoint_metadata(checkpoint_dir)
        if metadata is not None:
            checkpoints.append(metadata)
    return sorted(checkpoints, key=lambda item: (int(item["step"]), str(item["path"])))


def resolve_resume_behavior(cfg: Mapping[str, Any]) -> dict[str, Any]:
    output_dir = Path(str(cfg["output_dir"])).resolve()
    checkpoints = discover_checkpoints(output_dir)
    resume_checkpoints = discover_resume_checkpoints(output_dir)
    final_export_exists = has_final_export(output_dir)
    has_artifacts = bool(checkpoints or final_export_exists)
    mode = str(cfg["resume_mode"])

    if mode == "fail":
        if has_artifacts:
            raise ValueError("resume_mode=fail does not allow existing checkpoints or final-export artifacts")
        return {"mode": mode, "action": "start_fresh", "checkpoint_path": None}

    if mode == "resume_latest":
        if not resume_checkpoints:
            raise ValueError("resume_mode=resume_latest requires at least one resume-capable checkpoint")
        latest = resume_checkpoints[-1]
        return {
            "mode": mode,
            "action": "resume",
            "checkpoint_path": str(latest["path"]),
            "checkpoint_step": int(latest["step"]),
            "trainer_state_path": str(latest["trainer_state_path"]),
        }

    if mode == "resume_from_path":
        checkpoint_path = Path(str(cfg["resume_from_checkpoint"])).resolve()
        if not checkpoint_path.exists():
            raise ValueError(f"resume_from_checkpoint does not exist: {checkpoint_path}")
        if not checkpoint_path.is_dir():
            raise ValueError(f"resume_from_checkpoint must be a directory: {checkpoint_path}")
        if checkpoint_sort_key(checkpoint_path)[0] < 0:
            raise ValueError("resume_from_checkpoint must follow the checkpoint-* artifact naming contract")
        checkpoint_metadata = resolve_resume_checkpoint_metadata(checkpoint_path)
        if checkpoint_metadata is None:
            raise ValueError("resume_from_checkpoint must point to a resume-capable checkpoint artifact")
        return {
            "mode": mode,
            "action": "resume",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_step": int(checkpoint_metadata["step"]),
            "trainer_state_path": str(checkpoint_metadata["trainer_state_path"]),
        }

    if mode == "overwrite_empty_only":
        if has_artifacts:
            raise ValueError("overwrite_empty_only requires output_dir without checkpoints or final-export")
        return {"mode": mode, "action": "start_fresh", "checkpoint_path": None}

    raise ValueError(f"unsupported resume_mode: {mode}")


def collect_package_versions(package_names: tuple[str, ...] = REQUIRED_PACKAGES) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package_name in package_names:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = "missing"
    return versions


def collect_package_imports(package_names: tuple[str, ...] = REQUIRED_PACKAGES) -> dict[str, dict[str, Any]]:
    imports: dict[str, dict[str, Any]] = {}
    for package_name in package_names:
        try:
            importlib.import_module(package_name)
            imports[package_name] = {"importable": True}
        except Exception as exc:
            imports[package_name] = {"importable": False, "error": f"{type(exc).__name__}: {exc}"}
    return imports


def evaluate_package_compatibility(
    package_versions: Mapping[str, Any],
    package_imports: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    packages: dict[str, Any] = {}
    missing_packages: list[str] = []
    import_failed_packages: list[str] = []
    package_imports = package_imports or {}

    for package_name in REQUIRED_PACKAGES:
        raw_version = package_versions.get(package_name, "missing")
        version = str(raw_version).strip()
        import_info = dict(package_imports.get(package_name, {}))
        importable = import_info.get("importable")
        import_error = import_info.get("error")

        if not version or version == "missing":
            packages[package_name] = {
                "present": False,
                "version": "missing",
                "status": "missing",
                "compatibility": "blocked",
                "importable": bool(importable) if importable is not None else False,
            }
            if import_error:
                packages[package_name]["import_error"] = str(import_error)
            missing_packages.append(package_name)
            continue

        if importable is False:
            packages[package_name] = {
                "present": True,
                "version": version,
                "status": "import_failed",
                "compatibility": "blocked",
                "importable": False,
                "import_error": str(import_error or "import failed"),
            }
            import_failed_packages.append(package_name)
            continue

        packages[package_name] = {
            "present": True,
            "version": version,
            "status": "ok",
            "compatibility": "ok",
            "importable": True if importable is None else bool(importable),
        }

    compatibility: dict[str, Any] = {"status": "ok", "packages": packages}
    if missing_packages:
        compatibility["status"] = "blocked"
        compatibility["blocking_reason"] = f"missing required packages: {', '.join(missing_packages)}"
    elif import_failed_packages:
        compatibility["status"] = "blocked"
        compatibility["blocking_reason"] = f"packages installed but not importable: {', '.join(import_failed_packages)}"
    return compatibility


def collect_gpu_facts() -> tuple[int, list[float], bool]:
    try:
        import torch  # type: ignore
    except Exception:
        return 0, [], False

    if not torch.cuda.is_available():
        return 0, [], False

    gpu_count = int(torch.cuda.device_count())
    gpu_vram_gib: list[float] = []
    for index in range(gpu_count):
        props = torch.cuda.get_device_properties(index)
        gpu_vram_gib.append(round(float(props.total_memory) / GIB, 2))

    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    return gpu_count, gpu_vram_gib, bf16_supported


def collect_disk_free_bytes(output_dir: Path) -> int:
    target = output_dir.resolve()
    while not target.exists() and target.parent != target:
        target = target.parent
    return int(shutil.disk_usage(target).free)


def collect_readiness_facts(output_dir: Path) -> dict[str, Any]:
    gpu_count, gpu_vram_gib, bf16_supported = collect_gpu_facts()
    return {
        "gpu_count": gpu_count,
        "gpu_vram_gib": gpu_vram_gib,
        "disk_free_bytes": collect_disk_free_bytes(output_dir),
        "bf16_supported": bf16_supported,
        "package_versions": collect_package_versions(),
        "package_imports": collect_package_imports(),
    }


def evaluate_readiness_gate(facts: Mapping[str, Any]) -> dict[str, Any]:
    gpu_count = int(facts.get("gpu_count", 0))
    gpu_vram_gib = [float(value) for value in facts.get("gpu_vram_gib", [])]
    disk_free_bytes = int(facts.get("disk_free_bytes", 0))
    bf16_supported = bool(facts.get("bf16_supported", False))
    package_versions = {str(key): str(value) for key, value in dict(facts.get("package_versions", {})).items()}
    package_imports = dict(facts.get("package_imports", {}))
    package_compatibility = evaluate_package_compatibility(package_versions, package_imports)

    report: dict[str, Any] = {
        "status": "ready",
        "gpu_count": gpu_count,
        "gpu_vram_gib": gpu_vram_gib,
        "disk_free_bytes": disk_free_bytes,
        "bf16_supported": bf16_supported,
        "package_versions": package_versions,
        "package_compatibility": package_compatibility,
        "thresholds": {
            "min_gpu_count": MIN_GPU_COUNT,
            "min_gpu_vram_gib": MIN_GPU_VRAM_GIB,
            "min_disk_free_bytes": MIN_DISK_FREE_BYTES,
            "required_packages": list(REQUIRED_PACKAGES),
        },
    }

    if gpu_count < MIN_GPU_COUNT:
        report["status"] = "blocked"
        report["blocking_check"] = "gpu_count"
        report["blocking_reason"] = f"need at least {MIN_GPU_COUNT} visible GPUs"
        return report

    if len(gpu_vram_gib) < gpu_count or any(value < MIN_GPU_VRAM_GIB for value in gpu_vram_gib[:gpu_count]):
        report["status"] = "blocked"
        report["blocking_check"] = "gpu_vram_gib"
        report["blocking_reason"] = f"each visible GPU must provide at least {MIN_GPU_VRAM_GIB:.0f} GiB VRAM"
        return report

    if disk_free_bytes < MIN_DISK_FREE_BYTES:
        report["status"] = "blocked"
        report["blocking_check"] = "disk_free_bytes"
        report["blocking_reason"] = "insufficient free disk for full fine-tuning checkpoints"
        return report

    if not bf16_supported:
        report["status"] = "blocked"
        report["blocking_check"] = "bf16_supported"
        report["blocking_reason"] = "bf16 support is required"
        return report

    if package_compatibility["status"] != "ok":
        report["status"] = "blocked"
        report["blocking_check"] = "package_compatibility"
        report["blocking_reason"] = str(package_compatibility.get("blocking_reason", "package compatibility check failed"))
        return report

    return report


def determine_failure_report_path(config_path: Path) -> Path:
    try:
        raw_cfg = read_json(config_path)
    except Exception:
        return (config_path.parent / "train_result.json").resolve()

    report_path_value = raw_cfg.get("report_path")
    if isinstance(report_path_value, str) and report_path_value.strip():
        return Path(report_path_value).resolve()
    return (config_path.parent / "train_result.json").resolve()


def should_write_side_effects(distributed_info: Mapping[str, Any] | None) -> bool:
    if not distributed_info:
        return True
    return int(distributed_info.get("rank", 0)) == 0


def distributed_world_size(distributed_info: Mapping[str, Any] | None) -> int:
    if not distributed_info:
        return 1
    return max(1, int(distributed_info.get("world_size", 1)))


def write_report_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    distributed_info: Mapping[str, Any] | None = None,
) -> None:
    if should_write_side_effects(distributed_info):
        write_json(path, payload)


def resolve_distributed_init_path(cfg: Mapping[str, Any]) -> Path:
    return (Path(str(cfg["output_dir"])).resolve() / "distributed-init").resolve()


def resolve_sync_status_dir(cfg: Mapping[str, Any]) -> Path:
    return (Path(str(cfg["output_dir"])).resolve() / ".runner-sync").resolve()


def normalize_phase_name(phase_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", phase_name.lower()).strip("-") or "phase"


def resolve_sync_status_path(cfg: Mapping[str, Any], phase_name: str) -> Path:
    normalized = normalize_phase_name(phase_name)
    return (resolve_sync_status_dir(cfg) / f"{normalized}.json").resolve()


def resolve_rank_sync_status_path(cfg: Mapping[str, Any], phase_name: str, rank: int) -> Path:
    normalized = normalize_phase_name(phase_name)
    return (resolve_sync_status_dir(cfg) / f"{normalized}-rank-{rank}.json").resolve()


def resolve_checkpoint_output_path(cfg: Mapping[str, Any], global_step: int | None = None) -> Path:
    step = int(global_step) if global_step is not None else int(cfg["max_steps"])
    if step <= 0:
        step = int(cfg["save_steps"])
    return (Path(str(cfg["output_dir"])).resolve() / f"checkpoint-{step}").resolve()


def resolve_final_export_path(cfg: Mapping[str, Any]) -> Path:
    return (Path(str(cfg["output_dir"])).resolve() / "final-export").resolve()


def resolve_post_training_resume_checkpoint(output_dir: Path, preferred_path: Path | None = None) -> Path:
    if preferred_path is not None:
        preferred_metadata = resolve_resume_checkpoint_metadata(preferred_path)
        if preferred_metadata is not None:
            return Path(str(preferred_metadata["path"])).resolve()

    resume_checkpoints = discover_resume_checkpoints(output_dir)
    if resume_checkpoints:
        return Path(str(resume_checkpoints[-1]["path"])).resolve()

    raise RuntimeError("training did not produce a resume-capable checkpoint under checkpoint-*")


def build_artifact_paths(
    cfg: Mapping[str, Any] | None,
    *,
    distributed_init_path: Path | None = None,
    checkpoint_path: Path | None = None,
    final_export_path: Path | None = None,
) -> dict[str, Any]:
    if cfg is None:
        return {}

    artifacts: dict[str, Any] = {
        "output_dir": str(Path(str(cfg["output_dir"])).resolve()),
        "report_path": str(Path(str(cfg["report_path"])).resolve()),
    }
    if distributed_init_path is not None:
        artifacts["distributed_init_path"] = str(distributed_init_path.resolve())
    if checkpoint_path is not None:
        artifacts["checkpoint_path"] = str(checkpoint_path.resolve())
    if final_export_path is not None:
        artifacts["final_export_path"] = str(final_export_path.resolve())
    return artifacts


def render_messages_with_chat_template(tokenizer: Any, messages: Any) -> str:
    if not isinstance(messages, list) or not messages:
        raise ValueError("dataset record must include a non-empty messages list")
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise RuntimeError("tokenizer does not support apply_chat_template required for GPT-OSS full fine-tuning")
    rendered = apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    text = str(rendered).strip()
    if not text:
        raise ValueError("chat template rendering produced empty training text")
    return text


def validate_training_artifacts(checkpoint_dir: Path, final_export_dir: Path) -> None:
    if not checkpoint_dir.name.startswith("checkpoint-"):
        raise ValueError("checkpoint output must follow checkpoint-* naming")
    if not checkpoint_dir.is_dir():
        raise ValueError(f"checkpoint directory missing: {checkpoint_dir}")
    if not final_export_dir.is_dir():
        raise ValueError(f"final-export directory missing: {final_export_dir}")


def parse_distributed_env(env: Mapping[str, str] | None = None) -> dict[str, int]:
    env = env or os.environ

    def parse_int(name: str, default: int) -> int:
        raw_value = env.get(name)
        if raw_value is None or not str(raw_value).strip():
            return default
        try:
            return int(str(raw_value).strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer when set") from exc

    rank = parse_int("RANK", 0)
    world_size = parse_int("WORLD_SIZE", 1)
    local_rank = parse_int("LOCAL_RANK", 0)

    if rank < 0:
        raise ValueError("RANK must be greater than or equal to 0")
    if world_size <= 0:
        raise ValueError("WORLD_SIZE must be greater than 0")
    if local_rank < 0:
        raise ValueError("LOCAL_RANK must be greater than or equal to 0")
    if rank >= world_size:
        raise ValueError("RANK must be less than WORLD_SIZE")

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
    }


def use_env_rendezvous(env_info: Mapping[str, int], env: Mapping[str, str] | None = None) -> bool:
    env = env or os.environ
    if int(env_info.get("world_size", 1)) <= 1:
        return False
    required = ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    return all(env.get(name) is not None and str(env.get(name, "")).strip() for name in required)


def resolve_training_global_step(train_output: Any, cfg: Mapping[str, Any]) -> int:
    if isinstance(train_output, Mapping):
        raw_step = train_output.get("global_step")
        if isinstance(raw_step, int) and raw_step > 0:
            return raw_step
    raw_step = getattr(train_output, "global_step", None)
    if isinstance(raw_step, int) and raw_step > 0:
        return raw_step
    return int(cfg["max_steps"])


class FullFTRuntime:
    def __init__(self) -> None:
        self._trainer: Any | None = None
        self._tokenizer: Any | None = None
        self._distributed_initialized_by_runner = False

    def initialize_distributed(self, *, output_dir: Path, init_path: Path) -> dict[str, Any]:
        del output_dir
        self._distributed_initialized_by_runner = False
        torch = importlib.import_module("torch")
        distributed = getattr(torch, "distributed", None)
        if distributed is None or not bool(getattr(distributed, "is_available", lambda: False)()):
            raise RuntimeError("torch.distributed is not available")

        already_initialized = bool(getattr(distributed, "is_initialized", lambda: False)())
        backend = "nccl" if bool(getattr(torch.cuda, "is_available", lambda: False)()) else "gloo"
        env_info = parse_distributed_env()
        rank = int(env_info["rank"])
        world_size = int(env_info["world_size"])
        local_rank = int(env_info["local_rank"])
        launcher_managed = use_env_rendezvous(env_info)
        if launcher_managed:
            init_method = "env://"
        else:
            init_path.parent.mkdir(parents=True, exist_ok=True)
            if init_path.exists():
                init_path.unlink()
            init_method = f"file://{init_path.resolve()}"

        if not already_initialized:
            set_device = getattr(torch.cuda, "set_device", None)
            if backend == "nccl" and callable(set_device):
                set_device(local_rank)
            distributed.init_process_group(
                backend=backend,
                init_method=init_method,
                rank=rank,
                world_size=world_size,
            )
            self._distributed_initialized_by_runner = True
        else:
            get_rank = getattr(distributed, "get_rank", None)
            get_world_size = getattr(distributed, "get_world_size", None)
            if callable(get_rank):
                rank = int(get_rank())
            if callable(get_world_size):
                world_size = int(get_world_size())

        return {
            "backend": backend,
            "world_size": world_size,
            "rank": rank,
            "local_rank": local_rank,
            "init_method": init_method,
            "is_writer": rank == 0,
            "launcher_managed": launcher_managed,
            "initialized": True,
            "initialized_by_runner": self._distributed_initialized_by_runner,
        }

    def finalize_distributed(self, distributed_info: Mapping[str, Any] | None) -> None:
        if not distributed_info or not distributed_info.get("initialized_by_runner"):
            return
        torch = importlib.import_module("torch")
        distributed = getattr(torch, "distributed", None)
        if distributed is None or not bool(getattr(distributed, "is_available", lambda: False)()):
            return
        if bool(getattr(distributed, "is_initialized", lambda: False)()):
            distributed.destroy_process_group()

    def barrier(self, distributed_info: Mapping[str, Any] | None) -> None:
        if distributed_world_size(distributed_info) <= 1:
            return
        torch = importlib.import_module("torch")
        distributed = getattr(torch, "distributed", None)
        if distributed is None or not bool(getattr(distributed, "is_available", lambda: False)()):
            return
        if not bool(getattr(distributed, "is_initialized", lambda: False)()):
            return
        barrier = getattr(distributed, "barrier", None)
        if callable(barrier):
            barrier()

    def load_tokenizer(self, model_name: str, *, cache_dir: Path):
        transformers = importlib.import_module("transformers")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cache_dir.resolve()),
            trust_remote_code=True,
        )
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        self._tokenizer = tokenizer
        return tokenizer

    def load_model(self, model_name: str, *, cache_dir: Path):
        transformers = importlib.import_module("transformers")
        torch = importlib.import_module("torch")
        torch_dtype = torch.bfloat16 if bool(getattr(torch.cuda, "is_available", lambda: False)()) else torch.float32
        return transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=str(cache_dir.resolve()),
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
        )

    def load_dataset(self, dataset_path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for raw_line in dataset_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("dataset record must be a JSON object")
            records.append(payload)
        if not records:
            raise ValueError(f"dataset contains no records: {dataset_path}")
        return records

    def build_trainer(
        self,
        *,
        cfg: Mapping[str, Any],
        model: Any,
        tokenizer: Any,
        dataset: list[dict[str, Any]],
        output_dir: Path,
        deepspeed_config_path: Path,
        resume_info: Mapping[str, Any] | None,
    ) -> Any:
        del resume_info
        transformers = importlib.import_module("transformers")
        tokenized_records: list[dict[str, Any]] = []
        for record in dataset:
            rendered = render_messages_with_chat_template(tokenizer, record.get("messages"))
            batch = tokenizer(
                rendered,
                truncation=True,
                max_length=int(cfg["max_length"]),
                padding="max_length",
                return_tensors="pt",
            )
            item = {key: value[0] for key, value in batch.items()}
            item["labels"] = item["input_ids"].clone()
            tokenized_records.append(item)

        training_args = transformers.TrainingArguments(
            output_dir=str(output_dir.resolve()),
            per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
            max_steps=int(cfg["max_steps"]),
            learning_rate=float(cfg["learning_rate"]),
            logging_steps=int(cfg["logging_steps"]),
            save_steps=int(cfg["save_steps"]),
            save_total_limit=int(cfg["save_total_limit"]),
            bf16=bool(cfg["bf16"]),
            gradient_checkpointing=bool(cfg["gradient_checkpointing"]),
            deepspeed=str(deepspeed_config_path.resolve()),
            remove_unused_columns=False,
            report_to=[],
        )
        self._trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_records,
            processing_class=tokenizer,
            data_collator=transformers.default_data_collator,
        )
        return self._trainer

    def train(self, *, resume_from_checkpoint: str | None = None) -> dict[str, Any]:
        if self._trainer is None:
            raise RuntimeError("trainer has not been initialized")
        result = self._trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        global_step = getattr(result, "global_step", None)
        if not isinstance(global_step, int) or global_step <= 0:
            global_step = int(getattr(self._trainer.state, "global_step", 0))
        return {"global_step": global_step if global_step > 0 else None}

    def save_checkpoint(self, checkpoint_dir: Path) -> None:
        metadata = resolve_resume_checkpoint_metadata(checkpoint_dir)
        if metadata is None:
            raise RuntimeError("checkpoint_dir must point to an existing resume-capable checkpoint artifact")

    def save_final_export(self, export_dir: Path, *, distributed_info: Mapping[str, Any] | None = None) -> None:
        if self._trainer is None or self._tokenizer is None:
            raise RuntimeError("trainer artifacts are not ready")
        export_dir.mkdir(parents=True, exist_ok=True)
        self._trainer.save_model(str(export_dir.resolve()))
        if should_write_side_effects(distributed_info):
            self._tokenizer.save_pretrained(str(export_dir.resolve()))

    def validate_outputs(self, *, config_path: Path, checkpoint_dir: Path, final_export_dir: Path) -> None:
        validate_training_artifacts(checkpoint_dir, final_export_dir)
        validator = load_validation_module()
        result = validator.run_validation(
            config_path=config_path,
            model_path=str(final_export_dir.resolve()),
        )
        if result.get("status") != "success" or int(result.get("exit_code", 1)) != 0:
            error_message = str(result.get("error", "reload validation failed"))
            report_path = result.get("report_path")
            if report_path:
                raise RuntimeError(f"{error_message} (report: {report_path})")
            raise RuntimeError(error_message)


def build_report(
    *,
    cfg: Mapping[str, Any] | None,
    config_path: Path,
    dataset_info: Mapping[str, Any] | None,
    deepspeed_info: Mapping[str, Any] | None,
    readiness: Mapping[str, Any] | None,
    resume_info: Mapping[str, Any] | None,
    distributed_info: Mapping[str, Any] | None,
    artifact_paths: Mapping[str, Any] | None,
    dry_run: bool,
    status: str,
    error_message: str | None = None,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    config_summary: dict[str, Any] = {}
    if cfg is not None:
        artifacts = build_artifact_paths(cfg)
        config_summary = {
            "model_name": cfg["model_name"],
            "dataset_path": cfg["dataset_path"],
            "deepspeed_config_path": cfg["deepspeed_config_path"],
            "output_dir": cfg["output_dir"],
            "report_path": cfg["report_path"],
            "resume_mode": cfg["resume_mode"],
            "bf16": cfg["bf16"],
            "gradient_checkpointing": cfg["gradient_checkpointing"],
        }
    artifacts.update(dict(artifact_paths or {}))

    report: dict[str, Any] = {
        "status": status,
        "dry_run": dry_run,
        "config_path": str(config_path.resolve()),
        "config": config_summary,
        "dataset": dict(dataset_info or {}),
        "deepspeed_config": dict(deepspeed_info or {}),
        "readiness": dict(readiness or {}),
        "resume": dict(resume_info or {}),
        "distributed": dict(distributed_info or {}),
        "artifacts": artifacts,
    }
    if error_message is not None:
        report["error"] = {"message": error_message}
    return report


def coordinated_writer_phase(
    *,
    cfg: Mapping[str, Any],
    runtime: Any,
    distributed_info: Mapping[str, Any] | None,
    phase_name: str,
    action,
) -> None:
    barrier = getattr(runtime, "barrier", None)
    if callable(barrier):
        barrier(distributed_info)

    status_path = resolve_sync_status_path(cfg, phase_name)
    if should_write_side_effects(distributed_info):
        status_path.parent.mkdir(parents=True, exist_ok=True)
        result: dict[str, Any]
        try:
            action()
        except Exception as exc:
            result = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        else:
            result = {"status": "ok"}
        write_json(status_path, result)

    if callable(barrier):
        barrier(distributed_info)

    phase_result = read_json(status_path)
    if str(phase_result.get("status")) != "ok":
        message = str(phase_result.get("error_message", f"{phase_name} failed"))
        raise RuntimeError(message)


def coordinated_collective_phase(
    *,
    cfg: Mapping[str, Any],
    runtime: Any,
    distributed_info: Mapping[str, Any] | None,
    phase_name: str,
    action,
) -> None:
    barrier = getattr(runtime, "barrier", None)
    rank = int((distributed_info or {}).get("rank", 0))
    world_size = distributed_world_size(distributed_info)
    local_status_path = resolve_rank_sync_status_path(cfg, phase_name, rank)
    aggregate_status_path = resolve_sync_status_path(cfg, phase_name)

    if callable(barrier):
        barrier(distributed_info)

    local_status_path.parent.mkdir(parents=True, exist_ok=True)
    local_result: dict[str, Any]
    try:
        action()
    except Exception as exc:
        local_result = {
            "status": "error",
            "rank": rank,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    else:
        local_result = {"status": "ok", "rank": rank}
    write_json(local_status_path, local_result)

    if callable(barrier):
        barrier(distributed_info)

    if should_write_side_effects(distributed_info):
        aggregate_result = {"status": "ok"}
        for current_rank in range(world_size):
            current_status_path = resolve_rank_sync_status_path(cfg, phase_name, current_rank)
            current_result = read_json(current_status_path)
            if str(current_result.get("status")) == "ok":
                continue
            aggregate_result = {
                "status": "error",
                "error_rank": int(current_result.get("rank", current_rank)),
                "error_type": str(current_result.get("error_type", "RuntimeError")),
                "error_message": str(current_result.get("error_message", f"{phase_name} failed")),
            }
            break
        write_json(aggregate_status_path, aggregate_result)

    if callable(barrier):
        barrier(distributed_info)

    phase_result = read_json(aggregate_status_path)
    if str(phase_result.get("status")) != "ok":
        error_rank = phase_result.get("error_rank")
        message = str(phase_result.get("error_message", f"{phase_name} failed"))
        if error_rank is not None:
            raise RuntimeError(f"{phase_name} failed on rank {error_rank}: {message}")
        raise RuntimeError(message)


def reset_post_training_phase_statuses(
    *,
    cfg: Mapping[str, Any],
    runtime: Any,
    distributed_info: Mapping[str, Any] | None,
    phase_names: list[str],
) -> None:
    barrier = getattr(runtime, "barrier", None)
    if callable(barrier):
        barrier(distributed_info)

    if should_write_side_effects(distributed_info):
        for phase_name in phase_names:
            status_path = resolve_sync_status_path(cfg, phase_name)
            if status_path.exists():
                status_path.unlink()

    if callable(barrier):
        barrier(distributed_info)


def mark_distributed_finalized(distributed_info: Mapping[str, Any] | None) -> None:
    if isinstance(distributed_info, dict):
        distributed_info["initialized"] = False
        distributed_info["initialized_by_runner"] = False


def coordinated_post_training_writer_phase(
    *,
    cfg: Mapping[str, Any],
    runtime: Any,
    distributed_info: Mapping[str, Any] | None,
    phase_name: str,
    action,
) -> None:
    del runtime
    status_path = resolve_sync_status_path(cfg, phase_name)

    if should_write_side_effects(distributed_info):
        status_path.parent.mkdir(parents=True, exist_ok=True)
        result: dict[str, Any]
        try:
            action()
        except Exception as exc:
            result = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        else:
            result = {"status": "ok"}
        write_json(status_path, result)
    else:
        deadline = time.monotonic() + POST_TRAINING_PHASE_TIMEOUT_SECONDS
        while not status_path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for {phase_name} writer status: {status_path}")
            time.sleep(POST_TRAINING_PHASE_POLL_SECONDS)

    phase_result = read_json(status_path)
    if str(phase_result.get("status")) != "ok":
        message = str(phase_result.get("error_message", f"{phase_name} failed"))
        raise RuntimeError(message)


def run_from_config(
    config_path: Path | str,
    *,
    dry_run: bool = False,
    readiness_facts: Mapping[str, Any] | None = None,
    runtime: Any | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    cfg: dict[str, Any] | None = None
    dataset_info: dict[str, Any] | None = None
    deepspeed_info: dict[str, Any] | None = None
    readiness: dict[str, Any] | None = None
    resume_info: dict[str, Any] | None = None
    distributed_info: dict[str, Any] | None = None
    artifact_paths: dict[str, Any] | None = None
    runtime = runtime or FullFTRuntime()

    try:
        cfg = load_and_validate_config(config_path)
        dataset_info = ensure_existing_file(Path(str(cfg["dataset_path"])), label="dataset_path", require_non_empty=True)
        deepspeed_info = ensure_existing_file(
            Path(str(cfg["deepspeed_config_path"])),
            label="deepspeed_config_path",
            require_non_empty=False,
        )
        readiness = evaluate_readiness_gate(
            dict(readiness_facts) if readiness_facts is not None else collect_readiness_facts(Path(str(cfg["output_dir"])))
        )
        if readiness["status"] != "ready":
            report = build_report(
                cfg=cfg,
                config_path=config_path,
                dataset_info=dataset_info,
                deepspeed_info=deepspeed_info,
                readiness=readiness,
                resume_info=None,
                distributed_info=None,
                artifact_paths=None,
                dry_run=dry_run,
                status="blocked",
                error_message=str(readiness.get("blocking_reason", "readiness gate failed")),
            )
            write_report_json(Path(str(cfg["report_path"])).resolve(), report)
            return report

        resume_info = resolve_resume_behavior(cfg)
        if dry_run:
            report = build_report(
                cfg=cfg,
                config_path=config_path,
                dataset_info=dataset_info,
                deepspeed_info=deepspeed_info,
                readiness=readiness,
                resume_info=resume_info,
                distributed_info=None,
                artifact_paths=None,
                dry_run=True,
                status="dry_run_ready",
            )
            write_report_json(Path(str(cfg["report_path"])).resolve(), report)
            return report

        output_dir = Path(str(cfg["output_dir"])).resolve()
        cache_dir = Path(str(cfg["cache_dir"])).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        distributed_init_path = resolve_distributed_init_path(cfg)
        distributed_info = dict(runtime.initialize_distributed(output_dir=output_dir, init_path=distributed_init_path))
        artifact_paths = build_artifact_paths(cfg, distributed_init_path=distributed_init_path)

        tokenizer = runtime.load_tokenizer(str(cfg["model_name"]), cache_dir=cache_dir)
        model = runtime.load_model(str(cfg["model_name"]), cache_dir=cache_dir)
        dataset = runtime.load_dataset(Path(str(cfg["dataset_path"])).resolve())
        runtime.build_trainer(
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            output_dir=output_dir,
            deepspeed_config_path=Path(str(cfg["deepspeed_config_path"])).resolve(),
            resume_info=resume_info,
        )
        train_output = runtime.train(resume_from_checkpoint=resume_info.get("checkpoint_path") if resume_info else None)
        checkpoint_path = resolve_post_training_resume_checkpoint(
            output_dir,
            preferred_path=resolve_checkpoint_output_path(cfg, resolve_training_global_step(train_output, cfg)),
        )
        final_export_path = resolve_final_export_path(cfg)
        artifact_paths = build_artifact_paths(
            cfg,
            distributed_init_path=distributed_init_path,
            checkpoint_path=checkpoint_path,
        )
        coordinated_collective_phase(
            cfg=cfg,
            runtime=runtime,
            distributed_info=distributed_info,
            phase_name="final export",
            action=lambda: runtime.save_final_export(final_export_path, distributed_info=distributed_info),
        )
        artifact_paths = build_artifact_paths(
            cfg,
            distributed_init_path=distributed_init_path,
            checkpoint_path=checkpoint_path,
            final_export_path=final_export_path,
        )
        report = build_report(
            cfg=cfg,
            config_path=config_path,
            dataset_info=dataset_info,
            deepspeed_info=deepspeed_info,
            readiness=readiness,
            resume_info=resume_info,
            distributed_info=distributed_info,
            artifact_paths=artifact_paths,
            dry_run=False,
            status="success",
        )
        reset_post_training_phase_statuses(
            cfg=cfg,
            runtime=runtime,
            distributed_info=distributed_info,
            phase_names=["train result report"],
        )
        finalize = getattr(runtime, "finalize_distributed", None)
        if callable(finalize):
            finalize(distributed_info)
        mark_distributed_finalized(distributed_info)

        coordinated_post_training_writer_phase(
            cfg=cfg,
            runtime=runtime,
            distributed_info=distributed_info,
            phase_name="train result report",
            action=lambda: write_report_json(
                Path(str(cfg["report_path"])).resolve(),
                report,
                distributed_info=distributed_info,
            ),
        )
        return report
    except Exception as exc:
        report = build_report(
            cfg=cfg,
            config_path=config_path,
            dataset_info=dataset_info,
            deepspeed_info=deepspeed_info,
            readiness=readiness,
            resume_info=resume_info,
            distributed_info=distributed_info,
            artifact_paths=artifact_paths,
            dry_run=dry_run,
            status="failed",
            error_message=str(exc),
        )
        target_report_path = Path(str(cfg["report_path"])).resolve() if cfg is not None else determine_failure_report_path(config_path)
        write_report_json(target_report_path, report, distributed_info=distributed_info)
        return report
    finally:
        finalize = getattr(runtime, "finalize_distributed", None)
        if callable(finalize):
            finalize(distributed_info)


def _detect_gpu_count() -> int:
    """Detect GPU count without initializing CUDA (safe for orchestrator)."""
    import subprocess as _sp

    try:
        result = _sp.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return len(result.stdout.strip().splitlines())
    except Exception:
        pass
    return 1


def run_orchestrated(
    config_path: str | Path,
    *,
    dry_run: bool = False,
    nproc_per_node: int | None = None,
    skip_validation: bool = False,
) -> dict[str, Any]:
    """Run training via torchrun subprocess, then validation via separate subprocess."""
    import subprocess as _sp

    config_path = Path(config_path).resolve()
    cfg = read_json(config_path)

    if nproc_per_node is None:
        nproc_per_node = _detect_gpu_count()

    script_path = str(Path(__file__).resolve())
    venv_python = sys.executable

    train_cmd: list[str]
    if nproc_per_node > 1:
        torchrun_path = str(Path(venv_python).parent / "torchrun")
        train_cmd = [
            torchrun_path, "--standalone", f"--nproc_per_node={nproc_per_node}",
            script_path, "--config", str(config_path),
        ]
    else:
        train_cmd = [venv_python, script_path, "--config", str(config_path)]

    if dry_run:
        train_cmd.append("--dry-run")

    print(f"[orchestrate] training cmd: {' '.join(train_cmd)}", flush=True)
    train_proc = _sp.run(train_cmd, capture_output=False)

    report_path = Path(str(cfg.get("report_path", "train_result.json"))).resolve()
    if not report_path.exists():
        return {"status": "failed", "phase": "training", "error": "train_result.json not found after training"}

    train_report = read_json(report_path)
    if train_report.get("status") != "success":
        print(f"[orchestrate] training ended with status={train_report.get('status')}, skipping validation", flush=True)
        return train_report

    if dry_run or skip_validation:
        print("[orchestrate] skipping validation (dry-run or --skip-validation)", flush=True)
        return train_report

    final_export_path = resolve_final_export_path(cfg)
    if not final_export_path.exists():
        train_report["validation_skipped"] = "final-export directory not found"
        print(f"[orchestrate] final-export not found at {final_export_path}, skipping validation", flush=True)
        return train_report

    validator_script = str(Path(__file__).with_name("check_gpt_oss_full_ft_output.py").resolve())
    validate_cmd = [
        venv_python, validator_script,
        "--config", str(config_path),
        "--model-path", str(final_export_path),
    ]

    print(f"[orchestrate] validation cmd: {' '.join(validate_cmd)}", flush=True)
    validate_proc = _sp.run(validate_cmd, capture_output=False)

    train_report["validation"] = {
        "exit_code": validate_proc.returncode,
        "status": "success" if validate_proc.returncode == 0 else "failed",
    }
    write_json(report_path, train_report)

    print(f"[orchestrate] done. training={train_report['status']}, validation={'success' if validate_proc.returncode == 0 else 'failed'}", flush=True)
    return train_report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.orchestrate:
        result = run_orchestrated(
            args.config,
            dry_run=bool(args.dry_run),
            nproc_per_node=args.nproc_per_node,
            skip_validation=bool(args.skip_validation),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") in {"dry_run_ready", "success"} else 1
    report = run_from_config(args.config, dry_run=bool(args.dry_run))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"dry_run_ready", "success"} else 1


if __name__ == "__main__":
    sys.exit(main())
