from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ROUND1_FULL_FT_CONFIG_PATH = ROOT / "configs/gpt_oss_20b_seed_v8_round1_full_ft.json"
ROUND3_SECTION1_FULL_FT_CONFIG_PATH = ROOT / "configs/gpt_oss_20b_seed_v8_round3_section1_full_ft.json"
ROUND4_SECTION_ALL_FULL_FT_CONFIG_PATH = ROOT / "configs/gpt_oss_20b_seed_v8_round4_section_all_full_ft.json"
DEEPSPEED_CONFIG_PATH = ROOT / "configs/deepspeed/gpt_oss_20b_zero3_bf16.json"

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


def load_module(relative_path: str, module_name: str):
    path = ROOT / relative_path
    assert path.exists(), f"expected contract scaffold to exist: {path}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v8_full_ft_contract_uses_seed_v8_dataset_and_full_model_paths():
    module = load_module("scripts/v8_full_ft_contracts.py", "v8_full_ft_contracts")

    assert module.DATASET_OUTPUT_RELATIVE_PATH == "llm_datasets/seed_v8/seed_v8_round1_full_ft.jsonl"
    assert module.MODEL_OUTPUT_RELATIVE_DIR == "llm_model_full/gpt-oss-20b-seed-v8-round1-full-ft"
    assert module.TRAIN_REPORT_RELATIVE_PATH == "llm_model_full/gpt-oss-20b-seed-v8-round1-full-ft/train_result.json"
    assert module.FINAL_EXPORT_RELATIVE_DIR == "llm_model_full/gpt-oss-20b-seed-v8-round1-full-ft/final-export"
    assert "llm_model_lora" not in module.MODEL_OUTPUT_RELATIVE_DIR


def test_v8_full_ft_default_config_contains_required_keys_and_expected_paths():
    module = load_module("scripts/v8_full_ft_contracts.py", "v8_full_ft_contracts")

    cfg = module.build_default_config(ROOT)

    assert REQUIRED_CONFIG_KEYS.issubset(cfg.keys())
    assert cfg["dataset_path"].endswith(module.DATASET_OUTPUT_RELATIVE_PATH)
    assert cfg["output_dir"].endswith(module.MODEL_OUTPUT_RELATIVE_DIR)
    assert cfg["report_path"].endswith(module.TRAIN_REPORT_RELATIVE_PATH)
    assert cfg["prompt_source_path"].endswith(module.DATASET_OUTPUT_RELATIVE_PATH)
    assert cfg["deepspeed_config_path"].endswith("configs/deepspeed/gpt_oss_20b_zero3_bf16.json")

    validated = module.validate_full_ft_config(cfg)
    assert validated["resume_mode"] == "fail"


def test_v8_round1_full_ft_config_file_matches_contract_and_validates():
    module = load_module("scripts/v8_full_ft_contracts.py", "v8_full_ft_contracts")

    assert ROUND1_FULL_FT_CONFIG_PATH.exists()
    cfg = json.loads(ROUND1_FULL_FT_CONFIG_PATH.read_text(encoding="utf-8"))

    assert REQUIRED_CONFIG_KEYS.issubset(cfg.keys())
    assert cfg["dataset_path"] == str((ROOT / module.DATASET_OUTPUT_RELATIVE_PATH).resolve())
    assert cfg["prompt_source_path"] == str((ROOT / module.DATASET_OUTPUT_RELATIVE_PATH).resolve())
    assert cfg["output_dir"] == str((ROOT / module.MODEL_OUTPUT_RELATIVE_DIR).resolve())
    assert cfg["report_path"] == str((ROOT / module.TRAIN_REPORT_RELATIVE_PATH).resolve())
    assert cfg["deepspeed_config_path"] == str(DEEPSPEED_CONFIG_PATH.resolve())

    validated = module.validate_full_ft_config(cfg)
    assert validated["resume_mode"] == "fail"


def test_v8_round1_deepspeed_config_is_conservative_zero3_bf16():
    assert DEEPSPEED_CONFIG_PATH.exists()

    cfg = json.loads(DEEPSPEED_CONFIG_PATH.read_text(encoding="utf-8"))

    assert cfg["bf16"] == {"enabled": True}
    assert cfg["gradient_clipping"] == 1.0
    assert cfg["train_micro_batch_size_per_gpu"] == 1
    assert cfg["zero_optimization"]["stage"] == 3
    assert cfg["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] is True
    assert cfg["zero_optimization"]["overlap_comm"] is False
    assert cfg["zero_optimization"]["offload_optimizer"]["device"] == "none"
    assert cfg["zero_optimization"]["offload_param"]["device"] == "none"


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda cfg: cfg | {"resume_mode": "resume_from_path"}, "resume_from_checkpoint"),
        (
            lambda cfg: cfg
            | {
                "validation_model_path": "/tmp/final-export",
                "validation_checkpoint_path": "/tmp/checkpoint-10",
            },
            "mutually exclusive",
        ),
        (lambda cfg: cfg | {"save_steps": cfg["max_steps"] + 1}, "save_steps"),
        (lambda cfg: cfg | {"per_device_train_batch_size": 0}, "per_device_train_batch_size"),
        (lambda cfg: cfg | {"gradient_accumulation_steps": 0}, "gradient_accumulation_steps"),
        (lambda cfg: cfg | {"sample_prompt_count": 0}, "sample_prompt_count"),
        (lambda cfg: cfg | {"per_device_train_batch_size": True}, "per_device_train_batch_size"),
        (lambda cfg: cfg | {"gradient_accumulation_steps": False}, "gradient_accumulation_steps"),
        (lambda cfg: cfg | {"sample_prompt_count": True}, "sample_prompt_count"),
        (lambda cfg: cfg | {"save_steps": True}, "save_steps"),
        (lambda cfg: cfg | {"max_steps": False}, "max_steps"),
    ],
)
def test_v8_full_ft_config_validation_rejects_invalid_contracts(mutator, message):
    module = load_module("scripts/v8_full_ft_contracts.py", "v8_full_ft_contracts")

    cfg = mutator(module.build_default_config(ROOT))

    with pytest.raises(ValueError, match=message):
        module.validate_full_ft_config(cfg)


def test_v8_full_ft_validation_accepts_model_or_checkpoint_but_not_adapter():
    module = load_module("scripts/v8_full_ft_contracts.py", "v8_full_ft_contracts")

    cfg = module.build_default_config(ROOT)

    model_result = module.resolve_validation_target(cfg, model_path="/tmp/final-export")
    assert model_result == {"path": "/tmp/final-export", "source": "model_path"}

    checkpoint_result = module.resolve_validation_target(cfg, checkpoint_path="/tmp/checkpoint-10")
    assert checkpoint_result == {"path": "/tmp/checkpoint-10", "source": "checkpoint_path"}

    with pytest.raises(ValueError, match="adapter_path"):
        module.resolve_validation_target(cfg, adapter_path="/tmp/adapter")

    with pytest.raises(ValueError, match="mutually exclusive"):
        module.resolve_validation_target(
            cfg,
            model_path="/tmp/final-export",
            checkpoint_path="/tmp/checkpoint-10",
        )


def test_v8_full_ft_builder_parses_markdown_groups_and_expands_question_variants():
    module = load_module("scripts/build_v8_round1_full_ft_dataset.py", "build_v8_round1_full_ft_dataset")

    parsed = module.parse_markdown_source(ROOT / "scripts/data_source.md")
    groups = parsed["groups"]
    records = module.build_dataset_records(groups)

    assert parsed["columns"] == ["user", "assistant"]
    assert len(groups) == 5
    assert len(records) == sum(len(group["questions"]) for group in groups)
    assert records[0]["id"] == "zeroin.seed_v8_round1_full_ft_0001"
    assert records[-1]["id"] == "zeroin.seed_v8_round1_full_ft_0150"
    assert records[0]["messages"][0]["content"] == module.SYSTEM_PROMPT
    assert all([item["role"] for item in record["messages"]] == ["system", "user", "assistant"] for record in records)
    assert records[0]["meta"]["round"] == "round1"
    assert records[0]["meta"]["group_id"] == "group1"
    assert records[0]["meta"]["training_mode"] == "full_ft"


def test_v8_full_ft_builder_writes_exact_dataset_path_and_file_contents(tmp_path: Path):
    builder = load_module("scripts/build_v8_round1_full_ft_dataset.py", "build_v8_round1_full_ft_dataset")
    contracts = load_module("scripts/v8_full_ft_contracts.py", "v8_full_ft_contracts_for_builder_test")

    summary = builder.build_dataset(
        root=tmp_path,
        source_path=ROOT / "scripts/data_source.md",
    )

    dataset_path = tmp_path / contracts.DATASET_OUTPUT_RELATIVE_PATH
    assert summary["dataset_path"] == str(dataset_path.resolve())
    assert dataset_path.exists()

    lines = [line for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]

    assert len(records) == 150
    assert records[0]["messages"][1]["content"] == "유형생성의 기본원칙에서 충분성, 비교성, 지속성은 각각 무엇을 뜻하나요?"
    assert records[-1]["meta"]["group_id"] == "group5"


def test_v8_round3_section1_builder_parses_csv_groups_and_expands_question_variants():
    module = load_module("scripts/build_v8_round3_section1_full_ft_dataset.py", "build_v8_round3_section1_full_ft_dataset")

    parsed = module.parse_csv_source(ROOT / "docs/제로인방법론/source_csv/v8-r3 section-1.csv")
    groups = parsed["groups"]
    records = module.build_dataset_records(groups)

    assert parsed["columns"] == ["num", "단원", "chat input", "system", "user", "assistant", "chat input"]
    assert len(groups) == 39
    assert len(records) == 472
    assert records[0]["id"] == "zeroin.seed_v8_round3_section1_full_ft_0001"
    assert records[-1]["id"] == "zeroin.seed_v8_round3_section1_full_ft_0472"
    assert records[0]["messages"][0]["content"].startswith("당신은 제로인 펀드평가 방법론에 근거해 답변하는 도메인 어시스턴트입니다.")
    assert records[0]["messages"][1]["content"] == "유형생성의 기본원칙에서 충분성, 비교성, 지속성은 각각 무엇을 뜻하나요?"
    assert (
        records[-1]["messages"][1]["content"]
        == "운용펀드 2개 이상, 운용사 2개 이상, BM 안정화, 장기 자금 유지가 검증된 뒤에만 독립 유형 신설을 검토하는 이유를 설명해주세요."
    )
    assert records[0]["meta"]["round"] == "round3"
    assert records[0]["meta"]["section"] == "section1"
    assert records[0]["meta"]["source_strategy"] == "csv_multiline_question_expansion"
    assert records[-1]["meta"]["group_id"] == "group39"


def test_v8_round3_section1_builder_writes_exact_dataset_path_and_file_contents(tmp_path: Path):
    builder = load_module("scripts/build_v8_round3_section1_full_ft_dataset.py", "build_v8_round3_section1_full_ft_dataset_write")

    summary = builder.build_dataset(
        root=tmp_path,
        source_path=ROOT / "docs/제로인방법론/source_csv/v8-r3 section-1.csv",
    )

    dataset_path = tmp_path / "llm_datasets/seed_v8/seed_v8_round3_section1_full_ft.jsonl"
    assert summary["dataset_path"] == str(dataset_path.resolve())
    assert dataset_path.exists()

    lines = [line for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]

    assert summary["group_count"] == 39
    assert summary["record_count"] == 472
    assert len(records) == 472
    assert records[0]["messages"][1]["content"] == "유형생성의 기본원칙에서 충분성, 비교성, 지속성은 각각 무엇을 뜻하나요?"
    assert records[-1]["meta"]["group_id"] == "group39"


def test_v8_round3_section1_full_ft_config_file_matches_expected_paths_and_validates():
    contracts = load_module("scripts/v8_full_ft_contracts.py", "v8_full_ft_contracts_round3_section1")

    assert ROUND3_SECTION1_FULL_FT_CONFIG_PATH.exists()
    cfg = json.loads(ROUND3_SECTION1_FULL_FT_CONFIG_PATH.read_text(encoding="utf-8"))

    assert REQUIRED_CONFIG_KEYS.issubset(cfg.keys())
    assert cfg["dataset_path"] == str((ROOT / "llm_datasets/seed_v8/seed_v8_round3_section1_full_ft.jsonl").resolve())
    assert cfg["prompt_source_path"] == str((ROOT / "llm_datasets/seed_v8/seed_v8_round3_section1_full_ft.jsonl").resolve())
    assert cfg["output_dir"] == str((ROOT / "llm_model_full/gpt-oss-20b-seed-v8-round3-section1-full-ft").resolve())
    assert cfg["report_path"] == str(
        (ROOT / "llm_model_full/gpt-oss-20b-seed-v8-round3-section1-full-ft/train_result.json").resolve()
    )
    assert cfg["deepspeed_config_path"] == str(DEEPSPEED_CONFIG_PATH.resolve())

    validated = contracts.validate_full_ft_config(cfg)
    assert validated["resume_mode"] == "fail"


def test_v8_round4_section_all_builder_parses_csv_groups_and_expands_question_variants():
    module = load_module("scripts/build_v8_round4_section_all_full_ft_dataset.py", "build_v8_round4_section_all_full_ft_dataset")

    parsed = module.parse_csv_source(ROOT / "docs/제로인방법론/source_csv/v9-r5.csv")
    groups = parsed["groups"]
    records = module.build_dataset_records(groups)

    assert parsed["columns"] == ["num", "sec", "system", "user_q_base", "assistant", "user_q_ext"]
    assert parsed["skipped_rows"] == []
    assert len(groups) == 529
    assert len(records) == 5785
    assert records[0]["id"] == "zeroin.seed_v8_round4_section_all_full_ft_0001"
    assert records[-1]["id"] == "zeroin.seed_v8_round4_section_all_full_ft_5785"
    assert records[0]["messages"][0]["content"].startswith("당신은 제로인 펀드평가 방법론에 근거해 답변하는 도메인 어시스턴트입니다.")
    assert records[0]["messages"][1]["content"] == "유형생성의 기본원칙에서 충분성, 비교성, 지속성은 각각 무엇을 뜻하나요?"
    assert records[-1]["messages"][1]["content"] == "Zeroin"
    assert records[0]["meta"]["round"] == "round4"
    assert records[0]["meta"]["section"] == "section_all"
    assert records[0]["meta"]["source_strategy"] == "csv_multiline_question_expansion"
    assert records[-1]["meta"]["group_id"] == "group529"
    assert records[-1]["meta"]["chapter"] == "0"


def test_v8_round4_section_all_builder_writes_exact_dataset_path_and_file_contents(tmp_path: Path):
    builder = load_module(
        "scripts/build_v8_round4_section_all_full_ft_dataset.py",
        "build_v8_round4_section_all_full_ft_dataset_write",
    )

    summary = builder.build_dataset(
        root=tmp_path,
        source_path=ROOT / "docs/제로인방법론/source_csv/v9-r5.csv",
    )

    dataset_path = tmp_path / "llm_datasets/seed_v8/seed_v8_round4_section_all_full_ft.jsonl"
    assert summary["dataset_path"] == str(dataset_path.resolve())
    assert dataset_path.exists()

    lines = [line for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]

    assert summary["group_count"] == 529
    assert summary["record_count"] == 5785
    assert summary["skipped_rows"] == []
    assert len(records) == 5785
    assert records[0]["messages"][1]["content"] == "유형생성의 기본원칙에서 충분성, 비교성, 지속성은 각각 무엇을 뜻하나요?"
    assert records[-1]["meta"]["group_id"] == "group529"


def test_v8_round4_section_all_full_ft_config_file_matches_expected_paths_and_validates():
    contracts = load_module("scripts/v8_full_ft_contracts.py", "v8_full_ft_contracts_round4_section_all")
    round4_output_dir = Path("/home/work/llm_model_full/gpt-oss-20b-seed-v8-round4-section-all-full-ft")

    assert ROUND4_SECTION_ALL_FULL_FT_CONFIG_PATH.exists()
    cfg = json.loads(ROUND4_SECTION_ALL_FULL_FT_CONFIG_PATH.read_text(encoding="utf-8"))

    assert REQUIRED_CONFIG_KEYS.issubset(cfg.keys())
    assert cfg["dataset_path"] == str((ROOT / "llm_datasets/seed_v8/seed_v8_round4_section_all_full_ft.jsonl").resolve())
    assert cfg["prompt_source_path"] == str((ROOT / "llm_datasets/seed_v8/seed_v8_round4_section_all_full_ft.jsonl").resolve())
    assert cfg["output_dir"] == str(round4_output_dir)
    assert cfg["report_path"] == str(round4_output_dir / "train_result.json")
    assert cfg["deepspeed_config_path"] == str(DEEPSPEED_CONFIG_PATH.resolve())

    validated = contracts.validate_full_ft_config(cfg)
    assert validated["resume_mode"] == "fail"


def test_v8_full_ft_readiness_gate_reports_required_fields():
    runner = load_module("scripts/run_full_ft_gpt_oss_20b.py", "run_full_ft_gpt_oss_20b_readiness")

    report = runner.evaluate_readiness_gate(
        {
            "gpu_count": 3,
            "gpu_vram_gib": [80.0, 80.0, 80.0],
            "disk_free_bytes": 2 * runner.TIB,
            "bf16_supported": True,
            "package_versions": {
                "torch": "2.7.0",
                "transformers": "4.52.0",
                "accelerate": "1.7.0",
                "deepspeed": "0.16.0",
            },
            "package_imports": {
                "torch": {"importable": True},
                "transformers": {"importable": True},
                "accelerate": {"importable": True},
                "deepspeed": {"importable": True},
            },
        }
    )

    assert report["status"] == "ready"
    assert report["gpu_count"] == 3
    assert report["gpu_vram_gib"] == [80.0, 80.0, 80.0]
    assert report["disk_free_bytes"] == 2 * runner.TIB
    assert report["bf16_supported"] is True
    assert report["package_versions"]["torch"] == "2.7.0"
    assert report["package_compatibility"]["status"] == "ok"
    assert report["package_compatibility"]["packages"]["torch"]["status"] == "ok"
    assert report["package_compatibility"]["packages"]["deepspeed"]["version"] == "0.16.0"
    assert report["package_compatibility"]["packages"]["deepspeed"]["importable"] is True


def test_v8_full_ft_readiness_gate_accepts_one_tib_free_disk():
    runner = load_module("scripts/run_full_ft_gpt_oss_20b.py", "run_full_ft_gpt_oss_20b_readiness_one_tib")

    report = runner.evaluate_readiness_gate(
        {
            "gpu_count": 3,
            "gpu_vram_gib": [80.0, 80.0, 80.0],
            "disk_free_bytes": runner.TIB,
            "bf16_supported": True,
            "package_versions": {
                "torch": "2.7.0",
                "transformers": "4.52.0",
                "accelerate": "1.7.0",
                "deepspeed": "0.16.0",
            },
            "package_imports": {
                "torch": {"importable": True},
                "transformers": {"importable": True},
                "accelerate": {"importable": True},
                "deepspeed": {"importable": True},
            },
        }
    )

    assert report["status"] == "ready"
    assert report["thresholds"]["min_disk_free_bytes"] == runner.TIB


def test_v8_full_ft_readiness_gate_fails_fast_below_thresholds():
    runner = load_module("scripts/run_full_ft_gpt_oss_20b.py", "run_full_ft_gpt_oss_20b_readiness_fail")

    report = runner.evaluate_readiness_gate(
        {
            "gpu_count": 1,
            "gpu_vram_gib": [10.0],
            "disk_free_bytes": 10 * runner.GIB,
            "bf16_supported": False,
            "package_versions": {
                "torch": "2.7.0",
                "transformers": "4.52.0",
                "accelerate": "1.7.0",
                "deepspeed": "0.16.0",
            },
            "package_imports": {
                "torch": {"importable": True},
                "transformers": {"importable": True},
                "accelerate": {"importable": True},
                "deepspeed": {"importable": True},
            },
        }
    )

    assert report["status"] == "blocked"
    assert report["blocking_check"] == "gpu_count"
    assert "at least 2" in report["blocking_reason"]


def test_v8_full_ft_readiness_gate_reports_package_compatibility_failure():
    runner = load_module("scripts/run_full_ft_gpt_oss_20b.py", "run_full_ft_gpt_oss_20b_package_fail")

    report = runner.evaluate_readiness_gate(
        {
            "gpu_count": 3,
            "gpu_vram_gib": [80.0, 80.0, 80.0],
            "disk_free_bytes": 2 * runner.TIB,
            "bf16_supported": True,
            "package_versions": {
                "torch": "2.7.0",
                "transformers": "4.52.0",
                "accelerate": "1.7.0",
                "deepspeed": "missing",
            },
            "package_imports": {
                "torch": {"importable": True},
                "transformers": {"importable": True},
                "accelerate": {"importable": True},
                "deepspeed": {"importable": False, "error": "ModuleNotFoundError: No module named 'deepspeed'"},
            },
        }
    )

    assert report["status"] == "blocked"
    assert report["blocking_check"] == "package_compatibility"
    assert report["package_compatibility"]["status"] == "blocked"
    assert report["package_compatibility"]["packages"]["deepspeed"]["status"] == "missing"
    assert "deepspeed" in report["blocking_reason"]


def test_v8_full_ft_readiness_gate_blocks_when_metadata_exists_but_import_fails():
    runner = load_module("scripts/run_full_ft_gpt_oss_20b.py", "run_full_ft_gpt_oss_20b_import_fail")

    report = runner.evaluate_readiness_gate(
        {
            "gpu_count": 3,
            "gpu_vram_gib": [80.0, 80.0, 80.0],
            "disk_free_bytes": 2 * runner.TIB,
            "bf16_supported": True,
            "package_versions": {
                "torch": "2.7.0",
                "transformers": "4.52.0",
                "accelerate": "1.7.0",
                "deepspeed": "0.16.0",
            },
            "package_imports": {
                "torch": {"importable": True},
                "transformers": {"importable": True},
                "accelerate": {"importable": True},
                "deepspeed": {"importable": False, "error": "ImportError: libaio.so missing"},
            },
        }
    )

    assert report["status"] == "blocked"
    assert report["blocking_check"] == "package_compatibility"
    assert report["package_compatibility"]["status"] == "blocked"
    assert report["package_compatibility"]["packages"]["deepspeed"]["status"] == "import_failed"
    assert report["package_compatibility"]["packages"]["deepspeed"]["version"] == "0.16.0"
    assert "deepspeed" in report["blocking_reason"]


def write_resume_checkpoint(path: Path, *, global_step: int | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "weights.bin").write_text("checkpoint", encoding="utf-8")
    (path / "optimizer.pt").write_text("optimizer", encoding="utf-8")
    (path / "scheduler.pt").write_text("scheduler", encoding="utf-8")
    step = int(global_step) if global_step is not None else int(path.name.split("-")[-1])
    (path / "trainer_state.json").write_text(json.dumps({"global_step": step}, ensure_ascii=False), encoding="utf-8")
    (path / "resume-artifact.json").write_text(
        json.dumps(
            {
                "resume_capable": True,
                "checkpoint_path": str(path.resolve()),
                "global_step": step,
                "checkpoint_kind": "trainer_native",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def write_export_only_checkpoint(path: Path, *, global_step: int | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "weights.bin").write_text("checkpoint", encoding="utf-8")
    step = int(global_step) if global_step is not None else int(path.name.split("-")[-1])
    (path / "trainer_state.json").write_text(json.dumps({"global_step": step}, ensure_ascii=False), encoding="utf-8")
    (path / "resume-artifact.json").write_text(
        json.dumps(
            {
                "resume_capable": False,
                "checkpoint_path": str(path.resolve()),
                "global_step": step,
                "checkpoint_kind": "export_only",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_v8_full_ft_resolve_resume_checkpoint_metadata_requires_native_resume_evidence(tmp_path: Path):
    runner = load_module("scripts/run_full_ft_gpt_oss_20b.py", "run_full_ft_gpt_oss_20b_resume_metadata")

    native_checkpoint = write_resume_checkpoint(tmp_path / "checkpoint-20")
    export_only_checkpoint = write_export_only_checkpoint(tmp_path / "checkpoint-30")

    native_metadata = runner.resolve_resume_checkpoint_metadata(native_checkpoint)
    export_metadata = runner.resolve_resume_checkpoint_metadata(export_only_checkpoint)

    assert native_metadata is not None
    assert native_metadata["resume_capable"] is True
    assert native_metadata["step"] == 20
    assert export_metadata is None


def test_v8_full_ft_resume_mode_handling_resolves_checkpoint_behavior(tmp_path: Path):
    runner = load_module("scripts/run_full_ft_gpt_oss_20b.py", "run_full_ft_gpt_oss_20b_resume")
    contracts = load_module("scripts/v8_full_ft_contracts.py", "v8_full_ft_contracts_for_runner_resume")

    checkpoint_9 = tmp_path / "checkpoint-9"
    checkpoint_a = tmp_path / "checkpoint-10"
    checkpoint_b = tmp_path / "checkpoint-20"
    checkpoint_100 = tmp_path / "checkpoint-100"
    write_resume_checkpoint(checkpoint_9)
    write_resume_checkpoint(checkpoint_a)
    write_resume_checkpoint(checkpoint_b)
    write_export_only_checkpoint(checkpoint_100)

    with pytest.raises(ValueError, match="resume_mode=fail"):
        runner.resolve_resume_behavior(contracts.build_default_config(tmp_path) | {"output_dir": str(tmp_path)})

    latest = runner.resolve_resume_behavior(
        contracts.build_default_config(tmp_path)
        | {
            "output_dir": str(tmp_path),
            "resume_mode": "resume_latest",
        }
    )
    assert latest["action"] == "resume"
    assert latest["checkpoint_path"] == str(checkpoint_b.resolve())
    assert latest["checkpoint_step"] == 20

    explicit = runner.resolve_resume_behavior(
        contracts.build_default_config(tmp_path)
        | {
            "output_dir": str(tmp_path),
            "resume_mode": "resume_from_path",
            "resume_from_checkpoint": str(checkpoint_a),
        }
    )
    assert explicit["action"] == "resume"
    assert explicit["checkpoint_path"] == str(checkpoint_a.resolve())
    assert explicit["checkpoint_step"] == 10

    not_a_checkpoint_dir = tmp_path / "model-export"
    not_a_checkpoint_dir.mkdir()
    with pytest.raises(ValueError, match="checkpoint-\\*"):
        runner.resolve_resume_behavior(
            contracts.build_default_config(tmp_path)
            | {
                "output_dir": str(tmp_path),
                "resume_mode": "resume_from_path",
                "resume_from_checkpoint": str(not_a_checkpoint_dir),
            }
        )

    invalid_resume_checkpoint = tmp_path / "checkpoint-200"
    invalid_resume_checkpoint.mkdir()
    with pytest.raises(ValueError, match="resume-capable"):
        runner.resolve_resume_behavior(
            contracts.build_default_config(tmp_path)
            | {
                "output_dir": str(tmp_path),
                "resume_mode": "resume_from_path",
                "resume_from_checkpoint": str(invalid_resume_checkpoint),
            }
        )

    checkpoint_file = tmp_path / "checkpoint-55.json"
    checkpoint_file.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a directory"):
        runner.resolve_resume_behavior(
            contracts.build_default_config(tmp_path)
            | {
                "output_dir": str(tmp_path),
                "resume_mode": "resume_from_path",
                "resume_from_checkpoint": str(checkpoint_file),
            }
        )

    clean_output = tmp_path / "clean-output"
    clean_output.mkdir()
    overwrite = runner.resolve_resume_behavior(
        contracts.build_default_config(tmp_path)
        | {
            "output_dir": str(clean_output),
            "resume_mode": "overwrite_empty_only",
        }
    )
    assert overwrite == {"action": "start_fresh", "checkpoint_path": None, "mode": "overwrite_empty_only"}


def test_v8_full_ft_runtime_load_tokenizer_uses_cache_dir_and_remote_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = load_module("scripts/run_full_ft_gpt_oss_20b.py", "run_full_ft_gpt_oss_20b_load_tokenizer")
    runtime = runner.FullFTRuntime()
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeTokenizer:
        pad_token = None
        eos_token = "</s>"

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs):
            calls.append((model_name, kwargs))
            return FakeTokenizer()

    fake_transformers = type("FakeTransformers", (), {"AutoTokenizer": FakeAutoTokenizer})()
    real_import_module = runner.importlib.import_module

    def fake_import_module(name: str):
        if name == "transformers":
            return fake_transformers
        return real_import_module(name)

    monkeypatch.setattr(runner.importlib, "import_module", fake_import_module)

    tokenizer = runtime.load_tokenizer("unsloth/gpt-oss-20b-BF16", cache_dir=tmp_path / ".cache")

    assert calls == [
        (
            "unsloth/gpt-oss-20b-BF16",
            {
                "cache_dir": str((tmp_path / ".cache").resolve()),
                "trust_remote_code": True,
            },
        )
    ]
    assert tokenizer.pad_token == tokenizer.eos_token


def test_v8_full_ft_runtime_load_model_uses_gpt_oss_loading_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = load_module("scripts/run_full_ft_gpt_oss_20b.py", "run_full_ft_gpt_oss_20b_load_model")
    runtime = runner.FullFTRuntime()
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs):
            calls.append((model_name, kwargs))
            return {"kind": "model", "model_name": model_name}

    fake_transformers = type("FakeTransformers", (), {"AutoModelForCausalLM": FakeAutoModelForCausalLM})()
    fake_torch = type(
        "FakeTorch",
        (),
        {
            "bfloat16": "bf16",
            "float32": "fp32",
            "cuda": type("FakeCuda", (), {"is_available": staticmethod(lambda: True)})(),
        },
    )()
    real_import_module = runner.importlib.import_module

    def fake_import_module(name: str):
        if name == "transformers":
            return fake_transformers
        if name == "torch":
            return fake_torch
        return real_import_module(name)

    monkeypatch.setattr(runner.importlib, "import_module", fake_import_module)

    model = runtime.load_model("unsloth/gpt-oss-20b-BF16", cache_dir=tmp_path / ".cache")

    assert model == {"kind": "model", "model_name": "unsloth/gpt-oss-20b-BF16"}
    assert calls == [
        (
            "unsloth/gpt-oss-20b-BF16",
            {
                "cache_dir": str((tmp_path / ".cache").resolve()),
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "torch_dtype": "bf16",
            },
        )
    ]


def test_v8_full_ft_runtime_build_trainer_uses_processing_class_not_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    runner = load_module("scripts/run_full_ft_gpt_oss_20b.py", "run_full_ft_gpt_oss_20b_build_trainer")
    runtime = runner.FullFTRuntime()
    trainer_calls: list[dict[str, object]] = []

    class FakeTrainer:
        def __init__(self, **kwargs):
            trainer_calls.append(kwargs)

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTensor:
        def __init__(self, values):
            self.values = list(values)

        def __getitem__(self, index):
            assert index == 0
            return self

        def clone(self):
            return FakeTensor(self.values)

    class FakeTokenizer:
        def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False):
            del tokenize, add_generation_prompt
            return "|".join(f"{message['role']}={message['content']}" for message in messages)

        def __call__(self, text: str, **kwargs):
            del text, kwargs
            return {
                "input_ids": FakeTensor([1, 2, 3]),
                "attention_mask": FakeTensor([1, 1, 1]),
            }

    fake_transformers = type(
        "FakeTransformers",
        (),
        {
            "Trainer": FakeTrainer,
            "TrainingArguments": FakeTrainingArguments,
            "default_data_collator": object(),
        },
    )()
    real_import_module = runner.importlib.import_module

    def fake_import_module(name: str):
        if name == "transformers":
            return fake_transformers
        return real_import_module(name)

    monkeypatch.setattr(runner.importlib, "import_module", fake_import_module)

    trainer = runtime.build_trainer(
        cfg={
            "max_length": 16,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_steps": 10,
            "learning_rate": 1e-5,
            "logging_steps": 1,
            "save_steps": 5,
            "save_total_limit": 1,
            "bf16": False,
            "gradient_checkpointing": False,
        },
        model={"kind": "model"},
        tokenizer=FakeTokenizer(),
        dataset=[
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            }
        ],
        output_dir=tmp_path / "output",
        deepspeed_config_path=tmp_path / "deepspeed.json",
        resume_info=None,
    )

    assert trainer is runtime._trainer
    assert len(trainer_calls) == 1
    assert "processing_class" in trainer_calls[0]
    assert "tokenizer" not in trainer_calls[0]


def test_v8_full_ft_runtime_build_trainer_renders_messages_with_chat_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    runner = load_module("scripts/run_full_ft_gpt_oss_20b.py", "run_full_ft_gpt_oss_20b_chat_template")
    runtime = runner.FullFTRuntime()
    tokenizer_inputs: list[str] = []

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTensor:
        def __init__(self, values):
            self.values = list(values)

        def __getitem__(self, index):
            assert index == 0
            return self

        def clone(self):
            return FakeTensor(self.values)

    class FakeTokenizer:
        def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False):
            assert tokenize is False
            assert add_generation_prompt is False
            return "TEMPLATE::" + "|".join(f"{msg['role']}={msg['content']}" for msg in messages)

        def __call__(self, text: str, **kwargs):
            del kwargs
            tokenizer_inputs.append(text)
            return {
                "input_ids": FakeTensor([1, 2, 3]),
                "attention_mask": FakeTensor([1, 1, 1]),
            }

    fake_transformers = type(
        "FakeTransformers",
        (),
        {
            "Trainer": FakeTrainer,
            "TrainingArguments": FakeTrainingArguments,
            "default_data_collator": object(),
        },
    )()
    real_import_module = runner.importlib.import_module

    def fake_import_module(name: str):
        if name == "transformers":
            return fake_transformers
        return real_import_module(name)

    monkeypatch.setattr(runner.importlib, "import_module", fake_import_module)

    runtime.build_trainer(
        cfg={
            "max_length": 16,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_steps": 10,
            "learning_rate": 1e-5,
            "logging_steps": 1,
            "save_steps": 5,
            "save_total_limit": 1,
            "bf16": False,
            "gradient_checkpointing": False,
        },
        model={"kind": "model"},
        tokenizer=FakeTokenizer(),
        dataset=[
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            }
        ],
        output_dir=tmp_path / "output",
        deepspeed_config_path=tmp_path / "deepspeed.json",
        resume_info=None,
    )

    assert tokenizer_inputs == ["TEMPLATE::system=system|user=question|assistant=answer"]


def test_v8_full_ft_runtime_save_checkpoint_preserves_native_resume_artifacts(tmp_path: Path):
    runner = load_module("scripts/run_full_ft_gpt_oss_20b.py", "run_full_ft_gpt_oss_20b_save_checkpoint")
    runtime = runner.FullFTRuntime()

    checkpoint_dir = tmp_path / "checkpoint-12"
    write_resume_checkpoint(checkpoint_dir)
    original_metadata = json.loads((checkpoint_dir / "resume-artifact.json").read_text(encoding="utf-8"))

    runtime.save_checkpoint(checkpoint_dir)

    resume_metadata = json.loads((checkpoint_dir / "resume-artifact.json").read_text(encoding="utf-8"))
    assert resume_metadata == original_metadata
    assert resume_metadata["resume_capable"] is True
    assert resume_metadata["checkpoint_kind"] == "trainer_native"


def test_v8_full_ft_runtime_save_final_export_only_writes_tokenizer_on_writer_rank(tmp_path: Path):
    runner = load_module("scripts/run_full_ft_gpt_oss_20b.py", "run_full_ft_gpt_oss_20b_save_final_export")
    runtime = runner.FullFTRuntime()
    export_dir = tmp_path / "final-export"
    save_model_calls: list[str] = []
    save_pretrained_calls: list[str] = []

    class FakeTrainer:
        def save_model(self, path: str) -> None:
            save_model_calls.append(path)

    class FakeTokenizer:
        def save_pretrained(self, path: str) -> None:
            save_pretrained_calls.append(path)

    runtime._trainer = FakeTrainer()
    runtime._tokenizer = FakeTokenizer()

    runtime.save_final_export(export_dir, distributed_info={"rank": 1, "world_size": 2})

    assert save_model_calls == [str(export_dir.resolve())]
    assert save_pretrained_calls == []


def test_v8_full_ft_dry_run_writes_structured_train_result(tmp_path: Path):
    runner = load_module("scripts/run_full_ft_gpt_oss_20b.py", "run_full_ft_gpt_oss_20b_dry_run")
    contracts = load_module("scripts/v8_full_ft_contracts.py", "v8_full_ft_contracts_for_runner_dry_run")

    cfg = contracts.build_default_config(tmp_path)
    dataset_path = Path(cfg["dataset_path"])
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        json.dumps(
            {
                "id": "sample-1",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    deepspeed_path = Path(cfg["deepspeed_config_path"])
    deepspeed_path.parent.mkdir(parents=True, exist_ok=True)
    deepspeed_path.write_text("{}", encoding="utf-8")

    config_path = tmp_path / "full-ft-config.json"
    config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    result = runner.run_from_config(
        config_path,
        dry_run=True,
        readiness_facts={
            "gpu_count": 3,
            "gpu_vram_gib": [80.0, 80.0, 80.0],
            "disk_free_bytes": 2 * runner.TIB,
            "bf16_supported": True,
            "package_versions": {
                "torch": "2.7.0",
                "transformers": "4.52.0",
                "accelerate": "1.7.0",
                "deepspeed": "0.16.0",
            },
            "package_imports": {
                "torch": {"importable": True},
                "transformers": {"importable": True},
                "accelerate": {"importable": True},
                "deepspeed": {"importable": True},
            },
        },
    )

    report_path = Path(cfg["report_path"])
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result["status"] == "dry_run_ready"
    assert report["status"] == "dry_run_ready"
    assert report["dry_run"] is True
    assert report["config"]["model_name"] == cfg["model_name"]
    assert report["dataset"]["path"] == str(dataset_path.resolve())
    assert report["dataset"]["exists"] is True
    assert report["resume"]["action"] == "start_fresh"
    assert report["readiness"]["status"] == "ready"
    assert report["readiness"]["package_compatibility"]["status"] == "ok"
    assert report["readiness"]["package_compatibility"]["packages"]["deepspeed"]["importable"] is True
    assert report["artifacts"]["report_path"] == str(report_path.resolve())


READY_READINESS_FACTS = {
    "gpu_count": 3,
    "gpu_vram_gib": [80.0, 80.0, 80.0],
    "disk_free_bytes": 2 * (1024**4),
    "bf16_supported": True,
    "package_versions": {
        "torch": "2.7.0",
        "transformers": "4.52.0",
        "accelerate": "1.7.0",
        "deepspeed": "0.16.0",
    },
    "package_imports": {
        "torch": {"importable": True},
        "transformers": {"importable": True},
        "accelerate": {"importable": True},
        "deepspeed": {"importable": True},
    },
}


class FakeFullFTRuntime:
    def __init__(
        self,
        *,
        fail_on_train: str | None = None,
        fail_on_export: str | None = None,
        fail_on_validate: str | None = None,
        train_global_step: int = 12,
        native_checkpoint_steps: list[int] | None = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.fail_on_train = fail_on_train
        self.fail_on_export = fail_on_export
        self.fail_on_validate = fail_on_validate
        self.train_global_step = train_global_step
        self.native_checkpoint_steps = list(native_checkpoint_steps or [10])
        self.rank = rank
        self.world_size = world_size
        self.output_dir: Path | None = None
        self.calls: list[tuple[str, str]] = []

    def initialize_distributed(self, *, output_dir: Path, init_path: Path) -> dict[str, object]:
        self.calls.append(("initialize_distributed", str(init_path.resolve())))
        return {
            "backend": "gloo",
            "world_size": self.world_size,
            "rank": self.rank,
            "init_method": f"file://{init_path.resolve()}",
            "initialized_by_runner": True,
            "initialized": True,
        }

    def barrier(self, distributed_info) -> None:
        self.calls.append(("barrier", f"{distributed_info['rank']}|{distributed_info['world_size']}"))

    def finalize_distributed(self, distributed_info) -> None:
        self.calls.append(
            (
                "finalize_distributed",
                f"{distributed_info['rank']}|{distributed_info['world_size']}|{distributed_info.get('initialized_by_runner')}",
            )
        )

    def load_tokenizer(self, model_name: str, *, cache_dir: Path):
        del cache_dir
        self.calls.append(("load_tokenizer", model_name))
        return {"kind": "tokenizer", "model_name": model_name}

    def load_model(self, model_name: str, *, cache_dir: Path):
        del cache_dir
        self.calls.append(("load_model", model_name))
        return {"kind": "model", "model_name": model_name}

    def load_dataset(self, dataset_path: Path):
        self.calls.append(("load_dataset", str(dataset_path.resolve())))
        return [{"messages": [{"role": "user", "content": "question"}]}]

    def build_trainer(self, *, cfg, model, tokenizer, dataset, output_dir: Path, deepspeed_config_path: Path, resume_info):
        del cfg, model, tokenizer, dataset, deepspeed_config_path, resume_info
        self.output_dir = output_dir
        self.calls.append(("build_trainer", str(output_dir.resolve())))
        return self

    def train(self, *, resume_from_checkpoint: str | None = None) -> dict[str, int | None]:
        self.calls.append(("train", str(resume_from_checkpoint)))
        if self.fail_on_train is not None:
            raise RuntimeError(self.fail_on_train)
        assert self.output_dir is not None
        for step in self.native_checkpoint_steps:
            write_resume_checkpoint(self.output_dir / f"checkpoint-{step}", global_step=step)
        return {"global_step": self.train_global_step}

    def save_checkpoint(self, checkpoint_dir: Path) -> None:
        self.calls.append(("save_checkpoint", checkpoint_dir.name))

    def save_final_export(self, export_dir: Path, *, distributed_info=None) -> None:
        del distributed_info
        self.calls.append(("save_final_export", export_dir.name))
        if self.fail_on_export is not None:
            raise RuntimeError(self.fail_on_export)
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / "config.json").write_text("{}", encoding="utf-8")

    def validate_outputs(self, *, config_path: Path, checkpoint_dir: Path, final_export_dir: Path) -> None:
        del config_path
        self.calls.append(("validate_outputs", f"{checkpoint_dir.name}|{final_export_dir.name}"))
        if self.fail_on_validate is not None:
            raise ValueError(self.fail_on_validate)


class FakeTorchDistributed:
    def __init__(self) -> None:
        self.initialized = False
        self.destroy_calls = 0
        self.init_calls: list[dict[str, object]] = []

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return self.initialized

    def init_process_group(self, *, backend: str, init_method: str, rank: int, world_size: int) -> None:
        self.init_calls.append(
            {
                "backend": backend,
                "init_method": init_method,
                "rank": rank,
                "world_size": world_size,
            }
        )
        self.initialized = True

    def destroy_process_group(self) -> None:
        self.destroy_calls += 1
        self.initialized = False


class FakeTorchModule:
    def __init__(self, distributed: FakeTorchDistributed) -> None:
        self.distributed = distributed
        self.cuda = type("FakeCuda", (), {"is_available": staticmethod(lambda: False)})()


def write_full_ft_config_fixture(tmp_path: Path) -> tuple[object, object, Path, dict[str, object]]:
    runner = load_module("scripts/run_full_ft_gpt_oss_20b.py", f"run_full_ft_gpt_oss_20b_fixture_{tmp_path.name}")
    contracts = load_module("scripts/v8_full_ft_contracts.py", f"v8_full_ft_contracts_fixture_{tmp_path.name}")

    cfg = contracts.build_default_config(tmp_path)
    dataset_path = Path(cfg["dataset_path"])
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        json.dumps(
            {
                "id": "sample-1",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    deepspeed_path = Path(cfg["deepspeed_config_path"])
    deepspeed_path.parent.mkdir(parents=True, exist_ok=True)
    deepspeed_path.write_text("{}", encoding="utf-8")

    config_path = tmp_path / "full-ft-config.json"
    config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return runner, contracts, config_path, cfg


def test_v8_full_ft_runtime_resets_distributed_ownership_between_reused_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = load_module("scripts/run_full_ft_gpt_oss_20b.py", "run_full_ft_gpt_oss_20b_runtime_reuse")
    runtime = runner.FullFTRuntime()
    fake_distributed = FakeTorchDistributed()
    fake_torch = FakeTorchModule(fake_distributed)

    real_import_module = runner.importlib.import_module

    def fake_import_module(name: str):
        if name == "torch":
            return fake_torch
        return real_import_module(name)

    monkeypatch.setattr(runner.importlib, "import_module", fake_import_module)

    first_info = runtime.initialize_distributed(output_dir=tmp_path, init_path=tmp_path / "dist-1")
    assert first_info["initialized_by_runner"] is True
    runtime.finalize_distributed(first_info)
    assert fake_distributed.destroy_calls == 1

    fake_distributed.initialized = True
    second_info = runtime.initialize_distributed(output_dir=tmp_path, init_path=tmp_path / "dist-2")
    assert second_info["initialized_by_runner"] is False
    runtime.finalize_distributed(second_info)
    assert fake_distributed.destroy_calls == 1


def test_v8_full_ft_runtime_reads_launcher_distributed_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = load_module("scripts/run_full_ft_gpt_oss_20b.py", "run_full_ft_gpt_oss_20b_distributed_env")
    runtime = runner.FullFTRuntime()
    fake_distributed = FakeTorchDistributed()
    fake_torch = FakeTorchModule(fake_distributed)

    real_import_module = runner.importlib.import_module

    def fake_import_module(name: str):
        if name == "torch":
            return fake_torch
        return real_import_module(name)

    monkeypatch.setattr(runner.importlib, "import_module", fake_import_module)
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "1")

    distributed_info = runtime.initialize_distributed(output_dir=tmp_path, init_path=tmp_path / "dist-env")

    assert distributed_info["rank"] == 2
    assert distributed_info["world_size"] == 4
    assert distributed_info["local_rank"] == 1
    assert fake_distributed.init_calls == [
        {
            "backend": "gloo",
            "init_method": f"file://{(tmp_path / 'dist-env').resolve()}",
            "rank": 2,
            "world_size": 4,
        }
    ]


def test_v8_full_ft_runtime_uses_env_rendezvous_for_multi_process_launcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runner = load_module("scripts/run_full_ft_gpt_oss_20b.py", "run_full_ft_gpt_oss_20b_env_rendezvous")
    runtime = runner.FullFTRuntime()
    fake_distributed = FakeTorchDistributed()
    fake_torch = FakeTorchModule(fake_distributed)

    real_import_module = runner.importlib.import_module

    def fake_import_module(name: str):
        if name == "torch":
            return fake_torch
        return real_import_module(name)

    init_path = tmp_path / "dist-env"
    init_path.write_text("keep-me", encoding="utf-8")

    monkeypatch.setattr(runner.importlib, "import_module", fake_import_module)
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")

    distributed_info = runtime.initialize_distributed(output_dir=tmp_path, init_path=init_path)

    assert distributed_info["init_method"] == "env://"
    assert init_path.exists()
    assert fake_distributed.init_calls == [
        {
            "backend": "gloo",
            "init_method": "env://",
            "rank": 2,
            "world_size": 4,
        }
    ]


def test_v8_full_ft_runtime_validate_outputs_runs_reload_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = load_module("scripts/run_full_ft_gpt_oss_20b.py", "run_full_ft_gpt_oss_20b_validate_outputs")
    runtime = runner.FullFTRuntime()
    checkpoint_dir = tmp_path / "checkpoint-12"
    checkpoint_dir.mkdir()
    final_export_dir = tmp_path / "final-export"
    final_export_dir.mkdir()
    config_path = tmp_path / "full-ft-config.json"
    config_path.write_text("{}", encoding="utf-8")

    calls: list[dict[str, object]] = []

    class FakeValidationModule:
        @staticmethod
        def run_validation(**kwargs):
            calls.append(kwargs)
            return {
                "status": "failed",
                "exit_code": 1,
                "error": "reload validation exploded",
                "report_path": str(tmp_path / "tests/log/reload-validation.md"),
            }

    monkeypatch.setattr(runner, "load_validation_module", lambda: FakeValidationModule)

    with pytest.raises(RuntimeError, match="reload validation exploded"):
        runtime.validate_outputs(
            config_path=config_path,
            checkpoint_dir=checkpoint_dir,
            final_export_dir=final_export_dir,
        )

    assert calls == [
        {
            "config_path": config_path,
            "model_path": str(final_export_dir.resolve()),
        }
    ]


def test_v8_full_ft_non_dry_run_wires_runtime_and_writes_success_report(tmp_path: Path):
    runner, _contracts, config_path, cfg = write_full_ft_config_fixture(tmp_path)
    runtime = FakeFullFTRuntime()

    result = runner.run_from_config(
        config_path,
        dry_run=False,
        readiness_facts=READY_READINESS_FACTS,
        runtime=runtime,
    )

    report_path = Path(cfg["report_path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert result["status"] == "success"
    assert report["status"] == "success"
    assert report["dry_run"] is False
    assert result["distributed"]["init_method"].startswith("file://")
    assert Path(result["artifacts"]["distributed_init_path"]).name == "distributed-init"
    assert Path(result["artifacts"]["checkpoint_path"]).name == "checkpoint-10"
    assert Path(result["artifacts"]["final_export_path"]).name == "final-export"
    assert Path(result["artifacts"]["checkpoint_path"]).exists()
    assert Path(result["artifacts"]["final_export_path"]).exists()
    assert ("initialize_distributed", str((Path(cfg["output_dir"]) / "distributed-init").resolve())) in runtime.calls
    assert ("load_tokenizer", cfg["model_name"]) in runtime.calls
    assert ("load_model", cfg["model_name"]) in runtime.calls
    assert ("load_dataset", str(Path(cfg["dataset_path"]).resolve())) in runtime.calls
    assert not any(call[0] == "save_checkpoint" for call in runtime.calls)
    assert ("save_final_export", "final-export") in runtime.calls
    assert not any(call[0] == "validate_outputs" for call in runtime.calls)
    assert ("finalize_distributed", "0|1|True") in runtime.calls


def test_v8_full_ft_non_dry_run_succeeds_without_in_process_validation(tmp_path: Path):
    runner, _contracts, config_path, cfg = write_full_ft_config_fixture(tmp_path)
    runtime = FakeFullFTRuntime(fail_on_validate="should never be called")

    result = runner.run_from_config(
        config_path,
        dry_run=False,
        readiness_facts=READY_READINESS_FACTS,
        runtime=runtime,
    )

    report = json.loads(Path(cfg["report_path"]).read_text(encoding="utf-8"))
    assert result["status"] == "success"
    assert report["status"] == "success"
    assert not any(call[0] == "validate_outputs" for call in runtime.calls)
    assert Path(report["artifacts"]["checkpoint_path"]).name == "checkpoint-10"
    assert Path(report["artifacts"]["final_export_path"]).name == "final-export"


def test_v8_full_ft_non_dry_run_preserves_checkpoint_path_when_export_save_fails(tmp_path: Path):
    runner, _contracts, config_path, cfg = write_full_ft_config_fixture(tmp_path)
    runtime = FakeFullFTRuntime(fail_on_export="final export save failed")

    result = runner.run_from_config(
        config_path,
        dry_run=False,
        readiness_facts=READY_READINESS_FACTS,
        runtime=runtime,
    )

    report = json.loads(Path(cfg["report_path"]).read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert report["status"] == "failed"
    assert "final export save failed" in report["error"]["message"]
    assert Path(report["artifacts"]["checkpoint_path"]).name == "checkpoint-10"
    assert Path(report["artifacts"]["checkpoint_path"]).exists()
    assert "final_export_path" not in report["artifacts"]


def test_v8_full_ft_non_dry_run_marks_training_exception_as_failed(tmp_path: Path):
    runner, _contracts, config_path, cfg = write_full_ft_config_fixture(tmp_path)
    runtime = FakeFullFTRuntime(fail_on_train="training exploded")

    result = runner.run_from_config(
        config_path,
        dry_run=False,
        readiness_facts=READY_READINESS_FACTS,
        runtime=runtime,
    )

    report = json.loads(Path(cfg["report_path"]).read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert report["status"] == "failed"
    assert "training exploded" in report["error"]["message"]


def test_v8_full_ft_non_writer_rank_participates_in_final_export_collective_then_waits_on_post_training_writer_phase(
    tmp_path: Path,
):
    runner, _contracts, config_path, cfg = write_full_ft_config_fixture(tmp_path)
    runtime = FakeFullFTRuntime(rank=1, world_size=2)
    collective_phases: list[tuple[str, int]] = []
    post_training_writer_phases: list[tuple[str, int]] = []

    def fake_coordinated_collective_phase(*, cfg, runtime, distributed_info, phase_name, action):
        del cfg, runtime
        collective_phases.append((phase_name, int(distributed_info["rank"])))
        action()

    def fake_coordinated_post_training_writer_phase(*, cfg, runtime, distributed_info, phase_name, action):
        del cfg, runtime, action
        post_training_writer_phases.append((phase_name, int(distributed_info["rank"])))

    runner.coordinated_collective_phase = fake_coordinated_collective_phase
    runner.coordinated_post_training_writer_phase = fake_coordinated_post_training_writer_phase

    result = runner.run_from_config(
        config_path,
        dry_run=False,
        readiness_facts=READY_READINESS_FACTS,
        runtime=runtime,
    )

    assert result["status"] == "success"
    assert ("save_checkpoint", "checkpoint-12") not in runtime.calls
    assert ("save_final_export", "final-export") in runtime.calls
    assert not any(call[0] == "validate_outputs" for call in runtime.calls)
    assert ("finalize_distributed", "1|2|True") in runtime.calls
    assert collective_phases == [("final export", 1)]
    assert post_training_writer_phases == [
        ("train result report", 1),
    ]
    assert Path(cfg["report_path"]).exists() is False


def test_v8_full_ft_orchestrate_parses_args():
    runner = load_module("scripts/run_full_ft_gpt_oss_20b.py", "run_full_ft_gpt_oss_20b_orchestrate_args")
    args = runner.parse_args(["--config", "/tmp/cfg.json", "--orchestrate", "--nproc-per-node", "3"])
    assert args.orchestrate is True
    assert args.nproc_per_node == 3
    assert args.skip_validation is False
    args2 = runner.parse_args(["--config", "/tmp/cfg.json", "--orchestrate", "--skip-validation"])
    assert args2.skip_validation is True


def test_v8_full_ft_orchestrate_skips_validation_when_training_not_success(tmp_path: Path):
    runner = load_module("scripts/run_full_ft_gpt_oss_20b.py", "run_full_ft_gpt_oss_20b_orchestrate_skip")
    config_path, cfg = _write_orchestrate_config_fixture(tmp_path)
    report_path = Path(cfg["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    runner.write_json(report_path, {"status": "failed", "error": {"message": "boom"}})

    result = runner.run_orchestrated(config_path, nproc_per_node=1, skip_validation=True)
    assert result["status"] == "failed"


def test_v8_full_ft_orchestrate_detect_gpu_count():
    runner = load_module("scripts/run_full_ft_gpt_oss_20b.py", "run_full_ft_gpt_oss_20b_detect_gpu")
    count = runner._detect_gpu_count()
    assert isinstance(count, int)
    assert count >= 1


def _write_orchestrate_config_fixture(tmp_path: Path) -> tuple[Path, dict]:
    runner = load_module("scripts/run_full_ft_gpt_oss_20b.py", "run_full_ft_gpt_oss_20b_orch_fixture")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    cfg = {
        "model_name": "test-model",
        "cache_dir": str(tmp_path / "cache"),
        "dataset_path": str(tmp_path / "dataset.jsonl"),
        "output_dir": str(output_dir),
        "report_path": str(output_dir / "train_result.json"),
        "max_length": 512,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_steps": 10,
        "learning_rate": 1e-5,
        "logging_steps": 1,
        "save_steps": 5,
        "save_total_limit": 1,
        "seed": 42,
        "bf16": True,
        "gradient_checkpointing": True,
        "deepspeed_config_path": str(tmp_path / "ds.json"),
        "prompt_source_path": str(tmp_path / "dataset.jsonl"),
        "sample_prompt_count": 1,
        "resume_mode": "fail",
    }
    config_path = tmp_path / "config.json"
    runner.write_json(config_path, cfg)
    return config_path, cfg


class FakeFullFTValidatorRuntime:
    def __init__(
        self,
        *,
        answers_by_prompt: dict[str, str] | None = None,
        load_error: str | None = None,
        generation_error: str | None = None,
    ) -> None:
        self.answers_by_prompt = answers_by_prompt or {}
        self.load_error = load_error
        self.generation_error = generation_error
        self.loaded_model_sources: list[str] = []
        self.generated_prompts: list[str] = []

    def load_tokenizer(self, *, model_source: str, cache_dir: Path):
        del cache_dir
        if self.load_error is not None:
            raise RuntimeError(self.load_error)
        return {"kind": "tokenizer", "model_source": model_source}

    def load_model(self, *, model_source: str, cache_dir: Path):
        del cache_dir
        if self.load_error is not None:
            raise RuntimeError(self.load_error)
        self.loaded_model_sources.append(model_source)
        return {"kind": "model", "model_source": model_source}

    def generate_final_answer(
        self,
        *,
        model,
        tokenizer,
        prompt_text: str,
        generation_settings: dict[str, object],
    ) -> str:
        del model, tokenizer, generation_settings
        if self.generation_error is not None:
            raise RuntimeError(self.generation_error)
        self.generated_prompts.append(prompt_text)
        return self.answers_by_prompt.get(prompt_text, "")


def write_full_ft_validator_fixture(
    tmp_path: Path,
    *,
    prompt_records: list[dict[str, object]] | None = None,
    sample_prompt_count: int = 2,
) -> tuple[object, Path, dict[str, object], Path, Path]:
    validator = load_module("scripts/check_gpt_oss_full_ft_output.py", f"check_gpt_oss_full_ft_output_{tmp_path.name}")
    contracts = load_module("scripts/v8_full_ft_contracts.py", f"v8_full_ft_contracts_validator_{tmp_path.name}")

    cfg = contracts.build_default_config(tmp_path)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    final_export_dir = output_dir / "final-export"
    final_export_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoint-12"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    prompt_source_path = tmp_path / "prompt-source.jsonl"
    records = prompt_records or [
        {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "첫 번째 질문"},
                {"role": "assistant", "content": "첫 번째 답변"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "두 번째 질문"},
                {"role": "assistant", "content": "두 번째 답변"},
            ]
        },
    ]
    prompt_source_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    cfg["prompt_source_path"] = str(prompt_source_path.resolve())
    cfg["sample_prompt_count"] = sample_prompt_count

    config_path = tmp_path / "full-ft-validator-config.json"
    config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return validator, config_path, cfg, final_export_dir, checkpoint_dir


def test_v8_full_ft_validator_accepts_model_path_or_checkpoint_path(tmp_path: Path):
    validator, config_path, _cfg, final_export_dir, checkpoint_dir = write_full_ft_validator_fixture(tmp_path)

    model_runtime = FakeFullFTValidatorRuntime(answers_by_prompt={"첫 번째 질문": "모델 답변", "두 번째 질문": "모델 답변 2"})
    model_report_path = tmp_path / "tests/log/model-report.md"
    model_exit_code = validator.main(
        [
            "--config",
            str(config_path),
            "--model-path",
            str(final_export_dir),
            "--report-path",
            str(model_report_path),
        ],
        runtime=model_runtime,
    )
    assert model_exit_code == 0
    assert model_runtime.loaded_model_sources == [str(final_export_dir.resolve())]

    checkpoint_runtime = FakeFullFTValidatorRuntime(
        answers_by_prompt={"첫 번째 질문": "체크포인트 답변", "두 번째 질문": "체크포인트 답변 2"}
    )
    checkpoint_report_path = tmp_path / "tests/log/checkpoint-report.md"
    checkpoint_exit_code = validator.main(
        [
            "--config",
            str(config_path),
            "--checkpoint-path",
            str(checkpoint_dir),
            "--report-path",
            str(checkpoint_report_path),
        ],
        runtime=checkpoint_runtime,
    )
    assert checkpoint_exit_code == 0
    assert checkpoint_runtime.loaded_model_sources == [str(checkpoint_dir.resolve())]


def test_v8_full_ft_validator_loads_prompts_from_prompt_source_path(tmp_path: Path):
    prompt_records = [
        {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "prompt_source_path 질문 1"},
                {"role": "assistant", "content": "답변 1"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "prompt_source_path 질문 2"},
                {"role": "assistant", "content": "답변 2"},
            ]
        },
    ]
    validator, config_path, _cfg, final_export_dir, _checkpoint_dir = write_full_ft_validator_fixture(
        tmp_path,
        prompt_records=prompt_records,
    )
    runtime = FakeFullFTValidatorRuntime(
        answers_by_prompt={
            "prompt_source_path 질문 1": "생성 답변 1",
            "prompt_source_path 질문 2": "생성 답변 2",
        }
    )

    result = validator.run_validation(
        config_path=config_path,
        model_path=str(final_export_dir),
        report_path=tmp_path / "tests/log/prompt-source-report.md",
        runtime=runtime,
    )

    assert result["status"] == "success"
    assert runtime.generated_prompts == ["prompt_source_path 질문 1", "prompt_source_path 질문 2"]


def test_v8_full_ft_validator_writes_markdown_report_with_required_sections(tmp_path: Path):
    validator, config_path, cfg, final_export_dir, _checkpoint_dir = write_full_ft_validator_fixture(tmp_path)
    runtime = FakeFullFTValidatorRuntime(
        answers_by_prompt={
            "첫 번째 질문": "최종 생성 답변 1",
            "두 번째 질문": "최종 생성 답변 2",
        }
    )
    report_path = tmp_path / "tests/log/v8_round1_full_ft_test_report.md"

    result = validator.run_validation(
        config_path=config_path,
        model_path=str(final_export_dir),
        report_path=report_path,
        runtime=runtime,
    )

    report_text = report_path.read_text(encoding="utf-8")
    assert result["status"] == "success"
    assert "- Run Status: success" in report_text
    assert f"- Resolved Model Path: {final_export_dir.resolve()}" in report_text
    assert "- do_sample: False" in report_text
    assert "- max_new_tokens: 1024" in report_text
    assert "- temperature: 1.0" in report_text
    assert "- top_p: 1.0" in report_text
    assert "- Prompt Source Path: " + str(Path(cfg["prompt_source_path"]).resolve()) in report_text
    assert "### Sample 1" in report_text
    assert "첫 번째 질문" in report_text
    assert "최종 생성 답변 1" in report_text


def test_v8_full_ft_validator_extracts_final_channel_and_reports_raw_generation(tmp_path: Path):
    validator = load_module("scripts/check_gpt_oss_full_ft_output.py", f"check_gpt_oss_full_ft_output_raw_final_{tmp_path.name}")

    extracted = validator.extract_final_text(
        "<|channel|>analysis<|message|>생각<|end|><|start|>assistant<|channel|>final<|message|>최종 답변<|return|>"
    )

    assert extracted == "최종 답변"

    report = validator.render_markdown_report(
        {
            "status": "success",
            "config_path": str(tmp_path / "config.json"),
            "resolved_model_path": str(tmp_path / "final-export"),
            "target_source": "model_path",
            "prompt_source_path": str(tmp_path / "prompt-source.jsonl"),
            "generation_settings": validator.build_generation_settings(),
            "samples": [
                {
                    "prompt_text": "질문",
                    "raw_generation": "<|channel|>analysis<|message|>생각<|end|><|start|>assistant<|channel|>final<|message|>최종 답변<|return|>",
                    "final_answer": "최종 답변",
                }
            ],
        }
    )

    assert "**Raw Generation**" in report
    assert "**Generated Final Answer**" in report
    assert "최종 답변" in report


def test_v8_full_ft_validator_fails_nonzero_when_all_generations_are_empty(tmp_path: Path):
    validator, config_path, _cfg, final_export_dir, _checkpoint_dir = write_full_ft_validator_fixture(tmp_path)
    runtime = FakeFullFTValidatorRuntime(answers_by_prompt={"첫 번째 질문": "", "두 번째 질문": ""})
    report_path = tmp_path / "tests/log/empty-generation-report.md"

    exit_code = validator.main(
        [
            "--config",
            str(config_path),
            "--model-path",
            str(final_export_dir),
            "--report-path",
            str(report_path),
        ],
        runtime=runtime,
    )

    report_text = report_path.read_text(encoding="utf-8")
    assert exit_code == 1
    assert "- Run Status: failed" in report_text
    assert "all sampled generations were empty" in report_text


def test_v8_full_ft_validator_default_report_path_uses_run_id_naming(tmp_path: Path):
    validator, config_path, _cfg, final_export_dir, _checkpoint_dir = write_full_ft_validator_fixture(tmp_path)
    runtime = FakeFullFTValidatorRuntime(
        answers_by_prompt={
            "첫 번째 질문": "기본 경로 답변 1",
            "두 번째 질문": "기본 경로 답변 2",
        }
    )

    result = validator.run_validation(
        config_path=config_path,
        model_path=str(final_export_dir),
        runtime=runtime,
    )

    report_path = Path(result["report_path"])
    assert report_path.name == "v8_round1_full_ft_model-path-final-export-gpt-oss-20b-seed-v8-round1-full-ft_report.md"
    assert report_path.exists()
    assert str(report_path.parent).endswith("tests/log")


def test_v8_full_ft_validator_missing_model_report_still_includes_prompt_source_path(tmp_path: Path):
    validator, config_path, cfg, _final_export_dir, _checkpoint_dir = write_full_ft_validator_fixture(tmp_path)
    missing_model_path = tmp_path / "missing-final-export"
    report_path = tmp_path / "tests/log/missing-model-report.md"

    exit_code = validator.main(
        [
            "--config",
            str(config_path),
            "--model-path",
            str(missing_model_path),
            "--report-path",
            str(report_path),
        ]
    )

    report_text = report_path.read_text(encoding="utf-8")
    assert exit_code == 1
    assert "- Run Status: failed" in report_text
    assert "- Prompt Source Path: " + str(Path(cfg["prompt_source_path"]).resolve()) in report_text
    assert "resolved model path not found" in report_text
