from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path: str, module_name: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_train_dataset_accepts_message_records(tmp_path: Path):
    module = load_module("scripts/run_smoke_gpt_oss_20b.py", "run_smoke_gpt_oss_20b")

    class DummyTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            assert tokenize is False
            return " | ".join(f"{item['role']}:{item['content']}" for item in messages)

    dataset_path = tmp_path / "single_qa.jsonl"
    record = {
        "id": "single_qa_0001",
        "messages": [
            {"role": "system", "content": "시스템"},
            {"role": "user", "content": "질문"},
            {"role": "assistant", "content": "답변"},
        ],
    }
    dataset_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    dataset = module.load_train_dataset(dataset_path, DummyTokenizer())

    assert len(dataset) == 1
    assert dataset[0]["id"] == "single_qa_0001"
    assert dataset[0]["rendered_text"] == "system:시스템 | user:질문 | assistant:답변"


def test_v7_config_is_single_qa_local_training_only():
    config_path = ROOT / "configs/gpt_oss_20b_seed_v7_single_qa.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    assert cfg["dataset_path"].endswith("llm_datasets/seed_v7/seed_v7_single_qa.jsonl")
    assert "prompt_source_path" not in cfg
    assert cfg["max_steps"] == 960
    assert "validation_report_path" not in cfg
    assert "validation_questions" not in cfg
    assert "checkpoint_candidates" not in cfg


def test_v7_dataset_is_single_fixed_qa_record():
    dataset_path = ROOT / "llm_datasets/seed_v7/seed_v7_single_qa.jsonl"
    lines = [line for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(lines) == 1
    record = json.loads(lines[0])

    assert record["messages"][1]["content"] == "유형생성의 기본원칙에서 충분성, 비교성, 지속성은 각각 무엇을 뜻하나요?"
    assert "하나의 펀드 유형이 의미 있게 성립하기 위해 필요한 세 가지 조건" in record["messages"][2]["content"]
    assert "너무 소수이거나 특정 운용사에만 해당하면 독립된 유형으로 보기 어렵습니다." in record["messages"][2]["content"]


def test_render_markdown_report_writes_readable_smoke_log():
    module = load_module("tests/run_v7_single_qa_smoke.py", "run_v7_single_qa_smoke")

    report = {
        "status": "success",
        "model_name": "unsloth/gpt-oss-20b-BF16",
        "dataset_path": "/tmp/seed.jsonl",
        "output_dir": "/tmp/adapter",
        "resolved_target_modules": ["q_proj", "v_proj"],
        "train_metrics": {"train_loss": 0.1234},
        "system_prompt": "당신은 제로인 펀드평가 방법론에 근거해 답변하는 도메인 어시스턴트입니다.",
        "prompt_source_path": "/tmp/prompt.jsonl",
        "sample_inference": [
            {
                "id": "single_qa_0001",
                "prompt": "질문",
                "generation_final": "최종 답변",
                "generation_raw": "최종 답변<|return|>",
            }
        ],
    }

    rendered = module.render_markdown_report(report)

    assert "# GPT-OSS Smoke Report" in rendered
    assert "## Sample Inference" in rendered
    assert "- status: `success`" in rendered
    assert "## System Prompt" in rendered
    assert "당신은 제로인 펀드평가 방법론에 근거해 답변하는 도메인 어시스턴트입니다." in rendered
    assert "### single_qa_0001" in rendered
    assert "질문" in rendered
    assert "최종 답변" in rendered


def test_v7_smoke_config_writes_markdown_log_under_tests_log():
    config_path = ROOT / "tests/smoke_v7_single_qa_config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    assert cfg["adapter_path"].endswith("llm_model_lora/gpt-oss-20b-seed-v7-single-qa")
    assert cfg["prompt_source_path"].endswith("tests/smoke_v7_single_qa_prompt.jsonl")
    assert cfg["report_path"].endswith("tests/log/v7_single_qa_smoke_report.md")


def test_v7_smoke_prompt_source_contains_single_question():
    prompt_path = ROOT / "tests/smoke_v7_single_qa_prompt.jsonl"
    lines = [line for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["messages"][0]["role"] == "user"
    assert record["messages"][0]["content"].strip()


def test_build_smoke_messages_prepends_fixed_system_prompt():
    module = load_module("tests/run_v7_single_qa_smoke.py", "run_v7_single_qa_smoke")

    messages = [{"role": "user", "content": "질문"}]
    normalized = module.build_smoke_messages(messages)

    assert normalized[0]["role"] == "system"
    assert normalized[0]["content"] == "당신은 제로인 펀드평가 방법론에 근거해 답변하는 도메인 어시스턴트입니다."
    assert normalized[1:] == messages


def test_build_smoke_messages_ignores_training_assistant_turns():
    module = load_module("tests/run_v7_single_qa_smoke.py", "run_v7_single_qa_smoke")

    messages = [
        {"role": "system", "content": "학습용 시스템"},
        {"role": "user", "content": "질문"},
        {"role": "assistant", "content": "정답"},
    ]

    normalized = module.build_smoke_messages(messages)

    assert normalized == [
        {"role": "system", "content": "당신은 제로인 펀드평가 방법론에 근거해 답변하는 도메인 어시스턴트입니다."},
        {"role": "user", "content": "질문"},
    ]


def test_v7_round2_config_targets_new_single_qa_and_1000_steps():
    config_path = ROOT / "configs/gpt_oss_20b_seed_v7_single_qa_round2.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    assert cfg["dataset_path"].endswith("llm_datasets/seed_v7/seed_v7_single_qa_round2.jsonl")
    assert cfg["output_dir"].endswith("llm_model_lora/gpt-oss-20b-seed-v7-single-qa-round2")
    assert cfg["report_path"].endswith("llm_model_lora/gpt-oss-20b-seed-v7-single-qa-round2/train_result.json")
    assert cfg["max_steps"] == 1000


def test_v7_round2_dataset_matches_requested_question_and_answer():
    dataset_path = ROOT / "llm_datasets/seed_v7/seed_v7_single_qa_round2.jsonl"
    lines = [line for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["messages"][1]["content"] == "제로인 펀드평가 방법론의 전체 구성은 어떻게 이루어져 있나요?"
    answer = record["messages"][2]["content"]
    assert "제로인 펀드평가 방법론은 펀드를 체계적으로 비교·평가하기 위해 여러 단계의 구성요소로 이루어져 있습니다." in answer
    assert "유형분류 → 평가 및 등급 산출 → 위험·성과 분석 → 포트폴리오 및 부가 분석" in answer


def test_early_stop_callback_triggers_when_loss_and_accuracy_threshold_met():
    module = load_module("scripts/run_smoke_gpt_oss_20b.py", "run_smoke_gpt_oss_20b")

    callback = module.MetricThresholdStopCallback(loss_threshold=1.0, accuracy_threshold=1.0)

    assert callback.should_stop_from_logs({"loss": 0.9, "mean_token_accuracy": 1.0}) is True
    assert callback.should_stop_from_logs({"loss": 1.1, "mean_token_accuracy": 1.0}) is False
    assert callback.should_stop_from_logs({"loss": 0.9, "mean_token_accuracy": 0.99}) is False


def test_v7_round2_config_enables_metric_threshold_stop():
    config_path = ROOT / "configs/gpt_oss_20b_seed_v7_single_qa_round2.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    assert cfg["stop_when_loss_below"] == 1.0
    assert cfg["stop_when_mean_token_accuracy_at_least"] == 1.0


def test_v7_round3_config_targets_multi_question_dataset():
    config_path = ROOT / "configs/gpt_oss_20b_seed_v7_single_qa_round3.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    assert cfg["dataset_path"].endswith("llm_datasets/seed_v7/seed_v7_single_qa_round3.jsonl")
    assert cfg["output_dir"].endswith("llm_model_lora/gpt-oss-20b-seed-v7-single-qa-round3")
    assert cfg["report_path"].endswith("llm_model_lora/gpt-oss-20b-seed-v7-single-qa-round3/train_result.json")
    assert cfg["stop_when_loss_below"] == 1.0
    assert cfg["stop_when_mean_token_accuracy_at_least"] == 0.98


def test_v7_round3_dataset_contains_all_question_variants_with_shared_answer():
    dataset_path = ROOT / "llm_datasets/seed_v7/seed_v7_single_qa_round3.jsonl"
    lines = [line for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(lines) == 11
    records = [json.loads(line) for line in lines]
    questions = [record["messages"][1]["content"] for record in records]
    answers = [record["messages"][2]["content"] for record in records]

    assert questions[0] == "제로인 펀드평가 방법론의 전체 구성은 어떻게 이루어져 있나요?"
    assert "제로인 펀드평가 방법론은 어떤 단계로 구성되어 있나요?" in questions
    assert "제로인 펀드평가 방법론의 전반적인 구성과 흐름을 설명해 주세요." in questions
    assert len(set(answers)) == 1
    assert "유형분류 → 평가 및 등급 산출 → 위험·성과 분석 → 포트폴리오 및 부가 분석" in answers[0]


def test_v7_round4_config_targets_multi_group_dataset_from_base_model():
    config_path = ROOT / "configs/gpt_oss_20b_seed_v7_single_qa_round4.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    assert cfg["dataset_path"].endswith("llm_datasets/seed_v7/seed_v7_single_qa_round4.jsonl")
    assert cfg["output_dir"].endswith("llm_model_lora/gpt-oss-20b-seed-v7-single-qa-round4")
    assert cfg["report_path"].endswith("llm_model_lora/gpt-oss-20b-seed-v7-single-qa-round4/train_result.json")
    assert cfg["max_steps"] == 1000
    assert cfg["stop_when_loss_below"] == 1.0
    assert cfg["stop_when_mean_token_accuracy_at_least"] == 0.98
    assert "init_adapter_path" not in cfg


def test_v7_round4_dataset_contains_66_records_from_six_prompt_groups():
    dataset_path = ROOT / "llm_datasets/seed_v7/seed_v7_single_qa_round4.jsonl"
    lines = [line for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(lines) == 66
    records = [json.loads(line) for line in lines]
    questions = [record["messages"][1]["content"] for record in records]
    answers = [record["messages"][2]["content"] for record in records]
    group_ids = {record["meta"]["group_id"] for record in records}

    assert questions[0] == "유형생성의 기본원칙에서 충분성, 비교성, 지속성은 각각 무엇을 뜻하나요?"
    assert "제로인 방법론은 펀드를 어떤 기준과 절차로 평가하도록 설계되어 있나요?" in questions
    assert "제로인 펀드평가 방법론에서 유형분류, BM 설정, 성과평가는 어떤 기능적 차이가 있나요?" in questions
    assert any("유형은 충분한 규모가 있고(충분성), 서로 비교 가능하며(비교성), 장기적으로 유지될 수 있어야(지속성) 의미 있는 분류로 인정됩니다." in answer for answer in answers)
    assert any("유형분류 → BM 설정 → 평가대상 선정 → 위험조정 성과 계산 → ZI 표준화 및 순위화 → 등급 부여의 흐름으로 이루어집니다." in answer for answer in answers)
    assert "유형분류는 비교 집단을 정의하고," in answers[-1]
    assert group_ids == {"group1", "group2", "group3", "group4", "group5", "group6"}


def test_v7_round_builder_infers_round5_and_expands_data_source(tmp_path: Path):
    module = load_module("scripts/build_v7_round_from_data_source.py", "build_v7_round_from_data_source")

    source_path = ROOT / "scripts/data_source.md"
    groups = module.parse_markdown_table_groups(source_path)

    (tmp_path / "configs").mkdir()
    (tmp_path / "llm_datasets/seed_v7").mkdir(parents=True)
    (tmp_path / "llm_model_lora").mkdir()
    (tmp_path / "tests").mkdir()
    for round_number in (2, 3, 4):
        (tmp_path / "configs" / f"gpt_oss_20b_seed_v7_single_qa_round{round_number}.json").write_text(
            "{}\n",
            encoding="utf-8",
        )

    assert module.infer_next_round(tmp_path) == 5
    assert len(groups) == 5
    assert sum(len(group["questions"]) for group in groups) == 150
    assert groups[0]["questions"][0] == "유형생성의 기본원칙에서 충분성, 비교성, 지속성은 각각 무엇을 뜻하나요?"
    assert groups[-1]["questions"][-1] == "펀드평가 체계에서 유형분류, 벤치마크 설정, 성과평가는 어떤 구조적 역할을 기반으로 구성되나요?"


def test_v7_round5_config_targets_new_dataset_from_base_model():
    config_path = ROOT / "configs/gpt_oss_20b_seed_v7_single_qa_round5.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    assert cfg["model_name"] == "unsloth/gpt-oss-20b-BF16"
    assert cfg["dataset_path"].endswith("llm_datasets/seed_v7/seed_v7_single_qa_round5.jsonl")
    assert cfg["output_dir"].endswith("llm_model_lora/gpt-oss-20b-seed-v7-single-qa-round5")
    assert cfg["report_path"].endswith("llm_model_lora/gpt-oss-20b-seed-v7-single-qa-round5/train_result.json")
    assert cfg["max_steps"] == 1000
    assert cfg["stop_when_loss_below"] == 1.0
    assert cfg["stop_when_mean_token_accuracy_at_least"] == 0.98
    assert "init_adapter_path" not in cfg


def test_v7_round5_dataset_contains_150_records_from_five_prompt_groups():
    dataset_path = ROOT / "llm_datasets/seed_v7/seed_v7_single_qa_round5.jsonl"
    lines = [line for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(lines) == 150
    records = [json.loads(line) for line in lines]
    group_ids = {record["meta"]["group_id"] for record in records}

    assert records[0]["id"] == "zeroin.seed_v7_single_qa_round5_0001"
    assert records[-1]["id"] == "zeroin.seed_v7_single_qa_round5_0150"
    assert records[0]["messages"][0]["content"] == "당신은 제로인 펀드평가 방법론에 근거해 답변하는 도메인 어시스턴트입니다."
    assert records[0]["messages"][1]["content"] == "유형생성의 기본원칙에서 충분성, 비교성, 지속성은 각각 무엇을 뜻하나요?"
    assert records[-1]["messages"][1]["content"] == "펀드평가 체계에서 유형분류, 벤치마크 설정, 성과평가는 어떤 구조적 역할을 기반으로 구성되나요?"
    assert records[0]["meta"]["training_mode"] == "base_model_round5"
    assert records[0]["meta"]["round"] == "round5"
    assert group_ids == {"group1", "group2", "group3", "group4", "group5"}


def test_v7_round5_smoke_config_targets_all_questions_report():
    config_path = ROOT / "tests/smoke_v7_single_qa_round5_config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    assert cfg["adapter_path"].endswith("llm_model_lora/gpt-oss-20b-seed-v7-single-qa-round5")
    assert cfg["prompt_source_path"].endswith("llm_datasets/seed_v7/seed_v7_single_qa_round5.jsonl")
    assert cfg["report_path"].endswith("tests/log/v7_single_qa_round5_all_questions_report.md")
