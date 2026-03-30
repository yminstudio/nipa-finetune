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


def test_seed_system_prompt_keeps_p6_independent_understanding_rule():
    module = load_module("scripts/gen_dataset_v5/06_build_seed_v5.py", "build_seed_v5")

    assert "답변은 독립적으로 이해 가능해야 하지만" in module.SYSTEM_CONTENT
    assert "질문의 대상을 짧게 다시 잡아 주는 수준" in module.SYSTEM_CONTENT


def test_answer_prompt_allows_short_context_bridge_without_expansion():
    module = load_module("scripts/gen_dataset_v5/04_gen_answers.py", "gen_answers_v5")

    question = {
        "chapter": "1. 유형분류 기준",
        "qa_type": "criteria",
        "question": "유형생성의 기본원칙에서 충분성, 비교성, 지속성은 각각 무엇을 뜻하나요?",
    }
    policy = module.classify_answer_policy(question)
    prompt = module.build_prompt(question, "", policy)

    assert "기본 맥락을 1문장 이내로 짚고" in policy["policy_text"]
    assert "독립적으로 이해 가능해야 하지만" in prompt
    assert "질문의 대상을 짧게 다시 잡아 주는 수준" in prompt


def test_validation_prompt_includes_system_message():
    module = load_module("scripts/check_gpt_oss_model_output.py", "check_gpt_oss_model_output")

    class DummyTokenizer:
        def __init__(self):
            self.messages = None

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            self.messages = messages
            return "PROMPT"

        def __call__(self, prompt_text, return_tensors="pt"):
            return {"prompt_text": prompt_text, "return_tensors": return_tensors}

    tokenizer = DummyTokenizer()
    encoded = module.build_prompt_inputs(tokenizer, "질문", "시스템 규칙")

    assert tokenizer.messages == [
        {"role": "system", "content": "시스템 규칙"},
        {"role": "user", "content": "질문"},
    ]
    assert encoded["prompt_text"].endswith(module.FINAL_CHANNEL_PREFIX)


def test_validation_system_prompt_can_be_loaded_from_prompt_source(tmp_path: Path):
    module = load_module("scripts/check_gpt_oss_model_output.py", "check_gpt_oss_model_output_for_prompt")

    prompt_source = tmp_path / "seed.jsonl"
    record = {
        "messages": [
            {"role": "system", "content": "학습용 시스템 프롬프트"},
            {"role": "user", "content": "질문"},
            {"role": "assistant", "content": "답변"},
        ]
    }
    prompt_source.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    system_prompt = module.resolve_system_prompt({"prompt_source_path": str(prompt_source)})

    assert "학습용 시스템 프롬프트" in system_prompt
    assert "답변은 독립적으로 이해 가능해야 하지만" in system_prompt
    assert "질문의 대상을 짧게 다시 잡아 주는 수준" in system_prompt
