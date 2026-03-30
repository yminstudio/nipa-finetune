import importlib.util
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "scripts/gen_dataset_v4/02_gen_questions.py"
ANSWERS_PATH = ROOT / "scripts/gen_dataset_v4/04_gen_answers.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


questions = load_module("gen_questions", QUESTIONS_PATH)
answers = load_module("gen_answers", ANSWERS_PATH)


class PromptContractTests(unittest.TestCase):
    def test_question_prompt_blocks_general_knowledge(self):
        self.assertIn(
            "업로드 문서를 보지 않고도 일반 금융 상식만으로 답할 수 있는 질문은 생성하지 않는다",
            questions.QUESTION_PROMPT,
        )
        self.assertIn("생성 후 자체 점검", questions.QUESTION_PROMPT)

    def test_question_model_prefers_q_openai_model(self):
        with mock.patch.dict(
            "os.environ",
            {"Q_OPENAI_MODEL": "gpt-question", "OPENAI_MODEL": "gpt-fallback"},
            clear=False,
        ):
            self.assertEqual(questions.resolve_question_model(), "gpt-question")

    def test_answer_system_prompt_demands_minimum_scope(self):
        self.assertIn("질문에 필요한 최소 범위까지만 답한다", answers.SYSTEM_PROMPT)
        self.assertIn("문서 밖 일반 지식으로 빈칸을 메우지 않는다", answers.SYSTEM_PROMPT)

    def test_answer_user_prompt_is_question_only(self):
        question = {
            "question": "제로인 펀드평가 대상 선정 기준에서 '선정'의 정의는 무엇인가?",
            "chapter": "2. 펀드평가 방법론",
            "section": "2.1. 제로인 펀드평가 대상 선정 기준",
            "subsection": "",
            "seed_title": "2.1. 제로인 펀드평가 대상 선정 기준",
            "seed_nouns": ["선정", "순위"],
        }

        prompt = answers.build_prompt(question, section_content="", answer_policy={"policy_text": "unused"})

        self.assertEqual(prompt, "제로인 펀드평가 대상 선정 기준에서 '선정'의 정의는 무엇인가?")

    def test_structured_policy_prefers_shorter_grounded_output(self):
        policy = answers.classify_answer_policy(
            {"chapter": "2.", "qa_type": "criteria", "section": "평가대상 선정"},
            "| 항목 | 기준 |\n| --- | --- |\n| 대상 선정 | 기준 |\n",
        )
        self.assertLessEqual(policy["max_chars"], 1400)

    def test_answer_model_prefers_a_openai_model(self):
        with mock.patch.dict(
            "os.environ",
            {"A_OPENAI_MODEL": "gpt-answer", "OPENAI_MODEL": "gpt-fallback"},
            clear=False,
        ):
            self.assertEqual(answers.resolve_answer_model(), "gpt-answer")

    def test_file_search_payload_uses_vector_store_tool(self):
        payload = answers.build_responses_payload(
            model="gpt-answer",
            user_prompt="질문 본문",
            vector_store_id="vs_test_123",
        )

        self.assertEqual(payload["model"], "gpt-answer")
        self.assertEqual(
            payload["input"][0],
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": answers.SYSTEM_PROMPT}],
            },
        )
        self.assertEqual(
            payload["tools"],
            [{"type": "file_search", "vector_store_ids": ["vs_test_123"]}],
        )
        self.assertEqual(
            payload["input"][1],
            {"role": "user", "content": [{"type": "input_text", "text": "질문 본문"}]},
        )
        self.assertEqual(payload["text"], {"format": {"type": "text"}, "verbosity": "medium"})
        self.assertEqual(payload["reasoning"], {"effort": "medium", "summary": "auto"})
        self.assertTrue(payload["store"])
        self.assertIn("reasoning.encrypted_content", payload["include"])

    def test_question_prompt_log_uses_question_text_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = questions.write_prompt_log(
                log_dir=Path(tmpdir),
                stage="questions",
                record_id="rec-001",
                question_text="펀드 %순위 산출 대상은 어떤 기준으로 선정하나요?",
                model="gpt-test",
                api_base="https://example.invalid/v1",
                system_prompt="system text",
                user_prompt="user text",
                extra={"chapter": "01"},
            )
            body = log_path.read_text(encoding="utf-8")

        self.assertEqual(log_path.name, "펀드 %순위 산출 대상은 어떤 기준으로 선정하나요?.md")
        self.assertEqual(log_path.suffix, ".md")
        self.assertIn("## Stage", body)
        self.assertIn("questions", body)
        self.assertIn("## Question", body)
        self.assertIn("펀드 %순위 산출 대상은 어떤 기준으로 선정하나요?", body)
        self.assertIn("system text", body)
        self.assertIn("user text", body)
        self.assertIn('"chapter": "01"', body)

    def test_answer_prompt_log_fallback_uses_question_text_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = answers.resolve_prompt_log_path(
                {"question": "제로인 선정 기준은 무엇인가?"},
                Path(tmpdir),
            )

        self.assertEqual(path.name, "제로인 선정 기준은 무엇인가?.md")

    def test_answer_prompt_log_appends_under_question_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = questions.write_prompt_log(
                log_dir=Path(tmpdir),
                stage="questions",
                record_id="v4_qraw_0001",
                question_text="리서치등급에서 정량평가와 정성평가는 각각 무엇을 보나요?",
                model="gpt-test",
                api_base="https://example.invalid/v1",
                system_prompt="question system",
                user_prompt="question user",
                extra={"chapter": "02"},
            )
            answers.append_prompt_log(
                log_path=log_path,
                stage="answers",
                record_id="v4_qraw_0001",
                model="gpt-answer",
                api_base="https://example.invalid/v1",
                system_prompt="answer system",
                user_prompt="answer user",
                extra={"vector_store_id": "vs-123"},
            )
            answers.append_response_log(
                log_path=log_path,
                answer_text="최종 답변 텍스트",
                extra={"answer_status": "generated"},
            )
            body = log_path.read_text(encoding="utf-8")

        self.assertEqual(log_path.name, "리서치등급에서 정량평가와 정성평가는 각각 무엇을 보나요?.md")
        self.assertEqual(log_path.suffix, ".md")
        self.assertEqual(body.count("## Stage"), 2)
        self.assertIn("question system", body)
        self.assertIn("answer system", body)
        self.assertIn("answer user", body)
        self.assertIn('"vector_store_id": "vs-123"', body)
        self.assertIn("## Assistant Response", body)
        self.assertIn("최종 답변 텍스트", body)


if __name__ == "__main__":
    unittest.main()
