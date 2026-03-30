#!/usr/bin/env python3
"""v4 답변 생성기.

필터링된 질문을 바탕으로 Responses API + file_search로 근거형 답변을 만든다.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = Path(__file__).resolve().parent / "state"
DEFAULT_PROMPT_LOG_DIR = Path(__file__).resolve().parent / "prompt_log"
DEFAULT_INPUT = STATE_DIR / "questions_filtered.jsonl"
DEFAULT_OUTPUT = STATE_DIR / "answers_raw.jsonl"
DEFAULT_DOC = ROOT / "docs/제로인방법론/Zeroin 펀드평가 방법론 - only_text.md"
VECTOR_STORE_ID = "vs_68a80414c938819189ac784ba37c10ee"
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

SYSTEM_PROMPT = """제로인은 펀드평가사로 펀드평가의 방법론을 가지고 있습니다.
당신은 제로인 방법론에 근거해 답변하는 도메인 어시스턴트다.
반드시 한국어로만 답한다.
업로드된 문서에 근거가 없는 기준, 수치, 예외를 만들지 않는다.
질문에 필요한 최소 범위까지만 답한다.
문서 밖 일반 지식으로 빈칸을 메우지 않는다.
답변은 짧고 구조적으로 유지한다.
출처/원문/문서/장/절을 직접 언급하지 않는다.
일반 금융 상식으로 확장하지 않는다."""

def load_dotenv_file() -> None:
    for dotenv_path in (ROOT / ".env", ROOT.parent / ".env"):
        if not dotenv_path.exists():
            continue
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"').replace("\r", "")
            if key and key not in os.environ:
                os.environ[key] = value


load_dotenv_file()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate grounded answers for v4 questions.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to questions_filtered.jsonl")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to answers_raw.jsonl")
    parser.add_argument(
        "--prompt-log-dir",
        default=str(DEFAULT_PROMPT_LOG_DIR),
        help="Directory to store prompt logs for AI API requests.",
    )
    parser.add_argument("--doc", default=str(DEFAULT_DOC), help="Uploaded source markdown path.")
    parser.add_argument(
        "--api-base",
        default=os.getenv("OPENAI_API_BASE", DEFAULT_API_BASE),
    )
    parser.add_argument("--model", default=resolve_answer_model())
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N questions.")
    parser.add_argument("--force", action="store_true", help="Overwrite output even if it already exists.")
    return parser.parse_args()


def require_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return api_key.strip().replace("\r", "")


def resolve_answer_model() -> str:
    for key in ("A_OPENAI_MODEL", "OPENAI_MODEL"):
        value = os.getenv(key, "").strip()
        if value:
            return value
    return DEFAULT_MODEL


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_response_text(body: dict) -> str:
    if isinstance(body.get("output_text"), str) and body["output_text"].strip():
        return body["output_text"].strip()

    parts: list[str] = []
    for item in body.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(part for part in parts if part).strip()


def build_responses_payload(*, model: str, user_prompt: str, vector_store_id: str) -> dict:
    return {
        "model": model.strip().replace("\r", ""),
        "input": [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            },
        ],
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "reasoning": {"effort": "medium", "summary": "auto"},
        "tools": [{"type": "file_search", "vector_store_ids": [vector_store_id]}],
        "store": True,
        "include": [
            "reasoning.encrypted_content",
            "web_search_call.action.sources",
        ],
    }


def call_responses_with_file_search(
    *,
    api_base: str,
    model: str,
    api_key: str,
    vector_store_id: str,
    user_prompt: str,
    max_retries: int,
) -> str:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("openai package is required to generate answers.") from exc

    client = OpenAI(
        api_key=api_key.strip().replace("\r", ""),
        base_url=api_base.strip().replace("\r", ""),
    )
    payload = build_responses_payload(
        model=model,
        user_prompt=user_prompt,
        vector_store_id=vector_store_id.strip().replace("\r", ""),
    )

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.responses.create(**payload)
            output_text = getattr(response, "output_text", None)
            if isinstance(output_text, str) and output_text.strip():
                return output_text.strip()
            if hasattr(response, "model_dump"):
                text = parse_response_text(response.model_dump())
                if text:
                    return text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"responses call failed: {last_error!r}")


def classify_answer_policy(question: dict, section_content: str = "") -> dict[str, object]:
    chapter = str(question.get("chapter", "")).strip()
    qa_type = str(question.get("qa_type", "")).strip()
    text = "\n".join(
        [
            str(question.get("section", "")),
            str(question.get("subsection", "")),
            str(question.get("seed_title", "")),
            section_content,
        ]
    )

    has_formula = "$$" in section_content or "Where" in section_content
    has_step_flow = "**Step" in section_content or "절차" in text or "산출 절차" in text
    has_table = "|" in section_content or "표" in text
    has_dense_conditions = text.count("- ") >= 6 or "기준" in text or "대상 선정" in text
    structured = has_formula or has_step_flow or has_table or has_dense_conditions

    if chapter.startswith("1."):
        return {
            "policy_name": "chapter1_summary",
            "policy_text": "1단원 요약형으로 답한다. 2~5개 불릿 또는 2~4문장으로 정리하고, 분류 원리와 핵심 비교만 남긴다. 표 전체를 장황하게 나열하지 않는다.",
            "max_chars": 900,
        }

    if structured and qa_type in {"criteria", "application"}:
        return {
            "policy_name": "structured_long",
            "policy_text": "구조 보존형 기준/적용 답변으로 3~6개 불릿까지 허용한다. 질문에 필요한 조건, 적용 범위, 예외만 남기고 표 전체 해설이나 긴 배경설명은 금지한다.",
            "max_chars": 1400,
        }

    if structured:
        return {
            "policy_name": "structured_medium",
            "policy_text": "구조 보존형 답변으로 3~5개 불릿까지 허용한다. 표/절차/산식의 핵심 항목만 순서대로 정리하고, 질문에 없는 배경설명은 넣지 않는다.",
            "max_chars": 1200,
        }

    if qa_type == "definition":
        return {
            "policy_name": "definition_compact",
            "policy_text": "정의형 답변으로 2~4개 불릿 또는 2~3문장으로 정리한다. 핵심 정의와 바로 필요한 보조 설명만 남기고 일반론 확장은 금지한다.",
            "max_chars": 900,
        }

    if qa_type in {"criteria", "application"}:
        return {
            "policy_name": "criteria_medium",
            "policy_text": "기준/적용형 답변으로 3~5개 불릿까지 허용한다. 판단 조건, 적용 범위, 예외만 남기고 일반 설명은 쓰지 않는다.",
            "max_chars": 1100,
        }

    return {
        "policy_name": "comparison_medium",
        "policy_text": "비교형 답변으로 3~5개 불릿까지 허용한다. 차이점과 공통점 중 질문에 직접 필요한 비교축만 정리한다.",
        "max_chars": 1000,
    }


def build_prompt(question: dict, section_content: str, answer_policy: dict[str, object]) -> str:
    _ = section_content, answer_policy
    return str(question.get("question", "")).strip()


def append_prompt_log(
    *,
    log_path: Path,
    stage: str,
    record_id: str,
    model: str,
    api_base: str,
    system_prompt: str,
    user_prompt: str,
    extra: dict,
) -> Path:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    section_title = "답변 프롬프트 로그" if stage == "answers" else f"{stage} 프롬프트 로그"
    body = "\n".join(
        [
            "",
            "---",
            "",
            "## Stage",
            stage,
            "",
            "## Record ID",
            record_id,
            "",
            "## Model",
            model,
            "",
            "## API Base",
            api_base,
            "",
            "## Extra",
            "```json",
            json.dumps(extra, ensure_ascii=False, indent=2),
            "```",
            "",
            "## System Prompt",
            "```text",
            system_prompt,
            "```",
            "",
            "## User Prompt",
            "```text",
            user_prompt,
            "```",
            "",
        ]
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(body)
    return log_path


def append_response_log(*, log_path: Path, answer_text: str, extra: dict) -> Path:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        [
            "",
            "---",
            "",
            "## Assistant Response",
            "```text",
            answer_text,
            "```",
            "",
            "## Response Extra",
            "```json",
            json.dumps(extra, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(body)
    return log_path


def resolve_prompt_log_path(question: dict, prompt_log_dir: Path) -> Path:
    value = str(question.get("prompt_log_file", "")).strip()
    if value:
        path = Path(value)
        return path if path.is_absolute() else (ROOT / path)

    safe_question = re.sub(r"[\\/\r\n\t]+", "_", str(question.get("question", "")).strip())
    safe_question = re.sub(r"\s+", " ", safe_question).strip().rstrip(". ")
    safe_question = (safe_question or "질문")[:80].rstrip(" ._")
    return prompt_log_dir / f"{safe_question}.md"


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    if output_path.exists() and not args.force:
        raise SystemExit(f"output already exists: {output_path} (use --force to overwrite)")

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")
    prompt_log_dir = Path(args.prompt_log_dir).resolve()

    doc_path = Path(args.doc).resolve()
    if not doc_path.exists():
        raise FileNotFoundError(f"document not found: {doc_path}")

    api_key = require_api_key()

    questions = read_jsonl(input_path)
    if args.limit > 0:
        questions = questions[: args.limit]

    results: list[dict] = []
    for index, question in enumerate(questions, start=1):
        answer_policy = classify_answer_policy(question)
        prompt = build_prompt(question, section_content="", answer_policy=answer_policy)
        record_id = str(question.get("id", f"question_{index:04d}"))
        prompt_log_path = resolve_prompt_log_path(question, prompt_log_dir)
        append_prompt_log(
            log_path=prompt_log_path,
            stage="answers",
            record_id=record_id,
            model=args.model,
            api_base=args.api_base,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            extra={
                "vector_store_id": VECTOR_STORE_ID,
                "chapter": question.get("chapter", ""),
                "section": question.get("section", ""),
                "subsection": question.get("subsection", ""),
                "seed_title": question.get("seed_title", ""),
                "question": question.get("question", ""),
                "answer_policy": answer_policy["policy_name"],
            },
        )
        answer = call_responses_with_file_search(
            api_base=args.api_base,
            model=args.model,
            api_key=api_key,
            vector_store_id=VECTOR_STORE_ID,
            user_prompt=prompt,
            max_retries=args.max_retries,
        ).strip()
        status = "generated"
        issues: list[str] = []
        if not answer:
            status = "empty"
            issues.append("empty_answer")

        append_response_log(
            log_path=prompt_log_path,
            answer_text=answer,
            extra={
                "answer_status": status,
                "answer_issues": issues,
                "answer_policy": answer_policy["policy_name"],
                "vector_store_id": VECTOR_STORE_ID,
            },
        )

        results.append(
            {
                **question,
                "answer_id": f"v4_a_{index:04d}",
                "answer": answer,
                "answer_source_file": doc_path.relative_to(ROOT).as_posix(),
                "answer_vector_store_id": VECTOR_STORE_ID,
                "answer_status": status,
                "answer_issues": issues,
                "answer_policy": answer_policy["policy_name"],
                "answer_max_chars": answer_policy["max_chars"],
                "prompt_log_file": prompt_log_path.relative_to(ROOT).as_posix(),
            }
        )

    write_jsonl(output_path, results)
    print(f"Generated {len(results)} answers -> {output_path}")


if __name__ == "__main__":
    main()
