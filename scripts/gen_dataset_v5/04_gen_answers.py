#!/usr/bin/env python3
"""v5 답변 생성기.

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
DEFAULT_INPUT = STATE_DIR / "questions_filtered_ch01_round1.jsonl"
DEFAULT_OUTPUT = STATE_DIR / "answers_raw_ch01_round1.jsonl"
DEFAULT_DOC = ROOT / "docs/제로인방법론/Zeroin 펀드평가 방법론 - only_text.md"
VECTOR_STORE_ID = "vs_68a80414c938819189ac784ba37c10ee"
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

SYSTEM_PROMPT = """제로인은 펀드평가사로 펀드평가의 방법론을 가지고 있습니다.
- 당신은 제로인 방법론에 근거해 답변하는 도메인 어시스턴트다.
- 반드시 한국어로만 답한다.
- 답변은 질문에 직접 필요한 범위를 넘기지 않되, 독립적으로 이해 가능하도록 필요한 짧은 전제 설명은 포함할 수 있다.
- 답변은 독립적으로 이해 가능해야 한다. 하지만 그 이해 가능성을 이유로 새 정보를 만들면 안 된다.
- 답변은 불필요하게 짧게 끊지 말고, 구조적으로 정리하되 필요한 연결 문장은 남긴다.
- 문서에 없는 내용은 추가하지 않되, 의미 연결을 위한 최소한의 설명만 허용한다.
- 이 최소 설명은 용어를 풀거나 생략된 주어와 관계를 잇는 수준, 또는 질문의 대상을 짧게 다시 잡아 주는 수준에 한한다.
- 답변은 자연스러운 문장으로 연결하되, 새로운 정보 추가 없이 기존 내용을 매끄럽게 표현한다.
- 질문이 묻지 않은 다른 판단 기준으로 확장하지 않는다.
- 출처/원문/문서/장/절을 직접 언급하지 않는다.
- 일반 금융 상식으로 확장하지 않는다.
- **반드시 답변은 독립적으로 이해 가능해야 한다.**"""

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
    parser = argparse.ArgumentParser(description="Generate grounded answers for v5 questions.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to questions_filtered_ch01_round1.jsonl")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to answers_raw_ch01_round1.jsonl")
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
    parser.add_argument("--request-timeout", type=float, default=120.0, help="Per-request timeout in seconds.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N questions.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing partial output file.")
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
    request_timeout: float,
) -> str:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("openai package is required to generate answers.") from exc

    client = OpenAI(
        api_key=api_key.strip().replace("\r", ""),
        base_url=api_base.strip().replace("\r", ""),
        timeout=request_timeout,
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
            "policy_text": "요약형으로 답한다. 먼저 질문의 대상이 놓인 기본 맥락을 짧게 짚고, 이어서 핵심 정의, 기준, 비교만 남긴다. 문맥 연결을 위한 최소 설명은 허용하되 표 전체를 장황하게 나열하지 않는다.",
        }

    if structured and qa_type in {"criteria", "application"}:
        return {
            "policy_name": "structured_long",
            "policy_text": "구조 보존형 기준/적용 답변으로 답한다. 질문이 가리키는 대상이나 절차의 맥락을 짧게 잡아 준 뒤, 필요한 조건, 적용 범위, 예외만 남긴다. 표 전체 해설이나 문서 밖 배경설명은 금지한다.",
        }

    if structured:
        return {
            "policy_name": "structured_medium",
            "policy_text": "구조 보존형 답변으로 답한다. 표, 절차, 산식이 무엇을 설명하는지 짧게 밝힌 뒤 핵심 항목만 순서대로 남기고, 문맥 연결을 위한 최소 설명 외의 배경설명은 넣지 않는다.",
        }

    if qa_type == "definition":
        return {
            "policy_name": "definition_compact",
            "policy_text": "정의형 답변으로 답한다. 필요하면 첫 문장에서 질문 대상의 기본 맥락을 짧게 짚고, 이어서 핵심 정의를 분명히 제시한다. 일반론 확장이나 새 기준 추가는 금지한다.",
        }

    if qa_type == "criteria":
        return {
            "policy_name": "criteria_medium",
            "policy_text": "기준형 답변으로 답한다. 먼저 그 기준이 무엇을 판단하기 위한 것인지 짧게 짚고, 이어서 판단 기준을 제시한다. 적용 절차, 다른 대상 비교, 일반 배경설명으로 확장하지 않는다.",
        }

    if qa_type == "application":
        return {
            "policy_name": "application_medium",
            "policy_text": "적용형 답변으로 답한다. 적용 대상과 맥락을 짧게 밝힌 뒤 적용 순서, 조건, 전환 규칙만 남기고 정의 재설명이나 불필요한 비교는 쓰지 않는다.",
        }

    return {
        "policy_name": "comparison_medium",
        "policy_text": "비교형 답변으로 답한다. 비교 대상이 놓인 공통 맥락을 짧게 밝힌 뒤 질문에 직접 필요한 비교축의 차이만 정리하고, 각 대상의 정의나 절차 설명으로 확장하지 않는다.",
    }


def resolve_type_hard_rules(qa_type: str) -> list[str]:
    qa_type = qa_type.strip()
    if qa_type == "definition":
        return [
            "- 정의형: 답변에 정의문이 반드시 포함되어야 한다",
            "- 정의형: 절차, 비교, 예외는 질문이 직접 묻지 않으면 포함하지 않는다",
        ]
    if qa_type == "criteria":
        return [
            "- 기준형: 답변은 판단 기준을 중심으로 제시하되, 그 기준이 무엇을 판단하기 위한 것인지 짧게 밝혀 독립적으로 이해 가능하게 한다",
            "- 기준형: 적용 절차나 비교 설명은 포함하지 않는다",
        ]
    if qa_type == "comparison":
        return [
            "- 비교형: 답변은 질문에 직접 필요한 비교축의 차이만 제시한다",
            "- 비교형: 각 대상의 정의, 절차, 예외 설명으로 확장하지 않는다",
        ]
    if qa_type == "application":
        return [
            "- 적용형: 답변은 적용 순서, 적용 조건, 전환 규칙만 제시한다",
            "- 적용형: 정의 재설명이나 불필요한 비교는 포함하지 않는다",
        ]
    return []


def build_prompt(question: dict, section_content: str, answer_policy: dict[str, object]) -> str:
    _ = section_content
    question_text = str(question.get("question", "")).strip()
    qa_type = str(question.get("qa_type", "")).strip()
    policy_text = str(answer_policy.get("policy_text", "")).strip()
    type_rules = resolve_type_hard_rules(qa_type)
    return "\n".join(
        [
            f"[질문]\n{question_text}",
            "",
            "[답변 스타일]",
            policy_text,
            "- 문서에 없는 내용은 추가하지 않되, 의미 연결을 위한 최소 설명만 허용한다",
            "- 이 최소 설명은 용어 풀이, 생략된 주어/관계 보완, 질문의 대상을 짧게 다시 잡아 주는 수준에 한한다",
            "- 답변은 독립적으로 이해 가능해야 하지만, 그 이해 가능성을 이유로 새 정보를 만들면 안 된다",
            "- 답변은 자연스러운 문장으로 연결하되, 새로운 정보 추가 없이 기존 내용을 매끄럽게 표현한다",
            "- 새로운 판단 기준, 규칙, 예외, 질문에 없는 대상/비교는 추가하지 않는다",
            *type_rules,
            "- 질문에 직접 필요한 기준만 남기되, 새 정보가 아닌 범위에서 짧은 전제 설명은 허용한다",
            "- 제목, 번호 섹션, 마크다운 헤더(예: ##, 1) )는 쓰지 않는다",
        ]
    ).strip()


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
    if output_path.exists() and not (args.force or args.resume):
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
    start_index = 0
    if output_path.exists() and args.resume:
        results = read_jsonl(output_path)
        start_index = len(results)
        if start_index:
            print(f"Resuming from {start_index} existing answers in {output_path}")

    total_questions = len(questions)
    for index, question in enumerate(questions[start_index:], start=start_index + 1):
        print(f"[{index}/{total_questions}] generating answer", flush=True)
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
            request_timeout=args.request_timeout,
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
                "answer_id": f"v5_a_{index:04d}",
                "answer": answer,
                "answer_source_file": doc_path.relative_to(ROOT).as_posix(),
                "answer_vector_store_id": VECTOR_STORE_ID,
                "answer_status": status,
                "answer_issues": issues,
                "answer_policy": answer_policy["policy_name"],
                "prompt_log_file": prompt_log_path.relative_to(ROOT).as_posix(),
            }
        )
        write_jsonl(output_path, results)
        print(f"[{index}/{total_questions}] wrote partial output -> {output_path}", flush=True)

    write_jsonl(output_path, results)
    print(f"Generated {len(results)} answers -> {output_path}")


if __name__ == "__main__":
    main()
