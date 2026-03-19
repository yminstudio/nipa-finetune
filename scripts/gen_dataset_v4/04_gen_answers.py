#!/usr/bin/env python3
"""v4 답변 생성기.

필터링된 질문과 업로드된 `only_text.md`를 바탕으로 짧고 구조적인 답변을 만든다.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = Path(__file__).resolve().parent / "state"
DEFAULT_INPUT = STATE_DIR / "questions_filtered.jsonl"
DEFAULT_OUTPUT = STATE_DIR / "answers_raw.jsonl"
DEFAULT_DOC = ROOT / "docs/제로인방법론/Zeroin 펀드평가 방법론 - only_text.md"
DEFAULT_UPLOAD_STATE = STATE_DIR / "upload_state.json"
LEGACY_UPLOAD_STATE = ROOT / "scripts/gen_dataset_v3/state/upload_state.json"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
DEFAULT_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

SYSTEM_PROMPT = """당신은 제로인 방법론에 근거해 답변하는 도메인 어시스턴트다.
반드시 한국어로만 답한다.
업로드된 문서에 근거가 없는 기준, 수치, 예외를 만들지 않는다.
답변은 짧고 구조적으로 유지한다.
출처/원문/문서/장/절을 직접 언급하지 않는다.
일반 금융 상식으로 확장하지 않는다."""

ANSWER_PROMPT = """업로드된 문서에서 아래 범위에 해당하는 내용만 근거로 질문에 답해줘.

[단원]
{chapter}

[절]
{section}

[소절]
{subsection}

[핵심 제목]
{seed_title}

[핵심 명사]
{seed_nouns}

[참고 섹션 발췌]
\"\"\"
{section_content}
\"\"\"

[질문]
{question}

규칙:
- 답변 구조 정책:
  {answer_policy}
- 질문과 직접 관련된 내용만 답변
- 업로드된 문서에 없는 내용 추가 금지
- 원문 고유 표기(예: CE, ZI, %Rank) 외에는 불필요한 영문 혼용 금지
- "문서에 따르면", "원문에 따르면", "문서에 따른", "다음은 문서에 따른", "일반적으로는", "실무상" 같은 표현 금지
- 조언, 권유, 지시, 안내형 문장 금지
- "필요 시", "확인해 주세요", "점검해 주세요", "참고하세요", "유의하세요" 같은 마무리 금지
"""

BAD_PHRASES = (
    "문서에 따르면",
    "원문에 따르면",
    "문서에 따른",
    "다음은 문서에 따른",
    "문서에 근거한",
    "본 문서",
    "자료에 따르면",
    "일반적으로는",
    "보통은",
    "실무상",
    "필요 시",
    "확인해 주세요",
    "점검해 주세요",
    "참고하세요",
    "유의하세요",
    "살펴보세요",
    "확인 바랍니다",
    "질문:",
    "답변:",
)


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
    parser.add_argument("--doc", default=str(DEFAULT_DOC), help="Uploaded source markdown path.")
    parser.add_argument(
        "--upload-state",
        default=str(DEFAULT_UPLOAD_STATE),
        help="Path to upload_state.json containing file_id. Falls back to v3 upload state when missing.",
    )
    parser.add_argument("--file-id", help="Explicit OpenAI file id. Overrides upload-state.")
    parser.add_argument("--api-base", default=os.getenv("OPENAI_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N questions.")
    parser.add_argument("--force", action="store_true", help="Overwrite output even if it already exists.")
    return parser.parse_args()


def require_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return api_key.strip().replace("\r", "")


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


def parse_doc_sections(text: str) -> dict[str, str]:
    lines = text.splitlines()
    headings: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines):
        match = re.match(r"^(#{2,6})\s+(.+)$", line)
        if match:
            headings.append((index, len(match.group(1)), match.group(2).strip()))

    section_map: dict[str, str] = {}
    for idx, (line_no, level, title) in enumerate(headings):
        end = len(lines)
        for next_line_no, next_level, _ in headings[idx + 1 :]:
            if next_level <= level:
                end = next_line_no
                break
        section_map[title] = "\n".join(lines[line_no:end]).strip()
    return section_map


def find_section_content(section_map: dict[str, str], question: dict) -> str:
    keys = [
        str(question.get("subsection", "")).strip(),
        str(question.get("section", "")).strip(),
        str(question.get("seed_title", "")).strip(),
        str(question.get("chapter", "")).strip(),
    ]
    for key in keys:
        if not key:
            continue
        if key in section_map:
            return section_map[key]
        for title, content in section_map.items():
            if key in title or title in key:
                return content
    return ""


def resolve_file_id(file_id: str | None, upload_state_path: Path) -> str:
    if file_id:
        return file_id.strip().replace("\r", "")

    candidate_paths = [upload_state_path]
    if upload_state_path == DEFAULT_UPLOAD_STATE:
        candidate_paths.append(LEGACY_UPLOAD_STATE)

    for candidate in candidate_paths:
        if candidate.exists():
            data = json.loads(candidate.read_text(encoding="utf-8"))
            value = str(data.get("file_id", "")).strip()
            if value:
                return value.replace("\r", "")
    raise FileNotFoundError(
        f"upload state not found or file_id missing: {[str(path) for path in candidate_paths]}"
    )


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


def call_responses_with_file(
    *,
    api_base: str,
    model: str,
    api_key: str,
    file_id: str,
    user_prompt: str,
    max_retries: int,
) -> str:
    api_base = api_base.strip().replace("\r", "")
    model = model.strip().replace("\r", "")
    api_key = api_key.strip().replace("\r", "")
    file_id = file_id.strip().replace("\r", "")
    payload = {
        "model": model,
        "instructions": SYSTEM_PROMPT,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_file", "file_id": file_id},
                ],
            }
        ],
    }
    request = urllib.request.Request(
        url=f"{api_base}/responses",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                body = json.loads(response.read().decode("utf-8"))
                text = parse_response_text(body)
                if text:
                    return text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"responses call failed: {last_error!r}")


def has_bad_phrase(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip()
    return any(phrase in normalized for phrase in BAD_PHRASES)


def sanitize_answer(text: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        if any(phrase in line for phrase in BAD_PHRASES):
            continue
        if re.search(r"(해주세요|해 주세요|하세요|바랍니다)(?:[.!\s]|$)", line):
            continue
        cleaned_lines.append(raw_line.rstrip())

    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()
    return "\n".join(cleaned_lines).strip()


def classify_answer_policy(question: dict, section_content: str) -> dict[str, object]:
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
            "policy_text": "구조 보존형으로 답한다. 5~10개 불릿까지 허용하며, 조건/단계/표 항목/산식 의미를 빠뜨리지 않는다. 필요한 경우 짧은 도입 1문장 후 항목별로 정리한다.",
            "max_chars": 2400,
        }

    if structured:
        return {
            "policy_name": "structured_medium",
            "policy_text": "구조 보존형으로 답한다. 4~8개 불릿까지 허용하며, 표/절차/산식의 핵심 항목을 순서대로 정리한다. 긴 배경설명보다 항목 구조를 우선한다.",
            "max_chars": 2000,
        }

    if qa_type == "definition":
        return {
            "policy_name": "definition_compact",
            "policy_text": "정의형 답변으로 2~5개 불릿 또는 2~4문장으로 정리한다. 핵심 정의와 보조 설명만 남기고 과도한 확장은 피한다.",
            "max_chars": 1200,
        }

    if qa_type in {"criteria", "application"}:
        return {
            "policy_name": "criteria_medium",
            "policy_text": "기준/적용형 답변으로 4~8개 불릿까지 허용한다. 판단 조건, 적용 범위, 예외가 있으면 누락 없이 적는다.",
            "max_chars": 1600,
        }

    return {
        "policy_name": "comparison_medium",
        "policy_text": "비교형 답변으로 3~6개 불릿까지 허용한다. 차이점과 공통점 중 질문에 직접 필요한 요소만 정리한다.",
        "max_chars": 1400,
    }


def build_prompt(question: dict, section_content: str, answer_policy: dict[str, object]) -> str:
    return ANSWER_PROMPT.format(
        chapter=question.get("chapter", ""),
        section=question.get("section", ""),
        subsection=question.get("subsection", ""),
        seed_title=question.get("seed_title", ""),
        seed_nouns=", ".join(question.get("seed_nouns", [])),
        section_content=section_content,
        question=question.get("question", ""),
        answer_policy=answer_policy["policy_text"],
    )


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    if output_path.exists() and not args.force:
        raise SystemExit(f"output already exists: {output_path} (use --force to overwrite)")

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    doc_path = Path(args.doc).resolve()
    if not doc_path.exists():
        raise FileNotFoundError(f"document not found: {doc_path}")

    api_key = require_api_key()
    upload_state_path = Path(args.upload_state).resolve()
    file_id = resolve_file_id(args.file_id, upload_state_path)

    questions = read_jsonl(input_path)
    if args.limit > 0:
        questions = questions[: args.limit]

    section_map = parse_doc_sections(doc_path.read_text(encoding="utf-8"))

    results: list[dict] = []
    for index, question in enumerate(questions, start=1):
        section_content = find_section_content(section_map, question)
        answer_policy = classify_answer_policy(question, section_content)
        prompt = build_prompt(question, section_content=section_content, answer_policy=answer_policy)
        answer = call_responses_with_file(
            api_base=args.api_base,
            model=args.model,
            api_key=api_key,
            file_id=file_id,
            user_prompt=prompt,
            max_retries=args.max_retries,
        ).strip()
        answer = sanitize_answer(answer)
        status = "generated"
        issues: list[str] = []
        if not answer:
            status = "empty"
            issues.append("empty_answer")
        elif has_bad_phrase(answer):
            status = "needs_review"
            issues.append("bad_phrase")

        results.append(
            {
                **question,
                "answer_id": f"v4_a_{index:04d}",
                "answer": answer,
                "answer_source_file": doc_path.relative_to(ROOT).as_posix(),
                "answer_file_id": file_id,
                "answer_status": status,
                "answer_issues": issues,
                "answer_policy": answer_policy["policy_name"],
                "answer_max_chars": answer_policy["max_chars"],
            }
        )

    write_jsonl(output_path, results)
    print(f"Generated {len(results)} answers -> {output_path}")


if __name__ == "__main__":
    main()
