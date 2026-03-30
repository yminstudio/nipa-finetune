#!/usr/bin/env python3
"""v5 질문 필터기.

질문 후보를 정규화하고 중복, 장황함, 범위 밖 패턴을 제거한다.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_INPUT = Path(__file__).resolve().parent / "state/questions_raw_ch01_round1.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "state/questions_filtered_ch01_round1.jsonl"

ALLOWED_QA_TYPES = {"definition", "criteria", "comparison", "application"}
BANNED_PHRASES = (
    "투자 추천",
    "매수",
    "매도",
    "포트폴리오",
    "시장 전망",
    "일반적으로",
    "보통은",
    "실무상",
    "어떻게 투자",
)
QUESTION_TEMPLATES = {
    "definition": "X는 무엇인가요?",
    "criteria": "X는 어떤 기준으로 판단하나요?",
    "comparison": "X와 Y의 차이는 무엇인가요?",
    "application": "어떤 경우에 X를 적용하나요?",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter and normalize raw v5 questions.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to questions_raw_ch01_round1.jsonl")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to questions_filtered_ch01_round1.jsonl")
    parser.add_argument(
        "--exclude-input",
        action="append",
        default=[],
        help="Existing filtered jsonl file whose normalized questions should be excluded.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output even if it already exists.")
    return parser.parse_args()


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


def normalize_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    normalized = normalized.rstrip("?!.")
    return normalized


def infer_qa_type(question: str) -> str:
    if "차이" in question or "비교" in question:
        return "comparison"
    if "어떤 경우" in question or "언제" in question or "적용" in question:
        return "application"
    if "기준" in question or "판단" in question or "조건" in question:
        return "criteria"
    return "definition"


def looks_like_title_copy(question: str, record: dict) -> bool:
    question_text = normalize_text(question)
    for field in ("seed_title", "section", "subsection", "chapter"):
        value = normalize_text(str(record.get(field, "")))
        if value and question_text == value:
            return True
    return False


def is_too_long(question: str) -> bool:
    return len(normalize_text(question)) > 90


def is_complex_question(question: str) -> bool:
    return question.count("?") > 1 or " 그리고 " in question or " 또한 " in question


def has_banned_phrase(question: str) -> bool:
    normalized = normalize_text(question)
    return any(phrase in normalized for phrase in BANNED_PHRASES)


def filter_records(records: list[dict], *, excluded_questions: set[str] | None = None) -> list[dict]:
    filtered: list[dict] = []
    seen_questions: set[str] = set(excluded_questions or set())
    question_index = 1

    for record in records:
        question = str(record.get("question", "")).strip()
        if not question:
            continue
        normalized = normalize_text(question)
        if normalized in seen_questions:
            continue
        if is_too_long(question):
            continue
        if is_complex_question(question):
            continue
        if has_banned_phrase(question):
            continue
        if looks_like_title_copy(question, record):
            continue

        qa_type = str(record.get("qa_type", "")).strip()
        if qa_type not in ALLOWED_QA_TYPES:
            qa_type = infer_qa_type(question)

        filtered.append(
            {
                "id": f"v5_q_{question_index:04d}",
                "original_id": record.get("id", ""),
                "question": question,
                "qa_type": qa_type,
                "question_template": record.get("question_template") or QUESTION_TEMPLATES[qa_type],
                "chapter": record.get("chapter", ""),
                "section": record.get("section", ""),
                "subsection": record.get("subsection", ""),
                "seed_title": record.get("seed_title", ""),
                "seed_nouns": record.get("seed_nouns", []),
                "source_file": record.get("source_file", ""),
                "generation_mode": record.get("generation_mode", "title_heading_noun_to_grounded_qa"),
                "review_status": "pass",
                "prompt_log_file": record.get("prompt_log_file", ""),
            }
        )
        seen_questions.add(normalized)
        question_index += 1

    return filtered


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    if output_path.exists() and not args.force:
        raise SystemExit(f"output already exists: {output_path} (use --force to overwrite)")

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    records = read_jsonl(input_path)
    excluded_questions: set[str] = set()
    for raw_path in args.exclude_input:
        exclude_path = Path(raw_path).resolve()
        if not exclude_path.exists():
            raise FileNotFoundError(f"exclude input file not found: {exclude_path}")
        for record in read_jsonl(exclude_path):
            question = str(record.get("question", "")).strip()
            if question:
                excluded_questions.add(normalize_text(question))

    filtered = filter_records(records, excluded_questions=excluded_questions)
    write_jsonl(output_path, filtered)
    print(f"Filtered {len(filtered)} questions -> {output_path}")


if __name__ == "__main__":
    main()
