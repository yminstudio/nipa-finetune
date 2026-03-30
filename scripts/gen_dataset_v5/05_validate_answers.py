#!/usr/bin/env python3
"""v5 답변 검증기."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


STATE_DIR = Path(__file__).resolve().parent / "state"
DEFAULT_INPUT = STATE_DIR / "answers_raw_ch01_round1.jsonl"
DEFAULT_OUTPUT = STATE_DIR / "answers_validated_ch01_round1.jsonl"
DEFAULT_REPORT = STATE_DIR / "validation_report_ch01_round1.json"

BAD_PHRASES = (
    "문서에 따르면",
    "원문에 따르면",
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
MAX_ANSWER_LEN = 900
MAX_ANSWER_LEN_CH2 = 1800
MAX_ANSWER_LEN_STRUCTURED = 2400
MIN_ANSWER_LEN = 20
GENERIC_TOKENS = {
    "무엇",
    "의미",
    "설명",
    "기준",
    "정의",
    "적용",
    "평가",
    "방법",
    "차이",
    "비교",
    "구분",
    "핵심",
    "개념",
    "국내",
    "국외",
    "유형",
    "펀드",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate v5 generated answers.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to answers_raw_ch01_round1.jsonl")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to answers_validated_ch01_round1.jsonl")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="Path to validation_report_ch01_round1.json")
    parser.add_argument("--force", action="store_true", help="Overwrite outputs even if they exist.")
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


def tokenize_korean_english(text: str) -> list[str]:
    tokens = re.findall(r"[가-힣A-Za-z][가-힣A-Za-z0-9/%+-]{1,30}", text)
    return [token for token in tokens if len(token) >= 2]


def normalize_tokens(tokens: list[str]) -> list[str]:
    normalized: list[str] = []
    for token in tokens:
        cleaned = re.sub(r"^\d+(?:\.\d+)*", "", token).strip()
        if not cleaned or cleaned in GENERIC_TOKENS:
            continue
        if cleaned not in normalized:
            normalized.append(cleaned)
    return normalized


def answer_length_limit(record: dict) -> int:
    answer_policy = str(record.get("answer_policy", "")).strip()
    if answer_policy == "structured_long":
        return MAX_ANSWER_LEN_STRUCTURED
    if answer_policy == "structured_medium":
        return MAX_ANSWER_LEN_CH2
    chapter = str(record.get("chapter", "")).strip()
    if chapter.startswith("2."):
        return MAX_ANSWER_LEN_CH2
    return MAX_ANSWER_LEN


def has_token_overlap(anchor_tokens: list[str], answer: str) -> bool:
    for token in anchor_tokens:
        if token in answer:
            return True
        if len(token) >= 4:
            prefix = token[: max(2, len(token) - 1)]
            if prefix in answer:
                return True
    return False


def detect_issues(record: dict) -> list[str]:
    answer = str(record.get("answer", "")).strip()
    question = str(record.get("question", "")).strip()
    normalized = re.sub(r"\s+", " ", answer)
    issues: list[str] = []
    max_answer_len = answer_length_limit(record)

    if not answer:
        issues.append("empty_answer")
        return issues
    if len(answer) < MIN_ANSWER_LEN:
        issues.append("too_short")
    if len(answer) > max_answer_len:
        issues.append("too_long")
    for phrase in BAD_PHRASES:
        if phrase in normalized:
            issues.append(f"bad_phrase:{phrase}")
    if re.search(r"(해주세요|해 주세요|하세요|바랍니다)(?:[.!\s]|$)", answer):
        issues.append("directive_tone")
    if re.search(r"<\|.*?\|>", answer):
        issues.append("strange_token")
    if answer.count("�") > 0:
        issues.append("broken_character")

    question_tokens = normalize_tokens(tokenize_korean_english(question))
    section_tokens = normalize_tokens(
        tokenize_korean_english(
            " ".join(
                [
                    str(record.get("section", "")),
                    str(record.get("subsection", "")),
                    str(record.get("seed_title", "")),
                    " ".join(str(item) for item in record.get("seed_nouns", [])),
                ]
            )
        )
    )
    anchor_tokens = []
    for token in question_tokens[:5] + section_tokens[:8]:
        if token not in anchor_tokens:
            anchor_tokens.append(token)

    # low_relevance는 강한 실패가 아니라 수동 검토 신호다.
    # 질문/섹션/시드 명사 어느 쪽과도 겹치지 않을 때만 걸어 오검출을 줄인다.
    if anchor_tokens and not has_token_overlap(anchor_tokens, answer):
        if len(answer) < 120 or len(question_tokens) >= 2:
            issues.append("low_relevance")

    return issues


def review_status_from_issues(issues: list[str]) -> str:
    if not issues:
        return "pass"
    severe = {"empty_answer", "strange_token", "broken_character"}
    if any(issue in severe for issue in issues):
        return "drop"
    return "manual_review"


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    if (output_path.exists() or report_path.exists()) and not args.force:
        raise SystemExit("validation outputs already exist (use --force to overwrite)")

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    records = read_jsonl(input_path)
    validated: list[dict] = []
    failures: list[dict] = []

    for record in records:
        issues = detect_issues(record)
        review_status = review_status_from_issues(issues)
        validated_record = {
            **record,
            "validation_issues": issues,
            "review_status": review_status,
        }
        validated.append(validated_record)
        if issues:
            failures.append({"id": record.get("id", "?"), "issues": issues, "review_status": review_status})

    write_jsonl(output_path, validated)

    passed = sum(1 for record in validated if record["review_status"] == "pass")
    report = {
        "total": len(validated),
        "passed": passed,
        "manual_review": sum(1 for record in validated if record["review_status"] == "manual_review"),
        "dropped": sum(1 for record in validated if record["review_status"] == "drop"),
        "failures": failures,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Validated {len(validated)} answers -> {output_path}")
    print(f"Report saved -> {report_path}")


if __name__ == "__main__":
    main()
