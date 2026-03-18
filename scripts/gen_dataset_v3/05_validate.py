#!/usr/bin/env python3
"""05_validate.py — seed_v3_all.jsonl 품질 검증."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


BAD_PHRASES = (
    "위 문서",
    "아래 문서",
    "문서 본문",
    "질문:",
    "답변:",
    "초안",
    "재작성 대상",
    "포맷:",
    "제약:",
    "주제:",
    "위 내용",
    "첨부 문서를 참고",
)

MIN_ANSWER_LEN = 50
MAX_ANSWER_LEN = 5000  # v3는 원문 구조 반영 상세 답변이므로 상한 완화


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate seed_v3_all.jsonl quality.")
    parser.add_argument(
        "--input",
        default=str(
            Path(__file__).resolve().parents[2] / "llm_datasets/seed_v3/seed_v3_all.jsonl"
        ),
    )
    parser.add_argument(
        "--report",
        default=str(Path(__file__).resolve().parent / "state/validation_report.json"),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def has_bad_phrase(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    return [p for p in BAD_PHRASES if p in normalized]


def validate_record(rec: dict) -> list[str]:
    issues = []
    messages = rec.get("messages", [])

    # messages 구조 검증
    roles = [m.get("role") for m in messages]
    if roles != ["system", "user", "assistant"]:
        issues.append(f"messages 구조 오류: {roles}")
        return issues

    user_content = messages[1].get("content", "").strip()
    assistant_content = messages[2].get("content", "").strip()

    # 빈 답변
    if not assistant_content:
        issues.append("빈 답변")
        return issues

    # 빈 질문
    if not user_content:
        issues.append("빈 질문")

    # 답변 길이
    if len(assistant_content) < MIN_ANSWER_LEN:
        issues.append(f"답변 너무 짧음 ({len(assistant_content)}자 < {MIN_ANSWER_LEN}자)")

    if len(assistant_content) > MAX_ANSWER_LEN:
        issues.append(f"답변 너무 김 ({len(assistant_content)}자 > {MAX_ANSWER_LEN}자)")

    # 메타 표현 잔존
    bad = has_bad_phrase(assistant_content)
    if bad:
        issues.append(f"메타 표현 잔존: {bad}")

    bad_q = has_bad_phrase(user_content)
    if bad_q:
        issues.append(f"질문에 메타 표현: {bad_q}")

    return issues


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}\nRun 04_build_jsonl.py first.")

    records = read_jsonl(input_path)
    total = len(records)
    failures: list[dict] = []

    for rec in records:
        issues = validate_record(rec)
        if issues:
            failures.append({"id": rec.get("id", "?"), "issues": issues})

    passed = total - len(failures)

    # 터미널 출력
    print(f"\n{'='*50}")
    print(f"검증 결과: {total}건 중 {passed}건 통과, {len(failures)}건 실패")
    print(f"{'='*50}")
    if failures:
        print("\n[실패 목록]")
        for f in failures:
            print(f"  {f['id']}: {', '.join(f['issues'])}")
    else:
        print("모든 레코드 통과!")

    # 리포트 저장
    report = {
        "total": total,
        "passed": passed,
        "failed": len(failures),
        "pass_rate": round(passed / total * 100, 1) if total else 0,
        "failures": failures,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport saved: {report_path}")

    # 실패 있으면 비정상 종료
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
