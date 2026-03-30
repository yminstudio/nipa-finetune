#!/usr/bin/env python3
"""답변이 붙은 프롬프트 로그에서 raw 답변 jsonl을 복구한다."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover raw answers from prompt log markdown files.")
    parser.add_argument("--input", required=True, help="Filtered question jsonl path.")
    parser.add_argument("--output", required=True, help="Recovered raw answer jsonl path.")
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


def extract_last_json_block(text: str, title: str) -> dict:
    pattern = rf"## {re.escape(title)}\n```json\s*(\{{.*?\}})\s*```"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if not matches:
        return {}
    return json.loads(matches[-1])


def extract_last_text_block(text: str, title: str) -> str:
    pattern = rf"## {re.escape(title)}\n```text\s*(.*?)\s*```"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if not matches:
        return ""
    return matches[-1].strip()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    if output_path.exists() and not args.force:
        raise SystemExit(f"output already exists: {output_path} (use --force to overwrite)")

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    recovered: list[dict] = []
    for index, question in enumerate(read_jsonl(input_path), start=1):
        prompt_log_value = str(question.get("prompt_log_file", "")).strip()
        if not prompt_log_value:
            continue
        log_path = Path(prompt_log_value)
        log_path = log_path if log_path.is_absolute() else (ROOT / log_path)
        if not log_path.exists():
            continue

        text = log_path.read_text(encoding="utf-8")
        answer = extract_last_text_block(text, "Assistant Response")
        if not answer:
            continue
        extra = extract_last_json_block(text, "Response Extra")
        recovered.append(
            {
                **question,
                "answer_id": f"v5_a_{index:04d}",
                "answer": answer,
                "answer_source_file": "docs/제로인방법론/Zeroin 펀드평가 방법론 - only_text.md",
                "answer_vector_store_id": str(extra.get("vector_store_id", "")).strip(),
                "answer_status": str(extra.get("answer_status", "generated")).strip() or "generated",
                "answer_issues": extra.get("answer_issues", []),
                "answer_policy": str(extra.get("answer_policy", "")).strip(),
                "answer_max_chars": 0,
                "prompt_log_file": log_path.relative_to(ROOT).as_posix(),
            }
        )

    write_jsonl(output_path, recovered)
    print(f"Recovered {len(recovered)} answers -> {output_path}")


if __name__ == "__main__":
    main()
