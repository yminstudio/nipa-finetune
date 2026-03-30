#!/usr/bin/env python3
"""질문 프롬프트 로그에서 raw 질문 jsonl을 복구한다."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = Path(__file__).resolve().parent / "state"
DEFAULT_STRUCTURE = STATE_DIR / "structure_ch01.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover raw question records from prompt log markdown files.")
    parser.add_argument("--log-dir", required=True, help="Prompt log directory to recover from.")
    parser.add_argument("--structure", default=str(DEFAULT_STRUCTURE), help="Path to structure_ch01.jsonl")
    parser.add_argument("--output", required=True, help="Recovered raw output jsonl path.")
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


def extract_section(text: str, title: str) -> str:
    pattern = rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)"
    match = re.search(pattern, text, flags=re.MULTILINE | re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def extract_extra_json(text: str) -> dict:
    section = extract_section(text, "Extra")
    match = re.search(r"```json\s*(\{.*?\})\s*```", section, flags=re.DOTALL)
    if not match:
        return {}
    return json.loads(match.group(1))


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip().rstrip("?!.")


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    if output_path.exists() and not args.force:
        raise SystemExit(f"output already exists: {output_path} (use --force to overwrite)")

    structure_records = {
        str(record.get("id", "")).strip(): record
        for record in read_jsonl(Path(args.structure).resolve())
    }
    log_dir = Path(args.log_dir).resolve()
    if not log_dir.is_dir():
        raise FileNotFoundError(f"log directory not found: {log_dir}")

    recovered: list[dict] = []
    seen_questions: set[str] = set()
    index = 1

    for log_path in sorted(log_dir.glob("*.md")):
        text = log_path.read_text(encoding="utf-8")
        record_id = extract_section(text, "Record ID").splitlines()[0].strip()
        question = extract_section(text, "Question").splitlines()[0].strip()
        if not record_id or not question:
            continue
        normalized = normalize_question(question)
        if not normalized or normalized in seen_questions:
            continue

        extra = extract_extra_json(text)
        structure_record = structure_records.get(record_id, {})
        recovered.append(
            {
                "id": f"v5_qraw_{index:04d}",
                "question": question,
                "qa_type": str(extra.get("qa_type", "")).strip(),
                "question_template": str(extra.get("question_template", "")).strip(),
                "chapter": extra.get("chapter", structure_record.get("chapter", "")),
                "section": extra.get("section", structure_record.get("section", "")),
                "subsection": extra.get("subsection", structure_record.get("subsection", "")),
                "seed_title": extra.get("seed_title", structure_record.get("seed_title", "")),
                "seed_nouns": structure_record.get("seed_nouns", []),
                "source_file": structure_record.get("source_file", ""),
                "generation_mode": "title_heading_noun_to_grounded_qa",
                "review_status": "pending",
                "source_record_id": record_id,
                "prompt_log_file": log_path.relative_to(ROOT).as_posix(),
            }
        )
        seen_questions.add(normalized)
        index += 1

    write_jsonl(output_path, recovered)
    print(f"Recovered {len(recovered)} raw questions -> {output_path}")


if __name__ == "__main__":
    main()
