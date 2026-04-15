#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ROUND_NAME = "round4"
SECTION_NAME = "section_all"
DATASET_OUTPUT_RELATIVE_PATH = "llm_datasets/seed_v8/seed_v8_round4_section_all_full_ft.jsonl"
DEFAULT_SOURCE_RELATIVE_PATH = "docs/제로인방법론/source_csv/v8-r4 section-all.csv"


def normalize_cell(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def parse_question_variants(text: str) -> list[str]:
    return [line.strip() for line in normalize_cell(text).splitlines() if line.strip()]


def parse_csv_source(source_path: Path) -> dict[str, Any]:
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = [str(name).strip() for name in (reader.fieldnames or [])]
        expected_columns = ["num", "sec", "system", "user_q_base", "assistant", "user_q_ext"]
        if columns != expected_columns:
            raise ValueError(f"expected CSV columns {expected_columns}, got {columns}")

        groups: list[dict[str, Any]] = []
        skipped_rows: list[str] = []
        for row in reader:
            if row is None or not any(str(value or "").strip() for value in row.values()):
                continue

            source_row_num = normalize_cell(str(row.get("num", "")))
            chapter = normalize_cell(str(row.get("sec", "")))
            system_prompt = normalize_cell(str(row.get("system", "")))
            assistant_text = normalize_cell(str(row.get("assistant", "")))
            base_question = normalize_cell(str(row.get("user_q_base", "")))
            extended_questions = parse_question_variants(str(row.get("user_q_ext", "")))

            if not extended_questions and base_question:
                extended_questions = [base_question]

            if not extended_questions:
                raise ValueError(f"empty user question variants in row: {source_row_num}")
            if not system_prompt:
                raise ValueError(f"empty system prompt in row: {source_row_num}")
            if not assistant_text:
                skipped_rows.append(source_row_num)
                continue

            groups.append(
                {
                    "group_id": f"group{len(groups) + 1}",
                    "source_row_num": source_row_num,
                    "chapter": chapter,
                    "questions": extended_questions,
                    "system": system_prompt,
                    "answer": assistant_text,
                }
            )

    if not groups:
        raise ValueError(f"no data rows found in {source_path}")

    return {"columns": expected_columns, "groups": groups, "skipped_rows": skipped_rows}


def build_dataset_records(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for group in groups:
        for question_index, question in enumerate(group["questions"], start=1):
            record_index = len(records) + 1
            records.append(
                {
                    "id": f"zeroin.seed_v8_{ROUND_NAME}_{SECTION_NAME}_full_ft_{record_index:04d}",
                    "messages": [
                        {"role": "system", "content": group["system"]},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": group["answer"]},
                    ],
                    "meta": {
                        "dataset_version": f"seed_v8_{ROUND_NAME}_{SECTION_NAME}_full_ft",
                        "round": ROUND_NAME,
                        "section": SECTION_NAME,
                        "chapter": group["chapter"],
                        "group_id": group["group_id"],
                        "question_variant_index": question_index,
                        "source_row_num": group["source_row_num"],
                        "training_mode": "full_ft",
                        "source_strategy": "csv_multiline_question_expansion",
                    },
                }
            )

    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
    path.write_text(content, encoding="utf-8")


def build_dataset(root: Path | str = ROOT, source_path: Path | str | None = None) -> dict[str, Any]:
    base = Path(root).resolve()
    resolved_source = (
        Path(source_path).resolve() if source_path is not None else (base / DEFAULT_SOURCE_RELATIVE_PATH).resolve()
    )
    parsed = parse_csv_source(resolved_source)
    records = build_dataset_records(parsed["groups"])
    dataset_path = (base / DATASET_OUTPUT_RELATIVE_PATH).resolve()
    write_jsonl(dataset_path, records)
    return {
        "source_path": str(resolved_source),
        "dataset_path": str(dataset_path),
        "columns": parsed["columns"],
        "group_count": len(parsed["groups"]),
        "record_count": len(records),
        "skipped_rows": parsed["skipped_rows"],
        "round": ROUND_NAME,
        "section": SECTION_NAME,
        "training_mode": "full_ft",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the v8 round4 section-all full fine-tuning dataset from the source CSV."
    )
    parser.add_argument("--root", default=str(ROOT), help="Repository root for output paths.")
    parser.add_argument("--source", default=None, help="Optional CSV source path.")
    args = parser.parse_args()

    summary = build_dataset(root=args.root, source_path=args.source)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
