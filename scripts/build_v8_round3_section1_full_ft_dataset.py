#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ROUND_NAME = "round3"
SECTION_NAME = "section1"
DATASET_OUTPUT_RELATIVE_PATH = "llm_datasets/seed_v8/seed_v8_round3_section1_full_ft.jsonl"
DEFAULT_SOURCE_RELATIVE_PATH = "docs/제로인방법론/source_csv/v8-r3 section-1.csv"


def normalize_cell(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def parse_question_variants(text: str) -> list[str]:
    return [line.strip() for line in normalize_cell(text).splitlines() if line.strip()]


def parse_csv_source(source_path: Path) -> dict[str, Any]:
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))

    if len(rows) < 3:
        raise ValueError(f"expected at least 3 rows in CSV source: {source_path}")

    columns = [cell.strip() for cell in rows[1]]
    expected_columns = ["num", "단원", "chat input", "system", "user", "assistant", "chat input"]
    if columns != expected_columns:
        raise ValueError(f"expected CSV columns {expected_columns}, got {columns}")

    groups: list[dict[str, Any]] = []
    for row in rows[2:]:
        if not any(cell.strip() for cell in row):
            continue
        padded = row + [""] * max(0, len(columns) - len(row))
        row_data = dict(zip(columns, padded[: len(columns)]))

        questions = parse_question_variants(row_data["user"])
        if not questions:
            raise ValueError(f"empty user question variants in row: {row_data.get('num', '')}")

        system_prompt = normalize_cell(row_data["system"])
        assistant_text = normalize_cell(row_data["assistant"])
        if not system_prompt:
            raise ValueError(f"empty system prompt in row: {row_data.get('num', '')}")
        if not assistant_text:
            raise ValueError(f"empty assistant answer in row: {row_data.get('num', '')}")

        groups.append(
            {
                "group_id": f"group{len(groups) + 1}",
                "source_row_num": normalize_cell(row_data["num"]),
                "chapter": normalize_cell(row_data["단원"]),
                "questions": questions,
                "system": system_prompt,
                "answer": assistant_text,
            }
        )

    if not groups:
        raise ValueError(f"no data rows found in {source_path}")

    return {"columns": columns, "groups": groups}


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
        "round": ROUND_NAME,
        "section": SECTION_NAME,
        "training_mode": "full_ft",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the v8 round3 section1 full fine-tuning dataset from the source CSV."
    )
    parser.add_argument("--root", default=str(ROOT), help="Repository root for output paths.")
    parser.add_argument("--source", default=None, help="Optional CSV source path.")
    args = parser.parse_args()

    summary = build_dataset(root=args.root, source_path=args.source)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
