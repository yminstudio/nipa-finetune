#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = "당신은 제로인 펀드평가 방법론에 근거해 답변하는 도메인 어시스턴트입니다."
ROUND_NAME = "round1"
DATASET_OUTPUT_RELATIVE_PATH = "llm_datasets/seed_v8/seed_v8_round1_full_ft.jsonl"


def normalize_markdown_cell(text: str) -> str:
    normalized = text.strip()
    normalized = normalized.replace("<br>", "\n")
    normalized = normalized.replace("\\.", ".")
    return normalized.strip()


def parse_question_variants(text: str) -> list[str]:
    normalized = text.strip()
    if "<br>" in normalized:
        return [question.strip() for question in normalized.split("<br>") if question.strip()]
    return [question.strip() for question in re.split(r"(?<=\?)\s+", normalized) if question.strip()]


def parse_markdown_source(source_path: Path) -> dict[str, Any]:
    columns: list[str] | None = None
    groups: list[dict[str, Any]] = []

    for raw_line in source_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("| "):
            continue
        if line in {"|     |     |", "| --- | --- |"}:
            continue

        cells = line[2:-2].split(" | ", 1)
        if len(cells) != 2:
            raise ValueError(f"unexpected table row format: {raw_line}")

        left_cell, right_cell = (cell.strip() for cell in cells)
        if not left_cell and not right_cell:
            continue
        if set(left_cell) == {"-"} and set(right_cell) == {"-"}:
            continue
        if columns is None:
            columns = [left_cell, right_cell]
            continue

        questions = parse_question_variants(left_cell)
        if not questions:
            raise ValueError(f"empty question cell: {raw_line}")

        answer = normalize_markdown_cell(right_cell)
        if not answer:
            raise ValueError(f"empty answer cell: {raw_line}")

        groups.append(
            {
                "group_id": f"group{len(groups) + 1}",
                "questions": questions,
                "answer": answer,
            }
        )

    if columns != ["user", "assistant"]:
        raise ValueError(f"expected markdown columns ['user', 'assistant'], got {columns}")
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
                    "id": f"zeroin.seed_v8_{ROUND_NAME}_full_ft_{record_index:04d}",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": group["answer"]},
                    ],
                    "meta": {
                        "dataset_version": f"seed_v8_{ROUND_NAME}_full_ft",
                        "round": ROUND_NAME,
                        "group_id": group["group_id"],
                        "question_variant_index": question_index,
                        "training_mode": "full_ft",
                        "source_strategy": "markdown_table_br_expansion",
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
    resolved_source = Path(source_path).resolve() if source_path is not None else base / "scripts/data_source.md"
    parsed = parse_markdown_source(resolved_source)
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
        "training_mode": "full_ft",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the v8 round1 full fine-tuning dataset from scripts/data_source.md."
    )
    parser.add_argument("--root", default=str(ROOT), help="Repository root for output paths.")
    parser.add_argument("--source", default=None, help="Optional Markdown source table path.")
    args = parser.parse_args()

    summary = build_dataset(root=args.root, source_path=args.source)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
