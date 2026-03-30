#!/usr/bin/env python3
"""v5 seed round 병합기."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple v5 seed jsonl files into one deduplicated dataset.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input seed jsonl files in merge order.")
    parser.add_argument("--output", required=True, help="Merged output jsonl path.")
    parser.add_argument("--dataset-version", default="v5_round12", help="dataset_version to stamp on merged records.")
    parser.add_argument("--dataset-prefix", default="zeroin.seed_v5_round12", help="Merged record id prefix.")
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


def user_question(record: dict) -> str:
    messages = record.get("messages", [])
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return ""


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    if output_path.exists() and not args.force:
        raise SystemExit(f"output already exists: {output_path} (use --force to overwrite)")

    merged: list[dict] = []
    seen_questions: set[str] = set()
    next_index = 1

    for raw_input in args.inputs:
        input_path = Path(raw_input).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"input file not found: {input_path}")
        for record in read_jsonl(input_path):
            question = user_question(record)
            if not question or question in seen_questions:
                continue
            copied = json.loads(json.dumps(record, ensure_ascii=False))
            copied["id"] = f"{args.dataset_prefix}_{next_index:04d}"
            meta = copied.setdefault("meta", {})
            meta["dataset_version"] = args.dataset_version
            meta["merged_from"] = str(input_path)
            merged.append(copied)
            seen_questions.add(question)
            next_index += 1

    write_jsonl(output_path, merged)
    print(f"Merged {len(merged)} records -> {output_path}")


if __name__ == "__main__":
    main()
