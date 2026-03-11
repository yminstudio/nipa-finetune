#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def dataset_root() -> Path:
    return Path(__file__).resolve().parent


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"input file is empty: {path}")

    if path.suffix == ".jsonl":
        records = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"line {line_no}: each JSONL line must be an object")
            records.append(obj)
        return records

    data = json.loads(text)
    if isinstance(data, list):
        if not all(isinstance(item, dict) for item in data):
            raise ValueError("JSON array input must contain only objects")
        return data

    if isinstance(data, dict):
        return [data]

    raise ValueError("input must be a JSON object, JSON array, or JSONL file")


def default_output_path(input_path: Path) -> Path:
    rendered_dir = dataset_root() / "rendered" / "gpt-oss"
    return rendered_dir / f"{input_path.stem}_harmony.jsonl"


def render_records(
    records: list[dict[str, Any]],
    *,
    tokenizer_name: str,
    source_file: str,
) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    rendered_records: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        if "messages" not in record or not isinstance(record["messages"], list):
            raise ValueError(f"record {index}: missing messages list")

        rendered_text = tokenizer.apply_chat_template(record["messages"], tokenize=False)
        rendered_record: dict[str, Any] = {
            "id": record.get("id", f"record_{index:04d}"),
            "source_file": source_file,
            "rendered_text": rendered_text,
        }

        if "meta" in record:
            rendered_record["meta"] = record["meta"]

        rendered_records.append(rendered_record)

    return rendered_records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render canonical chat samples into GPT-OSS Harmony text."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to canonical dataset file (.json or .jsonl).",
    )
    parser.add_argument(
        "--output",
        help="Path to output JSONL. Defaults to llm_datasets/rendered/gpt-oss/<input>_harmony.jsonl",
    )
    parser.add_argument(
        "--tokenizer",
        default="openai/gpt-oss-20b",
        help="Tokenizer/model name used for apply_chat_template().",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    root = dataset_root()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve() if args.output else default_output_path(input_path)

    try:
        source_file = input_path.relative_to(root).as_posix()
    except ValueError:
        source_file = str(input_path)

    records = load_records(input_path)
    rendered_records = render_records(
        records,
        tokenizer_name=args.tokenizer,
        source_file=source_file,
    )
    write_jsonl(output_path, rendered_records)

    print(f"Rendered {len(rendered_records)} records")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
