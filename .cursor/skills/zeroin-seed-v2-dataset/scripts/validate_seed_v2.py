#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


REQUIRED_TOP_LEVEL = {"id", "messages", "meta"}
REQUIRED_META = {
    "dataset_version",
    "language",
    "chapter_id",
    "chapter_title",
    "section_path",
    "qa_type",
    "scope",
    "created_at",
    "source",
    "section_title",
    "difficulty",
    "answer_key_points",
    "tags",
}
REQUIRED_SOURCE = {"doc_path", "doc_hash", "anchors"}
ALLOWED_DIFFICULTY = {"easy", "medium", "hard"}
ALLOWED_DATASET_VERSIONS = {"v2", "v2.1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate seed_v2 JSONL records for Zeroin dataset authoring."
    )
    parser.add_argument("--input", required=True, help="Input JSON or JSONL file.")
    return parser.parse_args()


def iter_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("input file is empty")

    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError("input must be a JSON object, array, or JSONL")


def validate_timestamp(value: str) -> None:
    datetime.fromisoformat(value.replace("Z", "+00:00"))


def validate_record(record: dict, index: int) -> list[str]:
    errors: list[str] = []

    missing_top = REQUIRED_TOP_LEVEL - set(record.keys())
    if missing_top:
        errors.append(f"record {index}: missing top-level keys {sorted(missing_top)}")
        return errors

    if not isinstance(record["id"], str) or not record["id"].strip():
        errors.append(f"record {index}: id must be a non-empty string")

    messages = record["messages"]
    if not isinstance(messages, list) or len(messages) < 3:
        errors.append(f"record {index}: messages must be a list with at least 3 items")
    else:
        roles = [msg.get("role") for msg in messages if isinstance(msg, dict)]
        if roles[:3] != ["system", "user", "assistant"]:
            errors.append(
                f"record {index}: first three message roles must be system, user, assistant"
            )
        for msg_index, msg in enumerate(messages, start=1):
            if not isinstance(msg, dict):
                errors.append(f"record {index}: message {msg_index} must be an object")
                continue
            if not isinstance(msg.get("content"), str) or not msg["content"].strip():
                errors.append(
                    f"record {index}: message {msg_index} content must be a non-empty string"
                )

    meta = record["meta"]
    if not isinstance(meta, dict):
        errors.append(f"record {index}: meta must be an object")
        return errors

    missing_meta = REQUIRED_META - set(meta.keys())
    if missing_meta:
        errors.append(f"record {index}: missing meta keys {sorted(missing_meta)}")
        return errors

    if meta["dataset_version"] not in ALLOWED_DATASET_VERSIONS:
        errors.append(
            f"record {index}: meta.dataset_version must be one of {sorted(ALLOWED_DATASET_VERSIONS)}"
        )
    if meta["language"] != "ko":
        errors.append(f"record {index}: meta.language must be 'ko'")
    if meta["difficulty"] not in ALLOWED_DIFFICULTY:
        errors.append(f"record {index}: invalid difficulty {meta['difficulty']!r}")
    if not isinstance(meta["answer_key_points"], list) or not meta["answer_key_points"]:
        errors.append(f"record {index}: answer_key_points must be a non-empty list")
    if not isinstance(meta["tags"], list) or not meta["tags"]:
        errors.append(f"record {index}: tags must be a non-empty list")

    try:
        validate_timestamp(meta["created_at"])
    except Exception as exc:
        errors.append(f"record {index}: invalid created_at ({exc})")

    source = meta["source"]
    if not isinstance(source, dict):
        errors.append(f"record {index}: meta.source must be an object")
        return errors

    missing_source = REQUIRED_SOURCE - set(source.keys())
    if missing_source:
        errors.append(f"record {index}: missing source keys {sorted(missing_source)}")
        return errors

    doc_path = source["doc_path"]
    if not isinstance(doc_path, str) or "docs/제로인방법론/divide/" not in doc_path:
        errors.append(
            f"record {index}: meta.source.doc_path must point to docs/제로인방법론/divide/"
        )
    if not isinstance(source["anchors"], list):
        errors.append(f"record {index}: meta.source.anchors must be a list")
    if not isinstance(source["doc_hash"], str) or not source["doc_hash"].strip():
        errors.append(f"record {index}: meta.source.doc_hash must be a non-empty string")

    return errors


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise SystemExit(f"input file not found: {input_path}")

    records = iter_records(input_path)
    all_errors: list[str] = []

    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            all_errors.append(f"record {index}: record must be an object")
            continue
        all_errors.extend(validate_record(record, index))

    if all_errors:
        for error in all_errors:
            print(error)
        raise SystemExit(f"validation failed: {len(all_errors)} error(s)")

    print(f"validation passed: {len(records)} record(s)")


if __name__ == "__main__":
    main()
