#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a document manifest for Zeroin seed_v2 dataset authoring."
    )
    parser.add_argument(
        "--docs-dir",
        required=True,
        help="Directory containing divide markdown documents.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path.",
    )
    return parser.parse_args()


def infer_doc_id(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def infer_title(path: Path, text: str) -> str:
    stem = path.stem
    if "_" in stem:
        return stem.split("_", 1)[1].replace("_", " ")

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return stem


def infer_kind(path: Path) -> str:
    if path.name in {"목차.md", "개정사항.md"}:
        return "supporting"
    return "chapter"


def first_heading(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir).resolve()
    output_path = Path(args.output).resolve()

    if not docs_dir.is_dir():
        raise SystemExit(f"docs directory not found: {docs_dir}")

    paths = sorted(docs_dir.glob("*.md"))
    if not paths:
        raise SystemExit(f"no markdown files found in: {docs_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for index, path in enumerate(paths, start=1):
            text = path.read_text(encoding="utf-8")
            title = infer_title(path, text)
            doc_id = infer_doc_id(path)
            record = {
                "seq": index,
                "doc_id": doc_id,
                "kind": infer_kind(path),
                "doc_path": path.as_posix(),
                "output_stub": path.stem,
                "title": title,
                "first_heading": first_heading(text),
            }
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")

    print(f"Wrote {len(paths)} records to {output_path}")


if __name__ == "__main__":
    main()
