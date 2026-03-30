#!/usr/bin/env python3
"""v4 최종 seed 빌더."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


STATE_DIR = Path(__file__).resolve().parent / "state"
DEFAULT_INPUT = STATE_DIR / "answers_validated.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "llm_datasets/seed_v4"
DEFAULT_DATASET_VERSION = "v4"
DEFAULT_DATASET_PREFIX = "zeroin.seed_v4"
DEFAULT_CHAPTER_OUTPUT_PREFIX = "seed_v4"

SYSTEM_CONTENT = (
    "당신은 제로인 펀드평가 방법론에 근거해 답변하는 도메인 어시스턴트입니다. "
    "한국어로 답변하며 출처나 원문을 직접 언급하지 않습니다. "
    "제로인 펀드평가 방법론에 명시되지 않은 기준, 수치, 예외는 추정하거나 임의로 추가하지 않습니다."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical v4 seed jsonl files.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to answers_validated.jsonl")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for chapter jsonl files")
    parser.add_argument("--dataset-version", default=DEFAULT_DATASET_VERSION, help="Dataset version metadata")
    parser.add_argument("--dataset-prefix", default=DEFAULT_DATASET_PREFIX, help="Record id prefix")
    parser.add_argument(
        "--chapter-output-prefix",
        default=DEFAULT_CHAPTER_OUTPUT_PREFIX,
        help="Output filename prefix for chapter jsonl files",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing chapter files.")
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


def chapter_to_id(chapter: str) -> str:
    match = re.match(r"^\s*(\d+)", chapter)
    if match:
        return match.group(1).zfill(2)
    return "xx"


def build_record(
    index: int,
    record: dict,
    *,
    dataset_version: str = DEFAULT_DATASET_VERSION,
    dataset_prefix: str = DEFAULT_DATASET_PREFIX,
) -> dict | None:
    question = str(record.get("question", "")).strip()
    answer = str(record.get("answer", "")).strip()
    if not question or not answer:
        return None
    if str(record.get("review_status", "")) != "pass":
        return None

    chapter = str(record.get("chapter", "")).strip()
    record_id = f"{dataset_prefix}_{index:04d}"
    return {
        "id": record_id,
        "messages": [
            {"role": "system", "content": SYSTEM_CONTENT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "meta": {
            "dataset_version": dataset_version,
            "chapter": chapter,
            "section": record.get("section", ""),
            "subsection": record.get("subsection", ""),
            "source_file": record.get("answer_source_file") or record.get("source_file", ""),
            "generation_mode": record.get("generation_mode", "title_heading_noun_to_grounded_qa"),
            "qa_type": record.get("qa_type", ""),
            "seed_title": record.get("seed_title", ""),
            "seed_nouns": record.get("seed_nouns", []),
            "question_template": record.get("question_template", ""),
            "review_status": record.get("review_status", ""),
            "original_id": record.get("original_id", record.get("id", "")),
            "answer_vector_store_id": record.get("answer_vector_store_id", ""),
        },
    }


def resolve_output_paths(output_dir: Path, *, chapter_output_prefix: str) -> tuple[Path, Path]:
    return (
        output_dir / f"{chapter_output_prefix}_ch01.jsonl",
        output_dir / f"{chapter_output_prefix}_ch02.jsonl",
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    output_dir = Path(args.output_dir).resolve()
    ch01_path, ch02_path = resolve_output_paths(
        output_dir,
        chapter_output_prefix=args.chapter_output_prefix,
    )
    if (ch01_path.exists() or ch02_path.exists()) and not args.force:
        raise SystemExit("chapter output files already exist (use --force to overwrite)")

    source_records = read_jsonl(input_path)
    chapter_records: dict[str, list[dict]] = {"01": [], "02": []}
    built_count = 0

    for record in source_records:
        built = build_record(built_count + 1, record)
        if built is None:
            continue
        chapter_id = chapter_to_id(str(record.get("chapter", "")))
        if chapter_id in chapter_records:
            built_count += 1
            built["id"] = f"{args.dataset_prefix}_{built_count:04d}"
            built["meta"]["dataset_version"] = args.dataset_version
            chapter_records[chapter_id].append(built)

    write_jsonl(ch01_path, chapter_records["01"])
    write_jsonl(ch02_path, chapter_records["02"])
    print(f"Built {len(chapter_records['01'])} records -> {ch01_path}")
    print(f"Built {len(chapter_records['02'])} records -> {ch02_path}")


if __name__ == "__main__":
    main()
