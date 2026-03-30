#!/usr/bin/env python3
"""v5 최종 seed 빌더."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


STATE_DIR = Path(__file__).resolve().parent / "state"
DEFAULT_INPUT = STATE_DIR / "answers_validated_ch01_round1.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "llm_datasets/seed_v5"
DEFAULT_DATASET_VERSION = "v5"
DEFAULT_DATASET_PREFIX = "zeroin.seed_v5"
DEFAULT_CHAPTER_OUTPUT_PREFIX = "seed_v5_round1"

SYSTEM_CONTENT = (
    "당신은 제로인 펀드평가 방법론에 근거해 답변하는 도메인 어시스턴트입니다. "
    "한국어로 답변하며 출처나 원문을 직접 언급하지 않습니다. "
    "제로인 펀드평가 방법론에 명시되지 않은 기준, 수치, 예외는 추정하거나 임의로 추가하지 않습니다. "
    "답변은 독립적으로 이해 가능해야 하지만, 그 이해 가능성을 이유로 새 정보를 만들면 안 됩니다. "
    "문서에 없는 내용은 추가하지 않되, 의미 연결을 위한 최소한의 설명만 허용합니다. "
    "이 최소 설명은 용어 풀이, 생략된 주어/관계 보완, 질문의 대상을 짧게 다시 잡아 주는 수준에 한합니다. "
    "답변은 자연스러운 문장으로 연결하되, 새로운 정보 추가 없이 기존 내용을 매끄럽게 표현합니다. "
    "새로운 판단 기준, 규칙, 질문에 없는 대상이나 비교는 추가하지 않습니다."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical v5 seed jsonl files.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to answers_validated_ch01_round1.jsonl")
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
    appendix_match = re.match(r"^\s*X(?:\.?(\d+)|(\d+))", chapter, flags=re.IGNORECASE)
    if appendix_match:
        suffix = appendix_match.group(1) or appendix_match.group(2) or "x"
        return f"x{suffix}"
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


def chapter_output_suffix(chapter_id: str) -> str:
    if chapter_id.isdigit():
        return f"ch{chapter_id}"
    if chapter_id.startswith("x"):
        return f"ch{chapter_id}"
    return f"ch_{chapter_id}"


def resolve_output_path(output_dir: Path, *, chapter_output_prefix: str, chapter_id: str) -> Path:
    return output_dir / f"{chapter_output_prefix}_{chapter_output_suffix(chapter_id)}.jsonl"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    output_dir = Path(args.output_dir).resolve()
    source_records = read_jsonl(input_path)
    chapter_records: dict[str, list[dict]] = {}
    built_count = 0

    for record in source_records:
        built = build_record(built_count + 1, record)
        if built is None:
            continue
        chapter_id = chapter_to_id(str(record.get("chapter", "")))
        if chapter_id == "xx":
            continue
        built_count += 1
        built["id"] = f"{args.dataset_prefix}_{built_count:04d}"
        built["meta"]["dataset_version"] = args.dataset_version
        chapter_records.setdefault(chapter_id, []).append(built)

    output_paths = [
        resolve_output_path(output_dir, chapter_output_prefix=args.chapter_output_prefix, chapter_id=chapter_id)
        for chapter_id in chapter_records
    ]
    all_output_path = output_dir / f"{args.chapter_output_prefix}_all.jsonl"
    if not args.force:
        existing_paths = [path for path in output_paths + [all_output_path] if path.exists()]
        if existing_paths:
            existing_text = ", ".join(str(path) for path in existing_paths)
            raise SystemExit(f"chapter output files already exist (use --force to overwrite): {existing_text}")

    merged_records: list[dict] = []
    for chapter_id in sorted(chapter_records):
        chapter_path = resolve_output_path(output_dir, chapter_output_prefix=args.chapter_output_prefix, chapter_id=chapter_id)
        write_jsonl(chapter_path, chapter_records[chapter_id])
        merged_records.extend(chapter_records[chapter_id])
        print(f"Built {len(chapter_records[chapter_id])} records -> {chapter_path}")

    if not merged_records:
        raise SystemExit("no pass records available to build seed dataset")

    write_jsonl(all_output_path, merged_records)
    print(f"Built {len(merged_records)} records -> {all_output_path}")


if __name__ == "__main__":
    main()
