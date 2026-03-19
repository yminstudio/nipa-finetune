#!/usr/bin/env python3
"""v4 구조 추출기.

1~2단원 분할 문서를 읽어 `단원 제목 + 절/소절 제목 + 핵심 명사` 재료를 만든다.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = [
    ROOT / "docs/제로인방법론/divide/01_유형분류_기준.md",
    ROOT / "docs/제로인방법론/divide/02_펀드평가_방법론.md",
]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "state/structure.jsonl"

GENERIC_TERMS = {
    "가",
    "나",
    "다",
    "기준",
    "개요",
    "기본",
    "세부",
    "적용",
    "방법",
    "대상",
    "내용",
    "설명",
    "부분",
    "사항",
    "유형",
    "펀드",
    "국내펀드",
    "해외펀드",
    "국내",
    "해외",
    "제로인",
    "이상",
    "개정",
    "br",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract v4 structure materials from chapter documents.")
    parser.add_argument(
        "--input-files",
        nargs="*",
        default=[str(path) for path in DEFAULT_INPUTS],
        help="Markdown files to parse. Defaults to chapter 1-2 divide files.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="JSONL output path. Defaults to scripts/gen_dataset_v4/state/structure.jsonl",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output even if it already exists.")
    return parser.parse_args()


def strip_markdown(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    return text.strip()


def split_title_fragments(text: str) -> list[str]:
    raw_parts = re.split(r"/|,|·|:|\(|\)|\s+및\s+|\s+및|\s*->\s*", strip_markdown(text))
    parts = [part.strip() for part in raw_parts if part.strip()]
    return parts


def normalize_candidate(text: str) -> str:
    text = strip_markdown(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^\d+(?:\.\d+)*\s*", "", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip(" -:|>")
    return text


def is_valid_candidate(text: str) -> bool:
    if len(text) < 2 or len(text) > 40:
        return False
    if text in GENERIC_TERMS:
        return False
    if re.fullmatch(r"[0-9./%()-]+", text):
        return False
    if text.lower().startswith("step "):
        return False
    return bool(re.search(r"[가-힣A-Za-z]", text))


def extract_seed_nouns(seed_title: str, section_content: str) -> list[str]:
    candidates: list[str] = []

    candidates.append(seed_title)
    candidates.append(normalize_candidate(seed_title))
    candidates.extend(split_title_fragments(seed_title))

    for match in re.findall(r"\*\*([^*]+)\*\*", section_content):
        candidates.append(match)

    for line in section_content.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        bullet = re.match(r"^[-*]\s+(.+?)\s*:\s*", strip_markdown(cleaned))
        if bullet:
            candidates.append(bullet.group(1))

    for token in re.findall(r"[가-힣A-Za-z][가-힣A-Za-z0-9/%+-]{1,30}", strip_markdown(section_content)):
        candidates.append(token)

    unique: list[str] = []
    for raw in candidates:
        candidate = normalize_candidate(raw)
        if not is_valid_candidate(candidate):
            continue
        if candidate not in unique:
            unique.append(candidate)

    return unique[:12]


def read_sections(path: Path) -> list[dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    headings: list[tuple[int, int, str]] = []

    for index, line in enumerate(lines):
        match = re.match(r"^(#{2,4})\s+(.+)$", line)
        if not match:
            continue
        headings.append((index, len(match.group(1)), match.group(2).strip()))

    records: list[dict[str, object]] = []
    current_chapter = ""
    current_section = ""
    item_index = 1

    for idx, (line_no, level, title) in enumerate(headings):
        if level == 2:
            current_chapter = title
            current_section = ""
            continue
        if level == 3:
            current_section = title
        if level not in (3, 4):
            continue

        end = len(lines)
        for next_line_no, next_level, _ in headings[idx + 1 :]:
            if next_level <= level:
                end = next_line_no
                break

        content = "\n".join(lines[line_no:end]).strip()
        seed_title = title
        seed_nouns = extract_seed_nouns(seed_title=seed_title, section_content=content)

        records.append(
            {
                "id": f"{path.stem}_structure_{item_index:04d}",
                "chapter": current_chapter,
                "section": current_section if level == 4 else title,
                "subsection": title if level == 4 else "",
                "seed_title": seed_title,
                "seed_nouns": seed_nouns,
                "section_content": content,
                "source_file": path.relative_to(ROOT).as_posix(),
            }
        )
        item_index += 1

    return records


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    if output_path.exists() and not args.force:
        raise SystemExit(f"output already exists: {output_path} (use --force to overwrite)")

    input_paths = [Path(value).resolve() for value in args.input_files]
    all_records: list[dict[str, object]] = []
    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"input file not found: {input_path}")
        all_records.extend(read_sections(input_path))

    write_jsonl(output_path, all_records)
    print(f"Extracted {len(all_records)} structure records -> {output_path}")


if __name__ == "__main__":
    main()
