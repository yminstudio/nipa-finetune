#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


SYSTEM_PROMPT = (
    "당신은 제로인 방법론에 근거해 답변하는 도메인 어시스턴트입니다. "
    "한국어로 답변합니다. "
    "상위 제목 질문은 반드시 하위 제목 전체를 포함해 요약합니다. "
    "문서에 없는 내용은 추가하지 않되, 의미 연결을 위한 최소한의 설명만 허용합니다. "
    "답변은 자연스러운 문장으로 연결하되, 새로운 정보 추가 없이 기존 내용을 매끄럽게 표현합니다. "
    "새로운 판단 기준, 규칙, 질문에 없는 대상이나 비교는 추가하지 않습니다."
)
CREATED_AT = "2026-03-16T00:00:00Z"


@dataclass
class Section:
    index: int
    level: int
    title: str
    start: int
    end: int
    path_titles: list[str]
    body: str
    parent_index: int | None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def docs_dir() -> Path:
    return repo_root() / "docs" / "제로인방법론" / "divide"


def output_dir() -> Path:
    return repo_root() / "llm_datasets" / "seed_v2"


def clean_heading(text: str) -> str:
    text = re.sub(r"`+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_markdown(text: str) -> str:
    text = text.replace("<br>", ", ").replace("<br/>", ", ").replace("<br />", ", ")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    text = re.sub(r"\$+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_sections(text: str) -> list[Section]:
    lines = text.splitlines()
    headings: list[tuple[int, int, str]] = []
    for i, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if match:
            headings.append((i, len(match.group(1)), clean_heading(match.group(2))))

    if not headings:
        return []

    sections: list[Section] = []
    stack: list[tuple[int, int, str]] = []

    for idx, (line_no, level, title) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()

        parent_index = stack[-1][1] if stack else None
        path_titles = [item[2] for item in stack] + [title]
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        body = "\n".join(lines[line_no + 1 : end]).strip()
        sections.append(
            Section(
                index=idx,
                level=level,
                title=title,
                start=line_no + 1,
                end=end,
                path_titles=path_titles,
                body=body,
                parent_index=parent_index,
            )
        )
        stack.append((level, idx, title))

    return sections


def remove_tables_and_formulas(text: str) -> str:
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    cleaned_lines: list[str] = []
    in_table = False
    for line in text.splitlines():
        if line.strip().startswith("|"):
            in_table = True
            continue
        if in_table and not line.strip():
            in_table = False
            continue
        if in_table:
            continue
        if re.match(r"^#{1,6}\s+", line):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def extract_text_points(text: str, limit: int = 6) -> list[str]:
    body = remove_tables_and_formulas(text)
    points: list[str] = []
    for raw in body.splitlines():
        line = strip_markdown(raw)
        if not line:
            continue
        if line in {"---", "Where"}:
            continue
        if re.match(r"^\|", line):
            continue
        points.append(line.lstrip("- ").strip())
    deduped: list[str] = []
    seen: set[str] = set()
    for point in points:
        if point not in seen:
            deduped.append(point)
            seen.add(point)
        if len(deduped) >= limit:
            break
    return deduped


def extract_table_blocks(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("|"):
            current.append(line.rstrip())
        else:
            if current:
                blocks.append(current)
                current = []
    if current:
        blocks.append(current)
    return blocks


def parse_table(block: list[str]) -> tuple[list[str], list[list[str]]]:
    rows: list[list[str]] = []
    for line in block:
        cols = [strip_markdown(col.strip()) for col in line.strip().strip("|").split("|")]
        rows.append(cols)

    meaningful_rows = [
        row
        for row in rows
        if any(cell for cell in row)
        and not all(set(cell) <= {"-"} for cell in row if cell)
    ]
    if len(meaningful_rows) < 2:
        return [], []

    header = meaningful_rows[0]
    data_rows = meaningful_rows[1:]
    return header, data_rows


def extract_formula_blocks(text: str) -> list[str]:
    return [block.strip() for block in re.findall(r"\$\$(.*?)\$\$", text, flags=re.DOTALL)]


def infer_chapter_id(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def infer_chapter_title(path: Path, sections: list[Section]) -> str:
    stem = path.stem
    if "_" in stem:
        return stem.split("_", 1)[1]
    if sections:
        return sections[0].title
    return stem


def infer_difficulty(section: Section) -> str:
    if extract_table_blocks(section.body) or extract_formula_blocks(section.body):
        return "hard"
    if len(extract_text_points(section.body, limit=10)) >= 4:
        return "medium"
    return "easy"


def infer_qa_type(section: Section) -> str:
    title = section.title
    if "기준" in title or "구분" in title or "등급" in title:
        return "rule"
    if extract_formula_blocks(section.body):
        return "formula"
    if "정의" in title or "개요" in title:
        return "definition"
    return "concept"


def make_record(
    rec_id: str,
    user: str,
    assistant: str,
    *,
    chapter_id: str,
    chapter_title: str,
    section_path: str,
    qa_type: str,
    section_title: str,
    difficulty: str,
    answer_key_points: list[str],
    tags: list[str],
    doc_path: str,
    doc_hash: str,
    anchors: list[str],
) -> dict:
    return {
        "id": rec_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "meta": {
            "dataset_version": "v2",
            "language": "ko",
            "chapter_id": chapter_id,
            "chapter_title": chapter_title,
            "section_path": section_path,
            "qa_type": qa_type,
            "scope": "in_scope",
            "created_at": CREATED_AT,
            "source": {
                "doc_path": doc_path,
                "doc_hash": doc_hash,
                "anchors": anchors,
            },
            "section_title": section_title,
            "difficulty": difficulty,
            "answer_key_points": answer_key_points[:5] or [section_title],
            "tags": tags,
        },
    }


def build_coverage_answer(section: Section, children: list[Section]) -> str:
    lines = [f"- **핵심 주제**: `{section.title}` 아래의 내용을 전체 구조 기준으로 정리합니다."]
    if children:
        lines.append(
            "- **하위 구성**: " + ", ".join(f"`{child.title}`" for child in children)
        )
        for child in children[:8]:
            points = extract_text_points(child.body, limit=1)
            if points:
                lines.append(f"- **{child.title}**: {points[0]}")
            else:
                lines.append(f"- **{child.title}**: 해당 세부 기준과 설명이 이어집니다.")
    else:
        points = extract_text_points(section.body, limit=4)
        for point in points:
            lines.append(f"- {point}")
    return "\n".join(lines)


def build_section_answer(section: Section) -> str:
    points = extract_text_points(section.body, limit=6)
    lines = [f"- **주제**: `{section.title}`의 핵심 내용을 정리합니다."]
    if points:
        for point in points:
            lines.append(f"- {point}")
    if extract_formula_blocks(section.body):
        lines.append("- **추가 요소**: 이 섹션에는 계산식 또는 변수 정의가 포함됩니다.")
    if extract_table_blocks(section.body):
        lines.append("- **추가 요소**: 이 섹션에는 표 형태의 구분 기준 또는 비교 정보가 포함됩니다.")
    return "\n".join(lines)


def build_formula_answer(section: Section) -> str:
    formulas = extract_formula_blocks(section.body)
    points = extract_text_points(section.body, limit=4)
    lines = [f"- **대상**: `{section.title}`에서 사용하는 계산식과 관련 설명입니다."]
    for idx, formula in enumerate(formulas[:3], start=1):
        one_line = strip_markdown(formula.replace("\n", " "))
        lines.append(f"- **식 {idx}**: `{one_line}`")
    for point in points:
        lines.append(f"- {point}")
    return "\n".join(lines)


def build_table_answer(section: Section, header: list[str], rows: list[list[str]]) -> str:
    lines = [f"- **대상 표**: `{section.title}` 아래 표의 기준과 구분을 설명합니다."]
    if header:
        lines.append("- **열 구성**: " + ", ".join(f"`{col}`" for col in header if col))
    for row in rows[:20]:
        cells = [cell for cell in row if cell]
        if not cells:
            continue
        if header and len(cells) == len(header):
            parts = [
                f"{header[i]}={cells[i]}"
                for i in range(len(cells))
                if i < len(header) and header[i]
            ]
            lines.append("- " + ", ".join(parts))
        else:
            lines.append("- " + " / ".join(cells))
    if len(rows) > 20:
        lines.append("- **추가 행**: 나머지 행도 동일한 기준 구조로 이어집니다.")
    return "\n".join(lines)


def build_records_for_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    sections = parse_sections(text)
    chapter_id = infer_chapter_id(path)
    chapter_title = infer_chapter_title(path, sections)
    doc_path = path.relative_to(repo_root()).as_posix()
    children_map: dict[int, list[Section]] = {section.index: [] for section in sections}
    for section in sections:
        if section.parent_index is not None:
            children_map[section.parent_index].append(section)

    records: list[dict] = []
    serial = 1

    for section in sections:
        section_path = ">".join(section.path_titles)
        tags_base = ["seed", "zeroin", f"chapter{chapter_id}"]
        children = children_map.get(section.index, [])

        if children:
            rec_id = f"zeroin.seed_v2_{chapter_id}_{serial:04d}"
            serial += 1
            records.append(
                make_record(
                    rec_id,
                    f"`{section.title}` 아래 내용을 전체적으로 설명해줘.",
                    build_coverage_answer(section, children),
                    chapter_id=chapter_id,
                    chapter_title=chapter_title,
                    section_path=section_path,
                    qa_type="coverage",
                    section_title=section.title,
                    difficulty="medium" if children else infer_difficulty(section),
                    answer_key_points=[section.title] + [child.title for child in children[:4]],
                    tags=tags_base + ["coverage"],
                    doc_path=doc_path,
                    doc_hash=doc_hash,
                    anchors=section.path_titles,
                )
            )

        rec_id = f"zeroin.seed_v2_{chapter_id}_{serial:04d}"
        serial += 1
        section_qa_type = infer_qa_type(section)
        records.append(
            make_record(
                rec_id,
                f"`{section.title}`을 설명해줘.",
                build_section_answer(section),
                chapter_id=chapter_id,
                chapter_title=chapter_title,
                section_path=section_path,
                qa_type=section_qa_type,
                section_title=section.title,
                difficulty=infer_difficulty(section),
                answer_key_points=extract_text_points(section.body, limit=3) or [section.title],
                tags=tags_base + [section_qa_type],
                doc_path=doc_path,
                doc_hash=doc_hash,
                anchors=section.path_titles,
            )
        )

        formulas = extract_formula_blocks(section.body)
        if formulas:
            rec_id = f"zeroin.seed_v2_{chapter_id}_{serial:04d}"
            serial += 1
            records.append(
                make_record(
                    rec_id,
                    f"`{section.title}`에서 쓰는 계산식과 변수 의미를 설명해줘.",
                    build_formula_answer(section),
                    chapter_id=chapter_id,
                    chapter_title=chapter_title,
                    section_path=section_path,
                    qa_type="formula",
                    section_title=section.title,
                    difficulty="hard",
                    answer_key_points=["계산식 포함", "변수 정의 포함", section.title],
                    tags=tags_base + ["formula"],
                    doc_path=doc_path,
                    doc_hash=doc_hash,
                    anchors=section.path_titles,
                )
            )

        table_blocks = extract_table_blocks(section.body)
        for table_idx, block in enumerate(table_blocks, start=1):
            header, rows = parse_table(block)
            if not header:
                continue
            rec_id = f"zeroin.seed_v2_{chapter_id}_{serial:04d}"
            serial += 1
            records.append(
                make_record(
                    rec_id,
                    f"`{section.title}`에 있는 표의 기준과 구분값을 설명해줘.",
                    build_table_answer(section, header, rows),
                    chapter_id=chapter_id,
                    chapter_title=chapter_title,
                    section_path=f"{section_path}>table{table_idx}",
                    qa_type="table_explainer",
                    section_title=section.title,
                    difficulty="hard",
                    answer_key_points=[section.title, "표 기준 설명", "행/열 의미"],
                    tags=tags_base + ["table_explainer"],
                    doc_path=doc_path,
                    doc_hash=doc_hash,
                    anchors=section.path_titles + [f"table{table_idx}"],
                )
            )

    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def main() -> None:
    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_paths = sorted(docs_dir().glob("*.md"))
    total_records = 0

    for path in doc_paths:
        records = build_records_for_file(path)
        output_path = out_dir / f"seed_v2_{path.stem}.jsonl"
        write_jsonl(output_path, records)
        total_records += len(records)
        print(f"{path.name} -> {output_path.name} ({len(records)} records)")

    print(f"generated {len(doc_paths)} files, {total_records} records")


if __name__ == "__main__":
    main()
