#!/usr/bin/env python3
"""03_gen_answers.py — 각 섹션의 원문 전체(서브섹션 포함)를 컨텍스트로 주고 상세 답변을 생성한다."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path


SYSTEM_PROMPT = (
    "당신은 제로인 방법론에 근거해 답변하는 도메인 어시스턴트입니다. "
    "한국어로 답변합니다. "
    "제공된 원문 섹션의 구조와 규칙을 충실히 반영해 상세하게 설명합니다. "
    "답변 첫머리에 '질문', '답변', '원문', '문서' 같은 메타 표현을 쓰지 않고 바로 내용으로 시작합니다."
)

ANSWER_PROMPT_TEMPLATE = """\
아래 [원문 섹션]을 기반으로 질문에 대해 상세하게 답변해줘.

규칙:
- 원문의 구조(서브섹션 제목, 항목, 기준값, 표 등)를 충분히 반영할 것
- 서브섹션이 있으면 각 서브섹션을 소제목으로 나눠 설명할 것
- 원문에 있는 수치, 조건, 기준을 빠짐없이 포함할 것
- 답변은 길고 상세하게 작성할 것 (요약하지 말 것)
- 메타 표현('원문', '문서', '본문', '위 내용') 없이 바로 내용으로 시작할 것

질문: {question}

[원문 섹션]
\"\"\"
{section_content}
\"\"\"
"""

BAD_PHRASES = (
    "위 문서", "아래 문서", "문서 본문", "원문에 따르면",
    "질문:", "답변:", "초안", "메타",
)

DEFAULT_DOC = str(
    Path(__file__).resolve().parents[2]
    / "docs/제로인방법론/Zeroin 펀드평가 방법론 - only_text.md"
)


def load_dotenv_file() -> None:
    dotenv_path = Path(__file__).resolve().parents[2] / ".env"
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv_file()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate detailed answers from section content.")
    parser.add_argument("--doc", default=DEFAULT_DOC)
    parser.add_argument("--state-dir", default=str(Path(__file__).resolve().parent / "state"))
    parser.add_argument("--api-base", default=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-nano"))
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--start-id", help="이 id부터 재시작.")
    return parser.parse_args()


def require_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return key


def read_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_doc_sections(text: str) -> dict[str, str]:
    """헤딩 → 해당 섹션 전체 내용(서브섹션 포함) 매핑 반환."""
    lines = text.splitlines()
    headings: list[tuple[int, int, str]] = []
    for i, line in enumerate(lines):
        m = re.match(r"^(#{2,4})\s+(.+)$", line)
        if m:
            headings.append((i, len(m.group(1)), m.group(2).strip()))

    section_map: dict[str, str] = {}
    for idx, (line_no, level, title) in enumerate(headings):
        # 같은 레벨 또는 상위 레벨 헤딩이 나올 때까지가 이 섹션의 범위
        end = len(lines)
        for next_line_no, next_level, _ in headings[idx + 1:]:
            if next_level <= level:
                end = next_line_no
                break
        content = "\n".join(lines[line_no:end]).strip()
        section_map[title] = content
    return section_map


def find_section_content(section_map: dict[str, str], section_title: str) -> str:
    """섹션 제목으로 내용을 검색 (부분 매칭 포함)."""
    # 정확 매칭
    if section_title in section_map:
        return section_map[section_title]
    # 부분 매칭
    for title, content in section_map.items():
        if section_title in title or title in section_title:
            return content
    return ""


def has_bad_phrase(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip()
    return any(phrase in normalized for phrase in BAD_PHRASES)


def call_chat(
    api_base: str,
    model: str,
    api_key: str,
    system: str,
    user_text: str,
    max_retries: int,
) -> str:
    payload = {
        "model": model,
        "max_completion_tokens": 8000,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=f"{api_base}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                content = body["choices"][0]["message"].get("content", "")
                if isinstance(content, list):
                    content = "".join(
                        item.get("text", "") for item in content if isinstance(item, dict)
                    )
                return content
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"chat completion failed: {last_error!r}")


def generate_answer(
    api_base: str,
    model: str,
    api_key: str,
    q: dict,
    section_content: str,
    max_retries: int,
) -> str:
    user_text = ANSWER_PROMPT_TEMPLATE.format(
        question=q["question"],
        section_content=section_content,
    )
    for attempt in range(max_retries):
        answer = call_chat(api_base, model, api_key, SYSTEM_PROMPT, user_text, max_retries)
        answer = answer.strip()
        if not answer:
            time.sleep(1)
            continue
        if has_bad_phrase(answer):
            print(f"  [retry] bad phrase in {q['id']}, attempt {attempt + 1}")
            time.sleep(1)
            continue
        return answer
    raise RuntimeError(f"failed to generate answer for {q['id']}")


def main() -> None:
    args = parse_args()
    api_key = require_api_key()

    state_dir = Path(args.state_dir)
    questions = read_jsonl(state_dir / "questions.jsonl")
    answers_path = state_dir / "answers.jsonl"

    doc_path = Path(args.doc).resolve()
    doc_text = doc_path.read_text(encoding="utf-8")
    section_map = parse_doc_sections(doc_text)
    print(f"loaded {len(section_map)} sections from {doc_path.name}")

    # 기존 answers 로드 (재시작 지원)
    existing: dict[str, dict] = {}
    if answers_path.exists():
        for rec in read_jsonl(answers_path):
            existing[rec["id"]] = rec

    started = args.start_id is None
    results: list[dict] = []

    for q in questions:
        qid = q["id"]

        if not started:
            if qid == args.start_id:
                started = True
            else:
                results.append(existing.get(qid, {**q, "answer": ""}))
                continue

        if qid in existing and existing[qid].get("answer"):
            results.append(existing[qid])
            print(f"skip (done): {qid}")
            continue

        # chapter(## 레벨) 우선, 없으면 section으로 fallback
        section_content = find_section_content(section_map, q.get("chapter", q["section"]))
        if not section_content:
            section_content = find_section_content(section_map, q["section"])
        if not section_content:
            print(f"  [warn] section not found: {q.get('chapter', q['section'])} ({qid})")
            results.append({**q, "answer": ""})
            continue

        print(f"[{len(results)+1}/{len(questions)}] {qid} | {q['question'][:50]}...")
        answer = generate_answer(
            args.api_base, args.model, api_key, q, section_content, args.max_retries
        )
        rec = {**q, "answer": answer}
        results.append(rec)
        existing[qid] = rec
        write_jsonl(answers_path, results)

    write_jsonl(answers_path, results)
    done = sum(1 for r in results if r.get("answer"))
    print(f"\ndone: {done}/{len(results)} answers -> {answers_path}")


if __name__ == "__main__":
    main()
