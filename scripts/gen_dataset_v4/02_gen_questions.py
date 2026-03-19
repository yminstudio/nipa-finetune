#!/usr/bin/env python3
"""v4 질문 후보 생성기.

구조 추출 결과를 바탕으로 문서 범위 안의 짧은 질문 후보를 생성한다.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STRUCTURE = Path(__file__).resolve().parent / "state/structure.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "state/questions_raw.jsonl"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
DEFAULT_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

SYSTEM_PROMPT = """당신은 제로인 방법론 기반 QA 데이터셋의 질문 생성기다.
반드시 한국어로만 답하고, 제공된 재료 범위 안에서만 질문을 만든다.
질문은 짧고 단일 쟁점 중심이어야 한다.
일반 금융상식, 투자조언, 문서 바깥 배경지식 질문은 금지한다."""

QUESTION_PROMPT = """아래 재료만 사용해서 자연스러운 질문 후보를 생성해줘.

[단원 제목]
{chapter}

[절 제목]
{section}

[소절 제목]
{subsection}

[핵심 제목]
{seed_title}

[핵심 명사]
{seed_nouns}

규칙:
- 질문은 짧고 단일 쟁점 중심
- 문서 범위를 벗어난 질문 금지
- 제목을 그대로 복붙한 문장 금지
- 일반 금융상식, 투자조언, 시장전망 질문 금지
- 질문 유형은 definition, criteria, comparison, application 중 하나만 사용
- 최대 {question_count}개까지만 생성

반환 형식:
순수 JSON 배열만 출력한다.
[
  {{
    "question": "질문 내용",
    "qa_type": "definition|criteria|comparison|application",
    "question_template": "템플릿 이름"
  }}
]
"""


def load_dotenv_file() -> None:
    for dotenv_path in (ROOT / ".env", ROOT.parent / ".env"):
        if not dotenv_path.exists():
            continue
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"').replace("\r", "")
            if key and key not in os.environ:
                os.environ[key] = value


load_dotenv_file()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate raw v4 questions from structure records.")
    parser.add_argument("--input", default=str(DEFAULT_STRUCTURE), help="Path to structure.jsonl")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to questions_raw.jsonl")
    parser.add_argument("--api-base", default=os.getenv("OPENAI_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--questions-per-record", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N structure records.")
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


def require_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return api_key.strip().replace("\r", "")


def call_chat(
    *,
    api_base: str,
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int,
) -> str:
    api_base = api_base.strip().replace("\r", "")
    model = model.strip().replace("\r", "")
    api_key = api_key.strip().replace("\r", "")
    payload = {
        "model": model,
        "max_completion_tokens": 4000,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    request = urllib.request.Request(
        url=f"{api_base}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.loads(response.read().decode("utf-8"))
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


def fix_json_escapes(text: str) -> str:
    return re.sub(r'\\([^"\\/bfnrtu])', r"\1", text)


def parse_json_array(raw: str) -> list[dict]:
    text = raw.strip()
    if not text:
        raise ValueError("empty response")
    if text.startswith("```"):
        lines = text.splitlines()
        end_index = next((i for i, line in enumerate(lines[1:], 1) if line.strip() == "```"), len(lines))
        text = "\n".join(lines[1:end_index])
    text = text.strip()
    if text.lower().startswith("json"):
        text = text[4:].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        repaired = fix_json_escapes(text)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            start = repaired.find("[")
            end = repaired.rfind("]")
            if start == -1 or end == -1 or start >= end:
                raise ValueError(f"response is not a JSON array: {raw[:200]!r}")
            data = json.loads(repaired[start : end + 1])
    if not isinstance(data, list):
        raise ValueError("question generator must return a JSON array")
    return [item for item in data if isinstance(item, dict)]


def build_prompt(record: dict, question_count: int) -> str:
    nouns = ", ".join(record.get("seed_nouns", []))
    return QUESTION_PROMPT.format(
        chapter=record.get("chapter", ""),
        section=record.get("section", ""),
        subsection=record.get("subsection", ""),
        seed_title=record.get("seed_title", ""),
        seed_nouns=nouns,
        question_count=question_count,
    )


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    if output_path.exists() and not args.force:
        raise SystemExit(f"output already exists: {output_path} (use --force to overwrite)")

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    structure_records = read_jsonl(input_path)
    if args.limit > 0:
        structure_records = structure_records[: args.limit]

    api_key = require_api_key()
    results: list[dict] = []
    question_index = 1

    for record in structure_records:
        prompt = build_prompt(record, question_count=args.questions_per_record)
        last_error: Exception | None = None
        items: list[dict] = []
        for _attempt in range(args.max_retries):
            try:
                raw = call_chat(
                    api_base=args.api_base,
                    model=args.model,
                    api_key=api_key,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt,
                    max_retries=args.max_retries,
                )
                items = parse_json_array(raw)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(1)
        if last_error and not items:
            print(f"skip question generation for {record.get('id', '?')}: {last_error}")
            continue
        for item in items:
            question = str(item.get("question", "")).strip()
            if not question:
                continue
            results.append(
                {
                    "id": f"v4_qraw_{question_index:04d}",
                    "question": question,
                    "qa_type": str(item.get("qa_type", "")).strip(),
                    "question_template": str(item.get("question_template", "")).strip(),
                    "chapter": record.get("chapter", ""),
                    "section": record.get("section", ""),
                    "subsection": record.get("subsection", ""),
                    "seed_title": record.get("seed_title", ""),
                    "seed_nouns": record.get("seed_nouns", []),
                    "source_file": record.get("source_file", ""),
                    "generation_mode": "title_heading_noun_to_grounded_qa",
                    "review_status": "pending",
                }
            )
            question_index += 1

    write_jsonl(output_path, results)
    print(f"Generated {len(results)} raw questions -> {output_path}")


if __name__ == "__main__":
    main()
