#!/usr/bin/env python3
"""02_gen_questions.py — 문서를 챕터별로 AI에게 전달해 자연스러운 질문 목록을 생성한다."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path


DEFAULT_DOC = str(
    Path(__file__).resolve().parents[2]
    / "docs/제로인방법론/Zeroin 펀드평가 방법론 - only_text.md"
)

QUESTION_GEN_PROMPT = """\
아래 [섹션 내용]은 '제로인 유형분류 및 펀드평가 방법론' 문서의 일부이다.

이 내용을 이해하고, 실제 업무 담당자(펀드 평가 실무자)가 물어볼 법한 자연스러운 한국어 질문을 생성해줘.

규칙:
- 섹션의 주요 개념, 기준, 규칙, 절차를 빠짐없이 커버할 것
- 하위 소제목이 있으면 각 소제목에 대해서도 질문을 만들 것
- 질문은 실용적이고 구체적으로 (예: "주식형 펀드로 분류되려면 어떤 조건을 충족해야 하나요?")
- '문서', '본문', '위 내용' 같은 참조 표현 사용 금지
- '[참고: ...]', 'Step N:' 같은 기술적 표기가 제목인 경우 질문으로 만들지 말 것
- 각 질문마다 해당 소제목(section)도 함께 기록할 것
- 최소 {min_count}개 이상 생성할 것

반환 형식 — 순수 JSON 배열만 출력 (코드블록, 설명 없이):
[
  {{"question": "질문 내용", "section": "해당 소제목"}},
  ...
]

[섹션 내용]
\"\"\"
{section_content}
\"\"\"
"""


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
    parser = argparse.ArgumentParser(description="Generate questions via AI API per chapter.")
    parser.add_argument("--doc", default=DEFAULT_DOC)
    parser.add_argument("--state-dir", default=str(Path(__file__).resolve().parent / "state"))
    parser.add_argument("--api-base", default=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-nano"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def require_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return key


def call_chat(api_base: str, model: str, api_key: str, user_text: str, max_retries: int = 3) -> str:
    payload = {
        "model": model,
        "max_completion_tokens": 8000,
        "messages": [{"role": "user", "content": user_text}],
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=f"{api_base}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                content = body["choices"][0]["message"].get("content", "")
                if isinstance(content, list):
                    content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
                return content
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"chat completion failed: {last_error!r}")


def fix_json_escapes(text: str) -> str:
    return re.sub(r'\\([^"\\/bfnrtu])', r'\1', text)


def parse_json_array(raw: str) -> list[dict]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        end = next((i for i, l in enumerate(lines[1:], 1) if l.strip() == "```"), len(lines))
        text = "\n".join(lines[1:end])
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(fix_json_escapes(text))


def split_into_chapters(doc_text: str) -> list[tuple[str, str]]:
    """## 레벨 챕터 단위로 문서를 분할. (제목, 내용) 리스트 반환."""
    lines = doc_text.splitlines()
    chapters: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = re.match(r"^##\s+(.+)$", line)
        if m:
            title = m.group(1).strip()
            if re.search(r"목차|개정사항", title):
                continue
            chapters.append((i, title))

    result = []
    for idx, (line_no, title) in enumerate(chapters):
        end = chapters[idx + 1][0] if idx + 1 < len(chapters) else len(lines)
        content = "\n".join(lines[line_no:end]).strip()
        result.append((title, content))
    return result


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    api_key = require_api_key()

    state_dir = Path(args.state_dir)
    questions_path = state_dir / "questions.jsonl"

    if questions_path.exists() and not args.force:
        count = sum(1 for line in questions_path.read_text(encoding="utf-8").splitlines() if line.strip())
        print(f"questions.jsonl already exists ({count} questions). Use --force to regenerate.")
        return

    doc_path = Path(args.doc).resolve()
    doc_text = doc_path.read_text(encoding="utf-8")
    chapters = split_into_chapters(doc_text)
    print(f"found {len(chapters)} chapters in {doc_path.name}")

    all_questions: list[dict] = []
    global_idx = 1

    for ch_title, ch_content in chapters:
        # 짧은 챕터는 최소 3개, 긴 챕터는 최소 8개 요청
        min_count = 8 if len(ch_content) > 3000 else 3
        prompt = QUESTION_GEN_PROMPT.format(section_content=ch_content, min_count=min_count)

        print(f"  [{ch_title[:40]}] 질문 생성 중 (min {min_count}개)...")
        for attempt in range(3):
            try:
                raw = call_chat(args.api_base, args.model, api_key, prompt)
                items = parse_json_array(raw)
                break
            except Exception as e:
                print(f"    [retry {attempt+1}] {e}")
                time.sleep(2)
                items = []

        for item in items:
            question = item.get("question", "").strip()
            section = item.get("section", ch_title).strip()
            if not question:
                continue
            all_questions.append({
                "id": f"v3_q_{global_idx:04d}",
                "question": question,
                "section": section,
                "chapter": ch_title,
                "qa_type": "section_qa",
            })
            global_idx += 1

        print(f"    → {len(items)}개 생성 (누적 {len(all_questions)}개)")

    write_jsonl(questions_path, all_questions)
    print(f"\n총 {len(all_questions)}개 질문 -> {questions_path}")


if __name__ == "__main__":
    main()
