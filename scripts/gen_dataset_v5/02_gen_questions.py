#!/usr/bin/env python3
"""v5 질문 후보 생성기.

구조 추출 결과를 바탕으로 문서 범위 안의 짧은 질문 후보를 생성한다.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STRUCTURE = Path(__file__).resolve().parent / "state/structure_ch01.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "state/questions_raw_ch01_round1.jsonl"
DEFAULT_PROMPT_LOG_DIR = Path(__file__).resolve().parent / "prompt_log"
DEFAULT_MODEL = "gpt-4.1-nano"
DEFAULT_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
DEFAULT_TARGET_TOTAL = 100
QA_TYPE_WEIGHTS = {
    "criteria": 55,
    "definition": 15,
    "comparison": 15,
    "application": 15,
}

SYSTEM_PROMPT = """당신은 제로인 방법론 기반 QA 데이터셋의 질문 생성기다.
반드시 한국어로만 답하고, 제공된 재료 범위 안에서만 질문을 만든다.
질문은 짧고 단일 쟁점 중심이어야 한다.
일반 금융상식, 투자조언, 문서 바깥 배경지식 질문은 금지한다.
업로드 문서를 보지 않고도 일반 금융 상식만으로 답할 수 있는 질문은 생성하지 않는다.
질문은 반드시 문서 안의 특정 정의, 기준, 비교, 적용 중 하나에 직접 매핑되어야 한다.
정답은 문서의 핵심 정보와 의미 연결을 위한 최소 설명만으로 성립해야 하며, 문서 밖 지식 확장을 요구하는 질문은 만들지 않는다.
정답이 자연스러운 문장으로 연결될 수는 있어도, 그 연결을 위해 새로운 정보가 필요해지는 질문은 만들지 않는다."""

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

[질문 유형 목표]
{type_mix}

[이번 호출에서 허용되는 질문 유형]
{allowed_types}

[이번 호출의 생성 목표]
{call_goal}

규칙:
- 질문은 짧고 단일 쟁점 중심
- 문서 범위를 벗어난 질문 금지
- 제목을 그대로 복붙한 문장 금지
- 일반 금융상식, 투자조언, 시장전망 질문 금지
- 업로드 문서를 보지 않고도 일반 금융 상식만으로 답할 수 있는 질문은 생성하지 않는다
- 답이 표 전체 요약이나 장문 설명이 되도록 만드는 질문은 금지하고, 특정 기준 하나만 묻는 질문으로 만든다
- 질문 하나에는 하나의 판단축만 남긴다. 정의와 기준, 기준과 예외를 한 문장에 함께 묻지 않는다
- 정답은 문서의 핵심 정보와 의미 연결을 위한 최소 설명만으로 완결될 수 있어야 한다
- 정답이 자연스러운 문장으로 연결되더라도 새로운 정보 추가 없이 답할 수 있어야 한다
- 질문이 새로운 판단 기준, 질문에 없는 대상, 불필요한 비교를 답변에 끌어오게 만들면 폐기한다
- definition 질문은 정의문 하나로 직접 답할 수 있어야 하며 절차·비교·예외를 동시에 요구하지 않는다
- criteria 질문은 판단 기준만 묻게 하고 적용 절차나 다른 대상 비교를 함께 요구하지 않는다
- comparison 질문은 필요한 비교축만 묻게 하고 각 대상의 정의나 절차 설명까지 요구하지 않는다
- application 질문은 적용 순서나 전환 조건을 묻되 정의/비교를 한 문장에 함께 묻지 않는다
- 질문 유형은 definition, criteria, comparison, application 중 하나만 사용
- 허용되지 않은 질문 유형은 절대 생성하지 않는다
- 최대 {question_count}개까지만 생성

생성 후 자체 점검:
- 이 질문은 문서의 특정 표, 기준, 정의, 절차가 없으면 답하기 어려운가?
- 이 질문에 일반 금융 상식으로 그럴듯하게 답할 수 있으면 폐기한다
- 질문이 길거나 복합적이면 더 짧은 한 쟁점 질문으로 다시 쓴다
- 이 질문에 답하려고 할 때 질문에 없는 대상/비교/예외를 덧붙이게 되면 폐기한다

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


def resolve_question_model() -> str:
    for key in ("Q_OPENAI_MODEL", "OPENAI_MODEL"):
        value = os.getenv(key, "").strip()
        if value:
            return value
    return DEFAULT_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate raw v5 chapter 1 questions from structure records.")
    parser.add_argument("--input", default=str(DEFAULT_STRUCTURE), help="Path to structure_ch01.jsonl")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to questions_raw_ch01_round1.jsonl")
    parser.add_argument(
        "--prompt-log-dir",
        default=str(DEFAULT_PROMPT_LOG_DIR),
        help="Directory to store prompt logs for AI API requests.",
    )
    parser.add_argument("--api-base", default=os.getenv("OPENAI_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--model", default=resolve_question_model())
    parser.add_argument("--target-total", type=int, default=DEFAULT_TARGET_TOTAL)
    parser.add_argument("--request-buffer", type=int, default=2, help="Extra question count per record.")
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


def parse_response_text(body: dict) -> str:
    if isinstance(body.get("output_text"), str) and body["output_text"].strip():
        return body["output_text"].strip()

    parts: list[str] = []
    for item in body.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(part for part in parts if part).strip()


def call_responses(
    *,
    api_base: str,
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int,
) -> str:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("openai package is required to generate questions.") from exc

    client = OpenAI(
        api_key=api_key.strip().replace("\r", ""),
        base_url=api_base.strip().replace("\r", ""),
        timeout=120.0,
    )
    payload = {
        "model": model.strip().replace("\r", ""),
        "input": [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            },
        ],
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "reasoning": {"effort": "medium", "summary": "auto"},
    }

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.responses.create(**payload)
            output_text = getattr(response, "output_text", None)
            if isinstance(output_text, str) and output_text.strip():
                return output_text.strip()
            if hasattr(response, "model_dump"):
                text = parse_response_text(response.model_dump())
                if text:
                    return text
            last_error = RuntimeError("empty response")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"responses call failed: {last_error!r}")


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


def build_type_targets(target_total: int) -> dict[str, int]:
    if target_total <= 0:
        return {key: 0 for key in QA_TYPE_WEIGHTS}

    total_weight = sum(QA_TYPE_WEIGHTS.values())
    base_counts: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    assigned = 0
    for qa_type, weight in QA_TYPE_WEIGHTS.items():
        raw_count = target_total * weight / total_weight
        count = int(raw_count)
        base_counts[qa_type] = count
        assigned += count
        remainders.append((raw_count - count, qa_type))

    for _, qa_type in sorted(remainders, reverse=True)[: target_total - assigned]:
        base_counts[qa_type] += 1
    return base_counts


def build_record_allocations(record_count: int, target_total: int) -> list[int]:
    if record_count <= 0:
        return []
    base = target_total // record_count
    remainder = target_total % record_count
    return [base + (1 if index < remainder else 0) for index in range(record_count)]


def build_type_mix_text(type_targets: dict[str, int]) -> str:
    ordered = ["criteria", "definition", "comparison", "application"]
    parts = [f"{qa_type} {type_targets.get(qa_type, 0)}개" for qa_type in ordered]
    return ", ".join(parts)


def build_call_goal(type_targets: dict[str, int], allowed_types: list[str]) -> str:
    if len(allowed_types) == 1:
        qa_type = allowed_types[0]
        return f"{qa_type} 질문만 생성한다. 다른 유형은 금지한다."
    ordered_parts = [f"{qa_type} {type_targets.get(qa_type, 0)}개" for qa_type in allowed_types]
    return ", ".join(ordered_parts) + "를 우선 맞춘다."


def build_prompt(
    record: dict,
    question_count: int,
    *,
    type_targets: dict[str, int],
    allowed_types: list[str],
) -> str:
    nouns = ", ".join(record.get("seed_nouns", []))
    return QUESTION_PROMPT.format(
        chapter=record.get("chapter", ""),
        section=record.get("section", ""),
        subsection=record.get("subsection", ""),
        seed_title=record.get("seed_title", ""),
        seed_nouns=nouns,
        type_mix=build_type_mix_text(type_targets),
        allowed_types=", ".join(allowed_types),
        call_goal=build_call_goal(type_targets, allowed_types),
        question_count=question_count,
    )


def sanitize_filename_component(text: str, limit: int = 80) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    sanitized = re.sub(r"[\\/\r\n\t]+", "_", collapsed)
    sanitized = sanitized.rstrip(". ")
    if not sanitized:
        sanitized = "질문"
    return sanitized[:limit].rstrip(" ._")


def write_prompt_log(
    *,
    log_dir: Path,
    stage: str,
    record_id: str,
    question_text: str,
    model: str,
    api_base: str,
    system_prompt: str,
    user_prompt: str,
    extra: dict,
) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_question = sanitize_filename_component(question_text)
    log_path = log_dir / f"{safe_question}.md"
    body = "\n".join(
        [
            "# 질문 프롬프트 로그",
            "",
            "## Stage",
            stage,
            "",
            "## Record ID",
            record_id,
            "",
            "## Question",
            question_text,
            "",
            "## Model",
            model,
            "",
            "## API Base",
            api_base,
            "",
            "## Extra",
            "```json",
            json.dumps(extra, ensure_ascii=False, indent=2),
            "```",
            "",
            "## System Prompt",
            "```text",
            system_prompt,
            "```",
            "",
            "## User Prompt",
            "```text",
            user_prompt,
            "```",
            "",
        ]
    )
    log_path.write_text(body, encoding="utf-8")
    return log_path


def normalize_question_key(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip().rstrip("?!.")


def filter_type_targets(type_targets: dict[str, int], allowed_types: list[str]) -> dict[str, int]:
    return {qa_type: type_targets.get(qa_type, 0) for qa_type in allowed_types if type_targets.get(qa_type, 0) > 0}


def pick_candidate(
    candidates: list[dict],
    *,
    used_ids: set[str],
    seen_questions: set[str],
    remaining_type_targets: dict[str, int] | None = None,
) -> dict | None:
    for candidate in candidates:
        candidate_id = str(candidate.get("id", ""))
        if candidate_id in used_ids:
            continue
        question_key = normalize_question_key(str(candidate.get("question", "")))
        if not question_key or question_key in seen_questions:
            continue
        qa_type = str(candidate.get("qa_type", "")).strip()
        if remaining_type_targets is not None and remaining_type_targets.get(qa_type, 0) <= 0:
            continue
        return candidate
    return None


def select_questions(
    *,
    candidates: list[dict],
    record_order: list[str],
    record_targets: dict[str, int],
    type_targets: dict[str, int],
    target_total: int,
) -> list[dict]:
    grouped: dict[str, list[dict]] = {record_id: [] for record_id in record_order}
    for candidate in candidates:
        record_id = str(candidate.get("source_record_id", ""))
        grouped.setdefault(record_id, []).append(candidate)

    selected: list[dict] = []
    used_ids: set[str] = set()
    seen_questions: set[str] = set()
    remaining_per_record = dict(record_targets)
    remaining_types = dict(type_targets)

    progress = True
    while progress and len(selected) < target_total:
        progress = False
        for record_id in record_order:
            if remaining_per_record.get(record_id, 0) <= 0:
                continue
            candidate = pick_candidate(
                grouped.get(record_id, []),
                used_ids=used_ids,
                seen_questions=seen_questions,
                remaining_type_targets=remaining_types,
            )
            if candidate is None:
                continue
            selected.append(candidate)
            used_ids.add(str(candidate.get("id", "")))
            seen_questions.add(normalize_question_key(str(candidate.get("question", ""))))
            remaining_per_record[record_id] -= 1
            remaining_types[str(candidate.get("qa_type", "")).strip()] -= 1
            progress = True

    for record_id in record_order:
        while remaining_per_record.get(record_id, 0) > 0 and len(selected) < target_total:
            candidate = pick_candidate(
                grouped.get(record_id, []),
                used_ids=used_ids,
                seen_questions=seen_questions,
            )
            if candidate is None:
                break
            selected.append(candidate)
            used_ids.add(str(candidate.get("id", "")))
            seen_questions.add(normalize_question_key(str(candidate.get("question", ""))))
            remaining_per_record[record_id] -= 1
            qa_type = str(candidate.get("qa_type", "")).strip()
            if remaining_types.get(qa_type, 0) > 0:
                remaining_types[qa_type] -= 1

    while len(selected) < target_total:
        candidate = pick_candidate(
            candidates,
            used_ids=used_ids,
            seen_questions=seen_questions,
            remaining_type_targets=remaining_types,
        ) or pick_candidate(
            candidates,
            used_ids=used_ids,
            seen_questions=seen_questions,
        )
        if candidate is None:
            break
        selected.append(candidate)
        used_ids.add(str(candidate.get("id", "")))
        seen_questions.add(normalize_question_key(str(candidate.get("question", ""))))
        qa_type = str(candidate.get("qa_type", "")).strip()
        if remaining_types.get(qa_type, 0) > 0:
            remaining_types[qa_type] -= 1

    return selected[:target_total]


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    if output_path.exists() and not args.force:
        raise SystemExit(f"output already exists: {output_path} (use --force to overwrite)")

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")
    prompt_log_dir = Path(args.prompt_log_dir).resolve()

    structure_records = read_jsonl(input_path)
    if args.limit > 0:
        structure_records = structure_records[: args.limit]
    if not structure_records:
        raise RuntimeError("no structure records found")

    api_key = require_api_key()
    raw_candidates: list[dict] = []
    candidate_index = 1
    record_allocations = build_record_allocations(len(structure_records), args.target_total)
    type_targets = build_type_targets(args.target_total)
    record_order = [str(record.get("id", f"record_{index:04d}")) for index, record in enumerate(structure_records, start=1)]
    record_target_map = dict(zip(record_order, record_allocations))

    for record, allocation in zip(structure_records, record_allocations):
        question_count = max(1, allocation + args.request_buffer)
        local_type_targets = build_type_targets(question_count)
        source_record_id = str(record.get("id", f"record_{candidate_index:04d}"))
        call_plans = [
            {
                "allowed_types": ["criteria"],
                "type_targets": filter_type_targets(local_type_targets, ["criteria"]),
            },
            {
                "allowed_types": ["definition", "comparison", "application"],
                "type_targets": filter_type_targets(local_type_targets, ["definition", "comparison", "application"]),
            },
        ]

        for call_plan in call_plans:
            call_type_targets = call_plan["type_targets"]
            if not call_type_targets:
                continue
            prompt = build_prompt(
                record,
                question_count=sum(call_type_targets.values()),
                type_targets=call_type_targets,
                allowed_types=call_plan["allowed_types"],
            )
            last_error: Exception | None = None
            items: list[dict] = []
            for _attempt in range(args.max_retries):
                try:
                    raw = call_responses(
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
                print(
                    "skip question generation for"
                    f" {record.get('id', '?')} {call_plan['allowed_types']}: {last_error}"
                )
                continue
            for item in items:
                question = str(item.get("question", "")).strip()
                qa_type = str(item.get("qa_type", "")).strip()
                if not question or qa_type not in call_plan["allowed_types"]:
                    continue
                log_path = write_prompt_log(
                    log_dir=prompt_log_dir,
                    stage="questions",
                    record_id=source_record_id,
                    question_text=question,
                    model=args.model,
                    api_base=args.api_base,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt,
                    extra={
                        "chapter": record.get("chapter", ""),
                        "section": record.get("section", ""),
                        "subsection": record.get("subsection", ""),
                        "seed_title": record.get("seed_title", ""),
                        "record_target": allocation,
                        "requested_question_count": question_count,
                        "call_requested_question_count": sum(call_type_targets.values()),
                        "call_allowed_types": call_plan["allowed_types"],
                        "call_type_targets": call_type_targets,
                        "target_total": args.target_total,
                        "generated_question": question,
                        "qa_type": qa_type,
                        "question_template": str(item.get("question_template", "")).strip(),
                    },
                )
                raw_candidates.append(
                    {
                        "id": f"v5_qraw_candidate_{candidate_index:04d}",
                        "question": question,
                        "qa_type": qa_type,
                        "question_template": str(item.get("question_template", "")).strip(),
                        "chapter": record.get("chapter", ""),
                        "section": record.get("section", ""),
                        "subsection": record.get("subsection", ""),
                        "seed_title": record.get("seed_title", ""),
                        "seed_nouns": record.get("seed_nouns", []),
                        "source_file": record.get("source_file", ""),
                        "generation_mode": "title_heading_noun_to_grounded_qa",
                        "review_status": "pending",
                        "source_record_id": source_record_id,
                        "prompt_log_file": log_path.relative_to(ROOT).as_posix(),
                    }
                )
                candidate_index += 1

    selected = select_questions(
        candidates=raw_candidates,
        record_order=record_order,
        record_targets=record_target_map,
        type_targets=type_targets,
        target_total=args.target_total,
    )
    results = []
    for index, item in enumerate(selected, start=1):
        results.append(
            {
                **item,
                "id": f"v5_qraw_{index:04d}",
            }
        )

    write_jsonl(output_path, results)
    print(f"Generated {len(results)} raw questions -> {output_path}")


if __name__ == "__main__":
    main()
