#!/usr/bin/env python3
"""04_build_jsonl.py — questions + answers를 최종 seed_v3_all.jsonl로 조립한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM_CONTENT = (
    "당신은 제로인 유형분류 및 펀드평가 방법론 전문가입니다. "
    "제로인 방법론에 근거해 정확하고 상세하게 한국어로 답변합니다. "
    "문서에 없는 내용은 추가하지 않되, 의미 연결을 위한 최소한의 설명만 허용합니다. "
    "답변은 자연스러운 문장으로 연결하되, 새로운 정보 추가 없이 기존 내용을 매끄럽게 표현합니다. "
    "새로운 판단 기준, 규칙, 질문에 없는 대상이나 비교는 추가하지 않습니다."
)

SOURCE_FILE = "Zeroin 펀드평가 방법론 - only_text.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final seed_v3_all.jsonl.")
    parser.add_argument("--state-dir", default=str(Path(__file__).resolve().parent / "state"))
    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).resolve().parents[2] / "llm_datasets/seed_v3/seed_v3_all.jsonl"
        ),
    )
    return parser.parse_args()


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


def build_record(idx: int, answer_rec: dict) -> dict | None:
    question = answer_rec.get("question", "").strip()
    answer = answer_rec.get("answer", "").strip()
    if not question or not answer:
        return None

    record_id = f"zeroin.seed_v3_{idx:04d}"
    return {
        "id": record_id,
        "messages": [
            {"role": "system", "content": SYSTEM_CONTENT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "meta": {
            "dataset_version": "v3",
            "section": answer_rec.get("section", ""),
            "qa_type": answer_rec.get("qa_type", ""),
            "source_file": SOURCE_FILE,
            "original_id": answer_rec.get("id", ""),
        },
    }


def main() -> None:
    args = parse_args()
    state_dir = Path(args.state_dir)
    answers_path = state_dir / "answers.jsonl"
    questions_path = state_dir / "questions.jsonl"

    if not answers_path.exists():
        raise FileNotFoundError("answers.jsonl not found. Run 03_gen_answers.py first.")

    answers = read_jsonl(answers_path)

    # questions.jsonl의 최신 질문으로 덮어씀 (질문 재생성 시 answers에 구버전이 남는 문제 방지)
    if questions_path.exists():
        q_map = {r["id"]: r["question"] for r in read_jsonl(questions_path)}
        for ans in answers:
            if ans["id"] in q_map:
                ans["question"] = q_map[ans["id"]]

    records = []
    skipped = 0

    for idx, ans in enumerate(answers, start=1):
        rec = build_record(idx, ans)
        if rec is None:
            skipped += 1
            print(f"  skip (empty): {ans.get('id', '?')}")
            continue
        records.append(rec)

    output_path = Path(args.output)
    write_jsonl(output_path, records)
    print(f"built {len(records)} records (skipped {skipped}) -> {output_path}")


if __name__ == "__main__":
    main()
