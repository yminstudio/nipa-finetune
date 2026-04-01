#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = "당신은 제로인 펀드평가 방법론에 근거해 답변하는 도메인 어시스턴트입니다."
ROUND_PATTERN = re.compile(r"round(\d+)")


def normalize_markdown_cell(text: str) -> str:
    normalized = text.strip()
    normalized = normalized.replace("<br>", "\n")
    normalized = normalized.replace("\\.", ".")
    return normalized.strip()


def parse_markdown_table_groups(source_path: Path) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for raw_line in source_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("| "):
            continue
        if line in {"|     |     |", "| --- | --- |", "| user | assistant |"}:
            continue

        cells = line[2:-2].split(" | ", 1)
        if len(cells) != 2:
            raise ValueError(f"unexpected table row format: {raw_line}")

        user_cell, assistant_cell = cells
        questions = [question.strip() for question in user_cell.split("<br>") if question.strip()]
        if not questions:
            raise ValueError(f"empty question cell: {raw_line}")

        answer = normalize_markdown_cell(assistant_cell)
        if not answer:
            raise ValueError(f"empty answer cell: {raw_line}")

        groups.append(
            {
                "group_id": f"group{len(groups) + 1}",
                "questions": questions,
                "answer": answer,
            }
        )

    if not groups:
        raise ValueError(f"no data rows found in {source_path}")
    return groups


def infer_next_round(root: Path) -> int:
    candidates = [
        root / "configs",
        root / "llm_datasets/seed_v7",
        root / "llm_model_lora",
        root / "tests",
    ]
    max_round = 0
    for base in candidates:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            match = ROUND_PATTERN.search(path.name)
            if match:
                max_round = max(max_round, int(match.group(1)))
    return max_round + 1


def build_round_paths(root: Path, round_number: int) -> dict[str, Path]:
    round_label = f"round{round_number}"
    return {
        "dataset": root / f"llm_datasets/seed_v7/seed_v7_single_qa_{round_label}.jsonl",
        "config": root / f"configs/gpt_oss_20b_seed_v7_single_qa_{round_label}.json",
        "adapter_dir": root / f"llm_model_lora/gpt-oss-20b-seed-v7-single-qa-{round_label}",
        "train_result": root / f"llm_model_lora/gpt-oss-20b-seed-v7-single-qa-{round_label}/train_result.json",
        "smoke_config": root / f"tests/smoke_v7_single_qa_{round_label}_config.json",
        "smoke_report": root / f"tests/log/v7_single_qa_{round_label}_all_questions_report.md",
    }


def load_latest_round_config(root: Path) -> dict[str, Any]:
    config_dir = root / "configs"
    latest_path: Path | None = None
    latest_round = -1
    for path in config_dir.glob("gpt_oss_20b_seed_v7_single_qa_round*.json"):
        match = ROUND_PATTERN.search(path.name)
        if not match:
            continue
        round_number = int(match.group(1))
        if round_number > latest_round:
            latest_round = round_number
            latest_path = path

    if latest_path is None:
        raise FileNotFoundError("no prior round config found")
    return json.loads(latest_path.read_text(encoding="utf-8"))


def build_train_config(root: Path, round_number: int, paths: dict[str, Path]) -> dict[str, Any]:
    cfg = load_latest_round_config(root)
    cfg["model_name"] = "unsloth/gpt-oss-20b-BF16"
    cfg["dataset_path"] = str(paths["dataset"])
    cfg["output_dir"] = str(paths["adapter_dir"])
    cfg["report_path"] = str(paths["train_result"])
    cfg["stop_when_loss_below"] = float(cfg.get("stop_when_loss_below", 1.0))
    cfg["stop_when_mean_token_accuracy_at_least"] = float(
        cfg.get("stop_when_mean_token_accuracy_at_least", 0.98)
    )
    cfg.pop("init_adapter_path", None)
    return cfg


def build_smoke_config(train_cfg: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "model_name": train_cfg["model_name"],
        "cache_dir": train_cfg["cache_dir"],
        "adapter_path": str(paths["adapter_dir"]),
        "prompt_source_path": str(paths["dataset"]),
        "report_path": str(paths["smoke_report"]),
        "dtype": "bfloat16",
        "max_new_tokens": 384,
    }


def build_dataset_records(groups: list[dict[str, Any]], round_number: int) -> list[dict[str, Any]]:
    round_label = f"round{round_number}"
    dataset_version = f"v7_single_qa_{round_label}"
    today = date.today().isoformat()
    records: list[dict[str, Any]] = []

    for group in groups:
        for question_index, question in enumerate(group["questions"], start=1):
            record_index = len(records) + 1
            records.append(
                {
                    "id": f"zeroin.seed_v7_single_qa_{round_label}_{record_index:04d}",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": group["answer"]},
                    ],
                    "meta": {
                        "dataset_version": dataset_version,
                        "group_id": group["group_id"],
                        "date_key": today,
                        "question_variant_index": question_index,
                        "source_strategy": "multi_group_multi_question",
                        "training_mode": f"base_model_{round_label}",
                        "round": round_label,
                    },
                }
            )

    return records


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the next v7 round dataset and configs from scripts/data_source.md.")
    parser.add_argument("--source", default=str(ROOT / "scripts/data_source.md"), help="Markdown source table path.")
    parser.add_argument("--round", type=int, default=None, help="Optional explicit round number.")
    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    round_number = args.round if args.round is not None else infer_next_round(ROOT)
    paths = build_round_paths(ROOT, round_number)
    groups = parse_markdown_table_groups(source_path)
    records = build_dataset_records(groups, round_number)
    train_cfg = build_train_config(ROOT, round_number, paths)
    smoke_cfg = build_smoke_config(train_cfg, paths)

    write_jsonl(paths["dataset"], records)
    write_json(paths["config"], train_cfg)
    write_json(paths["smoke_config"], smoke_cfg)

    summary = {
        "source_path": str(source_path),
        "round_number": round_number,
        "group_count": len(groups),
        "question_count": len(records),
        "dataset_path": str(paths["dataset"]),
        "config_path": str(paths["config"]),
        "smoke_config_path": str(paths["smoke_config"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
