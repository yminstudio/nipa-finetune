#!/usr/bin/env python3
"""v5 전 챕터 데이터 생성, 학습, 검토 오케스트레이터."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = Path(__file__).resolve().parent / "state"
SEED_DIR = ROOT / "llm_datasets/seed_v5"
RENDERED_DIR = ROOT / "llm_datasets/rendered/gpt-oss"
DOC_PATH = ROOT / "docs/제로인방법론/Zeroin 펀드평가 방법론 - only_text.md"
DEFAULT_CONFIG = ROOT / "configs/gpt_oss_20b_seed_v5_all_round1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v5 pipeline for all chapters.")
    parser.add_argument("--doc", default=str(DOC_PATH), help="Source markdown path.")
    parser.add_argument(
        "--phase",
        choices=("data", "train", "full"),
        default="full",
        help="`data`: dataset only, `train`: train/validate only, `full`: dataset+train+validate",
    )
    parser.add_argument(
        "--chapter-prefixes",
        nargs="*",
        default=[],
        help="Optional explicit chapter prefixes such as `1.`, `2.`, `X1.`",
    )
    parser.add_argument("--round-label", default="round1", help="Round label used in file names.")
    parser.add_argument("--target-total", type=int, default=100, help="Questions to target per chapter.")
    parser.add_argument("--question-limit", type=int, default=0, help="Optional limit passed to question generation.")
    parser.add_argument("--answer-limit", type=int, default=0, help="Optional limit passed to answer generation.")
    parser.add_argument("--train-config", default=str(DEFAULT_CONFIG), help="Training config path.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training even in `full` phase.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def normalize_title(text: str) -> str:
    normalized = text.replace("\\.", ".").replace("\u00a0", " ")
    return re.sub(r"\s+", " ", normalized).strip()


def chapter_token(title: str) -> str:
    return normalize_title(title).split(" ", 1)[0]


def discover_chapter_prefixes(doc_path: Path) -> list[str]:
    prefixes: list[str] = []
    for line in doc_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^(#{1,2})\s+(.+)$", line)
        if not match:
            continue
        markdown_level = len(match.group(1))
        token = chapter_token(match.group(2))
        if not (
            re.fullmatch(r"\d+\.", token)
            or re.fullmatch(r"X\d*\.", token)
            or (markdown_level <= 2 and re.fullmatch(r"X\.\d+\.", token))
        ):
            continue
        if token not in prefixes:
            prefixes.append(token)
    return prefixes


def chapter_slug(prefix: str) -> str:
    normalized = prefix.strip().lower().replace(".", "").replace(" ", "")
    if normalized.startswith("x"):
        return f"ch{normalized}"
    if normalized.isdigit():
        return f"ch{int(normalized):02d}"
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_") or "chapter"


def run_command(command: list[str]) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=str(ROOT), check=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def state_path(*parts: str) -> Path:
    return STATE_DIR.joinpath(*parts)


def build_dataset(args: argparse.Namespace, chapter_prefixes: list[str]) -> None:
    validated_paths: list[Path] = []
    for prefix in chapter_prefixes:
        slug = chapter_slug(prefix)
        structure_path = state_path(f"structure_{slug}.jsonl")
        questions_raw_path = state_path(f"questions_raw_{slug}_{args.round_label}.jsonl")
        questions_filtered_path = state_path(f"questions_filtered_{slug}_{args.round_label}.jsonl")
        answers_raw_path = state_path(f"answers_raw_{slug}_{args.round_label}.jsonl")
        answers_validated_path = state_path(f"answers_validated_{slug}_{args.round_label}.jsonl")
        validation_report_path = state_path(f"validation_report_{slug}_{args.round_label}.json")
        prompt_log_dir = Path(__file__).resolve().parent / f"prompt_log_{slug}_{args.round_label}"

        command = [
            "python",
            str(Path(__file__).resolve().parent / "01_extract_structure.py"),
            "--input-files",
            str(Path(args.doc).resolve()),
            "--chapter-prefix",
            prefix,
            "--output",
            str(structure_path),
        ]
        if args.force:
            command.append("--force")
        run_command(command)

        command = [
            "python",
            str(Path(__file__).resolve().parent / "02_gen_questions.py"),
            "--input",
            str(structure_path),
            "--output",
            str(questions_raw_path),
            "--prompt-log-dir",
            str(prompt_log_dir),
            "--target-total",
            str(args.target_total),
        ]
        if args.question_limit > 0:
            command.extend(["--limit", str(args.question_limit)])
        if args.force:
            command.append("--force")
        run_command(command)

        command = [
            "python",
            str(Path(__file__).resolve().parent / "03_filter_questions.py"),
            "--input",
            str(questions_raw_path),
            "--output",
            str(questions_filtered_path),
        ]
        if args.force:
            command.append("--force")
        run_command(command)

        command = [
            "python",
            str(Path(__file__).resolve().parent / "04_gen_answers.py"),
            "--input",
            str(questions_filtered_path),
            "--output",
            str(answers_raw_path),
            "--prompt-log-dir",
            str(prompt_log_dir),
        ]
        if args.answer_limit > 0:
            command.extend(["--limit", str(args.answer_limit)])
        if args.force:
            command.append("--force")
        run_command(command)

        command = [
            "python",
            str(Path(__file__).resolve().parent / "05_validate_answers.py"),
            "--input",
            str(answers_raw_path),
            "--output",
            str(answers_validated_path),
            "--report",
            str(validation_report_path),
        ]
        if args.force:
            command.append("--force")
        run_command(command)
        validated_paths.append(answers_validated_path)

    merged_validated_path = state_path(f"answers_validated_{args.round_label}_all.jsonl")
    merged_records: list[dict] = []
    for path in validated_paths:
        merged_records.extend(read_jsonl(path))
    write_jsonl(merged_validated_path, merged_records)
    print(f"Merged {len(merged_records)} validated answers -> {merged_validated_path}")

    command = [
        "python",
        str(Path(__file__).resolve().parent / "06_build_seed_v5.py"),
        "--input",
        str(merged_validated_path),
        "--output-dir",
        str(SEED_DIR),
        "--chapter-output-prefix",
        f"seed_v5_{args.round_label}",
    ]
    if args.force:
        command.append("--force")
    run_command(command)

    all_seed_path = SEED_DIR / f"seed_v5_{args.round_label}_all.jsonl"
    all_rendered_path = RENDERED_DIR / f"seed_v5_{args.round_label}_all_harmony.jsonl"
    command = [
        "python",
        str(ROOT / "llm_datasets/render_gpt-oss-harmony.py"),
        "--input",
        str(all_seed_path),
        "--output",
        str(all_rendered_path),
    ]
    run_command(command)


def run_training(train_config_path: Path) -> None:
    command = [
        "python",
        str(ROOT / "scripts/run_smoke_gpt_oss_20b.py"),
        "--config",
        str(train_config_path),
    ]
    run_command(command)

    cfg = read_json(train_config_path)
    validation_questions = cfg.get("validation_questions")
    validation_question_count = (
        len([item for item in validation_questions if isinstance(item, str) and item.strip()])
        if isinstance(validation_questions, list)
        else 0
    )
    command = [
        "python",
        str(ROOT / "scripts/check_gpt_oss_model_output.py"),
        "--config",
        str(train_config_path),
        "--adapter-path",
        str(Path(cfg["output_dir"]).resolve()),
    ]
    if validation_question_count > 0:
        command.extend(["--question-limit", str(validation_question_count)])
    run_command(command)


def main() -> None:
    args = parse_args()
    doc_path = Path(args.doc).resolve()
    if not doc_path.exists():
        raise FileNotFoundError(f"document not found: {doc_path}")

    chapter_prefixes = args.chapter_prefixes or discover_chapter_prefixes(doc_path)
    if not chapter_prefixes:
        raise RuntimeError("no chapter prefixes discovered")

    print("Chapters:", ", ".join(chapter_prefixes))
    if args.phase in {"data", "full"}:
        build_dataset(args, chapter_prefixes)

    if args.phase in {"train", "full"} and not args.skip_train:
        run_training(Path(args.train_config).resolve())


if __name__ == "__main__":
    main()
