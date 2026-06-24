# V8 Preparation

## Goal

`v8`는 `v7`의 문서형 데이터 확장 흐름은 유지하되, LoRA 어댑터 반복 실험에서 벗어나 `gpt-oss-20b` 전체 가중치를 업데이트하는 `full_ft` 운영 경로로 전환한다.

## V7 vs V8

- `v7`는 LoRA 학습 결과를 `llm_model_lora/` 아래에 저장하고, 필요 시 smoke 스크립트로 생성 결과를 확인한다.
- `v8`는 full fine-tuning 결과를 `llm_model_full/` 아래에 저장하고, 실제 학습 전에 `--dry-run`으로 readiness gate와 입력 경로를 먼저 점검한다.
- `v7` 검증은 adapter 기반 smoke 흐름에 가깝고, `v8` 검증은 저장된 `final-export/` 또는 `checkpoint-*`를 다시 로드하는 post-training validation 흐름이다.

## Key Paths

- source markdown: `scripts/data_source.md`
- generated dataset: `llm_datasets/seed_v8/seed_v8_round1_full_ft.jsonl`
- round config: `configs/gpt_oss_20b_seed_v8_round1_full_ft.json`
- DeepSpeed config: `configs/deepspeed/gpt_oss_20b_zero3_bf16.json`
- training output dir: `llm_model_full/gpt-oss-20b-seed-v8-round1-full-ft/`
- train report: `llm_model_full/gpt-oss-20b-seed-v8-round1-full-ft/train_result.json`
- final export: `llm_model_full/gpt-oss-20b-seed-v8-round1-full-ft/final-export/`
- validation report pattern: `tests/log/v8_round1_full_ft_*_report.md`

## Operator Checklist

실제 full fine-tuning 시작 전에는 아래 preflight 순서를 그대로 따른다.

1. dataset build를 실행한다.
2. config validation을 실행한다.
3. `--dry-run`을 실행한다.
4. readiness report를 확인한다.
5. report gate가 통과한 경우에만 실제 full fine-tuning을 시작한다.

## Operator Flow

### 1. Build Dataset

`v8 round1` 데이터셋은 `scripts/data_source.md`를 읽어 생성한다.

```bash
python scripts/build_v8_round1_full_ft_dataset.py
```

정상 실행 시 JSONL 학습 데이터는 `llm_datasets/seed_v8/seed_v8_round1_full_ft.jsonl`에 기록된다.

### 2. Config Validation

실제 실행 전에 round config가 `full_ft` 계약을 만족하는지 먼저 확인한다.

```bash
python -c "import json; from pathlib import Path; from scripts.v8_full_ft_contracts import validate_full_ft_config; cfg = json.loads(Path('configs/gpt_oss_20b_seed_v8_round1_full_ft.json').read_text(encoding='utf-8')); validate_full_ft_config(cfg); print('config validation: OK')"
```

이 단계는 `scripts/v8_full_ft_contracts.py`의 `validate_full_ft_config()`를 직접 호출해 필수 키, `resume_mode`, 정수 필드, `save_steps <= max_steps` 제약을 먼저 확인한다.

### 3. Dry Run

실제 학습 전에 config 파싱, dataset 존재 여부, DeepSpeed 설정 경로, readiness gate, report 쓰기 경로를 먼저 확인한다.

```bash
python scripts/run_full_ft_gpt_oss_20b.py --config configs/gpt_oss_20b_seed_v8_round1_full_ft.json --dry-run
```

dry-run 결과는 `llm_model_full/gpt-oss-20b-seed-v8-round1-full-ft/train_result.json`에 기록된다. `resume_mode` 기본값은 `fail`이므로 기존 `checkpoint-*` 또는 `final-export/`가 남아 있으면 실제 학습은 중단된다.

### 4. Inspect Readiness Report

dry-run 이후에는 readiness report를 열어 gate 결과를 확인한다.

```bash
python -m json.tool llm_model_full/gpt-oss-20b-seed-v8-round1-full-ft/train_result.json
```

`status`가 `dry_run_ready`이고 `readiness.status`가 `ready`일 때만 다음 단계로 진행한다. `blocking_check` 또는 `blocking_reason`가 있으면 실제 학습을 시작하지 않는다.

### 5. Real Training

readiness gate를 통과한 뒤에만 실제 full fine-tuning을 시작한다.

```bash
python scripts/run_full_ft_gpt_oss_20b.py --config configs/gpt_oss_20b_seed_v8_round1_full_ft.json
```

실행 중 산출물은 `llm_model_full/gpt-oss-20b-seed-v8-round1-full-ft/` 아래에 쌓이며, 대표 산출물은 다음과 같다.

- `train_result.json`
- `checkpoint-*`
- `final-export/`

### 6. Post-Training Validation

학습 완료 후에는 저장된 full model export를 다시 로드해 검증 리포트를 생성한다.

```bash
python scripts/check_gpt_oss_full_ft_output.py --config configs/gpt_oss_20b_seed_v8_round1_full_ft.json
```

기본 검증 대상은 round config가 가리키는 `final-export/`이며, 특정 체크포인트를 검증하려면 아래처럼 명시한다.

```bash
python scripts/check_gpt_oss_full_ft_output.py --config configs/gpt_oss_20b_seed_v8_round1_full_ft.json --checkpoint-path llm_model_full/gpt-oss-20b-seed-v8-round1-full-ft/checkpoint-100
```

검증 리포트는 `tests/log/v8_round1_full_ft_*_report.md` 패턴으로 저장된다.
