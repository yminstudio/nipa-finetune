---
name: zeroin-training-validation
description: Analyze the current training state in this project, verify dataset and config readiness, run GPT-OSS LoRA training, perform direct adapter validation, and continue to vLLM serving checks. Use when the user asks to continue training, run all-chapter training, validate an adapter, or perform end-to-end review of the training pipeline in this repository.
---

# Zeroin Training Validation

## Goal

이 저장소에서 현재 데이터셋 준비 상태를 분석하고, 학습 가능 여부를 판정한 뒤, 가능한 경우 `학습 -> 직접 검증 -> vLLM 검토`까지 이어서 수행합니다.

기본 원칙:

- 먼저 현재 상태를 확인하고, 이미 끝난 단계는 건너뜁니다.
- `데이터 준비 완료`를 보고한 뒤 멈추지 말고, 사용자가 중단을 요청하지 않았다면 다음 단계로 계속 진행합니다.
- 전 챕터 학습 요청이면 `all-chapter` 산출물을 우선 사용합니다.
- config 경로와 실제 산출물 경로가 다르면 학습 전에 먼저 맞춥니다.
- 검증은 최소 `직접 검증`까지 수행하고, 가능하면 `vLLM 검토`까지 이어갑니다.

## When To Use

다음 요청에서 이 스킬을 적용합니다.

- `학습 진행`
- `전 챕터 학습`
- `검증까지 진행`
- `학습이 왜 멈췄는지 확인`
- `adapter 검증`
- `vLLM 서빙 검토`
- `데이터 생성 후 이어서 학습/검증`

## Quick Start

1. 현재 데이터 산출물과 config를 점검합니다.
2. 학습 입력 JSONL과 Harmony 파일이 실제로 있는지 확인합니다.
3. 필요하면 데이터 파이프라인을 먼저 끝까지 수행합니다.
4. `scripts/run_smoke_gpt_oss_20b.py`로 LoRA 학습을 실행합니다.
5. `scripts/check_gpt_oss_model_output.py`로 직접 검증합니다.
6. 필요하면 `scripts/run_vllm_adapter_server.sh`와 `scripts/check_vllm_adapter_server.py`로 서빙 검토를 이어갑니다.

## Required Checks

학습 전에 아래를 반드시 확인합니다.

- canonical seed JSONL 존재 여부
- Harmony 렌더 결과 존재 여부
- config의 `dataset_path`, `prompt_source_path`, `output_dir`가 실제 파일명과 일치하는지
- 기존 어댑터를 이어학습할 경우 `init_adapter_path` 존재 여부
- 이미 같은 출력 디렉터리에 학습 결과가 있으면 재학습인지 이어서 확인인지 구분

전 챕터 기본 산출물은 보통 아래 형태를 따릅니다.

```text
llm_datasets/seed_*/..._all.jsonl
llm_datasets/rendered/gpt-oss/..._all_harmony.jsonl
configs/gpt_oss_20b_*.json
llm_model_lora/<adapter-output-dir>/
```

## Workflow

다음 체크리스트를 그대로 사용합니다.

```text
Training Progress
- [ ] 1. 현재 산출물 점검
- [ ] 2. 학습 입력 파일 존재 확인
- [ ] 3. config 경로 정합성 확인
- [ ] 4. 데이터 미완료 시 데이터 파이프라인 완료
- [ ] 5. LoRA 학습 실행
- [ ] 6. 직접 검증 실행
- [ ] 7. vLLM 어댑터 서빙
- [ ] 8. vLLM 헬스체크
- [ ] 9. 결과 요약 및 남은 리스크 보고
```

### 1. 현재 산출물 점검

- `scripts/gen_dataset_*/state/`
- `llm_datasets/seed_*/`
- `llm_datasets/rendered/gpt-oss/`
- `llm_model_lora/`

를 먼저 확인합니다.

- 이미 `*_all.jsonl`과 `*_all_harmony.jsonl`이 있으면 데이터 단계는 완료로 간주합니다.
- 데이터가 완료되었는데 학습 결과가 없으면 바로 학습 단계로 넘어갑니다.

### 2. 학습 입력 파일 존재 확인

전 챕터 요청이면 canonical JSONL과 Harmony 파일이 모두 있어야 합니다.

예시:

```bash
ls llm_datasets/seed_v5/seed_v5_allchap_round1_all.jsonl
ls llm_datasets/rendered/gpt-oss/seed_v5_allchap_round1_all_harmony.jsonl
```

둘 중 하나라도 없으면 학습을 시작하지 말고 데이터 파이프라인부터 끝냅니다.

### 3. config 경로 정합성 확인

학습 config를 열어 다음 값을 실제 산출물과 비교합니다.

- `dataset_path`
- `prompt_source_path`
- `output_dir`
- `validation_report_path`
- `served_model_name`

파일명 규칙이 현재 산출물과 다르면 config를 먼저 수정합니다.

### 4. 데이터 미완료 시 데이터 파이프라인 완료

프로젝트에 상위 오케스트레이터가 있으면 그 경로를 우선 사용합니다.

현재 예시:

```bash
python scripts/gen_dataset_v5/10_run_v5_all_chapters.py \
  --phase data \
  --round-label allchap_round1 \
  --force
```

중요:

- 이 단계가 끝나면 멈추지 말고 바로 학습 단계로 이어갑니다.
- 사용자가 멈추라고 하지 않았다면 `학습 가능` 보고만 하고 대기하지 않습니다.

### 5. LoRA 학습 실행

기본 명령:

```bash
python scripts/run_smoke_gpt_oss_20b.py \
  --config <training-config.json>
```

이어학습이면:

- config에 `init_adapter_path`가 실제로 있는지 확인합니다.
- 없으면 이어학습을 시도하지 않습니다.

### 6. 직접 검증 실행

학습 직후 반드시 직접 검증을 수행합니다.

```bash
python scripts/check_gpt_oss_model_output.py \
  --config <training-config.json> \
  --adapter-path <adapter-output-dir>
```

검증 결과는 아래를 확인합니다.

- 질문별 생성이 비어 있지 않은지
- 도메인 이탈이나 일반론이 심하지 않은지
- 특정 챕터 질문에 챕터 고유 개념이 실제로 반영되는지

### 7. vLLM 어댑터 서빙

직접 검증이 끝나면 필요 시 어댑터를 서빙합니다.

```bash
PORT=8001 \
ADAPTER_NAME=<served-model-name> \
ADAPTER_PATH="<adapter-output-dir>" \
bash scripts/run_vllm_adapter_server.sh
```

포트가 이미 사용 중이면 기존 프로세스를 먼저 확인하고, 사용자가 요청한 포트를 우선 맞춥니다.

### 8. vLLM 헬스체크

서빙 후 바로 점검합니다.

```bash
python scripts/check_vllm_adapter_server.py \
  --port 8001 \
  --model <served-model-name>
```

가능하면 `/v1/models`와 샘플 응답 둘 다 확인합니다.

### 9. 결과 보고

보고는 아래 순서를 지킵니다.

1. 데이터 준비 완료 여부
2. 학습 완료 여부
3. 직접 검증 결과
4. vLLM 서빙/검토 결과
5. 남은 리스크

## Stop Conditions

아래 상황에서는 다음 단계로 자동 진행하지 말고 먼저 수정합니다.

- config의 입력 경로가 실제 산출물과 다름
- 학습 입력 JSONL 또는 Harmony 파일이 없음
- adapter 출력 디렉터리가 비정상 상태임
- 직접 검증 결과가 비어 있음
- vLLM 서빙이 포트 충돌 또는 메모리 부족으로 실패함

## Example

사용자 요청:

`모든 챕터 학습, 검토 진행해`

적용 방식:

1. 현재 `*_all.jsonl`과 Harmony 파일 존재 확인
2. config 경로가 실제 파일명과 맞는지 확인
3. 맞지 않으면 config 수정
4. LoRA 학습 실행
5. 직접 검증 실행
6. 어댑터 서빙과 헬스체크 실행
7. 완료 여부를 단계별로 보고
