---
name: zeroin-training-validation
description: Creates the next `seed_round+1` dataset/training run in this repository, updates round-specific config paths, runs GPT-OSS LoRA adapter training, and reviews outputs against the p6 philosophy. Use when the user asks to continue round training, make round2/round3 style follow-up datasets, train after prompt changes, validate adapters, or check whether training results stay document-based, minimally connected, and non-expansive.
---

# Zeroin Training Validation

## Goal

이 저장소에서 현재 프롬프트/데이터 상태를 기준으로 `seed_round+1` 산출물을 만들고, `학습 -> 직접 검증 -> p6 철학 검토`까지 이어서 수행합니다.

## p6 Core

반드시 아래 철학을 유지합니다.

- 문서 충실성을 가장 우선한다.
- 답변은 독립적으로 이해 가능해야 한다.
- 이해 가능성을 이유로 새 정보를 만들면 안 된다.
- 설명은 허용하되, 의미 연결 수준에만 머물러야 한다.
- 답변은 `문서 기반 + 최소 연결 + 무확장`이어야 한다.
- 자연스럽게 표현하되, 정보는 추가하지 않는다.

## Default Operating Rules

- 사용자가 명시하지 않으면 `adapter-only` 경로를 기본으로 하고, 병합은 수행하지 않습니다.
- 사용자가 `round+1`을 요청하면 현재 최신 round 산출물과 adapter를 먼저 찾고, 가능하면 이어학습을 기본값으로 검토합니다.
- 데이터 준비가 끝났다고 보고만 하고 멈추지 않습니다. 중단 요청이 없으면 학습과 검증으로 이어갑니다.
- 검증은 최소 `직접 adapter 검증`까지 수행합니다.
- 사용자가 서빙/포트 확인도 원하거나 기존 흐름상 필요하면 `vLLM` 검토까지 이어갑니다.
- config의 경로와 실제 산출물이 다르면 학습 전에 먼저 맞춥니다.

## When To Use

다음 요청에서 이 스킬을 적용합니다.

- `round 진행`
- `seed_round+1 버전 만들고 학습`
- `프롬프트 수정했으니 다음 라운드 학습`
- `학습까지 진행 후 검토`
- `adapter 검증`
- `p6 철학에 맞는지 검토`

## Workflow

다음 체크리스트를 그대로 사용합니다.

```text
Round Training Progress
- [ ] 1. 현재 round 상태와 최신 산출물 확인
- [ ] 2. round+1 이름과 출력 경로 확정
- [ ] 3. 데이터셋/질문/답변 파이프라인 실행 또는 갱신
- [ ] 4. canonical JSONL / Harmony / config 정합성 확인
- [ ] 5. 필요 시 config 보정 및 이어학습 경로 확인
- [ ] 6. GPT-OSS LoRA 학습 실행
- [ ] 7. 직접 adapter 검증 실행
- [ ] 8. p6 철학 기준 검토
- [ ] 9. 필요 시 vLLM adapter 서빙 및 헬스체크
- [ ] 10. 결과와 남은 리스크 보고
```

## Step Guidance

### 1. 현재 round 상태 확인

- `scripts/gen_dataset_*/state/`
- `llm_datasets/seed_*/`
- `llm_datasets/rendered/gpt-oss/`
- `configs/`
- `llm_model_lora/`

를 먼저 확인합니다.

- 최신 canonical JSONL, Harmony 파일, config, adapter를 찾습니다.
- 이미 완료된 단계는 재사용하되, 프롬프트가 바뀐 경우 해당 round는 새 round로 다시 만듭니다.

### 2. round+1 이름 확정

- 기존 round 명명 규칙을 따릅니다.
- 사용자가 특정 라운드 이름을 주지 않으면 최신 round 다음 번호를 사용합니다.
- 파일명, config명, output_dir, served_model_name이 모두 같은 round를 가리키도록 맞춥니다.

### 3. 데이터 파이프라인 실행

- 프로젝트에 상위 오케스트레이터가 있으면 그 경로를 우선 사용합니다.
- 질문/답변 생성 프롬프트가 바뀌었다면 새 round 데이터로 다시 생성합니다.
- 기존 round 데이터를 합칠지, 새 round만 따로 학습할지는 사용자 요청과 현재 전략에 맞춰 결정합니다.

### 4. 입력 정합성 확인

학습 전에 아래를 반드시 확인합니다.

- canonical seed JSONL 존재 여부
- Harmony 렌더 결과 존재 여부
- config의 `dataset_path`, `prompt_source_path`, `output_dir` 정합성
- 이어학습이면 `init_adapter_path` 존재 여부
- validation report와 served model name이 새 round를 가리키는지

### 5. 학습 실행

- 기본 학습 명령은 `scripts/run_smoke_gpt_oss_20b.py --config <config>`입니다.
- 이어학습이면 최신 안정 adapter를 기본 후보로 보고 실제 경로를 검증한 뒤 사용합니다.
- 사용자가 병합을 요청하지 않으면 merge 단계는 건너뜁니다.

### 6. 직접 검증

학습 직후 반드시 `scripts/check_gpt_oss_model_output.py`로 adapter 검증을 수행합니다.

- 생성이 비어 있지 않은지
- 질문 대상 개념이 실제로 답에 반영되는지
- 문서 밖 일반론으로 평탄화되지 않는지
- 메타 문구를 그대로 답하지 않는지

### 7. p6 검토

직접 검증 결과를 아래 기준으로 검토합니다.

- 문서 기반인가
- 독립적으로 이해 가능한가
- 자연스럽게 연결되지만 새 정보가 추가되지 않았는가
- 의미 연결 수준을 넘는 비교/예외/일반화가 없는가
- 타입별 정합성이 맞는가

타입별 최소 기준:

- `definition`: 정의문이 실제로 들어가고, 질문이 묻지 않은 절차/예외/비교가 없는가
- `criteria`: 판단 기준만 남고 적용 절차나 비교 설명이 섞이지 않았는가
- `comparison`: 필요한 비교축만 남고 각 대상 정의/절차로 새지 않는가
- `application`: 적용 순서/조건/전환 규칙 중심으로 답하고 정의 재설명으로 새지 않는가

### 8. vLLM 검토

필요 시 adapter 서빙과 헬스체크를 수행합니다.

- 사용자가 지정한 포트를 우선합니다.
- 포트 충돌이 있으면 먼저 기존 프로세스를 확인합니다.
- `/v1/models`와 샘플 응답을 둘 다 확인합니다.

## Stop Conditions

아래 상황에서는 다음 단계로 자동 진행하지 말고 먼저 수정합니다.

- 입력 JSONL 또는 Harmony 파일이 없음
- config 경로가 실제 산출물과 다름
- `init_adapter_path`가 필요한데 존재하지 않음
- 직접 검증 결과가 비어 있음
- 직접 검증에서 p6 위반이 반복적으로 확인됨
- vLLM 서빙이 포트 충돌 또는 메모리 부족으로 실패함

## Reporting

보고는 아래 순서를 지킵니다.

1. 이번 round 이름과 생성 범위
2. 데이터 준비 완료 여부
3. 학습 완료 여부
4. 직접 검증 결과
5. p6 철학 기준 평가
6. vLLM 서빙/헬스체크 결과
7. 남은 리스크와 다음 액션

## Additional Resources

- 상세 명령 예시와 p6 검토 질문은 [reference.md](reference.md)를 따릅니다.
