# GPT-OSS 20B 메인 계획안

## 목표

- 이번 프로젝트는 `gpt-oss-20b` 단일 트랙으로 진행한다.
- `gpt-oss-120b`는 이번 프로젝트의 파인튜닝 대상에서 제외한다.
- 목표는 작은 seed 데이터에서 시작해 학습 파이프라인을 검증하고, 최종적으로 `vLLM` 서빙 가능한 배포 artifact를 만드는 것이다.

## 모델 및 베이스

- 1차 학습 모델: `gpt-oss-20b`
- 1차 학습 베이스: `unsloth/gpt-oss-20b-BF16`
- 로딩 기본 원칙:
  - `Transformers` 기반
  - `trust_remote_code=True`
  - `torch_dtype=torch.bfloat16`

## 데이터 및 포맷

- 입력 seed 기준 파일:
  - `llm_datasets/seed_v1/seed_ch01_first3.json`
- `gpt-oss`는 일반 chat `messages`를 그대로 쓰는 것이 아니라 `Harmony` 렌더링을 전제로 한다.
- canonical 데이터는 `messages` 구조로 유지하고, 학습 직전 렌더링한다.
- 렌더링 스크립트:
  - `llm_datasets/render_gpt-oss-harmony.py`
- 렌더링 산출물 예시:
  - `llm_datasets/rendered/gpt-oss/seed_ch01_first3_harmony.jsonl`

## 학습 원칙

- OpenAI Cookbook 기준으로 `MXFP4` 상태에서 직접 학습하는 경로는 사용하지 않는다.
- 학습 기본 경로:
  - `BF16` 또는 필요 시 `QLoRA 4-bit nf4`
  - `LoRA` 파인튜닝
- 1차 기본안은 `BF16 LoRA`이다.
- 학습 스택 기본안:
  - `Transformers`
  - `TRL (SFTTrainer)`
  - `PEFT`

## LoRA 전략

- 초기 LoRA 타깃은 `attention only`로 시작한다.
- baseline 예시:
  - `q_proj`
  - `v_proj`
- `MoE expert projection`은 초기 기본안에서 제외한다.
- 도메인 적합성이 부족할 때만 확장 검토한다.

## 양자화 및 배포 전략

- `gpt-oss`는 `MXFP4` 자체 양자화 특성을 가진 모델이므로, 학습 정밀도와 배포 정밀도를 분리해서 본다.
- 최종 목표 경로:
  1. BF16 베이스 로드
  2. LoRA 학습
  3. LoRA 병합
  4. BF16 결과물 저장
  5. `MXFP4` 재양자화
  6. `vLLM` 서빙

## 서빙 방침

- 서빙 엔진은 `vLLM`으로 고정한다.
- 속도가 중요하므로 최종 목표는 `MXFP4` artifact 기반 서빙이다.
- `adapter` 서빙은 필요 시 중간 검증용으로만 사용한다.
- 메인 배포 경로는 `merge -> MXFP4 재양자화 -> vLLM`이다.

## 장비 환경

- GPU: `NVIDIA H200` 3장
- 드라이버: `570.86.10`
- CPU: `Intel Xeon Platinum 8570` 계열
- 논리 CPU: `224`
- 메모리: 약 `2.0 TB RAM`

## 환경 해석

- 현재 장비는 `gpt-oss-20b`의 BF16 학습, LoRA 병합, MXFP4 재양자화, vLLM 서빙을 모두 검토하기에 충분하다.
- 이번 프로젝트는 장비 부족보다 `데이터 규모`와 `파이프라인 안정화`가 더 중요한 단계다.

## 실행 순서

1. `unsloth/gpt-oss-20b-BF16`를 1차 베이스로 확정한다.
2. seed 데이터와 이후 생성할 소량 데이터로 Harmony 렌더링 파이프라인을 고정한다.
3. `BF16 LoRA` 스모크 테스트를 먼저 수행한다.
4. `attention-only LoRA` 설정을 안정화한다.
5. LoRA 병합 결과를 만든다.
6. `MXFP4` 재양자화 경로를 검증한다.
7. `vLLM`에서 실제 서빙 성능을 확인한다.
8. 테스트 문서에 결과를 기록하고 데이터셋 확장 여부를 판단한다.

## 보류 사항

- `BF16`와 `QLoRA nf4` 중 실제 1차 실행 설정을 어디까지 열어둘지
- `LoRA 병합 -> MXFP4 재양자화` 툴체인 선택
- `vLLM`에서 사용할 최종 배포 artifact 저장 구조
- 테스트 문서의 구체적인 기록 형식
