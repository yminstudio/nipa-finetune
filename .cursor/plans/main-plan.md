# GPT-OSS 20B 메인 계획안

## 목표

- 이번 프로젝트의 유일한 대상 모델은 `unsloth/gpt-oss-20b-BF16`이다.
- `gpt-oss-120b`는 이번 프로젝트 범위에서 제외한다.
- 목표는 작은 seed 데이터에서 시작해 학습 파이프라인을 검증하고, 최종적으로 `vLLM` 서빙 가능한 배포 artifact를 만드는 것이다.

## 모델 및 베이스

- 단일 학습/배포 모델: `unsloth/gpt-oss-20b-BF16`
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
- 현재 운영 기준은 `adapter` 서빙 우선이다.
- `LoRA` 품질이 충분히 확인되기 전에는 병합을 서두르지 않는다.
- 메인 장기 목표 경로는 `merge -> MXFP4 재양자화 -> vLLM`이지만, 현재 단계의 실제 검증 경로는 `adapter -> vLLM -> OpenWebUI`다.

## 장비 환경

- GPU: `NVIDIA H200` 3장
- 드라이버: `570.86.10`
- CPU: `Intel Xeon Platinum 8570` 계열
- 논리 CPU: `224`
- 메모리: 약 `2.0 TB RAM`

## 환경 해석

- 현재 장비는 `unsloth/gpt-oss-20b-BF16`의 BF16 학습, LoRA 병합, MXFP4 재양자화, vLLM 서빙을 모두 검토하기에 충분하다.
- 이번 프로젝트는 장비 부족보다 `데이터 규모`와 `파이프라인 안정화`가 더 중요한 단계다.

## 실행 순서

1. `unsloth/gpt-oss-20b-BF16`를 1차 베이스로 확정한다.
2. seed 데이터와 이후 생성할 소량 데이터로 Harmony 렌더링 파이프라인을 고정한다.
3. `BF16 LoRA` 스모크 테스트를 먼저 수행한다.
4. `attention-only LoRA` 설정을 안정화한다.
5. `adapter`를 `vLLM`과 `OpenWebUI`에서 직접 검증한다.
6. 품질이 충분할 때만 LoRA 병합 결과를 만든다.
7. 이후 `MXFP4` 재양자화 경로를 검증한다.
8. 테스트 문서에 결과를 기록하고 데이터셋 확장 여부를 판단한다.

## 보류 사항

- `BF16`와 `QLoRA nf4` 중 실제 1차 실행 설정을 어디까지 열어둘지
- `LoRA 병합 -> MXFP4 재양자화` 툴체인 선택
- `vLLM`에서 사용할 최종 배포 artifact 저장 구조
- 테스트 문서의 구체적인 기록 형식

## 현재 실행 자산

- 학습 config:
  - `configs/gpt_oss_20b_smoke.json`
  - `configs/gpt_oss_20b_seed_ch01_ch02.json`
  - `configs/gpt_oss_20b_seed_v2_all.json`
- 학습 실행:
  - `scripts/run_smoke_gpt_oss_20b.py`
- 병합:
  - `scripts/merge_gpt_oss_lora.py`
- 직접 검증:
  - `scripts/check_gpt_oss_model_output.py`
- vLLM 서빙:
  - `scripts/run_vllm_adapter_server.sh`
  - `scripts/run_vllm_model_server.sh`
  - `scripts/check_vllm_adapter_server.py`

## 현재 기준 실행 경로

1. `python scripts/run_smoke_gpt_oss_20b.py --config configs/gpt_oss_20b_seed_v2_all.json`
2. `python scripts/check_gpt_oss_model_output.py --config configs/gpt_oss_20b_seed_v2_all.json --model-path unsloth/gpt-oss-20b-BF16 --adapter-path llm_model_lora/gpt-oss-20b-seed-v2-all --report-path llm_model_lora/gpt-oss-20b-seed-v2-all/adapter_validation.json`
3. `ADAPTER_NAME=gpt-oss-20b-seed-v2-all ADAPTER_PATH=llm_model_lora/gpt-oss-20b-seed-v2-all bash scripts/run_vllm_adapter_server.sh`
4. `python scripts/check_vllm_adapter_server.py --port 8000 --model gpt-oss-20b-seed-v2-all`
5. `OpenWebUI`에서 `gpt-oss-20b-seed-v2-all`를 선택해 실제 대화 품질을 점검한다.
6. 병합은 위 검증이 통과한 뒤에만 진행한다.

## 2차 학습안

- 2차 config: `configs/gpt_oss_20b_seed_v2_all_round2.json`
- 2차 계획 문서: `docs/gpt_oss_round2_plan.md`
- 변경 요약:
  - `max_length=1024`
  - `gradient_accumulation_steps=4`
  - `max_steps=240`
  - `learning_rate=1e-4`
  - `lora_r=16`, `lora_alpha=32`
  - `target_modules=q_proj,k_proj,v_proj,o_proj`
- 2차 검증은 병합 전 `adapter + vLLM + OpenWebUI` 기준으로 진행한다.

## 다음 우선순위

- 1순위: `seed_v2_all` adapter를 `vLLM + OpenWebUI`에서 안정적으로 호출되는 상태로 고정한다.
- 2순위: `configs/gpt_oss_20b_seed_v2_all_round2.json` 기준 2차 학습을 실행하고, 대표 질문셋 응답 품질을 기록한다.
- 3순위: `round2` 결과를 바탕으로 canonical 데이터의 메타성 문구와 일반론 응답 유발 샘플을 정비한다.
- 4순위: 데이터 정비 후에만 다음 학습 라운드를 연다.
