# vLLM Serving

이 디렉터리는 `gpt-oss-20b`의 `adapter 검증 -> 필요 시 병합 BF16 검증 -> 최종 vLLM 서빙` 흐름을 정리한 런북을 둡니다.

또한 `NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`를 `vLLM`으로 서빙하기 위한 별도 런북과 파서 파일도 함께 둡니다.

## 현재 목표

- 베이스 모델: `unsloth/gpt-oss-20b-BF16`
- 2차 기본 adapter: `llm_model_lora/gpt-oss-20b-seed-v2-all`
- 병합 BF16 기본 경로: `llm_model_merged/gpt-oss-20b-seed-v2-all-bf16`
- 서빙 방식: `vLLM` OpenAI 호환 API

현재 운영 기본선은 `adapter 서빙`입니다.

즉, 품질이 충분히 확인되기 전에는 병합본보다 `LoRA adapter + vLLM + OpenWebUI` 조합을 우선 사용합니다.

현재 기본 검증 순서는 아래와 같습니다.

1. `seed_v2_all` 기준 LoRA 학습
2. LoRA adapter 직접 검증
3. LoRA adapter를 `vLLM`과 `OpenWebUI`에서 검증
4. adapter 품질이 충분할 때만 LoRA 병합 후 BF16 단일 모델 저장
5. 이후 병합 BF16 직접 검증
6. 마지막으로 `MXFP4` 재양자화 경로를 별도로 검증

## 계획 파일

- 메인 계획 파일: `.cursor/plans/main-plan.md`

## 실행 순서

### 1. vLLM 환경 준비

```bash
bash scripts/setup_vllm_env.sh
```

### 2. LoRA adapter 중간 검증

학습된 adapter만 먼저 붙여 `vLLM` 로딩과 응답 형식을 확인합니다.

```bash
ADAPTER_NAME=gpt-oss-20b-seed-v2-all \
ADAPTER_PATH=/home/work/dev_data/fine-tuning/llm_model_lora/gpt-oss-20b-seed-v2-all \
bash scripts/run_vllm_adapter_server.sh
```

```bash
python scripts/check_vllm_adapter_server.py --port 8000 --model gpt-oss-20b-seed-v2-all
```

현재 `OpenWebUI`에서 선택해야 하는 모델 이름도 `gpt-oss-20b-seed-v2-all`입니다.
`gpt-oss-20b-seed-v2-all-bf16`는 병합본을 따로 띄웠을 때만 사용하는 이름입니다.

### 3. LoRA 병합

이 단계는 adapter 품질이 충분히 확인된 뒤에만 진행합니다.

```bash
python scripts/merge_gpt_oss_lora.py \
  --config /home/work/dev_data/fine-tuning/configs/gpt_oss_20b_seed_v2_all.json
```

### 4. 병합 BF16 직접 검증

고정 질문셋은 config의 `validation_questions`를 사용합니다.

```bash
python scripts/check_gpt_oss_model_output.py \
  --config /home/work/dev_data/fine-tuning/configs/gpt_oss_20b_seed_v2_all.json
```

필요하면 adapter 상태도 직접 비교할 수 있습니다.

```bash
python scripts/check_gpt_oss_model_output.py \
  --config /home/work/dev_data/fine-tuning/configs/gpt_oss_20b_seed_v2_all.json \
  --model-path unsloth/gpt-oss-20b-BF16 \
  --adapter-path /home/work/dev_data/fine-tuning/llm_model_lora/gpt-oss-20b-seed-v2-all \
  --report-path /home/work/dev_data/fine-tuning/llm_model_lora/gpt-oss-20b-seed-v2-all/adapter_validation.json
```

### 5. 병합 BF16을 vLLM으로 서빙

```bash
MODEL_PATH=/home/work/dev_data/fine-tuning/llm_model_merged/gpt-oss-20b-seed-v2-all-bf16 \
SERVED_MODEL_NAME=gpt-oss-20b-seed-v2-all-bf16 \
bash scripts/run_vllm_model_server.sh
```

```bash
python scripts/check_vllm_adapter_server.py --port 8000 --model gpt-oss-20b-seed-v2-all-bf16
```

### 6. MXFP4 최종 목표

현재 저장소에선 `gpt-oss-20b`용 `MXFP4` 재양자화 스크립트를 아직 고정하지 않았습니다.
따라서 지금 단계의 기본선은 `adapter 직접 검증 -> adapter vLLM/OpenWebUI 검증`을 먼저 안정화하는 것입니다.

## 환경 분리 이유

현재 학습 환경은 커스텀 `torch` 빌드를 사용하고 있어 `vllm`과 직접 섞을 때 충돌 가능성이 있습니다.
따라서 `vLLM`은 `.venv-vllm`에 분리 설치합니다.

## 설치

```bash
bash scripts/setup_vllm_env.sh
```

## 서버 실행

adapter 검증 기본값은 단일 GPU, `127.0.0.1:8000`, adapter 이름 `gpt-oss-20b-smoke`입니다.

```bash
bash scripts/run_vllm_adapter_server.sh
```

병합 BF16 검증 기본값은 `gpt-oss-20b-seed-v2-all-bf16` 병합 디렉터리입니다.

```bash
bash scripts/run_vllm_model_server.sh
```

환경 변수로 주요 값을 바꿀 수 있습니다.

```bash
CUDA_VISIBLE_DEVICES=0 PORT=8001 ADAPTER_NAME=my-smoke bash scripts/run_vllm_adapter_server.sh
```

```bash
CUDA_VISIBLE_DEVICES=0 PORT=8001 MODEL_PATH=/path/to/merged-model SERVED_MODEL_NAME=my-merged bash scripts/run_vllm_model_server.sh
```

## 검증

```bash
python scripts/check_vllm_adapter_server.py --port 8000 --model gpt-oss-20b-smoke
```

검증 스크립트는 아래를 확인합니다.

- `/v1/models`에 base model과 LoRA adapter가 노출되는지
- `OpenWebUI`와 동일한 `chat/completions` 호출이 실제 `assistant` 출력 텍스트를 반환하는지
- `responses` API가 실패하면 그 오류도 함께 기록하는지

## Nemotron 3 Super NVFP4

`GGUF` 대신 `vLLM` 공식 실행 예시가 있는 `NVFP4` 변형을 사용합니다.

### 현재 판단

- 대상 모델: `unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`
- 서빙 엔진: `vLLM`
- 분리 환경: `.venv-vllm-nemotron3`
- 기본 포트: `5000`
- 기본 served model name: `nvidia/nemotron-3-super`

### 실행 순서

1. `bash scripts/setup_vllm_nemotron3_env.sh`
2. `bash scripts/prepare_nemotron3_nvfp4_assets.sh`
3. `bash scripts/run_vllm_nemotron3_nvfp4_server.sh`
4. `PYTHONNOUSERSITE=1 env -u PYTHONPATH ./.venv-vllm-nemotron3/bin/python scripts/check_vllm_nemotron3_nvfp4_server.py --port 5000`

### 기본 실행 예시

```bash
bash scripts/setup_vllm_nemotron3_env.sh
bash scripts/prepare_nemotron3_nvfp4_assets.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_vllm_nemotron3_nvfp4_server.sh
PYTHONNOUSERSITE=1 env -u PYTHONPATH ./.venv-vllm-nemotron3/bin/python scripts/check_vllm_nemotron3_nvfp4_server.py --port 5000
```

### 주요 환경 변수

```bash
CUDA_VISIBLE_DEVICES=0,1
PORT=5001
MAX_MODEL_LEN=262144
GPU_MEMORY_UTILIZATION=0.9
MODEL_NAME=unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
SERVED_MODEL_NAME=nvidia/nemotron-3-super
```

### 주의

- 모델 첫 실행 시 가중치는 `llm_model_base/nemotron-3-super-120b-a12b-nvfp4` 아래로 내려받습니다.
- 기본 컨텍스트는 `262144`로 잡았습니다. `1M` 컨텍스트는 메모리 여유를 확인한 뒤 별도 조정합니다.
- reasoning parser 파일은 `serving/vllm/nemotron3_super_nvfp4/super_v3_reasoning_parser.py`에 저장합니다.
- 현재 셸에 전역 `PYTHONPATH`가 잡혀 있다면 `env -u PYTHONPATH`로 제거한 뒤 전용 `venv`를 사용하는 편이 안전합니다.
