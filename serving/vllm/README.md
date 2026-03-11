# vLLM Adapter Serving

이 디렉터리는 `gpt-oss-20b` LoRA adapter를 `vLLM`으로 로컬 서빙하는 중간 검증 런북을 둡니다.

## 현재 목표

- 베이스 모델: `unsloth/gpt-oss-20b-BF16`
- adapter: `llm_model_lora/gpt-oss-20b-smoke`
- 서빙 방식: `vLLM` OpenAI 호환 API
- 범위: `adapter` 서빙 검증

최종 목표인 `merge -> MXFP4 재양자화 -> vLLM` 전 단계로, 먼저 LoRA adapter 자체가 `vLLM`에서 로드되고 요청을 처리하는지 확인합니다.

## 계획 파일

- 메인 계획 파일: `.cursor/plans/main-plan.md`

## 실행 순서

1. `scripts/setup_vllm_env.sh`
2. `scripts/run_vllm_adapter_server.sh`
3. `scripts/check_vllm_adapter_server.py`

## 환경 분리 이유

현재 학습 환경은 커스텀 `torch` 빌드를 사용하고 있어 `vllm`과 직접 섞을 때 충돌 가능성이 있습니다.
따라서 `vLLM`은 `.venv-vllm`에 분리 설치합니다.

## 설치

```bash
bash scripts/setup_vllm_env.sh
```

## 서버 실행

기본값은 단일 GPU, `127.0.0.1:8000`, adapter 이름 `gpt-oss-20b-smoke`입니다.

```bash
bash scripts/run_vllm_adapter_server.sh
```

환경 변수로 주요 값을 바꿀 수 있습니다.

```bash
CUDA_VISIBLE_DEVICES=0 PORT=8001 ADAPTER_NAME=my-smoke bash scripts/run_vllm_adapter_server.sh
```

## 검증

```bash
python scripts/check_vllm_adapter_server.py --port 8000 --model gpt-oss-20b-smoke
```

검증 스크립트는 아래를 확인합니다.

- `/v1/models`에 base model과 LoRA adapter가 노출되는지
- `/v1/responses` 요청이 실제 `assistant` 출력 텍스트를 반환하는지
