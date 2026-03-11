#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/work/dev_data/fine-tuning"
VENV_DIR="${ROOT_DIR}/.venv-vllm"
VENV_SITE_PACKAGES="${VENV_DIR}/lib/python3.12/site-packages"

MODEL_NAME="${MODEL_NAME:-unsloth/gpt-oss-20b-BF16}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-${ROOT_DIR}/llm_model_base}"
ADAPTER_NAME="${ADAPTER_NAME:-gpt-oss-20b-smoke}"
ADAPTER_PATH="${ADAPTER_PATH:-${ROOT_DIR}/llm_model_lora/gpt-oss-20b-smoke}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-bfloat16}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

if [[ ! -x "${VENV_DIR}/bin/vllm" ]]; then
  echo "vLLM is not installed. Run scripts/setup_vllm_env.sh first." >&2
  exit 1
fi

if [[ ! -d "${ADAPTER_PATH}" ]]; then
  echo "Adapter path not found: ${ADAPTER_PATH}" >&2
  exit 1
fi

export PYTHONPATH="${VENV_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1

exec "${VENV_DIR}/bin/vllm" serve "${MODEL_NAME}" \
  --download-dir "${MODEL_CACHE_DIR}" \
  --trust-remote-code \
  --dtype "${DTYPE}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --enable-lora \
  --lora-modules "${ADAPTER_NAME}=${ADAPTER_PATH}"
