#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/work/dev_data/fine-tuning"
VENV_DIR="${ROOT_DIR}/.venv-vllm"
VENV_SITE_PACKAGES="${VENV_DIR}/lib/python3.12/site-packages"

MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/llm_model_merged/gpt-oss-20b-seed-v2-all-bf16}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-${ROOT_DIR}/llm_model_base}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gpt-oss-20b-seed-v2-all-bf16}"
TOKENIZER_NAME="${TOKENIZER_NAME:-unsloth/gpt-oss-20b-BF16}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-bfloat16}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

if [[ ! -x "${VENV_DIR}/bin/vllm" ]]; then
  echo "vLLM is not installed. Run scripts/setup_vllm_env.sh first." >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]] && [[ ! "${MODEL_PATH}" =~ ^[^/]+/[^/]+$ ]]; then
  echo "Model path not found: ${MODEL_PATH}" >&2
  exit 1
fi

unset PYTHONPATH
export PYTHONPATH="${VENV_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1

exec "${VENV_DIR}/bin/vllm" serve "${MODEL_PATH}" \
  --download-dir "${MODEL_CACHE_DIR}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tokenizer "${TOKENIZER_NAME}" \
  --trust-remote-code \
  --dtype "${DTYPE}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}"
