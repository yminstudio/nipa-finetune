#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/work/dev_data/fine-tuning"
VENV_DIR="${ROOT_DIR}/.venv-vllm"

export PYTHONNOUSERSITE=1
unset PYTHONPATH

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/pip" install vllm
"${VENV_DIR}/bin/pip" install tokenizers requests huggingface_hub

echo "vLLM environment ready: ${VENV_DIR}"
