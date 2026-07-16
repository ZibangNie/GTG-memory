#!/usr/bin/env bash

GTG_ROOT="${GTG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
GTG_DATA="${GTG_DATA:-${GTG_ROOT}/data}"
GTG_CKPT_ROOT="${GTG_CKPT_ROOT:-${GTG_ROOT}/ckpts}"
GTG_CACHE="${GTG_CACHE:-${GTG_ROOT}/.cache}"

export GTG_ROOT GTG_DATA GTG_CKPT_ROOT GTG_CACHE
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${GTG_CACHE}/pip}"
export HF_HOME="${HF_HOME:-${GTG_CACHE}/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"

if [[ -n "${GTG_CONDA_ENV:-}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] GTG_CONDA_ENV is set but conda is not available." >&2
    return 1 2>/dev/null || exit 1
  fi
  eval "$(conda shell.bash hook)"
  conda activate "${GTG_CONDA_ENV}"
fi

mkdir -p "${GTG_CACHE}"
cd "${GTG_ROOT}"
