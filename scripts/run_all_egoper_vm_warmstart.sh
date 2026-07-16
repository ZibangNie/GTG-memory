#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASKS=("tea" "oatmeal" "pinwheels" "quesadilla" "coffee")
RUN_TAG="vm_warmstart"
RUN_TS="$(date +%m_%d_%H_%M_%S)"
BATCH_LOG_DIR="${REPO_ROOT}/logs/batch_runs/${RUN_TS}"

mkdir -p "${BATCH_LOG_DIR}"

cd "${REPO_ROOT}"
source env.sh
DATA_ROOT="${GTG_EGOPER_DATA_ROOT:-${GTG_DATA}/EgoPER}"

for task in "${TASKS[@]}"; do
  CFG="configs/EgoPER/${task}/generated/vc_4omini_post_db0.6.visual_memory.train.json"
  CKPT="${GTG_CKPT_ROOT}/EgoPER/${task}/best/best_checkpoint.pth"

  if [[ ! -f "${CFG}" ]]; then
    echo "[SKIP] missing config: ${CFG}"
    continue
  fi

  if [[ ! -f "${CKPT}" ]]; then
    echo "[SKIP] missing official best checkpoint: ${CKPT}"
    continue
  fi

  echo "=============================="
  echo "[VM][START] task=${task}"
  echo "config=${CFG}"
  echo "pretrained_backbone_ckpt=${CKPT}"
  echo "dir=${RUN_TAG}"
  echo "=============================="

  python main.py \
    --config "${CFG}" \
    --data-root "${DATA_ROOT}" \
    --ckpt-root "${GTG_CKPT_ROOT}" \
    --dir "${RUN_TAG}" \
    2>&1 | tee "${BATCH_LOG_DIR}/vm_${task}.log"

  echo "[VM][DONE] task=${task}"
done
