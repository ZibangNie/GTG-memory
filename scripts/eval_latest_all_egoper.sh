#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/eval_latest_all_egoper.sh <baseline_retrain|vm_warmstart>"
  exit 1
fi

RUN_TAG="$1"
REPO_ROOT="/root/autodl-tmp/GTG-memory"
TASKS=("tea" "oatmeal" "pinwheels" "quesadilla" "coffee")
RUN_TS="$(date +%m_%d_%H_%M_%S)"
BATCH_LOG_DIR="${REPO_ROOT}/logs/batch_runs/eval_${RUN_TS}_${RUN_TAG}"

mkdir -p "${BATCH_LOG_DIR}"

cd "${REPO_ROOT}"
source env.sh

for task in "${TASKS[@]}"; do
  if [[ "${RUN_TAG}" == "baseline_retrain" ]]; then
    CFG="configs/EgoPER/${task}/generated/vc_4omini_post_db0.6.baseline.train.json"
  elif [[ "${RUN_TAG}" == "vm_warmstart" ]]; then
    CFG="configs/EgoPER/${task}/generated/vc_4omini_post_db0.6.visual_memory.train.json"
  else
    echo "[ERROR] unsupported RUN_TAG: ${RUN_TAG}"
    exit 1
  fi

  if [[ ! -f "${CFG}" ]]; then
    echo "[SKIP] missing config: ${CFG}"
    continue
  fi

  LATEST_DIR="$(ls -dt ckpts/EgoPER/${task}/${RUN_TAG}_* 2>/dev/null | head -n 1 || true)"
  if [[ -z "${LATEST_DIR}" ]]; then
    echo "[SKIP] no run dir found for task=${task}, tag=${RUN_TAG}"
    continue
  fi

  LATEST_BASENAME="$(basename "${LATEST_DIR}")"

  echo "=============================="
  echo "[EVAL][START] task=${task}"
  echo "config=${CFG}"
  echo "dir=${LATEST_BASENAME}"
  echo "=============================="

  python main.py \
    --config "${CFG}" \
    --dir "${LATEST_BASENAME}" \
    --eval \
    2>&1 | tee "${BATCH_LOG_DIR}/eval_${RUN_TAG}_${task}.log"

  echo "[EVAL][DONE] task=${task}"
done
