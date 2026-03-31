#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/autodl-tmp/GTG-memory"
TASKS=("tea" "oatmeal" "pinwheels" "quesadilla" "coffee")
RUN_TAG="baseline_retrain"
RUN_TS="$(date +%m_%d_%H_%M_%S)"
BATCH_LOG_DIR="${REPO_ROOT}/logs/batch_runs/${RUN_TS}"

mkdir -p "${BATCH_LOG_DIR}"

cd "${REPO_ROOT}"
source env.sh

for task in "${TASKS[@]}"; do
  CFG="configs/EgoPER/${task}/generated/vc_4omini_post_db0.6.baseline.train.json"
  if [[ ! -f "${CFG}" ]]; then
    echo "[SKIP] missing config: ${CFG}"
    continue
  fi

  echo "=============================="
  echo "[BASELINE][START] task=${task}"
  echo "config=${CFG}"
  echo "dir=${RUN_TAG}"
  echo "=============================="

  python main.py \
    --config "${CFG}" \
    --dir "${RUN_TAG}" \
    2>&1 | tee "${BATCH_LOG_DIR}/baseline_${task}.log"

  echo "[BASELINE][DONE] task=${task}"
done
