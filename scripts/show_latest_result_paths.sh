#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/autodl-tmp/GTG-memory"
TASKS=("tea" "oatmeal" "pinwheels" "quesadilla" "coffee")

cd "${REPO_ROOT}"

for task in "${TASKS[@]}"; do
  BASE_DIR="$(ls -dt ckpts/EgoPER/${task}/baseline_retrain_* 2>/dev/null | head -n 1 || true)"
  VM_DIR="$(ls -dt ckpts/EgoPER/${task}/vm_warmstart_* 2>/dev/null | head -n 1 || true)"

  echo "----------------------------------------"
  echo "TASK: ${task}"
  echo "LATEST_BASELINE: ${BASE_DIR:-<none>}"
  echo "LATEST_VM:       ${VM_DIR:-<none>}"
done
