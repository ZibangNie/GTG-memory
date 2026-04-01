#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/autodl-tmp/GTG-memory"
PIPELINE_TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG_DIR="${REPO_ROOT}/logs/batch_runs/pipeline_available_${PIPELINE_TS}"
MASTER_LOG="${MASTER_LOG_DIR}/pipeline.log"

mkdir -p "${MASTER_LOG_DIR}"

NUM_EPOCHS="${NUM_EPOCHS:-10}"
NUM_ITERATIONS="${NUM_ITERATIONS:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LOG_FREQ="${LOG_FREQ:-10}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
BACKGROUND_WEIGHT="${BACKGROUND_WEIGHT:-2.0}"

SHORT_DIM="${SHORT_DIM:-256}"
LONG_DIM="${LONG_DIM:-384}"
FUSION_DIM="${FUSION_DIM:-256}"
LONG_WRITE_CAP="${LONG_WRITE_CAP:-0.2}"
FUSION_DROPOUT="${FUSION_DROPOUT:-0.1}"
BACKBONE_LR="${BACKBONE_LR:-5e-5}"
VM_LR="${VM_LR:-1e-4}"

cd "${REPO_ROOT}"
source env.sh

# 主日志：终端实时可见，同时保存到文件
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "========================================"
echo "[PIPELINE START] ${PIPELINE_TS}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "NUM_EPOCHS=${NUM_EPOCHS}"
echo "NUM_ITERATIONS=${NUM_ITERATIONS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "LOG_FREQ=${LOG_FREQ}"
echo "LEARNING_RATE=${LEARNING_RATE}"
echo "WEIGHT_DECAY=${WEIGHT_DECAY}"
echo "BACKGROUND_WEIGHT=${BACKGROUND_WEIGHT}"
echo "SHORT_DIM=${SHORT_DIM}"
echo "LONG_DIM=${LONG_DIM}"
echo "FUSION_DIM=${FUSION_DIM}"
echo "LONG_WRITE_CAP=${LONG_WRITE_CAP}"
echo "FUSION_DROPOUT=${FUSION_DROPOUT}"
echo "BACKBONE_LR=${BACKBONE_LR}"
echo "VM_LR=${VM_LR}"
echo "MASTER_LOG=${MASTER_LOG}"
echo "========================================"

echo
echo "[STEP 1/6] Build available-only splits + configs"
python scripts/build_available_only_egoper_splits_and_configs.py \
  --repo_root "${REPO_ROOT}" \
  --num_epochs "${NUM_EPOCHS}" \
  --num_iterations "${NUM_ITERATIONS}" \
  --batch_size "${BATCH_SIZE}" \
  --log_freq "${LOG_FREQ}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --background_weight "${BACKGROUND_WEIGHT}" \
  --short_dim "${SHORT_DIM}" \
  --long_dim "${LONG_DIM}" \
  --fusion_dim "${FUSION_DIM}" \
  --long_write_cap "${LONG_WRITE_CAP}" \
  --fusion_dropout "${FUSION_DROPOUT}" \
  --backbone_learning_rate "${BACKBONE_LR}" \
  --vm_learning_rate "${VM_LR}"

TASK_JSON="${REPO_ROOT}/reports/task_probe/egoper_available_only_latest.json"

if [[ ! -f "${TASK_JSON}" ]]; then
  echo "[ERROR] missing task json: ${TASK_JSON}"
  exit 1
fi

echo
echo "[STEP 2/6] Load ready tasks from available-only report"
python - <<'PY'
import json
from pathlib import Path

task_json = Path("/root/autodl-tmp/GTG-memory/reports/task_probe/egoper_available_only_latest.json")
payload = json.loads(task_json.read_text(encoding="utf-8"))
print("[READY TASKS]")
for t in payload["ready_tasks"]:
    print(t)
PY

READY_TASKS=$(python - <<'PY'
import json
from pathlib import Path

task_json = Path("/root/autodl-tmp/GTG-memory/reports/task_probe/egoper_available_only_latest.json")
payload = json.loads(task_json.read_text(encoding="utf-8"))
tasks = payload.get("ready_tasks", [])
print(" ".join(tasks))
PY
)

if [[ -z "${READY_TASKS}" ]]; then
  echo "[ERROR] no ready_tasks found in ${TASK_JSON}"
  exit 1
fi

echo
echo "[STEP 3/6] Train baseline on available-only ready tasks"
for task in ${READY_TASKS}; do
  CFG="configs/EgoPER/${task}/generated_available_only/vc_4omini_post_db0.6.available_only.baseline.train.json"

  if [[ ! -f "${CFG}" ]]; then
    echo "[ERROR] missing baseline config: ${CFG}"
    exit 1
  fi

  echo "----------------------------------------"
  echo "[BASELINE][START] task=${task}"
  echo "config=${CFG}"
  echo "dir=baseline_retrain"
  echo "----------------------------------------"

  python main.py \
    --config "${CFG}" \
    --dir baseline_retrain

  echo "[BASELINE][DONE] task=${task}"
done

echo
echo "[STEP 4/6] Train visual-memory on available-only ready tasks"
for task in ${READY_TASKS}; do
  CFG="configs/EgoPER/${task}/generated_available_only/vc_4omini_post_db0.6.available_only.visual_memory.train.json"
  CKPT="ckpts/EgoPER/${task}/best/best_checkpoint.pth"

  if [[ ! -f "${CFG}" ]]; then
    echo "[ERROR] missing visual-memory config: ${CFG}"
    exit 1
  fi

  if [[ ! -f "${CKPT}" ]]; then
    echo "[ERROR] missing warmstart checkpoint: ${CKPT}"
    exit 1
  fi

  echo "----------------------------------------"
  echo "[VM][START] task=${task}"
  echo "config=${CFG}"
  echo "pretrained_backbone_ckpt=${CKPT}"
  echo "dir=vm_warmstart"
  echo "----------------------------------------"

  python main.py \
    --config "${CFG}" \
    --dir vm_warmstart

  echo "[VM][DONE] task=${task}"
done

echo
echo "[STEP 5/6] Eval latest baseline / vm on available-only ready tasks"
for task in ${READY_TASKS}; do
  BASE_CFG="configs/EgoPER/${task}/generated_available_only/vc_4omini_post_db0.6.available_only.baseline.train.json"
  VM_CFG="configs/EgoPER/${task}/generated_available_only/vc_4omini_post_db0.6.available_only.visual_memory.train.json"

  BASE_DIR="$(ls -dt ckpts/EgoPER/${task}/baseline_retrain_* 2>/dev/null | head -n 1 || true)"
  VM_DIR="$(ls -dt ckpts/EgoPER/${task}/vm_warmstart_* 2>/dev/null | head -n 1 || true)"

  if [[ -z "${BASE_DIR}" ]]; then
    echo "[ERROR] no baseline run dir found for task=${task}"
    exit 1
  fi

  if [[ -z "${VM_DIR}" ]]; then
    echo "[ERROR] no vm run dir found for task=${task}"
    exit 1
  fi

  BASE_NAME="$(basename "${BASE_DIR}")"
  VM_NAME="$(basename "${VM_DIR}")"

  echo "----------------------------------------"
  echo "[EVAL][BASELINE][START] task=${task}"
  echo "config=${BASE_CFG}"
  echo "dir=${BASE_NAME}"
  echo "----------------------------------------"

  python main.py \
    --config "${BASE_CFG}" \
    --dir "${BASE_NAME}" \
    --eval

  echo "[EVAL][BASELINE][DONE] task=${task}"

  echo "----------------------------------------"
  echo "[EVAL][VM][START] task=${task}"
  echo "config=${VM_CFG}"
  echo "dir=${VM_NAME}"
  echo "----------------------------------------"

  python main.py \
    --config "${VM_CFG}" \
    --dir "${VM_NAME}" \
    --eval

  echo "[EVAL][VM][DONE] task=${task}"
done

echo
echo "[STEP 6/6] Generate comparison report for available-only ready tasks"
python scripts/compare_egoper_runs.py \
  --repo_root "${REPO_ROOT}" \
  --baseline_tag baseline_retrain \
  --vm_tag vm_warmstart \
  --task_list_json "${TASK_JSON}"

echo
echo "[PIPELINE DONE] log=${MASTER_LOG}"