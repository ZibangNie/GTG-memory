# 文件：scripts/run_available_egoper_pipeline.sh
# 作用：
# 1) 自动探测“当前数据可用”的 EgoPER 任务
# 2) 只为 ready tasks 生成 config
# 3) 只训练 ready tasks 的 baseline / visual-memory
# 4) 对 ready tasks 做 eval
# 5) 生成 ready tasks 的对比报告
#
# 一键运行：
# bash scripts/run_available_egoper_pipeline.sh

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

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "========================================"
echo "[PIPELINE START] ${PIPELINE_TS}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "========================================"

echo
echo "[STEP 1/6] Probe available EgoPER tasks"
python scripts/probe_available_egoper_tasks.py --repo_root "${REPO_ROOT}"

TASK_JSON="${REPO_ROOT}/reports/task_probe/egoper_ready_tasks_latest.json"

echo
echo "[STEP 2/6] Generate configs only for ready tasks"
python scripts/gen_all_egoper_task_configs.py \
  --repo_root "${REPO_ROOT}" \
  --task_list_json "${TASK_JSON}" \
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

echo
echo "[STEP 3/6] Train baseline on ready tasks"
python - <<'PY'
import json
from pathlib import Path
task_json = Path("/root/autodl-tmp/GTG-memory/reports/task_probe/egoper_ready_tasks_latest.json")
payload = json.loads(task_json.read_text())
for t in payload["ready_tasks"]:
    print(t)
PY
READY_TASKS=$(python - <<'PY'
import json
from pathlib import Path
task_json = Path("/root/autodl-tmp/GTG-memory/reports/task_probe/egoper_ready_tasks_latest.json")
payload = json.loads(task_json.read_text())
print(" ".join(payload["ready_tasks"]))
PY
)

for task in ${READY_TASKS}; do
  CFG="configs/EgoPER/${task}/generated/vc_4omini_post_db0.6.baseline.train.json"
  echo "[BASELINE][START] ${task}"
  python main.py --config "${CFG}" --dir baseline_retrain
  echo "[BASELINE][DONE] ${task}"
done

echo
echo "[STEP 4/6] Train visual-memory on ready tasks"
for task in ${READY_TASKS}; do
  CFG="configs/EgoPER/${task}/generated/vc_4omini_post_db0.6.visual_memory.train.json"
  echo "[VM][START] ${task}"
  python main.py --config "${CFG}" --dir vm_warmstart
  echo "[VM][DONE] ${task}"
done

echo
echo "[STEP 5/6] Eval latest baseline / vm on ready tasks"
for task in ${READY_TASKS}; do
  BASE_DIR=$(ls -dt ckpts/EgoPER/${task}/baseline_retrain_* | head -n 1)
  VM_DIR=$(ls -dt ckpts/EgoPER/${task}/vm_warmstart_* | head -n 1)

  BASE_NAME=$(basename "${BASE_DIR}")
  VM_NAME=$(basename "${VM_DIR}")

  python main.py \
    --config "configs/EgoPER/${task}/generated/vc_4omini_post_db0.6.baseline.train.json" \
    --dir "${BASE_NAME}" \
    --eval

  python main.py \
    --config "configs/EgoPER/${task}/generated/vc_4omini_post_db0.6.visual_memory.train.json" \
    --dir "${VM_NAME}" \
    --eval
done

echo
echo "[STEP 6/6] Generate comparison report for ready tasks only"
python scripts/compare_egoper_runs.py \
  --repo_root "${REPO_ROOT}" \
  --baseline_tag baseline_retrain \
  --vm_tag vm_warmstart \
  --task_list_json "${TASK_JSON}"

echo
echo "[PIPELINE DONE] log=${MASTER_LOG}"