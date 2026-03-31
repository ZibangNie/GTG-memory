# 文件 2：scripts/run_all_egoper_pipeline.sh
# 作用：
# 1) 生成全部任务 config
# 2) 跑全部 baseline_retrain
# 3) 跑全部 vm_warmstart
# 4) 对两者分别做 latest eval
# 5) 自动生成对比报告
#
# 一键运行方式：
#   bash scripts/run_all_egoper_pipeline.sh
#
# 可调环境变量（可选）：
#   NUM_EPOCHS=10 NUM_ITERATIONS=100 BATCH_SIZE=2 LOG_FREQ=10 bash scripts/run_all_egoper_pipeline.sh

cd /root/autodl-tmp/GTG-memory

cat > scripts/run_all_egoper_pipeline.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/autodl-tmp/GTG-memory"
PIPELINE_TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG_DIR="${REPO_ROOT}/logs/batch_runs/pipeline_${PIPELINE_TS}"
MASTER_LOG="${MASTER_LOG_DIR}/pipeline.log"

mkdir -p "${MASTER_LOG_DIR}"

# 可调参数（不给就用默认）
NUM_EPOCHS="${NUM_EPOCHS:-10}"
NUM_ITERATIONS="${NUM_ITERATIONS:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LOG_FREQ="${LOG_FREQ:-10}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
BACKGROUND_WEIGHT="${BACKGROUND_WEIGHT:-2.0}"
DROP_BASE="${DROP_BASE:-0.6}"

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
echo "NUM_EPOCHS=${NUM_EPOCHS}"
echo "NUM_ITERATIONS=${NUM_ITERATIONS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "LOG_FREQ=${LOG_FREQ}"
echo "========================================"

echo
echo "[STEP 1/5] Generate all configs"
python scripts/gen_all_egoper_task_configs.py \
  --repo_root "${REPO_ROOT}" \
  --num_epochs "${NUM_EPOCHS}" \
  --num_iterations "${NUM_ITERATIONS}" \
  --batch_size "${BATCH_SIZE}" \
  --log_freq "${LOG_FREQ}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --background_weight "${BACKGROUND_WEIGHT}" \
  --drop_base "${DROP_BASE}" \
  --short_dim "${SHORT_DIM}" \
  --long_dim "${LONG_DIM}" \
  --fusion_dim "${FUSION_DIM}" \
  --long_write_cap "${LONG_WRITE_CAP}" \
  --fusion_dropout "${FUSION_DROPOUT}" \
  --backbone_learning_rate "${BACKBONE_LR}" \
  --vm_learning_rate "${VM_LR}"

echo
echo "[STEP 2/5] Train all baseline tasks"
bash scripts/run_all_egoper_baseline.sh

echo
echo "[STEP 3/5] Train all visual-memory tasks"
bash scripts/run_all_egoper_vm_warmstart.sh

echo
echo "[STEP 4/5] Eval latest baseline tasks"
bash scripts/eval_latest_all_egoper.sh baseline_retrain

echo
echo "[STEP 5/5] Eval latest visual-memory tasks"
bash scripts/eval_latest_all_egoper.sh vm_warmstart

echo
echo "[FINAL] Generate comparison report"
python scripts/compare_egoper_runs.py \
  --repo_root "${REPO_ROOT}" \
  --baseline_tag baseline_retrain \
  --vm_tag vm_warmstart

echo
echo "[LATEST RUN DIRS]"
bash scripts/show_latest_result_paths.sh

echo
echo "[PIPELINE DONE] log=${MASTER_LOG}"
SH

chmod +x scripts/run_all_egoper_pipeline.sh