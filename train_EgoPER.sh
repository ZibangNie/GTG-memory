#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${REPO_ROOT}/env.sh"

DATA_ROOT="${GTG_EGOPER_DATA_ROOT:-${GTG_DATA}/EgoPER}"
CONFIGS=(
  "configs/EgoPER/tea/vc_4omini_post_db0.6.json"
  "configs/EgoPER/oatmeal/vc_4omini_post_db0.4.json"
  "configs/EgoPER/pinwheels/vc_4omini_post_db0.3.json"
  "configs/EgoPER/quesadilla/vc_4omini_post_db0.2.json"
  "configs/EgoPER/coffee/vc_4omini_post_db0.3_ndb0.json"
)

for config in "${CONFIGS[@]}"; do
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python main.py \
    --config "${config}" \
    --data-root "${DATA_ROOT}" \
    --ckpt-root "${GTG_CKPT_ROOT}" \
    --dir best
done
