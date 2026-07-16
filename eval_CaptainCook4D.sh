#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${REPO_ROOT}/env.sh"

DATA_ROOT="${GTG_CAPTAINCOOK_DATA_ROOT:-${GTG_DATA}/CaptainCook4D}"
CONFIGS=(
  "configs/CaptainCook4D/breakfastburritos/vc_4omini_post_db0.0_ndb0_win100.json"
  "configs/CaptainCook4D/microwaveeggsandwich/vc_4omini_post_db0.0_ndb0.json"
  "configs/CaptainCook4D/spicedhotchocolate/vc_4omini_post_db0.1_ndb0.json"
  "configs/CaptainCook4D/cucumberraita/vc_4omini_post_db-0.2_ndb0.json"
  "configs/CaptainCook4D/ramen/vc_4omini_post_db0.3.json"
)

for config in "${CONFIGS[@]}"; do
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python main.py \
    --config "${config}" \
    --data-root "${DATA_ROOT}" \
    --ckpt-root "${GTG_CKPT_ROOT}" \
    --dir best \
    --eval
done
