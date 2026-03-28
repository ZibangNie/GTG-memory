#!/usr/bin/env bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/GTG

export GTG_ROOT=/root/autodl-tmp/GTG-memory
export GTG_DATA=/root/autodl-tmp/data
export GTG_MODELS=/root/autodl-tmp/models
export GTG_CACHE=/root/autodl-tmp/.cache

export PIP_CACHE_DIR=/root/autodl-tmp/.cache/pip
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/.cache/huggingface/transformers
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/.cache/huggingface/hub

cd /root/autodl-tmp/GTG-memory
