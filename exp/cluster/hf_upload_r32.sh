#!/bin/bash
# Task #200 follow-up: replace the HF Qwen1.5 real-quant artifact (r20) with
# the r32 canonical re-export (rank=32, G=128, attn VQ2, 2.152 bits, PPL 7.138).
# Network-only (no GPU) — safe to run outside SLURM test:0.
# Usage:  export HF_TOKEN=hf_xxx   (do NOT hardcode secrets)
#         bash exp/cluster/hf_upload_r32.sh
set -uo pipefail
: "${HF_TOKEN:?set HF_TOKEN env var before running}"
CLI=/home/qyyang/repo/GLoRCQ/.venv/bin/huggingface-cli
SRC=/mnt/Data/yqy/resource_dir/glorcq_paper_exp/qwen15_r32_real
REPO=Tsingyow/GLoRCQ-qwen1.5-moe-a2.7b-fair-grassmann-real
echo "=========== [$(date)] uploading $SRC -> $REPO ==========="
# --delete mirrors: removes remote shards/CLI not present locally (r20 leftovers).
$CLI upload "$REPO" "$SRC" . \
    --token "$HF_TOKEN" \
    --delete "model-*.safetensors" --delete "cross_layer_info.pt" \
    --commit-message "Replace r20 artifact with r32 canonical: rank=32, G=128, attn VQ2 (2.152 bits), fake-quant PPL 7.138 - same operating point as paper Table 1"
echo "=========== [$(date)] UPLOAD DONE (exit $?) ==========="
