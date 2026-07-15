#!/bin/bash
# Download the 3 fair-grassmann-real models from HF for inference-framework testing.
# Network-only (no GPU) — safe to run outside SLURM test:0.
set -uo pipefail
# Set HF_TOKEN in your environment before running (do NOT hardcode secrets):
#   export HF_TOKEN=hf_xxx
: "${HF_TOKEN:?set HF_TOKEN env var before running}"
CLI=/home/qyyang/repo/GLoRCQ/.venv/bin/huggingface-cli
DL=/mnt/Data/yqy/resource_dir/hf_dl
mkdir -p "$DL"
for m in qwen1.5-moe-a2.7b mixtral-8x7b qwen3-30b-a3b; do
  repo="Tsingyow/GLoRCQ-${m}-fair-grassmann-real"
  echo "=========== [$(date)] downloading $repo ==========="
  $CLI download "$repo" --local-dir "$DL/GLoRCQ-${m}-fair-grassmann-real" \
      --token "$HF_TOKEN" 2>&1 | tail -3
  echo "[$(date)] done $m: $(du -sh "$DL/GLoRCQ-${m}-fair-grassmann-real" 2>/dev/null | cut -f1)"
done
echo "=========== [$(date)] ALL DOWNLOADS DONE ==========="
