#!/bin/bash
set -uo pipefail
PY=/home/qyyang/repo/GLoRCQ/.venv/bin/python
DL=/mnt/Data/yqy/resource_dir/hf_dl
LOG=/home/qyyang/repo/GLoRCQ/logs/cluster_validity
export CUDA_VISIBLE_DEVICES=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
for m in qwen1.5-moe-a2.7b mixtral-8x7b qwen3-30b-a3b; do
  echo "=========== [$(date)] INFER SANITY $m ==========="
  $PY /home/qyyang/repo/GLoRCQ/exp/cluster/infer_sanity.py \
      "$DL/GLoRCQ-${m}-fair-grassmann-real" > "$LOG/infer_${m}.log" 2>&1
  echo "[$(date)] $m -> $(grep -E '\[RESULT\]|\[logits\]' "$LOG/infer_${m}.log" | tr '\n' ' ')"
done
echo "=========== [$(date)] INFER SANITY ALL DONE ==========="
