#!/bin/bash
# Real-quant decode-speed benchmark (post shim-graph fix) for mixtral + qwen3.
# qwen1.5 already validated separately (Std 8.4 / Graph 19.8). GPU4, tmux test:0.
set -uo pipefail
PY=/home/qyyang/repo/GLoRCQ/.venv/bin/python
ES=/home/qyyang/repo/GLoRCQ/inference/eval_speed.py
DL=/mnt/Data/yqy/resource_dir/hf_dl
LOG=/home/qyyang/repo/GLoRCQ/logs/cluster_validity
export CUDA_VISIBLE_DEVICES=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
for m in mixtral-8x7b qwen3-30b-a3b; do
  echo "=========== [$(date)] SPEED $m ==========="
  $PY "$ES" --model_path "$DL/GLoRCQ-${m}-fair-grassmann-real" --real_quant \
      --batch_size 1 --prompt_len 128 --gen_len 128 --device cuda:0 \
      > "$LOG/speed_${m}.log" 2>&1
  echo "[$(date)] $m -> $(grep -E 'Standard:|Graph:|Speedup:' "$LOG/speed_${m}.log" | tr '\n' ' ')"
done
echo "=========== [$(date)] SPEED 2 DONE ==========="
