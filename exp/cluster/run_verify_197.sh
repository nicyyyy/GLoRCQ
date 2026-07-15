#!/bin/bash
# Task #197 verification: real-quant decode speed for all 3 models after the
# _vq4_cat_cache leak fix. Mixtral must now COMPLETE (previously OOM'd at 77GB);
# Qwen1.5 + Qwen3 are regressions. GPU4, tmux test:0.
set -uo pipefail
PY=/home/qyyang/repo/GLoRCQ/.venv/bin/python
ES=/home/qyyang/repo/GLoRCQ/inference/eval_speed.py
DL=/mnt/Data/yqy/resource_dir/hf_dl
LOG=/home/qyyang/repo/GLoRCQ/logs/cluster_validity
export CUDA_VISIBLE_DEVICES=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for m in qwen1.5-moe-a2.7b mixtral-8x7b qwen3-30b-a3b; do
  echo "=========== [$(date)] SPEED $m ==========="
  $PY "$ES" --model_path "$DL/GLoRCQ-${m}-fair-grassmann-real" --real_quant \
      --batch_size 1 --prompt_len 128 --gen_len 128 --max_seq_len 384 --device cuda:0 \
      > "$LOG/verify197_${m}.log" 2>&1
  echo "[$(date)] $m -> $(grep -E 'Standard:|Graph:|Speedup:|OutOfMemory' "$LOG/verify197_${m}.log" | tr '\n' ' ')"
done
echo "=========== [$(date)] VERIFY197 DONE ==========="
