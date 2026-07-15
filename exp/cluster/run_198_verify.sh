#!/bin/bash
# Task #198 Phase 2/3 verification (serial, GPU4 via tmux test:0):
#   1. GPTQ int8-store parity probe on Qwen1.5 (Edit A correctness)
#   2. eval_speed Qwen1.5  (regression: Standard + Graph must still work)
#   3. eval_speed Mixtral  (Standard tok/s + graph auto-disable + memory)
set -uo pipefail
PY=/home/qyyang/repo/GLoRCQ/.venv/bin/python
ES=/home/qyyang/repo/GLoRCQ/inference/eval_speed.py
DL=/mnt/Data/yqy/resource_dir/hf_dl
LOG=/home/qyyang/repo/GLoRCQ/logs/cluster_validity
cd /home/qyyang/repo/GLoRCQ
export CUDA_VISIBLE_DEVICES=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========== [$(date)] 1/3 GPTQ parity probe (qwen1.5) ==========="
$PY exp/cluster/gptq_parity_probe.py "$DL/GLoRCQ-qwen1.5-moe-a2.7b-fair-grassmann-real" \
    > "$LOG/parity_198_qwen15.log" 2>&1
echo "[$(date)] parity -> $(grep -E 'RESULT' "$LOG/parity_198_qwen15.log")"

echo "=========== [$(date)] 2/3 SPEED qwen1.5 (regression) ==========="
$PY "$ES" --model_path "$DL/GLoRCQ-qwen1.5-moe-a2.7b-fair-grassmann-real" --real_quant \
    --batch_size 1 --prompt_len 128 --gen_len 128 --max_seq_len 384 --device cuda:0 \
    > "$LOG/speed_198_qwen15.log" 2>&1
echo "[$(date)] qwen1.5 -> $(grep -E 'Standard:|Graph:|Speedup:' "$LOG/speed_198_qwen15.log" | tr '\n' ' ')"

echo "=========== [$(date)] 3/3 SPEED mixtral (edits active) ==========="
$PY "$ES" --model_path "$DL/GLoRCQ-mixtral-8x7b-fair-grassmann-real" --real_quant \
    --batch_size 1 --prompt_len 128 --gen_len 128 --max_seq_len 384 --device cuda:0 \
    > "$LOG/speed_198_mixtral.log" 2>&1
echo "[$(date)] mixtral -> $(grep -E 'Standard:|Graph:|Speedup:|Mixtral detected' "$LOG/speed_198_mixtral.log" | tr '\n' ' ')"
echo "=========== [$(date)] TASK198 VERIFY DONE ==========="
