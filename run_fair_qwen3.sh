#!/bin/bash
# GLoRCQ fair-bit config for Qwen3-30B-A3B (128 experts × top-8, 48 layers)
# Matches TileQ's Extra bits budget (+0.16) for apples-to-apples paper comparison.
#
# Result (2026-07-05, H200 quant + eval):
#   Extra bits = +0.1647 bits/param
#   WikiText-2 PPL = 9.42  (WIN 0.68 vs TileQ_v 10.1 @ 0.16 bits)
#   ZS 5-task avg = 63.00%
#   MMLU 5-shot = 65.52%
#
# Key change vs earlier E11 SOTA (run_e11_qwen3.sh, rank=32, +0.27 bits, PPL 9.51):
#   - fix_rank 32 -> 16   (Qwen3's per-expert weights are smaller so rank halves cleanly)

cd "$(dirname "$0")"

OUTPUT=/home/qyyang/resource_dir/GLoRCQ_out/qwen3_fair
LOG=logs/qwen3_fair.log
mkdir -p logs "$OUTPUT"

echo "[$(date)] Qwen3-30B-A3B fair-bit started" | tee $LOG

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python run_quantize.py \
    --model_path /home/qyyang/resource_dir/hf_cache/Qwen3-30B-A3B \
    --output_path $OUTPUT \
    --qbit 2 \
    --fix_rank 16 \
    --G 128 \
    --group_size 128 \
    --lora_bit 16 \
    --lora_iter 8 \
    --ha_bsize 256 \
    --id_bsize 256 \
    --attn_bits 4 \
    --int8_lora \
    --int8_lora_v \
    --pool_kmeans \
    2>&1 | tee -a $LOG

echo "[$(date)] Quant done." | tee -a $LOG
