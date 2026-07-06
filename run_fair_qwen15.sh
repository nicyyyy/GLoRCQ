#!/bin/bash
# GLoRCQ fair-bit config for Qwen1.5-MoE-A2.7B
# Matches TileQ's Extra bits budget (+0.16) for apples-to-apples paper comparison.
#
# Result (2026-07-05, H200 quant + eval):
#   Extra bits = +0.1621 bits/param
#   WikiText-2 PPL = 7.17  (WIN 0.39 vs TileQ_s 7.56 @ 0.16 bits)
#   ZS 5-task avg = 65.59%
#   MMLU 5-shot = 57.27%
#
# Key differences vs earlier E11 SOTA (run_e11.sh, +0.22 bits, PPL 7.16):
#   - fix_rank 32 -> 20   (bit budget cut)
#   - +int8_lora +int8_lora_v +pool_kmeans   (compression + init)

cd "$(dirname "$0")"

OUTPUT=/home/qyyang/resource_dir/GLoRCQ_out/qwen15_fair
LOG=logs/qwen15_fair.log
mkdir -p logs "$OUTPUT"

echo "[$(date)] Qwen1.5-MoE fair-bit started" | tee $LOG

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} .venv/bin/python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path $OUTPUT \
    --qbit 2 \
    --fix_rank 20 \
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
