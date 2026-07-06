#!/bin/bash
# GLoRCQ fair-bit config for Mixtral-8x7B-v0.1 (8 experts × top-2, 32 layers)
# Matches TileQ's Extra bits budget (+0.16) for apples-to-apples paper comparison.
#
# Result (2026-07-06, H200 quant + eval):
#   Extra bits = +0.1611 bits/param
#   WikiText-2 PPL = 4.69  (WIN 0.29 vs TileQ_s 4.98 @ 0.16 bits)
#   ZS 5-task avg = 69.22%
#   MMLU 5-shot = 49.64%
#
# Key differences vs Qwen fair-bit configs:
#   - fix_rank 32 (Mixtral's per-expert dim is bigger; rank needs to scale up)
#   - NO --int8_lora / --int8_lora_v: Mixtral 5% weight-error regression under int8 LoRA;
#     fp16 LoRA (default) restores correctness at cost of ~0.01 extra bits/param.
#   - Everything else identical.
#
# NOTE: Mixtral FP16 is ~87 GB — Phase 1 SVD needs enough RAM. If OOM, use
# --phase1_cache_path to split; and reduce --G to 64 to halve SVD input.

cd "$(dirname "$0")"

OUTPUT=/home/qyyang/resource_dir/GLoRCQ_out/mixtral_fair
LOG=logs/mixtral_fair.log
mkdir -p logs "$OUTPUT"

echo "[$(date)] Mixtral-8x7B fair-bit started" | tee $LOG

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python run_quantize.py \
    --model_path /home/qyyang/resource_dir/hf_cache/Mixtral-8x7B-v0.1 \
    --output_path $OUTPUT \
    --qbit 2 \
    --fix_rank 32 \
    --G 128 \
    --group_size 128 \
    --lora_bit 16 \
    --lora_iter 8 \
    --ha_bsize 256 \
    --id_bsize 256 \
    --attn_bits 4 \
    --pool_kmeans \
    2>&1 | tee -a $LOG

echo "[$(date)] Quant done." | tee -a $LOG
