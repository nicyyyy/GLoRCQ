#!/bin/bash
# E11 VQ4 on Qwen3-30B-A3B (128 experts × top-8, 48 layers, moe_intermediate=768)
#
# Same recipe as Mixtral: TileQ + GLoRCQ cross-layer sharing + VQ4 + attn 4-bit GPTQ.
# Output → resource_dir (avoid home quota).

cd "$(dirname "$0")"

OUTPUT=/home/qyyang/resource_dir/GLoRCQ_out/qwen3_e11_vq4
LOG=logs/e11_qwen3.log
mkdir -p logs "$OUTPUT"

echo "[$(date)] E11 Qwen3-30B-A3B started" | tee $LOG

CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python run_quantize.py \
    --model_path /home/qyyang/resource_dir/hf_cache/Qwen3-30B-A3B \
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
    --int8_lora \
    --int8_lora_v \
    --pool_kmeans \
    2>&1 | tee -a $LOG

echo "[$(date)] Quant done." | tee -a $LOG
