#!/bin/bash
# Run E11: TileQ + GLoRCQ cross-layer sharing + 4-bit attention GPTQ
#
# Result: PPL=6.90 @ ~3.47 effective bits on Qwen1.5-MoE-A2.7B
#   - MoE routing experts: 2-bit VQ with cross-layer shared U (rank=32, G=128)
#   - Attention layers: 4-bit scalar GPTQ (no LoRA)
#   - Shared expert + embeddings: FP16
#
# Output is saved to resource_dir to avoid home quota limits.

cd "$(dirname "$0")"

OUTPUT=/home/qyyang/resource_dir/GLoRCQ_out/e11_tileq_glorcq_attn4bit
LOG=logs/e11_tileq_glorcq_attn4bit.log

mkdir -p logs "$OUTPUT"

echo "[$(date)] E11 started" | tee $LOG

CUDA_VISIBLE_DEVICES=4 .venv/bin/python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
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
    2>&1 | tee -a $LOG

echo "[$(date)] Quant done. Running PPL eval..." | tee -a $LOG

CUDA_VISIBLE_DEVICES=4 .venv/bin/python evaluate/eval_ppl.py \
    --model_path $OUTPUT \
    --device cuda:0 \
    --output_json logs/e11_ppl.json \
    2>&1 | tee -a $LOG

echo "[$(date)] E11 ALL DONE!" | tee -a $LOG
