#!/bin/bash
# E11: E10 + 4-bit scalar GPTQ for attention layers
#
# E10 had attention in FP16 → effective bits=3.82, PPL=6.82.
# E11 quantizes attention to 4-bit → effective bits≈3.47.
# Reuses E10's Phase 1 cache (no feature re-collection needed).

cd /home/qyyang/repo/GLoRCQ

OUTPUT=/home/qyyang/resource_dir/GLoRCQ_out/e11_tileq_glorcq_attn4bit
LOG=logs/e11_tileq_glorcq_attn4bit.log

mkdir -p logs "$OUTPUT"

echo "[$(date)] E11: E10 + attn 4-bit GPTQ started" | tee $LOG

CUDA_VISIBLE_DEVICES=4 .venv/bin/python tileq1d/run_tileq_glorcq.py \
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
    --phase1_cache_path /home/qyyang/resource_dir/GLoRCQ_out/e10_tileq_glorcq_phase1_cache.pt \
    2>&1 | tee -a $LOG

echo "[$(date)] Quant done. Running PPL eval..." | tee -a $LOG

CUDA_VISIBLE_DEVICES=4 .venv/bin/python evaluate/eval_ppl.py \
    --model_path $OUTPUT \
    --device cuda:0 \
    --output_json logs/e11_ppl.json \
    2>&1 | tee -a $LOG

echo "[$(date)] E11 ALL DONE!" | tee -a $LOG
echo "Compare: E10(attn-FP16)=6.82@3.82bits | E11(attn-4bit)=?@~3.47bits | E5=7.55@3.23bits" | tee -a $LOG
