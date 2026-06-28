#!/bin/bash
# E10: TileQ framework + GLoRCQ cross-layer group sharing
#
# Key difference from TileQ 1D (E9):
#   - TileQ 1D shares U within ONE layer (60 experts/layer).
#   - E10 shares U across G=128 experts from DIFFERENT layers.
#   - Everything else is TileQ unchanged: activation scaling, rank-1 sketch, VQ quantization.
#
# Baseline comparisons:
#   E5 SOTA (GLoRCQ):     PPL=7.55, bits≈3.23
#   E9 (GLoRCQ+TileQ-1D): PPL=9.20 (using our framework)
#   TileQ-1D paper:        PPL=7.49 (no cross-layer share)
#   TileQ-2D paper:        PPL≈7.49 (with V-sharing, within-layer)
#   E10 target:            PPL<7.49 (cross-layer share should reduce U overhead)
#
# Cross-layer sharing reduces U bits from rank*in_d/expert to rank*in_d/(G*expert),
# which at G=128 frees up bit budget for larger rank or lower residual quantization error.

cd /home/qyyang/repo/GLoRCQ

OUTPUT=/home/qyyang/resource_dir/GLoRCQ_out/e10_tileq_glorcq
LOG=logs/e10_tileq_glorcq.log

mkdir -p logs "$OUTPUT"

echo "[$(date)] E10 TileQ+GLoRCQ cross-layer sharing started" | tee $LOG

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
    2>&1 | tee -a $LOG

echo "[$(date)] Quant done. Running PPL eval..." | tee -a $LOG

CUDA_VISIBLE_DEVICES=4 .venv/bin/python evaluate/eval_ppl.py \
    --model_path $OUTPUT \
    --device cuda:0 \
    --output_json logs/e10_ppl.json \
    2>&1 | tee -a $LOG

echo "[$(date)] E10 ALL DONE!" | tee -a $LOG
echo "[$(date)] Compare: TileQ-1D-paper=7.49 | E5(GLoRCQ)=7.55 | E9(our-fw+TileQ)=9.20" | tee -a $LOG
