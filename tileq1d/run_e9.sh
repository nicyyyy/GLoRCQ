#!/bin/bash
# E9: TileQ 1D + cross-layer share pipeline
#
# Key difference from E5 SOTA:
#   - calibration_only=True: Stage 0 collects Hessian/act_scale, skips MoE quantization
#   - cluster_on_original=True: cluster on W_orig (TileQ-style)
#   - fit_on_original=True: SVD(W_orig) → lora ≈ W_orig
#   - Stage 5: Q(R_k) where R_k = W_orig - lora (genuine low-rank residual)
#
# E5 SOTA: PPL=7.55, bits=3.2270
# E7 (cluster_on_original): PPL=7.56 — clustering not the issue
# E8 (requant_moe wrong approach): PPL=7.59 — fitted lora to residuals, not W_orig
# E9 (this): TileQ-correct approach — fit lora to W_orig first, then quantize residual
#
# TileQ paper: PPL=7.49, ~0.31 extra bits (their 1D baseline before V-sharing trick)
# If E9 matches ~7.49, the gap was purely quantization order.

cd /home/qyyang/repo/GLoRCQ

OUTPUT=/home/qyyang/resource_dir/GLoRCQ_out/e9_tileq1d
LOG=logs/e9_tileq1d.log

mkdir -p logs

echo "[$(date)] E9 TileQ-1D pipeline started" | tee $LOG

CUDA_VISIBLE_DEVICES=4 .venv/bin/python tileq1d/run_tileq1d.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path $OUTPUT \
    --qbit 2 --groupsize 128 --nsamples 128 \
    --rank 32 --rank_down 512 --rank_attn 512 --rank_cluster 32 \
    --n_iter 5 --n_lora_iter 1 \
    --G_moe 128 --G_attn 4 \
    --u_bits 8 --u_bits_attn 8 --sv_bits 8 \
    --w_clip --hessian_svd --recon_weight 0.7 \
    --search_act_alpha \
    2>&1 | tee -a $LOG

echo "[$(date)] Quant done. Running PPL eval..." | tee -a $LOG
CUDA_VISIBLE_DEVICES=4 .venv/bin/python evaluate/eval_ppl.py \
    --model_path $OUTPUT \
    --device cuda:0 \
    --output_json logs/e9_ppl.json \
    2>&1 | tee -a $LOG

echo "[$(date)] E9 ALL DONE!" | tee -a $LOG
echo "[$(date)] Compare: E5=7.55 | E7=7.56 | E8=7.59 | TileQ-1D-paper=7.49" | tee -a $LOG
