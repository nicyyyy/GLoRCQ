#!/bin/bash
# E7: TileQ-style clustering (cluster on original weights, not residuals)
# vs E5 SOTA (cluster on Hessian-weighted residuals)
#
# Hypothesis: clustering on activation/Hessian-scaled original FP16 weights
# finds more homogeneous groups → lower ‖E_cat‖ → better PPL at same rank.
#
# E5 SOTA config (PPL=7.55, bits=3.2270):
#   rank=32, rank_down=512, rank_attn=512, rank_cluster=32
#   n_iter=5, n_lora_iter=3, G_moe=128, G_attn=4
#   u_bits=8, u_bits_attn=8, sv_bits=8
#   w_clip, hessian_svd, recon_weight=0.7, turboquant, search_act_alpha
#
# E7 = E5 + --tileq_cluster  (only change: clustering input)

cd /home/qyyang/repo/GLoRCQ

OUTPUT=/home/qyyang/resource_dir/GLoRCQ_out/e7_tileq_cluster
LOG=logs/e7_tileq_cluster.log

echo "[$(date)] E7 TileQ-cluster started" | tee $LOG

CUDA_VISIBLE_DEVICES=4 .venv/bin/python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path $OUTPUT \
    --qbit 2 --groupsize 128 --nsamples 128 \
    --rank 32 --rank_down 512 --rank_attn 512 --rank_cluster 32 \
    --n_iter 5 --n_lora_iter 3 \
    --G_moe 128 --G_attn 4 \
    --u_bits 8 --u_bits_attn 8 --sv_bits 8 \
    --w_clip --hessian_svd --recon_weight 0.7 \
    --use_turboquant --search_act_alpha \
    --tileq_cluster \
    2>&1 | tee -a $LOG

echo "[$(date)] Quant done. Running PPL eval..." | tee -a $LOG
CUDA_VISIBLE_DEVICES=4 .venv/bin/python evaluate/eval_ppl.py \
    --model_path $OUTPUT \
    --device cuda:0 \
    --output_json logs/e7_ppl.json \
    2>&1 | tee -a $LOG

echo "[$(date)] E7 ALL DONE!" | tee -a $LOG
echo "[$(date)] Compare with E5: PPL=7.55, bits=3.2270" | tee -a $LOG
