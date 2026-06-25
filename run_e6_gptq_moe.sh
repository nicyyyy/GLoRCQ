#!/bin/bash
# E6: GPTQ for MoE experts (remove --use_turboquant)
# All other params identical to E5 SOTA (PPL=7.55, bits=3.2270)
# Run from test:0 (SLURM step cgroup)
set -uo pipefail
cd /home/qyyang/repo/GLoRCQ

GPU=4
DEVICE=cuda:0
E6_OUT=/home/qyyang/resource_dir/GLoRCQ_out/explore_e6_gptq_moe
LOG=logs/explore_e6_gptq_moe.log
mkdir -p logs

echo "[$(date '+%F %T')] E6 quantization started" | tee "$LOG"
echo "[$(date '+%F %T')] Config: GPTQ for all modules (no TurboQuant), E5 params otherwise" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path "$E6_OUT" \
    --qbit 2 --groupsize 128 --nsamples 128 \
    --rank 32 --rank_down 512 --rank_attn 512 --rank_cluster 32 \
    --n_iter 5 --n_lora_iter 3 \
    --G_moe 128 --G_attn 4 \
    --u_bits 8 --u_bits_attn 8 --sv_bits 8 \
    --w_clip --hessian_svd --recon_weight 0.7 \
    --search_act_alpha \
    >> "$LOG" 2>&1

echo "[$(date '+%F %T')] Quant done. Running PPL eval..." | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python evaluate/eval_ppl.py \
    --model_path "$E6_OUT" --device $DEVICE \
    > logs/eval_e6_gptq_moe.log 2>&1

E6_PPL=$(grep 'WikiText-2 PPL' logs/eval_e6_gptq_moe.log | awk '{print $NF}')
E6_BITS=$(grep 'Total average' "$LOG" | awk '{print $NF}' | tail -1)
echo "[$(date '+%F %T')] PPL=$E6_PPL  bits=$E6_BITS" | tee -a "$LOG"

echo "[$(date '+%F %T')] Running zero-shot eval (5 tasks)..." | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python evaluate/eval_zeroshot.py \
    --model_path "$E6_OUT" --device $DEVICE \
    --tasks arc_challenge,arc_easy,winogrande,hellaswag,piqa \
    --batch_size 1 \
    --output_json logs/zeroshot_e6_gptq_moe.json \
    >> logs/zeroshot_e6_run.log 2>&1

echo "[$(date '+%F %T')] E6 ALL DONE! PPL=$E6_PPL bits=$E6_BITS" | tee -a "$LOG"
echo "[$(date '+%F %T')] Compare with E5: PPL=7.55, bits=3.2270" | tee -a "$LOG"
