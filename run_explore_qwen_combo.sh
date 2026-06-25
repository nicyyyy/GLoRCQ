#!/bin/bash
# GLoRCQ Explore combos: E4 (rank128+iter3) and E5 (u8+iter3)
# Continuation from run_explore_qwen.sh (E1-E3 already done)
# Run from test:0 (SLURM step cgroup)

set -uo pipefail
cd /home/qyyang/repo/GLoRCQ

CHAIN_LOG=logs/explore_qwen_chain.log
RESULTS_LOG=logs/explore_qwen_results.md
GPU=4
DEVICE=cuda:0

wait_for_gpu() {
    while true; do
        USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=$GPU 2>/dev/null)
        if [ "$USED" -lt 2000 ]; then
            echo "[$(date '+%F %T')] GPU${GPU} free (${USED} MB)" | tee -a "$CHAIN_LOG"
            break
        fi
        echo "[$(date '+%F %T')] GPU${GPU} busy (${USED} MB), waiting 60s..." | tee -a "$CHAIN_LOG"
        sleep 60
    done
}

log() { echo "[$(date '+%F %T')] $*" | tee -a "$CHAIN_LOG"; }

log "=== explore combo chain started (E4+E5) ==="

# ============================================================
# E4: rank=128 gate/up + n_lora_iter=3 (E1+E3 combo)
# Expected bits: ~3.4513 (same as E1, iter doesn't change bits)
# ============================================================
wait_for_gpu
log "=== E4: rank=128 gate/up + n_lora_iter=3 (E1+E3 combo) ==="
E4_OUT=/home/qyyang/resource_dir/GLoRCQ_out/explore_e4_rank128_iter3
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path "$E4_OUT" \
    --qbit 2 --groupsize 128 --nsamples 128 \
    --rank 128 --rank_down 512 --rank_attn 512 --rank_cluster 32 \
    --n_iter 5 --n_lora_iter 3 \
    --G_moe 128 --G_attn 4 \
    --u_bits 4 --u_bits_attn 8 --sv_bits 8 \
    --w_clip --hessian_svd --recon_weight 0.7 \
    --use_turboquant --search_act_alpha \
    > logs/explore_e4_rank128_iter3.log 2>&1
log "E4 quant done"

wait_for_gpu
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python evaluate/eval_ppl.py \
    --model_path "$E4_OUT" --device $DEVICE \
    > logs/eval_explore_e4.log 2>&1
E4_PPL=$(grep 'WikiText-2 PPL' logs/eval_explore_e4.log | awk '{print $NF}')
E4_BITS=$(grep 'Total average' logs/explore_e4_rank128_iter3.log | awk '{print $NF}' | tail -1)
log "E4 RESULT: rank128+iter3 → PPL=${E4_PPL}  bits=${E4_BITS}"
echo "| E4 rank=128+iter3 | ${E4_PPL:-?} | ${E4_BITS:-?} | rank=128,rdown=512,rattn=512,u4sv8,iter3 |" >> "$RESULTS_LOG"

# ============================================================
# E5: u_bits=8 MoE + n_lora_iter=3 (E2+E3 combo)
# Expected bits: ~3.2270 (same as E2)
# ============================================================
wait_for_gpu
log "=== E5: u_bits=8 MoE + n_lora_iter=3 (E2+E3 combo) ==="
E5_OUT=/home/qyyang/resource_dir/GLoRCQ_out/explore_e5_ubits8_iter3
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path "$E5_OUT" \
    --qbit 2 --groupsize 128 --nsamples 128 \
    --rank 32 --rank_down 512 --rank_attn 512 --rank_cluster 32 \
    --n_iter 5 --n_lora_iter 3 \
    --G_moe 128 --G_attn 4 \
    --u_bits 8 --u_bits_attn 8 --sv_bits 8 \
    --w_clip --hessian_svd --recon_weight 0.7 \
    --use_turboquant --search_act_alpha \
    > logs/explore_e5_ubits8_iter3.log 2>&1
log "E5 quant done"

wait_for_gpu
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python evaluate/eval_ppl.py \
    --model_path "$E5_OUT" --device $DEVICE \
    > logs/eval_explore_e5.log 2>&1
E5_PPL=$(grep 'WikiText-2 PPL' logs/eval_explore_e5.log | awk '{print $NF}')
E5_BITS=$(grep 'Total average' logs/explore_e5_ubits8_iter3.log | awk '{print $NF}' | tail -1)
log "E5 RESULT: ubits8+iter3 → PPL=${E5_PPL}  bits=${E5_BITS}"
echo "| E5 u8+iter3 | ${E5_PPL:-?} | ${E5_BITS:-?} | rank=32,rdown=512,rattn=512,u8sv8,iter3 |" >> "$RESULTS_LOG"

log "=== E4+E5 combo experiments done ==="
cat "$RESULTS_LOG"
