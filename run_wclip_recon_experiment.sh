#!/bin/bash
# Experiment: SOTA config + w_clip + recon_weight=0.7
#
# E1 (rank_down=512) is INFEASIBLE: 3.1931 bits (over 2.5 limit)
# This experiment adds the two historically-effective flags that were
# in the 7.99 PPL config but absent from SOTA (8.48):
#   --w_clip: MSE-based weight clipping during GPTQ
#   --recon_weight 0.7: cross-reconstruction distance in Grassmannian clustering
#
# Bits: same as SOTA (~2.4787) — no rank change
# Expected PPL: 8.3-8.4?
#
# Run from test:0 (SLURM step cgroup):
#   tmux send-keys -t test:0 "bash run_wclip_recon_experiment.sh" Enter

set -uo pipefail
cd /home/qyyang/repo/GLoRCQ

CHAIN_LOG=logs/wclip_recon_chain.log
GPU=4
DEVICE=cuda:0

wait_for_gpu() {
    while true; do
        USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=$GPU 2>/dev/null)
        if [ "$USED" -lt 2000 ]; then
            echo "[$(date '+%F %T')] GPU${GPU} free (${USED} MB)"
            break
        fi
        echo "[$(date '+%F %T')] GPU${GPU} busy (${USED} MB), waiting 30s..."
        sleep 30
    done
}

log() { echo "[$(date '+%F %T')] $*" | tee -a "$CHAIN_LOG"; }

mkdir -p logs
log "=== SOTA + w_clip + recon_weight=0.7 experiment started ==="
log "Config: rank=32, rank_attn=512, G_moe=128, G_attn=24, u8sv8, n_lora_iter=2, +w_clip +recon_weight=0.7"
log "Bits budget: ~2.4787 (same as SOTA, no rank change)"

# ====================================================================
# Exp 1: SOTA + w_clip + recon_weight=0.7
# ====================================================================
wait_for_gpu
log "=== Exp1: SOTA + w_clip + recon_weight=0.7 [bits~=2.4787] ==="
EXP1_OUT=/home/qyyang/resource_dir/GLoRCQ_out/sota_wclip_recon07
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path "$EXP1_OUT" \
    --qbit 2 --groupsize 128 --nsamples 128 \
    --rank 32 --rank_attn 512 \
    --n_iter 5 --n_lora_iter 2 \
    --G_moe 128 --G_attn 24 \
    --u_bits 8 --sv_bits 8 \
    --w_clip --recon_weight 0.7 \
    --hessian_svd \
    --use_turboquant --search_act_alpha \
    > logs/sota_wclip_recon07.log 2>&1
log "Exp1 quant done"

wait_for_gpu
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python evaluate/eval_ppl.py \
    --model_path "$EXP1_OUT" --device $DEVICE \
    > logs/eval_sota_wclip_recon07.log 2>&1
EXP1_PPL=$(grep 'WikiText-2 PPL' logs/eval_sota_wclip_recon07.log | awk '{print $NF}')
EXP1_BITS=$(grep 'Total average' logs/sota_wclip_recon07.log | awk '{print $NF}' | head -1)
log "Exp1 PPL=$EXP1_PPL  bits=$EXP1_BITS"

# ====================================================================
# Exp 2: SOTA + w_clip only (no recon_weight change)
# ====================================================================
wait_for_gpu
log "=== Exp2: SOTA + w_clip only [bits~=2.4787] ==="
EXP2_OUT=/home/qyyang/resource_dir/GLoRCQ_out/sota_wclip_only
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path "$EXP2_OUT" \
    --qbit 2 --groupsize 128 --nsamples 128 \
    --rank 32 --rank_attn 512 \
    --n_iter 5 --n_lora_iter 2 \
    --G_moe 128 --G_attn 24 \
    --u_bits 8 --sv_bits 8 \
    --w_clip \
    --hessian_svd \
    --use_turboquant --search_act_alpha \
    > logs/sota_wclip_only.log 2>&1
log "Exp2 quant done"

wait_for_gpu
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python evaluate/eval_ppl.py \
    --model_path "$EXP2_OUT" --device $DEVICE \
    > logs/eval_sota_wclip_only.log 2>&1
EXP2_PPL=$(grep 'WikiText-2 PPL' logs/eval_sota_wclip_only.log | awk '{print $NF}')
EXP2_BITS=$(grep 'Total average' logs/sota_wclip_only.log | awk '{print $NF}' | head -1)
log "Exp2 PPL=$EXP2_PPL  bits=$EXP2_BITS"

# ====================================================================
# Summary
# ====================================================================
log "=== All experiments done ==="
log "| Exp | PPL | Bits |"
log "| SOTA (reference) | 8.48 | 2.4787 |"
log "| Exp1 (w_clip+recon0.7) | ${EXP1_PPL:-?} | ${EXP1_BITS:-?} |"
log "| Exp2 (w_clip only) | ${EXP2_PPL:-?} | ${EXP2_BITS:-?} |"

# Zero-shot for best if improved
BEST_PPL=8.48
BEST_OUT=""
for pair in "EXP1_PPL:EXP1_OUT" "EXP2_PPL:EXP2_OUT"; do
    ppl_var="${pair%%:*}"; out_var="${pair##*:}"
    ppl="${!ppl_var:-9.99}"; out="${!out_var}"
    if python3 -c "exit(0 if float('${ppl}') < float('${BEST_PPL}') else 1)" 2>/dev/null; then
        BEST_PPL="$ppl"; BEST_OUT="$out"
    fi
done

if [ -n "$BEST_OUT" ] && [ -d "$BEST_OUT" ]; then
    wait_for_gpu
    BEST_NAME=$(basename "$BEST_OUT")
    log "=== Best improved to PPL=$BEST_PPL — running zero-shot ==="
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python evaluate/eval_zeroshot.py \
        --model_path "$BEST_OUT" --device $DEVICE \
        --tasks arc_challenge,arc_easy,winogrande,hellaswag,piqa \
        --batch_size 1 --output_json logs/eval_${BEST_NAME}_zeroshot.json \
        > logs/eval_${BEST_NAME}_zeroshot.log 2>&1
    log "Zero-shot done."
else
    log "No improvement over SOTA (8.48). Zero-shot skipped."
fi

log "=== wclip+recon experiment chain completed ==="
