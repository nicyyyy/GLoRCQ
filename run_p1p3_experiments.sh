#!/bin/bash
# P1+P3 Algorithm Improvement Experiments
# P1: Batched Hessian-weighted SVD fix (joint_optim.py)
# P3: Kurtosis adaptive rank (cross_layer_share.py)
#
# Base: SOTA config (rank=32, rattn=512, G_attn=24, u8sv8, n_lora_iter=2)
# Bits: 2.4787 (same as SOTA — P1 is quality fix, P3 is budget-neutral) ✅
#
# Run from test:0 (SLURM step cgroup) to avoid 5-min kill:
#   tmux send-keys -t test:0 "bash run_p1p3_experiments.sh" Enter
#
# Experiment plan:
#   P1-only : SOTA config + hessian fix in batched rSVD (no adaptive_rank)
#   P1+P3   : SOTA config + hessian fix + --adaptive_rank kurtosis rank allocation
#
# Expected:
#   P1-only:  PPL ~8.3-8.4 (better init for MoE LoRA; baseline 8.48)
#   P1+P3:    PPL ~8.2-8.3 (P1 + higher rank for harder experts)

set -uo pipefail
cd /home/qyyang/repo/GLoRCQ

RESULTS_LOG=logs/p1p3_results.md
CHAIN_LOG=logs/p1p3_chain.log
GPU=4
DEVICE=cuda:0

wait_for_gpu() {
    echo "[$(date '+%F %T')] Waiting for GPU${GPU} to be free..."
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

append_result() {
    local name="$1" ppl="$2" bits="$3" config="$4"
    echo "| $name | $ppl | $bits | $config |" >> "$RESULTS_LOG"
    log "RESULT: $name → PPL=$ppl  bits=$bits"
}

mkdir -p logs
cat > "$RESULTS_LOG" << 'HEADER'
# P1+P3 Algorithm Improvement Results

| Experiment | PPL | Total Bits | Key Config |
|------------|-----|-----------|-----------|
| FP16 baseline | 6.79 | 16.0 | — |
| TileQ_v (target) | 7.35 | 2.16 | GPTQ + 2D-tiling |
| GLoRCQ SOTA | 8.48 | 2.4787 | rattn=512,Gattn=24,u8sv8,rdown=32,iter2 |

HEADER
log "=== P1+P3 improvement experiments started ==="

# ====================================================================
# P1-only: SOTA config + batched Hessian fix (hessian_svd already default)
# Bits: 2.4787 (unchanged) ✅
# ====================================================================
wait_for_gpu
log "=== P1-only: SOTA config + batched H fix [bits=2.4787] ==="
P1_OUT=/home/qyyang/resource_dir/GLoRCQ_out/p1_batched_hessian
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path "$P1_OUT" \
    --qbit 2 --groupsize 128 --nsamples 128 \
    --rank 32 --rank_attn 512 \
    --n_iter 5 --n_lora_iter 2 \
    --G_moe 128 --G_attn 24 \
    --u_bits 8 --sv_bits 8 \
    --hessian_svd \
    --use_turboquant --search_act_alpha \
    > logs/p1_batched_hessian.log 2>&1
log "P1-only quant done"

wait_for_gpu
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python evaluate/eval_ppl.py \
    --model_path "$P1_OUT" --device $DEVICE \
    > logs/eval_p1.log 2>&1
P1_PPL=$(grep 'WikiText-2 PPL' logs/eval_p1.log | awk '{print $NF}')
P1_BITS=$(grep 'Total average' logs/p1_batched_hessian.log | awk '{print $NF}' | head -1)
append_result "P1-only (batched H fix)" \
    "${P1_PPL:-?}" "${P1_BITS:-?}" \
    "rank=32,rattn=512,Gattn=24,u8sv8,n_lora_iter=2,hessian_svd"
log "P1-only PPL=$P1_PPL  bits=$P1_BITS"

# ====================================================================
# P1+P3: SOTA config + batched H fix + kurtosis adaptive rank
# Bits: ~2.4787 (P3 is budget-neutral: mean cluster rank = base rank) ✅
# ====================================================================
wait_for_gpu
log "=== P1+P3: batched H fix + adaptive_rank [bits≈2.4787] ==="
P1P3_OUT=/home/qyyang/resource_dir/GLoRCQ_out/p1p3_batched_hessian_adaptive
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path "$P1P3_OUT" \
    --qbit 2 --groupsize 128 --nsamples 128 \
    --rank 32 --rank_attn 512 \
    --n_iter 5 --n_lora_iter 2 \
    --G_moe 128 --G_attn 24 \
    --u_bits 8 --sv_bits 8 \
    --hessian_svd --adaptive_rank \
    --use_turboquant --search_act_alpha \
    > logs/p1p3_batched_hessian_adaptive.log 2>&1
log "P1+P3 quant done"

wait_for_gpu
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python evaluate/eval_ppl.py \
    --model_path "$P1P3_OUT" --device $DEVICE \
    > logs/eval_p1p3.log 2>&1
P1P3_PPL=$(grep 'WikiText-2 PPL' logs/eval_p1p3.log | awk '{print $NF}')
P1P3_BITS=$(grep 'Total average' logs/p1p3_batched_hessian_adaptive.log | awk '{print $NF}' | head -1)
append_result "P1+P3 (batched H + adaptive_rank)" \
    "${P1P3_PPL:-?}" "${P1P3_BITS:-?}" \
    "rank=32,rattn=512,Gattn=24,u8sv8,n_lora_iter=2,hessian_svd,adaptive_rank"
log "P1+P3 PPL=$P1P3_PPL  bits=$P1P3_BITS"

# ====================================================================
# Summary + zero-shot for best model
# ====================================================================
log "=== All PPL experiments done. Summary in $RESULTS_LOG ==="

BEST_PPL="9.99"
BEST_OUT=""
BEST_NAME=""
for exp in P1 P1P3; do
    eval "ppl=\${${exp}_PPL:-9.99}"
    eval "out=\${${exp}_OUT}"
    if [ -n "$ppl" ] && python3 -c "exit(0 if float('$ppl') < float('$BEST_PPL') else 1)" 2>/dev/null; then
        BEST_PPL="$ppl"
        BEST_OUT="$out"
        BEST_NAME="$exp"
    fi
done
log "Best: $BEST_NAME with PPL=$BEST_PPL"

if [ -n "$BEST_OUT" ] && [ -d "$BEST_OUT" ]; then
    wait_for_gpu
    log "=== Zero-shot eval on $BEST_NAME ==="
    BEST_ZS_LOG=logs/eval_${BEST_NAME}_zeroshot.log
    BEST_ZS_JSON=logs/eval_${BEST_NAME}_zeroshot.json
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python evaluate/eval_zeroshot.py \
        --model_path "$BEST_OUT" \
        --device $DEVICE \
        --tasks arc_challenge,arc_easy,winogrande,hellaswag,piqa \
        --batch_size 1 \
        --output_json "$BEST_ZS_JSON" \
        > "$BEST_ZS_LOG" 2>&1
    log "Zero-shot done."
    grep -E "arc_challenge|arc_easy|winogrande|hellaswag|piqa|Avg" "$BEST_ZS_LOG" | tail -10 | tee -a "$CHAIN_LOG"
fi

log "=== P1+P3 experiments completed ==="
cat "$RESULTS_LOG"
