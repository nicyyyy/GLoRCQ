#!/bin/bash
# P3 corrected approaches experiment
#
# SOTA baseline: PPL=8.48, bits=2.4787
# SOTA backed up to: loftq5_niter5_sv8_SOTA_backup
#
# Three corrections vs original P3:
#   1. P3v2 (--adaptive_rank_v2): kurtosis of W (not E=W-Q), exponential formula
#      2^(kurtosis_W + k_calibrated), cap [16, 64], budget-neutral k calibration
#   2. P3h (--adaptive_rank_h): H_eq_diag sum per cluster (activation importance
#      proxy), proportional allocation, cap [16, 64], budget-neutral by construction
#
# Both approaches are budget-neutral (mean cluster rank ≈ 32) → bits ≈ 2.4787 ✓
#
# Run from test:0:
#   tmux send-keys -t test:0 "bash run_p3v2_experiment.sh" Enter

set -uo pipefail
cd /home/qyyang/repo/GLoRCQ

CHAIN_LOG=logs/p3v2_chain.log
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
log "=== P3v2 corrected adaptive rank experiment started ==="
log "SOTA backed up at: loftq5_niter5_sv8_SOTA_backup (PPL=8.48, 2.4787 bits)"

# ====================================================================
# Exp1: P3v2 — kurtosis of W + exponential formula (MiLo-style)
# ====================================================================
wait_for_gpu
log "=== Exp1: P3v2 (W-kurtosis, exponential, calibrated k) [bits≈2.4787] ==="
EXP1_OUT=/home/qyyang/resource_dir/GLoRCQ_out/p3v2_wkurt_exp
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path "$EXP1_OUT" \
    --qbit 2 --groupsize 128 --nsamples 128 \
    --rank 32 --rank_attn 512 \
    --n_iter 5 --n_lora_iter 2 \
    --G_moe 128 --G_attn 24 \
    --u_bits 8 --sv_bits 8 \
    --adaptive_rank_v2 \
    --hessian_svd \
    --use_turboquant --search_act_alpha \
    > logs/p3v2_wkurt_exp.log 2>&1
log "Exp1 quant done"

wait_for_gpu
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python evaluate/eval_ppl.py \
    --model_path "$EXP1_OUT" --device $DEVICE \
    > logs/eval_p3v2_wkurt.log 2>&1
EXP1_PPL=$(grep 'WikiText-2 PPL' logs/eval_p3v2_wkurt.log | awk '{print $NF}')
EXP1_BITS=$(grep 'Total average' logs/p3v2_wkurt_exp.log | awk '{print $NF}' | head -1)
log "Exp1 PPL=$EXP1_PPL  bits=$EXP1_BITS"

# ====================================================================
# Exp2: P3h — H_eq_diag sum (activation importance), proportional
# ====================================================================
wait_for_gpu
log "=== Exp2: P3h (H-norm activation importance, proportional) [bits≈2.4787] ==="
EXP2_OUT=/home/qyyang/resource_dir/GLoRCQ_out/p3h_hnorm
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path "$EXP2_OUT" \
    --qbit 2 --groupsize 128 --nsamples 128 \
    --rank 32 --rank_attn 512 \
    --n_iter 5 --n_lora_iter 2 \
    --G_moe 128 --G_attn 24 \
    --u_bits 8 --sv_bits 8 \
    --adaptive_rank_h \
    --hessian_svd \
    --use_turboquant --search_act_alpha \
    > logs/p3h_hnorm.log 2>&1
log "Exp2 quant done"

wait_for_gpu
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python evaluate/eval_ppl.py \
    --model_path "$EXP2_OUT" --device $DEVICE \
    > logs/eval_p3h_hnorm.log 2>&1
EXP2_PPL=$(grep 'WikiText-2 PPL' logs/eval_p3h_hnorm.log | awk '{print $NF}')
EXP2_BITS=$(grep 'Total average' logs/p3h_hnorm.log | awk '{print $NF}' | head -1)
log "Exp2 PPL=$EXP2_PPL  bits=$EXP2_BITS"

# ====================================================================
# Summary
# ====================================================================
log "=== All experiments done ==="
log "| Experiment | PPL | Bits |"
log "| SOTA (baseline) | 8.48 | 2.4787 |"
log "| Exp1 P3v2 (W-kurtosis exp) | ${EXP1_PPL:-?} | ${EXP1_BITS:-?} |"
log "| Exp2 P3h (H-norm prop) | ${EXP2_PPL:-?} | ${EXP2_BITS:-?} |"

# Zero-shot for any improvement
for pair in "${EXP1_PPL:-9.99}:${EXP1_OUT}" "${EXP2_PPL:-9.99}:${EXP2_OUT}"; do
    ppl="${pair%%:*}"; out="${pair##*:}"
    if python3 -c "exit(0 if float('${ppl}') < 8.48 else 1)" 2>/dev/null; then
        wait_for_gpu
        BNAME=$(basename "$out")
        log "=== $BNAME improved to PPL=$ppl — running zero-shot ==="
        CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python evaluate/eval_zeroshot.py \
            --model_path "$out" --device $DEVICE \
            --tasks arc_challenge,arc_easy,winogrande,hellaswag,piqa \
            --batch_size 1 --output_json logs/eval_${BNAME}_zeroshot.json \
            > logs/eval_${BNAME}_zeroshot.log 2>&1
        log "Zero-shot done."
    fi
done

log "=== P3v2 experiment chain completed ==="
