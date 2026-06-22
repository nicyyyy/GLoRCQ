#!/bin/bash
# GLoRCQ Qwen1.5-MoE PPL Optimization Chain (redesigned v2)
# Goal: push PPL from 8.48 (SOTA) while keeping total bits ≤ 2.5
# Run from test:0 (SLURM step cgroup) to avoid 5-min kill
#
# Bits budget: extra overhead ≤ 0.5 bits/param (total ≤ 2.5 bits)
# Reference SOTA: 2.4787 bits (rank_attn=512, G_attn=24, u8sv8, rdown=32)
#
# NOTE: rank_down=512 gives 3.31 extra bits (WAY over budget) — DO NOT USE
#       rank_down=128 gives 2.58 total bits — also over budget
#       rank_down=64 gives 2.46 total bits — within budget ✅
#       Max feasible rank_down ≈ 88 (without rank_attn)
#
# Experiment order (bits pre-verified with delta analysis from SOTA):
#  E1: rdown=64, noRattn, G_attn=4,  u4sv8, mse, recon=0.7, iter2  → extra=0.458 ✅
#  E2: rdown=64, noRattn, G_attn=24, u4sv8, mse, recon=0.7, iter2  → extra=0.458 ✅
#  E3: rdown=32, rattn=512, G_attn=24, u4sv8, mse, recon=0.7, iter2 → extra=0.464 ✅
#  E4: rdown=64, noRattn, G_attn=4,  u4sv8, mse, recon=0.7, iter5  → extra=0.458 ✅
#  (Best model) + zero-shot eval

set -uo pipefail
cd /home/qyyang/repo/GLoRCQ

RESULTS_LOG=logs/improve_qwen_results.md
CHAIN_LOG=logs/improve_qwen_chain.log
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

# ---- Initialize results file ----
mkdir -p logs
cat > "$RESULTS_LOG" << 'HEADER'
# GLoRCQ Qwen1.5-MoE PPL Optimization Results (v2, bits-verified)

| Experiment | PPL | Total Bits | Key Config |
|------------|-----|-----------|-----------|
| FP16 baseline | 6.79 | 16.0 | — |
| TileQ_v (target) | 7.35 | 2.16 | GPTQ all + 2D-tiling |
| TileQ GPTQ baseline | 7.98 | 2.13 | GPTQ only |
| GLoRCQ SOTA | 8.48 | 2.4787 | rattn=512,Gattn=24,u8sv8,rdown=32 |

HEADER
log "=== Qwen1.5-MoE PPL optimization chain v2 started ==="

# =============================================
# E1: rdown=64, no rank_attn, G_attn=4, u4sv8, mse, recon=0.7, iter2
# Bits: extra≈0.458, total≈2.458 (verified by delta analysis)
# Hypothesis: combines rank_down improvement + G_attn=4 aggressive sharing
# =============================================
wait_for_gpu
log "=== E1: rdown=64, noRattn, Gattn=4, u4sv8, mse, recon=0.7, iter2 [extra≈0.458] ==="
E1_OUT=/home/qyyang/resource_dir/GLoRCQ_out/improve_e1_rdown64_gattn4_iter2
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path "$E1_OUT" \
    --qbit 2 --groupsize 128 --nsamples 128 \
    --rank 32 --rank_down 64 \
    --n_iter 5 --n_lora_iter 2 \
    --G_moe 128 --G_attn 4 \
    --u_bits 4 --sv_bits 8 \
    --w_clip --hessian_svd --recon_weight 0.7 \
    --use_turboquant --search_act_alpha \
    > logs/improve_e1_rdown64_gattn4_iter2.log 2>&1
log "E1 quant done"

wait_for_gpu
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python evaluate/eval_ppl.py \
    --model_path "$E1_OUT" --device $DEVICE \
    > logs/eval_improve_e1.log 2>&1
E1_PPL=$(grep 'WikiText-2 PPL' logs/eval_improve_e1.log | awk '{print $NF}')
E1_BITS=$(grep 'Total average' logs/improve_e1_rdown64_gattn4_iter2.log | awk '{print $NF}' | head -1)
append_result "E1 rdown=64,noRattn,Gattn=4,u4sv8,mse,recon=0.7,iter2" \
    "${E1_PPL:-?}" "${E1_BITS:-?}" \
    "rank=32,rank_down=64,G_attn=4,u4sv8,mse,recon=0.7,n_lora_iter=2"
log "E1 PPL=$E1_PPL  bits=$E1_BITS"

# =============================================
# E2: rdown=64, no rank_attn, G_attn=24, u4sv8, mse, recon=0.7, iter2
# Bits: extra≈0.458, total≈2.458 (verified)
# Hypothesis: E1 but G_attn=24 (less aggressive attention sharing)
# =============================================
wait_for_gpu
log "=== E2: rdown=64, noRattn, Gattn=24, u4sv8, mse, recon=0.7, iter2 [extra≈0.458] ==="
E2_OUT=/home/qyyang/resource_dir/GLoRCQ_out/improve_e2_rdown64_gattn24_iter2
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path "$E2_OUT" \
    --qbit 2 --groupsize 128 --nsamples 128 \
    --rank 32 --rank_down 64 \
    --n_iter 5 --n_lora_iter 2 \
    --G_moe 128 --G_attn 24 \
    --u_bits 4 --sv_bits 8 \
    --w_clip --hessian_svd --recon_weight 0.7 \
    --use_turboquant --search_act_alpha \
    > logs/improve_e2_rdown64_gattn24_iter2.log 2>&1
log "E2 quant done"

wait_for_gpu
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python evaluate/eval_ppl.py \
    --model_path "$E2_OUT" --device $DEVICE \
    > logs/eval_improve_e2.log 2>&1
E2_PPL=$(grep 'WikiText-2 PPL' logs/eval_improve_e2.log | awk '{print $NF}')
E2_BITS=$(grep 'Total average' logs/improve_e2_rdown64_gattn24_iter2.log | awk '{print $NF}' | head -1)
append_result "E2 rdown=64,noRattn,Gattn=24,u4sv8,mse,recon=0.7,iter2" \
    "${E2_PPL:-?}" "${E2_BITS:-?}" \
    "rank=32,rank_down=64,G_attn=24,u4sv8,mse,recon=0.7,n_lora_iter=2"
log "E2 PPL=$E2_PPL  bits=$E2_BITS"

# =============================================
# E3: rdown=32, rattn=512, G_attn=24, u4sv8, mse, recon=0.7, iter2
# Bits: extra≈0.464, total≈2.464 (verified)
# Hypothesis: SOTA config + mse + recon (isolate quality hyperparams effect)
# =============================================
wait_for_gpu
log "=== E3: rdown=32, rattn=512, Gattn=24, u4sv8, mse, recon=0.7, iter2 [extra≈0.464] ==="
E3_OUT=/home/qyyang/resource_dir/GLoRCQ_out/improve_e3_rattn512_mse_recon
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path "$E3_OUT" \
    --qbit 2 --groupsize 128 --nsamples 128 \
    --rank 32 --rank_attn 512 \
    --n_iter 5 --n_lora_iter 2 \
    --G_moe 128 --G_attn 24 \
    --u_bits 4 --sv_bits 8 \
    --w_clip --hessian_svd --recon_weight 0.7 \
    --use_turboquant --search_act_alpha \
    > logs/improve_e3_rattn512_mse_recon.log 2>&1
log "E3 quant done"

wait_for_gpu
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python evaluate/eval_ppl.py \
    --model_path "$E3_OUT" --device $DEVICE \
    > logs/eval_improve_e3.log 2>&1
E3_PPL=$(grep 'WikiText-2 PPL' logs/eval_improve_e3.log | awk '{print $NF}')
E3_BITS=$(grep 'Total average' logs/improve_e3_rattn512_mse_recon.log | awk '{print $NF}' | head -1)
append_result "E3 rdown=32,rattn=512,Gattn=24,u4sv8,mse,recon=0.7,iter2" \
    "${E3_PPL:-?}" "${E3_BITS:-?}" \
    "rank=32,rank_attn=512,G_attn=24,u4sv8,mse,recon=0.7,n_lora_iter=2"
log "E3 PPL=$E3_PPL  bits=$E3_BITS"

# =============================================
# E4: rdown=64, no rank_attn, G_attn=4, u4sv8, mse, recon=0.7, iter5
# Bits: extra≈0.458, total≈2.458 (verified)
# Hypothesis: E1 with more LoftQ iterations (5 vs 2)
# =============================================
wait_for_gpu
log "=== E4: rdown=64, noRattn, Gattn=4, u4sv8, mse, recon=0.7, iter5 [extra≈0.458] ==="
E4_OUT=/home/qyyang/resource_dir/GLoRCQ_out/improve_e4_rdown64_gattn4_iter5
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path "$E4_OUT" \
    --qbit 2 --groupsize 128 --nsamples 128 \
    --rank 32 --rank_down 64 \
    --n_iter 5 --n_lora_iter 5 \
    --G_moe 128 --G_attn 4 \
    --u_bits 4 --sv_bits 8 \
    --w_clip --hessian_svd --recon_weight 0.7 \
    --use_turboquant --search_act_alpha \
    > logs/improve_e4_rdown64_gattn4_iter5.log 2>&1
log "E4 quant done"

wait_for_gpu
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python evaluate/eval_ppl.py \
    --model_path "$E4_OUT" --device $DEVICE \
    > logs/eval_improve_e4.log 2>&1
E4_PPL=$(grep 'WikiText-2 PPL' logs/eval_improve_e4.log | awk '{print $NF}')
E4_BITS=$(grep 'Total average' logs/improve_e4_rdown64_gattn4_iter5.log | awk '{print $NF}' | head -1)
append_result "E4 rdown=64,noRattn,Gattn=4,u4sv8,mse,recon=0.7,iter5" \
    "${E4_PPL:-?}" "${E4_BITS:-?}" \
    "rank=32,rank_down=64,G_attn=4,u4sv8,mse,recon=0.7,n_lora_iter=5"
log "E4 PPL=$E4_PPL  bits=$E4_BITS"

# =============================================
# Summary and zero-shot for best model
# =============================================
log "=== All PPL experiments done. Summary in $RESULTS_LOG ==="

BEST_PPL="9.99"
BEST_OUT=""
BEST_NAME=""
for exp in E1 E2 E3 E4; do
    eval "ppl=\${${exp}_PPL:-9.99}"
    eval "out=\${${exp}_OUT}"
    if [ -n "$ppl" ] && python3 -c "exit(0 if float('$ppl') < float('$BEST_PPL') else 1)" 2>/dev/null; then
        BEST_PPL="$ppl"
        BEST_OUT="$out"
        BEST_NAME="$exp"
    fi
done
log "Best experiment: $BEST_NAME with PPL=$BEST_PPL from $BEST_OUT"

if [ -n "$BEST_OUT" ] && [ -d "$BEST_OUT" ]; then
    wait_for_gpu
    log "=== Zero-shot eval on best model ($BEST_NAME PPL=$BEST_PPL) ==="
    BEST_ZEROSHOT_LOG=logs/eval_improve_${BEST_NAME}_zeroshot.log
    BEST_ZEROSHOT_JSON=logs/eval_improve_${BEST_NAME}_zeroshot.json
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python evaluate/eval_zeroshot.py \
        --model_path "$BEST_OUT" \
        --device $DEVICE \
        --tasks arc_challenge,arc_easy,winogrande,hellaswag,piqa \
        --batch_size 1 \
        --output_json "$BEST_ZEROSHOT_JSON" \
        > "$BEST_ZEROSHOT_LOG" 2>&1
    log "Zero-shot eval done."
    grep -E "arc_challenge|arc_easy|winogrande|hellaswag|piqa|Avg" "$BEST_ZEROSHOT_LOG" | tail -10 | tee -a "$CHAIN_LOG"

    echo "" >> "$RESULTS_LOG"
    echo "## Best Model ($BEST_NAME, PPL=$BEST_PPL) Zero-shot Results" >> "$RESULTS_LOG"
    echo "\`\`\`" >> "$RESULTS_LOG"
    grep -E "arc_challenge|arc_easy|winogrande|hellaswag|piqa|Average" "$BEST_ZEROSHOT_LOG" | tail -10 >> "$RESULTS_LOG"
    echo "\`\`\`" >> "$RESULTS_LOG"
fi

log "=== Qwen1.5-MoE improvement chain v2 completed ==="
cat "$RESULTS_LOG"
