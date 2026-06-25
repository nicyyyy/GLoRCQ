#!/bin/bash
# P3 Kurtosis Adaptive Rank experiment
# Base: SOTA config (rank=32, rattn=512, G_attn=24, u8sv8, n_lora_iter=2)
# P3: --adaptive_rank assigns higher rank to clusters with higher kurtosis, budget-neutral
# Bits: ~2.4787 (budget-neutral by design) ✅
#
# Run from test:0 (SLURM step cgroup):
#   tmux send-keys -t test:0 "bash run_p3_experiment.sh" Enter

set -uo pipefail
cd /home/qyyang/repo/GLoRCQ

RESULTS_LOG=logs/p3_results.md
CHAIN_LOG=logs/p3_chain.log
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
cat > "$RESULTS_LOG" << 'HEADER'
# P3 Kurtosis Adaptive Rank Results

| Experiment | PPL | Total Bits | Key Config |
|------------|-----|-----------|-----------|
| FP16 baseline | 6.79 | 16.0 | — |
| TileQ_v (target) | 7.35 | 2.16 | GPTQ + 2D-tiling |
| GLoRCQ SOTA | 8.48 | 2.4787 | rattn=512,Gattn=24,u8sv8,rdown=32,iter2 |

HEADER
log "=== P3 kurtosis adaptive rank experiment started ==="

# ====================================================================
# P3: SOTA config + kurtosis adaptive rank
# ====================================================================
wait_for_gpu
log "=== P3: SOTA config + --adaptive_rank [bits≈2.4787] ==="
P3_OUT=/home/qyyang/resource_dir/GLoRCQ_out/p3_kurtosis_adaptive_rank
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path "$P3_OUT" \
    --qbit 2 --groupsize 128 --nsamples 128 \
    --rank 32 --rank_attn 512 \
    --n_iter 5 --n_lora_iter 2 \
    --G_moe 128 --G_attn 24 \
    --u_bits 8 --sv_bits 8 \
    --hessian_svd --adaptive_rank \
    --use_turboquant --search_act_alpha \
    > logs/p3_kurtosis_adaptive_rank.log 2>&1
log "P3 quant done"

wait_for_gpu
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python evaluate/eval_ppl.py \
    --model_path "$P3_OUT" --device $DEVICE \
    > logs/eval_p3.log 2>&1
P3_PPL=$(grep 'WikiText-2 PPL' logs/eval_p3.log | awk '{print $NF}')
P3_BITS=$(grep 'Total average' logs/p3_kurtosis_adaptive_rank.log | awk '{print $NF}' | head -1)
echo "| P3 (kurtosis adaptive rank) | ${P3_PPL:-?} | ${P3_BITS:-?} | rank=32,rattn=512,Gattn=24,u8sv8,n_lora_iter=2,adaptive_rank |" >> "$RESULTS_LOG"
log "P3 PPL=$P3_PPL  bits=$P3_BITS"

# Zero-shot if PPL improved over SOTA
if python3 -c "exit(0 if float('${P3_PPL:-9.99}') < 8.48 else 1)" 2>/dev/null; then
    wait_for_gpu
    log "=== P3 improved! Running zero-shot eval ==="
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python evaluate/eval_zeroshot.py \
        --model_path "$P3_OUT" --device $DEVICE \
        --tasks arc_challenge,arc_easy,winogrande,hellaswag,piqa \
        --batch_size 1 --output_json logs/eval_p3_zeroshot.json \
        > logs/eval_p3_zeroshot.log 2>&1
    log "Zero-shot done."
    grep -E "arc_challenge|arc_easy|winogrande|hellaswag|piqa|Avg" logs/eval_p3_zeroshot.log | tail -10 | tee -a "$CHAIN_LOG"
fi

log "=== P3 experiment completed ==="
cat "$RESULTS_LOG"
