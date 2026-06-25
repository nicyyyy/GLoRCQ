#!/bin/bash
# GLoRCQ Qwen1.5-MoE: Explore improvements on SOTA PPL=7.62
# Base: rank=32, rank_down=512, rank_attn=512, u4/u_attn8/sv8, G_moe=128, n_lora_iter=2
# Run from test:0 (SLURM step cgroup) to avoid 5-min kill
#
# E1: rank=128 for gate/up (was 32) → hypothesis: gate/up are bottlenecks
# E2: u_bits=8 for MoE U (was 4)   → hypothesis: int4 U precision is limiting
# E3: n_lora_iter=3 (was 2)         → hypothesis: more LoftQ iters improve PPL
# PPL only (no zero-shot), save time

set -uo pipefail
cd /home/qyyang/repo/GLoRCQ

# Lock to prevent duplicate runs (in case multiple instances were queued)
LOCKFILE=/tmp/explore_qwen.lock
if [ -f "$LOCKFILE" ]; then
    echo "[$(date '+%F %T')] Another instance already running (lockfile exists). Exiting."
    exit 0
fi
echo $$ > "$LOCKFILE"
trap "rm -f $LOCKFILE" EXIT

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

# ---- Init results file ----
mkdir -p logs
cat > "$RESULTS_LOG" << 'HEADER'
# GLoRCQ Explore: Improvements on SOTA PPL=7.62

| Experiment | PPL | Total Bits | Notes |
|------------|-----|-----------|-------|
| SOTA (repro_v2_lora2) | 7.62 | 3.1931 | rank=32, rdown=512, rattn=512, u4/u_attn8/sv8, iter2 |
HEADER

log "=== explore_qwen chain started (base: SOTA PPL=7.62) ==="

# ============================================================
# E1: rank=128 for gate/up (was 32), all else same as SOTA
# ============================================================
wait_for_gpu
log "=== E1: rank=128 gate/up (was 32), rank_down=512, rank_attn=512, u4sv8, iter2 ==="
E1_OUT=/home/qyyang/resource_dir/GLoRCQ_out/explore_e1_rank128
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path "$E1_OUT" \
    --qbit 2 --groupsize 128 --nsamples 128 \
    --rank 128 --rank_down 512 --rank_attn 512 --rank_cluster 32 \
    --n_iter 5 --n_lora_iter 2 \
    --G_moe 128 --G_attn 4 \
    --u_bits 4 --u_bits_attn 8 --sv_bits 8 \
    --w_clip --hessian_svd --recon_weight 0.7 \
    --use_turboquant --search_act_alpha \
    > logs/explore_e1_rank128.log 2>&1
log "E1 quant done"

wait_for_gpu
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python evaluate/eval_ppl.py \
    --model_path "$E1_OUT" --device $DEVICE \
    > logs/eval_explore_e1.log 2>&1
E1_PPL=$(grep 'WikiText-2 PPL' logs/eval_explore_e1.log | awk '{print $NF}')
E1_BITS=$(grep 'Total average' logs/explore_e1_rank128.log | awk '{print $NF}' | tail -1)
log "E1 RESULT: rank128_gate_up → PPL=${E1_PPL}  bits=${E1_BITS}"
echo "| E1 rank=128 gate/up | ${E1_PPL:-?} | ${E1_BITS:-?} | rank=128,rdown=512,rattn=512,u4sv8,iter2 |" >> "$RESULTS_LOG"

# ============================================================
# E2: u_bits=8 for MoE U (was 4), all else same as SOTA
# ============================================================
wait_for_gpu
log "=== E2: u_bits=8 MoE U (was 4), rank=32, rank_down=512, rank_attn=512, sv8, iter2 ==="
E2_OUT=/home/qyyang/resource_dir/GLoRCQ_out/explore_e2_ubits8
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path "$E2_OUT" \
    --qbit 2 --groupsize 128 --nsamples 128 \
    --rank 32 --rank_down 512 --rank_attn 512 --rank_cluster 32 \
    --n_iter 5 --n_lora_iter 2 \
    --G_moe 128 --G_attn 4 \
    --u_bits 8 --u_bits_attn 8 --sv_bits 8 \
    --w_clip --hessian_svd --recon_weight 0.7 \
    --use_turboquant --search_act_alpha \
    > logs/explore_e2_ubits8.log 2>&1
log "E2 quant done"

wait_for_gpu
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python evaluate/eval_ppl.py \
    --model_path "$E2_OUT" --device $DEVICE \
    > logs/eval_explore_e2.log 2>&1
E2_PPL=$(grep 'WikiText-2 PPL' logs/eval_explore_e2.log | awk '{print $NF}')
E2_BITS=$(grep 'Total average' logs/explore_e2_ubits8.log | awk '{print $NF}' | tail -1)
log "E2 RESULT: ubits8_moe → PPL=${E2_PPL}  bits=${E2_BITS}"
echo "| E2 u_bits=8 MoE | ${E2_PPL:-?} | ${E2_BITS:-?} | rank=32,rdown=512,rattn=512,u8sv8,iter2 |" >> "$RESULTS_LOG"

# ============================================================
# E3: n_lora_iter=3 (was 2), all else same as SOTA
# ============================================================
wait_for_gpu
log "=== E3: n_lora_iter=3 (was 2), rank=32, rank_down=512, rank_attn=512, u4sv8 ==="
E3_OUT=/home/qyyang/resource_dir/GLoRCQ_out/explore_e3_iter3
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path "$E3_OUT" \
    --qbit 2 --groupsize 128 --nsamples 128 \
    --rank 32 --rank_down 512 --rank_attn 512 --rank_cluster 32 \
    --n_iter 5 --n_lora_iter 3 \
    --G_moe 128 --G_attn 4 \
    --u_bits 4 --u_bits_attn 8 --sv_bits 8 \
    --w_clip --hessian_svd --recon_weight 0.7 \
    --use_turboquant --search_act_alpha \
    > logs/explore_e3_iter3.log 2>&1
log "E3 quant done"

wait_for_gpu
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python evaluate/eval_ppl.py \
    --model_path "$E3_OUT" --device $DEVICE \
    > logs/eval_explore_e3.log 2>&1
E3_PPL=$(grep 'WikiText-2 PPL' logs/eval_explore_e3.log | awk '{print $NF}')
E3_BITS=$(grep 'Total average' logs/explore_e3_iter3.log | awk '{print $NF}' | tail -1)
log "E3 RESULT: iter3 → PPL=${E3_PPL}  bits=${E3_BITS}"
echo "| E3 n_lora_iter=3 | ${E3_PPL:-?} | ${E3_BITS:-?} | rank=32,rdown=512,rattn=512,u4sv8,iter3 |" >> "$RESULTS_LOG"

log "=== All explore experiments done ==="
cat "$RESULTS_LOG"

# ============================================================
# E4: rank=128 gate/up + n_lora_iter=3 (E1+E3 combo)
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

log "=== All explore experiments (E4+E5 combos) done ==="
cat "$RESULTS_LOG"
