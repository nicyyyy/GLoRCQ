#!/bin/bash
# ===========================================================================
# g8_grassmannian FINAL — the G-sweep winner, locked for the paper.
#   Re-quantize g8 (phase1 cache reused, ~2.5h) then FULL PPL + FULL ZS
#   (NO --limit, the definitive numbers comparable to the baseline's full ZS
#   64.48). bs=8 amortizes the CPU-offload cost (loglikelihood is batch-invariant).
#
# g8 = qbit2 / fix_rank32 / G8 / grassmannian(32 clusters, real) / attn 4-bit /
#   fp16-LoRA / seed42. Triage (limit500) gave PPL 4.50 / ZS 66.56.
#
# nwonga100 test:0 / GPU4 only (0-3 busy) -> fake eval via device_map=auto
# (GPU4 + ~14GB CPU offload). Keeps the 94GB fake at the end (winner artifact).
# ===========================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

VENV=/home/qyyang/repo/GLoRCQ/.venv/bin/python
MODEL=/mnt/Data/yqy/resource_dir/hf_cache/models--mistralai--Mixtral-8x7B-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0
OUTBASE=/mnt/Data/yqy/resource_dir/glorcq_paper_exp
PHASE1=$OUTBASE/mixtral_gsweep_phase1_cache.pt
OUT=$OUTBASE/mixtral_g8_final
LOGDIR=/home/qyyang/repo/GLoRCQ/logs/mixtral_gsweep
RESULTS=$LOGDIR/results
mkdir -p "$RESULTS"

export HF_HOME=/mnt/Data/yqy/resource_dir/hf_cache
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TASKS=arc_challenge,arc_easy,winogrande,hellaswag,piqa

echo "[$(date)] g8 FINAL START" | tee "$LOGDIR/final.log"

# ---- Re-quantize g8 (phase1 cache reused) -> 94GB fake ----
$VENV run_quantize.py \
    --model_path "$MODEL" --output_path "$OUT" \
    --qbit 2 --fix_rank 32 --G 8 --group_size 128 \
    --lora_bit 16 --lora_iter 8 --ha_bsize 256 --id_bsize 256 \
    --attn_bits 4 --cluster_method grassmannian --cluster_seed 42 \
    --phase1_cache_path "$PHASE1" --no_export_real_quant \
    2>&1 | tee "$LOGDIR/quant_g8_final.log"
if ! ls "$OUT"/model*.safetensors >/dev/null 2>&1; then
    echo "[$(date)] g8 FINAL QUANT FAILED" | tee -a "$LOGDIR/final.log"; exit 1
fi
grep -E "TOTAL=.*bits/param" "$LOGDIR/quant_g8_final.log" | tail -1 | tee -a "$LOGDIR/final.log"

# ---- FULL PPL (confirm; should reproduce 4.50 bit-identical) ----
$VENV evaluate/eval_ppl.py --model_path "$OUT" --device auto \
    --max_length 2048 --stride 512 \
    --output_json "$RESULTS/ppl_g8_final.json" 2>&1 | tee "$LOGDIR/ppl_g8_final.log"
grep -E "WikiText-2 PPL" "$LOGDIR/ppl_g8_final.log" | tail -1 | tee -a "$LOGDIR/final.log"

# ---- FULL ZS (NO --limit; definitive, comparable to baseline full 64.48) ----
$VENV evaluate/eval_zeroshot.py --model_path "$OUT" --device auto \
    --tasks "$TASKS" --num_fewshot 0 --batch_size 8 \
    --add_bos --metric_mode acc \
    --output_json "$RESULTS/zs_full_g8_final.json" 2>&1 | tee "$LOGDIR/zs_full_g8_final.log"
grep -E "Average" "$LOGDIR/zs_full_g8_final.log" | tail -1 | tee -a "$LOGDIR/final.log"

echo "[$(date)] g8 FINAL DONE. fake kept at $OUT  MIXTRAL_G8_FINAL_DONE" | tee -a "$LOGDIR/final.log"
