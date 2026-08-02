#!/usr/bin/env bash
# ===========================================================================
# CLASP — Qwen3-30B-A3B (canonical configuration)
#   Same recipe as Qwen1.5-MoE, at the fair-bit rank for Qwen3's smaller
#   per-expert matrices: rank 16, G=128, 4-bit GPTQ attention, Grassmannian
#   clustering, int8 pooled factors.
#   Runs: quantize -> WikiText-2 PPL -> 5-task zero-shot.
#   Expected: ~2.16 bits, PPL ~8.97.
#   NOTE: quantization loads the fp16 model (~60 GB) — needs a >=96 GB GPU
#   (e.g. H200); the quantized model itself evaluates on much less.
# Every parameter below can be overridden via environment variables.
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen3-30B-A3B}
OUT=${OUT:-./outputs/qwen3_30b}
PY=${PY:-python}
DEVICE=${DEVICE:-cuda:0}
TASKS=${TASKS:-arc_challenge,arc_easy,piqa,winogrande,hellaswag}

QBIT=${QBIT:-2}
FIX_RANK=${FIX_RANK:-16}
G=${G:-128}
GROUP_SIZE=${GROUP_SIZE:-128}
LORA_BIT=${LORA_BIT:-16}
LORA_ITER=${LORA_ITER:-8}
ATTN_BITS=${ATTN_BITS:-4}
CLUSTER_METHOD=${CLUSTER_METHOD:-grassmannian}
CLUSTER_RANK=${CLUSTER_RANK:-32}
CLUSTER_RECON_WEIGHT=${CLUSTER_RECON_WEIGHT:-0.0}
CLUSTER_SEED=${CLUSTER_SEED:-42}

mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "[$(date)] CLASP quantize: $MODEL -> $OUT"
$PY run_quantize.py \
    --model_path "$MODEL" --output_path "$OUT" \
    --qbit "$QBIT" --fix_rank "$FIX_RANK" --G "$G" --group_size "$GROUP_SIZE" \
    --lora_bit "$LORA_BIT" --lora_iter "$LORA_ITER" \
    --attn_bits "$ATTN_BITS" \
    --int8_lora --int8_lora_v --pool_kmeans \
    --cluster_method "$CLUSTER_METHOD" --cluster_rank "$CLUSTER_RANK" \
    --cluster_recon_weight "$CLUSTER_RECON_WEIGHT" --cluster_seed "$CLUSTER_SEED" \
    --export_real_quant \
    --phase1_cache_path "$OUT/phase1_cache.pt"

echo "[$(date)] WikiText-2 PPL"
$PY evaluate/eval_ppl.py \
    --model_path "$OUT" --device "$DEVICE" \
    --max_length 2048 --stride 512 \
    --output_json "$OUT/ppl.json"

echo "[$(date)] Zero-shot (5 tasks, acc)"
$PY evaluate/eval_zeroshot.py \
    --model_path "$OUT" --device "$DEVICE" \
    --tasks "$TASKS" --num_fewshot 0 --batch_size 8 --metric_mode acc \
    --output_json "$OUT/zeroshot.json"

echo "[$(date)] Done. Results in $OUT/ppl.json and $OUT/zeroshot.json"

# --------------------------------------------------------------------------
# Optional: decode-speed benchmark on the real-quantized checkpoint.
# Requires the CUDA kernels (bash scripts/build_kernels.sh). To shrink the
# checkpoint to its real footprint first, re-run run_quantize.py with the
# extra flag --strip_fp16_quantized.
#
# $PY inference/eval_speed.py --model_path "$OUT" \
#     --batch_size 1 --prompt_len 128 --gen_len 128 --max_seq_len 2048
# --------------------------------------------------------------------------
