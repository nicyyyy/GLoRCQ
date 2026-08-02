#!/usr/bin/env bash
# ===========================================================================
# CLASP — Mixtral-8x7B-v0.1 (canonical configuration)
#   2-bit VQ routed experts + cross-layer pooling with SMALL clusters
#   (G=8 experts/cluster -> 32 Grassmannian clusters over 256 experts),
#   rank 32, 4-bit GPTQ attention, fp16 pooled factors (no int8 factors:
#   Mixtral's huge expert dims make the factor overhead negligible).
#   Runs: quantize -> WikiText-2 PPL -> 5-task zero-shot.
#   Expected: ~2.17 bits, PPL ~4.50, zero-shot avg ~66.6.
#   NOTE: quantization loads the fp16 model (~94 GB) — needs a >=141 GB GPU
#   (e.g. H200) or multi-GPU; the fake-quant checkpoint is also ~94 GB, and
#   evaluation below uses device_map=auto (--device auto).
# Every parameter below can be overridden via environment variables.
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-mistralai/Mixtral-8x7B-v0.1}
OUT=${OUT:-./outputs/mixtral_8x7b}
PY=${PY:-python}
DEVICE=${DEVICE:-auto}
TASKS=${TASKS:-arc_challenge,arc_easy,winogrande,hellaswag,piqa}

QBIT=${QBIT:-2}
FIX_RANK=${FIX_RANK:-32}
G=${G:-8}
GROUP_SIZE=${GROUP_SIZE:-128}
LORA_BIT=${LORA_BIT:-16}
LORA_ITER=${LORA_ITER:-8}
ATTN_BITS=${ATTN_BITS:-4}
CLUSTER_METHOD=${CLUSTER_METHOD:-grassmannian}
CLUSTER_SEED=${CLUSTER_SEED:-42}

mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "[$(date)] CLASP quantize: $MODEL -> $OUT"
$PY run_quantize.py \
    --model_path "$MODEL" --output_path "$OUT" \
    --qbit "$QBIT" --fix_rank "$FIX_RANK" --G "$G" --group_size "$GROUP_SIZE" \
    --lora_bit "$LORA_BIT" --lora_iter "$LORA_ITER" \
    --ha_bsize 256 --id_bsize 256 \
    --attn_bits "$ATTN_BITS" \
    --cluster_method "$CLUSTER_METHOD" --cluster_seed "$CLUSTER_SEED" \
    --no_export_real_quant \
    --phase1_cache_path "$OUT/phase1_cache.pt"

echo "[$(date)] WikiText-2 PPL"
$PY evaluate/eval_ppl.py \
    --model_path "$OUT" --device "$DEVICE" \
    --max_length 2048 --stride 512 \
    --output_json "$OUT/ppl.json"

echo "[$(date)] Zero-shot (5 tasks, acc, add_bos)"
$PY evaluate/eval_zeroshot.py \
    --model_path "$OUT" --device "$DEVICE" \
    --tasks "$TASKS" --num_fewshot 0 --batch_size 8 \
    --add_bos --metric_mode acc \
    --output_json "$OUT/zeroshot.json"

echo "[$(date)] Done. Results in $OUT/ppl.json and $OUT/zeroshot.json"

# --------------------------------------------------------------------------
# Optional: decode-speed benchmark. This needs a REAL-quant checkpoint —
# re-run run_quantize.py above replacing --no_export_real_quant with
# --export_real_quant --strip_fp16_quantized (writes cross_layer_info.pt and
# strips fp16 expert weights), build the CUDA kernels
# (bash scripts/build_kernels.sh), then:
#
# GLORCQ_MIXTRAL_GRAPH=1 GLORCQ_MIXTRAL_IDXKERNEL=1 \
# GLORCQ_PREFILL_DEQUANT=1 GLORCQ_GPTQ_ILP=1 \
# $PY inference/eval_speed.py --model_path "$OUT" \
#     --batch_size 1 --prompt_len 128 --gen_len 512 --max_seq_len 768
#
# fp16 baseline (same harness):
# $PY inference/eval_speed.py --model_path "$MODEL" --no_real_quant \
#     --batch_size 1 --prompt_len 128 --gen_len 512 --max_seq_len 768
# --------------------------------------------------------------------------
