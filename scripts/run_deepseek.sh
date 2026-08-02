#!/usr/bin/env bash
# ===========================================================================
# CLASP — DeepSeek-MoE-16B (canonical configuration, "config B")
#   2-bit VQ routed experts (rank 16, G=128, Grassmannian, int8 pooled
#   factors) + 4-bit GPTQ attention + 4-bit GPTQ shared experts / layer-0
#   dense MLP (--shared_bits 4), clustering seed 0.
#   Runs: quantize -> WikiText-2 PPL -> 5-task zero-shot.
#   Expected: ~2.21 bits (attn+FFN, excl. embed/lm_head), PPL ~6.60,
#   zero-shot avg ~61.9.
# Every parameter below can be overridden via environment variables.
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-deepseek-ai/deepseek-moe-16b-base}
OUT=${OUT:-./outputs/deepseek_moe16b}
PY=${PY:-python}
DEVICE=${DEVICE:-cuda:0}
TASKS=${TASKS:-arc_challenge,arc_easy,winogrande,hellaswag,piqa}

QBIT=${QBIT:-2}
FIX_RANK=${FIX_RANK:-16}
G=${G:-128}
GROUP_SIZE=${GROUP_SIZE:-128}
LORA_BIT=${LORA_BIT:-16}
LORA_ITER=${LORA_ITER:-8}
ATTN_BITS=${ATTN_BITS:-4}
SHARED_BITS=${SHARED_BITS:-4}
CLUSTER_METHOD=${CLUSTER_METHOD:-grassmannian}
CLUSTER_SEED=${CLUSTER_SEED:-0}

mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "[$(date)] CLASP quantize: $MODEL -> $OUT"
$PY run_quantize.py \
    --model_path "$MODEL" --output_path "$OUT" \
    --qbit "$QBIT" --fix_rank "$FIX_RANK" --G "$G" --group_size "$GROUP_SIZE" \
    --lora_bit "$LORA_BIT" --lora_iter "$LORA_ITER" \
    --ha_bsize 256 --id_bsize 256 \
    --attn_bits "$ATTN_BITS" --shared_bits "$SHARED_BITS" \
    --int8_lora --int8_lora_v --pool_kmeans \
    --cluster_method "$CLUSTER_METHOD" --cluster_seed "$CLUSTER_SEED" \
    --export_real_quant \
    --phase1_cache_path "$OUT/phase1_cache.pt"

# DeepSeek-MoE is a trust_remote_code model: save_pretrained copies config +
# tokenizer but NOT the custom modeling .py files. Copy them from the HF
# snapshot so the quantized dir loads standalone.
$PY - "$MODEL" "$OUT" <<'PYEOF'
import os, shutil, sys
from huggingface_hub import snapshot_download
src, dst = sys.argv[1], sys.argv[2]
snap = src if os.path.isdir(src) else snapshot_download(
    src, allow_patterns=["*.py", "tokenizer*"])
for name in ("configuration_deepseek.py", "modeling_deepseek.py",
             "tokenizer.json", "tokenizer_config.json"):
    try:
        shutil.copy(f"{snap}/{name}", f"{dst}/{name}")
    except FileNotFoundError:
        pass
PYEOF

echo "[$(date)] WikiText-2 PPL"
$PY evaluate/eval_ppl.py \
    --model_path "$OUT" --device "$DEVICE" \
    --max_length 2048 --stride 512 \
    --output_json "$OUT/ppl.json"

echo "[$(date)] Zero-shot (5 tasks, acc, add_bos)"
$PY evaluate/eval_zeroshot.py \
    --model_path "$OUT" --device "$DEVICE" \
    --tasks "$TASKS" --num_fewshot 0 --batch_size 16 \
    --metric_mode acc --add_bos \
    --output_json "$OUT/zeroshot.json"

echo "[$(date)] Done. Results in $OUT/ppl.json and $OUT/zeroshot.json"

# --------------------------------------------------------------------------
# Optional: decode-speed benchmark. Needs a stripped real-quant checkpoint —
# re-run run_quantize.py above with the extra flag --strip_fp16_quantized,
# build the CUDA kernels (bash scripts/build_kernels.sh), then:
#
# GLORCQ_DEEPSEEK_GATHER=1 GLORCQ_DEEPSEEK_IDXKERNEL=1 \
# $PY evaluate/bench_deepseek.py --config full_graph \
#     --real_path "$OUT" --prompt_len 128 --gen_len 128 --max_seq_len 384
#
# fp16 baseline (same harness):
# $PY evaluate/bench_deepseek.py --config fp16 --fp16_path "$MODEL" \
#     --prompt_len 128 --gen_len 128 --max_seq_len 384
# --------------------------------------------------------------------------
