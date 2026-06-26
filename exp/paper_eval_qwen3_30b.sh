#!/bin/bash
# ============================================================
#  GLoRCQ Paper Evaluation — Qwen3-30B-A3B
# ============================================================
#
# Runs the full paper evaluation chain for Qwen3-30B-A3B:
#   Step 1: WikiText-2 PPL                   (~30 min, fake-quant model)
#   Step 2: Zero-shot 5 tasks (0-shot)       (~90 min, fake-quant model)
#   Step 3: MMLU (5-shot)                    (~120 min, fake-quant model)
#   Step 4: GLoRCQ inference speed           (~5 min,  real-quant model)
#   Step 5: vLLM FP16 baseline speed         (~15 min, Docker required)
#
# GPU memory requirements:
#   - Steps 1-3 (fake-quant): Qwen3-30B 2-bit ≈ 15 GB → single 24 GB GPU OK
#   - Step 4 (real-quant):    same as above
#   - Step 5 (vLLM FP16):    Qwen3-30B FP16 ≈ 60 GB + KV cache
#                             → needs H200 141G or B200 192G (A100 80G OOM)
#
# Usage:
#   bash exp/paper_eval_qwen3_30b.sh <fake_quant_dir> <real_quant_dir> [gpu_id]
#
# Arguments:
#   fake_quant_dir  Path to the GLoRCQ fake-quant model
#   real_quant_dir  Path to the GLoRCQ real-quant model
#                   Pass "skip" to skip the GLoRCQ speed test
#   gpu_id          GPU device index (default: 0)
#
# Example:
#   bash exp/paper_eval_qwen3_30b.sh \
#       /data/models/Qwen3-30B-A3B-GLoRCQ-2bit \
#       /data/models/Qwen3-30B-A3B-GLoRCQ-2bit-realquant \
#       0

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/.."

HF_MODEL="Qwen/Qwen3-30B-A3B"
FAKE_QUANT="${1:?Usage: $0 <fake_quant_dir> <real_quant_dir> [gpu_id]}"
REAL_QUANT="${2:?Usage: $0 <fake_quant_dir> <real_quant_dir> [gpu_id]}"
GPU_ID="${3:-0}"
DEVICE="cuda:${GPU_ID}"
OUT="logs/paper_eval_qwen3_30b"
mkdir -p "$OUT"

echo "============================================================"
echo "  GLoRCQ Paper Eval — Qwen3-30B-A3B"
echo "  fake_quant : $FAKE_QUANT"
echo "  real_quant : $REAL_QUANT"
echo "  GPU        : cuda:${GPU_ID}"
echo "  output     : $OUT"
echo "============================================================"

# ── Step 1: PPL ──────────────────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Step 1/5 — WikiText-2 PPL"
python evaluate/eval_ppl.py \
    --model_path "$FAKE_QUANT" \
    --device "$DEVICE" \
    --max_length 2048 --stride 512 \
    --output_json "$OUT/ppl.json"

# ── Step 2: Zero-shot 5 tasks ────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Step 2/5 — Zero-shot (ARC-C/E, WinoGrande, HellaSwag, PIQA)"
python evaluate/eval_zeroshot.py \
    --model_path "$FAKE_QUANT" \
    --device "$DEVICE" \
    --tasks arc_challenge,arc_easy,winogrande,hellaswag,piqa \
    --num_fewshot 0 --batch_size 1 \
    --output_json "$OUT/zeroshot_5task.json"

# ── Step 3: MMLU 5-shot ──────────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Step 3/5 — MMLU (5-shot)"
python evaluate/eval_zeroshot.py \
    --model_path "$FAKE_QUANT" \
    --device "$DEVICE" \
    --tasks mmlu \
    --num_fewshot 5 --batch_size 1 \
    --output_json "$OUT/zeroshot_mmlu.json"

# ── Step 4: GLoRCQ inference speed ───────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Step 4/5 — GLoRCQ inference speed"
if [ "$REAL_QUANT" = "skip" ]; then
    echo "  Skipped (real_quant_dir = 'skip')"
else
    python evaluate/eval_speed.py \
        --model_path "$REAL_QUANT" \
        --hf_model_path "$HF_MODEL" \
        --device "$DEVICE" \
        --prompt_len 128 --gen_len 128 \
        --num_warmup 2 --num_runs 5 \
        --output_json "$OUT/speed_glorcq.json"
fi

# ── Step 5: vLLM FP16 baseline ───────────────────────────────
# NOTE: Qwen3-30B FP16 ≈ 60 GB + KV cache overhead.
# A100 80G is at the limit and likely OOM. Use H200 or B200.
echo ""
echo "[$(date '+%H:%M:%S')] Step 5/5 — vLLM FP16 baseline speed"
echo "  WARNING: Qwen3-30B FP16 ≈ 60 GB + KV cache. Needs H200/B200."
if ! command -v docker &> /dev/null; then
    echo "  Skipped (Docker not found)"
else
    bash exp/bench_vllm.sh "$HF_MODEL" "$GPU_ID" 8022 \
        2>&1 | tee -a "$OUT/vllm_speed.log"
    cp "logs/vllm_bench_qwen3-30b-a3b.json" \
        "$OUT/speed_vllm.json" 2>/dev/null || true
fi

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Writing summary..."
python evaluate/print_eval_summary.py \
    --output_dir "$OUT" \
    --model_path "$FAKE_QUANT"

echo ""
echo "============================================================"
echo "  Done. Results in: $OUT"
echo "  Files: ppl.json | zeroshot_5task.json | zeroshot_mmlu.json"
echo "         speed_glorcq.json | speed_vllm.json | summary.txt"
echo "============================================================"
