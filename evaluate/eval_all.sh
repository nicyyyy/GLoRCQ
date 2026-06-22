#!/bin/bash
# Unified evaluation script for GLoRCQ fake-quant models.
#
# Runs three evaluations in sequence:
#   1. WikiText-2 PPL
#   2. Zero-shot accuracy (5 tasks, 0-shot)
#   3. MMLU accuracy (5-shot)
#   4. Print summary
#
# Usage:
#   bash evaluate/eval_all.sh <model_path> [output_dir] [gpu_id]
#
# Arguments:
#   model_path : HuggingFace repo ID or local path to a fake-quant model
#   output_dir : Directory to save result JSONs  (default: logs/eval_<model_basename>)
#   gpu_id     : CUDA device index               (default: 0)
#
# Examples:
#   bash evaluate/eval_all.sh your_user/Qwen1.5-MoE-A2.7B-GLoRCQ-2bit
#   bash evaluate/eval_all.sh /data/models/mixtral_glorcq logs/mixtral_eval 0

set -e

MODEL_PATH="${1}"
if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <model_path> [output_dir] [gpu_id]"
    exit 1
fi

MODEL_BASENAME=$(basename "$MODEL_PATH")
OUTPUT_DIR="${2:-logs/eval_${MODEL_BASENAME}}"
GPU_ID="${3:-0}"
DEVICE="cuda:${GPU_ID}"

# Run from repo root regardless of where the script is called from
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/.."

echo "============================================"
echo " GLoRCQ Eval Pipeline"
echo "  model   : $MODEL_PATH"
echo "  output  : $OUTPUT_DIR"
echo "  device  : $DEVICE"
echo "============================================"

mkdir -p "$OUTPUT_DIR"

# ── Step 1: WikiText-2 PPL ────────────────────────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Step 1/3: WikiText-2 PPL"
python evaluate/eval_ppl.py \
    --model_path "$MODEL_PATH" \
    --device "$DEVICE" \
    --max_length 2048 \
    --stride 512 \
    --output_json "$OUTPUT_DIR/ppl.json"
echo "[$(date '+%H:%M:%S')] Step 1 done"

# ── Step 2: Zero-shot (5 tasks, 0-shot) ──────────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Step 2/3: Zero-shot (ARC-C/E, WinoGrande, HellaSwag, PIQA)"
python evaluate/eval_zeroshot.py \
    --model_path "$MODEL_PATH" \
    --device "$DEVICE" \
    --tasks arc_challenge,arc_easy,winogrande,hellaswag,piqa \
    --num_fewshot 0 \
    --batch_size 1 \
    --output_json "$OUTPUT_DIR/zeroshot_5task.json"
echo "[$(date '+%H:%M:%S')] Step 2 done"

# ── Step 3: MMLU (5-shot) ─────────────────────────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Step 3/3: MMLU (5-shot)"
python evaluate/eval_zeroshot.py \
    --model_path "$MODEL_PATH" \
    --device "$DEVICE" \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size 1 \
    --output_json "$OUTPUT_DIR/zeroshot_mmlu.json"
echo "[$(date '+%H:%M:%S')] Step 3 done"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Writing summary..."
python evaluate/print_eval_summary.py \
    --output_dir "$OUTPUT_DIR" \
    --model_path "$MODEL_PATH"

echo ""
echo "============================================"
echo " All evaluations complete."
echo " Results saved to: $OUTPUT_DIR"
echo "============================================"
