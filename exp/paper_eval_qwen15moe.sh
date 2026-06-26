#!/bin/bash
# ============================================================
#  GLoRCQ Paper Evaluation — Qwen1.5-MoE-A2.7B
# ============================================================
#
# Runs the full paper evaluation chain for Qwen1.5-MoE-A2.7B:
#   Step 1: WikiText-2 PPL                   (~15 min, fake-quant model)
#   Step 2: Zero-shot 5 tasks (0-shot)       (~30 min, fake-quant model)
#   Step 3: MMLU (5-shot)                    (~60 min, fake-quant model)
#   Step 4: GLoRCQ inference speed           (~5 min,  real-quant model)
#   Step 5: vLLM FP16 baseline speed         (~10 min, Docker required)
#
# Usage:
#   bash exp/paper_eval_qwen15moe.sh <fake_quant_dir> <real_quant_dir> [gpu_id]
#
# Arguments:
#   fake_quant_dir  Path to the GLoRCQ fake-quant model
#                   (HuggingFace safetensors format, contains model.safetensors + config.json)
#   real_quant_dir  Path to the GLoRCQ real-quant model
#                   (contains glorcq_model.pt + cross_layer_info.pt)
#                   Pass "skip" to skip the GLoRCQ speed test
#   gpu_id          GPU device index (default: 0)
#
# Example:
#   bash exp/paper_eval_qwen15moe.sh \
#       /data/models/Qwen1.5-MoE-A2.7B-GLoRCQ-2bit \
#       /data/models/Qwen1.5-MoE-A2.7B-GLoRCQ-2bit-realquant \
#       0
#
# Requirements:
#   - Python environment with GLoRCQ deps:
#       pip install -e .   (or: uv sync)
#   - For Step 4: GLoRCQ CUDA kernels must be compiled:
#       bash scripts/build_kernels.sh
#   - For Step 5: Docker with NVIDIA Container Toolkit
#       https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/.."

HF_MODEL="Qwen/Qwen1.5-MoE-A2.7B"
FAKE_QUANT="${1:?Usage: $0 <fake_quant_dir> <real_quant_dir> [gpu_id]}"
REAL_QUANT="${2:?Usage: $0 <fake_quant_dir> <real_quant_dir> [gpu_id]}"
GPU_ID="${3:-0}"
DEVICE="cuda:${GPU_ID}"
OUT="logs/paper_eval_qwen15moe"
mkdir -p "$OUT"

echo "============================================================"
echo "  GLoRCQ Paper Eval — Qwen1.5-MoE-A2.7B"
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
echo ""
echo "[$(date '+%H:%M:%S')] Step 5/5 — vLLM FP16 baseline speed"
if ! command -v docker &> /dev/null; then
    echo "  Skipped (Docker not found)"
else
    bash exp/bench_vllm.sh "$HF_MODEL" "$GPU_ID" 8020 \
        2>&1 | tee -a "$OUT/vllm_speed.log"
    cp "logs/vllm_bench_$(basename $HF_MODEL | tr '/' '_' | tr '[:upper:]' '[:lower:]').json" \
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
