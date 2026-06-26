#!/bin/bash
# ============================================================
#  GLoRCQ — Complete Paper Evaluation Suite
# ============================================================
#
# Runs quantization + full evaluation for all three models sequentially.
# Each model: fake-quant → real-quant → PPL → zero-shot → MMLU → speed
#
# Before running, set the OUTPUT_BASE and per-model base dirs below.
# Each model's output will be organized as:
#   <model_base>/fake_quant/   — fake-quant safetensors (PPL/zeroshot)
#   <model_base>/real_quant/   — real-quant .pt files (speed test)
#   <model_base>/logs/         — JSON + summary results
#
# Quantization is skipped if the model is already present.
#
# Usage:
#   bash exp/run_paper_evals.sh [gpu_id]
#
#   gpu_id : CUDA device index to use (default: 0)
#
# To run a single model only:
#   bash exp/paper_eval_qwen15moe.sh  <output_dir> [gpu_id]
#   bash exp/paper_eval_mixtral.sh    <output_dir> [gpu_id]
#   bash exp/paper_eval_qwen3_30b.sh  <output_dir> [gpu_id]
#
# ============================================================
#  SETUP INSTRUCTIONS
# ============================================================
#
# 1. Clone repo and install dependencies:
#      git clone https://github.com/nicyyyy/GLoRCQ.git
#      cd GLoRCQ
#      pip install -e .          # or: uv sync
#
# 2. Build CUDA kernels (required for GLoRCQ speed test only):
#      bash scripts/build_kernels.sh
#
# 3. Install Docker + NVIDIA Container Toolkit (for vLLM speed test):
#      https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
#
# 4. Set output directories in the PATHS section below.
#
# 5. Run:
#      bash exp/run_paper_evals.sh 0
#
# ============================================================
#  PATHS — Set output directories before running
# ============================================================
#
# IMPORTANT: Paths must be on a filesystem with sufficient space:
#   Qwen1.5-MoE-A2.7B : fake-quant ~7 GB, real-quant ~7 GB
#   Mixtral-8x7B-v0.1  : fake-quant ~22 GB, real-quant ~22 GB
#   Qwen3-30B-A3B      : fake-quant ~15 GB, real-quant ~15 GB
#
# Recommended: use /home/qyyang/resource_dir/GLoRCQ_out/ (large quota)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/.."

GPU_ID="${1:-0}"

# ── Output directories — EDIT THESE ──────────────────────────────────────────

# Qwen1.5-MoE-A2.7B (FP16 ≈ 28 GB; quantization OK on A100 80G)
QWEN15_OUT="/path/to/glorcq_out/qwen15moe"

# Mixtral-8x7B-v0.1 (FP16 ≈ 94 GB; requires H200/B200 for quantization)
# Config: E5-pattern with rank_down=512, rank_attn=512, G_moe=32 (32L×8exp=256, 8/cluster)
MIXTRAL_OUT="/path/to/glorcq_out/mixtral"

# Qwen3-30B-A3B (FP16 ≈ 60 GB; requires H200/B200 for quantization)
QWEN3_OUT="/path/to/glorcq_out/qwen3_30b"

# ─────────────────────────────────────────────────────────────────────────────

# Validate paths are set
for VAR in QWEN15_OUT MIXTRAL_OUT QWEN3_OUT; do
    VAL="${!VAR}"
    if [[ "$VAL" == /path/to/* ]]; then
        echo "ERROR: $VAR is not set. Edit the PATHS section in $0 first."
        exit 1
    fi
done

TOTAL_START=$(date '+%s')
echo "============================================================"
echo "  GLoRCQ Paper Evaluation Suite (full chain)"
echo "  GPU : cuda:${GPU_ID}"
echo "  Time: $(date '+%F %T')"
echo "============================================================"

# ── Model 1: Qwen1.5-MoE-A2.7B ───────────────────────────────────────────────
echo ""
echo "▶▶▶  Model 1/3: Qwen1.5-MoE-A2.7B"
bash exp/paper_eval_qwen15moe.sh "$QWEN15_OUT" "$GPU_ID"

# ── Model 2: Mixtral-8x7B-v0.1 ───────────────────────────────────────────────
echo ""
echo "▶▶▶  Model 2/3: Mixtral-8x7B-v0.1"
bash exp/paper_eval_mixtral.sh "$MIXTRAL_OUT" "$GPU_ID"

# ── Model 3: Qwen3-30B-A3B ────────────────────────────────────────────────────
echo ""
echo "▶▶▶  Model 3/3: Qwen3-30B-A3B"
bash exp/paper_eval_qwen3_30b.sh "$QWEN3_OUT" "$GPU_ID"

# ── Final summary ─────────────────────────────────────────────────────────────
ELAPSED=$(( $(date '+%s') - TOTAL_START ))
echo ""
echo "============================================================"
echo "  ALL DONE  (total: ${ELAPSED}s)"
echo ""
echo "  Result directories:"
echo "    ${QWEN15_OUT}/logs/"
echo "    ${MIXTRAL_OUT}/logs/"
echo "    ${QWEN3_OUT}/logs/"
echo ""
echo "  Quick summary:"
for MODEL_DIR in "$QWEN15_OUT" "$MIXTRAL_OUT" "$QWEN3_OUT"; do
    SUMMARY="${MODEL_DIR}/logs/summary.txt"
    if [ -f "$SUMMARY" ]; then
        echo "  ── $(basename $MODEL_DIR) ──"
        cat "$SUMMARY" | sed 's/^/    /'
        echo ""
    fi
done
echo "============================================================"
