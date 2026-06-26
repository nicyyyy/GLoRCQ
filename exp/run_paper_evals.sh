#!/bin/bash
# ============================================================
#  GLoRCQ — Complete Paper Evaluation Suite
# ============================================================
#
# Runs all three models' evaluation chains sequentially.
# Each model: PPL + zero-shot (5 tasks) + MMLU + speed (GLoRCQ + vLLM FP16)
#
# Before running, set the model paths below (MODELS section).
#
# Usage:
#   bash exp/run_paper_evals.sh [gpu_id]
#
#   gpu_id : CUDA device index to use (default: 0)
#
# To run a single model only:
#   bash exp/paper_eval_qwen15moe.sh  <fake_quant> <real_quant> [gpu_id]
#   bash exp/paper_eval_mixtral.sh    <fake_quant> <real_quant> [gpu_id]
#   bash exp/paper_eval_qwen3_30b.sh  <fake_quant> <real_quant> [gpu_id]
#
# Outputs (one directory per model under logs/):
#   logs/paper_eval_qwen15moe/
#   logs/paper_eval_mixtral/
#   logs/paper_eval_qwen3_30b/
#
# Each directory contains:
#   ppl.json            WikiText-2 perplexity
#   zeroshot_5task.json ARC-C, ARC-E, WinoGrande, HellaSwag, PIQA (0-shot)
#   zeroshot_mmlu.json  MMLU (5-shot)
#   speed_glorcq.json   GLoRCQ decode throughput (tok/s)
#   speed_vllm.json     vLLM FP16 decode throughput (tok/s)
#   summary.txt         Human-readable summary table
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
# 4. Set model paths in the MODELS section below.
#
# 5. Run:
#      bash exp/run_paper_evals.sh 0
#
# ============================================================
#  MODELS — Fill in paths before running
# ============================================================
#
# Each model needs two paths:
#   FAKE_QUANT : HuggingFace-format directory with 2-bit GLoRCQ weights
#                (model.safetensors files + config.json + tokenizer files)
#                Used for PPL and zero-shot evaluation.
#   REAL_QUANT : Directory with glorcq_model.pt + cross_layer_info.pt
#                Used for inference speed benchmark.
#                Set to "skip" if not available.
#
# NOTE: These models must be obtained from the GLoRCQ authors or
#       downloaded from HuggingFace (links TBD).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/.."

GPU_ID="${1:-0}"

# ── Model paths — EDIT THESE ─────────────────────────────────────────────────

# Qwen1.5-MoE-A2.7B (FP16 ≈ 28 GB; 2-bit fake-quant ≈ 7 GB)
QWEN15_FAKE="/path/to/Qwen1.5-MoE-A2.7B-GLoRCQ-2bit"
QWEN15_REAL="/path/to/Qwen1.5-MoE-A2.7B-GLoRCQ-2bit-realquant"

# Mixtral-8x7B-v0.1 (FP16 ≈ 94 GB; 2-bit fake-quant ≈ 22 GB)
# NOTE: vLLM FP16 speed test needs H200/B200 (FP16 OOM on A100 80G)
MIXTRAL_FAKE="/path/to/Mixtral-8x7B-v0.1-GLoRCQ-2bit"
MIXTRAL_REAL="/path/to/Mixtral-8x7B-v0.1-GLoRCQ-2bit-realquant"

# Qwen3-30B-A3B (FP16 ≈ 60 GB; 2-bit fake-quant ≈ 15 GB)
# NOTE: vLLM FP16 speed test needs H200/B200 (A100 80G OOM with KV cache)
QWEN3_FAKE="/path/to/Qwen3-30B-A3B-GLoRCQ-2bit"
QWEN3_REAL="/path/to/Qwen3-30B-A3B-GLoRCQ-2bit-realquant"

# ─────────────────────────────────────────────────────────────────────────────

# Validate paths are set
for VAR in QWEN15_FAKE QWEN15_REAL MIXTRAL_FAKE MIXTRAL_REAL QWEN3_FAKE QWEN3_REAL; do
    VAL="${!VAR}"
    if [[ "$VAL" == /path/to/* ]]; then
        echo "ERROR: $VAR is not set. Edit the MODELS section in $0 first."
        exit 1
    fi
done

TOTAL_START=$(date '+%s')
echo "============================================================"
echo "  GLoRCQ Paper Evaluation Suite"
echo "  GPU : cuda:${GPU_ID}"
echo "  Time: $(date '+%F %T')"
echo "============================================================"

# ── Model 1: Qwen1.5-MoE-A2.7B ───────────────────────────────────────────────
echo ""
echo "▶▶▶  Model 1/3: Qwen1.5-MoE-A2.7B"
bash exp/paper_eval_qwen15moe.sh "$QWEN15_FAKE" "$QWEN15_REAL" "$GPU_ID"

# ── Model 2: Mixtral-8x7B-v0.1 ───────────────────────────────────────────────
echo ""
echo "▶▶▶  Model 2/3: Mixtral-8x7B-v0.1"
bash exp/paper_eval_mixtral.sh "$MIXTRAL_FAKE" "$MIXTRAL_REAL" "$GPU_ID"

# ── Model 3: Qwen3-30B-A3B ────────────────────────────────────────────────────
echo ""
echo "▶▶▶  Model 3/3: Qwen3-30B-A3B"
bash exp/paper_eval_qwen3_30b.sh "$QWEN3_FAKE" "$QWEN3_REAL" "$GPU_ID"

# ── Final summary ─────────────────────────────────────────────────────────────
ELAPSED=$(( $(date '+%s') - TOTAL_START ))
echo ""
echo "============================================================"
echo "  ALL DONE  (total: ${ELAPSED}s)"
echo ""
echo "  Result directories:"
echo "    logs/paper_eval_qwen15moe/"
echo "    logs/paper_eval_mixtral/"
echo "    logs/paper_eval_qwen3_30b/"
echo ""
echo "  Quick summary:"
for MODEL_DIR in logs/paper_eval_qwen15moe logs/paper_eval_mixtral logs/paper_eval_qwen3_30b; do
    if [ -f "$MODEL_DIR/summary.txt" ]; then
        echo "  ── $(basename $MODEL_DIR) ──"
        cat "$MODEL_DIR/summary.txt" | sed 's/^/    /'
        echo ""
    fi
done
echo "============================================================"
