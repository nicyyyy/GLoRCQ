#!/bin/bash
# vLLM inference benchmark for GLoRCQ speed comparison.
#
# Starts a vLLM server in Docker, runs the benchmark client, then cleans up.
# The measured decode throughput (tok/s at batch_size=1) can be directly
# compared against GLoRCQ's eval_speed.py results.
#
# Prerequisites:
#   - Docker with NVIDIA Container Toolkit installed
#   - HuggingFace model cached locally (or HUGGINGFACE_HUB_TOKEN set)
#
# Usage:
#   bash exp/bench_vllm.sh <hf_model> [gpu_id] [port]
#
# Examples:
#   bash exp/bench_vllm.sh Qwen/Qwen1.5-MoE-A2.7B 4
#   bash exp/bench_vllm.sh mistralai/Mixtral-8x7B-v0.1 4 8011
#
# Output:
#   logs/vllm_bench_<model_basename>.json

set -e

MODEL="${1:-Qwen/Qwen1.5-MoE-A2.7B}"
GPU_ID="${2:-4}"
PORT="${3:-8010}"
CONTAINER_NAME="glorcq_vllm_bench_${GPU_ID}"
MODEL_BASENAME=$(basename "$MODEL")
LOG_DIR="logs"
OUTPUT_JSON="${LOG_DIR}/vllm_bench_${MODEL_BASENAME}.json"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/.."
mkdir -p "$LOG_DIR"

echo "========================================"
echo " vLLM Benchmark"
echo "  model  : $MODEL"
echo "  GPU    : $GPU_ID"
echo "  port   : $PORT"
echo "  output : $OUTPUT_JSON"
echo "========================================"

# ── Cleanup helper ────────────────────────────────────────────────────────────
cleanup() {
    echo "[$(date '+%H:%M:%S')] Stopping vLLM container..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
}
trap cleanup EXIT

# ── Start vLLM server ─────────────────────────────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Starting vLLM server (GPU $GPU_ID, port $PORT)..."

# Mount HuggingFace cache so model doesn't need to re-download
HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"

docker run -d \
    --name "$CONTAINER_NAME" \
    --rm \
    --gpus "device=${GPU_ID}" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -p "${PORT}:8000" \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model "$MODEL" \
    --max-model-len 512 \
    --trust-remote-code \
    --dtype half \
    --disable-log-requests \
    2>&1

# ── Wait for server ready ─────────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] Waiting for server to be ready..."
MAX_WAIT=180
ELAPSED=0
until curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; do
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: vLLM server did not start within ${MAX_WAIT}s"
        docker logs "$CONTAINER_NAME" 2>&1 | tail -20
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "  ... waited ${ELAPSED}s"
done
echo "[$(date '+%H:%M:%S')] Server ready."

# ── Run benchmark ─────────────────────────────────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Running benchmark (batch_size=1, prompt=128, gen=128)..."
python exp/bench_vllm_client.py \
    --host "localhost" \
    --port "$PORT" \
    --model "$MODEL" \
    --prompt_len 128 \
    --gen_len 128 \
    --warmup 2 \
    --runs 5 \
    --output_json "$OUTPUT_JSON"

echo ""
echo "[$(date '+%H:%M:%S')] Benchmark complete. Results: $OUTPUT_JSON"
