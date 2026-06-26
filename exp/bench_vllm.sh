#!/bin/bash
# vLLM FP16 baseline benchmark for GLoRCQ paper speed comparison.
#
# Starts a vLLM server in Docker, runs the benchmark client, then cleans up.
# Results can be directly compared with evaluate/eval_speed.py (Our Method column).
#
# Prerequisites:
#   - Docker with NVIDIA Container Toolkit installed
#   - HuggingFace model cached at $HF_HOME (~/.cache/huggingface by default)
#   - For gated models: export HUGGINGFACE_HUB_TOKEN=hf_xxx
#
# Usage:
#   bash exp/bench_vllm.sh <hf_model> [gpu_id] [port]
#
# Examples:
#   bash exp/bench_vllm.sh Qwen/Qwen1.5-MoE-A2.7B 0
#   bash exp/bench_vllm.sh Qwen/Qwen3-30B-A3B 0 8011
#   bash exp/bench_vllm.sh mistralai/Mixtral-8x7B-v0.1 0 8012
#
# Output:
#   logs/vllm_bench_<model>.log   (server logs + client output)
#   logs/vllm_bench_<model>.json  (JSON metrics for paper table)

set -euo pipefail

MODEL="${1:?Usage: $0 <hf_model> [gpu_id] [port]}"
GPU_ID="${2:-0}"
PORT="${3:-8010}"

# Must match evaluate/eval_speed.py defaults for fair comparison
PROMPT_LEN=128
GEN_LEN=128
WARMUP=2
RUNS=5

MODEL_SLUG=$(basename "$MODEL" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
CONTAINER="glorcq_vllm_${GPU_ID}"
LOG_DIR="logs"
LOG="${LOG_DIR}/vllm_bench_${MODEL_SLUG}.log"
OUT="${LOG_DIR}/vllm_bench_${MODEL_SLUG}.json"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/.."
mkdir -p "$LOG_DIR"

echo "======================================================="
echo "  vLLM FP16 Baseline Benchmark"
echo "  model      : $MODEL"
echo "  GPU        : cuda:${GPU_ID}"
echo "  port       : $PORT"
echo "  prompt_len : $PROMPT_LEN   gen_len : $GEN_LEN"
echo "  output     : $OUT"
echo "======================================================="
echo ""

# ── Cleanup ──────────────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[$(date '+%H:%M:%S')] Stopping vLLM container..."
    docker stop "$CONTAINER" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Kill any leftover container from a previous crashed run
docker stop "$CONTAINER" 2>/dev/null || true
sleep 1

# ── HuggingFace cache ─────────────────────────────────────────────────────────
# Prefer $HF_HOME, then standard per-user location
HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"
if [ ! -d "$HF_CACHE" ]; then
    echo "WARNING: HF cache not found at $HF_CACHE"
    echo "  Model will be downloaded inside Docker (slow on first run)."
fi

EXTRA_DOCKER_ARGS=""
if [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    EXTRA_DOCKER_ARGS="--env HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}"
fi

# ── Start vLLM server ─────────────────────────────────────────────────────────
# --dtype auto   : lets vLLM pick bfloat16/float16 per model config
#                  (Qwen3 needs bfloat16; Qwen1.5/Mixtral work with float16)
# --max-model-len: just enough for our benchmark; reduces GPU memory needed
MAX_CTX=$((PROMPT_LEN + GEN_LEN + 64))

echo "[$(date '+%H:%M:%S')] Starting vLLM server (GPU ${GPU_ID}, port ${PORT})..."
docker run -d \
    --name "$CONTAINER" \
    --rm \
    --gpus "device=${GPU_ID}" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -p "${PORT}:8000" \
    --ipc=host \
    ${EXTRA_DOCKER_ARGS} \
    vllm/vllm-openai:latest \
        --model "$MODEL" \
        --max-model-len "$MAX_CTX" \
        --trust-remote-code \
        --dtype auto \
        --disable-log-requests \
    >> "$LOG" 2>&1

# ── Wait for /health ──────────────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] Waiting for server (model load: 1-5 min)..."
MAX_WAIT=600
ELAPSED=0
until curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo ""
        echo "ERROR: vLLM did not start within ${MAX_WAIT}s. Last 30 lines of log:"
        docker logs "$CONTAINER" 2>&1 | tail -30
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    printf "  ... %ds\r" "$ELAPSED"
done
echo ""
echo "[$(date '+%H:%M:%S')] Server ready."

# ── Run benchmark ─────────────────────────────────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Running benchmark..."
python exp/bench_vllm_client.py \
    --host "localhost" \
    --port "$PORT" \
    --model "$MODEL" \
    --prompt_len "$PROMPT_LEN" \
    --gen_len "$GEN_LEN" \
    --warmup "$WARMUP" \
    --runs "$RUNS" \
    --output_json "$OUT" \
    2>&1 | tee -a "$LOG"

echo ""
echo "[$(date '+%H:%M:%S')] Benchmark complete. Results: $OUT"

# ── Print comparison hint ─────────────────────────────────────────────────────
python3 - "$OUT" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    r = json.load(f)
g = r.get("generate_tok_s_mean", 0)
d = r.get("decode_tok_s_mean", 0)
print("")
print("── Paper table values ──────────────────────────────────")
print(f"  vLLM FP16  (Our Method comparison) : {g:.1f} tok/s")
print(f"  vLLM FP16  (decode-only, ref)      : {d:.1f} tok/s")
print(f"  TTFT                               : {r.get('ttft_ms_mean',0):.0f} ms")
print("  GLoRCQ 'Our Method' from eval_speed.py → Graph column")
print("────────────────────────────────────────────────────────")
PYEOF
