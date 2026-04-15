#!/usr/bin/env bash
# Speed benchmark on a real-quant model
# Usage: bash glorcq/scripts/run_eval_speed.sh <model_path> [output_json]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

export HF_HOME=/nvme1/yqy/huggingface

MODEL_PATH=${1:?"Usage: $0 <model_path> [output_json]"}
OUTPUT_JSON=${2:-"${MODEL_PATH}/speed.json"}

uv run python evaluate/eval_speed.py \
    --model_path "$MODEL_PATH" \
    --prompt_len 128 --gen_len 128 \
    --output_json "$OUTPUT_JSON"
