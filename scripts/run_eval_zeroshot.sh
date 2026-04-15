#!/usr/bin/env bash
# Zero-shot benchmark on a fake-quant model
# Usage: bash glorcq/scripts/run_eval_zeroshot.sh <model_path> [output_json]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

export HF_HOME=/nvme1/yqy/huggingface

MODEL_PATH=${1:?"Usage: $0 <model_path> [output_json]"}
OUTPUT_JSON=${2:-"${MODEL_PATH}/zeroshot.json"}

uv run python evaluate/eval_zeroshot.py \
    --model_path "$MODEL_PATH" \
    --output_json "$OUTPUT_JSON"
