#!/bin/bash
# Step 3/3: run speed test on 3 models.
#
# Prereqs:
#   - vast_1_install.sh done (venv, glorcq, kernels)
#   - vast_2_download.sh done (ckpts/{qwen1.5-moe-a2.7b, mixtral-8x7b, qwen3-30b-a3b})
#
# Usage:
#   bash GLoRCQ/scripts/vast_3_speed.sh
#   # or with specific model only:
#   bash GLoRCQ/scripts/vast_3_speed.sh qwen3-30b-a3b
set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLORCQ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK=${WORK:-/workspace/glorcq_speed}
export VIRTUAL_ENV=$WORK/.venv
export PATH=$HOME/.local/bin:$PATH
PY=$VIRTUAL_ENV/bin/python
cd "$WORK"

if [ ! -x "$PY" ]; then
    echo "ERROR: venv not found. Run vast_1_install.sh first."
    exit 1
fi

# Filter to specific models via CLI args, or all 3 by default
MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=(qwen1.5-moe-a2.7b mixtral-8x7b qwen3-30b-a3b)
fi

# Prevent OOM from residual fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p speed_results
for m in "${MODELS[@]}"; do
    if [ ! -d "ckpts/$m" ]; then
        echo "SKIP $m (no ckpt). Run vast_2_download.sh first."
        continue
    fi
    echo ""
    echo "========== $m =========="
    CUDA_VISIBLE_DEVICES=0 $PY "$GLORCQ_ROOT/inference/eval_speed.py" \
        --model_path "ckpts/$m" --batch_size 1 --prompt_len 128 --gen_len 128 --max_seq_len 512 \
        2>&1 | tee "speed_results/${m}.log"
done

echo ""
echo "===== [$(date)] SUMMARY ====="
grep -H -E "Standard:|Graph:|Speedup:" speed_results/*.log
