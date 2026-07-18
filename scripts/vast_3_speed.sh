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
if [ -z "${WORK:-}" ]; then
    if [ -d /workspace ] && [ -w /workspace ]; then WORK=/workspace/glorcq_speed
    else WORK=$HOME/glorcq_speed; fi
fi
export VIRTUAL_ENV=$WORK/.venv
export PATH=$HOME/.local/bin:$PATH
# Overridable knobs (defaults = the paper's Table-2 harness):
PY=${PY:-$VIRTUAL_ENV/bin/python}
CKPT_DIR=${CKPT_DIR:-$WORK/ckpts}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-384}   # 384 fits an 80GB card for all 3 (only 256 tokens needed)
GPU=${CUDA_VISIBLE_DEVICES:-0}
cd "$WORK"

if [ ! -x "$PY" ]; then
    echo "ERROR: python not found at $PY. Run vast_1_install.sh first (or set PY=...)."
    exit 1
fi

# Filter to specific models via CLI args, or all 3 by default
MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=(qwen1.5-moe-a2.7b mixtral-8x7b qwen3-30b-a3b)
fi
# Batch sizes to sweep (override: BATCH_SIZES="1 4" bash ...)
BATCH_SIZES=${BATCH_SIZES:-"1 4 16 64"}

# Prevent OOM from allocator fragmentation (both spellings: torch <=2.7 / >=2.8)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p speed_results
for m in "${MODELS[@]}"; do
    if [ ! -d "$CKPT_DIR/$m" ]; then
        echo "SKIP $m (no ckpt at $CKPT_DIR/$m). Run vast_2_download.sh first."
        continue
    fi
    # Mixtral only: the launch-optimized decode path (_forward_decode, used when
    # tokens N <= 4) has a shape bug with fp16-shim experts, so batch sizes in
    # (1, 4] crash. bs=1 (N=1) and bs>4 (routes to _forward_sparse) are fine.
    # Skip just the 2..4 range for Mixtral; keep 1 and >4.
    _bs_list=""
    for b in $BATCH_SIZES; do
        if [ "$m" = "mixtral-8x7b" ] && [ "$b" -gt 1 ] && [ "$b" -le 4 ]; then
            echo "  [note] mixtral-8x7b: skip bs=$b (decode-path bug at 1<bs<=4; 1 and >4 ok)"
            continue
        fi
        _bs_list="$_bs_list $b"
    done
    for bs in $_bs_list; do
        echo ""
        echo "========== $m  (batch_size=$bs) =========="
        # bs=1 reference (A100-80G): qwen1.5 ~10.4/22.2, mixtral ~8.3/8.7
        # (prints "Mixtral detected -> CUDA Graph disabled" — expected),
        # qwen3 ~2.8/6.4 tok/s. Larger batch raises TOTAL throughput; big
        # batch may OOM (esp. Mixtral) — set +e above lets the sweep continue.
        CUDA_VISIBLE_DEVICES=$GPU $PY "$GLORCQ_ROOT/inference/eval_speed.py" \
            --model_path "$CKPT_DIR/$m" --batch_size "$bs" --prompt_len 128 --gen_len 128 \
            --max_seq_len "$MAX_SEQ_LEN" \
            2>&1 | tee "speed_results/${m}_bs${bs}.log"
    done
done

echo ""
echo "===== [$(date)] SUMMARY ====="
grep -H -E "Standard:|Graph:|Speedup:" speed_results/*.log
