#!/bin/bash
# Step 2/3: download 3 real-quant checkpoints from HF (parallel).
#
# Prereq: bash vast_1_install.sh done (venv + huggingface-cli available).
#
# Usage:
#   bash GLoRCQ/scripts/vast_2_download.sh
set -e

WORK=${WORK:-/workspace/glorcq_speed}
export VIRTUAL_ENV=$WORK/.venv
export PATH=$HOME/.local/bin:$PATH
export HF_HOME=$WORK/hf_cache
mkdir -p "$HF_HOME" "$WORK/ckpts"
cd "$WORK"

if [ ! -x "$VIRTUAL_ENV/bin/huggingface-cli" ]; then
    echo "ERROR: venv not found at $VIRTUAL_ENV. Run vast_1_install.sh first."
    exit 1
fi

echo "===== [$(date)] Download 3 models in parallel ====="
declare -A DL_PID
for m in qwen1.5-moe-a2.7b mixtral-8x7b qwen3-30b-a3b; do
    if [ -f "ckpts/$m/config.json" ] && ls ckpts/$m/model-*.safetensors >/dev/null 2>&1; then
        echo "  already have $m ($(du -sh ckpts/$m | cut -f1))"
        continue
    fi
    mkdir -p "ckpts/$m"
    echo "  starting $m ..."
    $VIRTUAL_ENV/bin/huggingface-cli download "Tsingyow/GLoRCQ-${m}-real" \
        --local-dir "ckpts/$m" --max-workers 8 > "/tmp/dl_${m}.log" 2>&1 &
    DL_PID[$m]=$!
done

# Wait for all downloads
for m in "${!DL_PID[@]}"; do
    wait "${DL_PID[$m]}" && echo "  ✓ $m done" || echo "  ✗ $m failed (see /tmp/dl_${m}.log)"
done

echo ""
echo "===== [$(date)] Verify each checkpoint ====="
all_ok=1
for m in qwen1.5-moe-a2.7b mixtral-8x7b qwen3-30b-a3b; do
    printf "  %-22s " "$m"
    du -sh "ckpts/$m" 2>/dev/null | cut -f1
    for f in config.json cross_layer_info.pt model.safetensors.index.json; do
        if [ -f "ckpts/$m/$f" ]; then
            printf "    %s ✓\n" "$f"
        else
            printf "    %s ✗ MISSING\n" "$f"
            all_ok=0
        fi
    done
    n=$(ls "ckpts/$m/model-"*.safetensors 2>/dev/null | wc -l)
    printf "    shards=%d\n" $n
    [ "$n" -eq 0 ] && all_ok=0
done

if [ "$all_ok" -eq 0 ]; then
    echo ""
    echo "ERROR: some checkpoints incomplete. Re-run this script to resume."
    echo "See /tmp/dl_*.log for individual download errors."
    exit 1
fi

echo ""
echo "===== [$(date)] STEP 2 DONE ====="
df -h /workspace | tail -1
echo "Next: bash $(dirname $(realpath $0))/vast_3_speed.sh"
