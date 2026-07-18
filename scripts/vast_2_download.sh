#!/bin/bash
# Step 2/3: download the 3 real-quant checkpoints from HF (parallel, ~40 GiB).
#
# Repos are the paper's fair-grassmann artifacts (Qwen1.5 = the r32 canonical,
# same operating point as Table 1). All three are PUBLIC — no HF token needed.
# Checkpoints are self-contained (config + tokenizer + stripped safetensors +
# cross_layer_info.pt): the original fp16 base models are NOT required.
#
# Prereq: bash vast_1_install.sh done (venv + huggingface-cli available).
#
# Usage:
#   bash GLoRCQ/scripts/vast_2_download.sh
set -e

if [ -z "${WORK:-}" ]; then
    if [ -d /workspace ] && [ -w /workspace ]; then WORK=/workspace/glorcq_speed
    else WORK=$HOME/glorcq_speed; fi
fi
export VIRTUAL_ENV=$WORK/.venv
export PATH=$HOME/.local/bin:$PATH
export HF_HOME=$WORK/hf_cache
mkdir -p "$HF_HOME" "$WORK/ckpts"
cd "$WORK"

if [ ! -x "$VIRTUAL_ENV/bin/huggingface-cli" ]; then
    echo "ERROR: venv not found at $VIRTUAL_ENV. Run vast_1_install.sh first."
    exit 1
fi

# Repos are public: ignore any stale token cached on this machine (an expired
# stored token makes HF answer "Repository Not Found" for public repos!).
# An explicitly exported HF_TOKEN is still honored.
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
TOKEN_ARGS=()
[ -n "${HF_TOKEN:-}" ] && TOKEN_ARGS=(--token "$HF_TOKEN")

echo "===== [$(date)] Download 3 models in parallel ====="
declare -A DL_PID
for m in qwen1.5-moe-a2.7b mixtral-8x7b qwen3-30b-a3b; do
    if [ -f "ckpts/$m/config.json" ] && ls ckpts/$m/model-*.safetensors >/dev/null 2>&1; then
        echo "  already have $m ($(du -sh ckpts/$m | cut -f1))"
        continue
    fi
    mkdir -p "ckpts/$m"
    echo "  starting $m ..."
    $VIRTUAL_ENV/bin/huggingface-cli download "Tsingyow/GLoRCQ-${m}-fair-grassmann-real" \
        --local-dir "ckpts/$m" --max-workers 8 "${TOKEN_ARGS[@]}" > "/tmp/dl_${m}.log" 2>&1 &
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
    # tokenizer must ship with the ckpt (Mixtral uses sentencepiece tokenizer.model)
    if [ -f "ckpts/$m/tokenizer.json" ] || [ -f "ckpts/$m/tokenizer.model" ]; then
        printf "    tokenizer ✓\n"
    else
        printf "    tokenizer ✗ MISSING\n"; all_ok=0
    fi
done

if [ "$all_ok" -eq 0 ]; then
    echo ""
    echo "ERROR: some checkpoints incomplete. Re-run this script to resume."
    echo "See /tmp/dl_*.log for individual download errors."
    exit 1
fi

echo ""
echo "===== [$(date)] STEP 2 DONE ====="
df -h "$WORK" | tail -1
echo "Next: bash $(dirname $(realpath $0))/vast_3_speed.sh"
