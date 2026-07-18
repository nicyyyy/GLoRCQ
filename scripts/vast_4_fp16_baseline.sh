#!/bin/bash
# Step 4 (optional): FP16 baseline for the speedup claim.
#
# Downloads the 3 ORIGINAL fp16 base models and measures their decode speed
# through plain HF model.generate() — eager, NO CUDA graph, NO custom kernels
# (vLLM/PagedAttention are a different track; this is the same-framework
# denominator that isolates our method, as most weight-quant papers report).
# Then prints the speedup of our real-quant+graph over fp16.
#
# Prereqs: vast_1_install.sh (env) + vast_3_speed.sh done (so speed_results/
#          holds our real-quant Standard/Graph numbers to compare against).
# Needs ~170 GiB extra disk (Qwen1.5 27 + Mixtral 87 + Qwen3 57, safetensors
# only) and a GPU big enough to hold fp16: Mixtral-8x7B fp16 ~87 GiB needs an
# H200-class (>=96 GiB) card on a single GPU. Qwen1.5/Qwen3 fit on 80 GiB.
#
# Usage:
#   bash GLoRCQ/scripts/vast_4_fp16_baseline.sh              # all 3
#   bash GLoRCQ/scripts/vast_4_fp16_baseline.sh qwen1.5-moe-a2.7b
set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLORCQ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -z "${WORK:-}" ]; then
    if [ -d /workspace ] && [ -w /workspace ]; then WORK=/workspace/glorcq_speed
    else WORK=$HOME/glorcq_speed; fi
fi
export VIRTUAL_ENV=$WORK/.venv
export PATH=$HOME/.local/bin:$PATH
export HF_HOME=$WORK/hf_cache
# Public repos: ignore any stale cached token (expired token => bogus 404).
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
PY=${PY:-$VIRTUAL_ENV/bin/python}
GPU=${CUDA_VISIBLE_DEVICES:-0}
cd "$WORK"

if [ ! -x "$PY" ]; then echo "ERROR: venv not found. Run vast_1_install.sh."; exit 1; fi

# model-key -> original fp16 HF repo id
declare -A FP16_REPO=(
  [qwen1.5-moe-a2.7b]="Qwen/Qwen1.5-MoE-A2.7B"
  [mixtral-8x7b]="mistralai/Mixtral-8x7B-v0.1"
  [qwen3-30b-a3b]="Qwen/Qwen3-30B-A3B"
)

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(qwen1.5-moe-a2.7b mixtral-8x7b qwen3-30b-a3b)
# Batch sizes to sweep (override: BATCH_SIZES="1 4" bash ...); matches vast_3
BATCH_SIZES=${BATCH_SIZES:-"1 4 16 64"}

mkdir -p fp16_ckpts fp16_results

for m in "${MODELS[@]}"; do
    repo="${FP16_REPO[$m]}"
    [ -z "$repo" ] && { echo "SKIP unknown model $m"; continue; }
    dst="fp16_ckpts/$m"
    if ! ls "$dst"/*.safetensors >/dev/null 2>&1; then
        echo "===== [$(date)] downloading fp16 $repo (safetensors only) ====="
        # exclude the duplicate consolidated *.pt/*.bin (Mixtral ships ~90 GB of them)
        $VIRTUAL_ENV/bin/huggingface-cli download "$repo" \
            --local-dir "$dst" --max-workers 8 \
            --exclude "*.pt" "*.bin" "*.pth" "consolidated*" 2>&1 | tail -3
    else
        echo "  already have fp16 $m ($(du -sh "$dst" | cut -f1))"
    fi
    for bs in $BATCH_SIZES; do
        echo ""
        echo "========== fp16 baseline: $m  (batch_size=$bs) =========="
        # fp16 model may OOM at big batch (esp. Mixtral 87GB weights + KV);
        # the probe catches OOM and records it, sweep continues.
        CUDA_VISIBLE_DEVICES=$GPU $PY "$GLORCQ_ROOT/scripts/fp16_speed_probe.py" \
            --model_path "$dst" --prompt_len 128 --gen_len 128 --batch_size "$bs" \
            --device cuda:0 --output_json "fp16_results/${m}_bs${bs}.json" \
            2>&1 | tee "fp16_results/${m}_bs${bs}.log"
    done
done

echo ""
echo "===== [$(date)] SPEEDUP vs fp16 (HF eager, same framework) ====="
printf "%-20s %5s %10s %10s %10s %10s\n" model bs fp16_tot ourStd_tot ourGraph_tot "graph/fp16"
for m in "${MODELS[@]}"; do
    for bs in $BATCH_SIZES; do
        # fp16 total tok/s (per-seq x batch); our real-quant std/graph total from vast_3
        fp16=$(grep -oP "total \K[0-9.]+" "fp16_results/${m}_bs${bs}.log" 2>/dev/null | tail -1)
        ostd=$(grep -oP "Standard:\s*\K[0-9.]+" "speed_results/${m}_bs${bs}.log" 2>/dev/null | tail -1)
        ogra=$(grep -oP "Graph:\s*\K[0-9.]+" "speed_results/${m}_bs${bs}.log" 2>/dev/null | tail -1)
        # eval_speed prints per-seq tok/s; convert our numbers to total (x bs)
        ratio=$($PY -c "
try: print(f'{($ogra*$bs)/$fp16:.2f}x')
except Exception: print('n/a')" 2>/dev/null)
        oatot=$($PY -c "
try: print(f'{$ogra*$bs:.1f}')
except Exception: print('?')" 2>/dev/null)
        ostot=$($PY -c "
try: print(f'{$ostd*$bs:.1f}')
except Exception: print('?')" 2>/dev/null)
        printf "%-20s %5s %10s %10s %10s %10s\n" "$m" "$bs" "${fp16:-OOM}" "${ostot:-?}" "${oatot:-?}" "$ratio"
    done
done
echo "Note: fp16 = plain HF generate(), eager, no CUDA graph (isolates the method;"
echo "      not a serving-system comparison vs vLLM/PagedAttention). Numbers are"
echo "      TOTAL tok/s (per-seq x batch). fp16_tot=OOM means that batch didn't fit."
