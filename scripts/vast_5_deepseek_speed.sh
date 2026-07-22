#!/bin/bash
# Step 5 (optional add-on): DeepSeek-MoE-16B real-quant vs fp16 decode speed.
#
# Separate from vast_3 (which covers qwen1.5 / mixtral / qwen3) because DeepSeek
# uses its own stable harness exp/bench_deepseek_stable.py and its own gated
# fast paths (gather-graph + vq4-indexed kernel, DeepSeek-only, Qwen/Mixtral
# byte-identical).
#
# Prereq: vast_1_install.sh done (uv .venv + glorcq + kernels). This script
#   reuses the SAME uv venv ($WORK/.venv/bin/python) and auto-rebuilds the CUDA
#   kernel if the compiled .so predates the vq4-indexed kernel (new in a6bb8d5).
#
# Downloads (public, no token):
#   - config-B real-quant  Tsingyow/GLoRCQ-deepseek-moe-16b-real  (~7.4 GB)
#   - fp16 base            deepseek-ai/deepseek-moe-16b-base       (~31 GB; needs
#                          a ~35 GB+ GPU to load for the fp16 baseline)
#
# Usage:
#   bash GLoRCQ/scripts/vast_5_deepseek_speed.sh
#   # skip the 31 GB fp16 baseline (only time the real-quant configs):
#   SKIP_FP16=1 bash GLoRCQ/scripts/vast_5_deepseek_speed.sh
#   # full ablation (all four configs):
#   CONFIGS="eager moe_graph full_graph fp16" bash GLoRCQ/scripts/vast_5_deepseek_speed.sh
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
PY=${PY:-$VIRTUAL_ENV/bin/python}
CKPT_DIR=${CKPT_DIR:-$WORK/ckpts}
GPU=${CUDA_VISIBLE_DEVICES:-0}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-384}
PROMPT=${PROMPT:-128}; GEN=${GEN:-128}; WARMUP=${WARMUP:-2}; RUNS=${RUNS:-3}
# Default = the two the user asked for. Override CONFIGS=... for the full ablation.
CONFIGS=${CONFIGS:-"fp16 full_graph"}
mkdir -p "$HF_HOME" "$CKPT_DIR" "$WORK/speed_results"
cd "$WORK"

if [ ! -x "$PY" ]; then
    echo "ERROR: python not found at $PY. Run vast_1_install.sh first (or set PY=...)."
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1     # repos are public; ignore stale cached token
TOKEN_ARGS=(); [ -n "${HF_TOKEN:-}" ] && TOKEN_ARGS=(--token "$HF_TOKEN")

REAL_REPO=Tsingyow/GLoRCQ-deepseek-moe-16b-real
BASE_REPO=deepseek-ai/deepseek-moe-16b-base
REAL_DIR=$CKPT_DIR/deepseek-moe-16b-real
BASE_DIR=$CKPT_DIR/deepseek-moe-16b-base

# ── Kernel freshness: the vq4-indexed kernel is new; a stale .so from an earlier
#    vast_1 run would silently lack it. Auto-rebuild with the SAME venv python. ──
echo "===== [$(date)] Check CUDA kernel (vq4-indexed) ====="
HAS_IDX=$("$PY" - <<'PYEOF' 2>/dev/null
try:
    from inference import kernels as k
    print(1 if hasattr(getattr(k, "_vq4_cuda_ext", None), "vq4_dequant_grouped_gemv_indexed") else 0)
except Exception:
    print(0)
PYEOF
)
if [ "$HAS_IDX" != "1" ]; then
    echo "  vq4-indexed kernel MISSING from built .so — FORCE clean rebuild (nuke stale build cache)"
    # torch is pinned to a CUDA version (e.g. cu128) that may differ from the
    # system default nvcc (e.g. 13.0) — build_ext then errors on version mismatch.
    # If a matching /usr/local/cuda-<torch.version.cuda> toolkit exists, use its nvcc.
    TCUDA=$("$PY" -c "import torch; print(torch.version.cuda or '')" 2>/dev/null)
    # Search /usr/local AND conda envs for an nvcc whose release == torch's CUDA.
    NVCC_MATCH=""
    for n in $(which -a nvcc 2>/dev/null) "/usr/local/cuda-$TCUDA/bin/nvcc" \
             /opt/conda/bin/nvcc /opt/conda/envs/*/bin/nvcc /venv/*/bin/nvcc \
             "$HOME"/miniconda3/bin/nvcc "$HOME"/miniconda3/envs/*/bin/nvcc \
             "$HOME"/anaconda3/bin/nvcc "$HOME"/anaconda3/envs/*/bin/nvcc; do
        [ -x "$n" ] || continue
        v=$("$n" --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+')
        if [ -n "$TCUDA" ] && [ "$v" = "$TCUDA" ]; then NVCC_MATCH="$n"; break; fi
    done
    (
        cd "$GLORCQ_ROOT/inference/kernels" && rm -rf build ./*.so
        if [ -n "$NVCC_MATCH" ]; then
            echo "  using matching CUDA $TCUDA nvcc at $NVCC_MATCH"
            export CUDA_HOME="$(dirname "$(dirname "$NVCC_MATCH")")" PATH="$(dirname "$NVCC_MATCH"):$PATH"
        fi
        "$PY" setup.py build_ext --inplace 2>&1 | tail -12
    )
    HAS_IDX=$("$PY" - <<'PYEOF' 2>/dev/null
try:
    from inference import kernels as k
    print(1 if hasattr(getattr(k, "_vq4_cuda_ext", None), "vq4_dequant_grouped_gemv_indexed") else 0)
except Exception:
    print(0)
PYEOF
)
    if [ "$HAS_IDX" = "1" ]; then
        echo "  rebuild OK — vq4-indexed kernel now present ✓"
    else
        echo "  NOTE: could not build the vq4-indexed kernel (e.g. CUDA toolkit vs torch"
        echo "        version mismatch). Falling back to the non-indexed gather path"
        echo "        (GLORCQ_DEEPSEEK_IDXKERNEL=0) — valid gather-graph number, ~a few % below peak."
    fi
else
    echo "  vq4-indexed kernel present ✓"
fi
# idx kernel on only if the symbol is actually available; else non-indexed gather fallback
IDXVAL=$([ "$HAS_IDX" = "1" ] && echo 1 || echo 0)

# ── Download checkpoints ──
echo ""
echo "===== [$(date)] Download DeepSeek checkpoints ====="
if [ -f "$REAL_DIR/cross_layer_info.pt" ] && [ -f "$REAL_DIR/config.json" ]; then
    echo "  already have real-quant ($(du -sh "$REAL_DIR" 2>/dev/null | cut -f1))"
else
    echo "  downloading real-quant $REAL_REPO ..."
    "$VIRTUAL_ENV/bin/huggingface-cli" download "$REAL_REPO" --local-dir "$REAL_DIR" \
        --max-workers 8 "${TOKEN_ARGS[@]}" 2>&1 | tail -4
fi
# verify real-quant essentials
for f in config.json cross_layer_info.pt modeling_deepseek.py; do
    [ -f "$REAL_DIR/$f" ] && echo "    $f ✓" || { echo "    $f ✗ MISSING — real-quant configs will fail"; }
done

WANT_FP16=0
for c in $CONFIGS; do [ "$c" = "fp16" ] && WANT_FP16=1; done
if [ "$WANT_FP16" = "1" ] && [ "${SKIP_FP16:-0}" != "1" ]; then
    if [ -f "$BASE_DIR/config.json" ] && ls "$BASE_DIR"/model-*.safetensors >/dev/null 2>&1; then
        echo "  already have fp16 base ($(du -sh "$BASE_DIR" 2>/dev/null | cut -f1))"
    else
        echo "  downloading fp16 base $BASE_REPO (~31 GB) ..."
        "$VIRTUAL_ENV/bin/huggingface-cli" download "$BASE_REPO" --local-dir "$BASE_DIR" \
            --max-workers 8 "${TOKEN_ARGS[@]}" 2>&1 | tail -4
    fi
fi

# ── Run benchmarks ──
echo ""
echo "===== [$(date)] Bench (prompt=$PROMPT gen=$GEN warmup=$WARMUP runs=$RUNS gpu=$GPU) ====="
for cfg in $CONFIGS; do
    LOG="$WORK/speed_results/deepseek_${cfg}.log"
    echo ""
    echo "========== DeepSeek-MoE-16B  config=$cfg =========="
    if [ "$cfg" = "fp16" ]; then
        if [ "${SKIP_FP16:-0}" = "1" ]; then echo "  (SKIP_FP16=1, skipping)"; continue; fi
        CUDA_VISIBLE_DEVICES=$GPU "$PY" "$GLORCQ_ROOT/exp/bench_deepseek_stable.py" \
            --config fp16 --fp16_path "$BASE_DIR" \
            --prompt_len "$PROMPT" --gen_len "$GEN" --max_seq_len "$MAX_SEQ_LEN" \
            --warmup "$WARMUP" --runs "$RUNS" 2>&1 | tee "$LOG"
    elif [ "$cfg" = "full_graph" ]; then
        # best real-quant path: gather-graph + vq4-indexed kernel (IDXVAL=0 => non-indexed fallback)
        echo "  (GLORCQ_DEEPSEEK_GATHER=1 GLORCQ_DEEPSEEK_IDXKERNEL=$IDXVAL)"
        CUDA_VISIBLE_DEVICES=$GPU GLORCQ_DEEPSEEK_GATHER=1 GLORCQ_DEEPSEEK_IDXKERNEL=$IDXVAL \
            "$PY" "$GLORCQ_ROOT/exp/bench_deepseek_stable.py" \
            --config full_graph --real_path "$REAL_DIR" \
            --prompt_len "$PROMPT" --gen_len "$GEN" --max_seq_len "$MAX_SEQ_LEN" \
            --warmup "$WARMUP" --runs "$RUNS" 2>&1 | tee "$LOG"
    else   # eager | moe_graph (real-quant; idx-kernel if available, else fallback)
        CUDA_VISIBLE_DEVICES=$GPU GLORCQ_DEEPSEEK_IDXKERNEL=$IDXVAL \
            "$PY" "$GLORCQ_ROOT/exp/bench_deepseek_stable.py" \
            --config "$cfg" --real_path "$REAL_DIR" \
            --prompt_len "$PROMPT" --gen_len "$GEN" --max_seq_len "$MAX_SEQ_LEN" \
            --warmup "$WARMUP" --runs "$RUNS" 2>&1 | tee "$LOG"
    fi
done

echo ""
echo "===== [$(date)] SUMMARY (tok/s) ====="
grep -h -E "^RESULT config=" "$WORK"/speed_results/deepseek_*.log 2>/dev/null
echo ""
echo "Reference (A100-80G, session-dependent): fp16 ~17-19, full_graph+gather+idx ~25 = ~1.34-1.40x."
echo "VAST5_DEEPSEEK_DONE"
