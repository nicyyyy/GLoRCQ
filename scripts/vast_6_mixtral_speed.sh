#!/bin/bash
# Mixtral-8x7B real-quant decode speed vs fp16 — needs a GPU that fits the 94 GB
# fp16 base (H200 141G / B200). Our real-quant win is the gather-graph + indexed
# VQ4 kernel (GLORCQ_MIXTRAL_GRAPH=1 + GLORCQ_MIXTRAL_IDXKERNEL=1), an OPT-IN path
# gated on num_experts<=16 so Qwen/Qwen3/DeepSeek are byte-identical.
#
# On A100-80G we measured real-quant 8.8 -> 12.7 tok/s = 1.44x over our OWN
# baseline (NOT over fp16 — fp16 Mixtral is 94 GB and won't fit an 80 GB card, so
# there is no clean A100 fp16 number). This script gets the real "vs fp16" ratio
# on a big-GPU machine.
#
# Prereq: vast_1_install.sh (uv .venv + kernels). Reuses that venv + auto-rebuilds
# the CUDA kernel if the vq4-indexed symbol is missing (matching-CUDA nvcc auto-
# detected in /usr/local + conda).
#
# Mixtral fp16 base (mistralai/Mixtral-8x7B-v0.1) is GATED on HF: accept its
# license + export a token with access:  HF_TOKEN=hf_xxx bash .../vast_6_mixtral_speed.sh
# Skip the fp16 baseline (only time the real-quant configs) with SKIP_FP16=1.
#
# Usage:
#   HF_TOKEN=hf_xxx bash GLoRCQ/scripts/vast_6_mixtral_speed.sh
#   CONFIGS="A C" ... (A=baseline real-quant, B=inline-LoRA, C=gather+idx, FP16)
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
PROMPT=${PROMPT:-128}; GEN=${GEN:-128}
# Which configs: FP16 baseline + C (our best, incl. prefill-dequant) at gen=128
# AND gen=512 (prefill amortizes => the decode advantage shows). A/B = ablation.
CONFIGS=${CONFIGS:-"FP16 C FP16_G512 C_G512"}
mkdir -p "$HF_HOME" "$CKPT_DIR" "$WORK/speed_results"
cd "$WORK"
if [ ! -x "$PY" ]; then echo "ERROR: python not at $PY. Run vast_1_install.sh first."; exit 1; fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
TOKEN_ARGS=(); [ -n "${HF_TOKEN:-}" ] && TOKEN_ARGS=(--token "$HF_TOKEN")

# Repos (overridable). REAL default = the public g64 fair-grassmann artifact;
# decode SPEED is config-independent (expert dims identical to g8), so this
# measures the gather+idx speedup faithfully. Point REAL_REPO at a g8 repo if
# you upload one.
REAL_REPO=${REAL_REPO:-Tsingyow/GLoRCQ-mixtral-8x7b-fair-grassmann-real}
BASE_REPO=${BASE_REPO:-mistralai/Mixtral-8x7B-v0.1}
REAL_DIR=$CKPT_DIR/mixtral-real
BASE_DIR=$CKPT_DIR/mixtral-fp16

# ── Kernel freshness: vq4-indexed symbol; rebuild with a matching-CUDA nvcc ──
echo "===== [$(date)] Check CUDA kernel (vq4-indexed) ====="
HAS_IDX=$("$PY" - <<'PYEOF' 2>/dev/null
try:
    import inspect
    from inference import kernels as k
    ok = hasattr(getattr(k, "_vq4_cuda_ext", None), "vq4_dequant_grouped_gemv_indexed")
    # also require the r6 gptq kernel signature (reorder_ok) — an older .so has
    # the vq4 symbol but a 7-arg gptq op, which would silently fall back to python
    gp = getattr(k, "_gptq_cuda_ext", None)
    ok = ok and gp is not None and "reorder_ok" in (gp.gptq_dequant_matmul.__doc__ or "")
    print(1 if ok else 0)
except Exception:
    print(0)
PYEOF
)
if [ "$HAS_IDX" != "1" ]; then
    echo "  vq4-indexed kernel MISSING — clean rebuild with matching CUDA nvcc"
    TCUDA=$("$PY" -c "import torch;print(torch.version.cuda or '')" 2>/dev/null)
    NVCC_MATCH=""
    for n in $(which -a nvcc 2>/dev/null) "/usr/local/cuda-$TCUDA/bin/nvcc" \
             /opt/conda/bin/nvcc /opt/conda/envs/*/bin/nvcc /venv/*/bin/nvcc \
             "$HOME"/miniconda3/bin/nvcc "$HOME"/miniconda3/envs/*/bin/nvcc \
             "$HOME"/anaconda3/bin/nvcc "$HOME"/anaconda3/envs/*/bin/nvcc; do
        [ -x "$n" ] || continue
        v=$("$n" --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+')
        if [ -n "$TCUDA" ] && [ "$v" = "$TCUDA" ]; then NVCC_MATCH="$n"; break; fi
    done
    ( cd "$GLORCQ_ROOT/inference/kernels" && rm -rf build ./*.so
      if [ -n "$NVCC_MATCH" ]; then
          echo "  using matching CUDA $TCUDA nvcc at $NVCC_MATCH"
          export CUDA_HOME="$(dirname "$(dirname "$NVCC_MATCH")")" PATH="$(dirname "$NVCC_MATCH"):$PATH"
      fi
      "$PY" setup.py build_ext --inplace 2>&1 | tail -10 )
else
    echo "  vq4-indexed kernel present ✓"
fi

# ── Download ──
echo ""
echo "===== [$(date)] Download Mixtral real-quant + fp16 base ====="
# Disk precheck: fp16 base ~94GB + real 21GB (+ transient hub/xet cache). The
# cache is cleaned after each download below, but you still need the headline
# space. Bail early instead of dying mid-download with a full disk.
FREE_GB=$(df -BG --output=avail "$WORK" | tail -1 | tr -dc '0-9')
_want_fp16_pre=0; for c in $CONFIGS; do case "$c" in FP16*) _want_fp16_pre=1;; esac; done
NEED_GB=30; [ "$_want_fp16_pre" = "1" ] && [ "${SKIP_FP16:-0}" != "1" ] && NEED_GB=130
# already-downloaded base counts toward the need
[ -d "$BASE_DIR" ] && NEED_GB=$((NEED_GB - $(du -sBG "$BASE_DIR" 2>/dev/null | tr -dc '0-9' || echo 0)))
if [ -n "$FREE_GB" ] && [ "$FREE_GB" -lt "$NEED_GB" ]; then
    echo "ERROR: only ${FREE_GB}GB free at $WORK but ~${NEED_GB}GB needed"
    echo "       (fp16 Mixtral is ~94GB). Use a bigger-disk instance, free space,"
    echo "       or run with SKIP_FP16=1 (real-quant only)."
    exit 1
fi
if [ -f "$REAL_DIR/cross_layer_info.pt" ]; then
    echo "  real-quant present ($(du -sh "$REAL_DIR" 2>/dev/null | cut -f1))"
else
    echo "  downloading real-quant $REAL_REPO ..."
    "$VIRTUAL_ENV/bin/huggingface-cli" download "$REAL_REPO" --local-dir "$REAL_DIR" --max-workers 8 "${TOKEN_ARGS[@]}" 2>&1 | tail -3
fi
WANT_FP16=0; for c in $CONFIGS; do [ "$c" = "FP16" ] && WANT_FP16=1; done
if [ "$WANT_FP16" = "1" ] && [ "${SKIP_FP16:-0}" != "1" ]; then
    if [ -f "$BASE_DIR/config.json" ] && ls "$BASE_DIR"/*.safetensors >/dev/null 2>&1; then
        echo "  fp16 base present ($(du -sh "$BASE_DIR" 2>/dev/null | cut -f1))"
    else
        echo "  downloading fp16 base $BASE_REPO (~94 GB, GATED — needs HF_TOKEN + accepted license) ..."
        "$VIRTUAL_ENV/bin/huggingface-cli" download "$BASE_REPO" --local-dir "$BASE_DIR" --max-workers 8 "${TOKEN_ARGS[@]}" 2>&1 | tail -3
    fi
fi
# hf download to --local-dir ALSO fills the hub/xet cache with the same bytes —
# on a tight disk that doubles the footprint. The local dirs are the source of
# truth for the bench; drop the transient cache.
rm -rf "$HF_HOME/xet" "$HF_HOME/hub" 2>/dev/null || true

# ── Bench ──
echo ""
echo "===== [$(date)] Bench (bs1 prompt=$PROMPT gen=$GEN gpu=$GPU) ====="
ES="$GLORCQ_ROOT/inference/eval_speed.py"
COMMON="--batch_size 1 --prompt_len $PROMPT --gen_len $GEN --max_seq_len $MAX_SEQ_LEN"
for cfg in $CONFIGS; do
    LOG="$WORK/speed_results/mixtral_$cfg.log"
    echo ""; echo "========== Mixtral  config=$cfg =========="
    case "$cfg" in
      FP16)  [ "${SKIP_FP16:-0}" = "1" ] && { echo "(SKIP_FP16=1)"; continue; }
             CUDA_VISIBLE_DEVICES=$GPU "$PY" "$ES" --model_path "$BASE_DIR" --no_real_quant $COMMON 2>&1 | tee "$LOG" ;;
      A)     CUDA_VISIBLE_DEVICES=$GPU GLORCQ_MIXTRAL_INLINE_LORA=0 "$PY" "$ES" --model_path "$REAL_DIR" $COMMON 2>&1 | tee "$LOG" ;;
      B)     CUDA_VISIBLE_DEVICES=$GPU GLORCQ_MIXTRAL_INLINE_LORA=1 "$PY" "$ES" --model_path "$REAL_DIR" $COMMON 2>&1 | tee "$LOG" ;;
      C)     CUDA_VISIBLE_DEVICES=$GPU GLORCQ_MIXTRAL_GRAPH=1 GLORCQ_MIXTRAL_IDXKERNEL=1 \
             GLORCQ_PREFILL_DEQUANT=1 GLORCQ_GPTQ_ILP=1 "$PY" "$ES" --model_path "$REAL_DIR" $COMMON 2>&1 | tee "$LOG" ;;
      # long-gen variants: prefill amortizes over more tokens => decode advantage shows
      FP16_G512) [ "${SKIP_FP16:-0}" = "1" ] && continue
             CUDA_VISIBLE_DEVICES=$GPU "$PY" "$ES" --model_path "$BASE_DIR" --no_real_quant \
             --batch_size 1 --prompt_len $PROMPT --gen_len 512 --max_seq_len 768 2>&1 | tee "$LOG" ;;
      C_G512) CUDA_VISIBLE_DEVICES=$GPU GLORCQ_MIXTRAL_GRAPH=1 GLORCQ_MIXTRAL_IDXKERNEL=1 \
             GLORCQ_PREFILL_DEQUANT=1 GLORCQ_GPTQ_ILP=1 "$PY" "$ES" --model_path "$REAL_DIR" \
             --batch_size 1 --prompt_len $PROMPT --gen_len 512 --max_seq_len 768 2>&1 | tee "$LOG" ;;
    esac
done

echo ""
echo "===== [$(date)] SUMMARY (tok/s) ====="
for cfg in $CONFIGS; do echo "-- $cfg --"; grep -hE "Standard:|Graph:|Speedup:" "$WORK/speed_results/mixtral_$cfg.log" 2>/dev/null; done
echo ""
echo "Read: FP16 'Standard' = fp16 eager baseline; C 'Graph' = our gather+indexed real-quant."
echo "A100 ref (no fp16, OOM): A(baseline) 8.8 -> C(gather+idx) 12.7 = 1.44x self."
echo "VAST6_MIXTRAL_DONE"
