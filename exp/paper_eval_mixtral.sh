#!/bin/bash
# ============================================================
#  GLoRCQ Paper Evaluation — Mixtral-8x7B-v0.1
# ============================================================
#
# Full chain: quantize → eval PPL → zero-shot → MMLU → speed
#
#   Step 0a: Fake-quant (SOTA config)        (~3 h, GPU ≥24 GB)
#   Step 0b: Real-quant (for speed test)     (~3 h, GPU ≥24 GB)
#   Step 1:  WikiText-2 PPL                  (~20 min)
#   Step 2:  Zero-shot 5 tasks (0-shot)      (~60 min)
#   Step 3:  MMLU (5-shot)                   (~90 min)
#   Step 4:  GLoRCQ inference speed          (~5 min)
#   Step 5:  vLLM FP16 baseline speed        (~10 min, Docker required)
#
# Quantization is skipped automatically if the output directory
# already contains a completed model (config.json / glorcq_model.pt).
#
# GPU memory requirements:
#   Steps 0a/0b (quantize): Mixtral FP16 ≈ 94 GB → requires H200/B200
#                            (A100 80G OOM)
#   Steps 1-3 (fake-quant inference): 2-bit ≈ 22 GB → A100 80G OK
#   Step 4 (real-quant speed test): same as above
#   Step 5 (vLLM FP16): Mixtral FP16 ≈ 94 GB → needs H200/B200 or 2×A100
#     NOTE: modify bench_vllm.sh to add --tensor-parallel-size 2 for 2-GPU setup
#
# SOTA config (Mixtral, PPL=?):
#   rank=32, n_iter=5, G_moe=32, G_attn=32
#   uv_bits=8, sv_bits=8, n_lora_iter=2
#   use_turboquant, hessian_svd, search_act_alpha
#   (no rank_down/rank_attn, no w_clip, no recon_weight)
#
# Usage:
#   bash exp/paper_eval_mixtral.sh <output_dir> [gpu_id]
#
# Arguments:
#   output_dir  Base directory for all outputs.
#               fake-quant model  → <output_dir>/fake_quant/
#               real-quant model  → <output_dir>/real_quant/
#               eval results      → <output_dir>/logs/
#   gpu_id      CUDA device index (default: 0)
#
# Example:
#   bash exp/paper_eval_mixtral.sh \
#       /home/qyyang/resource_dir/GLoRCQ_out/mixtral_paper 0
#
# Requirements:
#   - pip install -e .   (or: uv sync)
#   - For Step 4: bash scripts/build_kernels.sh
#   - For Step 5: Docker + NVIDIA Container Toolkit

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/.."

HF_MODEL="mistralai/Mixtral-8x7B-v0.1"
BASE_DIR="${1:?Usage: $0 <output_dir> [gpu_id]}"
GPU_ID="${2:-0}"
DEVICE="cuda:${GPU_ID}"
FAKE_QUANT="${BASE_DIR}/fake_quant"
REAL_QUANT="${BASE_DIR}/real_quant"
EVAL_OUT="${BASE_DIR}/logs"
mkdir -p "$FAKE_QUANT" "$REAL_QUANT" "$EVAL_OUT"

echo "============================================================"
echo "  GLoRCQ Paper Eval — Mixtral-8x7B-v0.1"
echo "  HF model   : $HF_MODEL"
echo "  fake_quant : $FAKE_QUANT"
echo "  real_quant : $REAL_QUANT"
echo "  eval logs  : $EVAL_OUT"
echo "  GPU        : cuda:${GPU_ID}"
echo "============================================================"

# ── Step 0a: Fake-quant ───────────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Step 0a/7 — Fake-quant (SOTA config)"
echo "  NOTE: Mixtral FP16 ≈ 94 GB. Requires H200 141G or B200 192G."
if [ -f "${FAKE_QUANT}/config.json" ]; then
    echo "  Skipped: ${FAKE_QUANT}/config.json already exists"
else
    CUDA_VISIBLE_DEVICES=${GPU_ID} python run_quantize.py \
        --model_path "$HF_MODEL" \
        --output_path "$FAKE_QUANT" \
        --qbit 2 --groupsize 128 --nsamples 128 \
        --rank 32 \
        --n_iter 5 --n_lora_iter 2 \
        --G_moe 32 --G_attn 32 \
        --uv_bits 8 --sv_bits 8 \
        --use_turboquant --hessian_svd --search_act_alpha \
        2>&1 | tee "${EVAL_OUT}/quant_fake.log"
    echo "[$(date '+%H:%M:%S')] Step 0a done"
fi

# ── Step 0b: Real-quant ───────────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Step 0b/7 — Real-quant (for speed test)"
if [ -f "${REAL_QUANT}/glorcq_model.pt" ]; then
    echo "  Skipped: ${REAL_QUANT}/glorcq_model.pt already exists"
else
    CUDA_VISIBLE_DEVICES=${GPU_ID} python run_quantize.py \
        --model_path "$HF_MODEL" \
        --output_path "$REAL_QUANT" \
        --qbit 2 --groupsize 128 --nsamples 128 \
        --rank 32 \
        --n_iter 5 --n_lora_iter 2 \
        --G_moe 32 --G_attn 32 \
        --uv_bits 8 --sv_bits 8 \
        --use_turboquant --hessian_svd --search_act_alpha \
        --real_quant \
        2>&1 | tee "${EVAL_OUT}/quant_real.log"
    echo "[$(date '+%H:%M:%S')] Step 0b done"
fi

# ── Step 1: PPL ──────────────────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Step 1/7 — WikiText-2 PPL"
python evaluate/eval_ppl.py \
    --model_path "$FAKE_QUANT" \
    --device "$DEVICE" \
    --max_length 2048 --stride 512 \
    --output_json "${EVAL_OUT}/ppl.json"

# ── Step 2: Zero-shot 5 tasks ────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Step 2/7 — Zero-shot (ARC-C/E, WinoGrande, HellaSwag, PIQA)"
python evaluate/eval_zeroshot.py \
    --model_path "$FAKE_QUANT" \
    --device "$DEVICE" \
    --tasks arc_challenge,arc_easy,winogrande,hellaswag,piqa \
    --num_fewshot 0 --batch_size 1 \
    --output_json "${EVAL_OUT}/zeroshot_5task.json"

# ── Step 3: MMLU 5-shot ──────────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Step 3/7 — MMLU (5-shot)"
python evaluate/eval_zeroshot.py \
    --model_path "$FAKE_QUANT" \
    --device "$DEVICE" \
    --tasks mmlu \
    --num_fewshot 5 --batch_size 1 \
    --output_json "${EVAL_OUT}/zeroshot_mmlu.json"

# ── Step 4: GLoRCQ inference speed ───────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Step 4/7 — GLoRCQ inference speed"
python evaluate/eval_speed.py \
    --model_path "$REAL_QUANT" \
    --hf_model_path "$HF_MODEL" \
    --device "$DEVICE" \
    --prompt_len 128 --gen_len 128 \
    --num_warmup 2 --num_runs 5 \
    --output_json "${EVAL_OUT}/speed_glorcq.json"

# ── Step 5: vLLM FP16 baseline ───────────────────────────────
# NOTE: Mixtral FP16 ≈ 94 GB. Needs H200/B200, or 2×A100 with tensor parallelism.
echo ""
echo "[$(date '+%H:%M:%S')] Step 5/7 — vLLM FP16 baseline speed"
echo "  WARNING: Mixtral FP16 ≈ 94 GB. Ensure sufficient GPU memory."
if ! command -v docker &> /dev/null; then
    echo "  Skipped (Docker not found)"
else
    bash exp/bench_vllm.sh "$HF_MODEL" "$GPU_ID" 8021 \
        2>&1 | tee -a "${EVAL_OUT}/vllm_speed.log"
    cp "logs/vllm_bench_mixtral-8x7b-v0.1.json" \
        "${EVAL_OUT}/speed_vllm.json" 2>/dev/null || true
fi

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Writing summary..."
python evaluate/print_eval_summary.py \
    --output_dir "$EVAL_OUT" \
    --model_path "$FAKE_QUANT"

echo ""
echo "============================================================"
echo "  Done. Results in: $EVAL_OUT"
echo "  Files: ppl.json | zeroshot_5task.json | zeroshot_mmlu.json"
echo "         speed_glorcq.json | speed_vllm.json | summary.txt"
echo "============================================================"
