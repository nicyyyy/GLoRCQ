#!/bin/bash
# ===========================================================================
# Eval DeepSeek-MoE-16B CLASP fake-quant: WikiText-2 PPL + 5-task zero-shot.
# CANONICAL Table-1 protocol (matches the CLASP paper main table):
#   acc (NOT acc_norm), batch=1, add_bos=True, 0-shot,
#   tasks = ARC-C / ARC-E / PIQA / WinoGrande / HellaSwag.
# Also runs the fp16 baseline PPL so the quantization gap is on the same harness.
#
# NOTE vs TileQ paper: TileQ reports 6 tasks incl. MMLU with a different harness;
#   for apples-to-apples use OUR 5-task acc numbers here, and cite TileQ's DS-16B
#   numbers (below) only as an external reference (different harness/metric).
#
# NO-GPU-SAFE: LAUNCHES GPU JOBS. Run ONLY when GPU4 free, in tmux `test:0`.
#   Requires exp/deepseek_moe16b_quant.sh to have produced the fake dir.
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

FAKE=/mnt/Data/yqy/resource_dir/glorcq_paper_exp/deepseek_moe16b_fake
FP16=/mnt/Data/yqy/resource_dir/deepseek-moe-16b
OUT=logs/deepseek_moe16b
mkdir -p "$OUT"
export HF_HOME=/mnt/Data/yqy/resource_dir/hf_cache
GPU=${CUDA_VISIBLE_DEVICES:-4}
TASKS=arc_challenge,arc_easy,winogrande,hellaswag,piqa

echo "[$(date)] === PPL (fake-quant) ==="
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python evaluate/eval_ppl.py --model_path "$FAKE" --device cuda:0 \
    --max_length 2048 --stride 512 --output_json "$OUT/ppl_fake.json" 2>&1 | tee "$OUT/eval_ppl_fake.log"

echo "[$(date)] === PPL (fp16 baseline) ==="
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python evaluate/eval_ppl.py --model_path "$FP16" --device cuda:0 \
    --max_length 2048 --stride 512 --output_json "$OUT/ppl_fp16.json" 2>&1 | tee "$OUT/eval_ppl_fp16.log"

echo "[$(date)] === Zero-shot 5-task (fake-quant, canonical acc/add_bos/bs1) ==="
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python evaluate/eval_zeroshot.py --model_path "$FAKE" --device cuda:0 \
    --tasks $TASKS --num_fewshot 0 --batch_size 1 --metric_mode acc --add_bos \
    --output_json "$OUT/zeroshot_fake.json" 2>&1 | tee "$OUT/eval_zs_fake.log"

echo "[$(date)] === Zero-shot 5-task (fp16 baseline, canonical acc/add_bos/bs1) ==="
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python evaluate/eval_zeroshot.py --model_path "$FP16" --device cuda:0 \
    --tasks $TASKS --num_fewshot 0 --batch_size 1 --metric_mode acc --add_bos \
    --output_json "$OUT/zeroshot_fp16.json" 2>&1 | tee "$OUT/eval_zs_fp16.log"

echo "[$(date)] MOE16B_EVAL_DONE"
