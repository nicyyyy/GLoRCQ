#!/bin/bash
# After g8 real-quant: (1) Qwen byte-identical regression (Mixtral gates must NOT
# leak into Qwen), (2) Mixtral g8-real decode A/B/C to measure #2 (inline LoRA)
# and #1 (gather + indexed kernel). GPU4/test:0.
set -uo pipefail
cd "$(dirname "$0")/.."
VENV=/home/qyyang/repo/GLoRCQ/.venv/bin/python
G8=/mnt/Data/yqy/resource_dir/glorcq_paper_exp/mixtral_g8_real
QWEN=/mnt/Data/yqy/resource_dir/glorcq_paper_exp/qwen15_r32_real
LOG=/home/qyyang/repo/GLoRCQ/logs/mixtral_gsweep
export HF_HOME=/mnt/Data/yqy/resource_dir/hf_cache
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "===== [$(date)] Qwen byte-identical regression (Mixtral gates must not leak) ====="
$VENV exp/qwen_regression_check.py "$QWEN" 2>&1 | tee "$LOG/qwen_regr_default.log"
GLORCQ_MIXTRAL_INLINE_LORA=1 GLORCQ_MIXTRAL_GRAPH=1 GLORCQ_MIXTRAL_IDXKERNEL=1 \
    $VENV exp/qwen_regression_check.py "$QWEN" 2>&1 | tee "$LOG/qwen_regr_mixenv.log"
echo "--- Qwen QWEN_GEN_IDS must be IDENTICAL across the two (proves gates don't leak) ---"
grep QWEN_GEN_IDS "$LOG/qwen_regr_default.log" "$LOG/qwen_regr_mixenv.log"

echo ""
echo "===== [$(date)] Mixtral g8-real decode A/B/C (bs1 / prompt128 / gen128) ====="
echo "--- A: baseline (inline LoRA OFF, no graph) = old standard ---"
GLORCQ_MIXTRAL_INLINE_LORA=0 \
    $VENV inference/eval_speed.py --model_path "$G8" --batch_size 1 \
    --prompt_len 128 --gen_len 128 --max_seq_len 384 2>&1 | tee "$LOG/g8_speed_A.log"
echo "--- B: #2 inline LoRA ON (no graph) ---"
GLORCQ_MIXTRAL_INLINE_LORA=1 \
    $VENV inference/eval_speed.py --model_path "$G8" --batch_size 1 \
    --prompt_len 128 --gen_len 128 --max_seq_len 384 2>&1 | tee "$LOG/g8_speed_B.log"
echo "--- C: #1+#2 gather-graph + indexed kernel ---"
GLORCQ_MIXTRAL_INLINE_LORA=1 GLORCQ_MIXTRAL_GRAPH=1 GLORCQ_MIXTRAL_IDXKERNEL=1 \
    $VENV inference/eval_speed.py --model_path "$G8" --batch_size 1 \
    --prompt_len 128 --gen_len 128 --max_seq_len 384 2>&1 | tee "$LOG/g8_speed_C.log"

echo ""
echo "===== [$(date)] SUMMARY (tok/s) ====="
for c in A B C; do echo "-- config $c --"; grep -hE "Standard:|Graph:|Speedup:" "$LOG/g8_speed_$c.log" 2>/dev/null; done
echo "MIXTRAL_G8_SPEEDTEST_DONE"
