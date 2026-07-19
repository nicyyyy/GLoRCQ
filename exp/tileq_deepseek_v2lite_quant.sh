#!/bin/bash
# TileQ_s baseline quantization for DeepSeek-V2-Lite (deepseek_v2, trust_remote_code).
# ---------------------------------------------------------------------------
# Produces a same-method TileQ_s number for V2-Lite at OUR fair-bit budget
# (2-bit + ~0.16 extra bits) so it can be evaluated under our Table-1 protocol
# (see exp/tileq_deepseek_v2lite_eval.sh).
#
# WHAT GETS QUANTIZED (de-risk, mirrors the GLoRCQ V2-Lite port):
#   * ONLY the 64 routed experts (mlp.experts.*.{gate,up,down}_proj) are quantized:
#     2-bit scalar residual (WeightQuantizer) + a TileQ 2D low-rank term shared
#     across the 8x8 expert tile (rank -> the "extra bits").
#   * MLA attention (q/kv_a/kv_b/o_proj), the 2 shared experts, the router
#     (MoEGate, not nn.Linear) and the layer-0 dense MLP (first_k_dense_replace=1)
#     all stay fp16. This matches how GLoRCQ evaluates V2-Lite, so the two numbers
#     are apples-to-apples (same quantized weight set, same footprint convention).
#
# BIT BUDGET (printed as "loraq-bit(group)" at the end of the run; accounting is
#   over the quantized experts only):
#     loraq_bit = qbit + 16 * lora_size / total_size
#   For V2-Lite experts (out/in in {1408,2048}, 64 experts in an 8x8 tile,
#   lora_bit=16): extra_bits ~= fix_rank * 0.0024. So fix_rank=64 -> ~2.15,
#   fix_rank=67 -> ~2.16. TUNE fix_rank to land exactly on 2.16 after the first
#   run prints the achieved loraq_bit (the sketch may return srank <= fix_rank).
#
# NO-GPU-SAFE: this script starts a GPU job; run it only when GPU 4 is free,
#   inside tmux `test:0` (nwonga100 rule). It is NOT auto-launched.
#
# Usage (later, when a GPU frees up):
#   tmux attach -t test        # window :0 must be free
#   bash exp/tileq_deepseek_v2lite_quant.sh
# ---------------------------------------------------------------------------
set -euo pipefail

TILEQ_DIR=/home/qyyang/repo/tileq
VENV_PY=/home/qyyang/repo/GLoRCQ/.venv/bin/python     # tf 4.51.3, torch 2.6, has fast_hadamard_transform
MODEL=/mnt/Data/yqy/resource_dir/deepseek-v2-lite
OUTPUT=/mnt/Data/yqy/resource_dir/glorcq_paper_exp/tileq_s_deepseek_v2lite_fake
LOGDIR=/home/qyyang/repo/GLoRCQ/logs/tileq_deepseek_v2lite
LOG=$LOGDIR/quant.log

# TileQ_s scalar config @ fair-bit ~2.16
QBIT=2                # 2-bit scalar residual on the routed experts
FIX_RANK=64           # ~+0.15 extra bits; bump toward 67 for exactly 2.16
LORA_BIT=16           # U,V stored fp16 (matches the extra-bit formula above)
GROUPSIZE=128         # per-group scales (same as GLoRCQ V2-Lite)
LORA_ITER=8
TILE_ROW=8            # 64 experts -> 8x8 tile
METHOD=gptq           # scalar GPTQ path (gptq_fwrd_lora)
QUANTIZER=scalar      # WeightQuantizer(qbit); "binary" would be 1-bit BiWeightQuantizer

mkdir -p "$LOGDIR" "$OUTPUT"

# Calibration data lives in resource_dir (TileQ's default c4 path is repointed here).
export TILEQ_C4_PATH=/mnt/Data/yqy/resource_dir/c4/c4-train.00000-of-01024.json.gz
# WikiText-2 (gptq dataloader) + tokenizer cache.
export HF_HOME=/mnt/Data/yqy/resource_dir/hf_cache

echo "[$(date)] TileQ_s DeepSeek-V2-Lite quant starting" | tee "$LOG"
echo "  model=$MODEL" | tee -a "$LOG"
echo "  output=$OUTPUT" | tee -a "$LOG"
echo "  qbit=$QBIT fix_rank=$FIX_RANK lora_bit=$LORA_BIT groupsize=$GROUPSIZE tile_row=$TILE_ROW quantizer=$QUANTIZER" | tee -a "$LOG"

cd "$TILEQ_DIR"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$VENV_PY" run_quantize.py \
    --model_path "$MODEL" \
    --output_path "$OUTPUT" \
    --qbit $QBIT \
    --fix_rank $FIX_RANK \
    --lora_bit $LORA_BIT \
    --groupsize $GROUPSIZE \
    --lora_iter $LORA_ITER \
    --tile_row $TILE_ROW \
    --method $METHOD \
    --quantizer $QUANTIZER \
    --loratool sketch \
    --ha_bsize 256 \
    --id_bsize 256 \
    2>&1 | tee -a "$LOG"

echo "[$(date)] TileQ_s DeepSeek-V2-Lite quant done -> $OUTPUT   TILEQ_QUANT_DONE" | tee -a "$LOG"
echo "Next: bash exp/tileq_deepseek_v2lite_eval.sh   (Table-1 PPL + 5-task zero-shot)"
