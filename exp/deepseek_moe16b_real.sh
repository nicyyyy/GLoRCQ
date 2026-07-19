#!/bin/bash
# ===========================================================================
# CLASP/GLoRCQ REAL-quant export for DeepSeek-MoE-16B (v1, DeepseekForCausalLM).
#   Same recipe as exp/deepseek_moe16b_quant.sh but writes a real-quant
#   checkpoint: cross_layer_info.pt (int8 U/SV + Sa + VQ4 residuals) + strips the
#   fp16 weights of the 64x27 quantized routing experts from the safetensors
#   (--strip_fp16_quantized). Attention / shared experts / router / layer-0 dense
#   remain fp16 in the safetensors (not stripped) so inference keeps them exact.
#   Reuses the Phase-1 cache from the fake run (same path) so only Phase 2/3 rerun.
#
#   Inference reconstructs each expert W ~= VQ4(residual) + Sa .* (U @ SV) at
#   load time (inference/model_builder._fast_meta_load path).
#
# NO-GPU-SAFE: LAUNCHES A GPU JOB. Run ONLY when GPU4 is free, in tmux `test:0`.
#   Run exp/deepseek_moe16b_quant.sh first (produces the shared Phase-1 cache).
#
# Usage:
#   tmux attach -t test
#   bash exp/deepseek_moe16b_real.sh
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=/mnt/Data/yqy/resource_dir/deepseek-moe-16b
OUTPUT=/mnt/Data/yqy/resource_dir/glorcq_paper_exp/deepseek_moe16b_real
PHASE1=/mnt/Data/yqy/resource_dir/glorcq_paper_exp/deepseek_moe16b_phase1_cache.pt
LOGDIR=logs/deepseek_moe16b
LOG=$LOGDIR/quant_real.log
mkdir -p "$LOGDIR" "$OUTPUT"

echo "[$(date)] DeepSeek-MoE-16B CLASP REAL-quant started" | tee "$LOG"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python run_quantize.py \
    --model_path "$MODEL" \
    --output_path "$OUTPUT" \
    --qbit 2 \
    --fix_rank 32 \
    --G 128 \
    --group_size 128 \
    --lora_bit 16 \
    --lora_iter 8 \
    --ha_bsize 256 \
    --id_bsize 256 \
    --attn_bits 16 \
    --int8_lora \
    --int8_lora_v \
    --pool_kmeans \
    --cluster_method grassmannian \
    --cluster_seed 42 \
    --strip_fp16_quantized \
    --phase1_cache_path "$PHASE1" \
    2>&1 | tee -a "$LOG"

# Real-quant inference loads via trust_remote_code -> the .py modeling files MUST
# be in the checkpoint dir (save_pretrained does not copy them). Also needed for HF.
for f in configuration_deepseek.py modeling_deepseek.py tokenizer.json tokenizer_config.json; do
    [ -f "$MODEL/$f" ] && cp -n "$MODEL/$f" "$OUTPUT/$f" || true
done

echo "[$(date)] DeepSeek-MoE-16B REAL-quant done -> $OUTPUT   MOE16B_REAL_DONE" | tee -a "$LOG"
echo "Next: bash exp/deepseek_moe16b_speed.sh    (bs=1 decode: full_graph vs fp16)"
echo "      bash exp/deepseek_moe16b_upload.sh    (push to HF)"
