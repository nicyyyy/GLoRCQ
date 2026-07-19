#!/bin/bash
# GLoRCQ fair-bit port for DeepSeek-V2-Lite (deepseek_v2, custom modeling via trust_remote_code)
#   27 layers; layer 0 = dense MLP (first_k_dense_replace=1, kept fp16);
#   64 routed + 2 shared experts, top-6; expert dims 2048<->1408 == Qwen1.5-MoE.
#
# DE-RISK pass 1: attn_bits=16 -> MLA attention stays fp16 (Phase 2.5 skipped,
#   Phase 3 only quantizes routing experts). Shared experts + router + layer-0
#   dense MLP also stay fp16. Only the 64x26 routing experts get VQ4 + cross-layer
#   Grassmannian-shared LoRA (the paper contribution).
#
# Recipe: r=32 G=128 (experts identical to Qwen1.5), qbit=2, grassmannian clustering.
#
# Usage:
#   FAKE:  bash exp/run_deepseek_v2lite.sh fake
#   REAL:  bash exp/run_deepseek_v2lite.sh real

cd "$(dirname "$0")/.."

MODE=${1:-fake}
MODEL=/mnt/Data/yqy/resource_dir/deepseek-v2-lite

if [ "$MODE" = "real" ]; then
    OUTPUT=/mnt/Data/yqy/resource_dir/glorcq_paper_exp/deepseek_v2lite_real
    LOG=logs/deepseek_v2lite/quant_real.log
    REAL_ARGS="--strip_fp16_quantized"
else
    OUTPUT=/mnt/Data/yqy/resource_dir/glorcq_paper_exp/deepseek_v2lite_fake
    LOG=logs/deepseek_v2lite/quant_fake.log
    REAL_ARGS=""
fi

mkdir -p logs/deepseek_v2lite "$OUTPUT"

echo "[$(date)] DeepSeek-V2-Lite GLoRCQ ($MODE) started" | tee $LOG

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python run_quantize.py \
    --model_path $MODEL \
    --output_path $OUTPUT \
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
    --phase1_cache_path /mnt/Data/yqy/resource_dir/glorcq_paper_exp/deepseek_v2lite_phase1_cache.pt \
    $REAL_ARGS \
    2>&1 | tee -a $LOG

echo "[$(date)] DeepSeek-V2-Lite quant ($MODE) done. GLORCQ_QUANT_DONE" | tee -a $LOG
