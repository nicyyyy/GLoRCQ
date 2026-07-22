#!/bin/bash
# g8_grassmannian REAL-quant (for Mixtral decode-speed work). Same config as the
# G-sweep winner but --export_real_quant + strip => ~21GB packed checkpoint that
# fits GPU4 (80G) for profiling/bench. phase1 cache reused (~2.5h).
set -uo pipefail
cd "$(dirname "$0")/.."
VENV=/home/qyyang/repo/GLoRCQ/.venv/bin/python
MODEL=/mnt/Data/yqy/resource_dir/hf_cache/models--mistralai--Mixtral-8x7B-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0
OUTBASE=/mnt/Data/yqy/resource_dir/glorcq_paper_exp
PHASE1=$OUTBASE/mixtral_gsweep_phase1_cache.pt
OUT=$OUTBASE/mixtral_g8_real
LOGDIR=/home/qyyang/repo/GLoRCQ/logs/mixtral_gsweep
export HF_HOME=/mnt/Data/yqy/resource_dir/hf_cache
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date)] g8 REAL-quant START" | tee "$LOGDIR/g8_realquant.log"
$VENV run_quantize.py \
    --model_path "$MODEL" --output_path "$OUT" \
    --qbit 2 --fix_rank 32 --G 8 --group_size 128 \
    --lora_bit 16 --lora_iter 8 --ha_bsize 256 --id_bsize 256 \
    --attn_bits 4 --cluster_method grassmannian --cluster_seed 42 \
    --phase1_cache_path "$PHASE1" --export_real_quant --strip_fp16_quantized \
    2>&1 | tee -a "$LOGDIR/g8_realquant.log"
echo "[$(date)] g8 REAL-quant DONE -> $OUT  MIXTRAL_G8_REAL_DONE" | tee -a "$LOGDIR/g8_realquant.log"
