#!/bin/bash
set -uo pipefail
PY=/home/qyyang/repo/GLoRCQ/.venv/bin/python
RQ=/home/qyyang/repo/GLoRCQ/run_quantize.py
PPL=/home/qyyang/repo/GLoRCQ/evaluate/eval_ppl.py
MODEL=/mnt/Data/yqy/resource_dir/hf_cache/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9
OUT=/mnt/Data/yqy/resource_dir/glorcq_grassmann/repro_fromscratch
RES=/home/qyyang/repo/GLoRCQ/exp/cluster/results/ablation
LOG=/home/qyyang/repo/GLoRCQ/logs/cluster_validity
export CUDA_VISIBLE_DEVICES=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
rm -rf "$OUT"; mkdir -p "$OUT"
echo "[$(date)] REPRO CHECK: fresh from-scratch (own new Phase-1, seeded=42), base r20 G128 grassmannian"
# Fresh Phase 1: point phase1_cache to a NEW path so it recomputes + saves there (does NOT touch stripped_v1 cache)
$PY -u "$RQ" --model_path "$MODEL" --output_path "$OUT" \
    --qbit 2 --fix_rank 20 --G 128 --group_size 128 --lora_bit 16 --lora_iter 8 \
    --int8_lora --int8_lora_v --pool_kmeans \
    --cluster_method grassmannian --cluster_rank 32 --cluster_recon_weight 0.0 \
    --export_real_quant \
    --phase1_cache_path "$OUT/fresh_phase1_cache.pt" \
    > "$LOG/repro_fromscratch_quant.log" 2>&1
$PY "$PPL" --model_path "$OUT" --device cuda:0 --max_length 2048 --stride 512 \
    --output_json "$RES/repro_fromscratch_ppl.json" > "$LOG/repro_fromscratch_ppl.log" 2>&1
echo "[$(date)] REPRO DONE PPL=$(grep -o '\"wikitext2_ppl\": [0-9.]*' "$RES/repro_fromscratch_ppl.json")"
# keep the fresh phase1 cache + ckpt for now (user may want to inspect); note disk
