#!/bin/bash
set -uo pipefail
PY=/home/qyyang/repo/GLoRCQ/.venv/bin/python
RQ=/home/qyyang/repo/GLoRCQ/run_quantize.py
PPL=/home/qyyang/repo/GLoRCQ/evaluate/eval_ppl.py
MODEL=/mnt/Data/yqy/resource_dir/hf_cache/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9
CACHE=/mnt/Data/yqy/resource_dir/glorcq_smoketest/qwen1.5-moe_stripped_v1_phase1_cache.pt
OUT=/mnt/Data/yqy/resource_dir/glorcq_grassmann/repro_cacheseed
RES=/home/qyyang/repo/GLoRCQ/exp/cluster/results/ablation
LOG=/home/qyyang/repo/GLoRCQ/logs/cluster_validity
FRESH_JSON=$RES/repro_fromscratch_ppl.json

# Gate: wait until the fresh from-scratch PPL json exists (fresh run done -> GPU 4 free)
echo "[$(date)] waiting for fresh validation to finish ($FRESH_JSON) ..."
while [ ! -f "$FRESH_JSON" ]; do sleep 60; done
echo "[$(date)] fresh done: PPL=$(grep -o '\"wikitext2_ppl\": [0-9.]*' "$FRESH_JSON"). Starting cache+seed."
sleep 30  # let fresh proc fully release GPU

export CUDA_VISIBLE_DEVICES=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
rm -rf "$OUT"; mkdir -p "$OUT"
# cache+seed: reuse the canonical stripped_v1 cache (frozen Stage-1) + the new seed fix
$PY -u "$RQ" --model_path "$MODEL" --output_path "$OUT" \
    --qbit 2 --fix_rank 20 --G 128 --group_size 128 --lora_bit 16 --lora_iter 8 \
    --int8_lora --int8_lora_v --pool_kmeans \
    --cluster_method grassmannian --cluster_rank 32 --cluster_recon_weight 0.0 \
    --export_real_quant --phase1_cache_path "$CACHE" \
    > "$LOG/repro_cacheseed_quant.log" 2>&1
$PY "$PPL" --model_path "$OUT" --device cuda:0 --max_length 2048 --stride 512 \
    --output_json "$RES/repro_cacheseed_ppl.json" > "$LOG/repro_cacheseed_ppl.log" 2>&1
echo "[$(date)] CACHE+SEED DONE PPL=$(grep -o '\"wikitext2_ppl\": [0-9.]*' "$RES/repro_cacheseed_ppl.json")"
rm -rf "$OUT"
