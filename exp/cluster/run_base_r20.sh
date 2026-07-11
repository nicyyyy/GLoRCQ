#!/bin/bash
set -uo pipefail
PY=/home/qyyang/repo/GLoRCQ/.venv/bin/python
RQ=/home/qyyang/repo/GLoRCQ/run_quantize.py
PPL=/home/qyyang/repo/GLoRCQ/evaluate/eval_ppl.py
ZS=/home/qyyang/repo/GLoRCQ/evaluate/eval_zeroshot.py
MODEL=/mnt/Data/yqy/resource_dir/hf_cache/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9
CACHE=/mnt/Data/yqy/resource_dir/glorcq_smoketest/qwen1.5-moe_stripped_v1_phase1_cache.pt
OUT=/mnt/Data/yqy/resource_dir/glorcq_grassmann/abl/base_r20
RES=/home/qyyang/repo/GLoRCQ/exp/cluster/results/ablation
LOG=/home/qyyang/repo/GLoRCQ/logs/cluster_validity
export CUDA_VISIBLE_DEVICES=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
rm -rf "$OUT"; mkdir -p "$OUT"
echo "[$(date)] START base_r20 (fair-bit r20 G128 grassmannian, A100+cache)"
$PY -u "$RQ" --model_path "$MODEL" --output_path "$OUT" \
    --qbit 2 --fix_rank 20 --G 128 --group_size 128 --lora_bit 16 --lora_iter 8 \
    --int8_lora --int8_lora_v --pool_kmeans \
    --cluster_method grassmannian --cluster_rank 32 --cluster_recon_weight 0.0 \
    --export_real_quant --phase1_cache_path "$CACHE" \
    > "$LOG/abl_base_r20_quant.log" 2>&1
$PY "$PPL" --model_path "$OUT" --device cuda:0 --max_length 2048 --stride 512 \
    --output_json "$RES/base_r20_ppl.json" > "$LOG/abl_base_r20_ppl.log" 2>&1
$PY "$ZS" --model_path "$OUT" --device cuda:0 \
    --tasks arc_challenge,arc_easy,piqa,winogrande,hellaswag \
    --num_fewshot 0 --batch_size 8 --metric_mode acc \
    --output_json "$RES/base_r20_zs.json" > "$LOG/abl_base_r20_zs.log" 2>&1
echo "[$(date)] DONE base_r20 PPL=$(grep -o '\"wikitext2_ppl\": [0-9.]*' "$RES/base_r20_ppl.json")"
rm -rf "$OUT"
