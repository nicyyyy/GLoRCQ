#!/bin/bash
# Apples-to-apples WinoGrande (+ full 5-task) parity check: fp16 vs fair-bit r32,
# SAME A100 + SAME eval_zeroshot.py harness (add_bos=False, batch=8, num_fewshot=0, acc).
# Addresses reviewer concern that quantized WinoG (69.38) > cross-machine fp16 (68.75).
set -uo pipefail
PY=/home/qyyang/repo/GLoRCQ/.venv/bin/python
RQ=/home/qyyang/repo/GLoRCQ/run_quantize.py
ZS=/home/qyyang/repo/GLoRCQ/evaluate/eval_zeroshot.py
MODEL=/mnt/Data/yqy/resource_dir/hf_cache/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9
CACHE=/mnt/Data/yqy/resource_dir/glorcq_smoketest/qwen1.5-moe_stripped_v1_phase1_cache.pt
OUT=/mnt/Data/yqy/resource_dir/glorcq_grassmann/abl_seeded/r32_wino
RES=/home/qyyang/repo/GLoRCQ/exp/cluster/results/ablation_seeded
LOG=/home/qyyang/repo/GLoRCQ/logs/cluster_validity
TASKS=arc_challenge,arc_easy,piqa,winogrande,hellaswag
mkdir -p "$RES" "$LOG"
export CUDA_VISIBLE_DEVICES=4
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=========== [$(date)] STEP 1: fp16 base 5-task (same harness) ==========="
$PY "$ZS" --model_path "$MODEL" --device cuda:0 --tasks "$TASKS" \
    --num_fewshot 0 --batch_size 8 --metric_mode acc \
    --output_json "$RES/fp16_a100_sameharness_zs.json" > "$LOG/fp16_a100_zs.log" 2>&1
echo "[$(date)] fp16 done. WinoG=$(python3 -c "import json;print(round(json.load(open('$RES/fp16_a100_sameharness_zs.json'))['task_results']['winogrande']['value']*100,2))" 2>/dev/null)"

echo "=========== [$(date)] STEP 2: re-quant fair-bit base r32 (seeded) ==========="
rm -rf "$OUT"; mkdir -p "$OUT"
$PY -u "$RQ" --model_path "$MODEL" --output_path "$OUT" \
    --qbit 2 --fix_rank 32 --G 128 --group_size 128 --lora_bit 16 --lora_iter 8 \
    --int8_lora --int8_lora_v --pool_kmeans \
    --cluster_method grassmannian --cluster_rank 32 --cluster_recon_weight 0.0 \
    --export_real_quant --phase1_cache_path "$CACHE" \
    > "$LOG/r32_wino_quant.log" 2>&1
if [ ! -f "$OUT/config.json" ]; then echo "QUANT FAILED"; exit 1; fi

echo "=========== [$(date)] STEP 3: quantized r32 5-task (same harness) ==========="
$PY "$ZS" --model_path "$OUT" --device cuda:0 --tasks "$TASKS" \
    --num_fewshot 0 --batch_size 8 --metric_mode acc \
    --output_json "$RES/r32_rerun_sameharness_zs.json" > "$LOG/r32_wino_zs.log" 2>&1
echo "[$(date)] quantized done. WinoG=$(python3 -c "import json;print(round(json.load(open('$RES/r32_rerun_sameharness_zs.json'))['task_results']['winogrande']['value']*100,2))" 2>/dev/null)"
rm -rf "$OUT"
echo "=========== [$(date)] WINO PARITY DONE ==========="
