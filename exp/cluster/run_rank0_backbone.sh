#!/bin/bash
# True LoRA-off ablation (§6.3): bare 2-bit VQ backbone, no low-rank compensation.
# Requires the backbone_only patch (this branch exp/lora-off-rank0). fake-quant only
# (no export) -> PPL + ZS. Same harness/cache/seed as seeded ablations.
set -uo pipefail
PY=/home/qyyang/repo/GLoRCQ/.venv/bin/python
RQ=/home/qyyang/repo/GLoRCQ/run_quantize.py
PPL=/home/qyyang/repo/GLoRCQ/evaluate/eval_ppl.py
ZS=/home/qyyang/repo/GLoRCQ/evaluate/eval_zeroshot.py
MODEL=/mnt/Data/yqy/resource_dir/hf_cache/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9
CACHE=/mnt/Data/yqy/resource_dir/glorcq_smoketest/qwen1.5-moe_stripped_v1_phase1_cache.pt
OUT=/mnt/Data/yqy/resource_dir/glorcq_grassmann/abl_seeded/r0_backbone
RES=/home/qyyang/repo/GLoRCQ/exp/cluster/results/ablation_seeded
LOG=/home/qyyang/repo/GLoRCQ/logs/cluster_validity
TASKS=arc_challenge,arc_easy,piqa,winogrande,hellaswag
mkdir -p "$RES" "$LOG"
export CUDA_VISIBLE_DEVICES=4
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=========== [$(date)] rank=0 TRUE backbone (backbone_only patch) ==========="
rm -rf "$OUT"; mkdir -p "$OUT"
$PY -u "$RQ" --model_path "$MODEL" --output_path "$OUT" \
    --qbit 2 --fix_rank 0 --G 128 --group_size 128 --lora_bit 16 --lora_iter 8 \
    --int8_lora --int8_lora_v --pool_kmeans \
    --cluster_method grassmannian --cluster_rank 32 --cluster_recon_weight 0.0 \
    --phase1_cache_path "$CACHE" \
    > "$LOG/r0_backbone_quant.log" 2>&1
if [ ! -f "$OUT/config.json" ]; then echo "QUANT FAILED"; exit 1; fi
echo "[$(date)] bits: $(grep -oE 'TOTAL=[0-9.]+ bits/param +\(Extra[^)]*\)' "$LOG/r0_backbone_quant.log" | tail -1)"
echo "[$(date)] MoE line: $(grep -oE 'MoE: +[0-9.]+ B params @ [0-9]+-bit weight +\( +[0-9.]+ M bits\)' "$LOG/r0_backbone_quant.log" | tail -1)"
$PY "$PPL" --model_path "$OUT" --device cuda:0 --max_length 2048 --stride 512 \
    --output_json "$RES/r0_backbone_ppl.json" > "$LOG/r0_backbone_ppl.log" 2>&1
$PY "$ZS" --model_path "$OUT" --device cuda:0 --tasks "$TASKS" \
    --num_fewshot 0 --batch_size 8 --metric_mode acc \
    --output_json "$RES/r0_backbone_zs.json" > "$LOG/r0_backbone_zs.log" 2>&1
echo "[$(date)] R0 BACKBONE DONE  PPL=$(grep -o '\"wikitext2_ppl\": [0-9.]*' "$RES/r0_backbone_ppl.json")"
rm -rf "$OUT"
