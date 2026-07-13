#!/bin/bash
# Fill the two missing §6.1/6.3 ablation points, gated on wino parity finishing.
#   rank=0  : LoRA off (bare 2-bit VQ backbone) -> §6.3 LoRA on/off
#   G=32    : G-sweep small-G point (r=32, 45 clusters, valid Grassmannian, no fallback)
# Each: quant (cache hit) -> PPL -> ZS 5-task -> rm ckpt. Same harness as seeded ablations.
set -uo pipefail
PY=/home/qyyang/repo/GLoRCQ/.venv/bin/python
RQ=/home/qyyang/repo/GLoRCQ/run_quantize.py
PPL=/home/qyyang/repo/GLoRCQ/evaluate/eval_ppl.py
ZS=/home/qyyang/repo/GLoRCQ/evaluate/eval_zeroshot.py
MODEL=/mnt/Data/yqy/resource_dir/hf_cache/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9
CACHE=/mnt/Data/yqy/resource_dir/glorcq_smoketest/qwen1.5-moe_stripped_v1_phase1_cache.pt
OUTROOT=/mnt/Data/yqy/resource_dir/glorcq_grassmann/abl_seeded
RES=/home/qyyang/repo/GLoRCQ/exp/cluster/results/ablation_seeded
LOG=/home/qyyang/repo/GLoRCQ/logs/cluster_validity
TASKS=arc_challenge,arc_easy,piqa,winogrande,hellaswag
mkdir -p "$OUTROOT" "$RES" "$LOG"
export CUDA_VISIBLE_DEVICES=4
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Gate: wait for wino parity to finish
echo "[$(date)] waiting for WINO PARITY DONE ..."
while ! grep -q 'WINO PARITY DONE' "$LOG/wino_parity_queue.log" 2>/dev/null; do sleep 60; done
echo "[$(date)] wino parity done -> starting fill ablations"; sleep 20

run_one () {
  local tag="$1"; shift; local extra="$*"; local out="$OUTROOT/$tag"
  echo "=========== [$(date)] START $tag : $extra ==========="
  rm -rf "$out"; mkdir -p "$out"
  $PY -u "$RQ" --model_path "$MODEL" --output_path "$out" \
      --qbit 2 --group_size 128 --lora_bit 16 --lora_iter 8 \
      --int8_lora --int8_lora_v --pool_kmeans \
      --cluster_method grassmannian --cluster_rank 32 --cluster_recon_weight 0.0 \
      --export_real_quant --phase1_cache_path "$CACHE" \
      $extra > "$LOG/ablseed_${tag}_quant.log" 2>&1
  if [ ! -f "$out/config.json" ]; then echo "[$tag] QUANT FAILED — see log"; return 1; fi
  local bits=$(grep -oE 'TOTAL=[0-9.]+' "$LOG/ablseed_${tag}_quant.log" | tail -1)
  $PY "$PPL" --model_path "$out" --device cuda:0 --max_length 2048 --stride 512 \
      --output_json "$RES/${tag}_ppl.json" > "$LOG/ablseed_${tag}_ppl.log" 2>&1
  $PY "$ZS" --model_path "$out" --device cuda:0 --tasks "$TASKS" \
      --num_fewshot 0 --batch_size 8 --metric_mode acc \
      --output_json "$RES/${tag}_zs.json" > "$LOG/ablseed_${tag}_zs.log" 2>&1
  echo "[$(date)] DONE $tag  $bits  PPL=$(grep -o '\"wikitext2_ppl\": [0-9.]*' "$RES/${tag}_ppl.json" 2>/dev/null)"
  rm -rf "$out"
}

run_one r0_loraoff --fix_rank 0  --G 128     # §6.3 LoRA off (grouping-agnostic; G ignored)
run_one r32_G32    --fix_rank 32 --G 32      # G-sweep small-G (45 clusters, valid Grassmannian)
echo "=========== [$(date)] FILL ABLATIONS DONE ==========="
