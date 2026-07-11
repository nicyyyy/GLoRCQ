#!/bin/bash
# Grassmannian ablation campaign on Qwen1.5-MoE (A100, run inside SLURM test:0).
# Each config: quant (reuse phase1 cache) -> PPL -> zeroshot -> delete ckpt.
# Results JSONs kept under exp/cluster/results/ablation/.
set -uo pipefail

PY=/home/qyyang/repo/GLoRCQ/.venv/bin/python
RQ=/home/qyyang/repo/GLoRCQ/run_quantize.py
PPL=/home/qyyang/repo/GLoRCQ/evaluate/eval_ppl.py
ZS=/home/qyyang/repo/GLoRCQ/evaluate/eval_zeroshot.py
MODEL=/mnt/Data/yqy/resource_dir/hf_cache/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9
CACHE=/mnt/Data/yqy/resource_dir/glorcq_smoketest/qwen1.5-moe_stripped_v1_phase1_cache.pt
OUTROOT=/mnt/Data/yqy/resource_dir/glorcq_grassmann/abl
RES=/home/qyyang/repo/GLoRCQ/exp/cluster/results/ablation
LOG=/home/qyyang/repo/GLoRCQ/logs/cluster_validity
mkdir -p "$OUTROOT" "$RES" "$LOG"
export CUDA_VISIBLE_DEVICES=4
export PYTORCH_ALLOC_CONF=expandable_segments:True

# tag : extra run_quantize args (on top of the shared Grassmannian base)
run_one () {
  local tag="$1"; shift
  local extra="$*"
  local out="$OUTROOT/$tag"
  echo "=========== [$(date)] START $tag : $extra ==========="
  rm -rf "$out"; mkdir -p "$out"
  $PY -u "$RQ" --model_path "$MODEL" --output_path "$out" \
      --qbit 2 --G 128 --group_size 128 --lora_bit 16 --lora_iter 8 \
      --int8_lora --int8_lora_v --pool_kmeans \
      --cluster_method grassmannian --cluster_rank 32 --cluster_recon_weight 0.0 \
      --export_real_quant --phase1_cache_path "$CACHE" \
      $extra > "$LOG/abl_${tag}_quant.log" 2>&1
  if [ ! -f "$out/config.json" ]; then
      echo "[$tag] QUANT FAILED — see $LOG/abl_${tag}_quant.log"; return 1
  fi
  $PY "$PPL" --model_path "$out" --device cuda:0 --max_length 2048 --stride 512 \
      --output_json "$RES/${tag}_ppl.json" > "$LOG/abl_${tag}_ppl.log" 2>&1
  $PY "$ZS" --model_path "$out" --device cuda:0 \
      --tasks arc_challenge,arc_easy,piqa,winogrande,hellaswag \
      --num_fewshot 0 --batch_size 8 --metric_mode acc \
      --output_json "$RES/${tag}_zs.json" > "$LOG/abl_${tag}_zs.log" 2>&1
  echo "[$(date)] DONE $tag  PPL=$(grep -o '\"wikitext2_ppl\": [0-9.]*' "$RES/${tag}_ppl.json" 2>/dev/null)"
  rm -rf "$out"   # free disk; only JSONs kept
}

# --- rank sweep (fix_rank; base r20 = v3a 7.37 already have) ---
run_one r16 --fix_rank 16
run_one r32 --fix_rank 32
run_one r64 --fix_rank 64
# --- G sweep (base G128 have; G64 is the real-Grassmannian small-G point) ---
run_one G64 --fix_rank 20 --G 64
echo "=========== [$(date)] ALL ABLATIONS DONE ==========="
