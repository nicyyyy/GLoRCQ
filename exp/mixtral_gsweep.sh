#!/bin/bash
# ===========================================================================
# Mixtral-8x7B cluster-size (G) sweep — recover the zero-shot collapse.
# ---------------------------------------------------------------------------
# WHY: the Table-1 fair Mixtral run used G=64 (fix_rank=32, fp16 LoRA, attn 4-bit).
#   Mixtral has only 8 experts/layer x 32 layers = 256 experts per wtype, so
#   G=64 => 4 giant clusters each spanning 8 layers, AND n_clusters=4<8 makes
#   --cluster_method grassmannian silently fall back to traversal (contiguous
#   8-layer blocks). Result: PPL WINS (4.69 < TileQ_s 4.98) but 5-task ZS
#   COLLAPSES (64.48 vs TileQ_s 70.92) — a big heterogeneous cross-layer group
#   forced onto ONE shared U + ONE shared Sigma loses per-expert capability.
#
# HYPOTHESIS (user): smaller G => smaller/coherent clusters => better per-expert
#   reconstruction => ZS recovers. For Mixtral, shrinking G is ~free in bits
#   (huge expert dims make the shared-U amortization negligible: G 64->8 is only
#   +~0.007 bit), and G<=32 makes grassmannian clustering actually engage
#   (n_clusters = ceil(256/G) >= 8).
#
# WHAT THIS DOES (serial, one GPU job at a time, nwonga100 test:0 / GPU 4):
#   Phase-1 activation cache computed ONCE (G-independent) and reused.
#   For each (G, cluster_method): quantize -> save FAKE-quant fp16 checkpoint
#   (~94 GB) -> eval via device_map=auto (GPU4 + CPU offload for the ~14 GB that
#   exceeds 80 GB, since only GPU4 is free) -> WikiText-2 PPL -> 5-task ZS TRIAGE
#   (--limit 500) -> save both JSONs -> delete the 94 GB checkpoint (disk).
#   FAKE-quant matches the Table-1 baseline (4.69 / 64.48) methodology exactly.
#   Winner gets a FULL (unlimited) ZS re-run afterwards (separate step).
#
# EVAL PROTOCOL matches the 64.48 baseline exactly: acc / add_bos / bs=1 / 0-shot,
#   tasks arc_challenge,arc_easy,winogrande,hellaswag,piqa.
# NOTE: fake-quant (identical to baseline). Eval is slow (CPU offload for the
#   ~14 GB over 80 GB) but correct/comparable. Only GPU4 free (0-3 busy).
# ===========================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

VENV=/home/qyyang/repo/GLoRCQ/.venv/bin/python
MODEL=/mnt/Data/yqy/resource_dir/hf_cache/models--mistralai--Mixtral-8x7B-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0
OUTBASE=/mnt/Data/yqy/resource_dir/glorcq_paper_exp
PHASE1=$OUTBASE/mixtral_gsweep_phase1_cache.pt
LOGDIR=/home/qyyang/repo/GLoRCQ/logs/mixtral_gsweep
RESULTS=$LOGDIR/results
GPU=${CUDA_VISIBLE_DEVICES:-4}
mkdir -p "$LOGDIR" "$RESULTS"

export HF_HOME=/mnt/Data/yqy/resource_dir/hf_cache
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TASKS=arc_challenge,arc_easy,winogrande,hellaswag,piqa

# (G, cluster_method) pairs, strongest-signal first.
# 2 decisive configs first (offload eval is slow); add g8:traversal / g16 later if the
# small-G-helps-ZS trend confirms.
CONFIGS=(
  # "8:grassmannian"  # DONE: PPL 4.50 ZS 66.56
  "32:grassmannian"
)

echo "[$(date)] Mixtral G-sweep START  model=$MODEL  gpu=$GPU" | tee "$LOGDIR/sweep.log"

for cfg in "${CONFIGS[@]}"; do
  G="${cfg%%:*}"; METHOD="${cfg##*:}"
  TAG="g${G}_${METHOD}"
  OUT=$OUTBASE/mixtral_${TAG}
  QLOG=$LOGDIR/quant_${TAG}.log
  echo "" | tee -a "$LOGDIR/sweep.log"
  echo "==================================================================" | tee -a "$LOGDIR/sweep.log"
  echo "[$(date)] CONFIG $TAG  (G=$G method=$METHOD fix_rank=32 attn=4bit fp16-LoRA)" | tee -a "$LOGDIR/sweep.log"
  echo "==================================================================" | tee -a "$LOGDIR/sweep.log"

  # ---- Quantize (real-quant + strip). Phase-1 cache built on first run, reused. ----
  $VENV run_quantize.py \
      --model_path "$MODEL" --output_path "$OUT" \
      --qbit 2 --fix_rank 32 --G "$G" --group_size 128 \
      --lora_bit 16 --lora_iter 8 --ha_bsize 256 --id_bsize 256 \
      --attn_bits 4 \
      --cluster_method "$METHOD" --cluster_seed 42 \
      --phase1_cache_path "$PHASE1" \
      --no_export_real_quant \
      2>&1 | tee "$QLOG"
  if ! ls "$OUT"/model*.safetensors >/dev/null 2>&1; then
      echo "[$(date)] $TAG QUANT FAILED (no fake checkpoint) — skipping evals" | tee -a "$LOGDIR/sweep.log"
      continue
  fi
  grep -E "TOTAL=.*bits/param" "$QLOG" | tail -1 | tee -a "$LOGDIR/sweep.log"

  # ---- PPL (real-quant) ----
  $VENV evaluate/eval_ppl.py \
      --model_path "$OUT" --device auto \
      --max_length 2048 --stride 512 \
      --output_json "$RESULTS/ppl_${TAG}.json" 2>&1 | tee "$LOGDIR/ppl_${TAG}.log"
  grep -E "WikiText-2 PPL" "$LOGDIR/ppl_${TAG}.log" | tail -1 | tee -a "$LOGDIR/sweep.log"

  # ---- ZS TRIAGE (limit 500/task, acc/add_bos/bs1/0-shot) ----
  $VENV evaluate/eval_zeroshot.py \
      --model_path "$OUT" --device auto \
      --tasks "$TASKS" --num_fewshot 0 --batch_size 8 \
      --add_bos --metric_mode acc --limit 500 \
      --output_json "$RESULTS/zs_lim500_${TAG}.json" 2>&1 | tee "$LOGDIR/zs_${TAG}.log"
  grep -E "Average" "$LOGDIR/zs_${TAG}.log" | tail -1 | tee -a "$LOGDIR/sweep.log"

  # ---- Free disk: drop the checkpoint (JSONs already saved) ----
  rm -rf "$OUT"
  echo "[$(date)] $TAG DONE, checkpoint removed. Disk: $(df -h /mnt/Data | awk 'NR==2{print $4" free"}')" | tee -a "$LOGDIR/sweep.log"
done

echo "" | tee -a "$LOGDIR/sweep.log"
echo "[$(date)] Mixtral G-sweep ALL DONE.  Results JSON in $RESULTS/  MIXTRAL_GSWEEP_DONE" | tee -a "$LOGDIR/sweep.log"
