#!/bin/bash
# Correctness gate for Mixtral #1(gather+idx)/#2(inline LoRA): greedy token match.
set -uo pipefail
cd "$(dirname "$0")/.."
VENV=/home/qyyang/repo/GLoRCQ/.venv/bin/python
G8=/mnt/Data/yqy/resource_dir/glorcq_paper_exp/mixtral_g8_real
LOG=/home/qyyang/repo/GLoRCQ/logs/mixtral_gsweep
export HF_HOME=/mnt/Data/yqy/resource_dir/hf_cache
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "== baseline (standard, inline OFF) =="
GLORCQ_MIXTRAL_INLINE_LORA=0 $VENV exp/mixtral_decode_ids.py "$G8" 2>&1 | grep -E "gates|MIXTRAL_GEN_IDS" | tee "$LOG/ids_baseline.log"
echo "== #2 inline LoRA (standard) =="
GLORCQ_MIXTRAL_INLINE_LORA=1 $VENV exp/mixtral_decode_ids.py "$G8" 2>&1 | grep -E "gates|MIXTRAL_GEN_IDS" | tee "$LOG/ids_inline.log"
echo "== #1 gather + indexed kernel =="
GLORCQ_MIXTRAL_INLINE_LORA=1 GLORCQ_MIXTRAL_GRAPH=1 GLORCQ_MIXTRAL_IDXKERNEL=1 $VENV exp/mixtral_decode_ids.py "$G8" 2>&1 | grep -E "gates|MIXTRAL_GEN_IDS" | tee "$LOG/ids_gatheridx.log"
echo "== gather WITHOUT idx (prior 1.02x path) =="
GLORCQ_MIXTRAL_INLINE_LORA=1 GLORCQ_MIXTRAL_GRAPH=1 GLORCQ_MIXTRAL_IDXKERNEL=0 $VENV exp/mixtral_decode_ids.py "$G8" 2>&1 | grep -E "gates|MIXTRAL_GEN_IDS" | tee "$LOG/ids_gather_noidx.log"
echo "===== TOKEN MATCH ====="
b=$(grep MIXTRAL_GEN_IDS "$LOG/ids_baseline.log")
for f in inline gatheridx gather_noidx; do
  v=$(grep MIXTRAL_GEN_IDS "$LOG/ids_$f.log")
  [ "$b" = "$v" ] && echo "$f: MATCH baseline ✓" || echo "$f: DIFFER ✗"
done
echo "MIXTRAL_VERIFY_DONE"
