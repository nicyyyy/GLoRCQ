#!/bin/bash
# Task #200 Phase 2: re-measure speed/memory on the r32 canonical real-quant ckpt.
# Serial GPU jobs (one at a time) on CUDA_VISIBLE_DEVICES=4 inside tmux test:0.
set -uo pipefail
PY=/home/qyyang/repo/GLoRCQ/.venv/bin/python
CKPT=/mnt/Data/yqy/resource_dir/glorcq_paper_exp/qwen15_r32_real
LOG=/home/qyyang/repo/GLoRCQ/logs/cluster_validity
cd /home/qyyang/repo/GLoRCQ
export CUDA_VISIBLE_DEVICES=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [$(date)] task200 phase2 start ==="

# 1. sanity
$PY exp/cluster/infer_sanity.py "$CKPT" > "$LOG/task200_infer_sanity.log" 2>&1
grep "RESULT" "$LOG/task200_infer_sanity.log" || { echo "SANITY FAILED"; exit 1; }

# 2. speed b1 (standard + graph), Table-2 tool
$PY evaluate/eval_speed.py --model_path "$CKPT" --batch_size 1 \
    --prompt_len 128 --gen_len 128 \
    --output_json "$LOG/task200_speed_b1.json" > "$LOG/task200_speed_b1.log" 2>&1
echo "[$(date)] speed b1 done"

# 3. memory probe (same args as #199: max_seq_len 384)
$PY exp/cluster/mem_probe.py --model_path "$CKPT" --max_seq_len 384 \
    > "$LOG/task200_mem_probe.log" 2>&1
echo "[$(date)] mem probe done"

# 4. batch sweep (same defaults as #199: bs 1 4 8 16, prompt 128, gen 128, max_seq 384)
$PY exp/cluster/run_199_batch_sweep.py --model_path "$CKPT" \
    --output_json "$LOG/task200_batch_sweep.json" > "$LOG/task200_batch_sweep.log" 2>&1
echo "[$(date)] batch sweep done"

# 5. hit-rate re-join (CPU-light, reuses existing trace npz)
$PY exp/cluster/cluster_hitrate.py analyze \
    --trace_npz /mnt/Data/yqy/resource_dir/glorcq_paper_exp/exp5_routing_trace.npz \
    --cli r32=$CKPT/cross_layer_info.pt \
          r20=/mnt/Data/yqy/resource_dir/hf_dl/GLoRCQ-qwen1.5-moe-a2.7b-fair-grassmann-real/cross_layer_info.pt \
    --out_json "$LOG/task200_hitrate.json" > "$LOG/task200_hitrate.log" 2>&1
echo "=== [$(date)] task200 phase2 DONE ==="
