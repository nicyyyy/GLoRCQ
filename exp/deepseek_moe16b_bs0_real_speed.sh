#!/bin/bash
# ===========================================================================
# STEP 2: real-quant export of the WINNING fair-budget config-B (seed 0) with
# attention + shared + layer-0 FFN ACTUALLY quantized, then bs=1 decode speed.
#   config B: attn 4-bit / shared 4-bit / layer-0 4-bit / routed rank 16,
#   cluster_seed=0, honest 2.2099 bits. PPL 6.6046 / acc 61.93 (beats TileQ_s,
#   reproducible bit-identical).
# Speed configs (stable median, warmup+3, prompt128/gen128): fp16 / full_graph /
#   full_graph+gather (GLORCQ_DEEPSEEK_GATHER=1), all vs the SAME-session fp16.
# GPU: tmux test:0, GPU4, one job at a time.
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=/mnt/Data/yqy/resource_dir/deepseek-moe-16b
REAL=/mnt/Data/yqy/resource_dir/glorcq_paper_exp/deepseek_moe16b_sw_Bs0_real
PHASE1=/mnt/Data/yqy/resource_dir/glorcq_paper_exp/deepseek_moe16b_phase1_cache.pt
OUT=logs/deepseek_moe16b
mkdir -p "$OUT" "$REAL"
export HF_HOME=/mnt/Data/yqy/resource_dir/hf_cache
GPU=${CUDA_VISIBLE_DEVICES:-4}

echo "[$(date)] === REAL-quant export config-B seed0 (attn+shared+layer0 quantized) ===" | tee "$OUT/bs0_real.log"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python run_quantize.py --model_path "$MODEL" --output_path "$REAL" \
    --qbit 2 --fix_rank 16 --G 128 --group_size 128 --lora_bit 16 --lora_iter 8 \
    --ha_bsize 256 --id_bsize 256 --attn_bits 4 --shared_bits 4 \
    --int8_lora --int8_lora_v --pool_kmeans --cluster_method grassmannian \
    --cluster_seed 0 --strip_fp16_quantized --phase1_cache_path "$PHASE1" \
    2>&1 | tee -a "$OUT/bs0_real.log"
for f in configuration_deepseek.py modeling_deepseek.py tokenizer.json tokenizer_config.json; do
    [ -f "$MODEL/$f" ] && cp -n "$MODEL/$f" "$REAL/$f" || true
done

echo "[$(date)] === DECODE SPEED (stable median) ===" | tee "$OUT/bs0_speed.log"
for CFG in fp16 full_graph; do
    CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        .venv/bin/python exp/bench_deepseek_stable.py --config $CFG \
        --real_path "$REAL" --fp16_path "$MODEL" 2>&1 | tee -a "$OUT/bs0_speed.log"
done
CUDA_VISIBLE_DEVICES=$GPU GLORCQ_DEEPSEEK_GATHER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python exp/bench_deepseek_stable.py --config full_graph \
    --real_path "$REAL" --fp16_path "$MODEL" 2>&1 | tee -a "$OUT/bs0_speed.log"

echo "[$(date)] BS0_REAL_SPEED_DONE" | tee -a "$OUT/bs0_speed.log"
echo done > "$OUT/bs0_speed.flag"
