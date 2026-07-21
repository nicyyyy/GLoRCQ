#!/bin/bash
# ===========================================================================
# CANONICAL DeepSeek-MoE-16B variant: 2-bit routed experts + 4-bit GPTQ
# attention (--attn_bits 4) + fp16 shared experts + fp16 router/layer0.
# This matches the Qwen1.5/Mixtral/Qwen3 recipe ("2-bit tile VQ + 4-bit GPTQ
# attention") and makes the reported bit budget HONEST (attention is now
# actually quantized, not fp16).
#
# Separate output dirs (do NOT touch the accuracy-correct fp16-attn checkpoint):
#   fake -> deepseek_moe16b_attn4_fake   (PPL + zero-shot acc)
#   real -> deepseek_moe16b_attn4_real   (bs=1 decode speed)
# Reuses the shared Phase-1 cache (expert clustering; independent of attn_bits;
# attention GPTQ is Phase 2.5, computed fresh).
#
# GPU discipline: run ONLY in tmux test:0, GPU4, one job at a time.
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=/mnt/Data/yqy/resource_dir/deepseek-moe-16b
FAKE=/mnt/Data/yqy/resource_dir/glorcq_paper_exp/deepseek_moe16b_attn4_fake
REAL=/mnt/Data/yqy/resource_dir/glorcq_paper_exp/deepseek_moe16b_attn4_real
PHASE1=/mnt/Data/yqy/resource_dir/glorcq_paper_exp/deepseek_moe16b_phase1_cache.pt
FP16=/mnt/Data/yqy/resource_dir/deepseek-moe-16b
OUT=logs/deepseek_moe16b
mkdir -p "$OUT" "$FAKE" "$REAL"
export HF_HOME=/mnt/Data/yqy/resource_dir/hf_cache
GPU=${CUDA_VISIBLE_DEVICES:-4}
TASKS=arc_challenge,arc_easy,winogrande,hellaswag,piqa

COMMON="--qbit 2 --fix_rank 32 --G 128 --group_size 128 --lora_bit 16 \
  --lora_iter 8 --ha_bsize 256 --id_bsize 256 --attn_bits 4 \
  --int8_lora --int8_lora_v --pool_kmeans --cluster_method grassmannian \
  --cluster_seed 42 --phase1_cache_path $PHASE1"

echo "[$(date)] === (1/4) FAKE-quant attn_bits=4 ===" | tee "$OUT/attn4_fake.log"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python run_quantize.py --model_path "$MODEL" --output_path "$FAKE" \
    $COMMON 2>&1 | tee -a "$OUT/attn4_fake.log"
for f in configuration_deepseek.py modeling_deepseek.py tokenizer.json tokenizer_config.json; do
    [ -f "$MODEL/$f" ] && cp -n "$MODEL/$f" "$FAKE/$f" || true
done

echo "[$(date)] === (2/4) PPL + zero-shot (fake) ===" | tee "$OUT/attn4_eval.log"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python evaluate/eval_ppl.py --model_path "$FAKE" --device cuda:0 \
    --max_length 2048 --stride 512 --output_json "$OUT/attn4_ppl_fake.json" \
    2>&1 | tee -a "$OUT/attn4_eval.log"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python evaluate/eval_zeroshot.py --model_path "$FAKE" --device cuda:0 \
    --tasks $TASKS --num_fewshot 0 --batch_size 16 --metric_mode acc --add_bos \
    --output_json "$OUT/attn4_zeroshot_fake.json" 2>&1 | tee -a "$OUT/attn4_eval.log"

echo "[$(date)] === (3/4) REAL-quant attn_bits=4 ===" | tee "$OUT/attn4_real.log"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python run_quantize.py --model_path "$MODEL" --output_path "$REAL" \
    $COMMON --strip_fp16_quantized 2>&1 | tee -a "$OUT/attn4_real.log"
for f in configuration_deepseek.py modeling_deepseek.py tokenizer.json tokenizer_config.json; do
    [ -f "$MODEL/$f" ] && cp -n "$MODEL/$f" "$REAL/$f" || true
done

echo "[$(date)] === (4/4) bs=1 decode speed (real) ===" | tee "$OUT/attn4_speed.log"
for CFG in fp16 full_graph; do
    CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        .venv/bin/python exp/bench_deepseek_stable.py --config $CFG \
        --real_path "$REAL" --fp16_path "$FP16" 2>&1 | tee -a "$OUT/attn4_speed.log"
done
CUDA_VISIBLE_DEVICES=$GPU GLORCQ_DEEPSEEK_GATHER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python exp/bench_deepseek_stable.py --config full_graph \
    --real_path "$REAL" --fp16_path "$FP16" 2>&1 | tee -a "$OUT/attn4_speed.log"

echo "[$(date)] ATTN4_ALL_DONE" | tee -a "$OUT/attn4_speed.log"
echo "ATTN4_DONE" > "$OUT/attn4_done.flag"
