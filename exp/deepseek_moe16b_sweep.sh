#!/bin/bash
# ===========================================================================
# Fair-budget sweep for DeepSeek-MoE-16B: quantize attention (4-bit GPTQ) AND
# all non-routed FFN (shared experts + layer-0 dense) at SHARED_BITS, with
# routed-expert LoRA rank = RANK, targeting ~2.16 avg over (attn+FFN, excl
# embed/lm_head). Produces a FAKE checkpoint and evals PPL + 5-task acc so we
# can check the TileQ_s bar (PPL<7.06, acc>61.52).
#
# Env in:  SHARED_BITS, RANK, TAG   (e.g. SHARED_BITS=4 RANK=8 TAG=A)
# Reuses the shared Phase-1 cache (rank-independent; Phase 2 reclusters at RANK).
# GPU: tmux test:0, GPU4, one job at a time.
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

SHARED_BITS=${SHARED_BITS:?set SHARED_BITS}
RANK=${RANK:?set RANK}
TAG=${TAG:?set TAG}
ATTN_BITS=${ATTN_BITS:-4}
CLUSTER_SEED=${CLUSTER_SEED:-42}

MODEL=/mnt/Data/yqy/resource_dir/deepseek-moe-16b
FAKE=/mnt/Data/yqy/resource_dir/glorcq_paper_exp/deepseek_moe16b_sw_${TAG}_fake
PHASE1=/mnt/Data/yqy/resource_dir/glorcq_paper_exp/deepseek_moe16b_phase1_cache.pt
OUT=logs/deepseek_moe16b
mkdir -p "$OUT" "$FAKE"
export HF_HOME=/mnt/Data/yqy/resource_dir/hf_cache
GPU=${CUDA_VISIBLE_DEVICES:-4}
TASKS=arc_challenge,arc_easy,winogrande,hellaswag,piqa
L=$OUT/sweep_${TAG}.log

echo "[$(date)] === SWEEP $TAG: attn=$ATTN_BITS shared=$SHARED_BITS rank=$RANK cluster_seed=$CLUSTER_SEED ===" | tee "$L"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python run_quantize.py --model_path "$MODEL" --output_path "$FAKE" \
    --qbit 2 --fix_rank $RANK --G 128 --group_size 128 --lora_bit 16 --lora_iter 8 \
    --ha_bsize 256 --id_bsize 256 --attn_bits $ATTN_BITS --shared_bits $SHARED_BITS \
    --int8_lora --int8_lora_v --pool_kmeans --cluster_method grassmannian \
    --cluster_seed $CLUSTER_SEED --no_export_real_quant --phase1_cache_path "$PHASE1" \
    2>&1 | tee -a "$L"
for f in configuration_deepseek.py modeling_deepseek.py tokenizer.json tokenizer_config.json; do
    [ -f "$MODEL/$f" ] && cp -n "$MODEL/$f" "$FAKE/$f" || true
done

echo "[$(date)] === PPL (sweep $TAG) ===" | tee -a "$L"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python evaluate/eval_ppl.py --model_path "$FAKE" --device cuda:0 \
    --max_length 2048 --stride 512 --output_json "$OUT/sweep_${TAG}_ppl.json" 2>&1 | tee -a "$L"

echo "[$(date)] === Zero-shot (sweep $TAG, acc/bs16/add_bos) ===" | tee -a "$L"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python evaluate/eval_zeroshot.py --model_path "$FAKE" --device cuda:0 \
    --tasks $TASKS --num_fewshot 0 --batch_size 16 --metric_mode acc --add_bos \
    --output_json "$OUT/sweep_${TAG}_zs.json" 2>&1 | tee -a "$L"

echo "[$(date)] SWEEP_${TAG}_DONE" | tee -a "$L"
echo "SWEEP_${TAG}_DONE" > "$OUT/sweep_${TAG}.flag"
