#!/bin/bash
# ===========================================================================
# CLASP/GLoRCQ fair-bit FAKE-quant for DeepSeek-MoE-16B (Dai 2024, ORIGINAL v1).
#   arch DeepseekForCausalLM, model_type `deepseek`, trust_remote_code.
#   28 layers; layer 0 = dense DeepseekMLP (first_k_dense_replace=1, kept fp16);
#   64 routed + 2 shared experts, top-6; STANDARD symmetric MHA (heads=kv=16,
#   head_dim=128 — NO MLA). Expert dims 1408<->2048 == Qwen1.5-MoE == V2-Lite,
#   so the VQ4 kernel / clustering need ZERO changes.
#
# DE-RISK (mirrors the V2-Lite CLASP port, same footprint convention):
#   attn_bits=16 -> MHA attention stays fp16 (Phase 2.5 skipped). Shared experts
#   + router (MoEGate) + layer-0 dense MLP also stay fp16. Only the 64x27 routing
#   experts get VQ4 + cross-layer Grassmannian-shared LoRA (the paper contribution).
#   Verified no-GPU: the pipeline selects exactly 5184 routed-expert linears
#   (64*3*27) and skips attention/shared/router/layer0-dense.
#
# Recipe = IDENTICAL to the V2-Lite fair-bit run (experts have identical dims):
#   r=32 G=128 qbit=2 grassmannian, int8 U + int8 V -> ~2.16 bits over experts.
#
# NO-GPU-SAFE: this script LAUNCHES A GPU JOB. Run it ONLY when GPU4 is free,
#   inside tmux `test:0` (nwonga100 rule: never run GPU work outside test:0 or
#   the 5-min non-SLURM killer cron reaps it). It is NOT auto-launched.
#
# Usage (later, when GPU4 frees up):
#   tmux attach -t test          # window :0 must be free
#   bash exp/deepseek_moe16b_quant.sh          # fake-quant + eval-ready dir
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=/mnt/Data/yqy/resource_dir/deepseek-moe-16b
OUTPUT=/mnt/Data/yqy/resource_dir/glorcq_paper_exp/deepseek_moe16b_fake
PHASE1=/mnt/Data/yqy/resource_dir/glorcq_paper_exp/deepseek_moe16b_phase1_cache.pt
LOGDIR=logs/deepseek_moe16b
LOG=$LOGDIR/quant_fake.log
mkdir -p "$LOGDIR" "$OUTPUT"

echo "[$(date)] DeepSeek-MoE-16B CLASP FAKE-quant started" | tee "$LOG"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python run_quantize.py \
    --model_path "$MODEL" \
    --output_path "$OUTPUT" \
    --qbit 2 \
    --fix_rank 32 \
    --G 128 \
    --group_size 128 \
    --lora_bit 16 \
    --lora_iter 8 \
    --ha_bsize 256 \
    --id_bsize 256 \
    --attn_bits 16 \
    --int8_lora \
    --int8_lora_v \
    --pool_kmeans \
    --cluster_method grassmannian \
    --cluster_seed 42 \
    --phase1_cache_path "$PHASE1" \
    2>&1 | tee -a "$LOG"

# save_pretrained copies config.json + tokenizer, but NOT the trust_remote_code
# .py files. Copy them so the fake dir can be loaded standalone / uploaded.
for f in configuration_deepseek.py modeling_deepseek.py tokenizer.json tokenizer_config.json; do
    [ -f "$MODEL/$f" ] && cp -n "$MODEL/$f" "$OUTPUT/$f" || true
done

echo "[$(date)] DeepSeek-MoE-16B FAKE-quant done -> $OUTPUT   MOE16B_QUANT_DONE" | tee -a "$LOG"
echo "Next: bash exp/deepseek_moe16b_eval.sh   (PPL + 5-task zero-shot, canonical acc)"
echo "      bash exp/deepseek_moe16b_real.sh   (real-quant export + strip fp16)"
