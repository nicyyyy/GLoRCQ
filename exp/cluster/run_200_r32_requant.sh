#!/bin/bash
# Task #200: re-export Qwen1.5-MoE real-quant ckpt at the CANONICAL seeded r32 base
# (same operating point as Table 1: seed 42, fix_rank=32, G=128, grassmannian,
#  cluster_rank=32, recon_weight=0.0 — EXACT command from run_seeded_ablation.sh r32).
# Steps: quant (bit-reproducible, expect fake PPL 7.1378) -> PPL gate -> post-hoc
# fp16 strip (same _strip_fp16_quantized_weights the --strip_fp16_quantized flag calls).
# Strip only runs if PPL matches within 0.01 of the canonical 7.137814.
set -uo pipefail
PY=/home/qyyang/repo/GLoRCQ/.venv/bin/python
RQ=/home/qyyang/repo/GLoRCQ/run_quantize.py
PPL=/home/qyyang/repo/GLoRCQ/evaluate/eval_ppl.py
MODEL=/mnt/Data/yqy/resource_dir/hf_cache/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9
CACHE=/mnt/Data/yqy/resource_dir/glorcq_smoketest/qwen1.5-moe_stripped_v1_phase1_cache.pt
OUT=/mnt/Data/yqy/resource_dir/glorcq_paper_exp/qwen15_r32_real
RES=/home/qyyang/repo/GLoRCQ/exp/cluster/results
LOG=/home/qyyang/repo/GLoRCQ/logs/cluster_validity
mkdir -p "$OUT" "$LOG"
export CUDA_VISIBLE_DEVICES=4
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=========== [$(date)] TASK200 START r32 canonical re-quant ==========="

# --- Step 1: quant (EXACT seeded r32 command; unstripped so fake PPL is checkable)
$PY -u "$RQ" --model_path "$MODEL" --output_path "$OUT" \
    --qbit 2 --G 128 --group_size 128 --lora_bit 16 --lora_iter 8 \
    --int8_lora --int8_lora_v --pool_kmeans \
    --cluster_method grassmannian --cluster_rank 32 --cluster_recon_weight 0.0 \
    --export_real_quant --phase1_cache_path "$CACHE" \
    --fix_rank 32 > "$LOG/task200_r32_quant.log" 2>&1
if [ ! -f "$OUT/config.json" ] || [ ! -f "$OUT/cross_layer_info.pt" ]; then
    echo "[task200] QUANT FAILED — see $LOG/task200_r32_quant.log"; exit 1
fi
grep -o 'TOTAL=[0-9.]*' "$LOG/task200_r32_quant.log" | tail -1

# --- Step 2: fake-quant PPL gate (canonical = 7.137814)
echo "[$(date)] task200: quant done, running fake-quant PPL check ..."
$PY "$PPL" --model_path "$OUT" --device cuda:0 --max_length 2048 --stride 512 \
    --output_json "$RES/task200_r32_ppl.json" > "$LOG/task200_r32_ppl.log" 2>&1
PPL_OK=$($PY - <<'EOF'
import json
try:
    d = json.load(open('/home/qyyang/repo/GLoRCQ/exp/cluster/results/task200_r32_ppl.json'))
    ppl = d['wikitext2_ppl']
    print('OK' if abs(ppl - 7.137814044952393) <= 0.01 else f'MISMATCH:{ppl}')
except Exception as e:
    print(f'ERROR:{e}')
EOF
)
echo "[$(date)] task200: PPL gate = $PPL_OK"
if [ "$PPL_OK" != "OK" ]; then
    echo "[task200] STOP — fake PPL does not reproduce canonical 7.1378 ($PPL_OK). NOT stripping."
    exit 1
fi

# --- Step 3: post-hoc strip (identical to --strip_fp16_quantized code path)
echo "[$(date)] task200: PPL matched — stripping fp16 quantized weights ..."
$PY - <<'EOF' > "$LOG/task200_r32_strip.log" 2>&1
import sys, torch
sys.path.insert(0, '/home/qyyang/repo/GLoRCQ')
from run_quantize import _strip_fp16_quantized_weights
out = '/mnt/Data/yqy/resource_dir/glorcq_paper_exp/qwen15_r32_real'
d = torch.load(out + '/cross_layer_info.pt', map_location='cpu', weights_only=False)
_strip_fp16_quantized_weights(out, d['assignments'], d['attn_gptq_packs'], d['vq_residuals'])
print('config:', d['config'])
EOF
tail -3 "$LOG/task200_r32_strip.log"
ls -la "$OUT/.stripped_real_quant" && du -sh "$OUT"
echo "=========== [$(date)] TASK200 PHASE1 DONE ==========="
