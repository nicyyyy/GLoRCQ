#!/bin/bash
# Evaluate the TileQ_s DeepSeek-V2-Lite fake-quant checkpoint under OUR Table-1
# protocol, reusing GLoRCQ's own eval entry points so the numbers are directly
# comparable to the GLoRCQ V2-Lite row.
#   * WikiText-2 PPL: sliding window, max_length 2048, stride 512.
#   * 5-task zero-shot: ARC-C, ARC-E, PIQA, WinoGrande, HellaSwag; metric acc,
#     batch_size 1, num_fewshot 0, add_bos_token=True.
#
# TileQ saves a STANDARD fp16-shaped HF checkpoint (fake-quant bakes the quantized
# values into the Linear weights + copies the deepseek_v2 remote modeling via
# save_pretrained's auto_map), so it loads with trust_remote_code=True exactly like
# the GLoRCQ fake-quant dir -- no TileQ-specific loader needed.
#
# NOTE (flag provenance): the committed GLoRCQ driver exp/eval_deepseek_v2lite.sh
#   runs zero-shot with the DEFAULTS (no --add_bos, --metric_mode auto -> acc_norm
#   first, batch 8/1). The task spec for Table-1 says add_bos=True + metric acc +
#   batch 1, which is what this script uses. If the curated GLoRCQ V2-Lite number
#   was produced with the driver defaults instead, re-run this with ZS_FLAGS
#   overridden to match, so TileQ_s and GLoRCQ use identical flags.
#
# NO-GPU-SAFE: starts GPU jobs; run only when GPU 4 is free, in tmux `test:0`.
# Usage (after quant): bash exp/tileq_deepseek_v2lite_eval.sh
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

FAKE=/mnt/Data/yqy/resource_dir/glorcq_paper_exp/tileq_s_deepseek_v2lite_fake
OUT=logs/tileq_deepseek_v2lite
mkdir -p "$OUT"
export HF_HOME=/mnt/Data/yqy/resource_dir/hf_cache

GPU=${CUDA_VISIBLE_DEVICES:-4}
ZS_FLAGS=${ZS_FLAGS:-"--num_fewshot 0 --batch_size 1 --add_bos --metric_mode acc"}

echo "[$(date)] === TileQ_s V2-Lite PPL (WikiText-2, 2048/512) ==="
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python evaluate/eval_ppl.py --model_path "$FAKE" --device cuda:0 \
    --max_length 2048 --stride 512 --output_json "$OUT/ppl_tileq_s.json" 2>&1 | tee "$OUT/eval_ppl.log"

echo "[$(date)] === TileQ_s V2-Lite zero-shot 5-task (acc, bs1, add_bos) ==="
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python evaluate/eval_zeroshot.py --model_path "$FAKE" --device cuda:0 \
    --tasks arc_challenge,arc_easy,winogrande,hellaswag,piqa \
    $ZS_FLAGS --output_json "$OUT/zeroshot_tileq_s.json" 2>&1 | tee "$OUT/eval_zs.log"

echo "[$(date)] TILEQ_S_DEEPSEEK_EVAL_DONE  (results in $OUT/*.json)"
