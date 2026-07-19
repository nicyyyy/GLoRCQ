#!/bin/bash
# ===========================================================================
# bs=1 decode throughput for DeepSeek-MoE-16B real-quant vs fp16 eager.
# Uses the stable harness (exp/bench_deepseek_stable.py): warmup-discard + N
# timed runs, median reported; each config in a fresh process. prompt128/gen128.
#
# THE KEY DELIVERABLE of this port. DeepSeek-MoE-16B has STANDARD MHA (unlike
# V2-Lite's MLA), so the full-model decode graph (attention + MoE captured
# together, via the SYMMETRIC _MHAStaticCache in inference/deepseek_support.py)
# should apply — the analogue of the Qwen bs=1 4-5x win (many-small-expert MoE
# is launch-bound; the graph eliminates eager per-expert dispatch AND the
# per-step attention launch overhead). Configs:
#   fp16       : fp16 eager baseline
#   eager      : real-quant, no graphs
#   moe_graph  : per-MoE-block CUDA graphs + eager attention
#   full_graph : full decode-step graph (attention + MoE together)  <-- headline
# The full_graph config also prints a byte-identical check vs eager (see the
# python harness / exp/decode_graph_deepseek.py). WATCH FOR: capture success
# (this could not be verified without a GPU) and full_graph >> fp16.
#
# NO-GPU-SAFE: LAUNCHES GPU JOBS. Run ONLY when GPU4 free, in tmux `test:0`.
#   Requires exp/deepseek_moe16b_real.sh to have produced the real dir.
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

REAL=/mnt/Data/yqy/resource_dir/glorcq_paper_exp/deepseek_moe16b_real
FP16=/mnt/Data/yqy/resource_dir/deepseek-moe-16b
OUT=logs/deepseek_moe16b
mkdir -p "$OUT"
export HF_HOME=/mnt/Data/yqy/resource_dir/hf_cache
GPU=${CUDA_VISIBLE_DEVICES:-4}
PROMPT=${PROMPT_LEN:-128}
GEN=${GEN_LEN:-128}
MAXSEQ=${MAX_SEQ_LEN:-384}     # full_graph fixed KV window; must be > prompt+gen

for CFG in fp16 eager moe_graph full_graph; do
    echo "[$(date)] === speed config=$CFG ==="
    CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        .venv/bin/python exp/bench_deepseek_stable.py \
        --config $CFG --prompt_len $PROMPT --gen_len $GEN --max_seq_len $MAXSEQ \
        --real_path "$REAL" --fp16_path "$FP16" \
        2>&1 | tee "$OUT/speed_${CFG}.log"
done

echo "[$(date)] MOE16B_SPEED_DONE"
echo "Headline = full_graph median tok/s vs fp16 median tok/s (expect a Qwen-like"
echo "bs=1 win because MHA + many-small-experts is launch-bound, unlike V2-Lite MLA)."
