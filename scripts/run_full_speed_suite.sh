#!/bin/bash
# One-shot: env + download + real-quant speed sweep + fp16 baseline + speedup table.
# Runs the full GLoRCQ decode-speed benchmark on a fresh GPU machine.
#
#   git clone -b exp/e11-tileq-cross-layer https://github.com/nicyyyy/GLoRCQ.git
#   bash GLoRCQ/scripts/run_full_speed_suite.sh
#
# Env knobs (all optional):
#   WORK=/path         work dir (default /workspace/glorcq_speed, else ~/glorcq_speed)
#   BATCH_SIZES="1 4 16 64"   batch sweep (default)
#   SKIP_FP16=1        skip the fp16 baseline (real-quant only; no 171 GB download)
#   GLORCQ_MIXTRAL_GRAPH=1    opt into Mixtral's top-k gather graph (default: standard decode)
#
# Disk: ~40 GB real-quant + ~171 GB fp16 (safetensors) => budget ~250 GB.
# GPU: >=48 GB runs Qwen1.5/Qwen3 real-quant; Mixtral fp16 (~87 GB) needs an
#      H200-class single card. Big-batch fp16 may OOM (recorded, sweep continues).
set +e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

step() { echo ""; echo "############################################################"; echo "# $*"; echo "############################################################"; }

step "[1/4] install env + build CUDA kernels"
bash "$SCRIPT_DIR/vast_1_install.sh" || { echo "install failed"; exit 1; }

step "[2/4] download real-quant checkpoints (~40 GB)"
bash "$SCRIPT_DIR/vast_2_download.sh" || { echo "download failed"; exit 1; }

step "[3/4] real-quant decode speed sweep (batch ${BATCH_SIZES:-1 4 16 64})"
bash "$SCRIPT_DIR/vast_3_speed.sh"

if [ "${SKIP_FP16:-0}" = "1" ]; then
    step "[4/4] fp16 baseline SKIPPED (SKIP_FP16=1)"
else
    step "[4/4] fp16 baseline (~171 GB) + speedup table"
    bash "$SCRIPT_DIR/vast_4_fp16_baseline.sh"
fi

echo ""
echo "===== DONE. Results ====="
WORK=${WORK:-$([ -d /workspace ] && [ -w /workspace ] && echo /workspace/glorcq_speed || echo "$HOME/glorcq_speed")}
echo "  real-quant logs : $WORK/speed_results/*.log"
echo "  fp16 logs       : $WORK/fp16_results/*.log"
echo "  (the speedup table was printed at the end of step 4)"
