#!/bin/bash
# ===========================================================================
# Push the DeepSeek-MoE-16B CLASP real-quant checkpoint to HF.
#   repo = Tsingyow/GLoRCQ-deepseek-moe-16b-real  (GLoRCQ-<model>-real convention)
# Reuses exp/upload_deepseek_real.py via env overrides. The real dir must already
# contain the trust_remote_code .py files (deepseek_moe16b_real.sh copies them);
# without modeling_deepseek.py + configuration_deepseek.py the repo can't be
# loaded with trust_remote_code from HF.
#
# NETWORK ONLY (no GPU) — but per project policy the main agent runs it. Token
# via env HF_TOKEN (never hardcoded).
#
# Usage:
#   HF_TOKEN=... bash exp/deepseek_moe16b_upload.sh
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

: "${HF_TOKEN:?export HF_TOKEN=... first (read+write HF token)}"
LOCAL=/mnt/Data/yqy/resource_dir/glorcq_paper_exp/deepseek_moe16b_real

# Sanity: refuse to upload if remote-code files are missing (would break loading).
for f in modeling_deepseek.py configuration_deepseek.py config.json cross_layer_info.pt; do
    [ -f "$LOCAL/$f" ] || { echo "MISSING $LOCAL/$f — run deepseek_moe16b_real.sh first"; exit 1; }
done

GLORCQ_HF_REPO=Tsingyow/GLoRCQ-deepseek-moe-16b-real \
GLORCQ_LOCAL_DIR="$LOCAL" \
    .venv/bin/python exp/upload_deepseek_real.py

echo "MOE16B_UPLOAD_DONE"
