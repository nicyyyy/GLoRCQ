#!/usr/bin/env python3
"""Upload the DeepSeek-V2-Lite GLoRCQ real-quant checkpoint to HF.
Repo id follows the existing GLoRCQ-<model>-real convention (Tsingyow namespace).
Token via env HF_TOKEN only (never hardcoded)."""
import os, sys, time
from huggingface_hub import HfApi, HfFolder, upload_folder

HF_USERNAME = "Tsingyow"
# Defaults = V2-Lite (backward compatible). Override via env to reuse for other
# DeepSeek checkpoints, e.g. DeepSeek-MoE-16B:
#   GLORCQ_HF_REPO=Tsingyow/GLoRCQ-deepseek-moe-16b-real \
#   GLORCQ_LOCAL_DIR=/mnt/Data/yqy/resource_dir/glorcq_paper_exp/deepseek_moe16b_real \
#   HF_TOKEN=... python exp/upload_deepseek_real.py
REPO = os.environ.get("GLORCQ_HF_REPO", f"{HF_USERNAME}/GLoRCQ-deepseek-v2-lite-real")
LOCAL = os.environ.get("GLORCQ_LOCAL_DIR",
                       "/mnt/Data/yqy/resource_dir/glorcq_paper_exp/deepseek_v2lite_real")

tok = os.environ.get("HF_TOKEN", "")
if tok:
    HfFolder.save_token(tok)
elif not HfFolder.get_token():
    raise SystemExit("No HF_TOKEN — export HF_TOKEN=... before running")

api = HfApi()
api.create_repo(repo_id=REPO, private=False, exist_ok=True, repo_type="model")
print(f"uploading {LOCAL} -> {REPO}", flush=True)
t0 = time.time()
COMMIT_MSG = os.environ.get(
    "GLORCQ_COMMIT_MSG",
    "Upload GLoRCQ DeepSeek-V2-Lite real-quant checkpoint (experts VQ4 + cross-layer Grassmann LoRA; attn/shared/dense fp16)",
)
upload_folder(
    folder_path=LOCAL,
    repo_id=REPO,
    repo_type="model",
    commit_message=COMMIT_MSG,
    ignore_patterns=[".cache/**", "*.log", "*_phase1_cache.pt"],
)
print(f"UPLOAD_DONE in {(time.time()-t0)/60:.1f} min -> https://huggingface.co/{REPO}", flush=True)
