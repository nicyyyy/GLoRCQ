#!/usr/bin/env python
"""Retrying uploader for the Qwen1.5 r32 real-quant HF replacement.

Bypasses the CLI's create_repo preflight (the repo already exists; that call
is what kept hitting HF-side 504s) and goes straight to HfApi.upload_folder.
Retries on network/5xx errors every RETRY_WAIT seconds, up to MAX_TRIES.

Usage:  export HF_TOKEN=hf_xxx
        .venv/bin/python exp/cluster/hf_upload_r32_retry.py
"""
import os
import sys
import time

SRC = "/mnt/Data/yqy/resource_dir/glorcq_paper_exp/qwen15_r32_real"
REPO = "Tsingyow/GLoRCQ-qwen1.5-moe-a2.7b-fair-grassmann-real"
MSG = ("Replace r20 artifact with r32 canonical: rank=32, G=128, attn VQ2 "
       "(2.152 bits), fake-quant PPL 7.138 - same operating point as paper Table 1")
MAX_TRIES = 12
RETRY_WAIT = 600  # 10 min

token = os.environ.get("HF_TOKEN")
if not token:
    sys.exit("set HF_TOKEN env var before running")

from huggingface_hub import HfApi  # noqa: E402

api = HfApi(token=token)
for attempt in range(1, MAX_TRIES + 1):
    try:
        print(f"[{time.strftime('%F %T')}] attempt {attempt}/{MAX_TRIES}: "
              f"upload_folder {SRC} -> {REPO}", flush=True)
        url = api.upload_folder(
            folder_path=SRC,
            repo_id=REPO,
            repo_type="model",
            commit_message=MSG,
            # mirror-delete the r20 leftovers not present in the r32 export
            delete_patterns=["model-*.safetensors", "cross_layer_info.pt"],
        )
        print(f"[{time.strftime('%F %T')}] SUCCESS: {url}", flush=True)
        sys.exit(0)
    except Exception as e:  # network / 5xx — retry
        print(f"[{time.strftime('%F %T')}] attempt {attempt} failed: "
              f"{type(e).__name__}: {e}", flush=True)
        if attempt < MAX_TRIES:
            time.sleep(RETRY_WAIT)

sys.exit(f"gave up after {MAX_TRIES} attempts")
