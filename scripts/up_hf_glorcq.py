"""Batch-upload the 6 GLoRCQ quantized checkpoints to Hugging Face.

Structure of BASE_DIR (staged with symlinks):
  GLoRCQ-qwen1.5-moe-a2.7b-fake / -real
  GLoRCQ-mixtral-8x7b-fake / -real
  GLoRCQ-qwen3-30b-a3b-fake / -real

The 'fake' variant contains only fake-quant safetensors + configs
(loadable with `AutoModelForCausalLM.from_pretrained`); the 'real'
variant additionally contains `cross_layer_info.pt` for GLoRCQ's
`inference.model_builder.load_glorcq_model` path.

Adapted from /home/qyyang/resource_dir/FluxBin_quant_model/up_hf.py.
"""
import os
import sys
import time
from huggingface_hub import HfApi, HfFolder, upload_folder

HF_USERNAME = "Tsingyow"
BASE_DIR    = "/home/qyyang/resource_dir/hf_upload_staging"
# Never hardcode tokens (this file once did — that token is revoked; the repo
# is public, so anything committed here lives forever in git history).
HF_TOKEN    = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    HfFolder.save_token(HF_TOKEN)
elif not HfFolder.get_token():
    raise SystemExit("No HF_TOKEN found — export HF_TOKEN=... before running")

api = HfApi()


def upload_one(folder_name: str) -> bool:
    local_path = os.path.join(BASE_DIR, folder_name)
    if not os.path.isdir(local_path):
        print(f"❌ missing local dir: {local_path}", flush=True)
        return False
    repo_id = f"{HF_USERNAME}/{folder_name}"

    print("-" * 60, flush=True)
    print(f"🚀 {folder_name}", flush=True)
    print(f"   local:  {local_path}", flush=True)
    print(f"   repo:   {repo_id}", flush=True)

    # Report the effective payload size (resolving symlinks).
    total_bytes = 0
    n_files = 0
    for f in os.listdir(local_path):
        full = os.path.join(local_path, f)
        try:
            total_bytes += os.path.getsize(os.path.realpath(full))
            n_files += 1
        except OSError:
            pass
    print(f"   size:   {total_bytes/1e9:.1f} GB across {n_files} files", flush=True)

    try:
        api.create_repo(repo_id=repo_id, private=False, exist_ok=True, repo_type="model")
    except Exception as e:
        print(f"   ⚠️ create_repo warning: {e}", flush=True)

    t0 = time.time()
    try:
        upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload GLoRCQ quantized checkpoint: {folder_name}",
            # Skip .cache subdirs etc.
            ignore_patterns=[".cache/**", "*.log", "*_phase1_cache.pt"],
        )
    except Exception as e:
        print(f"   ❌ upload failed: {e}", flush=True)
        return False
    dt = time.time() - t0
    mb_per_s = total_bytes / 1e6 / dt if dt > 0 else 0.0
    print(f"   🎉 uploaded in {dt/60:.1f} min ({mb_per_s:.1f} MB/s)", flush=True)
    return True


def main():
    if not os.path.isdir(BASE_DIR):
        print(f"❌ BASE_DIR missing: {BASE_DIR}")
        sys.exit(1)
    all_folders = sorted(f.name for f in os.scandir(BASE_DIR) if f.is_dir())
    glorcq_folders = [f for f in all_folders if f.startswith("GLoRCQ-")]
    # Optional: allow subset via argv (e.g. only re-run failed ones)
    if len(sys.argv) > 1:
        wanted = set(sys.argv[1:])
        glorcq_folders = [f for f in glorcq_folders if f in wanted]

    print(f"📂 will upload {len(glorcq_folders)} repos: {glorcq_folders}", flush=True)
    ok, fail = [], []
    for folder in glorcq_folders:
        if upload_one(folder):
            ok.append(folder)
        else:
            fail.append(folder)

    print("\n" + "=" * 60, flush=True)
    print(f"✅ done: {len(ok)} success, {len(fail)} failed", flush=True)
    if ok:
        print("   success:", ok, flush=True)
    if fail:
        print("   failed :", fail, flush=True)


if __name__ == "__main__":
    main()
