"""GPU memory probe for GLoRCQ real-quant load path (task #197).

Loads a real-quant model via the SAME code path as inference/eval_speed.py
(load_glorcq_model) and logs GPU memory (resident torch.cuda.memory_allocated
AND peak max_memory_allocated) at each checkpoint, then triggers the CUDA-graph
cache build + capture (amplifier #1) and a single decode step.

Run inside tmux test:0 (GPU 4). Example:
  CUDA_VISIBLE_DEVICES=4 PYTORCH_ALLOC_CONF=expandable_segments:True \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/qyyang/repo/GLoRCQ/.venv/bin/python exp/cluster/mem_probe.py \
      --model_path /mnt/Data/yqy/resource_dir/hf_dl/GLoRCQ-mixtral-8x7b-fair-grassmann-real \
      --max_seq_len 384
"""
import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch


def _smi():
    """Resident GPU memory used by THIS process (MiB), from nvidia-smi."""
    try:
        pid = os.getpid()
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], text=True)
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if parts and parts[0] == str(pid):
                return f"{parts[1]} MiB (pid {pid})"
        return f"pid {pid} not in nvidia-smi apps yet"
    except Exception as e:
        return f"nvidia-smi err: {e}"


def mark(label, device):
    res = torch.cuda.memory_allocated(device) / 1024**3
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    resv = torch.cuda.memory_reserved(device) / 1024**3
    print(f"\n===[MEMPROBE] {label}===")
    print(f"    torch alloc = {res:6.2f} GB | reserved = {resv:6.2f} GB | "
          f"peak alloc = {peak:6.2f} GB")
    print(f"    nvidia-smi  = {_smi()}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_seq_len", type=int, default=384)
    ap.add_argument("--skip_graph", action="store_true")
    args = ap.parse_args()

    dev = args.device
    mark("baseline (before load)", dev)

    from inference.model_builder import load_glorcq_model
    # load_glorcq_model's _lap now prints per-phase GPU memory internally.
    model = load_glorcq_model(args.model_path, device=dev)
    mark("after load_glorcq_model (RESIDENT model)", dev)

    # Tokenizer-free single decode step via a tiny random prompt.
    import torch as _t
    input_ids = _t.randint(0, 100, (1, 8), device=dev)
    with _t.no_grad():
        out = model(input_ids=input_ids, use_cache=True, return_dict=False)
    _t.cuda.synchronize()
    mark("after 1 standard forward (prefill)", dev)

    if not args.skip_graph:
        from inference.graph_wrapper import GLoRCQGraphWrapper
        wrapper = GLoRCQGraphWrapper(
            model, max_batch_size=1, max_seq_len=args.max_seq_len)
        mark("after StaticCache alloc (wrapper init)", dev)
        wrapper.capture_graph()   # triggers _build_graph_cache_vq4 = amplifier #1
        _t.cuda.synchronize()
        mark("after capture_graph (graph cache + capture)", dev)

        # One decode replay
        nt = _t.randint(0, 100, (1, 1), device=dev)
        _ = wrapper.replay(nt, 8)
        _t.cuda.synchronize()
        mark("after 1 graph replay (decode)", dev)

    print("\n[MEMPROBE] done.", flush=True)


if __name__ == "__main__":
    main()
