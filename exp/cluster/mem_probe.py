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


def _rss():
    """Host resident set size of THIS process (GB), via psutil."""
    try:
        import psutil
        return f"{psutil.Process().memory_info().rss / 1024**3:.2f} GB"
    except Exception as e:
        return f"psutil err: {e}"


def mark(label, device):
    res = torch.cuda.memory_allocated(device) / 1024**3
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    resv = torch.cuda.memory_reserved(device) / 1024**3
    print(f"\n===[MEMPROBE] {label}===")
    print(f"    torch alloc = {res:6.2f} GB | reserved = {resv:6.2f} GB | "
          f"peak alloc = {peak:6.2f} GB")
    print(f"    nvidia-smi  = {_smi()}")
    print(f"    host RSS    = {_rss()}", flush=True)


def report_u_pool(model):
    """Measure the dequantized shared-U pool + per-expert SV sizes (task #199).

    Counts each shared tensor ONCE (dedupe by untyped_storage data_ptr) —
    the whole point of the cross-layer shared pool. Reports:
      - global concatenated pools installed on MoE blocks
        (moe._global_pool_gate / _global_pool_up; one tensor per wtype,
        shared by ALL layers)
      - unique per-cluster U tensors referenced by GLoRCQLinear.U
        (covers gate/up/down; gate+up duplicate the pool contents,
        down_proj has no concatenated pool)
      - per-expert SV tensors (GLoRCQLinear.SV, NOT shared)
    """
    from inference.moe_block import GraphCompatibleMoeBlock
    from inference.quantized_linear import GLoRCQLinear

    def _key(t):
        return (t.untyped_storage().data_ptr(), t.data_ptr(),
                t.numel(), t.element_size())

    pool_seen, pool_bytes, n_pool = set(), 0, 0
    u_seen, u_bytes, n_u_refs = set(), 0, 0
    sv_seen, sv_bytes, n_sv_refs = set(), 0, 0
    n_moe = 0
    for m in model.modules():
        if isinstance(m, GraphCompatibleMoeBlock):
            n_moe += 1
            for attr in ("_global_pool_gate", "_global_pool_up",
                         "_global_pool_down"):
                t = getattr(m, attr, None)
                if t is None:
                    continue
                k = _key(t)
                if k not in pool_seen:
                    pool_seen.add(k)
                    pool_bytes += t.numel() * t.element_size()
                    n_pool += 1
                    print(f"    [u-pool] unique pool tensor {attr}: "
                          f"shape={tuple(t.shape)} dtype={t.dtype} "
                          f"{t.numel()*t.element_size()/1024**2:.2f} MiB")
        if isinstance(m, GLoRCQLinear):
            if m.U is not None:
                n_u_refs += 1
                k = _key(m.U)
                if k not in u_seen:
                    u_seen.add(k)
                    u_bytes += m.U.numel() * m.U.element_size()
            if m.SV is not None:
                n_sv_refs += 1
                k = _key(m.SV)
                if k not in sv_seen:
                    sv_seen.add(k)
                    sv_bytes += m.SV.numel() * m.SV.element_size()

    print(f"\n===[MEMPROBE] shared-U pool accounting (deduped by data_ptr)===")
    print(f"    MoE blocks scanned            = {n_moe}")
    print(f"    global concat pools (gate/up) = {n_pool} tensors, "
          f"{pool_bytes/1024**2:.2f} MiB")
    print(f"    unique cluster U tensors      = {len(u_seen)} "
          f"(from {n_u_refs} GLoRCQLinear refs), {u_bytes/1024**2:.2f} MiB")
    print(f"    per-expert SV tensors         = {len(sv_seen)} unique "
          f"(from {n_sv_refs} refs), {sv_bytes/1024**2:.2f} MiB")
    print(f"    NOTE: concat pools are COPIES of the per-cluster U tensors "
          f"(gate/up only); both live on GPU → dequantized-U total = "
          f"{(pool_bytes+u_bytes)/1024**2:.2f} MiB", flush=True)


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
    report_u_pool(model)

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
