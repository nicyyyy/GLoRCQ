"""Kernel-level CUDA time breakdown INSIDE the DeepSeek full decode-step CUDA
graph (the real-quant `full_graph` config). Eager method-wrapping can't see into
a captured graph, so we use torch.profiler over graph replays and bucket kernels
by name into the architectural components.

Buckets (by CUDA kernel name substring):
  routed_vq4 : vq4 grouped-GEMV dequant kernels (routed experts)  [our contribution]
  attn_sdpa  : attention scaled-dot-product / flash / softmax kernels
  gemm       : cuBLAS/cutlass GEMMs (attn q/k/v/o proj, shared MLP, LoRA, lm_head)
  elementwise: rope, rmsnorm, add, copy, cast, index
  other      : anything else

Usage: python exp/profile_graph_kernels.py <real_dir> [n_decode]
"""
import os, sys, time, argparse, torch
sys.path.insert(0, "/home/qyyang/repo/GLoRCQ")

ap = argparse.ArgumentParser()
ap.add_argument("real")
ap.add_argument("n_decode", nargs="?", type=int, default=64)
ap.add_argument("--prompt_len", type=int, default=128)
ap.add_argument("--max_seq_len", type=int, default=384)
args = ap.parse_args()
DEV = "cuda:0"


def _p(*a):
    print(*a, flush=True)


from inference import deepseek_support
from inference.model_builder import load_glorcq_model

deepseek_support.patch_cache_compat()
_p(f"[gkprofile] loading {args.real}")
t0 = time.time()
ret = load_glorcq_model(args.real, device=DEV)
model = ret[0] if isinstance(ret, tuple) else ret
model.eval()
torch.cuda.synchronize()
_p(f"[gkprofile] load {time.time()-t0:.0f}s")

runner = deepseek_support.install_full_decode_graph(
    model, max_seq_len=args.max_seq_len, device=DEV)

torch.manual_seed(0)
ids = torch.randint(0, 30000, (1, args.prompt_len), device=DEV)

with torch.no_grad():
    # warmup: triggers prefill + capture + steady replays
    for _ in range(2):
        _ = runner.generate(ids, max_new_tokens=32)

    # timed wall for tok/s
    torch.cuda.synchronize(); w0 = time.time()
    out = runner.generate(ids, max_new_tokens=args.n_decode)
    torch.cuda.synchronize(); wall = time.time() - w0
    n = out.shape[1] - args.prompt_len
    _p(f"[gkprofile] full_graph wall {wall:.3f}s  {n/wall:.2f} tok/s")

    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        _ = runner.generate(ids, max_new_tokens=args.n_decode)
    torch.cuda.synchronize()

evts = prof.key_averages()
buckets = {"routed_vq4": 0.0, "attn_sdpa": 0.0, "gemm": 0.0,
           "elementwise": 0.0, "other": 0.0}
rows = []
total = 0.0
for e in evts:
    cuda_us = getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
    if cuda_us <= 0:
        continue
    total += cuda_us
    name = e.key.lower()
    if "vq4" in name or "grouped_gemv" in name or "dequant" in name:
        b = "routed_vq4"
    elif ("sdpa" in name or "attention" in name or "flash" in name or
          "softmax" in name or "fmha" in name or "scaled_dot" in name):
        b = "attn_sdpa"
    elif ("gemm" in name or "cutlass" in name or "cublas" in name or
          "ampere" in name or "sgemm" in name or "gemv" in name or "matmul" in name):
        b = "gemm"
    elif ("elementwise" in name or "vectorized" in name or "copy" in name or
          "index" in name or "cast" in name or "add" in name or "mul" in name or
          "rms" in name or "norm" in name or "rope" in name or "cat" in name):
        b = "elementwise"
    else:
        b = "other"
    buckets[b] += cuda_us
    rows.append((cuda_us, e.key))

_p("\n" + "=" * 70)
_p(f"  IN-GRAPH KERNEL BREAKDOWN  ({args.n_decode} decode steps, full_graph)")
_p("=" * 70)
for b, us in sorted(buckets.items(), key=lambda x: -x[1]):
    _p(f"    {b:14s} {us/1000:9.1f} ms  {us/total*100:5.1f}%")
_p(f"    {'TOTAL':14s} {total/1000:9.1f} ms")
_p("\n  --- TOP 20 KERNELS ---")
for us, name in sorted(rows, key=lambda x: -x[0])[:20]:
    _p(f"    {us/1000:9.1f} ms  {us/total*100:5.1f}%  {name[:70]}")
_p("=" * 70)
