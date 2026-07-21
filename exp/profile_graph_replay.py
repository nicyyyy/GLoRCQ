"""Kernel breakdown of DeepSeek DECODE-ONLY graph replays (excludes prefill).
Prefill + capture happen in warmup; the profiler wraps ONLY runner.graph.replay()
so we see the true steady-state decode kernels (the earlier profiler included the
one-time 128-token prefill, which inflates index/scatter kernels)."""
import os, sys, torch
sys.path.insert(0, "/home/qyyang/repo/GLoRCQ")
REAL = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 128
os.environ.setdefault("GLORCQ_DEEPSEEK_GATHER", "1")
from inference import deepseek_support
from inference.model_builder import load_glorcq_model
deepseek_support.patch_cache_compat()
ret = load_glorcq_model(REAL, device="cuda:0")
model = ret[0] if isinstance(ret, tuple) else ret; model.eval()
runner = deepseek_support.install_full_decode_graph(model, max_seq_len=384, device="cuda:0")
torch.manual_seed(0)
ids = torch.randint(0, 30000, (1,128), device="cuda:0")
with torch.no_grad():
    _ = runner.generate(ids, max_new_tokens=8)   # prefill + capture + a few replays (warmup)
    torch.cuda.synchronize()
    # steady-state replay loop (decode only), pos fixed mid-window
    runner.static_input_id.fill_(5); runner.static_position_ids.fill_(200); runner.cache.cache_position.fill_(200)
    for _ in range(8):
        runner.graph.replay()
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(N):
            runner.graph.replay()
    torch.cuda.synchronize()
buckets={"index_elem":0.0,"indexSelect":0.0,"scatter_gather":0.0,"vq4":0.0,"gptq_attn":0.0,
         "gemm":0.0,"sdpa":0.0,"copy_cast":0.0,"other":0.0}
rows=[]; total=0.0
for e in prof.key_averages():
    us=getattr(e,"self_device_time_total",0) or getattr(e,"self_cuda_time_total",0)
    if us<=0: continue
    total+=us; n=e.key.lower(); rows.append((us,e.key))
    if "index_elementwise" in n: b="index_elem"
    elif "indexselect" in n: b="indexSelect"
    elif "scatter_gather" in n or "gather" in n: b="scatter_gather"
    elif "vq4" in n: b="vq4"
    elif "gptq" in n: b="gptq_attn"
    elif "gemm" in n or "gemv" in n or "cutlass" in n or "sgemm" in n or "fmha" in n or "attention" in n: b="gemm" if "fmha" not in n and "attention" not in n else "sdpa"
    elif "copy" in n or "cast" in n or "elementwise" in n or "mul" in n or "add" in n: b="copy_cast"
    else: b="other"
    buckets[b]+=us
print(f"\n=== DECODE-ONLY graph replay kernels ({N} replays) ===",flush=True)
for b,us in sorted(buckets.items(),key=lambda x:-x[1]):
    print(f"  {b:16s} {us/1000:8.1f} ms  {us/total*100:5.1f}%")
print(f"  {'TOTAL':16s} {total/1000:8.1f} ms   ({total/1000/N:.3f} ms/replay)")
print("\n  --- top 15 kernels ---")
for us,name in sorted(rows,key=lambda x:-x[0])[:15]:
    print(f"    {us/1000:8.1f} ms  {us/total*100:5.1f}%  {name[:64]}")
