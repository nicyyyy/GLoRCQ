"""Identify the dominant index_elementwise kernel by printing top ATEN ops with
input shapes (record_shapes) inside the DeepSeek gather full_graph replay."""
import os, sys, torch
sys.path.insert(0, "/home/qyyang/repo/GLoRCQ")
REAL = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
from inference import deepseek_support
from inference.model_builder import load_glorcq_model
deepseek_support.patch_cache_compat()
ret = load_glorcq_model(REAL, device="cuda:0")
model = ret[0] if isinstance(ret, tuple) else ret
model.eval()
runner = deepseek_support.install_full_decode_graph(model, max_seq_len=384, device="cuda:0")
torch.manual_seed(0)
ids = torch.randint(0, 30000, (1, 128), device="cuda:0")
with torch.no_grad():
    for _ in range(2):
        _ = runner.generate(ids, max_new_tokens=32)
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        _ = runner.generate(ids, max_new_tokens=N)
    torch.cuda.synchronize()
rows = []
for e in prof.key_averages(group_by_input_shape=True):
    cuda_us = getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
    if cuda_us <= 0:
        continue
    if e.key.startswith("aten::") or "index" in e.key.lower():
        rows.append((cuda_us, e.key, str(getattr(e, "input_shapes", ""))[:60]))
print("\n=== TOP ATEN OPS by CUDA time (with input shapes) ===", flush=True)
for us, key, shp in sorted(rows, key=lambda x: -x[0])[:25]:
    print(f"  {us/1000:8.1f} ms  {key:32s} {shp}", flush=True)
