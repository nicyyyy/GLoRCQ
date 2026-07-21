"""Run the full-decode-graph runner's _decode_forward EAGERLY (not captured) so
aten ops emit with record_shapes — pinpoints the dominant index kernel's source."""
import os, sys, torch
sys.path.insert(0, "/home/qyyang/repo/GLoRCQ")
REAL = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
os.environ.setdefault("GLORCQ_DEEPSEEK_GATHER", "1")
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
    # prefill to populate cache, then drive _decode_forward eagerly
    _ = runner._prefill(ids)
    runner.cache.mode = "decode"
    runner._set_moe_graph_mode(True)
    runner.cache.cache_position.fill_(128)
    runner.static_position_ids.fill_(128)
    for _ in range(4):
        _ = runner._decode_forward()
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        for _ in range(N):
            _ = runner._decode_forward()
    torch.cuda.synchronize()
rows = []
for e in prof.key_averages(group_by_input_shape=True):
    cuda_us = getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
    if cuda_us <= 0:
        continue
    rows.append((cuda_us, e.key, str(getattr(e, "input_shapes", ""))[:55]))
print("\n=== TOP OPS by CUDA self-time (eager _decode_forward) ===", flush=True)
for us, key, shp in sorted(rows, key=lambda x: -x[0])[:28]:
    print(f"  {us/1000:8.1f} ms  {key:40s} {shp}", flush=True)
