"""Attribute the DeepSeek full-decode-graph runner's _decode_forward time per
section (attention vs MoE-block vs other) with CUDA-event buckets, run EAGERLY
(not captured) so per-module timing is possible. Pinpoints whether the big
index_elementwise kernel lives in attention (KV-cache index_copy / rope) or the
MoE top-k gather."""
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


class Bucket:
    def __init__(s): s.pairs=[];
    def wrap(s, fn):
        def inner(*a, **k):
            e0=torch.cuda.Event(enable_timing=True); e1=torch.cuda.Event(enable_timing=True)
            e0.record(); r=fn(*a,**k); e1.record(); s.pairs.append((e0,e1)); return r
        return inner
    def ms(s): return sum(a.elapsed_time(b) for a,b in s.pairs)
    def reset(s): s.pairs=[]

battn=Bucket(); bmoe=Bucket(); bcache=Bucket()
from inference.moe_block import GraphCompatibleMoeBlock
for layer in runner.layers:
    layer.self_attn.forward = battn.wrap(layer.self_attn.forward)
for m in model.modules():
    if isinstance(m, GraphCompatibleMoeBlock):
        m.forward = bmoe.wrap(m.forward)
runner.cache.update = bcache.wrap(runner.cache.update)   # KV-cache index_copy
# Time RoPE (module-global apply_rotary_pos_emb in the trust_remote_code modeling)
brope=Bucket()
_mod = sys.modules[type(model).__module__]
if hasattr(_mod, "apply_rotary_pos_emb"):
    _mod.apply_rotary_pos_emb = brope.wrap(_mod.apply_rotary_pos_emb)

torch.manual_seed(0)
ids = torch.randint(0, 30000, (1,128), device="cuda:0")
with torch.no_grad():
    _ = runner._prefill(ids)
    runner.cache.mode="decode"; runner._set_moe_graph_mode(True)
    runner.cache.cache_position.fill_(128); runner.static_position_ids.fill_(128)
    for _ in range(4): _=runner._decode_forward()
    torch.cuda.synchronize(); battn.reset(); bmoe.reset(); bcache.reset(); brope.reset()
    t0=torch.cuda.Event(enable_timing=True); t1=torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(N): _=runner._decode_forward()
    t1.record(); torch.cuda.synchronize()
tot=t0.elapsed_time(t1)
a=battn.ms(); mo=bmoe.ms()
print(f"\n=== _decode_forward sections ({N} steps, eager) ===", flush=True)
print(f"  total     {tot:8.1f} ms  ({tot/N:.2f} ms/tok)")
print(f"  attention {a:8.1f} ms  {a/tot*100:5.1f}%  (q/k/v/o gptq + rope + KV index_copy + sdpa)")
print(f"  moe-block {mo:8.1f} ms  {mo/tot*100:5.1f}%  (top-6 gather + vq4 + LoRA + shared)")
print(f"  other     {tot-a-mo:8.1f} ms  {(tot-a-mo)/tot*100:5.1f}%  (embed + mask + norm + lm_head)")
print(f"  [KV-cache index_copy]  {bcache.ms():8.1f} ms  {bcache.ms()/tot*100:5.1f}%"); print(f"  [RoPE apply_rotary]    {brope.ms():8.1f} ms  {brope.ms()/tot*100:5.1f}%")
