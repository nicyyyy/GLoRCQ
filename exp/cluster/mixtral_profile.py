"""
Task #198 Phase-1 PROFILE probe (read-only; not permanent instrumentation).

Loads a GLoRCQ real-quant model via model_builder and runs a graph-free
STANDARD decode loop, attributing GPU time to:
  - attn    (GLoRCQLinear gptq q/k/v/o + attn LoRA + rope/sdpa)
  - gate_up (_batched_proj_forward: vq4 grouped-GEMV + LoRA for gate & up)
  - down    (_batched_down_forward: vq4 grouped-GEMV + LoRA for down_proj)
  - other   (embed, router, rmsnorm, sampling, ...)  = total - the three above

Plus kernel-level counters (bucketed by n_cb so down (large in_d) is split from
gate/up) and a micro-benchmark of the int32->int8 GPTQ cast that Edit A removes.

Usage: python exp/cluster/mixtral_profile.py <model_dir> [n_decode]
Uses whatever GPU CUDA_VISIBLE_DEVICES selects (retry script sets =4 -> cuda:0).
"""
import os, sys, time, torch

sys.path.insert(0, "/home/qyyang/repo/GLoRCQ")

MODEL = sys.argv[1]
N_DECODE = int(sys.argv[2]) if len(sys.argv) > 2 else 40
DEV = "cuda:0"

import inference.moe_block as mb
import inference.quantized_linear as qlin
import inference.kernels as K
from inference.model_builder import load_glorcq_model


def _p(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------
# CUDA-event bucket timer (no mid-loop sync; elapsed summed after one sync)
# --------------------------------------------------------------------------
class Bucket:
    def __init__(self):
        self.pairs = []
        self.n = 0

    def wrap(self, fn):
        def inner(*a, **k):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            r = fn(*a, **k)
            e.record()
            self.pairs.append((s, e))
            self.n += 1
            return r
        return inner

    def reset(self):
        self.pairs = []
        self.n = 0

    def ms(self):
        return sum(s.elapsed_time(e) for s, e in self.pairs)


buckets = {"attn": Bucket(), "gate_up": Bucket(), "down": Bucket()}

# Method-level patches (decode-only methods)
mb.GraphCompatibleMoeBlock._batched_proj_forward = buckets["gate_up"].wrap(
    mb.GraphCompatibleMoeBlock._batched_proj_forward)
mb.GraphCompatibleMoeBlock._batched_down_forward = buckets["down"].wrap(
    mb.GraphCompatibleMoeBlock._batched_down_forward)

# Kernel-level counters
ggemv = {}          # n_cb -> Bucket()
_orig_ggemv = K.vq4_dequant_grouped_gemv


def ggemv_patched(x_grouped, codes_cat, centroids_cat, E, N, n_cb, codes_per_cb):
    b = ggemv.setdefault(int(n_cb), Bucket())
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    r = _orig_ggemv(x_grouped, codes_cat, centroids_cat, E, N, n_cb, codes_per_cb)
    e.record()
    b.pairs.append((s, e)); b.n += 1
    return r


K.vq4_dequant_grouped_gemv = ggemv_patched

gptq_b = Bucket()
_orig_gptq = qlin.gptq_dequant_matmul_fused


def gptq_patched(*a, **k):
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    r = _orig_gptq(*a, **k)
    e.record()
    gptq_b.pairs.append((s, e)); gptq_b.n += 1
    return r


qlin.gptq_dequant_matmul_fused = gptq_patched


# --------------------------------------------------------------------------
_p(f"[profile] loading {MODEL}")
t0 = time.time()
model = load_glorcq_model(MODEL, device=DEV)
model.eval()
torch.cuda.synchronize()
_p(f"[profile] load done in {time.time()-t0:.0f}s  "
   f"alloc={torch.cuda.memory_allocated(DEV)/1024**3:.2f}GB "
   f"reserved={torch.cuda.memory_reserved(DEV)/1024**3:.2f}GB")

# Patch attention instance forwards (after load, so all modules exist)
n_attn = 0
for m in model.modules():
    if m.__class__.__name__.endswith("Attention"):
        m.forward = buckets["attn"].wrap(m.forward)
        n_attn += 1
_p(f"[profile] wrapped {n_attn} attention modules")

# GPTQ cast micro-benchmark + value-range confirmation (Edit A safety)
qw = None
for m in model.modules():
    if getattr(m, "quant_type", None) == "gptq" and getattr(m, "qweight_int", None) is not None:
        qw = m.qweight_int
        break
if qw is not None:
    _p(f"[profile] gptq qweight_int dtype={qw.dtype} shape={tuple(qw.shape)} "
       f"min={int(qw.min())} max={int(qw.max())}")
    for _ in range(5):
        _ = qw.to(torch.int8)
    torch.cuda.synchronize(); a = time.time()
    R = 200
    for _ in range(R):
        _ = qw.to(torch.int8)
    torch.cuda.synchronize()
    cast_per = (time.time() - a) / R * 1000.0
    _p(f"[profile] int32->int8 cast per-proj: {cast_per:.4f} ms  (shape {tuple(qw.shape)})")
else:
    cast_per = 0.0
    _p("[profile] no gptq layer found (cannot micro-bench cast)")

# --------------------------------------------------------------------------
# Manual prefill + decode loop (decode-only bucket attribution)
# --------------------------------------------------------------------------
from transformers import DynamicCache
torch.manual_seed(0)
vocab = getattr(model.config, "vocab_size", 32000)
prompt = torch.randint(0, min(vocab, 30000), (1, 128), device=DEV)

with torch.no_grad():
    past = DynamicCache()
    out = model(input_ids=prompt, past_key_values=past, use_cache=True, return_dict=True)
    nxt = out.logits[:, -1:].argmax(-1)
    # warmup decode (not timed)
    for _ in range(4):
        out = model(input_ids=nxt, past_key_values=past, use_cache=True, return_dict=True)
        nxt = out.logits[:, -1:].argmax(-1)

    # reset all counters after warmup
    for b in buckets.values():
        b.reset()
    for b in ggemv.values():
        b.reset()
    gptq_b.reset()

    torch.cuda.synchronize(); td0 = time.time()
    for _ in range(N_DECODE):
        out = model(input_ids=nxt, past_key_values=past, use_cache=True, return_dict=True)
        nxt = out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize(); td1 = time.time()

decode_wall = td1 - td0
tok_s = N_DECODE / decode_wall

# --------------------------------------------------------------------------
_p("\n" + "=" * 64)
_p(f"  PROFILE: {os.path.basename(MODEL)}   ({N_DECODE} decode steps)")
_p("=" * 64)
_p(f"  decode wall: {decode_wall:.3f}s   throughput: {tok_s:.2f} tok/s")
_p(f"  (per-token wall: {decode_wall/N_DECODE*1000:.2f} ms)")

attn_ms = buckets["attn"].ms()
gu_ms = buckets["gate_up"].ms()
dn_ms = buckets["down"].ms()
region_sum = attn_ms + gu_ms + dn_ms
other_ms = max(0.0, decode_wall * 1000 - region_sum)
tot = decode_wall * 1000

_p("\n  --- METHOD-LEVEL (GPU time over decode, % of wall) ---")
for name, ms, n in [("attn (gptq q/k/v/o + LoRA + sdpa)", attn_ms, buckets["attn"].n),
                    ("gate_up (vq4 gGEMV + LoRA)", gu_ms, buckets["gate_up"].n),
                    ("down    (vq4 gGEMV + LoRA)", dn_ms, buckets["down"].n),
                    ("other   (embed/router/norm/sample)", other_ms, 0)]:
    _p(f"    {name:42s} {ms:9.1f} ms  {ms/tot*100:5.1f}%   calls={n}")

_p("\n  --- KERNEL-LEVEL ---")
for ncb in sorted(ggemv):
    b = ggemv[ncb]
    lbl = "down_proj" if ncb == max(ggemv) else "gate/up_proj"
    _p(f"    vq4_grouped_gemv n_cb={ncb:<3d} ({lbl:12s}) {b.ms():9.1f} ms  {b.ms()/tot*100:5.1f}%  calls={b.n}")
_p(f"    gptq_dequant_matmul_fused (attn all)   {gptq_b.ms():9.1f} ms  {gptq_b.ms()/tot*100:5.1f}%  calls={gptq_b.n}")

# Cast total estimate
n_gptq_per_tok = gptq_b.n / N_DECODE if N_DECODE else 0
cast_total = cast_per * gptq_b.n
_p(f"\n  int32->int8 cast: {n_gptq_per_tok:.0f} gptq calls/tok x {cast_per:.4f} ms "
   f"= {cast_total:.1f} ms total ({cast_total/tot*100:.2f}% of decode)  [Edit A removes this]")

_p(f"\n  resident: alloc={torch.cuda.memory_allocated(DEV)/1024**3:.2f}GB "
   f"reserved={torch.cuda.memory_reserved(DEV)/1024**3:.2f}GB")
_p("=" * 64)
