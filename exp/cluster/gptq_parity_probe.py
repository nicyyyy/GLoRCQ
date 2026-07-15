"""
Task #198 Edit-A parity probe: GPTQ qweight int8 store vs legacy int32 store.

Loads a GLoRCQ real-quant model (Qwen1.5-MoE — fast, has GPTQ attn), then:
  1. Value-range audit over ALL gptq layers (must fit int8).
  2. Per-layer consumer parity on a few layers:
       - fused path:  gptq_dequant_matmul_fused(x, qw_int8) vs (x, qw_int8.to(int32))
         (the int32 arg reproduces the legacy stored dtype through the same code)
       - python path: _dequant_gptq() with int8 stored vs int32 stored
  3. Full-model decode-logits parity: 8 greedy decode steps with int8 store,
     then ALL gptq layers swapped to int32 store, rerun, logits must be
     bit-identical (torch.equal).

Usage: python exp/cluster/gptq_parity_probe.py <model_dir>
"""
import sys, time, torch

sys.path.insert(0, "/home/qyyang/repo/GLoRCQ")

MODEL = sys.argv[1]
DEV = "cuda:0"

from inference.model_builder import load_glorcq_model
from inference.kernels import gptq_dequant_matmul_fused


def _p(*a):
    print(*a, flush=True)


_p(f"[parity] loading {MODEL}")
t0 = time.time()
model = load_glorcq_model(MODEL, device=DEV)
model.eval()
torch.cuda.synchronize()
_p(f"[parity] load done in {time.time()-t0:.0f}s  "
   f"alloc={torch.cuda.memory_allocated(DEV)/1024**3:.2f}GB "
   f"reserved={torch.cuda.memory_reserved(DEV)/1024**3:.2f}GB")

gptq_layers = [(n, m) for n, m in model.named_modules()
               if getattr(m, "quant_type", None) == "gptq"
               and getattr(m, "qweight_int", None) is not None]
_p(f"[parity] gptq layers: {len(gptq_layers)}")

# ---- 1. value range + dtype audit --------------------------------------
gmin, gmax = 10**9, -10**9
dtypes = set()
for _, m in gptq_layers:
    dtypes.add(str(m.qweight_int.dtype))
    gmin = min(gmin, int(m.qweight_int.min()))
    gmax = max(gmax, int(m.qweight_int.max()))
_p(f"[parity] qweight dtypes={sorted(dtypes)}  global min={gmin} max={gmax}")
assert -128 <= gmin and gmax <= 127, "VALUES DO NOT FIT INT8 — Edit A unsafe!"

# ---- 2. per-layer consumer parity ---------------------------------------
torch.manual_seed(0)
fail = 0
for name, m in gptq_layers[:4] + gptq_layers[-2:]:
    qw8 = m.qweight_int
    if qw8.dtype != torch.int8:
        _p(f"[parity]   {name}: stored dtype {qw8.dtype} (not int8) — skip")
        continue
    qw32 = qw8.to(torch.int32)
    x = torch.randn(3, m.in_features, dtype=torch.float16, device=qw8.device)

    y8 = gptq_dequant_matmul_fused(x, qw8, m.scales, m.zeros,
                                   m.gptq_groupsize, m.gptq_sym, lora_USV=None)
    y32 = gptq_dequant_matmul_fused(x, qw32, m.scales, m.zeros,
                                    m.gptq_groupsize, m.gptq_sym, lora_USV=None)
    ok_fused = torch.equal(y8, y32)

    # python fallback _dequant_gptq: swap stored tensor
    W8 = m._dequant_gptq()
    m.qweight_int = qw32
    W32 = m._dequant_gptq()
    m.qweight_int = qw8
    ok_py = torch.equal(W8, W32)

    _p(f"[parity]   {name}: fused_equal={ok_fused} python_equal={ok_py}")
    if not (ok_fused and ok_py):
        fail += 1
del qw32, W8, W32
torch.cuda.empty_cache()

# ---- 3. full-model decode-logits parity ---------------------------------
from transformers import DynamicCache
torch.manual_seed(1)
vocab = getattr(model.config, "vocab_size", 32000)
prompt = torch.randint(0, min(vocab, 30000), (1, 32), device=DEV)


def run_decode():
    torch.manual_seed(2)
    logits_seq = []
    with torch.no_grad():
        past = DynamicCache()
        out = model(input_ids=prompt, past_key_values=past,
                    use_cache=True, return_dict=True)
        nxt = out.logits[:, -1:].argmax(-1)
        for _ in range(8):
            out = model(input_ids=nxt, past_key_values=past,
                        use_cache=True, return_dict=True)
            logits_seq.append(out.logits[:, -1, :].clone())
            nxt = out.logits[:, -1:].argmax(-1)
    return torch.cat(logits_seq)


L8 = run_decode()

# swap ALL gptq layers to int32 store (legacy), rerun, restore
for _, m in gptq_layers:
    m.qweight_int = m.qweight_int.to(torch.int32)
L32 = run_decode()
for _, m in gptq_layers:
    m.qweight_int = m.qweight_int.to(torch.int8)

bitwise = torch.equal(L8, L32)
maxdiff = (L8 - L32).abs().max().item()
_p(f"[parity] decode-logits: bitwise_equal={bitwise}  max_abs_diff={maxdiff:.3e}")

if fail == 0 and bitwise:
    _p("[parity] RESULT: PASS — int8 store is bit-identical to int32 store")
else:
    _p(f"[parity] RESULT: FAIL — layer_fails={fail} bitwise={bitwise}")
    sys.exit(1)
