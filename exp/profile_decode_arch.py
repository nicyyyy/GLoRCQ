"""Per-component bs=1 decode-time breakdown for DeepSeek-MoE-16B.

Tests the hypothesis: at bs=1 decode (memory-bandwidth-bound), is the dominant
per-token cost the fp16 attention + fp16 shared experts, or the (already
quantized) routed experts?

Works on BOTH:
  * real-quant  (model_builder.load_glorcq_model): attn fp16, shared fp16,
    routed experts = VQ4+LoRA via GraphCompatibleMoeBlock.
  * fp16        (AutoModelForCausalLM): everything fp16, stock DeepseekMoE.

Buckets (CUDA-event timed, no mid-loop sync):
  attn    : all *Attention forwards
  shared  : all shared_experts (+ layer-0 dense DeepseekMLP) forwards
  routed  : routed experts. real-quant = _batched_proj_forward + _batched_down_forward;
            fp16 = DeepseekMoE.moe_infer
  lm_head : final projection to vocab
  other   : total wall - sum(above)   (embed, router gate, rmsnorm, rope, sampling)

Usage:
  python exp/profile_decode_arch.py <model_dir> [n_decode] [--fp16]
"""
import os, sys, time, argparse, torch

sys.path.insert(0, "/home/qyyang/repo/GLoRCQ")

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("n_decode", nargs="?", type=int, default=64)
ap.add_argument("--fp16", action="store_true", help="load as fp16 AutoModel (baseline)")
ap.add_argument("--prompt_len", type=int, default=128)
args = ap.parse_args()

MODEL = args.model
N_DECODE = args.n_decode
DEV = "cuda:0"


def _p(*a):
    print(*a, flush=True)


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


buckets = {"attn": Bucket(), "shared": Bucket(), "routed": Bucket(),
           "lm_head": Bucket()}

from inference import deepseek_support
deepseek_support.patch_cache_compat()

_p(f"[profile] loading {MODEL}  (fp16={args.fp16})")
t0 = time.time()
if args.fp16:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.float16).to(DEV)
    model.eval()
else:
    from inference.model_builder import load_glorcq_model
    ret = load_glorcq_model(MODEL, device=DEV)
    model = ret[0] if isinstance(ret, tuple) else ret
    model.eval()
torch.cuda.synchronize()
_p(f"[profile] load done in {time.time()-t0:.0f}s  "
   f"alloc={torch.cuda.memory_allocated(DEV)/1024**3:.2f}GB "
   f"reserved={torch.cuda.memory_reserved(DEV)/1024**3:.2f}GB")

# ---- wrap routed-expert compute ------------------------------------------
if args.fp16:
    # stock DeepseekMoE.moe_infer = routed experts (grouped). Wrap the class method.
    n_moe = 0
    for m in model.modules():
        if type(m).__name__ == "DeepseekMoE" and hasattr(m, "moe_infer"):
            m.moe_infer = buckets["routed"].wrap(m.moe_infer)
            n_moe += 1
    _p(f"[profile] wrapped {n_moe} DeepseekMoE.moe_infer (routed)")
else:
    import inference.moe_block as mb
    mb.GraphCompatibleMoeBlock._batched_proj_forward = buckets["routed"].wrap(
        mb.GraphCompatibleMoeBlock._batched_proj_forward)
    mb.GraphCompatibleMoeBlock._batched_down_forward = buckets["routed"].wrap(
        mb.GraphCompatibleMoeBlock._batched_down_forward)
    _p("[profile] wrapped GraphCompatibleMoeBlock routed proj/down")

# ---- wrap shared experts + layer-0 dense MLP -----------------------------
n_shared = 0
for m in model.modules():
    se = getattr(m, "shared_experts", None)
    if se is not None:
        se.forward = buckets["shared"].wrap(se.forward)
        n_shared += 1
# layer-0 dense MLP (DeepseekMLP that is NOT a shared_experts child): the
# decoder layer's `.mlp` when it's a bare DeepseekMLP (dense first layer).
n_dense = 0
base = getattr(model, "model", model)
for layer in base.layers:
    mlp = getattr(layer, "mlp", None)
    if mlp is not None and type(mlp).__name__ == "DeepseekMLP":
        mlp.forward = buckets["shared"].wrap(mlp.forward)
        n_dense += 1
_p(f"[profile] wrapped {n_shared} shared_experts + {n_dense} dense-MLP (layer0) -> shared bucket")

# ---- wrap attention ------------------------------------------------------
n_attn = 0
for m in model.modules():
    if m.__class__.__name__.endswith("Attention"):
        m.forward = buckets["attn"].wrap(m.forward)
        n_attn += 1
_p(f"[profile] wrapped {n_attn} attention modules")

# ---- wrap lm_head --------------------------------------------------------
model.lm_head.forward = buckets["lm_head"].wrap(model.lm_head.forward)

# ---- manual prefill + decode loop ----------------------------------------
from transformers import DynamicCache
torch.manual_seed(0)
vocab = getattr(model.config, "vocab_size", 32000)
prompt = torch.randint(0, min(vocab, 30000), (1, args.prompt_len), device=DEV)

with torch.no_grad():
    past = DynamicCache()
    out = model(input_ids=prompt, past_key_values=past, use_cache=True, return_dict=True)
    nxt = out.logits[:, -1:].argmax(-1)
    for _ in range(4):  # warmup decode
        out = model(input_ids=nxt, past_key_values=past, use_cache=True, return_dict=True)
        nxt = out.logits[:, -1:].argmax(-1)

    for b in buckets.values():
        b.reset()

    torch.cuda.synchronize(); td0 = time.time()
    for _ in range(N_DECODE):
        out = model(input_ids=nxt, past_key_values=past, use_cache=True, return_dict=True)
        nxt = out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize(); td1 = time.time()

decode_wall = td1 - td0
tok_s = N_DECODE / decode_wall
tot = decode_wall * 1000

_p("\n" + "=" * 66)
_p(f"  PROFILE: {os.path.basename(MODEL)}  (fp16={args.fp16})  {N_DECODE} decode steps")
_p("=" * 66)
_p(f"  decode wall: {decode_wall:.3f}s   throughput: {tok_s:.2f} tok/s   "
   f"per-token: {tot/N_DECODE:.2f} ms")

a_ms = buckets["attn"].ms()
s_ms = buckets["shared"].ms()
r_ms = buckets["routed"].ms()
l_ms = buckets["lm_head"].ms()
region = a_ms + s_ms + r_ms + l_ms
other = max(0.0, tot - region)

_p("\n  --- COMPONENT GPU TIME (% of decode wall) ---")
for name, ms, n in [
        ("attn      (fp16 q/k/v/o + sdpa + rope)", a_ms, buckets["attn"].n),
        ("shared    (fp16 shared MLP + layer0 dense)", s_ms, buckets["shared"].n),
        ("routed    (top-6 experts: VQ4+LoRA / fp16)", r_ms, buckets["routed"].n),
        ("lm_head   (fp16 -> vocab)", l_ms, buckets["lm_head"].n),
        ("other     (embed/router/norm/sample)", other, 0)]:
    _p(f"    {name:44s} {ms:9.1f} ms  {ms/tot*100:5.1f}%  calls={n}")

_p("\n  --- PER-TOKEN ---")
_p(f"    attn={a_ms/N_DECODE:.3f}  shared={s_ms/N_DECODE:.3f}  "
   f"routed={r_ms/N_DECODE:.3f}  lm_head={l_ms/N_DECODE:.3f}  "
   f"other={other/N_DECODE:.3f} ms")
fp16_dom = (a_ms + s_ms) / tot * 100
_p(f"\n  fp16 attn+shared = {fp16_dom:.1f}% of decode   "
   f"routed = {r_ms/tot*100:.1f}%")
_p(f"  resident: alloc={torch.cuda.memory_allocated(DEV)/1024**3:.2f}GB "
   f"reserved={torch.cuda.memory_reserved(DEV)/1024**3:.2f}GB")
_p("=" * 66)
