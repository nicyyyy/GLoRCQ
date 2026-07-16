import sys, torch
sys.path.insert(0, "/home/qyyang/repo/GLoRCQ")
from utils.model_loader import load_model_and_tokenizer
mp = sys.argv[1]
print(f"[load] {mp}", flush=True)
model, tok = load_model_and_tokenizer(mp, device="cuda:0", real_quant=True)
model.eval()

fired = []   # (order, name, has_nan, has_inf) in execution order
def mk(name):
    def hook(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        if isinstance(t, torch.Tensor) and t.is_floating_point():
            fired.append((name, bool(torch.isnan(t).any()), bool(torch.isinf(t).any())))
    return hook
hooks = [m.register_forward_hook(mk(n)) for n, m in model.named_modules() if n]

ids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda:0")
with torch.no_grad():
    logits = model(ids).logits
print(f"[final logits] NaN={torch.isnan(logits).any().item()} Inf={torch.isinf(logits).any().item()}", flush=True)

# first module (in execution order) that produced NaN or Inf
first = next(((n,nan,inf) for (n,nan,inf) in fired if nan or inf), None)
if first:
    print(f"[FIRST NaN/Inf] {first[0]}  NaN={first[1]} Inf={first[2]}", flush=True)
    # print a few modules around it for context
    idx = [i for i,(n,_,_) in enumerate(fired) if n==first[0]][0]
    print("[context around first bad module]", flush=True)
    for n,nan,inf in fired[max(0,idx-3):idx+2]:
        print(f"    {n:70s} NaN={nan} Inf={inf}", flush=True)
else:
    print("[FIRST NaN/Inf] none — all module outputs finite", flush=True)
print(f"[total modules fired] {len(fired)}", flush=True)
