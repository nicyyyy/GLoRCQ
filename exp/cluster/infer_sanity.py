import sys, torch
sys.path.insert(0, "/home/qyyang/repo/GLoRCQ")
from utils.model_loader import load_model_and_tokenizer

mp = sys.argv[1]
print(f"[load] {mp}", flush=True)
model, tok = load_model_and_tokenizer(mp, device="cuda:0", real_quant=True)
model.eval()
prompt = "The capital of France is"
ids = tok(prompt, return_tensors="pt").input_ids.to("cuda:0")
with torch.no_grad():
    # 1) forward logits NaN check
    logits = model(ids).logits
    nan = torch.isnan(logits).any().item()
    inf = torch.isinf(logits).any().item()
    print(f"[logits] shape={tuple(logits.shape)} NaN={nan} Inf={inf} max={logits.float().abs().max().item():.2f}", flush=True)
    # 2) greedy generate 40 tokens
    out = model.generate(ids, max_new_tokens=40, do_sample=False, use_cache=True)
txt = tok.decode(out[0], skip_special_tokens=True)
print(f"[gen] {txt!r}", flush=True)
ok = (not nan) and (not inf) and len(txt) > len(prompt)
print(f"[RESULT] {'PASS' if ok else 'FAIL'}", flush=True)
