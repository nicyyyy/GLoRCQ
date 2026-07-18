"""Task #202 parity probe: OLD (from_pretrained) vs NEW (meta-init fast) loader.

Same checkpoint, same kernels, same inputs -> logits MUST be bit-identical.
Any nonzero diff means the fast loader mis-loaded a surviving weight/buffer
(router gate, norm, fp16 attention, tie, rope) — a correctness bug, not noise.

Usage (run twice in tmux test:0, serial):
  GLORCQ_LEGACY_LOAD=1 python exp/cluster/loader_parity_probe.py <ckpt> old  <out_old.pt>
  GLORCQ_LEGACY_LOAD=0 python exp/cluster/loader_parity_probe.py <ckpt> new  <out_new.pt>
Then:
  python exp/cluster/loader_parity_probe.py --compare <out_old.pt> <out_new.pt>
"""
import os
import sys
import time
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def run(ckpt, tag, out_path, device="cuda:0"):
    from inference.model_builder import load_glorcq_model
    from transformers import AutoTokenizer

    t0 = time.time()
    model = load_glorcq_model(ckpt, device=device)
    load_s = time.time() - t0
    print(f"[{tag}] load wall = {load_s:.1f}s", flush=True)

    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True, use_fast=False)
    prompt = ("The capital of France is Paris. Quantization of mixture-of-experts "
              "language models trades a small amount of accuracy for a large "
              "reduction in memory footprint and")
    ids = tok(prompt, return_tensors="pt").input_ids[:, :32].to(device)

    all_logits = []
    gen_ids = []
    with torch.no_grad():
        out = model(ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :].float().cpu()
        all_logits.append(logits)
        nxt = logits.argmax(-1, keepdim=True).to(device)
        gen_ids.append(int(nxt))
        for _ in range(8):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].float().cpu()
            all_logits.append(logits)
            nxt = logits.argmax(-1, keepdim=True).to(device)
            gen_ids.append(int(nxt))

    torch.save({"tag": tag, "load_s": load_s,
                "logits": torch.stack(all_logits),  # (9, 1, vocab)
                "gen_ids": gen_ids,
                "text": tok.decode(gen_ids)}, out_path)
    print(f"[{tag}] gen_ids = {gen_ids}")
    print(f"[{tag}] text    = {tok.decode(gen_ids)!r}")
    print(f"[{tag}] saved -> {out_path}", flush=True)


def compare(p_old, p_new):
    a = torch.load(p_old, map_location="cpu")
    b = torch.load(p_new, map_location="cpu")
    print(f"OLD load {a['load_s']:.1f}s | NEW load {b['load_s']:.1f}s "
          f"| speedup {a['load_s']/max(b['load_s'],1e-6):.1f}x")
    la, lb = a["logits"], b["logits"]
    max_abs = (la - lb).abs().max().item()
    ids_match = a["gen_ids"] == b["gen_ids"]
    print(f"gen_ids OLD = {a['gen_ids']}")
    print(f"gen_ids NEW = {b['gen_ids']}")
    print(f"gen_ids identical : {ids_match}")
    print(f"logits max_abs_diff: {max_abs:.3e}")
    print(f"bitwise equal     : {torch.equal(la, lb)}")
    if ids_match and max_abs == 0.0:
        print("VERDICT: PASS (bit-identical)")
    elif ids_match and max_abs < 1e-2:
        print(f"VERDICT: PASS-ish (ids match, tiny fp diff {max_abs:.2e} — investigate source)")
    else:
        print("VERDICT: FAIL (loaders diverge — fast path has a correctness bug)")


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1], sys.argv[2], sys.argv[3])
