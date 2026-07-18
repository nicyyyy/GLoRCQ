"""Dump Qwen graph-decode logits for HEAD-vs-modified bit-parity A/B (Task #203).

Loads a Qwen GLoRCQ ckpt, prefill (graph_mode=False) then 8 N=1 decode steps
with graph_mode=True (all-experts graph path), saves the stacked decode logits
to the given output .pt. Run once on HEAD (changes git-stashed) and once with
the changes; torch.equal the two dumps.

Usage: python exp/cluster/qwen_bitparity_dump.py <qwen_dir> <out.pt>
"""
import sys
import torch

sys.path.insert(0, "/home/qyyang/repo/GLoRCQ")
from transformers import DynamicCache  # noqa: E402

DEV = "cuda:0"
QWEN = sys.argv[1]
OUT = sys.argv[2]

from inference.model_builder import load_glorcq_model  # noqa: E402
from inference.moe_block import GraphCompatibleMoeBlock  # noqa: E402

model = load_glorcq_model(QWEN, device=DEV)
model.eval()


def _set_gm(mode):
    for m in model.modules():
        if isinstance(m, GraphCompatibleMoeBlock):
            m.graph_mode = mode


torch.manual_seed(1)
vocab = min(getattr(model.config, "vocab_size", 30000), 30000)
prompt = torch.randint(0, vocab, (1, 32), device=DEV)

_set_gm(False)
logits = []
with torch.no_grad():
    past = DynamicCache()
    out = model(input_ids=prompt, past_key_values=past, use_cache=True,
                return_dict=True)
    nxt = out.logits[:, -1:].argmax(-1)
    _set_gm(True)
    for _ in range(8):
        out = model(input_ids=nxt, past_key_values=past, use_cache=True,
                    return_dict=True)
        logits.append(out.logits[:, -1, :].clone())
        nxt = out.logits[:, -1:].argmax(-1)

L = torch.cat(logits).cpu()
torch.save(L, OUT)
print(f"[dump] saved {OUT} shape={tuple(L.shape)} "
      f"sum={L.double().sum().item():.6f} ids={[int(a) for a in L.argmax(-1)]}",
      flush=True)
