"""Mixtral real-quant decode correctness: greedy-generate a fixed prompt and print
token IDs. Run under different GLORCQ_MIXTRAL_* env configs and diff the IDs — the
gather+indexed path (#1) and inline-LoRA (#2) must reproduce the standard baseline."""
import os, sys, torch
sys.path.insert(0, "/home/qyyang/repo/GLoRCQ")
REAL = sys.argv[1]
from inference.model_builder import load_glorcq_model
from inference.graph_wrapper import GLoRCQGraphWrapper
from inference.moe_block import GraphCompatibleMoeBlock
ret = load_glorcq_model(REAL, device="cuda:0")
model = ret[0] if isinstance(ret, tuple) else ret
model.eval()
inl = gg = idx = 0
for m in model.modules():
    if isinstance(m, GraphCompatibleMoeBlock):
        inl += int(getattr(m, "_inline_lora", False))
        gg  += int(getattr(m, "_use_gather_graph", False))
        idx += int(getattr(m, "_use_vq4_idx_kernel", False))
print(f"[gates] inline_lora={inl} gather={gg} idx={idx}  "
      f"env: GRAPH={os.environ.get('GLORCQ_MIXTRAL_GRAPH')} "
      f"IDX={os.environ.get('GLORCQ_MIXTRAL_IDXKERNEL')} "
      f"INLINE={os.environ.get('GLORCQ_MIXTRAL_INLINE_LORA')}", flush=True)
wrapper = GLoRCQGraphWrapper(model, max_batch_size=1, max_seq_len=384)
# REAL prompt (not random) so logits are non-flat and greedy is stable — a
# correct gather path should track the standard baseline for many tokens.
from transformers import AutoTokenizer
_tok = AutoTokenizer.from_pretrained(REAL, trust_remote_code=True, use_fast=False)
_prompt = ("The history of artificial intelligence began in antiquity, with myths "
           "and stories of artificial beings endowed with intelligence. In modern "
           "times, the field of AI research was founded at a workshop held on the "
           "campus of Dartmouth College during the summer of")
ids = _tok(_prompt, return_tensors="pt").input_ids.to("cuda:0")
with torch.no_grad():
    out = wrapper.generate(ids, max_new_tokens=48)
    if isinstance(out, tuple):
        out = out[0]
gen = out[0, ids.shape[1]:].tolist()
print(f"MIXTRAL_GEN_IDS={gen}", flush=True)
