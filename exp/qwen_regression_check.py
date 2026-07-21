"""Qwen1.5-MoE real-quant graph-decode regression check. Confirms the DeepSeek
gather / indexed-kernel code is NEVER reached for Qwen (gates off) and prints
generated token IDs so a new-.so vs old-.so run can be diffed byte-identically."""
import os, sys, torch
sys.path.insert(0, "/home/qyyang/repo/GLoRCQ")
REAL = sys.argv[1]
# set the DeepSeek env flags ON to prove they do NOT affect Qwen
os.environ["GLORCQ_DEEPSEEK_GATHER"] = "1"
os.environ["GLORCQ_DEEPSEEK_IDXKERNEL"] = "1"
from inference.model_builder import load_glorcq_model
from inference.moe_block import GraphCompatibleMoeBlock
from inference.graph_wrapper import GLoRCQGraphWrapper
ret = load_glorcq_model(REAL, device="cuda:0")
model = ret[0] if isinstance(ret, tuple) else ret
model.eval()
# ---- assert Qwen never enters the DeepSeek code paths ----
ngather = nidx = ndeepseek = nblocks = 0
for m in model.modules():
    if isinstance(m, GraphCompatibleMoeBlock):
        nblocks += 1
        if getattr(m, "_deepseek", False): ndeepseek += 1
        if getattr(m, "_use_gather_graph", False): ngather += 1
        if getattr(m, "_use_vq4_idx_kernel", False): nidx += 1
print(f"[qwen-check] MoE blocks={nblocks}  _deepseek={ndeepseek}  "
      f"_use_gather_graph={ngather}  _use_vq4_idx_kernel={nidx}", flush=True)
assert ndeepseek == 0 and ngather == 0 and nidx == 0, "REGRESSION: Qwen entered DeepSeek path!"
print("[qwen-check] PASS: Qwen blocks have all DeepSeek gates OFF (new code unreachable)", flush=True)
# ---- run graph decode, print tokens ----
wrapper = GLoRCQGraphWrapper(model, max_batch_size=1, max_seq_len=2048)
torch.manual_seed(0)
ids = torch.randint(0, 30000, (1, 64), device="cuda:0")
with torch.no_grad():
    out = wrapper.generate(ids, max_new_tokens=48)
    if isinstance(out, tuple):
        out = out[0]
gen = out[0, 64:].tolist()
print(f"QWEN_GEN_IDS={gen}", flush=True)
