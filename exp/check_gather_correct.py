"""Verify the DeepSeek top-k gather graph is numerically equivalent to the
all-experts graph: greedy-decode the same prompt with GLORCQ_DEEPSEEK_GATHER on
vs off (two fresh processes would differ only by RNG-free setup) — here we do it
in ONE process by capturing two runners? No: env is read at block init. So this
script runs ONE config (env decides) and prints the generated token ids; the
driver runs it twice and diffs the id lists.
"""
import os, sys, torch
sys.path.insert(0, "/home/qyyang/repo/GLoRCQ")
REAL = sys.argv[1]
from inference import deepseek_support
from inference.model_builder import load_glorcq_model
deepseek_support.patch_cache_compat()
ret = load_glorcq_model(REAL, device="cuda:0")
model = ret[0] if isinstance(ret, tuple) else ret
model.eval()
runner = deepseek_support.install_full_decode_graph(model, max_seq_len=384, device="cuda:0")
torch.manual_seed(0)
ids = torch.randint(0, 30000, (1, 128), device="cuda:0")
with torch.no_grad():
    out = runner.generate(ids, max_new_tokens=48)
gen = out[0, 128:].tolist()
gather = os.environ.get("GLORCQ_DEEPSEEK_GATHER", "0")
print(f"GATHER={gather} GEN_IDS={gen}", flush=True)
