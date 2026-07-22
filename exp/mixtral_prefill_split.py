"""Split prefill vs decode time for the Mixtral gather+idx path.
t(gen=1) ~= prefill + 1 decode step;  t(gen=129) ~= prefill + 129 steps.
decode tok/s = 128 / (t129 - t1);  prefill ~= t1 - one_step.
Run with GLORCQ_MIXTRAL_GRAPH=1 GLORCQ_MIXTRAL_IDXKERNEL=1."""
import os, sys, time, torch
sys.path.insert(0, "/home/qyyang/repo/GLoRCQ")
REAL = sys.argv[1]
from inference.model_builder import load_glorcq_model
from inference.graph_wrapper import GLoRCQGraphWrapper
model = load_glorcq_model(REAL, device="cuda:0")
ret = model
model = ret[0] if isinstance(ret, tuple) else ret
model.eval()
wrapper = GLoRCQGraphWrapper(model, max_batch_size=1, max_seq_len=384)
wrapper.capture_graph()
torch.manual_seed(0)
ids = torch.randint(0, 30000, (1, 128), device="cuda:0")

def timed(n):
    torch.cuda.synchronize(); t0 = time.time()
    out = wrapper.generate(ids, max_new_tokens=n)
    if isinstance(out, tuple): out = out[0]
    torch.cuda.synchronize()
    return time.time() - t0

# warmup (also warms prefill kernels)
_ = timed(4)
t1a, t1b = timed(1), timed(1)
t129a, t129b = timed(129), timed(129)
t1 = min(t1a, t1b); t129 = min(t129a, t129b)
dec = 128 / (t129 - t1)
print(f"PREFILL_SPLIT t(gen1)={t1:.3f}s  t(gen129)={t129:.3f}s  "
      f"decode_only={dec:.2f} tok/s  prefill~={t1:.3f}s  "
      f"naive_reported={129/t129:.2f} tok/s", flush=True)
