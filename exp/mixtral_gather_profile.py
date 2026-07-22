"""Profile the Mixtral gather+indexed-kernel DECODE path (the 12.7 tok/s path) to
find the next bottleneck. Sets the env gates, captures the graph, then runs N
decode steps under torch.profiler and prints the top CUDA ops by total time."""
import os, sys, torch
os.environ.setdefault("GLORCQ_MIXTRAL_GRAPH", "1")
os.environ.setdefault("GLORCQ_MIXTRAL_IDXKERNEL", "1")
os.environ.setdefault("GLORCQ_MIXTRAL_INLINE_LORA", "1")
sys.path.insert(0, "/home/qyyang/repo/GLoRCQ")
REAL = sys.argv[1]
NSTEP = int(sys.argv[2]) if len(sys.argv) > 2 else 64
from inference.model_builder import load_glorcq_model
from inference.graph_wrapper import GLoRCQGraphWrapper
ret = load_glorcq_model(REAL, device="cuda:0")
model = ret[0] if isinstance(ret, tuple) else ret
model.eval()
wrapper = GLoRCQGraphWrapper(model, max_batch_size=1, max_seq_len=384)
torch.manual_seed(0)
ids = torch.randint(0, 30000, (1, 128), device="cuda:0")
with torch.no_grad():
    # warmup + graph capture
    _ = wrapper.generate(ids, max_new_tokens=8)
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                 record_shapes=False) as prof:
        _ = wrapper.generate(ids, max_new_tokens=NSTEP)
        torch.cuda.synchronize()
print(f"\n===== TOP CUDA ops over {NSTEP} decode steps =====", flush=True)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
