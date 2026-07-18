"""
Task #203 block-level unit parity: gather graph path vs CANONICAL per-expert
reference, isolated from KV/attention/long-gen NaN accumulation.

For a chosen MoE block and random hidden states (1,1,H), run block.forward in 3
modes and compare outputs:
  * sparse  : graph_mode=False, _graph_threshold=0 -> _forward_sparse
              (canonical: each expert via GLoRCQLinear.forward / Fp16LinearShim,
               the reference math with the fp32-Sa fix). GROUND TRUTH.
  * decode  : graph_mode=False, _graph_threshold=4 -> _forward_decode (batched)
  * gather  : graph_mode=True                       -> _forward_graph_vq4_gather

Reports max|gather-sparse| and max|decode-sparse|. If gather ~ sparse and decode
diverges/NaNs, gather is correct and the batched decode has a pre-existing bug.

Runs on: a shim-FREE layer and a shim-CONTAINING layer (auto-detected).

Usage: python exp/cluster/gather_graph_unit_probe.py <mixtral_dir>
"""
import sys
import torch

sys.path.insert(0, "/home/qyyang/repo/GLoRCQ")

DEV = "cuda:0"
MODEL = sys.argv[1]


def _p(*a):
    print(*a, flush=True)


from inference.model_builder import load_glorcq_model  # noqa: E402
from inference.moe_block import GraphCompatibleMoeBlock  # noqa: E402

_p(f"[load] {MODEL}")
model = load_glorcq_model(MODEL, device=DEV)
model.eval()
H = model.config.hidden_size

# Collect MoE blocks with layer index and shim count.
blocks = []
m = getattr(model, "model", model)
for li, layer in enumerate(m.layers):
    for attr in ("block_sparse_moe", "mlp"):
        mod = getattr(layer, attr, None)
        if isinstance(mod, GraphCompatibleMoeBlock):
            n_shim = 0
            for e in mod.experts:
                for pn in ("gate_proj", "up_proj", "down_proj"):
                    p = getattr(e, pn, None)
                    if p is not None and getattr(p, "vq_codes", None) is None \
                            and getattr(p, "weight", None) is not None:
                        n_shim += 1
            blocks.append((li, mod, n_shim))

shimfree = next((b for b in blocks if b[2] == 0), None)
withshim = next((b for b in blocks if b[2] > 0), None)
_p(f"[info] {len(blocks)} MoE blocks; "
   f"shim-free layer={shimfree[0] if shimfree else None}; "
   f"with-shim layer={withshim[0] if withshim else None} "
   f"(shim projs={withshim[2] if withshim else 0})")


@torch.no_grad()
def run_mode(block, x, mode):
    saved_gm = block.graph_mode
    saved_th = block._graph_threshold
    try:
        if mode == "sparse":
            block.graph_mode = False
            block._graph_threshold = 0
        elif mode == "decode":
            block.graph_mode = False
            block._graph_threshold = 4
        elif mode == "gather":
            block.graph_mode = True
        out, _ = block(x)
        return out.float().reshape(-1)
    finally:
        block.graph_mode = saved_gm
        block._graph_threshold = saved_th


@torch.no_grad()
def compare(name, li, block):
    torch.manual_seed(0)
    diffs_gs, diffs_ds, diffs_gd, diffs_det = [], [], [], []
    nan_gather = nan_decode = nan_sparse = 0
    for t in range(16):
        x = torch.randn(1, 1, H, dtype=torch.float16, device=DEV) * 2.0
        y_sparse = run_mode(block, x, "sparse")   # fp32 canonical (down=fp32 python)
        y_decode = run_mode(block, x, "decode")   # fp16 batched decode (standard path)
        y_gather = run_mode(block, x, "gather")   # fp16 gather graph
        y_gather2 = run_mode(block, x, "gather")  # determinism floor
        nan_sparse += int(torch.isnan(y_sparse).any())
        nan_decode += int(torch.isnan(y_decode).any())
        nan_gather += int(torch.isnan(y_gather).any())
        denom = y_sparse.abs().max().clamp_min(1e-6)
        diffs_gs.append(((y_gather - y_sparse).abs().max() / denom).item())
        diffs_ds.append(((y_decode - y_sparse).abs().max() / denom).item())
        diffs_gd.append(((y_gather - y_decode).abs().max() / denom).item())
        diffs_det.append(((y_gather - y_gather2).abs().max() / denom).item())
    _p(f"\n[{name}] layer={li}")
    _p(f"   NaN counts / 16 trials: sparse={nan_sparse} decode={nan_decode} "
       f"gather={nan_gather}")
    _p(f"   gather-vs-fp32sparse  max={max(diffs_gs):.3e} mean={sum(diffs_gs)/len(diffs_gs):.3e}")
    _p(f"   decode-vs-fp32sparse  max={max(diffs_ds):.3e} mean={sum(diffs_ds)/len(diffs_ds):.3e}")
    _p(f"   gather-vs-decode(fp16)max={max(diffs_gd):.3e} mean={sum(diffs_gd)/len(diffs_gd):.3e}")
    _p(f"   gather determinism    max={max(diffs_det):.3e} (gather vs gather rerun)")
    return max(diffs_gd), nan_gather


results = {}
if shimfree:
    results["shim-free"] = compare("shim-free", shimfree[0], shimfree[1])
if withshim:
    results["with-shim"] = compare("with-shim", withshim[0], withshim[1])

_p("\n==== SUMMARY (gather vs STANDARD fp16 decode-path) ====")
ok = True
for k, (mx, nanc) in results.items():
    good = (mx < 2e-2) and (nanc == 0)   # gather vs decode: both fp16, ~noise floor
    ok = ok and good
    _p(f"  {k}: gather-vs-decode max_rel_diff={mx:.3e} gather_nan={nanc} "
       f"-> {'OK' if good else 'CHECK'}")
_p(f"OVERALL gather matches standard decode-path = {ok}")
