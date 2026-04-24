"""
Unit test: verify cluster-parallel LoRA gives identical results to per-expert LoRA.

Creates a small fake MoE block with shared U matrices, runs both paths,
and checks numerical equivalence. No model loading needed — runs in ~1s on GPU.

Usage:
    cd glorcq && python -m inference.test_cluster_parallel
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.moe_block import GraphCompatibleMoeBlock

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HIDDEN = 256
INTER = 512
RANK = 16
NUM_EXPERTS = 8
TOP_K = 2
NUM_CLUSTERS = 3  # fewer clusters than experts -> some share U
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32  # use float32 to avoid fp16 overflow in random weights


# ---------------------------------------------------------------------------
# Fake expert MLP using plain nn.Linear + manual LoRA
# ---------------------------------------------------------------------------
class FakeGLoRCQLinear(nn.Module):
    """Mimics GLoRCQLinear interface with plain linear + LoRA."""
    def __init__(self, in_f, out_f):
        super().__init__()
        self.in_features = in_f
        self.out_features = out_f
        self.linear = nn.Linear(in_f, out_f, bias=False)
        self.U = None
        self.S = None
        self.V = None
        self.cluster_id = None

    def load_lora(self, U, S, V, device="cuda"):
        self.U = U.to(device=device, dtype=DTYPE)
        self.S = S.to(device=device, dtype=DTYPE)
        self.V = V.to(device=device, dtype=DTYPE)

    def forward(self, x, precomputed_xU=None):
        y = self.linear(x)
        if self.U is not None:
            if precomputed_xU is not None:
                a = precomputed_xU
            else:
                a = x @ self.U
            b = a * self.S
            y = y + b @ self.V.T
        return y


class FakeExpertMLP(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.gate_proj = FakeGLoRCQLinear(hidden, inter)
        self.up_proj = FakeGLoRCQLinear(hidden, inter)
        self.down_proj = FakeGLoRCQLinear(inter, hidden)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class FakeOriginalBlock:
    """Mimics the original MoE block structure for GraphCompatibleMoeBlock."""
    def __init__(self):
        self.num_experts = NUM_EXPERTS
        self.top_k = TOP_K
        self.norm_topk_prob = True
        self.gate = nn.Linear(HIDDEN, NUM_EXPERTS, bias=False)
        self.experts = nn.ModuleList([
            FakeExpertMLP(HIDDEN, INTER) for _ in range(NUM_EXPERTS)
        ])
        self.shared_expert = nn.Sequential(nn.Linear(HIDDEN, HIDDEN, bias=False))
        self.shared_expert_gate = nn.Linear(HIDDEN, 1, bias=False)


def run_test():
    torch.manual_seed(42)
    print(f"Device: {DEVICE}, dtype: {DTYPE}")
    print(f"Config: {NUM_EXPERTS} experts, {NUM_CLUSTERS} clusters, "
          f"hidden={HIDDEN}, rank={RANK}")

    # Create shared U and S per cluster per proj type
    shared_Us = {}
    shared_Ss = {}
    for proj_name in ("gate_proj", "up_proj", "down_proj"):
        in_d = HIDDEN if proj_name in ("gate_proj", "up_proj") else INTER
        for cid in range(NUM_CLUSTERS):
            shared_Us[(proj_name, cid)] = torch.randn(in_d, RANK, device=DEVICE, dtype=DTYPE)
            shared_Ss[(proj_name, cid)] = torch.randn(RANK, device=DEVICE, dtype=DTYPE).abs()

    # Build MoE block
    orig_block = FakeOriginalBlock()
    for ei in range(NUM_EXPERTS):
        expert = orig_block.experts[ei]
        cid = ei % NUM_CLUSTERS
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(expert, proj_name)
            U = shared_Us[(proj_name, cid)]
            S = shared_Ss[(proj_name, cid)]
            V = torch.randn(proj.out_features, RANK, device=DEVICE, dtype=DTYPE)
            proj.load_lora(U, S, V, device=DEVICE)
            proj.cluster_id = (proj_name, cid)

    moe_block = GraphCompatibleMoeBlock(orig_block).to(DEVICE).to(DTYPE)

    # Test input
    B, T = 2, 4
    x = torch.randn(B, T, HIDDEN, device=DEVICE, dtype=DTYPE)

    # --- Path 1: baseline (no cluster-parallel, cluster_id=None) ---
    saved_cids = {}
    for ei in range(NUM_EXPERTS):
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(moe_block.experts[ei], proj_name)
            saved_cids[(ei, proj_name)] = proj.cluster_id
            proj.cluster_id = None
    moe_block._cluster_map_built = False

    moe_block.graph_mode = False
    y_baseline, _ = moe_block(x)

    moe_block.graph_mode = True
    y_graph_baseline, _ = moe_block(x)

    # --- Path 2: cluster-parallel ---
    for ei in range(NUM_EXPERTS):
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(moe_block.experts[ei], proj_name)
            proj.cluster_id = saved_cids[(ei, proj_name)]
    moe_block._cluster_map_built = False

    moe_block.graph_mode = False
    y_cluster, _ = moe_block(x)

    moe_block.graph_mode = True
    y_graph_cluster, _ = moe_block(x)

    # --- Compare sparse mode ---
    max_diff = (y_baseline - y_cluster).abs().max().item()
    rel_diff = max_diff / (y_baseline.abs().max().item() + 1e-10)
    print(f"\n=== Sparse mode (prefill) ===")
    print(f"  Max abs diff:  {max_diff:.2e}")
    print(f"  Rel diff:      {rel_diff:.2e}")
    assert rel_diff < 1e-5, f"FAIL: rel diff {rel_diff}"
    print("  PASS")

    # --- Compare graph mode ---
    max_diff_g = (y_graph_baseline - y_graph_cluster).abs().max().item()
    rel_diff_g = max_diff_g / (y_graph_baseline.abs().max().item() + 1e-10)
    print(f"\n=== Graph mode (decode) ===")
    print(f"  Max abs diff:  {max_diff_g:.2e}")
    print(f"  Rel diff:      {rel_diff_g:.2e}")
    assert max_diff_g < 1e-2, f"FAIL: max diff {max_diff_g}"
    print("  PASS")

    # --- Sharing stats ---
    gate_clusters = set()
    for ei in range(NUM_EXPERTS):
        gate_clusters.add(saved_cids[(ei, "gate_proj")])
    print(f"\n=== Cluster sharing stats ===")
    print(f"  {NUM_EXPERTS} experts, {len(gate_clusters)} unique gate_proj clusters")
    print(f"  x@U computations saved: {NUM_EXPERTS - len(gate_clusters)} per proj type")
    print(f"\nAll tests passed!")


if __name__ == "__main__":
    run_test()
