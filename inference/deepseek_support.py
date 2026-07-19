"""DeepSeek-V2 (deepseek_v2) inference-support helpers.

Keeps the shared inference files (moe_block.py, model_builder.py) lean: the
DeepSeek-specific behavior lives here and is called only behind an
`if self._deepseek:` / `if is_deepseek_moe_block(mod):` guard. Runtime behavior
for Qwen/Mixtral/Qwen3 is therefore unaffected — those paths never enter here.

DeepSeek-V2-Lite specifics handled:
  * Router is a custom `MoEGate` (returns (idx, weight, aux), not logits) whose
    config (norm_topk_prob, routed_scaling_factor) differs from Qwen. We compute
    routing directly from `gate.weight` so it works on flattened (N, h) inputs.
  * norm_topk_prob is False on V2-Lite → scale top-k weights by
    routed_scaling_factor instead of renormalizing (matches modeling_deepseek).
  * Plural, ungated `shared_experts` (a single wider MLP), added directly.
  * DeepseekV2DecoderLayer expects `mlp(x)` to return a bare tensor, unlike the
    Qwen/Mixtral decoders which unpack `(hidden_states, router_logits)`.
  * The remote modeling (written for tf ~4.36) calls DynamicCache.get_max_length()
    which was removed in tf 4.51 (renamed get_max_cache_shape). We shim it.
"""
import torch
import torch.nn.functional as F


def is_deepseek_block_obj(original_block) -> bool:
    """True if this HF MoE block is a DeepSeek-V2 routed-expert block."""
    return type(original_block).__name__.startswith("Deepseek")


def is_deepseek_moe_block(mod) -> bool:
    """True only for the routed DeepseekV2MoE block (NOT the dense layer-0
    DeepseekV2MLP, which has no `.experts`/`.gate`)."""
    return (
        type(mod).__name__ == "DeepseekV2MoE"
        and hasattr(mod, "experts")
        and hasattr(mod, "gate")
    )


def routing_config(original_block) -> dict:
    """Pull routing config off the DeepSeek MoEGate router."""
    gate = original_block.gate
    return {
        "norm_topk_prob": getattr(gate, "norm_topk_prob", False),
        "routed_scaling_factor": getattr(gate, "routed_scaling_factor", 1.0),
    }


def compute_routing(block, hidden_states):
    """DeepSeek MoEGate routing on flattened (N, h) hidden states.

    softmax over ALL experts (fp32) → greedy top-k. If norm_topk_prob: renormalize
    to sum 1; else scale by routed_scaling_factor (V2-Lite path). Computed from
    gate.weight directly (the MoEGate.forward expects 3-D input, so we can't call
    it on the flattened tensor).

    Returns (router_logits, routing_weights[dtype], selected_experts).
    """
    router_logits = F.linear(hidden_states.float(), block.gate.weight.float(), None)
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(
        routing_weights, block.top_k, dim=-1)
    if block.norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    else:
        routing_weights = routing_weights * block._routed_scaling_factor
    return router_logits, routing_weights.to(hidden_states.dtype), selected_experts


def add_shared_experts(block, final, hidden_states):
    """Add the ungated plural `shared_experts` MLP output (DeepSeek-V2)."""
    if getattr(block, "shared_experts", None) is not None:
        final = final + block.shared_experts(hidden_states)
    return final


def patch_cache_compat():
    """Add DynamicCache.get_max_length (removed in tf 4.51) so DeepSeek-V2's
    remote modeling generate() works. Idempotent; no-op if already present."""
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None


# ---------------------------------------------------------------------------
# bs=1 decode acceleration for DeepSeek-V2-Lite (per-MoE-block CUDA graphs)
# ---------------------------------------------------------------------------
# The shared graph_wrapper captures the WHOLE model as one CUDA graph, but that
# needs a StaticCache + cache_position. DeepSeek-V2's MLA attention decompresses
# and caches asymmetric multi-head K/V (key head_dim=192, value head_dim=128),
# which transformers' StaticCache (single head_dim) cannot hold, and its remote
# modeling uses the legacy DynamicCache API — so full-model capture is not
# available without touching shared code.
#
# Instead we capture ONLY the per-layer MoE block compute (the launch-bound part:
# 64 routed experts + shared expert, dispatched every decode token) into a
# per-block CUDA graph, and leave MLA attention running eager (it is fp16 and
# uses its own DynamicCache). This eliminates the MoE kernel-launch overhead
# while keeping attention correct. Fully DeepSeek-isolated: nothing here runs
# for Qwen/Mixtral, and the shared graph_wrapper is untouched.


class _MoEBlockGraph:
    """Wraps one GraphCompatibleMoeBlock: replays a captured fixed-shape (bs=1,
    seq=1) all-experts graph on decode; falls back to eager for prefill /
    other shapes."""

    def __init__(self, block, hidden_size, device, dtype):
        import torch
        self.block = block
        self.orig_forward = block.forward   # bound method (arch-aware, bare-tensor return)
        self.static_in = torch.zeros((1, 1, hidden_size), device=device, dtype=dtype)
        self.graph = None
        self.static_out = None
        block.graph_mode = True             # fixed-shape all-experts path
        # Warm up on a side stream: builds the lazy vq4 graph cache + cuBLAS/kernel
        # plans so nothing allocates during capture.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                out = self.orig_forward(self.static_in)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_out = self.orig_forward(self.static_in)

    def __call__(self, hidden_states, *args, **kwargs):
        # Decode step: (1, 1, hidden) → replay. Anything else (prefill) → eager.
        if (hidden_states.dim() == 3 and hidden_states.shape[0] == 1
                and hidden_states.shape[1] == 1 and not args and not kwargs):
            self.static_in.copy_(hidden_states)
            self.graph.replay()
            return self.static_out
        self.block.graph_mode = False
        out = self.orig_forward(hidden_states, *args, **kwargs)
        self.block.graph_mode = True
        return out


def install_moe_block_graphs(model, device="cuda:0"):
    """Capture a per-block CUDA graph for every DeepSeek MoE block so bs=1 decode
    replays the expert dispatch instead of launching it. Returns the list of
    runners (keep the reference alive; the graphs own their captured buffers).

    No-op-safe: only wraps GraphCompatibleMoeBlock instances (the routed MoE
    blocks). Dense layer-0 and attention are left untouched.
    """
    import torch
    from .moe_block import GraphCompatibleMoeBlock
    hidden = model.config.hidden_size
    dtype = next(model.parameters()).dtype
    runners = []
    n = 0
    for module in model.modules():
        if isinstance(module, GraphCompatibleMoeBlock):
            runner = _MoEBlockGraph(module, hidden, device, dtype)
            module.forward = runner          # nn.Module.__call__ dispatches to instance attr
            runners.append(runner)
            n += 1
    print(f"[deepseek] captured {n} per-MoE-block CUDA graphs (bs=1 decode)", flush=True)
    return runners
