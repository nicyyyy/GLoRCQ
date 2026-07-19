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
