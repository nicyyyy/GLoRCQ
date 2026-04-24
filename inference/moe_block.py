"""
Graph-compatible MoE block replacement for Qwen2MoeSparseMoeBlock.

Provides two forward paths:
  - graph_mode=False (default): Original HF sparse routing with data-dependent
    gather/scatter. Used during prefill where CUDA Graph is not active.
  - graph_mode=True: All experts execute on full input, with routing weights=0
    masking inactive experts. All operations have fixed tensor shapes, making
    this path compatible with CUDA Graph capture.

Mathematical equivalence:
  sum_{i in selected} w_i * expert_i(x) == sum_{i=0}^{N-1} full_w_i * expert_i(x)
  because full_w_i = 0 for unselected experts.

Cluster-parallel LoRA:
  Experts that share the same U matrix (same cluster) share the x@U computation.
  Only gate_proj and up_proj benefit (same input x); down_proj has different
  input per expert so it cannot be shared.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_cluster_id(expert, proj_name):
    """Get cluster_id from a GLoRCQLinear inside an expert, or None."""
    proj = getattr(expert, proj_name, None)
    if proj is not None and hasattr(proj, 'cluster_id'):
        return proj.cluster_id
    return None


def _get_U(expert, proj_name):
    """Get the shared U matrix from a GLoRCQLinear, or None."""
    proj = getattr(expert, proj_name, None)
    if proj is not None and hasattr(proj, 'U') and proj.U is not None:
        return proj.U
    return None


class GraphCompatibleMoeBlock(nn.Module):
    """Drop-in replacement for Qwen2MoeSparseMoeBlock with CUDA Graph support."""

    def __init__(self, original_block):
        super().__init__()
        # Transfer all sub-modules by reference (no weight copy)
        self.num_experts = original_block.num_experts
        self.top_k = original_block.top_k
        self.norm_topk_prob = original_block.norm_topk_prob
        self.gate = original_block.gate
        self.experts = original_block.experts
        self.shared_expert = original_block.shared_expert
        self.shared_expert_gate = original_block.shared_expert_gate
        self.graph_mode = False  # default: sparse routing

        # Cluster-parallel LoRA cache (built lazily on first forward)
        self._cluster_map_built = False
        # {proj_name: {cluster_id: (U_ref, [expert_indices])}}
        self._cluster_groups = {}

    def _build_cluster_map(self):
        """Build mapping from cluster_id to expert indices for shared U."""
        self._cluster_groups = {}
        for proj_name in ("gate_proj", "up_proj"):
            groups = {}  # cluster_id → (U_ref, [expert_idx, ...])
            for ei in range(self.num_experts):
                cid = _get_cluster_id(self.experts[ei], proj_name)
                if cid is None:
                    continue
                if cid not in groups:
                    U = _get_U(self.experts[ei], proj_name)
                    groups[cid] = (U, [])
                groups[cid][1].append(ei)
            self._cluster_groups[proj_name] = groups
        self._cluster_map_built = True

    def _precompute_xU(self, hidden_states):
        """Pre-compute x @ U_shared for each unique cluster.

        Returns:
            dict: {proj_name: {cluster_id: (B, rank) tensor}}
        """
        if not self._cluster_map_built:
            self._build_cluster_map()

        xU_cache = {}
        for proj_name, groups in self._cluster_groups.items():
            proj_cache = {}
            for cid, (U, _expert_ids) in groups.items():
                if U is not None:
                    proj_cache[cid] = hidden_states @ U  # (B, rank)
            xU_cache[proj_name] = proj_cache
        return xU_cache

    def _get_precomputed(self, xU_cache, expert_idx, proj_name):
        """Look up precomputed x@U for a specific expert and proj type."""
        cid = _get_cluster_id(self.experts[expert_idx], proj_name)
        if cid is not None and proj_name in xU_cache:
            return xU_cache[proj_name].get(cid)
        return None

    def _expert_forward(self, expert, x, xU_cache, expert_idx):
        """Forward through a single expert MLP with cluster-parallel LoRA."""
        gate_xU = self._get_precomputed(xU_cache, expert_idx, "gate_proj")
        up_xU = self._get_precomputed(xU_cache, expert_idx, "up_proj")

        gate_out = expert.gate_proj(x, precomputed_xU=gate_xU)
        up_out = expert.up_proj(x, precomputed_xU=up_xU)
        h = expert.act_fn(gate_out) * up_out
        return expert.down_proj(h)

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Router computation (fixed shape, graph-safe)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        # Pre-compute x @ U_shared for cluster-parallel LoRA
        xU_cache = self._precompute_xU(hidden_states)

        # Select forward path based on mode
        if self.graph_mode:
            final = self._forward_graph(
                hidden_states, routing_weights, selected_experts, hidden_dim,
                xU_cache)
        else:
            final = self._forward_sparse(
                hidden_states, routing_weights, selected_experts, hidden_dim,
                xU_cache)

        # Shared expert (already fixed-shape, graph-safe)
        shared_out = self.shared_expert(hidden_states)
        shared_out = (
            F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_out
        )
        final = final + shared_out

        final = final.reshape(batch_size, sequence_length, hidden_dim)
        return final, router_logits

    def _forward_sparse(self, hidden_states, routing_weights,
                        selected_experts, hidden_dim, xU_cache):
        """Original HF sparse routing. Used during prefill."""
        N = hidden_states.shape[0]
        final = torch.zeros(
            N, hidden_dim,
            dtype=hidden_states.dtype, device=hidden_states.device,
        )
        expert_mask = F.one_hot(
            selected_experts, num_classes=self.num_experts,
        ).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)

            # Gather precomputed x@U for selected tokens
            sparse_xU_cache = {}
            for proj_name in ("gate_proj", "up_proj"):
                cid = _get_cluster_id(expert_layer, proj_name)
                if cid is not None and proj_name in xU_cache and cid in xU_cache[proj_name]:
                    sparse_xU_cache[proj_name] = {cid: xU_cache[proj_name][cid][top_x]}

            current_out = self._expert_forward(
                expert_layer, current_state, sparse_xU_cache, expert_idx)
            current_out *= routing_weights[top_x, idx, None]
            final.index_add_(0, top_x, current_out.to(hidden_states.dtype))
        return final

    def _forward_graph(self, hidden_states, routing_weights,
                       selected_experts, hidden_dim, xU_cache):
        """
        Graph-safe forward: all experts execute, routing weights mask inactive ones.

        All operations are fixed-shape:
        - scatter_ writes into fixed-size tensor
        - expert_layer(hidden_states) input shape is constant
        - multiply and add are fixed-shape
        """
        N = hidden_states.shape[0]
        final = torch.zeros(
            N, hidden_dim,
            dtype=hidden_states.dtype, device=hidden_states.device,
        )

        # Build full per-expert routing weight matrix (N, num_experts)
        # selected_experts: (N, top_k), routing_weights: (N, top_k)
        full_weights = torch.zeros(
            N, self.num_experts,
            dtype=routing_weights.dtype, device=hidden_states.device,
        )
        full_weights.scatter_(1, selected_experts, routing_weights)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            expert_out = self._expert_forward(
                expert_layer, hidden_states, xU_cache, expert_idx)
            w = full_weights[:, expert_idx:expert_idx + 1]  # (N, 1)
            final = final + w * expert_out.to(final.dtype)

        return final
