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

        # Auto graph-mode threshold: when N (num tokens) <= this, use
        # _forward_graph to avoid expensive nonzero/torch.where sync.
        # Covers decode (N=1) and short sequences. Set to 0 to disable.
        self._graph_threshold = 4

        # Cluster-parallel LoRA cache (built lazily on first forward)
        self._cluster_map_built = False
        # {proj_name: {cluster_id: (U_ref, [expert_indices])}}
        self._cluster_groups = {}

        # Batched graph cache (built lazily on first graph-mode forward)
        self._graph_cache_built = False

        # Persistent side stream for parallel LoRA execution
        self._side_stream = None

    def _get_side_stream(self):
        """Get or create persistent side stream for parallel LoRA."""
        if self._side_stream is None:
            self._side_stream = torch.cuda.Stream()
        return self._side_stream

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

    def _build_graph_cache(self):
        """Pre-concatenate all experts' weights for batched graph-mode forward.

        Builds cat'd packed_indices/norms/SV for gate and up (same input x →
        1 turbo call per proj instead of 60).  Down proj still needs per-expert
        calls (different inputs), but batches rotation and LoRA.
        """
        E = self.num_experts

        for proj_attr, packed_attr, norms_attr, centroids_attr, rot_key_attr, out_d_attr in [
            ("gate_proj", "_packed_gate_all", "_norms_gate_all",
             "_gate_centroids", "_gate_x_rot_key", "_gate_out_d"),
            ("up_proj",   "_packed_up_all",   "_norms_up_all",
             "_up_centroids",   "_up_x_rot_key",   "_up_out_d"),
        ]:
            projs = [getattr(self.experts[ei], proj_attr) for ei in range(E)]
            setattr(self, packed_attr,
                    torch.cat([p.packed_indices for p in projs], dim=0))
            setattr(self, norms_attr,
                    torch.cat([p.norms for p in projs], dim=0))
            p0 = projs[0]
            rc = p0._rotation_cache
            setattr(self, centroids_attr,
                    rc.get_centroids(p0.turbo_dim, p0.turbo_bits, p0.turbo_seed))
            setattr(self, rot_key_attr,
                    (p0.turbo_dim, p0.turbo_bits, p0.turbo_seed))
            setattr(self, out_d_attr, p0.out_features)

            # LoRA: batch SV across all experts if they share the same cluster
            sv_attr = f"_SV_{proj_attr.split('_')[0]}_cat"  # _SV_gate_cat / _SV_up_cat
            u_attr  = f"_U_{proj_attr.split('_')[0]}"        # _U_gate / _U_up
            cids = [p.cluster_id for p in projs]
            if (len(set(cids)) == 1 and cids[0] is not None
                    and all(p.SV is not None for p in projs)):
                setattr(self, sv_attr,
                        torch.cat([p.SV for p in projs], dim=0))  # (E*out_d, rank)
                setattr(self, u_attr, projs[0].U)
            else:
                setattr(self, sv_attr, None)
                setattr(self, u_attr, None)

        # Down proj: one RHT call for all experts; batch LoRA via bmm
        down_projs = [self.experts[ei].down_proj for ei in range(E)]
        p0d = down_projs[0]
        rc_d = p0d._rotation_cache
        self._down_centroids = rc_d.get_centroids(
            p0d.turbo_dim, p0d.turbo_bits, p0d.turbo_seed)
        self._down_x_rot_key = (p0d.turbo_dim, p0d.turbo_bits, p0d.turbo_seed)
        self._rht_signs_down = getattr(p0d, '_rht_signs', None)

        down_cids = [getattr(p, 'cluster_id', None) for p in down_projs]
        if (len(set(down_cids)) == 1 and down_cids[0] is not None
                and all(p.SV is not None for p in down_projs)
                and down_projs[0].U is not None):
            self._U_down = down_projs[0].U                                         # (inter_d, rank)
            self._SV_T_down_all = torch.stack(
                [p.SV.T for p in down_projs], dim=0)                               # (E, rank, out_d)
        else:
            self._U_down = None
            self._SV_T_down_all = None

        self._graph_cache_built = True

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

    def _precompute_xU_for_experts(self, hidden_states, active_expert_indices):
        """Lazy xU: only compute x @ U for clusters that have active experts.

        At decode time (top-k << num_experts), most clusters are inactive.
        This avoids computing x @ U for all ~15 clusters per layer when only
        ~4 are needed, reducing cuBLAS GEMV launches significantly.

        Args:
            hidden_states: (B, hidden_dim) input
            active_expert_indices: list of int, expert indices that are active

        Returns:
            dict: {proj_name: {cluster_id: (B, rank) tensor}}
        """
        if not self._cluster_map_built:
            self._build_cluster_map()

        xU_cache = {}
        for proj_name, groups in self._cluster_groups.items():
            proj_cache = {}
            # Only compute xU for clusters that contain at least one active expert
            needed_cids = set()
            for ei in active_expert_indices:
                cid = _get_cluster_id(self.experts[ei], proj_name)
                if cid is not None:
                    needed_cids.add(cid)
            for cid in needed_cids:
                if cid in groups:
                    U, _ = groups[cid]
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

    def _precompute_rotations(self, hidden_states):
        """Pre-compute rotated input for unique rotation configs among gate/up experts.

        Supports both QR (x @ Pi.T) and Hadamard (online RHT) rotation.

        Returns:
            dict: {(dim, bits, seed): (B, K) float16 tensor}
        """
        rot_cache = {}
        for proj_name in ("gate_proj", "up_proj"):
            for ei in range(self.num_experts):
                proj = getattr(self.experts[ei], proj_name, None)
                if proj is None or not hasattr(proj, 'turbo_dim'):
                    continue
                key = (proj.turbo_dim, proj.turbo_bits, proj.turbo_seed)
                if key in rot_cache:
                    continue
                # Hadamard online rotation (zero HBM read)
                if hasattr(proj, '_rht_signs') and proj._rht_signs is not None:
                    from hadamard_rotation import rht_forward
                    rot_cache[key] = rht_forward(hidden_states.float(), proj._rht_signs).half()
                # QR rotation (read Pi from HBM)
                elif hasattr(proj, '_rotation_cache') and proj._rotation_cache is not None:
                    Pi = proj._rotation_cache.get_pi(
                        proj.turbo_dim, proj.turbo_bits, proj.turbo_seed)
                    rot_cache[key] = (hidden_states.float() @ Pi.float().T).half()
        return rot_cache

    def _get_x_rot(self, rot_cache, expert, proj_name):
        """Look up pre-computed x_rot for a specific expert proj."""
        proj = getattr(expert, proj_name, None)
        if proj is None or not hasattr(proj, 'turbo_dim') or proj._rotation_cache is None:
            return None
        key = (proj.turbo_dim, proj.turbo_bits, proj.turbo_seed)
        return rot_cache.get(key)

    def _expert_forward(self, expert, x, xU_cache, expert_idx, rot_cache=None):
        """Forward through a single expert MLP with cluster-parallel LoRA."""
        gate_xU = self._get_precomputed(xU_cache, expert_idx, "gate_proj")
        up_xU = self._get_precomputed(xU_cache, expert_idx, "up_proj")

        gate_rot = self._get_x_rot(rot_cache, expert, "gate_proj") if rot_cache else None
        up_rot = self._get_x_rot(rot_cache, expert, "up_proj") if rot_cache else None

        gate_out = expert.gate_proj(x, precomputed_xU=gate_xU, precomputed_x_rot=gate_rot)
        up_out = expert.up_proj(x, precomputed_xU=up_xU, precomputed_x_rot=up_rot)
        h = expert.act_fn(gate_out) * up_out
        # down_proj: different input (h), different Pi dim — no precompute
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

        # Pre-compute x @ Pi.T for TurboQuant rotation (once per unique Pi)
        rot_cache = self._precompute_rotations(hidden_states)

        # Select forward path:
        # - decode mode (N small): iterate selected_experts directly, no nonzero
        # - graph mode: all experts execute, fixed shape (for CUDA Graph capture)
        # - sparse mode: gather/scatter with nonzero (for prefill)
        N = hidden_states.shape[0]

        # Pre-compute x @ U_shared for cluster-parallel LoRA.
        # For decode (N small), use lazy version: only compute xU for clusters
        # that contain at least one active expert (routing already known).
        # For prefill/graph, compute all clusters upfront.
        if self.graph_mode:
            xU_cache = {}  # handled inside _forward_graph (batched)
        elif N <= self._graph_threshold:
            active_indices = selected_experts.reshape(-1).tolist()
            xU_cache = self._precompute_xU_for_experts(hidden_states, active_indices)
        else:
            xU_cache = self._precompute_xU(hidden_states)

        if self.graph_mode:
            final = self._forward_graph(
                hidden_states, routing_weights, selected_experts, hidden_dim,
                xU_cache, rot_cache)
        elif N <= self._graph_threshold:
            final = self._forward_decode(
                hidden_states, routing_weights, selected_experts, hidden_dim,
                xU_cache, rot_cache)
        else:
            final = self._forward_sparse(
                hidden_states, routing_weights, selected_experts, hidden_dim,
                xU_cache, rot_cache)

        # Shared expert (already fixed-shape, graph-safe)
        shared_out = self.shared_expert(hidden_states)
        shared_out = (
            F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_out
        )
        final = final + shared_out

        final = final.reshape(batch_size, sequence_length, hidden_dim)
        return final, router_logits

    def _forward_decode(self, hidden_states, routing_weights,
                        selected_experts, hidden_dim, xU_cache, rot_cache):
        """Decode-optimized: batch active experts to reduce kernel launches.

        For decode (N=1, top_k active experts), batches the quantized matmul
        across all active experts of the same proj type into a single kernel
        call. Reduces kernel launches from 4*3=12 to ~3 (one per proj type).
        """
        N = hidden_states.shape[0]
        final = torch.zeros(
            N, hidden_dim,
            dtype=hidden_states.dtype, device=hidden_states.device,
        )

        for token_idx in range(N):
            x_token = hidden_states[token_idx:token_idx + 1]  # (1, hidden_dim)
            expert_indices = selected_experts[token_idx].tolist()
            weights = [routing_weights[token_idx, k]
                       for k in range(self.top_k)]

            # --- Batched gate_proj + up_proj (same input x_token) ---
            gate_outs = self._batched_proj_forward(
                x_token, expert_indices, "gate_proj", xU_cache, rot_cache)
            up_outs_list = self._batched_proj_forward(
                x_token, expert_indices, "up_proj", xU_cache, rot_cache)

            # Compute h = silu(gate) * up per expert, then batch down_proj
            h_list = []
            for k in range(self.top_k):
                h = self.experts[expert_indices[k]].act_fn(gate_outs[k]) * up_outs_list[k]
                h_list.append(h)

            # --- Batched down_proj (different inputs h per expert) ---
            down_outs = self._batched_down_forward(
                h_list, expert_indices, xU_cache)

            # Weighted sum
            for k in range(self.top_k):
                final[token_idx] += weights[k] * down_outs[k].squeeze(0)

        return final

    def _batched_proj_forward(self, x, expert_indices, proj_name, xU_cache, rot_cache):
        """Batch multiple experts' gate/up proj with dual-stream parallel execution.

        Stream 0 (main): batched turbo kernel
        Stream 1 (side): LoRA computation (parallel); uses bmm when all experts
                         share the same cluster to replace K GEMV dispatches with 1.
        Then merge results.

        Returns list of (1, out_d) tensors, one per expert.
        """
        from inference.kernels import turbo_dequant_matmul_fused, is_cuda_available

        experts_data = []
        for ei in expert_indices:
            proj = getattr(self.experts[ei], proj_name)
            experts_data.append(proj)

        # Check if all are turbo with rotation
        all_turbo = all(p.quant_type == "turbo" and
                        (p._rotation_cache is not None or
                         (hasattr(p, '_rht_signs') and p._rht_signs is not None))
                        for p in experts_data)

        if not all_turbo or not is_cuda_available():
            return self._batched_proj_forward_raw(x, expert_indices, proj_name,
                                                  xU_cache, rot_cache)

        # Batch turbo kernel: concat packed_indices and norms
        out_dims = [p.out_features for p in experts_data]
        packed_cat = torch.cat([p.packed_indices for p in experts_data], dim=0)
        norms_cat = torch.cat([p.norms for p in experts_data], dim=0)

        p0 = experts_data[0]
        centroids = p0._rotation_cache.get_centroids(
            p0.turbo_dim, p0.turbo_bits, p0.turbo_seed)

        # Get precomputed rotation (Hadamard or QR, already cached)
        x_rot = None
        key = (p0.turbo_dim, p0.turbo_bits, p0.turbo_seed)
        if rot_cache:
            x_rot = rot_cache.get(key)
        # Fallback: compute rotation now
        if x_rot is None:
            if hasattr(p0, '_rht_signs') and p0._rht_signs is not None:
                from hadamard_rotation import rht_forward
                x_rot = rht_forward(x.float(), p0._rht_signs).half()
            else:
                Pi = p0._rotation_cache.get_pi(
                    p0.turbo_dim, p0.turbo_bits, p0.turbo_seed)
                x_rot = (x.float() @ Pi.float().T).half()

        # Pi for kernel: only needed for QR path where precomputed_x_rot is None.
        Pi = None if x_rot is not None else p0._rotation_cache.get_pi(
            p0.turbo_dim, p0.turbo_bits, p0.turbo_seed)

        K_exp = len(expert_indices)
        cids = [experts_data[k].cluster_id for k in range(K_exp)]
        all_same_cluster = (len(set(cids)) == 1 and cids[0] is not None)
        all_have_sv = all(experts_data[k].SV is not None for k in range(K_exp))

        # --- Side stream: LoRA computation (parallel with main-stream turbo) ---
        side = self._get_side_stream()
        lora_outs = [None] * K_exp
        lora_cat = None  # (1, total_out_d) when same-cluster path used

        with torch.cuda.stream(side):
            if all_same_cluster and all_have_sv:
                # All active experts share the same U → compute a = x@U once,
                # then do a single (1, rank) @ (rank, K*out_d) GEMM by concatenating SVs.
                cid = cids[0]
                a = (xU_cache.get(proj_name) or {}).get(cid)
                if a is None:
                    a = x @ experts_data[0].U  # (1, rank)
                # SV_cat: (K*out_d, rank) — concatenate row-wise
                SV_cat = torch.cat([experts_data[k].SV for k in range(K_exp)], dim=0)
                lora_cat = a @ SV_cat.T  # (1, K*out_d) — one cuBLAS call
            else:
                # Different clusters or missing SV: per-expert LoRA
                for k, ei in enumerate(expert_indices):
                    proj = experts_data[k]
                    if proj.SV is not None:
                        cid = proj.cluster_id
                        a_k = (xU_cache.get(proj_name) or {}).get(cid)
                        if a_k is None:
                            a_k = x @ proj.U
                        lora_outs[k] = a_k @ proj.SV.T

        # --- Main stream: batched turbo kernel (runs in parallel with side stream) ---
        y_cat = turbo_dequant_matmul_fused(
            x, packed_cat, norms_cat, Pi, centroids,
            p0.turbo_bits, p0.turbo_dim,
            lora_USV=None, precomputed_x_rot=x_rot)

        # Wait for side stream, then add LoRA
        torch.cuda.current_stream().wait_stream(side)
        if lora_cat is not None:
            y_cat = y_cat + lora_cat
        else:
            for k in range(K_exp):
                if lora_outs[k] is not None:
                    od = out_dims[k]
                    offset = sum(out_dims[:k])
                    y_cat[:, offset:offset + od] = y_cat[:, offset:offset + od] + lora_outs[k]

        # Split y_cat into per-expert results
        results = []
        offset = 0
        for od in out_dims:
            results.append(y_cat[:, offset:offset + od])
            offset += od
        return results

    def _batched_proj_forward_raw(self, x, expert_indices, proj_name,
                                   xU_cache, rot_cache):
        """Sequential fallback for proj forward."""
        results = []
        for ei in expert_indices:
            proj = getattr(self.experts[ei], proj_name)
            xU = self._get_precomputed(xU_cache, ei, proj_name)
            x_rot = self._get_x_rot(rot_cache, self.experts[ei], proj_name) if rot_cache else None
            y = proj(x, precomputed_xU=xU, precomputed_x_rot=x_rot)
            results.append(y)
        return results

    def _batched_down_forward(self, h_list, expert_indices, xU_cache):
        """Down_proj with dual-stream parallel: rotation+turbo ∥ LoRA.

        Each expert has different input h, so turbo kernel is per-expert.
        But rotation+turbo and LoRA are independent and run in parallel.

        When all experts share the same cluster (same U), batches:
          - h_cat @ U as a single GEMM: (K, inter_d) @ (inter_d, rank) → (K, rank)
          - K SV matmuls as a single bmm: (K, 1, rank) @ (K, rank, out_d) → (K, out_d)
        Replaces K individual GEMVs with 1 GEMM + 1 bmm.
        """
        from inference.kernels import turbo_dequant_matmul_fused, is_cuda_available

        experts_data = []
        for ei in expert_indices:
            experts_data.append(getattr(self.experts[ei], "down_proj"))

        all_turbo = all(p.quant_type == "turbo" and
                        (p._rotation_cache is not None or
                         (hasattr(p, '_rht_signs') and p._rht_signs is not None))
                        for p in experts_data)

        if not all_turbo or not is_cuda_available() or len(h_list) == 0:
            results = []
            for k, ei in enumerate(expert_indices):
                y = self.experts[ei].down_proj(h_list[k])
                results.append(y)
            return results

        p0 = experts_data[0]
        centroids = p0._rotation_cache.get_centroids(
            p0.turbo_dim, p0.turbo_bits, p0.turbo_seed)

        # Determine rotation type and batch h.float() cast upfront.
        # All down_proj experts share the same rotation config (same dim/seed),
        # so we can cast all h's at once (1 cast + 1 kernel) instead of per-expert.
        all_rht = all(
            hasattr(p, '_rht_signs') and p._rht_signs is not None
            for p in experts_data
        )
        if all_rht:
            from hadamard_rotation import rht_forward
            rht_signs = experts_data[0]._rht_signs
            # Single fp16→fp32 cast + single rht_forward call for all experts
            h_stacked = torch.cat(h_list, dim=0).float()  # (K, inter_d)
            h_rots = rht_forward(h_stacked, rht_signs).half()  # (K, inter_d) fp16
            Pi = None  # not needed when precomputed_x_rot is provided
        else:
            Pi = p0._rotation_cache.get_pi(
                p0.turbo_dim, p0.turbo_bits, p0.turbo_seed)
            h_rots = None

        K_exp = len(expert_indices)
        down_cids = [getattr(p, 'cluster_id', None) for p in experts_data]
        all_same_cluster = (len(set(down_cids)) == 1 and down_cids[0] is not None)
        all_have_sv = all(p.SV is not None for p in experts_data)

        # --- Side stream: batch LoRA (parallel with main turbo) ---
        # When all experts share U: one GEMM for h_cat@U → (K, rank),
        # then K separate GEMVs a[k]@SV_k.T → (K, out_d).
        side = self._get_side_stream()
        lora_outs = [None] * K_exp

        with torch.cuda.stream(side):
            if all_have_sv and all_same_cluster and experts_data[0].U is not None:
                h_cat = torch.cat(h_list, dim=0)           # (K, inter_d) fp16
                a_stacked = h_cat @ experts_data[0].U       # (K, rank) fp16 — one GEMM
                # Per-expert SV matmuls: each (1, rank) @ (rank, out_d)
                for k in range(K_exp):
                    lora_outs[k] = a_stacked[k:k + 1] @ experts_data[k].SV.T
            elif all_have_sv and not all_same_cluster:
                for k in range(K_exp):
                    proj = experts_data[k]
                    lora_outs[k] = (h_list[k] @ proj.U) @ proj.SV.T

        # --- Main stream: per-expert turbo kernel (runs in parallel with side) ---
        results = []
        for k, ei in enumerate(expert_indices):
            proj = experts_data[k]
            h = h_list[k]  # (1, inter_d)

            if all_rht:
                h_rot = h_rots[k:k + 1]  # (1, inter_d) fp16 view, no extra cast
            else:
                h_rot = (h_list[k].float() @ Pi.float().T).half()

            y_q = turbo_dequant_matmul_fused(
                h, proj.packed_indices, proj.norms, Pi, centroids,
                proj.turbo_bits, proj.turbo_dim,
                lora_USV=None, precomputed_x_rot=h_rot)
            results.append(y_q)

        # Wait for side stream, then add LoRA
        torch.cuda.current_stream().wait_stream(side)
        for k in range(K_exp):
            if lora_outs[k] is not None:
                results[k] = results[k] + lora_outs[k]

        return results

    def _forward_sparse(self, hidden_states, routing_weights,
                        selected_experts, hidden_dim, xU_cache, rot_cache):
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

            # Gather precomputed rotations for selected tokens
            sparse_rot_cache = {}
            for key, x_rot_full in rot_cache.items():
                sparse_rot_cache[key] = x_rot_full[top_x]

            current_out = self._expert_forward(
                expert_layer, current_state, sparse_xU_cache, expert_idx,
                rot_cache=sparse_rot_cache)
            current_out *= routing_weights[top_x, idx, None]
            final.index_add_(0, top_x, current_out.to(hidden_states.dtype))
        return final

    def _forward_graph(self, hidden_states, routing_weights,
                       selected_experts, hidden_dim, xU_cache, rot_cache):
        """Batched graph-mode forward: ~8 turbo calls/layer vs 180.

        Gate and up run as 1 batched turbo call each (all E experts share the
        same input hidden_states).  Down still needs E individual calls
        (different inputs h per expert), but rotation and LoRA are batched.
        """
        if not self._graph_cache_built:
            self._build_graph_cache()

        from inference.kernels import turbo_dequant_matmul_fused

        N = hidden_states.shape[0]
        E = self.num_experts

        # 1. Routing: (N, top_k) → (N, E)
        full_weights = torch.zeros(
            N, E, dtype=routing_weights.dtype, device=hidden_states.device)
        full_weights.scatter_(1, selected_experts, routing_weights)

        # 2. Gate proj: 1 turbo call for all E experts (same input)
        x_rot_gate = rot_cache.get(self._gate_x_rot_key)
        p0_gate = self.experts[0].gate_proj
        y_gate_all = turbo_dequant_matmul_fused(
            hidden_states, self._packed_gate_all, self._norms_gate_all,
            None, self._gate_centroids,
            p0_gate.turbo_bits, p0_gate.turbo_dim,
            lora_USV=None, precomputed_x_rot=x_rot_gate)            # (N, E*out_gate)
        if self._U_gate is not None:
            a_gate = hidden_states @ self._U_gate                    # (N, rank)
            y_gate_all = y_gate_all + a_gate @ self._SV_gate_cat.T  # (N, E*out_gate)

        # 3. Up proj: 1 turbo call for all E experts
        x_rot_up = rot_cache.get(self._up_x_rot_key)
        p0_up = self.experts[0].up_proj
        y_up_all = turbo_dequant_matmul_fused(
            hidden_states, self._packed_up_all, self._norms_up_all,
            None, self._up_centroids,
            p0_up.turbo_bits, p0_up.turbo_dim,
            lora_USV=None, precomputed_x_rot=x_rot_up)               # (N, E*out_up)
        if self._U_up is not None:
            a_up = hidden_states @ self._U_up
            y_up_all = y_up_all + a_up @ self._SV_up_cat.T

        # 4. Activation on stacked (N*E, inter_d) tensor
        out_gate = self._gate_out_d
        out_up   = self._up_out_d
        gate_all = y_gate_all.view(N * E, out_gate)
        up_all   = y_up_all.view(N * E, out_up)
        h_all = self.experts[0].act_fn(gate_all) * up_all            # (N*E, inter_d)

        # 5. Down rotation: 1 RHT call for all N*E vectors
        if self._rht_signs_down is not None:
            from hadamard_rotation import rht_forward
            h_rot_all = rht_forward(h_all.float(), self._rht_signs_down).half()
        else:
            Pi_down = self.experts[0].down_proj._rotation_cache.get_pi(
                *self._down_x_rot_key)
            h_rot_all = (h_all.float() @ Pi_down.float().T).half()   # (N*E, inter_d)

        # 6. Down turbo: E individual calls (different packed weights per expert)
        p0_down = self.experts[0].down_proj
        down_outs = []
        for k in range(E):
            proj_down = self.experts[k].down_proj
            y_k = turbo_dequant_matmul_fused(
                h_all[k * N: (k + 1) * N],
                proj_down.packed_indices, proj_down.norms,
                None, self._down_centroids,
                p0_down.turbo_bits, p0_down.turbo_dim,
                lora_USV=None,
                precomputed_x_rot=h_rot_all[k * N: (k + 1) * N])
            down_outs.append(y_k)

        # 7. Down LoRA: 1 GEMM + 1 bmm when all experts share the same cluster
        out_all = torch.stack(down_outs, dim=0)    # (E, N, hidden_dim)
        if self._U_down is not None:
            a_down = h_all @ self._U_down          # (N*E, rank)
            a_down_v = a_down.view(E, N, -1)       # (E, N, rank)
            lora_down = torch.bmm(a_down_v, self._SV_T_down_all)   # (E, N, hidden_dim)
            out_all = out_all + lora_down

        # 8. Routing accumulation: 1 bmm replaces E scatter-adds
        out_all_t = out_all.permute(1, 0, 2)                        # (N, E, hidden_dim)
        final = torch.bmm(
            full_weights.unsqueeze(1), out_all_t).squeeze(1)        # (N, hidden_dim)
        return final
