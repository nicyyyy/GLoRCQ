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
        # Mixtral's MixtralSparseMoeBlock stores `num_experts_per_tok` (top_k
        # alias) and does not expose `norm_topk_prob`. Qwen has both.
        self.norm_topk_prob = getattr(original_block, 'norm_topk_prob', True)
        self.gate = original_block.gate
        self.experts = original_block.experts
        # Qwen1.5-MoE has a shared expert with a scalar gate. Mixtral has neither.
        self.shared_expert = getattr(original_block, 'shared_expert', None)
        self.shared_expert_gate = getattr(original_block, 'shared_expert_gate', None)
        # Alias Mixtral's w1/w3/w2 → gate_proj/up_proj/down_proj so the rest of
        # this module can address experts uniformly. Attribute assignment on
        # nn.Module shares submodule identity (state_dict / .parameters() /
        # forward all still work through both names).
        if len(self.experts) > 0 and not hasattr(self.experts[0], 'gate_proj') \
                and hasattr(self.experts[0], 'w1'):
            for e in self.experts:
                e.gate_proj = e.w1
                e.up_proj   = e.w3
                e.down_proj = e.w2
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

        # Global U pool (shared across all layers, set by model_builder)
        self._global_pool_gate = None  # (hidden_dim, K_total * rank)
        self._global_pool_up   = None
        self._gate_id_map = None       # {(wtype_key, group_id): global_col_idx}
        self._up_id_map   = None

        # Persistent side stream for parallel LoRA execution
        self._side_stream = None

    def _get_side_stream(self):
        """Get or create persistent side stream for parallel LoRA."""
        if self._side_stream is None:
            self._side_stream = torch.cuda.Stream()
        return self._side_stream

    def set_global_pool(self, pools, id_maps):
        """Install cross-layer shared U matrix pool (called by model_builder).

        Pools are shared across all 24 MoE layers so that the same cluster's
        U matrix lives at one HBM address → GPU L2 cache reuse across layers.

        Args:
            pools: {'gate': (hidden_dim, K_total*rank) tensor,
                    'up':   (hidden_dim, K_total*rank) tensor}
            id_maps: {'gate': {(wtype_key, group_id): global_col_index}, ...}
        """
        self._global_pool_gate = pools.get('gate')
        self._global_pool_up   = pools.get('up')
        self._gate_id_map = id_maps.get('gate')
        self._up_id_map   = id_maps.get('up')
        # Build the cluster→expert mapping alongside the pool install.
        self._build_cluster_map()

    def _build_cluster_map(self):
        """Build mapping from cluster_id to expert indices for shared U.

        Called on demand from _precompute_xU* when set_global_pool has not
        run (e.g. Mixtral without a cross-layer pool).
        """
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

        # Down proj: one RHT call for all experts; batch LoRA via bmm;
        # grouped-GEMV via pre-concatenated weights
        down_projs = [self.experts[ei].down_proj for ei in range(E)]
        p0d = down_projs[0]
        rc_d = p0d._rotation_cache
        self._down_centroids = rc_d.get_centroids(
            p0d.turbo_dim, p0d.turbo_bits, p0d.turbo_seed)
        self._down_x_rot_key = (p0d.turbo_dim, p0d.turbo_bits, p0d.turbo_seed)
        self._rht_signs_down = getattr(p0d, '_rht_signs', None)

        # Pre-concatenate packed_indices and norms for grouped-GEMV
        self._packed_down_all = torch.cat(
            [p.packed_indices for p in down_projs], dim=0)   # (E*out_d, K/4)
        self._norms_down_all = torch.cat(
            [p.norms for p in down_projs], dim=0)            # (E*out_d,)
        self._down_out_d = p0d.out_features

        self._prefetch_stream = self._get_side_stream()

        # --- Batched LoRA for graph-mode parallel execution ---
        # Instead of per-cluster loops (many small kernels), batch everything
        # into 2 GPU ops per projection: 1 GEMV (U) + 1 BMM (SV).
        #
        # Gate/Up layout (input hidden_states is SHARED across all experts):
        #   _U_cat_{gate,up}: (hidden_dim, K*rank)  — K cluster U mats concatenated
        #   _SV_all_{gate,up}: (E, out_d, rank)     — per-expert SV matrices
        #   _cluster_idx_{gate,up}: (E,) long       — which cluster each expert uses
        #
        # Down layout (each expert has its OWN input h_all[i]):
        #   _U_per_expert_down: (E, inter_d, rank)  — U duplicated per expert
        #   _SV_all_down: (E, hidden_dim, rank)      — per-expert SV matrices
        #
        # Experts without LoRA get zero SV (and zero U for down) so their
        # contribution is automatically zero.

        def _build_gate_up_lora(projs, out_d, device, pool, id_map):
            """Build batched LoRA tensors for gate or up projection."""
            from collections import defaultdict
            cid_to_U = {}
            cid_list = []        # ordered cluster id list
            cid_to_idx = {}      # cid → index in cid_list
            rank = None
            for p in projs:
                cid = getattr(p, 'cluster_id', None)
                if cid is None or not hasattr(p, 'U') or p.U is None:
                    continue
                if not hasattr(p, 'SV') or p.SV is None:
                    continue
                if cid not in cid_to_idx:
                    cid_to_idx[cid] = len(cid_list)
                    cid_list.append(cid)
                    cid_to_U[cid] = p.U
                if rank is None:
                    rank = p.U.shape[1]

            if not cid_list:
                return None, None, None, rank or 0

            K = len(cid_list)
            # U_cat: (hidden_dim, K*rank)
            U_cat = torch.cat([cid_to_U[c] for c in cid_list], dim=1)

            E = len(projs)
            # Use the SV's own dtype so bf16 models (Mixtral) don't hit dtype-mismatch
            # errors in downstream bmm. Fallback to fp16 if no expert has SV yet.
            _sv_dtype = torch.float16
            for p in projs:
                if hasattr(p, 'SV') and p.SV is not None:
                    _sv_dtype = p.SV.dtype
                    break
            # SV_all: (E, out_d, rank) — zero for experts without LoRA
            SV_all = torch.zeros(E, out_d, rank, dtype=_sv_dtype, device=device)
            # cluster_idx: (E,) — default 0 (will be multiplied by zero SV anyway)
            cluster_idx = torch.zeros(E, dtype=torch.long, device=device)
            for ei, p in enumerate(projs):
                cid = getattr(p, 'cluster_id', None)
                if cid is not None and cid in cid_to_idx and hasattr(p, 'SV') and p.SV is not None:
                    SV_all[ei] = p.SV    # p.SV shape: (out_d, rank)
                    cluster_idx[ei] = cid_to_idx[cid]

            return U_cat, SV_all, cluster_idx, rank

        def _build_down_lora(projs, hidden_dim, device):
            """Build batched LoRA tensors for down projection."""
            rank = None
            E = len(projs)
            for p in projs:
                if hasattr(p, 'U') and p.U is not None:
                    rank = p.U.shape[1]
                    inter_d = p.U.shape[0]
                    break
            if rank is None:
                return None, None

            # Match projection's own dtype (bf16 for Mixtral, fp16 for Qwen).
            _u_dtype = torch.float16
            _sv_dtype = torch.float16
            for p in projs:
                if hasattr(p, 'U') and p.U is not None:
                    _u_dtype = p.U.dtype
                if hasattr(p, 'SV') and p.SV is not None:
                    _sv_dtype = p.SV.dtype
                if _u_dtype != torch.float16 or _sv_dtype != torch.float16:
                    break
            # U_per_expert: (E, inter_d, rank) — duplicate U per expert
            U_per = torch.zeros(E, inter_d, rank, dtype=_u_dtype, device=device)
            # SV_all: (E, hidden_dim, rank)
            SV_all = torch.zeros(E, hidden_dim, rank, dtype=_sv_dtype, device=device)
            any_lora = False
            for ei, p in enumerate(projs):
                if (hasattr(p, 'U') and p.U is not None
                        and hasattr(p, 'SV') and p.SV is not None):
                    U_per[ei] = p.U
                    SV_all[ei] = p.SV    # p.SV shape: (hidden_dim, rank)
                    any_lora = True
            if not any_lora:
                return None, None
            return U_per, SV_all

        _E = self.num_experts
        _dev = self.experts[0].gate_proj.packed_indices.device
        _gate_projs = [self.experts[ei].gate_proj for ei in range(_E)]
        _up_projs   = [self.experts[ei].up_proj   for ei in range(_E)]
        _down_projs = [self.experts[ei].down_proj for ei in range(_E)]

        # Gate / Up batched LoRA
        self._U_cat_gate, self._SV_all_gate, self._cluster_idx_gate, _ = \
            _build_gate_up_lora(_gate_projs, self._gate_out_d, _dev,
                                self._global_pool_gate, self._gate_id_map)
        self._U_cat_up,   self._SV_all_up,   self._cluster_idx_up,   _ = \
            _build_gate_up_lora(_up_projs,   self._up_out_d,   _dev,
                                self._global_pool_up,   self._up_id_map)

        # Down batched LoRA: down proj output = hidden_dim
        self._U_per_expert_down, self._SV_all_down = \
            _build_down_lora(_down_projs, self._down_out_d, _dev)

        # Sync events: side stream records when each LoRA is done
        self._ev_lora_gate = torch.cuda.Event()
        self._ev_lora_up   = torch.cuda.Event()
        self._ev_lora_down = torch.cuda.Event()

        self._graph_cache_built = True

    def _build_graph_cache_vq4(self):
        """Pre-concatenate VQ4 expert weights for graph-mode forward.

        For gate/up (input hidden_states shared across all experts):
          _vq4_codes_gate_all:      (E*out_d, in_d/vdim) uint8
          _vq4_centroids_gate_all:  (E*n_cb, K_CB, vdim)  fp16
          _vq4_sigma_gate:          (E, in_d) int32       — per-expert permutation
          _vq4_diagI_gate:          (in_d, in_d) fp16     — shared diagI
          _vq4_gate_out_d:          out_d
          _vq4_gate_ncb, _cpcb

        For down (each expert has own input h):
          _vq4_codes_down_all, _vq4_centroids_down_all, _vq4_sigma_down,
          _vq4_diagI_down, _vq4_down_out_d, _vq4_down_ncb, _cpcb

        LoRA (Sa breaks cluster-parallel U; use per-expert U in graph):
          _vq4_Sa_gate: (E, in_d) fp16
          _vq4_U_gate:  (E, in_d, rank) fp16 — per-expert U (larger than turbo's cat)
          _vq4_SV_gate: (E, out_d, rank) fp16
        """
        E = self.num_experts
        _dev = self.experts[0].gate_proj.vq_codes.device

        for proj_attr, prefix in [("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")]:
            projs = [getattr(self.experts[ei], proj_attr) for ei in range(E)]
            # Find a reference expert with valid vq_codes to derive shapes
            ref_p = None
            for p in projs:
                if getattr(p, 'vq_codes', None) is not None:
                    ref_p = p
                    break
            if ref_p is None:
                # No valid experts in this layer — very rare; disable graph fallback
                self.graph_mode = False
                self._graph_cache_built = False
                return

            # For experts missing vq_codes (Phase 2 max_err skip), pad with zero
            # codes/centroids so their output is 0 and they contribute nothing.
            def _get_codes(p):
                if getattr(p, 'vq_codes', None) is not None:
                    return p.vq_codes
                return torch.zeros_like(ref_p.vq_codes)
            def _get_centroids(p):
                if getattr(p, 'vq_centroids', None) is not None:
                    return p.vq_centroids
                return torch.zeros_like(ref_p.vq_centroids)
            def _get_sigma(p):
                if getattr(p, 'vq_perm_sigma', None) is not None:
                    return p.vq_perm_sigma.int()
                return torch.arange(ref_p.in_features,
                                     dtype=torch.int32,
                                     device=ref_p.vq_codes.device)
            def _get_diagI(p):
                if getattr(p, 'vq_diagI', None) is not None:
                    return p.vq_diagI
                return ref_p.vq_diagI
            def _get_infeat(p):
                return p.in_features if hasattr(p, 'in_features') else ref_p.in_features
            def _get_outfeat(p):
                return p.out_features if hasattr(p, 'out_features') else ref_p.out_features
            p0 = ref_p

            codes_cat = torch.cat([_get_codes(p) for p in projs], dim=0).contiguous()
            centroids_cat = torch.cat([_get_centroids(p) for p in projs], dim=0).contiguous()
            # sigma stack (E, in_d)
            sigma_stack = torch.stack([_get_sigma(p) for p in projs], dim=0).contiguous()

            setattr(self, f'_vq4_codes_{prefix}_all', codes_cat)
            setattr(self, f'_vq4_centroids_{prefix}_all', centroids_cat)
            setattr(self, f'_vq4_sigma_{prefix}', sigma_stack)

            # Flat int64 sigma for torch.index_select (skips gather+expand+.long() chain).
            # For gate/up: applied to (N, hidden) → (N, E*in_d). Flat index is just
            # sigma_stack.view(-1). For down: applied to (N, E*inter_d) — need
            # per-expert offset added to the index.
            _p0 = ref_p
            _in_d = _get_infeat(_p0)
            sigma_int64 = sigma_stack.long()                              # (E, in_d)
            if prefix == 'down':
                # h_all flattened to (N, E*inter_d); index needs offset e*inter_d
                offsets = torch.arange(E, device=sigma_int64.device,
                                        dtype=torch.int64).unsqueeze(1) * _in_d
                sigma_flat = (sigma_int64 + offsets).view(-1).contiguous()
            else:
                # x_h.view(-1) flattens (N, hidden) — but only valid for N=1.
                # For N>1 use index_select on dim=1 with sigma_stack.view(-1).
                sigma_flat = sigma_int64.view(-1).contiguous()             # (E*in_d,)
            setattr(self, f'_vq4_sigma_flat_{prefix}', sigma_flat)
            setattr(self, f'_vq4_diagI_{prefix}', p0.vq_diagI)   # shared
            setattr(self, f'_vq4_{prefix}_out_d', p0.out_features)
            setattr(self, f'_vq4_{prefix}_in_d',  p0.in_features)
            n_cb, K_cb, vdim = p0.vq_centroids.shape
            codes_per_row = p0.vq_codes.shape[1]
            setattr(self, f'_vq4_{prefix}_ncb', n_cb)
            setattr(self, f'_vq4_{prefix}_cpcb', codes_per_row // n_cb)
            setattr(self, f'_vq4_{prefix}_vdim', vdim)

            # LoRA tensors (per-expert): Sa (in_d), U (in_d, rank), SV (out_d, rank)
            rank = None
            for p in projs:
                if getattr(p, 'U', None) is not None:
                    rank = p.U.shape[1]
                    break
            in_d = _get_infeat(p0)
            out_d = _get_outfeat(p0)
            _R = rank or 32
            # Use the projection's own fp dtype (bf16 for Mixtral, fp16 for Qwen).
            # Sa/U/SV are typically stored at the same fp width, chosen from
            # whichever expert has non-None tensors first.
            _dt = torch.float16
            for p in projs:
                for a in ('Sa', 'U', 'SV'):
                    t = getattr(p, a, None)
                    if t is not None and t.dtype.is_floating_point:
                        _dt = t.dtype
                        break
                if _dt != torch.float16:
                    break
            Sa_all = torch.zeros(E, in_d, dtype=_dt, device=_dev)
            U_all  = torch.zeros(E, in_d, _R, dtype=_dt, device=_dev)
            SV_all = torch.zeros(E, out_d, _R, dtype=_dt, device=_dev)
            for ei, p in enumerate(projs):
                if getattr(p, 'Sa', None) is not None:
                    Sa_all[ei] = p.Sa
                else:
                    Sa_all[ei].fill_(1.0)   # multiplicative identity
                if getattr(p, 'U', None) is not None:
                    U_all[ei] = p.U
                if getattr(p, 'SV', None) is not None:
                    SV_all[ei] = p.SV
            setattr(self, f'_vq4_Sa_{prefix}', Sa_all)
            setattr(self, f'_vq4_U_{prefix}',  U_all)
            setattr(self, f'_vq4_SV_{prefix}', SV_all)
            setattr(self, f'_vq4_{prefix}_rank', _R)

            # Sa-fused U: U_sa[e, i, r] = Sa[e, i] * U[e, i, r].
            # This lets forward do x @ U_sa[e] directly instead of (x*Sa) @ U.
            U_sa = U_all * Sa_all.unsqueeze(-1)                     # (E, in_d, rank)
            setattr(self, f'_vq4_U_sa_{prefix}', U_sa)
            # For gate/up: input x_h is shared across E experts. Stack U_sa into
            # (in_d, E*rank) so LoRA becomes a single giant GEMM.
            U_sa_flat = U_sa.permute(1, 0, 2).contiguous().view(in_d, E * _R)
            setattr(self, f'_vq4_U_sa_flat_{prefix}', U_sa_flat)
            # For down_proj: K-parallel kernel needs U_sa transposed as (E, rank, in_d)
            # so K (in_d) is contiguous inner dim for coalesced reads.
            U_sa_T = U_sa.transpose(1, 2).contiguous()               # (E, rank, in_d)
            setattr(self, f'_vq4_U_sa_T_{prefix}', U_sa_T)
            # SV.T pre-transposed for bmm avoidance (E, rank, out_d) contiguous
            SV_T = SV_all.transpose(1, 2).contiguous()               # (E, rank, out_d)
            setattr(self, f'_vq4_SV_T_{prefix}', SV_T)
            # SV block-diagonal: not truly block-diagonal but (E, out_d, rank) → per-expert
            # For LoRA output y[n, e, o] = a[n, e, r] * SV[e, o, r], use bmm.

        # P3: detect if gate/up share the same PD_rot (sigma+diagI). If so, we
        # can compute x_gate_rot once and reuse for up_proj — skips one
        # index_select + one (E, in_d) @ (in_d, in_d) GEMM per layer.
        try:
            self._vq4_share_gate_up_rot = (
                self._vq4_gate_in_d == self._vq4_up_in_d
                and self._vq4_diagI_gate.data_ptr() == self._vq4_diagI_up.data_ptr()
                and torch.equal(self._vq4_sigma_flat_gate, self._vq4_sigma_flat_up)
            )
        except Exception:
            self._vq4_share_gate_up_rot = False

        # P4: concat U_sa_flat_gate + U_sa_flat_up into ONE wider tensor so we
        # do 1 GEMM instead of 2 (both operate on the same x_h). Requires same
        # rank for gate/up (typically true).
        try:
            if self._vq4_gate_rank == self._vq4_up_rank:
                self._vq4_U_sa_flat_gate_up = torch.cat(
                    [self._vq4_U_sa_flat_gate, self._vq4_U_sa_flat_up], dim=1
                ).contiguous()  # (in_d, 2*E*rank)
                self._vq4_fused_gate_up_U = True
            else:
                self._vq4_fused_gate_up_U = False
        except Exception:
            self._vq4_fused_gate_up_U = False

        self._prefetch_stream = self._get_side_stream()
        self._ev_lora_gate = torch.cuda.Event()
        self._ev_lora_up   = torch.cuda.Event()
        self._ev_lora_down = torch.cuda.Event()

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

    def _precompute_PD_rotations(self, hidden_states, active_expert_indices=None):
        """Pre-compute x @ PD_rot for VQ4 experts.

        Each expert has its own permutation sigma but shares the diagI matrix
        by (in_d, rotate_size, partial_size). Since sigma varies per expert,
        we can't dedup rotations across experts — but we can still cache when
        multiple projections have literally the same sigma tensor.

        If active_expert_indices is provided, only compute for those experts
        (avoids doing 120 rotations per layer when only 4 are active in decode).

        Returns:
            dict: {id(vq_perm_sigma): (B, in_d) fp16 tensor}
        """
        rot_cache = {}
        if active_expert_indices is not None:
            expert_iter = set(active_expert_indices)
        else:
            expert_iter = range(self.num_experts)
        for proj_name in ("gate_proj", "up_proj"):
            for ei in expert_iter:
                proj = getattr(self.experts[ei], proj_name, None)
                if proj is None or getattr(proj, 'quant_type', None) != 'vq4':
                    continue
                if getattr(proj, 'vq_perm_sigma', None) is None:
                    continue
                key = id(proj.vq_perm_sigma)
                if key in rot_cache:
                    continue
                x_permuted = hidden_states.half()[..., proj.vq_perm_sigma]
                rot_cache[key] = x_permuted @ proj.vq_diagI
        return rot_cache

    def _get_x_rot(self, rot_cache, expert, proj_name):
        """Look up pre-computed x_rot for a specific expert proj."""
        proj = getattr(expert, proj_name, None)
        if proj is None:
            return None
        # VQ4 path: key by id(vq_perm_sigma)
        if getattr(proj, 'quant_type', None) == 'vq4' and getattr(proj, 'vq_perm_sigma', None) is not None:
            return rot_cache.get(id(proj.vq_perm_sigma))
        # TurboQuant path: key by (dim, bits, seed)
        if not hasattr(proj, 'turbo_dim') or proj._rotation_cache is None:
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

        # In graph_mode, _forward_graph_vq4 recomputes rotations as batched
        # (E, in_d) @ (in_d, in_d) — the per-expert precompute at line ~508 is
        # redundant and expensive (60 * 24 = 1440 (1,2048)@(2048,2048) matmuls
        # per token, ~29ms wasted). Skip both precomputes in graph_mode.
        if self.graph_mode:
            rot_cache = {}
        else:
            # Pre-compute x @ Pi.T for TurboQuant rotation (once per unique Pi)
            rot_cache = self._precompute_rotations(hidden_states)

            # Pre-compute x @ PD_rot for VQ4 experts (dedup by id(vq_perm_sigma)).
            # For decode, only compute for active experts (huge savings when top_k<<E).
            N_tokens = hidden_states.shape[0]
            if N_tokens <= self._graph_threshold:
                active_ei = set(selected_experts.reshape(-1).tolist())
                pd_cache = self._precompute_PD_rotations(hidden_states, active_ei)
            else:
                pd_cache = self._precompute_PD_rotations(hidden_states)
            rot_cache.update(pd_cache)

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

        # Shared expert (Qwen only). Mixtral has no shared_expert → skip.
        if self.shared_expert is not None:
            shared_out = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
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
        try:
            from inference.kernels import (
                vq4_dequant_grouped_gemv, is_vq4_cuda_available,
            )
            _has_vq4 = is_vq4_cuda_available()
        except ImportError:
            _has_vq4 = False

        experts_data = []
        for ei in expert_indices:
            proj = getattr(self.experts[ei], proj_name)
            experts_data.append(proj)

        # Check if all are turbo with rotation
        all_turbo = all(getattr(p, 'quant_type', None) == "turbo" and
                        (p._rotation_cache is not None or
                         (hasattr(p, '_rht_signs') and p._rht_signs is not None))
                        for p in experts_data)

        # Check if all are vq4
        all_vq4 = all(getattr(p, 'quant_type', None) == "vq4"
                       and getattr(p, 'vq_codes', None) is not None
                       for p in experts_data) and _has_vq4

        if all_vq4:
            return self._batched_proj_forward_vq4(
                x, expert_indices, experts_data, proj_name, xU_cache, rot_cache)

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

    def _batched_proj_forward_vq4(self, x, expert_indices, experts_data,
                                    proj_name, xU_cache, rot_cache):
        """VQ4 dual-stream: grouped_gemv (main) ∥ LoRA (side).

        For MoE gate/up, the input x is the same for all active experts. So the
        grouped kernel needs one x_rot per unique PD_rot (typically 1). If all
        experts share the same PD_rot (usually true within one cluster), we can
        call grouped_gemv once with concatenated codes and centroids.
        """
        from inference.kernels import vq4_dequant_grouped_gemv, vq4_dequant_matmul

        K_exp = len(expert_indices)
        p0 = experts_data[0]

        # Precomputed x_rot: dedup by id(vq_perm_sigma) since sigma is per-expert.
        pd_ids = [id(p.vq_perm_sigma) for p in experts_data]
        all_same_pd = (len(set(pd_ids)) == 1)

        x_rot_shared = None
        if all_same_pd and rot_cache is not None:
            x_rot_shared = rot_cache.get(pd_ids[0])
        if x_rot_shared is None and all_same_pd:
            x_permuted = x.half()[..., p0.vq_perm_sigma]
            x_rot_shared = x_permuted @ p0.vq_diagI

        # Cluster info for LoRA
        cids = [experts_data[k].cluster_id for k in range(K_exp)]
        all_same_cluster = (len(set(cids)) == 1 and cids[0] is not None)
        all_have_sv = all(experts_data[k].SV is not None for k in range(K_exp))
        # Sa breaks cluster-parallel: any expert with Sa needs its own (x*Sa) @ U
        any_has_sa = any(experts_data[k].Sa is not None for k in range(K_exp))

        # ---- Side stream: LoRA computation ----
        side = self._get_side_stream()
        lora_outs = [None] * K_exp
        lora_cat = None

        with torch.cuda.stream(side):
            if all_same_cluster and all_have_sv and not any_has_sa:
                cid = cids[0]
                a = (xU_cache.get(proj_name) or {}).get(cid)
                if a is None:
                    a = x @ experts_data[0].U   # (1, rank)
                SV_cat = torch.cat([experts_data[k].SV for k in range(K_exp)], dim=0)
                lora_cat = a @ SV_cat.T   # (1, K*out_d)
            else:
                for k, ei in enumerate(expert_indices):
                    proj = experts_data[k]
                    if proj.SV is not None:
                        x_for_lora = x * proj.Sa if proj.Sa is not None else x
                        a_k = x_for_lora @ proj.U     # per-expert (1, rank)
                        lora_outs[k] = a_k @ proj.SV.T

        # ---- Main stream: batched vq4 kernel(s) ----
        # If all experts have same shape, group them into one grouped_gemv call
        # even when sigmas differ — each expert's rotation is computed and
        # stacked into x_grouped.
        out_dims = [p.out_features for p in experts_data]
        same_shape = all(out_dims[k] == out_dims[0] for k in range(K_exp))

        if same_shape:
            # Pre-cache concatenated codes+centroids per (proj_name, tuple(expert_indices))
            # so we don't torch.cat() every forward.
            cache_key = (proj_name, tuple(expert_indices))
            cached = getattr(self, '_vq4_cat_cache', {}).get(cache_key)
            if cached is None:
                codes_cat = torch.cat(
                    [experts_data[k].vq_codes for k in range(K_exp)], dim=0).contiguous()
                centroids_cat = torch.cat(
                    [experts_data[k].vq_centroids for k in range(K_exp)], dim=0).contiguous()
                if not hasattr(self, '_vq4_cat_cache'):
                    self._vq4_cat_cache = {}
                self._vq4_cat_cache[cache_key] = (codes_cat, centroids_cat)
            else:
                codes_cat, centroids_cat = cached

            # Build x_grouped from per-expert rotations
            if all_same_pd and x_rot_shared is not None:
                x_grouped = x_rot_shared.expand(K_exp, -1).contiguous()
            else:
                # Compute per-expert x_rot then stack
                x_rots = []
                for k in range(K_exp):
                    p = experts_data[k]
                    pd_id = id(p.vq_perm_sigma)
                    x_rot_k = rot_cache.get(pd_id) if rot_cache else None
                    if x_rot_k is None:
                        x_permuted = x.half()[..., p.vq_perm_sigma]
                        x_rot_k = x_permuted @ p.vq_diagI
                    x_rots.append(x_rot_k)
                x_grouped = torch.cat(x_rots, dim=0)   # (K_exp, in_d)

            n_cb, K_cb, vdim = experts_data[0].vq_centroids.shape
            codes_per_row = experts_data[0].vq_codes.shape[1]
            codes_per_cb = codes_per_row // n_cb
            N_out = out_dims[0]
            y_flat = vq4_dequant_grouped_gemv(
                x_grouped, codes_cat, centroids_cat,
                K_exp, N_out, n_cb, codes_per_cb,
            )
            y_cat = y_flat.view(K_exp, N_out)
        else:
            # Per-expert vq4 kernel calls
            y_pieces = []
            for k in range(K_exp):
                p = experts_data[k]
                pd_id = id(p.vq_perm_sigma)
                x_rot = rot_cache.get(pd_id) if rot_cache else None
                if x_rot is None:
                    x_permuted = x.half()[..., p.vq_perm_sigma]
                    x_rot = x_permuted @ p.vq_diagI
                n_cb, K_cb, vdim = p.vq_centroids.shape
                codes_per_row = p.vq_codes.shape[1]
                codes_per_cb = codes_per_row // n_cb
                y_k = vq4_dequant_matmul(
                    x_rot, p.vq_codes, p.vq_centroids, n_cb, codes_per_cb)
                y_pieces.append(y_k)
            y_cat = torch.cat(y_pieces, dim=0)   # (K, out_d) if same, else concat

        # ---- Sync side stream and add LoRA ----
        torch.cuda.current_stream().wait_stream(side)

        if lora_cat is not None:
            # lora_cat: (1, K*out_d) → reshape to (K, out_d)
            lora_reshaped = lora_cat.view(K_exp, -1)
            y_cat = y_cat + lora_reshaped
        else:
            for k in range(K_exp):
                if lora_outs[k] is not None:
                    y_cat[k:k+1] = y_cat[k:k+1] + lora_outs[k]

        # Return list of (1, out_d) tensors
        return [y_cat[k:k+1] for k in range(K_exp)]

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
        try:
            from inference.kernels import (
                vq4_dequant_grouped_gemv, vq4_dequant_matmul, is_vq4_cuda_available,
            )
            _has_vq4 = is_vq4_cuda_available()
        except ImportError:
            _has_vq4 = False

        experts_data = []
        for ei in expert_indices:
            experts_data.append(getattr(self.experts[ei], "down_proj"))

        all_turbo = all(getattr(p, 'quant_type', None) == "turbo" and
                        (p._rotation_cache is not None or
                         (hasattr(p, '_rht_signs') and p._rht_signs is not None))
                        for p in experts_data)

        all_vq4 = all(getattr(p, 'quant_type', None) == "vq4"
                       and getattr(p, 'vq_codes', None) is not None
                       for p in experts_data) and _has_vq4

        if all_vq4 and len(h_list) > 0:
            return self._batched_down_forward_vq4(h_list, expert_indices, experts_data, xU_cache)

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

    def _batched_down_forward_vq4(self, h_list, expert_indices, experts_data, xU_cache):
        """VQ4 down_proj: grouped_gemv (main) ∥ LoRA (side)."""
        from inference.kernels import vq4_dequant_grouped_gemv, vq4_dequant_matmul

        K_exp = len(expert_indices)
        p0 = experts_data[0]

        # Rotate each h. Since each expert has its own sigma, compute per-expert
        # then stack. This still enables single grouped_gemv even with different sigmas.
        pd_ids = [id(p.vq_perm_sigma) for p in experts_data]
        all_same_pd = (len(set(pd_ids)) == 1)

        if all_same_pd:
            h_cat = torch.cat(h_list, dim=0).half()   # (K, inter_d)
            h_perm = h_cat[..., p0.vq_perm_sigma]
            h_rots = h_perm @ p0.vq_diagI              # (K, inter_d)
        else:
            h_rot_list = []
            for k in range(K_exp):
                p = experts_data[k]
                h_perm_k = h_list[k].half()[..., p.vq_perm_sigma]
                h_rot_k = h_perm_k @ p.vq_diagI
                h_rot_list.append(h_rot_k)
            h_rots = torch.cat(h_rot_list, dim=0)      # (K, inter_d)

        # LoRA on side stream
        cids = [getattr(p, 'cluster_id', None) for p in experts_data]
        all_same_cluster = (len(set(cids)) == 1 and cids[0] is not None)
        all_have_sv = all(p.SV is not None for p in experts_data)
        any_has_sa = any(experts_data[k].Sa is not None for k in range(K_exp))

        side = self._get_side_stream()
        lora_outs = [None] * K_exp

        with torch.cuda.stream(side):
            if all_have_sv and all_same_cluster and p0.U is not None and not any_has_sa:
                h_cat_lora = torch.cat(h_list, dim=0)     # (K, inter_d)
                a_stacked = h_cat_lora @ p0.U             # (K, rank)
                for k in range(K_exp):
                    lora_outs[k] = a_stacked[k:k+1] @ experts_data[k].SV.T
            elif all_have_sv:
                for k in range(K_exp):
                    proj = experts_data[k]
                    h_lora = h_list[k] * proj.Sa if proj.Sa is not None else h_list[k]
                    lora_outs[k] = (h_lora @ proj.U) @ proj.SV.T

        # Main stream: grouped kernel with concatenated codes+centroids.
        # Now works even when sigmas differ (h_rots already computed per expert).
        out_dims = [p.out_features for p in experts_data]
        same_shape = all(out_dims[k] == out_dims[0] for k in range(K_exp))

        if same_shape:
            cache_key = ('down_proj', tuple(expert_indices))
            cached = getattr(self, '_vq4_cat_cache', {}).get(cache_key)
            if cached is None:
                codes_cat = torch.cat(
                    [experts_data[k].vq_codes for k in range(K_exp)], dim=0).contiguous()
                centroids_cat = torch.cat(
                    [experts_data[k].vq_centroids for k in range(K_exp)], dim=0).contiguous()
                if not hasattr(self, '_vq4_cat_cache'):
                    self._vq4_cat_cache = {}
                self._vq4_cat_cache[cache_key] = (codes_cat, centroids_cat)
            else:
                codes_cat, centroids_cat = cached

            n_cb, K_cb, vdim = p0.vq_centroids.shape
            codes_per_row = p0.vq_codes.shape[1]
            codes_per_cb = codes_per_row // n_cb
            N_out = out_dims[0]
            y_flat = vq4_dequant_grouped_gemv(
                h_rots, codes_cat, centroids_cat,
                K_exp, N_out, n_cb, codes_per_cb,
            )
            y_cat = y_flat.view(K_exp, N_out)
            results = [y_cat[k:k+1] for k in range(K_exp)]
        else:
            results = []
            for k in range(K_exp):
                p = experts_data[k]
                h_permuted = h_list[k].half()[..., p.vq_perm_sigma]
                h_rot_k = h_permuted @ p.vq_diagI
                n_cb, K_cb, vdim = p.vq_centroids.shape
                codes_per_row = p.vq_codes.shape[1]
                codes_per_cb = codes_per_row // n_cb
                y_k = vq4_dequant_matmul(h_rot_k, p.vq_codes, p.vq_centroids,
                                          n_cb, codes_per_cb)
                results.append(y_k)

        torch.cuda.current_stream().wait_stream(side)
        for k in range(K_exp):
            if lora_outs[k] is not None:
                results[k] = results[k] + lora_outs[k]
        return results

    def _forward_graph_vq4(self, hidden_states, routing_weights,
                            selected_experts, hidden_dim, xU_cache, rot_cache):
        """Graph-mode forward for VQ4 experts. All E experts execute on
        the full input; routing_weights scatter selects which contribute.

        Fixed-shape → CUDA Graph capture works.
        """
        if not self._graph_cache_built:
            self._build_graph_cache_vq4()

        from inference.kernels import (
            vq4_dequant_grouped_gemv, lora_grouped_gemv, lora_u_grouped_gemv,
            silu_and_mul,
        )

        N = hidden_states.shape[0]
        E = self.num_experts

        # ---- Gate proj ----
        # Use pre-built sigma_flat (int64) + torch.index_select to replace the
        # gather+expand+.long() chain — cuts ~5 kernel launches per proj per layer.
        x_h = hidden_states.half()   # (N, hidden)
        in_d_g = self._vq4_gate_in_d
        if N == 1:
            x_gate_perm_flat = torch.index_select(
                x_h.view(-1), 0, self._vq4_sigma_flat_gate
            ).view(E, in_d_g)                                          # (E, in_d)
        else:
            x_gate_perm_flat = x_h.index_select(
                1, self._vq4_sigma_flat_gate
            ).view(N * E, in_d_g)                                      # (N*E, in_d)
        x_gate_rot = x_gate_perm_flat @ self._vq4_diagI_gate            # (N*E, in_d)

        # grouped kernel: x_grouped=(N*E, in_d), codes=(E*out_d, in_d/vdim), centroids=(E*n_cb, K, vdim)
        # But we have N tokens times E experts → need to call once per token, or reshape as (N*E) grouped
        # Simplification for N=1 (decode): treat as E-way grouped call directly.
        # For N>1, do grouped per-token then concat.
        out_d_g = self._vq4_gate_out_d
        if N == 1:
            y_gate_flat = vq4_dequant_grouped_gemv(
                x_gate_rot, self._vq4_codes_gate_all, self._vq4_centroids_gate_all,
                E, out_d_g, self._vq4_gate_ncb, self._vq4_gate_cpcb,
            )
            y_gate_all = y_gate_flat.view(E, out_d_g).unsqueeze(0)   # (1, E, out_d)
        else:
            # Loop over tokens to preserve grouped batching per token
            outs = []
            for t in range(N):
                y_t = vq4_dequant_grouped_gemv(
                    x_gate_rot[t*E:(t+1)*E], self._vq4_codes_gate_all,
                    self._vq4_centroids_gate_all, E, out_d_g,
                    self._vq4_gate_ncb, self._vq4_gate_cpcb,
                )
                outs.append(y_t.view(E, out_d_g))
            y_gate_all = torch.stack(outs, dim=0)   # (N, E, out_d)

        # LoRA U for gate+up — if ranks match, do ONE wider GEMM instead of two.
        rank_g = self._vq4_gate_rank
        rank_u = self._vq4_up_rank
        if N == 1 and getattr(self, '_vq4_fused_gate_up_U', False):
            a_cat = x_h @ self._vq4_U_sa_flat_gate_up   # (1, 2*E*rank)
            a_flat = a_cat[:, :E * rank_g]
            a_flat_u = a_cat[:, E * rank_g:]
        else:
            a_flat = x_h @ self._vq4_U_sa_flat_gate
            a_flat_u = x_h @ self._vq4_U_sa_flat_up
        a_gate = a_flat.view(N, E, rank_g)
        if N == 1:
            lora_gate = lora_grouped_gemv(
                a_gate.squeeze(0),                                      # (E, rank)
                self._vq4_SV_T_gate                                     # (E, rank, out_d)
            ).unsqueeze(0)                                              # (N=1, E, out_d)
        else:
            lora_gate = torch.bmm(
                a_gate.transpose(0, 1),
                self._vq4_SV_T_gate
            ).transpose(0, 1)
        y_gate_all = y_gate_all + lora_gate

        # ---- Up proj ---- (same input; reuse gate rotation if configs match)
        in_d_u = self._vq4_up_in_d
        if getattr(self, '_vq4_share_gate_up_rot', False):
            # P3: gate/up share sigma+diagI → reuse x_gate_rot for up_proj
            x_up_rot = x_gate_rot
        else:
            if N == 1:
                x_up_perm_flat = torch.index_select(
                    x_h.view(-1), 0, self._vq4_sigma_flat_up
                ).view(E, in_d_u)
            else:
                x_up_perm_flat = x_h.index_select(
                    1, self._vq4_sigma_flat_up
                ).view(N * E, in_d_u)
            x_up_rot = x_up_perm_flat @ self._vq4_diagI_up

        out_d_u = self._vq4_up_out_d
        if N == 1:
            y_up_flat = vq4_dequant_grouped_gemv(
                x_up_rot, self._vq4_codes_up_all, self._vq4_centroids_up_all,
                E, out_d_u, self._vq4_up_ncb, self._vq4_up_cpcb,
            )
            y_up_all = y_up_flat.view(E, out_d_u).unsqueeze(0)
        else:
            outs = []
            for t in range(N):
                y_t = vq4_dequant_grouped_gemv(
                    x_up_rot[t*E:(t+1)*E], self._vq4_codes_up_all,
                    self._vq4_centroids_up_all, E, out_d_u,
                    self._vq4_up_ncb, self._vq4_up_cpcb,
                )
                outs.append(y_t.view(E, out_d_u))
            y_up_all = torch.stack(outs, dim=0)

        # a_flat_u was already computed above (fused or separate)
        a_up = a_flat_u.view(N, E, rank_u)
        if N == 1:
            lora_up = lora_grouped_gemv(
                a_up.squeeze(0),
                self._vq4_SV_T_up
            ).unsqueeze(0)
        else:
            lora_up = torch.bmm(
                a_up.transpose(0, 1),
                self._vq4_SV_T_up
            ).transpose(0, 1)
        y_up_all = y_up_all + lora_up

        # ---- Activation ---- (fused SiLU * mul)
        h_all = silu_and_mul(y_gate_all, y_up_all)                # (N, E, inter_d)

        # ---- Down proj ---- (different input per expert)
        in_d_d = self._vq4_down_in_d
        # sigma_flat_down includes per-expert offset (e*in_d) so a single flat
        # index_select on h_all.view(N, E*in_d) yields (N, E*in_d) permuted.
        h_all_flat = h_all.contiguous().view(N, E * in_d_d)
        h_perm_flat = h_all_flat.index_select(
            1, self._vq4_sigma_flat_down
        ).view(N * E, in_d_d)                                          # (N*E, in_d)
        h_rot = h_perm_flat @ self._vq4_diagI_down

        out_d_d = self._vq4_down_out_d
        if N == 1:
            y_down_flat = vq4_dequant_grouped_gemv(
                h_rot, self._vq4_codes_down_all, self._vq4_centroids_down_all,
                E, out_d_d, self._vq4_down_ncb, self._vq4_down_cpcb,
            )
            y_down_all = y_down_flat.view(E, out_d_d).unsqueeze(0)
        else:
            outs = []
            for t in range(N):
                y_t = vq4_dequant_grouped_gemv(
                    h_rot[t*E:(t+1)*E], self._vq4_codes_down_all,
                    self._vq4_centroids_down_all, E, out_d_d,
                    self._vq4_down_ncb, self._vq4_down_cpcb,
                )
                outs.append(y_t.view(E, out_d_d))
            y_down_all = torch.stack(outs, dim=0)

        rank_d = self._vq4_down_rank
        if N == 1:
            # K-parallel kernel with pre-transposed weight (E, rank, in_d)
            a_down_flat = lora_u_grouped_gemv(
                h_all.squeeze(0),                                       # (E, inter_d)
                self._vq4_U_sa_T_down                                   # (E, rank, inter_d)
            )                                                           # (E, rank)
            lora_down = lora_grouped_gemv(
                a_down_flat,                                            # (E, rank)
                self._vq4_SV_T_down                                     # (E, rank, out_d)
            ).unsqueeze(0)                                              # (N=1, E, out_d)
        else:
            a_down = torch.bmm(
                h_all.transpose(0, 1),                                 # (E, N, inter_d)
                self._vq4_U_sa_down                                    # (E, inter_d, rank)
            ).transpose(0, 1)                                          # (N, E, rank)
            lora_down = torch.bmm(
                a_down.transpose(0, 1),
                self._vq4_SV_T_down
            ).transpose(0, 1)
        y_down_all = y_down_all + lora_down

        # ---- Weighted accumulation ----
        # P2: gather selected experts (top_k, not E) and weighted-sum. Avoids
        # building a (N, E) full_weights + scatter + bmm-of-mostly-zeros.
        if N == 1:
            # selected_experts: (1, top_k) int64; routing_weights: (1, top_k) fp16
            sel = selected_experts[0]                          # (top_k,)
            y_sel = y_down_all[0].index_select(0, sel)         # (top_k, hidden)
            w = routing_weights[0].to(y_sel.dtype).unsqueeze(1)  # (top_k, 1)
            final = (w * y_sel).sum(0, keepdim=True)           # (1, hidden)
        else:
            full_weights = torch.zeros(
                N, E, dtype=routing_weights.dtype, device=hidden_states.device)
            full_weights.scatter_(1, selected_experts, routing_weights)
            final = torch.bmm(
                full_weights.to(y_down_all.dtype).unsqueeze(1),
                y_down_all
            ).squeeze(1)
        return final

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
        """Batched graph-mode forward: 1 grouped-GEMV for down_proj vs E individual calls.

        Gate and up run as 1 batched turbo call each (all E experts share the
        same input hidden_states).  Down now also uses a single grouped-GEMV
        kernel call (E experts each with their own input row and weight block).
        """
        # VQ4 experts: dispatch to vq4-specific graph forward.
        # Require ALL experts to be VQ4 — mixed VQ4 + fp16_passthrough (from
        # quant-skipped experts) would crash the batched vq4 path.
        all_vq4 = all(
            getattr(getattr(e, 'gate_proj', None), 'quant_type', None) == 'vq4'
            for e in self.experts
        )
        if all_vq4:
            return self._forward_graph_vq4(
                hidden_states, routing_weights, selected_experts,
                hidden_dim, xU_cache, rot_cache)

        if not self._graph_cache_built:
            self._build_graph_cache()

        from inference.kernels import turbo_dequant_matmul_fused

        N = hidden_states.shape[0]
        E = self.num_experts

        # 1. Routing: (N, top_k) → (N, E)
        full_weights = torch.zeros(
            N, E, dtype=routing_weights.dtype, device=hidden_states.device)
        full_weights.scatter_(1, selected_experts, routing_weights)

        # 2. Gate proj: side stream computes all per-cluster LoRA while main
        # stream runs turbo GEMV — both captured in CUDA Graph → real GPU
        # parallelism during replay (no Python overhead at runtime).
        x_rot_gate = rot_cache.get(self._gate_x_rot_key)
        p0_gate = self.experts[0].gate_proj

        ps  = self._prefetch_stream   # side stream (always valid)
        cur = torch.cuda.current_stream()

        # 2. Gate proj: side stream computes batched LoRA (1 GEMV + 1 BMM) while
        # main stream runs turbo GEMV — both captured in CUDA Graph.
        # Gate LoRA: (1,hidden) @ (hidden,K*rank) → gather → BMM with SV_all
        if self._U_cat_gate is not None:
            _rank_g = self._SV_all_gate.shape[2]
            _K_gate = self._U_cat_gate.shape[1] // _rank_g
            ps.wait_stream(cur)   # ensure hidden_states is ready on side stream

        # Submit turbo to main stream FIRST so both streams start simultaneously.
        # ps.wait_stream(cur) above records the main-stream fence before turbo,
        # so side stream only depends on routing (hidden_states), not on turbo.
        y_gate_all = turbo_dequant_matmul_fused(
            hidden_states, self._packed_gate_all, self._norms_gate_all,
            None, self._gate_centroids,
            p0_gate.turbo_bits, p0_gate.turbo_dim,
            lora_USV=None, precomputed_x_rot=x_rot_gate)            # (N, E*out_gate)

        if self._U_cat_gate is not None:
            with torch.cuda.stream(ps):
                _a_all_g = hidden_states @ self._U_cat_gate          # (N, K*rank)
                # Squeeze N=1, reshape to (K, rank), gather per expert → (E, rank)
                _a_exp_g = _a_all_g.squeeze(0).view(_K_gate, _rank_g)[
                    self._cluster_idx_gate]
                _lora_g = torch.bmm(
                    _a_exp_g.unsqueeze(1),                           # (E, 1, rank)
                    self._SV_all_gate.permute(0, 2, 1)               # (E, rank, out_d)
                ).squeeze(1)                                          # (E, out_d)
                self._ev_lora_gate.record()
            cur.wait_event(self._ev_lora_gate)
            y_gate_all = y_gate_all + _lora_g.view(N, -1)

        # 3. Up proj: same pattern.
        x_rot_up = rot_cache.get(self._up_x_rot_key)
        p0_up = self.experts[0].up_proj

        if self._U_cat_up is not None:
            _rank_u = self._SV_all_up.shape[2]
            _K_up = self._U_cat_up.shape[1] // _rank_u
            with torch.cuda.stream(ps):
                _a_all_u = hidden_states @ self._U_cat_up            # (N, K*rank)
                _a_exp_u = _a_all_u.squeeze(0).view(_K_up, _rank_u)[
                    self._cluster_idx_up]                            # (E, rank)
                _lora_u = torch.bmm(
                    _a_exp_u.unsqueeze(1),                           # (E, 1, rank)
                    self._SV_all_up.permute(0, 2, 1)                 # (E, rank, out_d)
                ).squeeze(1)                                          # (E, out_d)
                self._ev_lora_up.record()

        y_up_all = turbo_dequant_matmul_fused(
            hidden_states, self._packed_up_all, self._norms_up_all,
            None, self._up_centroids,
            p0_up.turbo_bits, p0_up.turbo_dim,
            lora_USV=None, precomputed_x_rot=x_rot_up)               # (N, E*out_up)

        if self._U_cat_up is not None:
            cur.wait_event(self._ev_lora_up)
            y_up_all = y_up_all + _lora_u.view(N, -1)

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

        # 6. Down turbo: 1 grouped-GEMV call for all E experts.
        # Simultaneously, side stream computes per-cluster down LoRA using h_all
        # (pre-rotation activations); h_all is ready on main stream at this point.
        # N=1 assumed (CUDA Graph is captured for fixed decode batch size).
        from inference.kernels import turbo_dequant_grouped_gemv_fused
        hidden_dim = hidden_states.shape[-1]
        _down_out_d = self._down_out_d

        if self._U_per_expert_down is not None:
            ps.wait_stream(cur)   # h_all ready on main stream

        # Submit grouped GEMV to main stream FIRST (same parallelism trick as gate).
        y_down_cat = turbo_dequant_grouped_gemv_fused(
            h_rot_all,                 # (N*E, K) fp16, N=1 in decode
            self._packed_down_all,     # (E*out_d, K/4) uint8
            self._norms_down_all,      # (E*out_d,) fp32
            self._down_centroids,      # (4,) fp32
            self._down_out_d)          # out_features per expert
        # y_down_cat: (E*out_d,) — reshape to (E, N, hidden_dim)
        out_all = y_down_cat.view(E, N, hidden_dim)

        # 7. Down LoRA on side stream (parallel to grouped GEMV above),
        # then sync and add.
        # h_all: (E, inter_d) for N=1 decode.
        # BMM1: (E, 1, inter_d) @ (E, inter_d, rank) → (E, 1, rank)
        # BMM2: (E, 1, rank)   @ (E, rank, hidden_d) → (E, 1, hidden_d)
        if self._U_per_expert_down is not None:
            with torch.cuda.stream(ps):
                _a_d = torch.bmm(
                    h_all.unsqueeze(1),                              # (E, 1, inter_d)
                    self._U_per_expert_down                          # (E, inter_d, rank)
                )                                                    # (E, 1, rank)
                _lora_d = torch.bmm(
                    _a_d,
                    self._SV_all_down.permute(0, 2, 1)               # (E, rank, hidden_d)
                ).squeeze(1)                                          # (E, hidden_d)
                self._ev_lora_down.record()
            cur.wait_event(self._ev_lora_down)
            out_all = out_all + _lora_d.unsqueeze(1)                 # (E, 1, hidden_dim)

        # 8. Routing accumulation: 1 bmm replaces E scatter-adds
        out_all_t = out_all.permute(1, 0, 2)                        # (N, E, hidden_dim)
        final = torch.bmm(
            full_weights.unsqueeze(1), out_all_t).squeeze(1)        # (N, hidden_dim)
        return final
