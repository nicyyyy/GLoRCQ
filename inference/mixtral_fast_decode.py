"""
Mixtral-specific fast MoE dispatch module (Candidate B: pre-baked expert dispatch).

Drop-in replacement path for GraphCompatibleMoeBlock._forward_decode when the
underlying MoE block is a Mixtral block (num_experts=8, top_k=2, VQ4 quant with
uniform rank and no per-expert Sa scale). Under those preconditions this class
pre-stacks per-expert quantized tensors (packed indices / vq_codes / vq_norms /
perm_sigma / U / SV) into contiguous (E, ...) tensors at build time so that
per-decode-step dispatch is:

  - 3x  torch.index_select on the E-stacked tensors (one per proj), no
    per-step torch.cat, no Python for-loop over the top_k slots;
  - 3x  vq4_dequant_grouped_gemv with a static G=top_k=2 batch shape;
  - 3x  torch.bmm for LoRA (two BMMs for gate/up because they share the input,
    a small third BMM for down);
  - 1x  weighted sum via broadcasted mul + sum.

When the preconditions do not hold, MixtralFastDispatch.try_build returns None
and the caller keeps the original _forward_decode path unchanged.

This module never touches inference/kernels/, it only re-uses the already-
exported vq4_dequant_grouped_gemv API.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------
def _proj_has_vq4(proj):
    return (
        proj is not None
        and getattr(proj, "quant_type", None) == "vq4"
        and getattr(proj, "vq_codes", None) is not None
        and getattr(proj, "vq_centroids", None) is not None
        and getattr(proj, "vq_perm_sigma", None) is not None
        and getattr(proj, "vq_diagI", None) is not None
    )


def _preconditions_ok(moe_block):
    """Return True iff Mixtral fast-dispatch can safely apply to this block.

    Required:
      - num_experts == 8 (Mixtral)
      - top_k == 2 (Mixtral)
      - Every expert has gate_proj / up_proj / down_proj as GLoRCQLinear
      - Every proj is VQ4 with valid vq_codes+centroids+sigma+diagI
      - All experts share the same rank for each of gate/up/down (uniform)
      - No expert has Sa (per-input activation scale)
    """
    if not hasattr(moe_block, "num_experts") or not hasattr(moe_block, "top_k"):
        return False
    if moe_block.num_experts != 8 or moe_block.top_k != 2:
        return False
    experts = getattr(moe_block, "experts", None)
    if experts is None or len(experts) != 8:
        return False

    ranks = {"gate_proj": None, "up_proj": None, "down_proj": None}
    for e in experts:
        for name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(e, name, None)
            if not _proj_has_vq4(proj):
                return False
            if getattr(proj, "Sa", None) is not None:
                return False
            if getattr(proj, "U", None) is None or getattr(proj, "SV", None) is None:
                return False
            r = proj.U.shape[1]
            if ranks[name] is None:
                ranks[name] = r
            elif ranks[name] != r:
                return False
    return True


# ---------------------------------------------------------------------------
# Fast dispatch
# ---------------------------------------------------------------------------
class MixtralFastDispatch:
    """Pre-baked expert dispatch for Mixtral VQ4 decode (N=1, top_k=2).

    Built once per MoE block at load time (or on first forward). Holds
    contiguous (E, ...) tensors so per-step forward is a small number of
    kernel launches with no Python loop over top_k and no per-step cat.
    """

    def __init__(self, moe_block):
        self.num_experts = moe_block.num_experts       # 8
        self.top_k = moe_block.top_k                    # 2
        self.gate = moe_block.gate                      # router linear
        self.norm_topk_prob = getattr(moe_block, "norm_topk_prob", True)

        experts = moe_block.experts
        E = self.num_experts

        # ------- Per-proj stacked tensors -------
        self._proj_meta = {}
        for name in ("gate_proj", "up_proj", "down_proj"):
            projs = [getattr(experts[e], name) for e in range(E)]
            p0 = projs[0]

            packed_codes = torch.stack(
                [p.vq_codes for p in projs], dim=0
            ).contiguous()                                        # (E, out_d, in_d/vdim) uint8
            packed_centroids = torch.stack(
                [p.vq_centroids for p in projs], dim=0
            ).contiguous()                                        # (E, n_cb, K_cb, vdim)
            perm_sigma = torch.stack(
                [p.vq_perm_sigma for p in projs], dim=0
            ).contiguous()                                        # (E, in_d) long
            U_stack = torch.stack(
                [p.U for p in projs], dim=0
            ).contiguous()                                        # (E, in_d, rank)
            SV_stack = torch.stack(
                [p.SV for p in projs], dim=0
            ).contiguous()                                        # (E, out_d, rank)
            # SV.T pre-transposed for bmm: (E, rank, out_d)
            SV_T_stack = SV_stack.transpose(1, 2).contiguous()

            # diagI is shared by construction (same in_d/rotate/partial across
            # experts of the same proj type) — grab from expert 0.
            diagI = p0.vq_diagI                                    # (in_d, in_d) fp16

            n_cb, K_cb, vdim = p0.vq_centroids.shape
            codes_per_row = p0.vq_codes.shape[1]
            codes_per_cb = codes_per_row // n_cb
            in_d = p0.in_features
            out_d = p0.out_features
            rank = p0.U.shape[1]

            self._proj_meta[name] = {
                "packed_codes": packed_codes,
                "packed_centroids": packed_centroids,
                "perm_sigma": perm_sigma,
                "U": U_stack,
                "SV": SV_stack,
                "SV_T": SV_T_stack,
                "diagI": diagI,
                "in_d": in_d,
                "out_d": out_d,
                "n_cb": n_cb,
                "K_cb": K_cb,
                "vdim": vdim,
                "codes_per_cb": codes_per_cb,
                "rank": rank,
                "act_fn": experts[0].act_fn if name == "gate_proj" else None,
            }

        # SiLU activation from expert 0 (all experts share act_fn under Mixtral).
        self.act_fn = experts[0].act_fn

        # Deferred import of the VQ4 grouped kernel — mirrors moe_block.py.
        from inference.kernels import (
            vq4_dequant_grouped_gemv, is_vq4_cuda_available,
        )
        if not is_vq4_cuda_available():
            raise RuntimeError(
                "MixtralFastDispatch requires the VQ4 CUDA kernel to be available."
            )
        self._vq4_grouped = vq4_dequant_grouped_gemv

    # ---- factory: returns None if preconditions fail ----
    @classmethod
    def try_build(cls, moe_block):
        if not _preconditions_ok(moe_block):
            return None
        try:
            return cls(moe_block)
        except Exception:
            return None

    # ---- proj helper: 2-way grouped VQ4 GEMV over experts (e0, e1) ----
    def _proj_forward(self, name, x_row, sel_idx):
        """Batched vq4 forward for two selected experts.

        Args:
            name: 'gate_proj' | 'up_proj' | 'down_proj'
            x_row: (1, in_d) fp16 — input for gate/up (shared) or per-slot h
                   stacked as (top_k, in_d) for down.
            sel_idx: (top_k,) long — expert index per slot.

        Returns:
            y: (top_k, out_d) fp16 — main VQ4 output + LoRA (SV branch only;
               shared U branch is added by the caller via one x@U per input).
        """
        meta = self._proj_meta[name]
        G = self.top_k                                      # 2

        # Gather stacked per-expert tensors for the two selected slots.
        codes_sel = meta["packed_codes"].index_select(0, sel_idx)          # (2, out_d, in_d/vdim)
        cents_sel = meta["packed_centroids"].index_select(0, sel_idx)      # (2, n_cb, K, vdim)
        sigma_sel = meta["perm_sigma"].index_select(0, sel_idx)            # (2, in_d)

        # Rotation: per-slot permute + shared diagI matmul.
        # x_row is either (1, in_d) (gate/up) or (G, in_d) (down; per-slot h).
        in_d = meta["in_d"]
        if x_row.shape[0] == 1:
            # gate/up: single input; broadcast permute across the 2 slots.
            x_h = x_row.half()                                             # (1, in_d)
            # (2, in_d) via gather; x_h.expand(2, -1) then gather.
            x_permuted = torch.gather(
                x_h.expand(G, -1), 1, sigma_sel
            )                                                              # (2, in_d)
        else:
            # down: (G, in_d) per-slot h, distinct per slot.
            x_h = x_row.half()                                             # (G, in_d)
            x_permuted = torch.gather(x_h, 1, sigma_sel)                   # (G, in_d)
        x_rot = x_permuted @ meta["diagI"]                                 # (G, in_d) fp16

        # Flatten stacked packed/centroids to (G*out_d, in_d/vdim) and
        # (G*n_cb, K, vdim) — the grouped kernel expects the same layout as
        # torch.cat over dim=0.
        out_d = meta["out_d"]
        n_cb = meta["n_cb"]
        vdim = meta["vdim"]
        codes_flat = codes_sel.reshape(G * out_d, -1).contiguous()
        cents_flat = cents_sel.reshape(G * n_cb, meta["K_cb"], vdim).contiguous()

        y_flat = self._vq4_grouped(
            x_rot, codes_flat, cents_flat,
            G, out_d, n_cb, meta["codes_per_cb"],
        )
        y_main = y_flat.view(G, out_d)                                     # (2, out_d)
        return y_main, sigma_sel

    # ---- per-step decode forward ----
    def _decode_step(self, hidden_states):
        """One decode step. Assumes N=1.

        Args:
            hidden_states: (1, hidden_dim) fp16/bf16.

        Returns:
            final: (1, hidden_dim), router_logits: (1, num_experts) — mirroring
            the ``forward`` return contract of GraphCompatibleMoeBlock.
        """
        # ---- Routing (mirrors GraphCompatibleMoeBlock.forward) ----
        router_logits = self.gate(hidden_states)                           # (1, E)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1)                          # (1, 2)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)          # (1, 2)

        sel_idx = selected_experts[0].to(torch.long)                       # (2,)
        G = self.top_k

        # ---- Gate / Up proj (shared input x_row = hidden_states) ----
        gate_meta = self._proj_meta["gate_proj"]
        up_meta   = self._proj_meta["up_proj"]

        # LoRA: x @ U for each of the 2 selected experts, batched.
        # (2, in_d) @ (2, in_d, rank) via bmm → (2, 1, rank)
        x_row = hidden_states                                               # (1, in_d)
        x_dt  = x_row.dtype

        U_gate_sel = gate_meta["U"].index_select(0, sel_idx)               # (2, in_d, r)
        U_up_sel   = up_meta["U"].index_select(0, sel_idx)                 # (2, in_d, r)
        SV_T_gate_sel = gate_meta["SV_T"].index_select(0, sel_idx)         # (2, r, out_d)
        SV_T_up_sel   = up_meta["SV_T"].index_select(0, sel_idx)           # (2, r, out_d)

        x_g = x_row.expand(G, -1).unsqueeze(1).to(U_gate_sel.dtype)        # (2, 1, in_d)
        a_gate = torch.bmm(x_g, U_gate_sel)                                # (2, 1, r)
        lora_gate = torch.bmm(a_gate, SV_T_gate_sel).squeeze(1)            # (2, out_d)
        a_up   = torch.bmm(x_g, U_up_sel)
        lora_up   = torch.bmm(a_up, SV_T_up_sel).squeeze(1)

        y_gate_main, _ = self._proj_forward("gate_proj", x_row, sel_idx)   # (2, out_d)
        y_up_main,   _ = self._proj_forward("up_proj",   x_row, sel_idx)   # (2, out_d)

        y_gate = y_gate_main + lora_gate.to(y_gate_main.dtype)
        y_up   = y_up_main   + lora_up.to(y_up_main.dtype)

        # ---- Activation ----
        h = self.act_fn(y_gate) * y_up                                     # (2, inter_d)

        # ---- Down proj (per-slot input h) ----
        down_meta = self._proj_meta["down_proj"]
        U_down_sel = down_meta["U"].index_select(0, sel_idx)               # (2, inter_d, r)
        SV_T_down_sel = down_meta["SV_T"].index_select(0, sel_idx)         # (2, r, hidden)

        h_bmm = h.unsqueeze(1).to(U_down_sel.dtype)                        # (2, 1, inter_d)
        a_down = torch.bmm(h_bmm, U_down_sel)                              # (2, 1, r)
        lora_down = torch.bmm(a_down, SV_T_down_sel).squeeze(1)            # (2, hidden)

        y_down_main, _ = self._proj_forward("down_proj", h, sel_idx)       # (2, hidden)
        y_down = y_down_main + lora_down.to(y_down_main.dtype)

        # ---- Weighted sum: (2, hidden) → (1, hidden) ----
        # routing_weights: (1, 2)  →  (2, 1)
        w = routing_weights[0].to(y_down.dtype).unsqueeze(1)               # (2, 1)
        final = (w * y_down).sum(dim=0, keepdim=True)                      # (1, hidden)

        return final, router_logits

    # ---- top-level forward matching GraphCompatibleMoeBlock.forward ----
    def forward(self, hidden_states):
        """Mirror GraphCompatibleMoeBlock.forward's contract.

        Only decode-shaped inputs (batch*seq == 1) are handled here; for any
        other shape MixtralFastMoeBlock delegates back to the fallback path.
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)
        assert flat.shape[0] == 1, "MixtralFastDispatch is decode-only (N=1)"
        final, router_logits = self._decode_step(flat)
        return final.reshape(batch_size, sequence_length, hidden_dim), router_logits


# ---------------------------------------------------------------------------
# Wrapper module: full drop-in replacement for GraphCompatibleMoeBlock on
# Mixtral, keeps the original block around as a fallback for prefill / any
# non-decode shape.
# ---------------------------------------------------------------------------
class MixtralFastMoeBlock(torch.nn.Module):
    """Drop-in replacement for GraphCompatibleMoeBlock on Mixtral.

    For hidden_states with N=1 (decode), delegates to MixtralFastDispatch's
    pre-baked path. For any other shape (prefill, batched decode), delegates
    to the wrapped original GraphCompatibleMoeBlock's forward — bit-exact
    fallback with zero behavioral change.
    """

    def __init__(self, original_block):
        super().__init__()
        # Keep the original block as a real submodule so its parameters and
        # buffers are preserved for prefill / non-decode. We do NOT copy any
        # tensors; MixtralFastDispatch takes references only.
        self._original = original_block
        # Router / experts references (share identity with _original).
        self.gate = original_block.gate
        self.experts = original_block.experts
        self.num_experts = original_block.num_experts
        self.top_k = original_block.top_k
        self.norm_topk_prob = getattr(original_block, "norm_topk_prob", True)

        # Build the fast dispatch. If preconditions fail we still act as an
        # identity wrapper (forward always goes through _original).
        self._fast = MixtralFastDispatch.try_build(original_block)

    @property
    def dispatch_mode(self):
        return "mixtral-fast" if self._fast is not None else "fallback"

    def forward(self, hidden_states):
        # Decode fast path — N=1 → use pre-baked dispatch.
        if (
            self._fast is not None
            and hidden_states.dim() == 3
            and hidden_states.shape[0] * hidden_states.shape[1] == 1
        ):
            return self._fast.forward(hidden_states)
        # Any other shape (prefill, batched decode): keep original path.
        return self._original.forward(hidden_states)


# ---------------------------------------------------------------------------
# Public helper: monkey-patch a loaded model in place.
# ---------------------------------------------------------------------------
def replace_mixtral_moe_blocks(model, verbose: bool = True):
    """Walk model.model.layers and replace each Mixtral MoE block with
    MixtralFastMoeBlock. Non-Mixtral layers / non-VQ4 blocks are left alone.

    Returns:
        (n_replaced, n_fallback): number of layers converted to the fast path
        and number of layers where preconditions failed and the wrapper falls
        back to the original block.
    """
    n_replaced = 0
    n_fallback = 0

    # Mixtral stores MoE under `block_sparse_moe`; Qwen uses `mlp`. We only
    # convert layers whose MoE container is a Mixtral-shaped GraphCompatible
    # block — i.e. num_experts == 8 and top_k == 2.
    try:
        layers = model.model.layers
    except AttributeError:
        if verbose:
            print("[mixtral_fast] model.model.layers not found; nothing to patch.")
        return 0, 0

    for layer_i, layer in enumerate(layers):
        for attr in ("block_sparse_moe", "mlp"):
            mod = getattr(layer, attr, None)
            if mod is None:
                continue
            # Only touch Mixtral-shaped blocks.
            if getattr(mod, "num_experts", None) != 8 or getattr(mod, "top_k", None) != 2:
                continue
            wrapper = MixtralFastMoeBlock(mod)
            setattr(layer, attr, wrapper)
            if wrapper.dispatch_mode == "mixtral-fast":
                n_replaced += 1
            else:
                n_fallback += 1

    if verbose:
        print(f"[mixtral_fast] fast-dispatch layers: {n_replaced}, "
              f"fallback layers: {n_fallback}")
    return n_replaced, n_fallback
