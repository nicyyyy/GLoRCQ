"""
CUDA Stream Timeline Profiler — Non-Graph (decode) path.

Profiles _forward_decode → _batched_proj_forward (gate/up) + _batched_down_forward,
showing the dual-stream parallelism between main stream (turbo) and side stream (LoRA).

Three charts:
  timeline_gate.png  — gate projection: main (turbo 4 experts) ∥ side (LoRA)
  timeline_up.png    — up   projection: main (turbo 4 experts) ∥ side (LoRA)
  timeline_down.png  — down projection: main (rotation+turbo loop) ∥ side (LoRA)

Usage:
    CUDA_VISIBLE_DEVICES=3 python exp/img/profile_streams.py \\
        --model_path /path/to/realquant_model
"""

import argparse, os, sys
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)


# ─────────────────────────────────────────────────────────────
# CUDA event helpers
# ─────────────────────────────────────────────────────────────

def _ev():
    return torch.cuda.Event(enable_timing=True)

def _rec(e, stream=None):
    if stream is None:
        e.record(torch.cuda.current_stream())
    else:
        with torch.cuda.stream(stream):
            e.record()


# ─────────────────────────────────────────────────────────────
# Build inputs / caches
# ─────────────────────────────────────────────────────────────

def make_inputs(block, hidden_dim, top_k, device):
    """Construct realistic decode inputs for a single token."""
    x = torch.randn(1, hidden_dim, dtype=torch.float16, device=device)

    # Pick top_k distinct experts at random
    expert_indices = torch.randperm(block.num_experts)[:top_k].tolist()

    # Pre-compute rotated input (mirrors _precompute_rotations)
    rot_cache = {}
    p0_gate = block.experts[expert_indices[0]].gate_proj
    key_gate = (p0_gate.turbo_dim, p0_gate.turbo_bits, p0_gate.turbo_seed)

    if hasattr(p0_gate, '_rht_signs') and p0_gate._rht_signs is not None:
        from hadamard_rotation import rht_forward
        rot_cache[key_gate] = rht_forward(x.float(), p0_gate._rht_signs).half()
    else:
        Pi = p0_gate._rotation_cache.get_pi(*key_gate)
        rot_cache[key_gate] = (x.float() @ Pi.float().T).half()

    p0_up = block.experts[expert_indices[0]].up_proj
    key_up = (p0_up.turbo_dim, p0_up.turbo_bits, p0_up.turbo_seed)
    if key_up != key_gate:
        if hasattr(p0_up, '_rht_signs') and p0_up._rht_signs is not None:
            from hadamard_rotation import rht_forward
            rot_cache[key_up] = rht_forward(x.float(), p0_up._rht_signs).half()
        else:
            Pi = p0_up._rotation_cache.get_pi(*key_up)
            rot_cache[key_up] = (x.float() @ Pi.float().T).half()
    else:
        rot_cache[key_up] = rot_cache[key_gate]

    return x, expert_indices, rot_cache


# ─────────────────────────────────────────────────────────────
# Instrumented _batched_proj_forward  (gate or up)
# ─────────────────────────────────────────────────────────────

def profile_proj(block, x, expert_indices, proj_name, rot_cache):
    """
    Replicate _batched_proj_forward with CUDA event instrumentation.

    GPU timeline (ideal):
      t=0  ──── side: xU GEMV starts         ───── main: turbo starts
      t=T1 ──── side: SV.T matmuls start
      t=T2 ──── side: LoRA done              (ev_side_e)
      t=T3 ──── main: turbo done             (ev_main_e)
      t=max(T2,T3) ── wait_stream(side) ── add LoRA
    """
    from inference.kernels import turbo_dequant_matmul_fused

    experts_data = [getattr(block.experts[ei], proj_name) for ei in expert_indices]
    p0 = experts_data[0]
    K  = len(expert_indices)

    # Packed concat
    packed_cat = torch.cat([p.packed_indices for p in experts_data], dim=0)
    norms_cat  = torch.cat([p.norms          for p in experts_data], dim=0)
    centroids  = p0._rotation_cache.get_centroids(p0.turbo_dim, p0.turbo_bits, p0.turbo_seed)

    key   = (p0.turbo_dim, p0.turbo_bits, p0.turbo_seed)
    x_rot = rot_cache.get(key)

    cids          = [p.cluster_id for p in experts_data]
    all_same_cid  = (len(set(cids)) == 1 and cids[0] is not None)
    all_have_sv   = all(p.SV is not None for p in experts_data)
    has_lora      = all_have_sv and all(p.U is not None for p in experts_data)

    side = block._get_side_stream()
    cur  = torch.cuda.current_stream()

    # ── anchor ──
    ev_anchor = _ev(); _rec(ev_anchor)

    # ── Side stream: LoRA  (starts concurrently with main turbo below) ──
    ev_side_s = _ev(); ev_side_e = _ev()
    _rec(ev_side_s, stream=side)
    with torch.cuda.stream(side):
        lora_cat  = None
        lora_outs = [None] * K
        if has_lora:
            if all_same_cid:
                # one GEMM for all active experts sharing same cluster
                a        = x @ experts_data[0].U          # (1, rank)
                SV_cat   = torch.cat([p.SV for p in experts_data], dim=0)
                lora_cat = a @ SV_cat.T                    # (1, K*out_d)
            else:
                for k in range(K):
                    p   = experts_data[k]
                    a_k = x @ p.U
                    lora_outs[k] = a_k @ p.SV.T
    _rec(ev_side_e, stream=side)

    # ── Main stream: batched turbo (all K experts in one call) ──
    ev_main_s = _ev(); _rec(ev_main_s)
    y_cat = turbo_dequant_matmul_fused(
        x, packed_cat, norms_cat, None, centroids,
        p0.turbo_bits, p0.turbo_dim,
        lora_USV=None, precomputed_x_rot=x_rot)
    ev_main_e = _ev(); _rec(ev_main_e)

    # ── Sync + add LoRA ──
    ev_sync = _ev()
    cur.wait_stream(side)
    _rec(ev_sync)
    out_dims = [p.out_features for p in experts_data]
    if lora_cat is not None:
        y_cat = y_cat + lora_cat
    else:
        offset = 0
        for k, od in enumerate(out_dims):
            if lora_outs[k] is not None:
                y_cat[:, offset:offset+od] += lora_outs[k]
            offset += od

    torch.cuda.synchronize()

    return dict(
        _anchor  = (ev_anchor, ev_anchor),
        main     = (ev_main_s, ev_main_e),
        side     = (ev_side_s, ev_side_e),
        sync     = (ev_sync,   ev_sync),
        has_lora = has_lora,
        all_same = all_same_cid,
        K        = K,
    ), [y_cat[:, sum(out_dims[:k]):sum(out_dims[:k+1])] for k in range(K)]


# ─────────────────────────────────────────────────────────────
# Instrumented _batched_down_forward
# ─────────────────────────────────────────────────────────────

def profile_down(block, h_list, expert_indices):
    """
    Replicate _batched_down_forward with CUDA event instrumentation.

    Side stream: batch LoRA  (h_cat @ U  +  per-expert a @ SV.T)
    Main stream: per-expert  [RHT rotation]  +  [turbo]  in a loop

    Each expert's rotation and turbo are timed individually so the
    Gantt chart shows the serial loop on main stream.
    """
    from inference.kernels import turbo_dequant_matmul_fused

    experts_data = [block.experts[ei].down_proj for ei in expert_indices]
    p0 = experts_data[0]
    K  = len(expert_indices)

    centroids  = p0._rotation_cache.get_centroids(p0.turbo_dim, p0.turbo_bits, p0.turbo_seed)
    all_rht    = all(hasattr(p, '_rht_signs') and p._rht_signs is not None for p in experts_data)

    down_cids     = [getattr(p, 'cluster_id', None) for p in experts_data]
    all_same_cid  = (len(set(down_cids)) == 1 and down_cids[0] is not None)
    all_have_sv   = all(p.SV is not None for p in experts_data)
    has_lora      = all_have_sv and all(p.U is not None for p in experts_data)

    side = block._get_side_stream()
    cur  = torch.cuda.current_stream()

    # ── Pre-batch h rotation (all experts share same rotation) ──
    ev_rot_s = _ev(); _rec(ev_rot_s)
    if all_rht:
        from hadamard_rotation import rht_forward
        rht_signs = experts_data[0]._rht_signs
        h_stacked = torch.cat(h_list, dim=0).float()
        h_rots    = rht_forward(h_stacked, rht_signs).half()
    else:
        Pi = p0._rotation_cache.get_pi(p0.turbo_dim, p0.turbo_bits, p0.turbo_seed)
        h_rots = torch.cat([(h.float() @ Pi.float().T).half() for h in h_list], dim=0)
    ev_rot_e = _ev(); _rec(ev_rot_e)

    # ── anchor (after rotation, before parallel section) ──
    ev_anchor = _ev(); _rec(ev_anchor)

    # ── Side stream: batch LoRA ──
    ev_side_s = _ev(); ev_side_e = _ev()
    _rec(ev_side_s, stream=side)
    with torch.cuda.stream(side):
        lora_outs = [None] * K
        if has_lora:
            if all_same_cid and experts_data[0].U is not None:
                h_cat      = torch.cat(h_list, dim=0)
                a_stacked  = h_cat @ experts_data[0].U     # (K, rank)
                for k in range(K):
                    lora_outs[k] = a_stacked[k:k+1] @ experts_data[k].SV.T
            else:
                for k in range(K):
                    p = experts_data[k]
                    if p.U is not None and p.SV is not None:
                        lora_outs[k] = (h_list[k] @ p.U) @ p.SV.T
    _rec(ev_side_e, stream=side)

    # ── Main stream: per-expert turbo loop ──
    ev_turbo_s = [_ev() for _ in range(K)]
    ev_turbo_e = [_ev() for _ in range(K)]
    results = []
    for k in range(K):
        _rec(ev_turbo_s[k])
        p     = experts_data[k]
        h_rot = h_rots[k:k+1]
        y_q   = turbo_dequant_matmul_fused(
            h_list[k], p.packed_indices, p.norms, None, centroids,
            p.turbo_bits, p.turbo_dim,
            lora_USV=None, precomputed_x_rot=h_rot)
        _rec(ev_turbo_e[k])
        results.append(y_q)

    # ── Sync + add LoRA ──
    ev_sync = _ev()
    cur.wait_stream(side)
    _rec(ev_sync)
    for k in range(K):
        if lora_outs[k] is not None:
            results[k] = results[k] + lora_outs[k]

    torch.cuda.synchronize()

    return dict(
        _anchor  = (ev_anchor, ev_anchor),
        rot      = (ev_rot_s,  ev_rot_e),
        side     = (ev_side_s, ev_side_e),
        sync     = (ev_sync,   ev_sync),
        turbo    = list(zip(ev_turbo_s, ev_turbo_e)),   # per-expert
        has_lora = has_lora,
        all_same = all_same_cid,
        K        = K,
    ), results


# ─────────────────────────────────────────────────────────────
# Timing helpers
# ─────────────────────────────────────────────────────────────

def to_us(raw):
    """Convert raw event dict to µs relative to _anchor[0]."""
    ref = raw["_anchor"][0]
    out = {}
    for k, v in raw.items():
        if k.startswith("_") or k in ("has_lora", "all_same", "K"):
            continue
        if k == "turbo":
            # list of (start_ev, end_ev)
            out["turbo"] = []
            for (es, ee) in v:
                try:
                    out["turbo"].append((ref.elapsed_time(es)*1e3,
                                         ref.elapsed_time(ee)*1e3))
                except RuntimeError:
                    out["turbo"].append((0.0, 0.0))
        else:
            es, ee = v
            try:
                out[k] = (ref.elapsed_time(es)*1e3, ref.elapsed_time(ee)*1e3)
            except RuntimeError:
                out[k] = (0.0, 0.0)
    out["has_lora"] = raw["has_lora"]
    out["all_same"] = raw["all_same"]
    out["K"]        = raw["K"]
    return out


def avg_us(all_us, key):
    """Average a list of timing dicts for a single key → (mean_t0, mean_t1)."""
    t0s = [u[key][0] for u in all_us]
    t1s = [u[key][1] for u in all_us]
    return float(np.mean(t0s)), float(np.mean(t1s))


def avg_turbo(all_us):
    """Average per-expert turbo timings → list of (mean_t0, mean_t1)."""
    K = all_us[0]["K"]
    result = []
    for k in range(K):
        t0s = [u["turbo"][k][0] for u in all_us]
        t1s = [u["turbo"][k][1] for u in all_us]
        result.append((float(np.mean(t0s)), float(np.mean(t1s))))
    return result


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

MAIN_Y  = 1.2
SIDE_Y  = 0.0
BAR_H   = 0.45

CLRS = dict(
    turbo    = "#2166ac",
    lora_xU  = "#74add1",
    lora_sv  = "#abd9e9",
    rot      = "#7fc97f",
    sync     = "#dddddd",
)
SIDE_CLR = "#f4a582"


def _bar(ax, t0, t1, y, color, label, min_w=0.0, fontsize=6.5):
    w = max(t1 - t0, min_w)
    if w <= 0:
        return
    ax.barh(y, w, left=t0, height=BAR_H,
            color=color, edgecolor="white", linewidth=0.5, align="center")
    if w > 2.5:
        ax.text(t0 + w/2, y, f"{label}\n{t1-t0:.1f}µs",
                ha="center", va="center", fontsize=fontsize,
                color="white", fontweight="bold", clip_on=True)


def _sync_line(ax, x, label=""):
    ax.axvline(x, color="#888888", linestyle="--", linewidth=0.8, alpha=0.6)
    if label:
        ax.text(x+0.3, 0.65, label, fontsize=6, color="#888888")


def _make_legend(has_lora, is_down=False):
    patches = [
        mpatches.Patch(color=CLRS["turbo"],   label="Turbo GEMV  (2-bit dequant+matmul)"),
    ]
    if is_down:
        patches.insert(0, mpatches.Patch(color=CLRS["rot"], label="RHT Rotation  (batched, all experts)"))
    if has_lora:
        patches += [
            mpatches.Patch(color=SIDE_CLR,            label="Side stream: x@U  (LoRA input proj)"),
            mpatches.Patch(color=CLRS["lora_sv"],      label="Side stream: a@SV.T  (LoRA output proj)"),
        ]
    else:
        patches.append(
            mpatches.Patch(color=SIDE_CLR, alpha=0.3,
                           label="Side stream: idle  (experts span multiple clusters)"))
    return patches


def plot_proj_section(avg_main, avg_side, has_lora, title, save_path,
                      rot_bar=None, turbo_per_expert=None):
    """
    Plot one projection section with main (top) and side (bottom) streams.

    avg_main / avg_side: (t0_µs, t1_µs) of the dominant block on each stream.
    rot_bar: optional (t0, t1) for a rotation bar shown before main turbo.
    turbo_per_expert: optional list of (t0,t1) for per-expert turbo bars (down proj).
    """
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.set_ylim(-0.6, 2.0)

    # ── Main stream ──
    if rot_bar is not None:
        _bar(ax, *rot_bar, MAIN_Y, CLRS["rot"], "RHT Rotation\n(all experts)")

    if turbo_per_expert is not None:
        for k, (t0, t1) in enumerate(turbo_per_expert):
            lbl = f"Turbo expert {k}\n{t1-t0:.1f}µs"
            _bar(ax, t0, t1, MAIN_Y, CLRS["turbo"], lbl)
    else:
        _bar(ax, *avg_main, MAIN_Y, CLRS["turbo"], "Turbo GEMV\n(all active experts)")

    # ── Side stream ──
    if has_lora:
        _bar(ax, *avg_side, SIDE_Y, SIDE_CLR, "LoRA\n(x@U + a@SV.T)")
    else:
        ax.text(3, SIDE_Y, "Side stream: idle  (experts span multiple clusters — LoRA computed serially)",
                fontsize=8, color="#999999", va="center", style="italic")

    # sync line
    sync_x = max(avg_main[1], avg_side[1] if has_lora else 0)
    _sync_line(ax, sync_x, "wait_stream")

    ax.set_yticks([SIDE_Y, MAIN_Y])
    ax.set_yticklabels(["Side Stream\n(LoRA)", "Main Stream\n(Turbo)"], fontsize=9)
    ax.set_xlabel("Time relative to projection start (µs)", fontsize=9)
    ax.set_title(title + "  (mean of 20 runs)", fontsize=11)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    ax.legend(handles=_make_legend(has_lora, rot_bar is not None),
              loc="upper right", fontsize=7, framealpha=0.85)

    # total annotation
    t_total = sync_x
    ax.text(0.99, 0.04, f"Section total: {t_total:.1f} µs",
            transform=ax.transAxes, ha="right", fontsize=8, color="#333333")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


def print_summary(gate_us_list, up_us_list, down_us_list):
    g = dict(main=avg_us(gate_us_list, "main"), side=avg_us(gate_us_list, "side"))
    u = dict(main=avg_us(up_us_list,   "main"), side=avg_us(up_us_list,   "side"))
    d_rot    = avg_us(down_us_list, "rot")
    d_side   = avg_us(down_us_list, "side")
    d_turbo  = avg_turbo(down_us_list)

    print(f"\n{'─'*65}")
    print(f"{'Section':<18} {'Stream':<8} {'t_start':>10} {'t_end':>10} {'dur(µs)':>10}")
    print(f"{'─'*65}")
    for sec, label, t0, t1 in [
        ("Gate",  "main", *g["main"]), ("Gate",  "side", *g["side"]),
        ("Up",    "main", *u["main"]), ("Up",    "side", *u["side"]),
        ("Down",  "rot",  *d_rot),     ("Down",  "side", *d_side),
    ]:
        print(f"{sec:<18} {label:<8} {t0:>10.2f} {t1:>10.2f} {t1-t0:>10.2f}")
    for k, (t0, t1) in enumerate(d_turbo):
        print(f"{'Down':<18} {f'turbo[{k}]':<8} {t0:>10.2f} {t1:>10.2f} {t1-t0:>10.2f}")
    print(f"{'─'*65}")
    g_total = max(g["main"][1], g["side"][1])
    u_total = max(u["main"][1], u["side"][1])
    d_total = max(max(t1 for _,t1 in d_turbo), d_side[1])
    print(f"{'Gate total':30} {g_total:>10.2f} µs")
    print(f"{'Up   total':30} {u_total:>10.2f} µs")
    print(f"{'Down total (after rot)':30} {d_total:>10.2f} µs")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--layer_idx", type=int, default=1)
    parser.add_argument("--warmup",    type=int, default=10)
    parser.add_argument("--runs",      type=int, default=20)
    parser.add_argument("--device",    default="cuda")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Load model ──
    from utils.model_loader import load_model_and_tokenizer
    print("Loading real-quant model ...")
    model, _ = load_model_and_tokenizer(
        args.model_path, device=args.device, real_quant=True)
    model.eval()

    # ── Find target MoE block ──
    from inference.moe_block import GraphCompatibleMoeBlock
    moe_layers = [(i, layer.mlp)
                  for i, layer in enumerate(model.model.layers)
                  if isinstance(getattr(layer, "mlp", None), GraphCompatibleMoeBlock)]
    if not moe_layers:
        raise RuntimeError("No GraphCompatibleMoeBlock found")

    layer_i, block = next(((i,b) for i,b in moe_layers if i==args.layer_idx),
                           moe_layers[0])
    print(f"Profiling layer {layer_i}  (E={block.num_experts} experts)")

    hidden_dim = model.config.hidden_size
    top_k      = getattr(model.config, "num_experts_per_tok",
                         getattr(model.config, "top_k", 4))
    device     = next(model.parameters()).device
    print(f"  hidden_dim={hidden_dim}, top_k={top_k} active experts per step")

    # ── Warmup ──
    print(f"\nWarming up ({args.warmup} steps) ...")
    for _ in range(args.warmup):
        x, eidx, rot = make_inputs(block, hidden_dim, top_k, device)
        raw_g, gate_outs = profile_proj(block, x, eidx, "gate_proj", rot)
        raw_u, up_outs   = profile_proj(block, x, eidx, "up_proj",   rot)
        inter = block.experts[0].down_proj.in_features
        h_list = [block.experts[eidx[k]].act_fn(gate_outs[k]) * up_outs[k]
                  for k in range(top_k)]
        raw_d, _         = profile_down(block, h_list, eidx)

    # ── Collect ──
    print(f"Collecting timing ({args.runs} steps) ...")
    gate_us_list, up_us_list, down_us_list = [], [], []
    for _ in range(args.runs):
        x, eidx, rot = make_inputs(block, hidden_dim, top_k, device)
        raw_g, gate_outs = profile_proj(block, x, eidx, "gate_proj", rot)
        raw_u, up_outs   = profile_proj(block, x, eidx, "up_proj",   rot)
        h_list = [block.experts[eidx[k]].act_fn(gate_outs[k]) * up_outs[k]
                  for k in range(top_k)]
        raw_d, _         = profile_down(block, h_list, eidx)
        gate_us_list.append(to_us(raw_g))
        up_us_list.append(to_us(raw_u))
        down_us_list.append(to_us(raw_d))

    print_summary(gate_us_list, up_us_list, down_us_list)

    # ── Plot ──
    g_main  = avg_us(gate_us_list, "main")
    g_side  = avg_us(gate_us_list, "side")
    u_main  = avg_us(up_us_list,   "main")
    u_side  = avg_us(up_us_list,   "side")
    d_rot   = avg_us(down_us_list, "rot")
    d_side  = avg_us(down_us_list, "side")
    d_turbo = avg_turbo(down_us_list)
    has_g   = gate_us_list[0]["has_lora"]
    has_u   = up_us_list[0]["has_lora"]
    has_d   = down_us_list[0]["has_lora"]

    # Shift rot bar to start at 0 for down chart
    rot_shift = d_rot[0]
    d_rot_s   = (0.0, d_rot[1] - rot_shift)
    d_side_s  = (d_side[0]-rot_shift, d_side[1]-rot_shift)
    d_turbo_s = [(t0-rot_shift, t1-rot_shift) for t0,t1 in d_turbo]

    plot_proj_section(
        g_main, g_side, has_g,
        f"Gate Projection  (top-{top_k} active experts)",
        os.path.join(out_dir, "timeline_gate.png"))

    plot_proj_section(
        u_main, u_side, has_u,
        f"Up Projection  (top-{top_k} active experts)",
        os.path.join(out_dir, "timeline_up.png"))

    plot_proj_section(
        (d_turbo_s[0][0], d_turbo_s[-1][1]),  # full turbo span
        d_side_s, has_d,
        f"Down Projection  (top-{top_k} active experts)",
        os.path.join(out_dir, "timeline_down.png"),
        rot_bar=d_rot_s,
        turbo_per_expert=d_turbo_s)

    print("\nDone.")


if __name__ == "__main__":
    main()
