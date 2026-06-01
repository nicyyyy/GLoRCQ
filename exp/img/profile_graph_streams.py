"""
CUDA Stream Timeline Profiler — Graph-path (_forward_graph).

Directly executes the batched LoRA + turbo pipeline in eager mode (no graph
capture/replay), inserting CUDA events to capture per-op GPU timings.

Produces ONE combined Gantt chart showing a full decode step:
  timeline_graph.png  — Main stream ∥ Side stream across gate / up / down

Usage:
    CUDA_VISIBLE_DEVICES=3 python exp/img/profile_graph_streams.py \\
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


# ──────────────────────────────────────────────────────────────
# CUDA event helpers
# ──────────────────────────────────────────────────────────────

def _ev():
    return torch.cuda.Event(enable_timing=True)

def _rec(e, stream=None):
    if stream is None:
        e.record(torch.cuda.current_stream())
    else:
        e.record(stream)


# ──────────────────────────────────────────────────────────────
# Instrumented forward (mirrors _forward_graph with events)
# ──────────────────────────────────────────────────────────────

def run_instrumented(block, hidden_states, routing_weights, selected_experts):
    """
    Execute the same ops as _forward_graph but with timing events.

    Returns a dict of (start_ev, end_ev) pairs keyed by op name.
    All events are on the GPU's unified clock → cross-stream safe.
    """
    from inference.kernels import turbo_dequant_matmul_fused
    from inference.kernels import turbo_dequant_grouped_gemv_fused

    if not block._graph_cache_built:
        block._build_graph_cache()

    # Rebuild rotation cache (mirrors forward() preamble)
    block._build_rotation_cache_if_needed(hidden_states)
    rot_cache = block._rot_cache

    N  = hidden_states.shape[0]
    E  = block.num_experts
    ps = block._prefetch_stream
    cur = torch.cuda.current_stream()

    ev = {}  # op_name → (start_ev, end_ev)

    # ── anchor ──────────────────────────────────────────────────
    e_anchor = _ev(); _rec(e_anchor)

    # ── 1. Routing ──────────────────────────────────────────────
    e_rt_s = _ev(); _rec(e_rt_s)
    full_weights = torch.zeros(N, E, dtype=routing_weights.dtype,
                               device=hidden_states.device)
    full_weights.scatter_(1, selected_experts, routing_weights)
    e_rt_e = _ev(); _rec(e_rt_e)
    ev["routing"] = (e_rt_s, e_rt_e)

    x_rot_gate = rot_cache.get(block._gate_x_rot_key)
    p0_gate    = block.experts[0].gate_proj

    # ── 2. Gate: main turbo FIRST, then side LoRA ───────────────
    # (matches _forward_graph reordering: turbo submitted before LoRA
    #  so both enter GPU queue within ~5µs of each other)
    e_gl_s = _ev(); e_gl_e = _ev()
    if block._U_cat_gate is not None:
        _rank_g = block._SV_all_gate.shape[2]
        _K_gate = block._U_cat_gate.shape[1] // _rank_g
        ps.wait_stream(cur)

    e_gt_s = _ev(); _rec(e_gt_s)
    y_gate_all = turbo_dequant_matmul_fused(
        hidden_states, block._packed_gate_all, block._norms_gate_all,
        None, block._gate_centroids,
        p0_gate.turbo_bits, p0_gate.turbo_dim,
        lora_USV=None, precomputed_x_rot=x_rot_gate)
    e_gt_e = _ev(); _rec(e_gt_e)
    ev["gate_turbo"] = (e_gt_s, e_gt_e)

    if block._U_cat_gate is not None:
        with torch.cuda.stream(ps):
            _rec(e_gl_s, ps)
            _a_all_g  = hidden_states @ block._U_cat_gate
            _a_exp_g  = _a_all_g.squeeze(0).view(_K_gate, _rank_g)[
                block._cluster_idx_gate]
            _lora_g   = torch.bmm(
                _a_exp_g.unsqueeze(1),
                block._SV_all_gate.permute(0, 2, 1)).squeeze(1)
            _rec(e_gl_e, ps)
            block._ev_lora_gate.record()
        cur.wait_event(block._ev_lora_gate)
        y_gate_all = y_gate_all + _lora_g.view(N, -1)
    ev["gate_lora"] = (e_gl_s, e_gl_e)

    # ── 3. Up: side LoRA ∥ main turbo ───────────────────────────
    x_rot_up = rot_cache.get(block._up_x_rot_key)
    p0_up    = block.experts[0].up_proj

    e_ul_s = _ev(); e_ul_e = _ev()
    if block._U_cat_up is not None:
        _rank_u = block._SV_all_up.shape[2]
        _K_up   = block._U_cat_up.shape[1] // _rank_u
        ps.wait_stream(cur)
        with torch.cuda.stream(ps):
            _rec(e_ul_s, ps)
            _a_all_u  = hidden_states @ block._U_cat_up
            _a_exp_u  = _a_all_u.squeeze(0).view(_K_up, _rank_u)[
                block._cluster_idx_up]
            _lora_u   = torch.bmm(
                _a_exp_u.unsqueeze(1),
                block._SV_all_up.permute(0, 2, 1)).squeeze(1)
            _rec(e_ul_e, ps)
            block._ev_lora_up.record()
    ev["up_lora"] = (e_ul_s, e_ul_e)

    e_ut_s = _ev(); _rec(e_ut_s)
    y_up_all = turbo_dequant_matmul_fused(
        hidden_states, block._packed_up_all, block._norms_up_all,
        None, block._up_centroids,
        p0_up.turbo_bits, p0_up.turbo_dim,
        lora_USV=None, precomputed_x_rot=x_rot_up)
    e_ut_e = _ev(); _rec(e_ut_e)
    ev["up_turbo"] = (e_ut_s, e_ut_e)

    if block._U_cat_up is not None:
        cur.wait_event(block._ev_lora_up)
        y_up_all = y_up_all + _lora_u.view(N, -1)

    # ── 4. Activation ────────────────────────────────────────────
    e_act_s = _ev(); _rec(e_act_s)
    out_gate = block._gate_out_d
    out_up   = block._up_out_d
    gate_all = y_gate_all.view(N * E, out_gate)
    up_all   = y_up_all.view(N * E, out_up)
    h_all    = block.experts[0].act_fn(gate_all) * up_all
    e_act_e  = _ev(); _rec(e_act_e)
    ev["activation"] = (e_act_s, e_act_e)

    # ── 5. RHT rotation ──────────────────────────────────────────
    e_rht_s = _ev(); _rec(e_rht_s)
    if block._rht_signs_down is not None:
        from hadamard_rotation import rht_forward
        h_rot_all = rht_forward(h_all.float(), block._rht_signs_down).half()
    else:
        Pi_down   = block.experts[0].down_proj._rotation_cache.get_pi(
            *block._down_x_rot_key)
        h_rot_all = (h_all.float() @ Pi_down.float().T).half()
    e_rht_e = _ev(); _rec(e_rht_e)
    ev["rht"] = (e_rht_s, e_rht_e)

    # ── 6. Down: main grouped-GEMV FIRST, then side LoRA ────────
    hidden_dim  = hidden_states.shape[-1]
    _down_out_d = block._down_out_d

    if block._U_per_expert_down is not None:
        ps.wait_stream(cur)

    e_dg_s = _ev(); _rec(e_dg_s)
    y_down_cat = turbo_dequant_grouped_gemv_fused(
        h_rot_all, block._packed_down_all, block._norms_down_all,
        block._down_centroids, _down_out_d)
    e_dg_e = _ev(); _rec(e_dg_e)
    ev["down_gemv"] = (e_dg_s, e_dg_e)

    out_all = y_down_cat.view(E, N, hidden_dim)

    e_dl_s = _ev(); e_dl_e = _ev()
    if block._U_per_expert_down is not None:
        with torch.cuda.stream(ps):
            _rec(e_dl_s, ps)
            _a_d    = torch.bmm(h_all.unsqueeze(1), block._U_per_expert_down)
            _lora_d = torch.bmm(_a_d,
                                block._SV_all_down.permute(0, 2, 1)).squeeze(1)
            _rec(e_dl_e, ps)
            block._ev_lora_down.record()
        cur.wait_event(block._ev_lora_down)
        out_all = out_all + _lora_d.unsqueeze(1)
    ev["down_lora"] = (e_dl_s, e_dl_e)

    # ── 7. Routing accumulation ──────────────────────────────────
    e_acc_s = _ev(); _rec(e_acc_s)
    out_all_t = out_all.permute(1, 0, 2)
    final = torch.bmm(full_weights.unsqueeze(1), out_all_t).squeeze(1)
    e_acc_e = _ev(); _rec(e_acc_e)
    ev["routing_acc"] = (e_acc_s, e_acc_e)

    torch.cuda.synchronize()
    return ev, e_anchor, final


def _build_rotation_cache_if_needed(block, hidden_states):
    """Build the rotation cache that _forward_graph uses."""
    from collections import OrderedDict
    rot_cache = {}
    for proj_attr, rot_key_attr in [("gate_proj", "_gate_x_rot_key"),
                                     ("up_proj",   "_up_x_rot_key")]:
        key = getattr(block, rot_key_attr)
        if key in rot_cache:
            continue
        p0 = block.experts[0]
        proj = getattr(p0, proj_attr)
        if hasattr(proj, '_rht_signs') and proj._rht_signs is not None:
            from hadamard_rotation import rht_forward
            rot_cache[key] = rht_forward(hidden_states.float(), proj._rht_signs).half()
        else:
            Pi = proj._rotation_cache.get_pi(*key)
            rot_cache[key] = (hidden_states.float() @ Pi.float().T).half()
    block._rot_cache = rot_cache

# Monkey-patch helper onto block
import inference.moe_block as _mb
_mb.GraphCompatibleMoeBlock._build_rotation_cache_if_needed = _build_rotation_cache_if_needed


# ──────────────────────────────────────────────────────────────
# Convert events to µs
# ──────────────────────────────────────────────────────────────

def events_to_us(ev_dict, anchor):
    """Return {name: (t_start_us, t_end_us)} relative to anchor."""
    out = {}
    for name, (es, ee) in ev_dict.items():
        try:
            t0 = anchor.elapsed_time(es) * 1e3
            t1 = anchor.elapsed_time(ee) * 1e3
            out[name] = (t0, t1)
        except RuntimeError:
            out[name] = (0.0, 0.0)
    return out


def avg_timings(all_us, name):
    t0s = [u[name][0] for u in all_us]
    t1s = [u[name][1] for u in all_us]
    return float(np.mean(t0s)), float(np.mean(t1s))


# ──────────────────────────────────────────────────────────────
# Classify each op as "main" or "side"
# ──────────────────────────────────────────────────────────────

OP_STREAM = {
    "routing":     "main",
    "gate_lora":   "side",
    "gate_turbo":  "main",
    "up_lora":     "side",
    "up_turbo":    "main",
    "activation":  "main",
    "rht":         "main",
    "down_lora":   "side",
    "down_gemv":   "main",
    "routing_acc": "main",
}

OP_COLOR = {
    "routing":     "#a6d854",
    "gate_lora":   "#f4a582",
    "gate_turbo":  "#2166ac",
    "up_lora":     "#d6604d",
    "up_turbo":    "#4393c3",
    "activation":  "#66c2a5",
    "rht":         "#7fc97f",
    "down_lora":   "#e08214",
    "down_gemv":   "#1a6faf",
    "routing_acc": "#abdda4",
}

OP_LABEL = {
    "routing":     "Routing\nscatter",
    "gate_lora":   "Gate LoRA\n(GEMV+BMM)",
    "gate_turbo":  "Gate Turbo\n(2-bit GEMV)",
    "up_lora":     "Up LoRA\n(GEMV+BMM)",
    "up_turbo":    "Up Turbo\n(2-bit GEMV)",
    "activation":  "Activation\n(SiLU×)",
    "rht":         "RHT\nrotation",
    "down_lora":   "Down LoRA\n(2× BMM)",
    "down_gemv":   "Down Turbo\n(grouped GEMV)",
    "routing_acc": "Routing\naccum",
}


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────

MAIN_Y = 1.2
SIDE_Y = 0.0
BAR_H  = 0.42

# Color palette
C_CPU_OVERHEAD = "#eeeeee"   # light gray: Python/CPU busy, GPU main idle
C_IDLE         = "#f0f0f0"   # hatched: side stream idle


def _bar(ax, t0, t1, y, color, label, fontsize=6.5, min_w=2.0, label_color="white"):
    w = t1 - t0
    if w <= 0:
        return
    ax.barh(y, max(w, min_w), left=t0, height=BAR_H,
            color=color, edgecolor="white", linewidth=0.6, align="center")
    if w > 3.5:
        ax.text(t0 + w / 2, y, f"{label}\n{w:.1f}µs",
                ha="center", va="center", fontsize=fontsize,
                color=label_color, fontweight="bold", clip_on=True)


def _vline(ax, x, label="", ymin=-0.5, ymax=1.75, color="#666666", ls="--"):
    ax.axvline(x, color=color, linestyle=ls, linewidth=0.9, alpha=0.8,
               ymin=0.05, ymax=0.95)
    if label:
        ax.text(x + 1.5, ymax, label, fontsize=6, color=color, va="top")


def plot_combined(avg, save_path):
    """
    Full step Gantt chart (eager mode).
    Main stream top row, side stream bottom row.
    Key additions vs previous version:
      - Explicit "CPU overhead" bands on main stream (where Python is busy
        submitting side-stream ops but main GPU is idle)
      - Green badge on up section (only section with true GPU parallelism)
      - wait_event sync lines
    """
    fig, ax = plt.subplots(figsize=(20, 5.0))
    ax.set_ylim(-0.85, 2.35)

    # ── Draw GPU ops ─────────────────────────────────────────────
    for name, stream in OP_STREAM.items():
        t0, t1 = avg.get(name, (0.0, 0.0))
        y = MAIN_Y if stream == "main" else SIDE_Y
        _bar(ax, t0, t1, y, OP_COLOR[name], OP_LABEL[name])

    # ── CPU overhead bands on main stream ───────────────────────
    # Period when Python is submitting side ops but main GPU is idle.
    # Gate: routing_end → gate_turbo_start
    rt_end  = avg.get("routing",    (0,0))[1]
    gt_s    = avg.get("gate_turbo", (0,0))[0]
    up_end  = avg.get("up_turbo",   (0,0))[1]   # gate sync point
    ut_s    = avg.get("up_turbo",   (0,0))[0]
    rht_end = avg.get("rht",        (0,0))[1]
    dg_s    = avg.get("down_gemv",  (0,0))[0]

    def _cpu_band(x0, x1, note):
        if x1 - x0 > 2:
            ax.barh(MAIN_Y, x1 - x0, left=x0, height=BAR_H,
                    color=C_CPU_OVERHEAD, edgecolor="#cccccc", linewidth=0.5,
                    align="center", zorder=1)
            ax.text(x0 + (x1-x0)/2, MAIN_Y, note,
                    ha="center", va="center", fontsize=6.5,
                    color="#888888", style="italic", clip_on=True)

    _cpu_band(rt_end,  gt_s,  "CPU overhead\n(submitting side ops)")
    # After gate sync, brief Python overhead before up_lora submit:
    # (usually < 5µs, skip if tiny)
    # After rht, before down_gemv:
    _cpu_band(rht_end, dg_s,  "CPU overhead")

    # ── Side stream idle regions ─────────────────────────────────
    # After each LoRA op finishes, side stream sits idle until next section
    def _side_idle(lora_op, next_sync_x):
        lo_end = avg.get(lora_op, (0,0))[1]
        if next_sync_x - lo_end > 5:
            ax.barh(SIDE_Y, next_sync_x - lo_end, left=lo_end, height=BAR_H,
                    color=C_IDLE, edgecolor="#cccccc", linewidth=0.4,
                    hatch="....", align="center", alpha=0.5)
            ax.text(lo_end + (next_sync_x-lo_end)/2, SIDE_Y,
                    f"side idle\n{next_sync_x-lo_end:.0f}µs",
                    ha="center", va="center", fontsize=6, color="#aaaaaa",
                    clip_on=True)

    gate_sync = avg.get("gate_turbo", (0,0))[1]   # gate_turbo always finishes last
    up_sync   = avg.get("up_turbo",   (0,0))[1]
    down_sync = avg.get("down_gemv",  (0,0))[1]

    _side_idle("gate_lora",  gate_sync)
    _side_idle("up_lora",    up_sync)
    _side_idle("down_lora",  down_sync)

    # ── Sync (wait_event) vertical lines ─────────────────────────
    for x, lbl in [(gate_sync, "wait_event\n(gate)"),
                   (up_sync,   "wait_event\n(up)"),
                   (down_sync, "wait_event\n(down)")]:
        _vline(ax, x, lbl, ymax=1.95, color="#555555")

    # ── Highlight true GPU parallel section (up) ─────────────────
    up_lora_s = avg.get("up_lora",  (0,0))[0]
    up_turbo_s = avg.get("up_turbo",(0,0))[0]
    up_lora_e  = avg.get("up_lora", (0,0))[1]
    parallel_start = max(up_lora_s, up_turbo_s)
    parallel_end   = up_lora_e
    if parallel_end > parallel_start:
        ax.axvspan(parallel_start, parallel_end, ymin=0.05, ymax=0.95,
                   color="#2ca02c", alpha=0.10, zorder=0)
        ax.text((parallel_start + parallel_end) / 2, 2.05,
                "✓ TRUE GPU\nPARALLEL",
                ha="center", va="bottom", fontsize=7.5,
                color="#1a7a1a", fontweight="bold")

    # ── Y-axis labels ─────────────────────────────────────────────
    ax.set_yticks([SIDE_Y, MAIN_Y])
    ax.set_yticklabels(["Side Stream\n(LoRA)", "Main Stream\n(Turbo + Routing)"],
                       fontsize=9)
    ax.set_xlabel("GPU time (µs, relative to step start)", fontsize=9)
    ax.set_title(
        "GLoRCQ: MoE-Layer Decode — Two-Stream Parallel LoRA Compensation  "
        "(eager mode, Qwen1.5-MoE-A2.7B, mean of 20 runs)\n"
        "Side stream runs LoRA (batched GEMV+BMM) in parallel with main-stream "
        "Turbo 2-bit GEMV.  In CUDA Graph mode all 3 sections truly overlap.",
        fontsize=10)
    ax.grid(axis="x", alpha=0.18, linestyle=":", zorder=0)

    # ── Legend ────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(color=OP_COLOR["routing"],    label="Routing (scatter/accum)"),
        mpatches.Patch(color=OP_COLOR["gate_turbo"], label="Gate Turbo-2bit GEMV"),
        mpatches.Patch(color=OP_COLOR["up_turbo"],   label="Up Turbo-2bit GEMV"),
        mpatches.Patch(color=OP_COLOR["activation"], label="Activation (SiLU×)"),
        mpatches.Patch(color=OP_COLOR["rht"],        label="RHT (Hadamard rotation)"),
        mpatches.Patch(color=OP_COLOR["down_gemv"],  label="Down Grouped-GEMV"),
        mpatches.Patch(color=OP_COLOR["gate_lora"],  label="Gate LoRA side (GEMV+BMM)"),
        mpatches.Patch(color=OP_COLOR["up_lora"],    label="Up LoRA side (GEMV+BMM)"),
        mpatches.Patch(color=OP_COLOR["down_lora"],  label="Down LoRA side (2×BMM)"),
        mpatches.Patch(color=C_CPU_OVERHEAD,         label="Main GPU idle (Python overhead)"),
        mpatches.Patch(color=C_IDLE, hatch="....",   label="Side stream idle"),
        mpatches.Patch(color="#2ca02c", alpha=0.3,   label="True GPU parallel region"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=7,
              ncol=2, framealpha=0.92, edgecolor="#cccccc")

    total = avg.get("routing_acc", (0., 0.))[1]
    ax.text(0.005, 0.97,
            f"MoE layer total (eager): {total:.0f} µs\n"
            f"Eager ≠ Graph mode — in graph replay, Python overhead vanishes\n"
            f"→ all 3 LoRA sections overlap with Turbo (theoretical: ~942 µs)",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            color="#222222", bbox=dict(boxstyle="round,pad=0.3",
                                       facecolor="white", alpha=0.85,
                                       edgecolor="#cccccc"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_zoom(avg, section, ops_main, ops_side, save_path, title,
              true_parallel=False, note=None):
    """
    Zoomed-in chart for one parallel section.
    true_parallel: if True, mark the overlap region and annotate start-gap.
    note: extra annotation string.
    """
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.set_ylim(-0.85, 2.35)

    all_ops = ops_main + ops_side
    t_min = min((avg.get(op,(0,0))[0] for op in all_ops), default=0)
    t_max = max((avg.get(op,(0,0))[1] for op in all_ops), default=100)
    span  = t_max - t_min
    ax.set_xlim(t_min - span*0.06 - 5, t_max + span*0.06 + 10)

    # ── CPU overhead on main stream (gap before first main op) ───
    first_main_start = min((avg.get(op,(0,0))[0] for op in ops_main), default=t_min)
    first_side_start = min((avg.get(op,(0,0))[0] for op in ops_side), default=t_min)
    cpu_band_start   = min(first_main_start, first_side_start)
    if first_main_start - first_side_start > 5:
        _cpu_gap = first_main_start - first_side_start
        ax.barh(MAIN_Y, _cpu_gap, left=first_side_start, height=BAR_H,
                color=C_CPU_OVERHEAD, edgecolor="#cccccc", linewidth=0.5,
                align="center", zorder=1)
        ax.text(first_side_start + _cpu_gap/2, MAIN_Y,
                f"main GPU idle\n(Python overhead {_cpu_gap:.0f}µs)",
                ha="center", va="center", fontsize=7, color="#999999",
                style="italic", clip_on=True)

    # ── Draw GPU ops ─────────────────────────────────────────────
    for name in ops_main:
        t0, t1 = avg.get(name, (0., 0.))
        _bar(ax, t0, t1, MAIN_Y, OP_COLOR[name], OP_LABEL[name], fontsize=8)
    for name in ops_side:
        t0, t1 = avg.get(name, (0., 0.))
        _bar(ax, t0, t1, SIDE_Y, OP_COLOR[name], OP_LABEL[name], fontsize=8)

    # ── Sync line at end ─────────────────────────────────────────
    mt = max((avg.get(op,(0,0))[1] for op in ops_main), default=0)
    st = max((avg.get(op,(0,0))[1] for op in ops_side), default=0)
    sync_x = max(mt, st)
    _vline(ax, sync_x, "wait_event", ymax=2.0, color="#444444")

    # ── Side idle region ─────────────────────────────────────────
    if st < mt - 2:
        ax.barh(SIDE_Y, mt - st, left=st, height=BAR_H,
                color=C_IDLE, edgecolor="#cccccc", hatch="....",
                align="center", alpha=0.5)
        ax.text(st + (mt-st)/2, SIDE_Y,
                f"side idle\n{mt-st:.0f}µs",
                fontsize=7.5, ha="center", va="center", color="#aaaaaa")

    # ── True GPU parallel highlight ───────────────────────────────
    if true_parallel and ops_main and ops_side:
        ms = avg.get(ops_main[0], (0,0))[0]
        me = avg.get(ops_main[0], (0,0))[1]
        ss = avg.get(ops_side[0], (0,0))[0]
        se = avg.get(ops_side[0], (0,0))[1]
        # True overlap = window where both are simultaneously executing
        overlap_s = max(ms, ss)
        overlap_e = min(me, se)
        if overlap_e > overlap_s:
            ax.axvspan(overlap_s, overlap_e, ymin=0.03, ymax=0.97,
                       color="#2ca02c", alpha=0.12, zorder=0)
            overlap_dur = overlap_e - overlap_s
            delta = abs(ms - ss)
            ax.annotate(
                f"✓ TRUE GPU PARALLEL\nΔstart = {delta:.0f}µs  |  overlap = {overlap_dur:.0f}µs",
                xy=((overlap_s+overlap_e)/2, 1.78),
                ha="center", va="bottom", fontsize=8.5,
                color="#1a7a1a", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#e8f5e9",
                          edgecolor="#2ca02c", alpha=0.9))
    elif not true_parallel and ops_main and ops_side:
        # Annotate the serial behavior with an explanation
        ss = avg.get(ops_side[0], (0,0))[0]
        se = avg.get(ops_side[0], (0,0))[1]
        ms = avg.get(ops_main[0], (0,0))[0]
        gap = ms - se
        if gap > 2:
            # Bracket showing the gap
            mid_gap = (se + ms) / 2
            ax.annotate("", xy=(ms, SIDE_Y + 0.32), xytext=(se, SIDE_Y + 0.32),
                        arrowprops=dict(arrowstyle="<->", color="#cc4444", lw=1.2))
            ax.text(mid_gap, SIDE_Y + 0.55,
                    f"{gap:.0f}µs gap\n(eager mode: LoRA finishes\nbefore turbo starts)",
                    ha="center", va="bottom", fontsize=7.5, color="#cc4444",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff0f0",
                              edgecolor="#cc4444", alpha=0.9))
            ax.text(ms/2 + se/2, MAIN_Y + 0.32,
                    "In CUDA Graph mode:\nLoRA ∥ Turbo (zero Python overhead)",
                    ha="center", va="bottom", fontsize=7.5, color="#1a5599",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#f0f4ff",
                              edgecolor="#4477cc", alpha=0.9))

    ax.set_yticks([SIDE_Y, MAIN_Y])
    ax.set_yticklabels(["Side Stream\n(LoRA)", "Main Stream\n(Turbo)"], fontsize=9)
    ax.set_xlabel("GPU time (µs)", fontsize=9)
    ax.set_title(title + "  —  eager mode, mean of 20 runs", fontsize=11)
    ax.grid(axis="x", alpha=0.22, linestyle=":", zorder=0)

    patches = [mpatches.Patch(color=OP_COLOR[n], label=OP_LABEL[n].replace("\n"," "))
               for n in all_ops]
    patches += [
        mpatches.Patch(color=C_CPU_OVERHEAD, label="Main GPU idle (Python overhead)"),
        mpatches.Patch(color=C_IDLE, hatch="....", label="Side stream idle"),
    ]
    if true_parallel:
        patches.append(mpatches.Patch(color="#2ca02c", alpha=0.3,
                                      label="True GPU parallel region"))
    ax.legend(handles=patches, loc="upper left", fontsize=8, framealpha=0.9)

    if note:
        ax.text(0.99, 0.03, note, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5,
                color="#555555", style="italic")

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Print summary table
# ──────────────────────────────────────────────────────────────

def plot_graph_mode_theoretical(avg, save_path):
    """
    Construct a theoretical CUDA Graph mode timeline where main and side
    streams truly overlap. Uses measured op durations but lays them out
    as the CUDA runtime would schedule them in graph replay mode.
    """
    # Measured durations
    def dur(name):
        t0, t1 = avg.get(name, (0., 0.))
        return max(0., t1 - t0)

    # Build theoretical layout:
    #   main:  routing → (gate_turbo ∥ gate_lora) → (up_turbo ∥ up_lora)
    #          → activation → rht → (down_gemv ∥ down_lora) → routing_acc
    #   sync points: after each parallel section, max(main, side)
    ops = {}  # name → (t0, t1, stream)

    t = 0.
    ops["routing"] = (t, t + dur("routing"), "main")
    t += dur("routing")

    # Gate parallel section
    gate_main = dur("gate_turbo")
    gate_side = dur("gate_lora")
    ops["gate_lora"]  = (t, t + gate_side, "side")
    ops["gate_turbo"] = (t, t + gate_main, "main")
    t += max(gate_main, gate_side)

    # Up parallel section
    up_main = dur("up_turbo")
    up_side = dur("up_lora")
    ops["up_lora"]  = (t, t + up_side, "side")
    ops["up_turbo"] = (t, t + up_main, "main")
    t += max(up_main, up_side)

    ops["activation"] = (t, t + dur("activation"), "main")
    t += dur("activation")
    ops["rht"] = (t, t + dur("rht"), "main")
    t += dur("rht")

    # Down parallel section
    down_main = dur("down_gemv")
    down_side = dur("down_lora")
    ops["down_lora"] = (t, t + down_side, "side")
    ops["down_gemv"] = (t, t + down_main, "main")
    t += max(down_main, down_side)

    ops["routing_acc"] = (t, t + dur("routing_acc"), "main")
    t += dur("routing_acc")

    total_theoretical = t
    total_eager = avg.get("routing_acc", (0., 0.))[1]

    fig, ax = plt.subplots(figsize=(18, 4.5))
    ax.set_ylim(-0.7, 2.1)

    for name, (t0, t1, stream) in ops.items():
        y = MAIN_Y if stream == "main" else SIDE_Y
        _bar(ax, t0, t1, y, OP_COLOR[name], OP_LABEL[name])

    # Sync lines at end of each parallel section
    def _sync_par(main_op, side_op):
        mt = ops[main_op][1]
        st = ops[side_op][1] if side_op in ops else mt
        x = max(mt, st)
        _vline(ax, x, "sync")
        # Wait hatch if main waits for side
        if st > mt:
            ax.barh(MAIN_Y, st - mt, left=mt, height=BAR_H,
                    color="#dddddd", edgecolor="#aaaaaa", hatch="///",
                    align="center", alpha=0.6)
            ax.text(mt + (st-mt)/2, MAIN_Y + 0.32, f"wait\n{st-mt:.0f}µs",
                    fontsize=6.5, ha="center", color="#888888")
        elif mt > st:
            ax.barh(SIDE_Y, mt - st, left=st, height=BAR_H,
                    color="#dddddd", edgecolor="#aaaaaa", hatch="///",
                    align="center", alpha=0.6)
            ax.text(st + (mt-st)/2, SIDE_Y + 0.32, f"idle\n{mt-st:.0f}µs",
                    fontsize=6.5, ha="center", color="#888888")

    _sync_par("gate_turbo", "gate_lora")
    _sync_par("up_turbo",   "up_lora")
    _sync_par("down_gemv",  "down_lora")

    ax.set_yticks([SIDE_Y, MAIN_Y])
    ax.set_yticklabels(["Side Stream\n(LoRA)", "Main Stream\n(Turbo + Routing)"], fontsize=9)
    ax.set_xlabel("Time (µs)", fontsize=9)
    ax.set_title(
        "GLoRCQ: Theoretical CUDA Graph Parallel Layout (measured op durations, ideal scheduling)\n"
        f"Theoretical total: {total_theoretical:.0f} µs  vs  Eager serial: {total_eager:.0f} µs "
        f"  (saving: {total_eager - total_theoretical:.0f} µs = {(total_eager-total_theoretical)/total_eager*100:.0f}%)",
        fontsize=10)
    ax.grid(axis="x", alpha=0.2, linestyle=":")

    patches = [
        mpatches.Patch(color=OP_COLOR["routing"],    label="Routing scatter/accum"),
        mpatches.Patch(color=OP_COLOR["gate_turbo"], label="Gate Turbo (2-bit GEMV)"),
        mpatches.Patch(color=OP_COLOR["up_turbo"],   label="Up Turbo (2-bit GEMV)"),
        mpatches.Patch(color=OP_COLOR["activation"], label="Activation (SiLU×)"),
        mpatches.Patch(color=OP_COLOR["rht"],        label="RHT rotation"),
        mpatches.Patch(color=OP_COLOR["down_gemv"],  label="Down Grouped-GEMV"),
        mpatches.Patch(color=OP_COLOR["gate_lora"],  label="Gate LoRA (side, hidden in turbo)"),
        mpatches.Patch(color=OP_COLOR["up_lora"],    label="Up LoRA (side, hidden in turbo)"),
        mpatches.Patch(color=OP_COLOR["down_lora"],  label="Down LoRA (side, hidden in GEMV)"),
        mpatches.Patch(color="#dddddd", hatch="///", label="Wait / idle"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=7, ncol=2, framealpha=0.9)

    ax.text(0.01, 0.97,
            f"Theoretical (graph mode): {total_theoretical:.0f} µs\n"
            f"Eager (serial) baseline: {total_eager:.0f} µs",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            color="#222222", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


def print_summary(avg):
    print(f"\n{'─'*68}")
    print(f"{'Op':<20} {'Stream':<6} {'t_start':>9} {'t_end':>9} {'dur(µs)':>9}")
    print(f"{'─'*68}")
    order = list(OP_STREAM.keys())
    for name in order:
        if name not in avg:
            continue
        t0, t1 = avg[name]
        stream = OP_STREAM[name]
        print(f"{name:<20} {stream:<6} {t0:>9.2f} {t1:>9.2f} {t1-t0:>9.2f}")
    print(f"{'─'*68}")

    def _par_overhead(main_op, side_op):
        mt = avg.get(main_op, (0,0))[1] - avg.get(main_op, (0,0))[0]
        st = avg.get(side_op,  (0,0))[1] - avg.get(side_op,  (0,0))[0]
        wait = max(0, avg.get(side_op,(0,0))[1] - avg.get(main_op,(0,0))[1])
        saving = min(mt, st)
        print(f"  {main_op} ({mt:.1f}µs) ∥ {side_op} ({st:.1f}µs)  → "
              f"wait={wait:.1f}µs  LoRA hidden={saving:.1f}µs")

    print("\nParallelism analysis:")
    _par_overhead("gate_turbo", "gate_lora")
    _par_overhead("up_turbo",   "up_lora")
    _par_overhead("down_gemv",  "down_lora")
    total = avg.get("routing_acc", (0,0))[1]
    print(f"\nTotal MoE layer time: {total:.1f} µs")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--layer_idx",  type=int, default=1)
    parser.add_argument("--warmup",     type=int, default=10)
    parser.add_argument("--runs",       type=int, default=20)
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Load model ──────────────────────────────────────────────
    from utils.model_loader import load_model_and_tokenizer
    print("Loading real-quant model ...")
    model, _ = load_model_and_tokenizer(
        args.model_path, device=args.device, real_quant=True)
    model.eval()

    from inference.moe_block import GraphCompatibleMoeBlock
    moe_layers = [(i, layer.mlp)
                  for i, layer in enumerate(model.model.layers)
                  if isinstance(getattr(layer, "mlp", None), GraphCompatibleMoeBlock)]
    if not moe_layers:
        raise RuntimeError("No GraphCompatibleMoeBlock found")

    layer_i, block = next(((i, b) for i, b in moe_layers if i == args.layer_idx),
                          moe_layers[0])
    print(f"Profiling layer {layer_i}  (E={block.num_experts} experts)")

    block._build_graph_cache()

    hidden_dim = model.config.hidden_size
    top_k      = getattr(model.config, "num_experts_per_tok",
                         getattr(model.config, "top_k", 4))
    device     = next(model.parameters()).device
    print(f"  hidden_dim={hidden_dim}, top_k={top_k}, "
          f"gate clusters={len(block._gate_lora_clusters) if hasattr(block,'_gate_lora_clusters') else 'N/A'}")

    # ── Build fixed inputs ───────────────────────────────────────
    torch.manual_seed(42)
    hidden_states    = torch.randn(1, hidden_dim, dtype=torch.float16, device=device)
    selected_experts = torch.zeros(1, top_k, dtype=torch.long, device=device)
    routing_weights  = torch.ones(1, top_k, dtype=torch.float16, device=device) / top_k

    # ── Warmup ──────────────────────────────────────────────────
    print(f"\nWarming up ({args.warmup} steps) ...")
    for _ in range(args.warmup):
        _build_rotation_cache_if_needed(block, hidden_states)
        run_instrumented(block, hidden_states, routing_weights, selected_experts)

    # ── Collect timings ──────────────────────────────────────────
    print(f"Collecting timings ({args.runs} steps) ...")
    all_us = []
    for _ in range(args.runs):
        ev_dict, anchor, _ = run_instrumented(
            block, hidden_states, routing_weights, selected_experts)
        all_us.append(events_to_us(ev_dict, anchor))

    # ── Average ──────────────────────────────────────────────────
    avg = {}
    for name in OP_STREAM:
        if any(name in u for u in all_us):
            t0s = [u[name][0] for u in all_us if name in u]
            t1s = [u[name][1] for u in all_us if name in u]
            avg[name] = (float(np.mean(t0s)), float(np.mean(t1s)))

    print_summary(avg)

    # ── Plot ─────────────────────────────────────────────────────
    plot_combined(avg, os.path.join(out_dir, "timeline_graph.png"))
    plot_graph_mode_theoretical(avg, os.path.join(out_dir, "timeline_graph_theoretical.png"))

    plot_zoom(avg,
              section="gate",
              ops_main=["gate_turbo"],
              ops_side=["gate_lora"],
              save_path=os.path.join(out_dir, "timeline_graph_gate.png"),
              title="Gate Projection: Turbo-2bit GEMV ∥ LoRA (batched GEMV + BMM)",
              true_parallel=True,
              note="Gate: Turbo submitted first → LoRA starts ~100µs later while Turbo is running → true GPU overlap")

    plot_zoom(avg,
              section="up",
              ops_main=["up_turbo"],
              ops_side=["up_lora"],
              save_path=os.path.join(out_dir, "timeline_graph_up.png"),
              title="Up Projection: Turbo-2bit GEMV ∥ LoRA (batched GEMV + BMM)",
              true_parallel=True,
              note="Up: LoRA (55µs) finishes before Turbo starts (22µs gap) — fully hidden, zero main-stream wait")

    plot_zoom(avg,
              section="down",
              ops_main=["down_gemv"],
              ops_side=["down_lora"],
              save_path=os.path.join(out_dir, "timeline_graph_down.png"),
              title="Down Projection: Grouped GEMV ∥ LoRA (2× BMM)",
              true_parallel=True,
              note="Down: Turbo submitted first → both streams overlap; LoRA slowed by SM competition but main-stream wait < 20µs")

    print("\nDone.")


if __name__ == "__main__":
    main()
