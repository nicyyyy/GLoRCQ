"""Fig 4a — layer x cluster composition heatmap (Qwen1.5-MoE).

Reads cluster assignments from cross_layer_info.pt. For each weight type, builds
a (n_layers x n_clusters) matrix where cell (l, c) = number of experts from
Transformer layer l assigned to cluster c. If clustering had degenerated to
per-layer grouping, mass would concentrate on a diagonal / block pattern
(each cluster drawn from one layer). Instead the mass is spread down every
column → clusters draw experts from across all layers.

Usage:
    python exp/cluster/plot_fig4a.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CROSS_LAYER_INFO = "/mnt/Data/yqy/resource_dir/glorcq_grassmann/qwen15_fair_grassmann_v3a_r32_recon0/cross_layer_info.pt"
OUT_DIR = os.path.join(_REPO_ROOT, "exp/cluster")

WTYPES = ("gate_proj", "up_proj", "down_proj")
N_LAYERS = 24


def plot_one(wtype: str, assignments_list):
    n_clusters = max(a["group_id"] for a in assignments_list) + 1
    counts = np.zeros((N_LAYERS, n_clusters), dtype=np.int32)
    for a in assignments_list:
        counts[a["layer"], a["group_id"]] += 1

    fig, ax = plt.subplots(figsize=(6.2, 6.4))
    im = ax.imshow(counts, aspect="auto", cmap="magma", origin="lower",
                   interpolation="nearest", vmin=0)

    ax.set_xlabel("Cluster (shared-U group)")
    ax.set_ylabel("Source Transformer layer")
    ax.set_title(f"Layer × cluster composition — {wtype}\n(Qwen1.5-MoE: 24 layers × 60 experts, 12 groups)")
    ax.set_xticks(range(n_clusters))
    ax.set_yticks(range(0, N_LAYERS, 2))

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("Experts drawn from this layer")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig4a_qwen15_{wtype}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Diagnostic: how uniform is each column across layers?
    # A perfectly cross-layer clustering has each cluster drawing ~equally from all layers.
    col_layers = (counts > 0).sum(axis=0)  # distinct layers per cluster
    print(f"[{wtype}] {n_clusters} clusters; distinct layers/cluster "
          f"min/max/mean = {col_layers.min()}/{col_layers.max()}/{col_layers.mean():.1f} "
          f"(intra-layer degenerate would be 1)", flush=True)


def main():
    print(f"[load] {CROSS_LAYER_INFO}", flush=True)
    xl = torch.load(CROSS_LAYER_INFO, map_location="cpu", weights_only=False)
    assignments = xl["assignments"]
    for wt in WTYPES:
        if wt in assignments:
            plot_one(wt, assignments[wt])


if __name__ == "__main__":
    main()
