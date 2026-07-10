"""Fig 4a for Qwen3-30B-A3B — layer x cluster composition heatmap.
Reads the small assignments pickle extracted from cross_layer_info.pt on H2.
"""
import os, pickle
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PKL = os.path.join(OUT_DIR, "qwen3_assignments.pkl")
WTYPES = ("gate_proj", "up_proj", "down_proj")
N_LAYERS = 48

def main():
    assign = pickle.load(open(PKL, "rb"))
    for wt in WTYPES:
        lst = assign[wt]
        n_clusters = max(a["group_id"] for a in lst) + 1
        counts = np.zeros((N_LAYERS, n_clusters), dtype=np.int32)
        for a in lst:
            counts[a["layer"], a["group_id"]] += 1
        fig, ax = plt.subplots(figsize=(8.5, 6.6))
        im = ax.imshow(counts, aspect="auto", cmap="magma", origin="lower",
                       interpolation="nearest", vmin=0)
        ax.set_xlabel("Cluster (shared-U group)")
        ax.set_ylabel("Source Transformer layer")
        ax.set_title(f"Layer × cluster composition — {wt}\n"
                     f"(Qwen3-30B-A3B: 48 layers × 128 experts, 48 groups; "
                     f"traversal at G=128 would be strictly one-layer-per-cluster)")
        ax.set_xticks(range(0, n_clusters, 4))
        ax.set_yticks(range(0, N_LAYERS, 4))
        cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label("Experts drawn from this layer")
        plt.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(OUT_DIR, f"fig4a_qwen3_{wt}.{ext}"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        col_layers = (counts > 0).sum(axis=0)
        print(f"[{wt}] {n_clusters} clusters; distinct layers/cluster "
              f"min/max/mean = {col_layers.min()}/{col_layers.max()}/{col_layers.mean():.1f} "
              f"(traversal degenerate = 1)", flush=True)


if __name__ == "__main__":
    main()
