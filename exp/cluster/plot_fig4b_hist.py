"""Fig 4b (final) — within vs between cluster distance histograms only.
The MDS scatter is dropped: with 48 clusters in a near-orthogonal high-dim
subspace, the 2D embedding is an uninformative blob. The histogram cleanly
shows same-cluster experts are closer in subspace than cross-cluster pairs.

Reads the activation-scaled D_grass npz. Qwen3 main; edit MODEL to switch.
"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
MODEL = "qwen3"           # figure tag
NPZ_TMPL = "D_grass_qwen3_scaled_{wt}.npz"
WTYPES = ("gate_proj", "up_proj", "down_proj")

def stats(wt):
    d = np.load(os.path.join(OUT, NPZ_TMPL.format(wt=wt)))
    D = d["D_grass"].astype(np.float64); gid = d["group_id"].astype(np.int32)
    N = D.shape[0]; D = 0.5*(D+D.T); np.fill_diagonal(D, 0.0)
    iu = np.triu_indices(N, k=1); same = gid[iu[0]]==gid[iu[1]]
    return D[iu][same], D[iu][~same]

# One row, three panels (gate / up / down)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
bins = np.linspace(0.3, 1.0, 43)
for ax, wt in zip(axes, WTYPES):
    within, between = stats(wt)
    ax.hist(between, bins=bins, density=True, alpha=0.55, color="#4c72b0",
            label=f"between (μ={between.mean():.3f})")
    ax.hist(within, bins=bins, density=True, alpha=0.65, color="#dd8452",
            label=f"within (μ={within.mean():.3f})")
    ax.axvline(within.mean(), color="#dd8452", ls="--", lw=1.2)
    ax.axvline(between.mean(), color="#4c72b0", ls="--", lw=1.2)
    ax.set_title(f"{wt}  (Δμ={between.mean()-within.mean():.3f})")
    ax.set_xlabel("Grassmannian distance (0=same subspace, 1=orthogonal)")
    ax.legend(loc="upper left", fontsize=8)
axes[0].set_ylabel("Density")
fig.suptitle("Within- vs between-cluster expert subspace distance — Qwen3-30B-A3B "
             "(same-cluster experts are closer)", y=1.02)
plt.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig4b_qwen3_hist.{ext}"), dpi=300, bbox_inches="tight")
print("wrote fig4b_qwen3_hist.png/pdf")
