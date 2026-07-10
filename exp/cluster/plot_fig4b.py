"""Fig 4b — cluster validity: 2D MDS scatter (colored by cluster) + within/between
distance histogram (Qwen1.5-MoE).

Two panels per weight type:
  (left)  2D MDS embedding of the pairwise Grassmannian distance matrix, one
          point per expert, colored by cluster assignment. Same-color points
          grouping together = the clustering captures real subspace structure.
  (right) Histogram of within-cluster vs between-cluster pairwise distances.
          A left-shifted within-cluster distribution = clustering is meaningful.

Uses the D_grass npz produced by compute_D_grass.py (distance matrix + group_id).

Usage:
    python exp/cluster/plot_fig4b.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(_REPO_ROOT, "exp/cluster")
WTYPES = ("gate_proj", "up_proj", "down_proj")
SEED = 42


def plot_one(wt: str):
    npz_path = os.path.join(OUT_DIR, f"D_grass_qwen15_{wt}.npz")
    if not os.path.exists(npz_path):
        print(f"[skip] {wt}: {npz_path} not found", flush=True)
        return
    data = np.load(npz_path)
    D = data["D_grass"].astype(np.float64)
    gid = data["group_id"].astype(np.int32)
    N = D.shape[0]
    # Symmetrize + zero diagonal (guard against tiny numerical asymmetry)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    n_clusters = int(gid.max()) + 1

    # --- 2D MDS embedding on precomputed distances ---
    print(f"[{wt}] MDS embedding {N} experts ...", flush=True)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=SEED,
              n_init=1, max_iter=300, normalized_stress="auto")
    xy = mds.fit_transform(D)
    print(f"[{wt}] MDS stress = {mds.stress_:.4f}", flush=True)

    # --- within / between distances ---
    iu = np.triu_indices(N, k=1)
    same = gid[iu[0]] == gid[iu[1]]
    within = D[iu][same]
    between = D[iu][~same]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.4),
                                   gridspec_kw={"width_ratios": [1.35, 1.0]})

    # Left: MDS scatter colored by cluster
    cmap = plt.get_cmap("tab20", n_clusters)
    axL.scatter(xy[:, 0], xy[:, 1], c=gid, cmap=cmap, s=9, alpha=0.75,
                linewidths=0)
    axL.set_title(f"2D MDS of expert subspaces — {wt}\n(color = shared-U cluster; same color together = coherent)")
    axL.set_xlabel("MDS dim 1")
    axL.set_ylabel("MDS dim 2")
    axL.set_xticks([]); axL.set_yticks([])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_clusters - 1))
    cbar = fig.colorbar(sm, ax=axL, pad=0.01, fraction=0.046, ticks=range(0, n_clusters, 2))
    cbar.set_label("Cluster")

    # Right: within vs between distance histograms
    bins = np.linspace(0.0, 1.0, 51)
    axR.hist(between, bins=bins, density=True, alpha=0.55, color="#4c72b0",
             label=f"between-cluster (μ={between.mean():.3f})")
    axR.hist(within, bins=bins, density=True, alpha=0.65, color="#dd8452",
             label=f"within-cluster (μ={within.mean():.3f})")
    axR.axvline(within.mean(), color="#dd8452", ls="--", lw=1.2)
    axR.axvline(between.mean(), color="#4c72b0", ls="--", lw=1.2)
    axR.set_title(f"Pairwise Grassmannian distance — {wt}")
    axR.set_xlabel("Distance (0 = same subspace, 1 = orthogonal)")
    axR.set_ylabel("Density")
    axR.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig4b_qwen15_{wt}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    sep = between.mean() - within.mean()
    print(f"[{wt}] within μ={within.mean():.4f}  between μ={between.mean():.4f}  "
          f"separation={sep:.4f}  ratio={within.mean()/between.mean():.3f}", flush=True)


def main():
    for wt in WTYPES:
        plot_one(wt)


if __name__ == "__main__":
    main()
