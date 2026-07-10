"""Fig 4b for Qwen3-30B-A3B — cluster validity: 2D MDS scatter (colored by
cluster) + within/between distance histogram.

MDS on 6k points is slow, so the scatter uses a stratified subsample (cap per
cluster) while the histogram uses ALL pairwise distances. Reads the Qwen3
D_grass npz produced by compute_D_grass_qwen3.py.

Usage:
    python exp/cluster/plot_fig4b_qwen3.py
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
SUBSAMPLE = 1600  # cap points fed to MDS for tractable 2D embedding


def plot_one(wt: str):
    npz_path = os.path.join(OUT_DIR, f"D_grass_qwen3_{wt}.npz")
    if not os.path.exists(npz_path):
        print(f"[skip] {wt}: {npz_path} not found", flush=True)
        return
    data = np.load(npz_path)
    D = data["D_grass"].astype(np.float64)
    gid = data["group_id"].astype(np.int32)
    N = D.shape[0]
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    n_clusters = int(gid.max()) + 1

    # --- histogram uses ALL pairs ---
    iu = np.triu_indices(N, k=1)
    same = gid[iu[0]] == gid[iu[1]]
    within = D[iu][same]
    between = D[iu][~same]

    # --- MDS on a stratified subsample ---
    rng = np.random.default_rng(SEED)
    if N > SUBSAMPLE:
        # stratified: proportional per cluster, >=1 each
        idx = []
        for c in range(n_clusters):
            members = np.where(gid == c)[0]
            k = max(1, int(round(len(members) * SUBSAMPLE / N)))
            idx.extend(rng.choice(members, size=min(k, len(members)), replace=False))
        idx = np.array(sorted(idx))
    else:
        idx = np.arange(N)
    Dsub = D[np.ix_(idx, idx)]
    gsub = gid[idx]
    print(f"[{wt}] MDS embedding {len(idx)} / {N} experts ...", flush=True)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=SEED,
              n_init=1, max_iter=300, normalized_stress="auto")
    xy = mds.fit_transform(Dsub)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.8, 5.4),
                                   gridspec_kw={"width_ratios": [1.35, 1.0]})
    cmap = plt.get_cmap("gist_ncar", n_clusters)
    axL.scatter(xy[:, 0], xy[:, 1], c=gsub, cmap=cmap, s=10, alpha=0.75, linewidths=0)
    axL.set_title(f"2D MDS of expert subspaces — {wt}\n"
                  f"(Qwen3-30B-A3B; {len(idx)}-point subsample; color = shared-U cluster)")
    axL.set_xlabel("MDS dim 1"); axL.set_ylabel("MDS dim 2")
    axL.set_xticks([]); axL.set_yticks([])

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
        fig.savefig(os.path.join(OUT_DIR, f"fig4b_qwen3_{wt}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[{wt}] within μ={within.mean():.4f}  between μ={between.mean():.4f}  "
          f"sep={between.mean()-within.mean():.4f}  ratio={within.mean()/between.mean():.3f}", flush=True)


def main():
    for wt in WTYPES:
        plot_one(wt)


if __name__ == "__main__":
    main()
