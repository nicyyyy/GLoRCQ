"""Faithful (activation-scaled) Qwen3 D_grass from the phase1 cache saved by the
random-cluster quant. Uses the SAME distance the clusterer used (diag(S_a)·W^T),
paired with the Grassmannian cluster labels from qwen3_assignments.pkl.
Writes D_grass_qwen3_scaled_{wtype}.npz.
"""
import os, sys, pickle, time
import numpy as np, torch
_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _R)
from cross_layer_share import _grassmannian_dist_matrix

CACHE = "/mnt/Data/yqy/resource_dir/glorcq_grassmann/qwen3_random_phase1_cache.pt"
OUT = os.path.join(_R, "exp/cluster")
PKL = os.path.join(OUT, "qwen3_assignments.pkl")
WTYPES = ("gate_proj", "up_proj", "down_proj")
RANK = 32

def parse(name):
    p = name.split(".")
    return p[-1], int(p[-2])

def main():
    assign = pickle.load(open(PKL, "rb"))
    print("[load] cache ...", flush=True)
    c = torch.load(CACHE, map_location="cpu", weights_only=False)
    recs = c["all_expert_recs"]
    # index by (wtype, layer, expert)
    by_key = {}
    for r in recs:
        if not r["name"].startswith("mlp.experts."):
            continue
        wt, ex = parse(r["name"])
        by_key[(wt, r["layer"], ex)] = r
    for wt in WTYPES:
        entries = assign[wt]
        E_T_list, gids = [], []
        for e in entries:
            r = by_key.get((wt, e["layer"], e["expert"]))
            if r is None:
                continue
            W = r["W_orig"].float().T          # (in_d, out_d)
            s = r["scales"].float()
            E_T_list.append(W * s.unsqueeze(1))  # activation-scaled
            gids.append(e["group_id"])
        gids = np.asarray(gids, dtype=np.int32)
        print(f"[{wt}] N={len(E_T_list)} (scaled); computing D_grass ...", flush=True)
        t0 = time.time()
        D, _ = _grassmannian_dist_matrix(E_T_list, rank=RANK, compute_grass=True)
        print(f"[{wt}] {D.shape} in {time.time()-t0:.1f}s", flush=True)
        np.savez(os.path.join(OUT, f"D_grass_qwen3_scaled_{wt}.npz"),
                 D_grass=D.astype(np.float32), group_id=gids, rank=np.int32(RANK), N=np.int32(len(gids)))
        del E_T_list; import gc; gc.collect()

if __name__ == "__main__":
    main()
