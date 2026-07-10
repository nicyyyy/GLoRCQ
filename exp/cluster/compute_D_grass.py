"""Compute pairwise Grassmannian distance matrix for Qwen1.5-MoE experts.

Loads the Phase-1 cache used by run_quantize.py's Grassmannian branch, rebuilds
the activation-scaled weight matrices E_T = diag(S_a) @ W_orig^T per expert,
and calls cross_layer_share._grassmannian_dist_matrix directly (the same
function the clusterer uses). Saves D_grass + the SVD-sharing group_id (from
cross_layer_info.pt) into a single .npz per weight type.

Usage:
    CUDA_VISIBLE_DEVICES=5 python exp/cluster/compute_D_grass.py
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

from cross_layer_share import _grassmannian_dist_matrix  # noqa: E402


PHASE1_CACHE = "/mnt/Data/yqy/resource_dir/glorcq_smoketest/qwen1.5-moe_stripped_v1_phase1_cache.pt"
CROSS_LAYER_INFO = "/mnt/Data/yqy/resource_dir/glorcq_grassmann/qwen15_fair_grassmann_v3a_r32_recon0/cross_layer_info.pt"
OUT_DIR = os.path.join(_REPO_ROOT, "exp/cluster")

WTYPES = ("gate_proj", "up_proj", "down_proj")
RANK = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=RANK)
    ap.add_argument("--wtypes", nargs="+", default=list(WTYPES))
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[load] Phase-1 cache: {PHASE1_CACHE}", flush=True)
    t0 = time.time()
    cache = torch.load(PHASE1_CACHE, map_location="cpu", weights_only=False)
    print(f"[load] done in {time.time()-t0:.1f}s; top-level keys: {list(cache.keys()) if isinstance(cache, dict) else type(cache)}", flush=True)

    if isinstance(cache, dict) and "all_expert_recs" in cache:
        recs = cache["all_expert_recs"]
    elif isinstance(cache, list):
        recs = cache
    else:
        raise RuntimeError(f"unrecognised phase1 cache structure: keys={list(cache.keys()) if isinstance(cache, dict) else '?'}")
    print(f"[load] {len(recs)} expert records total", flush=True)

    print(f"[load] cross_layer_info: {CROSS_LAYER_INFO}", flush=True)
    xl_info = torch.load(CROSS_LAYER_INFO, map_location="cpu", weights_only=False)
    assignments = xl_info["assignments"]
    print(f"[load] assignments wtypes: {list(assignments.keys())}", flush=True)

    def _parse_name(name: str):
        # e.g. "mlp.experts.37.gate_proj" -> ("gate_proj", 37)
        parts = name.split(".")
        wtype = parts[-1]
        try:
            expert = int(parts[-2])
        except (ValueError, IndexError):
            expert = -1
        return wtype, expert

    for wt in args.wtypes:
        wt_recs = []
        for r in recs:
            name = r.get("name", "")
            rec_wt, rec_expert = _parse_name(name)
            if rec_wt != wt or not name.startswith("mlp.experts."):
                continue
            r["_wt"] = rec_wt
            r["_expert"] = rec_expert
            wt_recs.append(r)
        if not wt_recs:
            print(f"[skip] {wt}: no records found in cache", flush=True)
            continue
        n = len(wt_recs)
        print(f"\n[{wt}] N={n}", flush=True)

        gid_lookup = {(a["layer"], a["expert"]): a["group_id"] for a in assignments.get(wt, [])}

        E_T_list = []
        gids = []
        for r in wt_recs:
            W = r["W_orig"]
            s = r["scales"]
            E_T = W.float().T * s.float().unsqueeze(1)
            E_T_list.append(E_T)
            gids.append(gid_lookup.get((r["layer"], r["_expert"]), -1))
        gids = np.asarray(gids, dtype=np.int32)
        print(f"[{wt}] built E_T list; group_id coverage: {(gids >= 0).sum()}/{n}", flush=True)

        t0 = time.time()
        D_grass, _D_recon = _grassmannian_dist_matrix(E_T_list, rank=args.rank, compute_grass=True)
        print(f"[{wt}] D_grass computed in {time.time()-t0:.1f}s; shape={D_grass.shape}, dtype={D_grass.dtype}", flush=True)

        out = os.path.join(OUT_DIR, f"D_grass_qwen15_{wt}.npz")
        np.savez(out, D_grass=D_grass.astype(np.float32), group_id=gids,
                 rank=np.int32(args.rank), N=np.int32(n))
        print(f"[{wt}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
