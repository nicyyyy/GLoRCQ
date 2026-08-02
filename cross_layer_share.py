"""
cross_layer_share.py — Grassmannian expert clustering for GLoRCQ cross-layer sharing.

Restored (minimal) subset from the pre-`ac9dd0b` version of this file. Only the
clustering machinery is kept: SVD basis extraction, pairwise Grassmannian +
cross-reconstruction distances, and spectral-cluster assignment. Stages 3-5
of the historical file (Hessian-weighted reconstruction, int-N storage, bit
accounting) are handled by the current `run_quantize.py::fill_phase2` pipeline
and are NOT reintroduced.

Usage from run_quantize.py::fill_phase2 (when --cluster_method grassmannian):
    from cross_layer_share import cluster_residuals
    assignments, wtype_indices = cluster_residuals(
        all_residuals, rank=fix_rank, G_moe=G, G_attn=G,
        cluster_on_original=True, hessian_svd=False,
        recon_weight=0.0, share_attn=False,
    )

Input record format for `all_residuals[i]` (when cluster_on_original=True):
    {
        'type':            'gate_proj' | 'up_proj' | 'down_proj' | ...
        'weight_orig':     torch.Tensor (out_d, in_d), fp16 CPU
        '_act_scale_cpu':  torch.Tensor (in_d,), fp32 CPU  (optional but preferred)
        'hessian_diag':    torch.Tensor (in_d,), fp32 CPU  (fallback if no act_scale)
    }
"""

import math
import numpy as np
import torch
from sklearn.cluster import SpectralClustering


DEV = torch.device("cuda")

_ATTN_WTYPES = {"q_proj", "k_proj", "v_proj", "o_proj"}
_MOE_WTYPES  = {"gate_proj", "up_proj", "down_proj"}
_ALL_WTYPES  = _ATTN_WTYPES | _MOE_WTYPES

# Mixtral uses w1/w2/w3 instead of gate_proj/up_proj/down_proj.
# The caller normalizes to canonical names before passing records here.
_MIXTRAL_WTYPE_MAP = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}


def _compute_svd_us(E_T: torch.Tensor, rank: int):
    """Randomized truncated SVD; returns (U, S) as float32 numpy arrays."""
    r = min(rank, E_T.shape[0], E_T.shape[1])
    try:
        U, S, _ = torch.svd_lowrank(E_T.float().to(DEV), q=r, niter=4)
        return (U.cpu().numpy().astype(np.float32),
                S.cpu().numpy().astype(np.float32))
    except Exception:
        return (np.zeros((E_T.shape[0], r), dtype=np.float32),
                np.zeros(r, dtype=np.float32))


def _grassmannian_dist_matrix(E_T_list, rank: int, compute_grass: bool = True):
    """
    Pairwise Grassmannian + cross-reconstruction distances.

    D_grass(i,j) = || arccos( sigma(U_i^T U_j) ) ||_2 / (sqrt(r) * pi/2)  in [0,1]
    D_recon(i,j) = 1 - 0.5 * (||M_ij S_j||^2 / ||S_j||^2
                              + ||diag(S_i) M_ij||^2 / ||S_i||^2)  in [0,1]

    where M_ij = U_i^T U_j (r x r, computed in the same inner loop).

    Args:
      compute_grass : if False, skip svdvals and return D_grass=zeros
                      (use when recon_weight=1.0).

    Returns (D_grass, D_recon) as (N,N) numpy float32 arrays.
    """
    N = len(E_T_list)
    print(f"  [grassmannian] SVD for {N} modules, rank={rank} ...", flush=True)
    results = [_compute_svd_us(E_T, rank) for E_T in E_T_list]
    U_np = np.stack([res[0] for res in results], axis=0)  # (N, in_d, r)
    S_np = np.stack([res[1] for res in results], axis=0)  # (N, r)
    _, _, r = U_np.shape
    max_d = np.sqrt(r) * (np.pi / 2)

    U_all  = torch.from_numpy(U_np).to(DEV)          # (N, in_d, r)
    S_all  = torch.from_numpy(S_np).to(DEV)          # (N, r)
    S_sq   = (S_all ** 2).sum(dim=1)                  # (N,) ||S_i||^2

    D_grass = torch.zeros(N, N, device=DEV)
    D_recon = torch.zeros(N, N, device=DEV)

    print(f"  [grassmannian] {N*(N-1)//2} pairs (GPU-batched) ...", flush=True)
    _chunk = 8192
    for i in range(N - 1):
        U_i_T   = U_all[i].T.unsqueeze(0)            # (1, r, in_d)
        rest    = U_all[i + 1:]                       # (M, in_d, r)
        M       = rest.shape[0]
        S_i_row = S_all[i].view(1, r, 1)             # (1, r, 1)
        dg_list, dr_list = [], []
        for c0 in range(0, M, _chunk):
            c1 = min(c0 + _chunk, M)
            M_batch = torch.bmm(
                U_i_T.expand(c1 - c0, -1, -1),       # (chunk, r, in_d)
                rest[c0:c1],                          # (chunk, in_d, r)
            )                                         # (chunk, r, r)

            # Grassmannian (skipped when compute_grass=False)
            if compute_grass:
                sigma  = torch.linalg.svdvals(M_batch).clamp(0.0, 1.0)
                angles = torch.arccos(sigma)
                dg     = angles.pow(2).sum(dim=1).sqrt() / (max_d + 1e-12)
                dg_list.append(dg)

            # Cross-reconstruction: D_recon(i,j) = 1 - 0.5*(R_ji + R_ij)
            S_j  = S_all[i + 1 + c0: i + 1 + c1]     # (chunk, r)
            ms_j = (M_batch * S_j.unsqueeze(1)).pow(2).sum(dim=(1, 2))    # (chunk,)
            ms_i = (M_batch * S_i_row).pow(2).sum(dim=(1, 2))             # (chunk,)
            dr   = (1.0 - 0.5 * (ms_j / (S_sq[i + 1 + c0: i + 1 + c1] + 1e-12)
                                 + ms_i / (S_sq[i] + 1e-12))).clamp(0.0, 1.0)
            dr_list.append(dr)

        if compute_grass:
            dg = torch.cat(dg_list)
            D_grass[i, i + 1:] = dg;  D_grass[i + 1:, i] = dg
        dr = torch.cat(dr_list)
        D_recon[i, i + 1:] = dr;  D_recon[i + 1:, i] = dr
        if (i + 1) % 200 == 0 or i == N - 2:
            print(f"    row {i+1}/{N-1}", flush=True)
    return D_grass.cpu().numpy(), D_recon.cpu().numpy()


def cluster_residuals(all_residuals, rank: int, G_moe: int, G_attn: int,
                      seed: int = 42, share_attn: bool = False,
                      hessian_svd: bool = True, recon_weight: float = 0.0,
                      rank_cluster: int = None,
                      rank_attn: int = None, rank_down: int = None,
                      cluster_on_original: bool = False):
    """
    Run Grassmannian SpectralClustering independently for each wtype.

    Only the `cluster_on_original=True` path (activation-scaled ORIGINAL weights)
    is exercised in the current GLoRCQ pipeline, since fill_phase2 groups
    experts BEFORE quantization (no residuals exist yet). The default path
    (Hessian-weighted quantization residuals) is retained for backward
    compatibility but requires `weight_quant` in each record.

    When share_attn=False (default), attention layers (q/k/v/o_proj) skip
    Grassmannian clustering and each record is assigned its own singleton
    group. The current pipeline routes attention through a separate
    non-clustering path (`gptq_attn_4bit`), so this branch is effectively
    dead when called from fill_phase2, but is preserved for API stability.

    Args:
      hessian_svd  : If True and records have 'hessian_diag', weight by
                     H^{1/2} before SVD.
      recon_weight : alpha in [0,1]. Combined distance =
                     (1-alpha)*D_grass + alpha*D_recon.
                     0 = pure Grassmannian (default).
      cluster_on_original : If True, cluster on original FP16 weights.

    Returns:
      assignments   : {wtype: ndarray(N,)} - group_id per record
      wtype_indices : {wtype: list[int]}   - indices into all_residuals
    """
    _rank_c_moe  = rank_cluster if rank_cluster is not None else rank
    _rank_c_down = rank_cluster if rank_cluster is not None else (rank_down if rank_down is not None else rank)
    _rank_c_attn = rank_cluster if rank_cluster is not None else (rank_attn if rank_attn is not None else rank)
    assignments   = {}
    wtype_indices = {}

    for wtype in sorted(_ALL_WTYPES):
        idxs = [i for i, r in enumerate(all_residuals) if r["type"] == wtype]
        if not idxs:
            continue

        N = len(idxs)

        # No-share attention: each record is its own singleton group
        if wtype in _ATTN_WTYPES and not share_attn:
            labels = np.arange(N, dtype=np.int64)
            print(f"\n[Stage 2] {wtype}: N={N}  [no-share-attn] each is own group",
                  flush=True)
            assignments[wtype]   = labels
            wtype_indices[wtype] = idxs
            continue

        G = G_attn if wtype in _ATTN_WTYPES else G_moe
        G = min(G, N)

        print(f"\n[Stage 2] {wtype}: N={N}, G={G}", flush=True)
        subset  = [all_residuals[i] for i in idxs]

        # Build E_T list (in_d x out_d), two modes
        E_T_list = []
        n_weighted = 0
        for r in subset:
            if cluster_on_original:
                E_T = r["weight_orig"].float().T  # (in_d, out_d)
                act_s = r.get("_act_scale_cpu")
                if act_s is not None:
                    E_T = E_T * act_s.float().to(E_T.device).unsqueeze(1)
                    n_weighted += 1
                elif hessian_svd and r.get("hessian_diag") is not None:
                    h_sqrt = r["hessian_diag"].float().sqrt().to(E_T.device)
                    E_T = E_T * h_sqrt.unsqueeze(1)
                    n_weighted += 1
            else:
                # Requires 'weight_quant'; not used by current fill_phase2 path.
                E_T = (r["weight_orig"] - r["weight_quant"]).float().T
                if hessian_svd and r.get("hessian_diag") is not None:
                    h_sqrt = r["hessian_diag"].float().sqrt().to(E_T.device)
                    E_T = E_T * h_sqrt.unsqueeze(1)
                    n_weighted += 1
            E_T_list.append(E_T)
        if n_weighted > 0:
            mode_str = "act_scale/hessian on orig-weight" if cluster_on_original else "hessian on residuals"
            print(f"  [cluster] {mode_str}: {n_weighted}/{N} modules weighted",
                  flush=True)

        # Select clustering rank for this wtype
        if wtype in _ATTN_WTYPES:
            _rank_c = _rank_c_attn
        elif wtype == "down_proj":
            _rank_c = _rank_c_down
        else:
            _rank_c = _rank_c_moe

        D_grass, D_recon = _grassmannian_dist_matrix(
            E_T_list, _rank_c, compute_grass=(recon_weight < 1.0)
        )
        if recon_weight > 0:
            D = (1.0 - recon_weight) * D_grass + recon_weight * D_recon
            print(f"  [recon_weight={recon_weight}] combined D_grass + D_recon",
                  flush=True)
        else:
            D = D_grass

        if G <= 1 or N == 1:
            labels = np.zeros(N, dtype=np.int64)
        elif G >= N:
            labels = np.arange(N, dtype=np.int64)
        else:
            pos_vals = D[D > 0]
            sigma2   = float(pos_vals.mean() ** 2) if len(pos_vals) else 1.0
            K        = np.exp(-D ** 2 / (2 * max(sigma2, 1e-6)))
            np.fill_diagonal(K, 1.0)
            sc = SpectralClustering(
                n_clusters=G, affinity="precomputed",
                random_state=seed, assign_labels="kmeans", n_init=5,
            )
            labels = sc.fit_predict(K).astype(np.int64)

        assignments[wtype]   = labels
        wtype_indices[wtype] = idxs
        counts = np.bincount(labels)
        print(f"  -> group sizes: min={counts.min()}, max={counts.max()}, "
              f"mean={counts.mean():.1f}", flush=True)

    return assignments, wtype_indices
