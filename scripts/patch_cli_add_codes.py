"""Patch cross_layer_info.pt: derive codes (uint8) from Q_rotated + centroids.

The quantization process assigned each vdim-vector of Q_rotated to a centroid;
the codebook+codes are saved, but currently only Q_rotated is in cross_layer_info.
We can recover the exact codes by nearest-centroid argmin:

    codes[n, k] = argmin_c ||Q_rot[n, k*vdim:(k+1)*vdim] - centroids[cb_id(k), c, :]||²

Since Q_rotated was set to a centroid value during quantization, this argmin
returns the true index (modulo tie-breaking on the rare case of duplicate centroids).

Stores codes back in vq_residuals[wt][local_idx]['codes'].

Usage:
    python scripts/patch_cli_add_codes.py
"""

import sys, os, torch
sys.path.insert(0, '/home/qyyang/repo/GLoRCQ')


def derive_codes(Q_rot, centroids, vdim):
    """
    Q_rot: (out_d, in_d) fp16 in rotated col space
    centroids: (n_codebooks, K, vdim) fp16
    returns codes (out_d, in_d/vdim) uint8 — assumes K <= 256.
    """
    out_d, in_d = Q_rot.shape
    n_cb, K, vd = centroids.shape
    assert vd == vdim
    assert in_d % vdim == 0
    codes_per_row = in_d // vdim
    assert codes_per_row % n_cb == 0, f"codes_per_row={codes_per_row} not divisible by n_cb={n_cb}"
    codes_per_cb = codes_per_row // n_cb

    # Reshape Q to (out_d, codes_per_row, vdim)
    Q_v = Q_rot.float().reshape(out_d, codes_per_row, vdim)
    codes = torch.empty(out_d, codes_per_row, dtype=torch.uint8, device=Q_rot.device)
    for cb_idx in range(n_cb):
        c0 = cb_idx * codes_per_cb
        c1 = c0 + codes_per_cb
        cb = centroids[cb_idx].float()                     # (K, vdim)
        # Distances: (out_d, codes_per_cb, K) = ||Q[:, c0:c1, None, :] - cb[None, None, :, :]||²
        # Memory-friendly chunking: do per-row group
        Q_slice = Q_v[:, c0:c1, :]                          # (out_d, codes_per_cb, vdim)
        # Broadcast distance:
        diff = Q_slice.unsqueeze(2) - cb.unsqueeze(0).unsqueeze(0)   # (out_d, cpc, K, vdim)
        dist = (diff * diff).sum(dim=-1)                              # (out_d, cpc, K)
        codes[:, c0:c1] = dist.argmin(dim=-1).to(torch.uint8)
    return codes


def main():
    model_path = '/home/qyyang/resource_dir/GLoRCQ_out/e11_vq4_gs_hwl'
    cli_path = os.path.join(model_path, 'cross_layer_info.pt')
    print(f'Loading {cli_path}', flush=True)
    info = torch.load(cli_path, map_location='cpu', weights_only=False)
    vqr = info['vq_residuals']

    n_patched = 0
    max_err_global = 0.0
    for wt in vqr:
        for vq in vqr[wt]:
            if vq is None:
                continue
            Q_rot = vq['Q_rotated'].cuda()
            centroids = vq['centroids'].cuda()
            vdim = vq['vdim']
            codes = derive_codes(Q_rot, centroids, vdim)

            # Verify: dequantize via codes and compare to Q_rot
            out_d, in_d = Q_rot.shape
            n_cb, K, _ = centroids.shape
            codes_per_row = in_d // vdim
            codes_per_cb = codes_per_row // n_cb
            Q_recon = torch.empty_like(Q_rot)
            for cb_idx in range(n_cb):
                c0 = cb_idx * codes_per_cb
                c1 = c0 + codes_per_cb
                cb = centroids[cb_idx]                           # (K, vdim)
                codes_blk = codes[:, c0:c1].long()                # (out_d, codes_per_cb)
                looked = cb[codes_blk]                            # (out_d, codes_per_cb, vdim)
                Q_recon[:, c0*vdim:c1*vdim] = looked.reshape(out_d, codes_per_cb * vdim)
            err = (Q_rot.float() - Q_recon.float()).abs().max().item()
            if err > max_err_global:
                max_err_global = err

            vq['codes'] = codes.cpu()
            n_patched += 1
            if n_patched % 200 == 0:
                print(f'  patched {n_patched}, max_err so far = {max_err_global:.6f}', flush=True)

    print(f'\nSaving {cli_path} with {n_patched} codes tensors.', flush=True)
    print(f'Max reconstruction error (Q_recon vs Q_rot): {max_err_global:.6f}', flush=True)
    torch.save(info, cli_path)
    print('Done.')


if __name__ == '__main__':
    main()
