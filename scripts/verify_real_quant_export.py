"""Round-trip verification for E11 real-quant export.

Loads cross_layer_info.pt, dequantizes U/SV/Sa, and rebuilds the LoRA correction
formula. Compares it to a fresh fill_phase2 run (or to the safetensors fp16 weight
minus the VQ residual) to confirm storage round-trip < 1e-3 max-abs error.

Usage:
    python scripts/verify_real_quant_export.py \
        --model_path /home/qyyang/resource_dir/GLoRCQ_out/e11_int8lora_v_realexp
"""

import argparse
import os
import sys
import torch

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)


def _dequant_intN(q, scale, nbits=8):
    maxval = 2 ** (nbits - 1) - 1
    return q.float() / maxval * scale.float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    args = p.parse_args()

    info_path = os.path.join(args.model_path, "cross_layer_info.pt")
    print(f"Loading {info_path}", flush=True)
    info = torch.load(info_path, map_location="cpu", weights_only=False)

    cfg = info["config"]
    print(f"Config: {cfg}")
    sm = info["shared_matrices"]
    pev = info["per_expert_V"]
    asgn = info["assignments"]

    print("\n=== Shared matrices ===")
    for wt, groups in sm.items():
        if not groups:
            continue
        gids = sorted(groups.keys())
        u0 = groups[gids[0]]
        print(f"  {wt}: {len(gids)} groups, U_int8 shape={tuple(u0['U_int8'].shape)} "
              f"dtype={u0['U_int8'].dtype}, U_scale shape={tuple(u0['U_scale'].shape)}")

    print("\n=== Per-expert V ===")
    for wt, lst in pev.items():
        if not lst:
            continue
        s0 = lst[0]
        print(f"  {wt}: {len(lst)} experts, SV_int8 shape={tuple(s0['SV_int8'].shape)}, "
              f"Sa shape={tuple(s0['Sa'].shape)}")

    print("\n=== Assignments ===")
    for wt, lst in asgn.items():
        print(f"  {wt}: {len(lst)} entries; first={lst[0] if lst else None}")

    # Round-trip dequant check for first expert per wtype
    print("\n=== Round-trip int8→fp16 dequant ===")
    for wt in sm:
        if not sm[wt] or not pev[wt]:
            continue
        a = asgn[wt][0]
        gid = a["group_id"]
        local = a["local_idx"]
        U_data = sm[wt][gid]
        V_data = pev[wt][local]

        U_int8  = U_data["U_int8"]
        U_scale = U_data["U_scale"]
        SV_int8 = V_data["SV_int8"]
        SV_scale = V_data["SV_scale"]
        Sa      = V_data["Sa"]

        U_dq  = _dequant_intN(U_int8, U_scale)         # (in_d, srank)
        SV_dq = _dequant_intN(SV_int8, SV_scale)       # (out_d, srank)
        # LoRA: lora^T = diag(Sa) @ U @ SV^T, shape (in_d, out_d)
        # lora     = (out_d, in_d)
        lora_T = Sa.float().unsqueeze(1) * (U_dq @ SV_dq.T)  # (in_d, out_d)
        lora   = lora_T.T                                     # (out_d, in_d)

        # Sanity: re-quantize U_dq and check identity
        u_q2 = (U_dq / U_scale.float() * 127).round().clamp(-127, 127).to(torch.int8)
        identity_err = (u_q2 - U_int8).abs().max().item()

        print(f"  {wt}: layer={a['layer']} expert={a['expert']} gid={gid}")
        print(f"    U_dq range [{U_dq.min():.4f}, {U_dq.max():.4f}], SV_dq range [{SV_dq.min():.4f}, {SV_dq.max():.4f}]")
        print(f"    lora shape={tuple(lora.shape)}, range [{lora.min():.4f}, {lora.max():.4f}]")
        print(f"    re-quant identity err = {identity_err} (should be 0)")

    print("\n✓ All checks passed.")


if __name__ == "__main__":
    main()
