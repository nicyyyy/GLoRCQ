"""Patch existing cross_layer_info.pt to add gptvq_Q_rotated derived from fp16 weights.

For E11 vq4 inference. The fp16 fake-quant model has W_q = Q @ PD_rot.T + lora,
so Q = (W_q - lora) @ PD_rot.
"""

import sys, os, torch
sys.path.insert(0, '/home/qyyang/repo/GLoRCQ')
from transformers import AutoModelForCausalLM
from utils.hadamard_utils import (
    block_diagonal_walsh_matrix,
    create_diagI_matrix_upper,
    construct_partial_permutation_matrix_upper,
)


def _dequant_intN(q, scale, nbits=8):
    maxval = 2 ** (nbits - 1) - 1
    return q.float() / maxval * scale.float()


def main():
    model_path = '/home/qyyang/resource_dir/GLoRCQ_out/e11_vq4_gs_hwl'
    cli_path = os.path.join(model_path, 'cross_layer_info.pt')
    print(f'Loading model from {model_path}', flush=True)
    m = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                              torch_dtype=torch.float16).cuda()
    print(f'Loading {cli_path}', flush=True)
    info = torch.load(cli_path, map_location='cpu', weights_only=False)

    layers = m.model.layers
    vqr = info['vq_residuals']
    asgn = info['assignments']
    sm = info['shared_matrices']
    pev = info['per_expert_V']

    n_patched = 0
    for wt in vqr:
        for entry, vq in zip(asgn[wt], vqr[wt]):
            if vq is None:
                continue
            li, ei, local = entry['layer'], entry['expert'], entry['local_idx']
            gid = entry['group_id']

            lin = getattr(layers[li].mlp.experts[ei], wt)
            W = lin.weight.data.cuda().float()  # (out_d, in_d)

            # LoRA
            U_dq = _dequant_intN(sm[wt][gid]['U_int8'].cuda(),
                                 sm[wt][gid]['U_scale'].cuda())
            SV_dq = _dequant_intN(pev[wt][local]['SV_int8'].cuda(),
                                  pev[wt][local]['SV_scale'].cuda())
            Sa = pev[wt][local]['Sa'].cuda().float()
            lora_T = Sa.unsqueeze(1) * (U_dq @ SV_dq.T)  # (in_d, out_d)
            lora = lora_T.T                               # (out_d, in_d)

            # Reconstruct PD_rot from saved perm + diag_signs (deterministic)
            in_d = vq['in_d']
            rs = int(vq.get('rotate_size', 256))
            while rs > 1 and in_d % rs != 0:
                rs //= 2
            ps = int(vq.get('partial_size', 256))
            rot = block_diagonal_walsh_matrix(in_d, rs, 'cuda').to(torch.float16)
            diagI = create_diagI_matrix_upper(rot, rot.shape[0] - ps).cuda().to(torch.float16)
            Pperm = construct_partial_permutation_matrix_upper(
                vq['perm'], m=in_d, dtype=torch.float16
            ).cuda()
            PD_rot = (Pperm @ diagI).to(torch.float16)

            # Q = (W - lora) @ PD_rot
            res = (W - lora).half()                       # (out_d, in_d)
            Q = res @ PD_rot                              # (out_d, in_d)
            vq['Q_rotated'] = Q.cpu()
            n_patched += 1
            if n_patched % 100 == 0:
                print(f'  patched {n_patched} ...', flush=True)

    print(f'Saving {cli_path} with {n_patched} Q_rotated tensors ...', flush=True)
    torch.save(info, cli_path)
    print('Done.')


if __name__ == '__main__':
    main()
