import torch
import torch.nn as nn
import argparse
import os
import json
from utils.qlayer_name_utils import *
from utils.moe_utils import *
from quantizer.quantizer import *

from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer, MixtralRMSNorm
from utils.hadamard_utils import *
from utils.qlayer_name_utils import *
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Any

SCALE_CLAMP_MIN = 1e-4


@torch.no_grad()
def get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)



def place_blocks_optimized_uv_coherence(
    B_map: Dict[Any, torch.Tensor],
    M: int,
    N: int,
    s: int = 16,
    device: torch.device = None
) -> Tuple[torch.Tensor, Dict[Any, Tuple[int, int]]]:
    if not B_map:
        raise ValueError("B_map is empty")
    
    keys = list(B_map.keys())
    B_list = [B_map[k] for k in keys]
    k = len(B_list)
    p, q = B_list[0].shape

    if device is None:
        device = B_list[0].device

    # --- Step 1: Extract top-s SVD features ---
    #B_cpu = [B.detach().cpu() for B in B_list]
    U_features = []
    V_features = []

    for B in B_list:
        normB = torch.norm(B).item()
        if normB < 1e-12:
            U_s = torch.zeros(p, s)
            V_s = torch.zeros(q, s)
        else:
            min_dim = min(p, q)
            s_use = min(s, min_dim)

            W2, U, Vh, S,max_0,max_now,srank = get_best_sketch_fp16_ret(B, 16, fix_rank = s_use)
            # try:
            #     U, S, Vh = torch.linalg.svd(B, full_matrices=False)
            # except Exception:
            #     U, S, Vh = torch.svd_lowrank(B, q=s_use)

            U_s = [tensor.to(torch.float16) for tensor in U]#= U[:, :s_use]
            V_s = [tensor.to(torch.float16) for tensor in Vh]#= Vh[:s_use, :].T
            U_s = torch.vstack(U_s[:srank])
            V_s = torch.vstack(V_s[:srank])
            V_s = V_s.T
            if s_use < s:
                U_s = torch.cat([U_s, torch.zeros(p, s - s_use)], dim=1)
                V_s = torch.cat([V_s, torch.zeros(q, s - s_use)], dim=1)

        U_features.append(U_s.cpu().reshape(-1).numpy())
        V_features.append(V_s.cpu().reshape(-1).numpy())

    U_features = np.array(U_features)
    V_features = np.array(V_features)

    # Normalize
    def normalize(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return X / norms

    U_features = normalize(U_features)
    V_features = normalize(V_features)

    # --- Step 2: Choose R, C ---
    R, C = M, N

    # --- Step 3: Cluster U and V ---
    if R == 1:
        row_groups = np.zeros(k, dtype=int)
    else:
        kmeans_u = KMeans(n_clusters=R, random_state=0, n_init=10)
        row_groups = kmeans_u.fit_predict(U_features)

    if C == 1:
        col_groups = np.zeros(k, dtype=int)
    else:
        kmeans_v = KMeans(n_clusters=C, random_state=0, n_init=10)
        col_groups = kmeans_v.fit_predict(V_features)

    # --- Step 4: Group blocks by (row_group, col_group) ---
    group_to_blocks = defaultdict(list)
    for idx, key in enumerate(keys):
        rg = row_groups[idx]
        cg = col_groups[idx]
        group_to_blocks[(rg, cg)].append(key)

    # --- Step 5: Map ideal (rg, cg) to actual (r, c) in MxN grid ---
    # We assume R <= M and C <= N, so we can embed directly.
    # If not, we compress via modulo or clustering again (not needed here).

    A_big = torch.zeros(M * p, N * q, device=device, dtype=B_list[0].dtype)
    used = torch.zeros(M, N, dtype=torch.bool)
    placement = {}

    # We'll assign each ideal group to a base position (rg, cg)
    # Then assign its blocks to nearby free cells in that "region"
    for (rg, cg), block_keys in group_to_blocks.items():
        # Base actual position
        base_r = min(rg, M - 1)
        base_c = min(cg, N - 1)

        # Collect all free positions in the grid (could precompute, but k is small)
        # We'll search in expanding window around (base_r, base_c)
        assigned = 0
        max_radius = max(M, N)
        for radius in range(max_radius + 1):
            if assigned >= len(block_keys):
                break
            # Spiral or ring search: iterate over offsets with |dr|+|dc| == radius
            for dr in range(-radius, radius + 1):
                for dc in [-(radius - abs(dr)), radius - abs(dr)] if radius - abs(dr) != 0 else [0]:
                    r = (base_r + dr) % M
                    c = (base_c + dc) % N
                    if not used[r, c] and assigned < len(block_keys):
                        key = block_keys[assigned]
                        i0, j0 = r * p, c * q
                        A_big[i0:i0 + p, j0:j0 + q] = B_map[key].to(device=device)
                        used[r, c] = True
                        placement[key] = (r, c)
                        assigned += 1
                        if assigned >= len(block_keys):
                            break
                if assigned >= len(block_keys):
                    break

        if assigned < len(block_keys):
            # Fallback: fill any remaining free slots
            for r in range(M):
                for c in range(N):
                    if assigned >= len(block_keys):
                        break
                    if not used[r, c]:
                        key = block_keys[assigned]
                        i0, j0 = r * p, c * q
                        A_big[i0:i0 + p, j0:j0 + q] = B_map[key].to(device=device)
                        used[r, c] = True
                        placement[key] = (r, c)
                        assigned += 1
                if assigned >= len(block_keys):
                    break

    return A_big, placement


@torch.no_grad()
def get_experts_2Dlora(weightsArr, input_featArr, w_bit, group_size, fix_rank, ratio, quant_infos,  wIndex={"exp0": [0,0]} ,row = 4,  max_clip=0.5, Q = None, lora_bit = 16, loratool="sketch", lora_iter = 8, name = ""):
    col = math.ceil(len(wIndex)/row)
    first_key, first_value = next(iter(weightsArr.items()))
    scaleARR = {}
    big_tensor = torch.zeros(row * first_value.shape[1], col * first_value.shape[0], dtype=torch.float64, device=first_value.device)

    for name in weightsArr:
        weight = weightsArr[name]
        input_feat = input_featArr[name]
        minrank = min(weight.shape[0], weight.shape[1])
        fix_rank = min(fix_rank, minrank)

        def get_proper_scale(input_feat, scale_time, dev):
            mean_feat = input_feat.abs().view(-1, input_feat.shape[-1]).mean(0)
            mean_feat = mean_feat.pow(scale_time).to(dev)
            mean_feat[torch.isinf(mean_feat)] = torch.finfo(mean_feat.dtype).max/2
            if mean_feat.dtype==torch.float16:
                mean_feat = mean_feat.clamp(min=SCALE_CLAMP_MIN)
            if mean_feat.dtype==torch.bfloat16:
                mean_feat = mean_feat.clamp(min=1e-14)
            else:
                mean_feat = mean_feat.clamp(min=1e-15)
            scales = mean_feat / ((mean_feat.max().float() * mean_feat.min().float()).sqrt()).to(mean_feat.dtype)
            return scales

        max_attempts = 10
        attempt = 0
        scale_time = 2.4
        scales = None

        while attempt < max_attempts:
            scales = get_proper_scale(input_feat, scale_time, first_value.device)
            ratio = scales.max() / scales.min()

            if torch.any(torch.isnan(scales)):
                continue
            if ratio < 1e5:
                break
            scale_time = max(0.5, scale_time - 0.2)
            attempt += 1

        if input_feat.numel() == 0 or torch.any(torch.isnan(scales)):
            scales = torch.ones(( weight.shape[1]), device=weight.device,dtype=weight.dtype)
        Sa = (torch.tensor(1.0, dtype=torch.float32)/scales.float())
        W_scale_T = torch.diag(scales.to(torch.float64)).cuda() @ weight.T.to(torch.float64)
        
        weightsArr[name] = W_scale_T
        scaleARR[name] = Sa
    del input_feat,  scales
    gc.collect()


    big_tensor, placement = place_blocks_optimized_uv_coherence(
        weightsArr, M=row, N=col, s=fix_rank//2, device='cuda'
    )


    W2,U_trunc,Vh_trunc,S_trunc,max_0,max_now,srank = get_best_sketch_fp8_ret(big_tensor, w_bit, ratio = ratio, fix_rank = fix_rank, max_sketch_iter = lora_iter)
    U_trunc = [tensor.to(torch.float16) for tensor in U_trunc]
    Vh_trunc = [tensor.to(torch.float16) for tensor in Vh_trunc]

    U_trunc = torch.vstack(U_trunc[:srank]).to(first_value.device)
    Vh_trunc = torch.vstack(Vh_trunc[:srank]).to(first_value.device)
    S_trunc = torch.tensor(S_trunc).to(first_value.device)
    del big_tensor
    gc.collect()
    torch.cuda.empty_cache()


    quant_infos["lora_rank"] = quant_infos["lora_rank"] + srank * len(wIndex)/(col+row)
    quant_infos["lora_size"] =  quant_infos["lora_size"] + srank * (row*first_value.size(0) + col*first_value.size(1))*lora_bit
    quant_infos["total_size"] = quant_infos["total_size"] + len(wIndex) * first_value.size(0) * first_value.size(1)*16
    quant_infos["quant_size"] = quant_infos["quant_size"] + len(wIndex) * first_value.size(0) * first_value.size(1)*w_bit
    quant_infos["layer_cnt"] = quant_infos["layer_cnt"] + len(wIndex)

    return U_trunc, S_trunc, Vh_trunc,scaleARR, placement
    

@torch.no_grad()
def get_normal_lora(layer, input_feat, w_bit, group_size, fix_rank, ratio, quant_infos,max_clip=0.5, Q = None, lora_bit = 16, loratool="sketch", lora_iter = 8, name = ""):
    minrank = min(layer.weight.shape[0],layer.weight.shape[1])
    fix_rank = min(fix_rank, minrank)
    mean_feat = input_feat.abs().view(-1, input_feat.shape[-1]).mean(0)

    scale_time = 2.4
    mean_feat = mean_feat.pow(scale_time)
    mean_feat[torch.isinf(mean_feat)] = torch.finfo(mean_feat.dtype).max/2
    if mean_feat.dtype==torch.float16:
        mean_feat = mean_feat.clamp(min=SCALE_CLAMP_MIN)
    if mean_feat.dtype==torch.bfloat16:
        mean_feat = mean_feat.clamp(min=1e-14)
    else:
        mean_feat = mean_feat.clamp(min=1e-15)
    scales = mean_feat / ((mean_feat.max().float() * mean_feat.min().float()).sqrt()).to(mean_feat.dtype)
    if input_feat.numel() == 0:
        scales = torch.ones(( layer.weight.shape[1]), device=layer.weight.device,dtype=layer.weight.dtype)

    input_feat = input_feat.cuda()

    if loratool == "svd":
        lora_W_struct,srank = sketch_pre_svd_split(layer.weight, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size, lora_bit = lora_bit)
    else:
        lora_W_struct,srank = sketch_pre_split(layer.weight, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size, lora_bit = lora_bit, lora_iter = lora_iter)


    quant_infos["lora_rank"] = quant_infos["lora_rank"] + srank
    quant_infos["lora_size"] =  quant_infos["lora_size"] + srank * (layer.weight.size(0) + layer.weight.size(1))*lora_bit
    quant_infos["total_size"] = quant_infos["total_size"] + layer.weight.size(0) * layer.weight.size(1)*16
    quant_infos["quant_size"] = quant_infos["quant_size"] + layer.weight.size(0) * layer.weight.size(1)*w_bit
    quant_infos["layer_cnt"] = quant_infos["layer_cnt"] + 1
    
    del input_feat, mean_feat, scales
    

    return lora_W_struct


@torch.no_grad()
def get_layers_2Dlora(layer, input_feat, quant_infos, w_bit=4, fix_rank = 0, ratio = 0.1, group_size = 128, index = 1, Q = None, lora_bit = 16,loratool = "sketch",lora_iter = 8 ,row = 8):
    

    def get_module(root_module, module_path):
        current = root_module
        for part in module_path.split("."):
            current = getattr(current, part)
        return current
    W_R_layer = {}
    subset = find_layers(layer)
    normal_qlist, shared_qlist, regular_qlist = get_moe_qlayers_name(subset)
    row_len = row
    qlist = ["gate", "up", "down"]
    if isinstance(layer, (MixtralDecoderLayer)):
        normal_qlist = MixtralQuantLayerA[0]
        shared_qlist = []
        regular_qlist = MixtralQuantLayerA[1]
        qlist = ["w1", "w2", "w3"]
        row_len = 4
    

    for name in normal_qlist:
        if name in input_feat:
            module = get_module(layer, name)
            lora_stu= get_normal_lora(module , input_feat[name] , w_bit, group_size, fix_rank, ratio, quant_infos, lora_bit = lora_bit, loratool =loratool, lora_iter = lora_iter)
            W_R_layer[name] = lora_stu              
        else:
            print(f"Warning: {name} not found in qkvo input_feat")
    for name in shared_qlist:
        if name in input_feat:
            module = get_module(layer, name)
            lora_stu= get_normal_lora(module , input_feat[name] , w_bit, group_size, fix_rank, ratio, quant_infos, lora_bit = lora_bit, loratool =loratool, lora_iter = lora_iter, name = name)
            W_R_layer[name] = lora_stu
        else:
            print(f"Warning: {name} not found in share input_feat")

    for qn in qlist:
        i = 0
        weightArr = {}
        featArr = {}
        indexARR={}
        for name in regular_qlist:
            if name not in input_feat:
                module = get_module(layer, name)
                input_feat[name] = torch.ones(( 2, module.weight.shape[1]), device=module.weight.device,dtype=module.weight.dtype)

            if name in input_feat and qn in name:
                module = get_module(layer, name)


                weightArr[name] = module.weight.data.detach()
                featArr[name] = input_feat[name]
                
                row = i//row_len
                col = i%row_len
                indexARR[name] = (row, col)
                i = i + 1
        if weightArr=={}:
            continue
        U_all,S_all,V_all,ScaleAll,new_placement = get_experts_2Dlora(weightArr, featArr, w_bit, group_size, fix_rank, ratio, quant_infos,  row = row_len, wIndex = indexARR ,lora_bit = lora_bit, loratool = loratool, lora_iter = lora_iter)
        i = 0
        for name in regular_qlist:
            if name in input_feat and qn in name:
                r,c = new_placement[name]
                module = get_module(layer, name)
                W_R_layer[name] = {}
                W_R_layer[name]["U"] = U_all[:,r*module.weight.shape[1]:(r+1)*module.weight.shape[1]].T
                W_R_layer[name]["V"] = V_all[:,c*module.weight.shape[0]:(c+1)*module.weight.shape[0]]
                W_R_layer[name]["Si"] = S_all.to(module.weight.dtype)
                W_R_layer[name]["Sa"] = ScaleAll[name].to(module.weight.dtype)
                i = i + 1

                lora = (W_R_layer[name]["U"].to(module.weight.dtype) @ torch.diag(W_R_layer[name]["Si"])@ W_R_layer[name]["V"].to(module.weight.dtype))
                lora = (torch.diag(W_R_layer[name]["Sa"]) @ lora).T
                if (module.weight - lora).abs().max() > 60:
                    W_R_layer[name]["U"] = None
                    print(name, "Error too large!", ((module.weight - lora).abs()>60).sum().item(), ((module.weight - lora).abs()>10).sum().item() , ((module.weight - lora).abs()>5).sum().item(), W_R_layer[name]["Sa"].max(), W_R_layer[name]["Sa"].min(),)
        del U_all,S_all,V_all, lora
    gc.collect()
    torch.cuda.empty_cache()
    return layer, W_R_layer
