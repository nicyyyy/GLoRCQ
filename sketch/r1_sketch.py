import torch
import torch.nn as nn
import argparse
import os
import json
import math

from numpy import random



def find_max_abs_value(A):
    abs_A = torch.abs(A)
    max_abs_value = torch.max(abs_A)
    return max_abs_value

def compute_r1sketch_fp8_ret(A,iter = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, n = A.shape
    x_numpy = random.normal(loc=0, scale=1, size=(n))
    x = torch.from_numpy(x_numpy)
    x = x.to(torch.float64)
    x = x.to(device)
    y = torch.matmul(A, x)
    
    for i in range(iter):
        tmp = torch.matmul(A.T, y)
        y = torch.matmul(A, tmp)
    A_L = y
    A_R = torch.matmul(A.T, A_L)
    normP = torch.norm(A_L, p=2)
    normQ = torch.norm(A_R, p=2)
    S = normQ/normP
    Var_AL = 1.0/normP
    Var_AR = 1.0/normQ
    A_R = A_R*Var_AR
    A_L = A_L*Var_AL
    A_L = A_L.to(torch.float8_e5m2)
    A_R = A_R.to(torch.float8_e5m2)
    return A_L, A_R, S

def compute_r1sketch_fp16_ret(A,iter = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, n = A.shape
    x_numpy = random.normal(loc=0, scale=1, size=(n))
    x = torch.from_numpy(x_numpy)
    x = x.to(torch.float64)
    x = x.to(device)
    y = torch.matmul(A, x)

    for i in range(iter):
        tmp = torch.matmul(A.T, y)
        y = torch.matmul(A, tmp)
    A_L = y
    A_R = torch.matmul(A.T, A_L)
    normP = torch.norm(A_L, p=2)
    normQ = torch.norm(A_R, p=2)

    # Zero-norm guard: residual matrix is effectively zero; return zero vectors
    if normP < 1e-10 or normQ < 1e-10:
        A_L = torch.zeros(m, dtype=torch.float64, device=device)
        A_R = torch.zeros(n, dtype=torch.float64, device=device)
        S   = torch.tensor(0.0, dtype=torch.float64, device=device)
        return A_L, A_R, S

    S = normQ/normP
    Var_AL = 1.0/normP
    Var_AR = 1.0/normQ
    A_R = A_R*Var_AR
    A_L = A_L*Var_AL
    return A_L, A_R, S


def get_best_sketch_fp8_ret(weights, bits, ratio=0.01, max_sketch_iter = 8, fix_rank = 0):
    row = weights.size(0)
    col = weights.size(1)
    min_rank = min(row,col)
    weight_cp = weights
    if weights.dtype == torch.float16:
        weights = weights.to(torch.float64)

    skethc_L = []
    skethc_R = []
    max_iter = {}
    max_absW_0 = 0
    VS_L = None
    VS_R = None
    VS_L_16 = None
    VS_R_16 = None
    work_rank = 0
    S_arr = []
    work_rank = fix_rank
    for i in range(0,work_rank):
        r1_L,r1_R,S = compute_r1sketch_fp8_ret(weights,max_sketch_iter)
        r1_L_FP16 = r1_L.to(torch.float16)
        r1_R_FP16 = r1_R.to(torch.float16)
        r1_L_FP16 = r1_L_FP16* S
        r1_matrix = torch.outer(r1_L_FP16, r1_R_FP16)
        weights = weights - r1_matrix
        skethc_L.append(r1_L)
        skethc_R.append(r1_R)
        S_arr.append(S.item())
    VS_L = skethc_L[:work_rank]
    VS_R = skethc_R[:work_rank]
    S_arr = S_arr[:work_rank]

    if work_rank!=0:
        weight_cp = weight_cp
        max_now = find_max_abs_value(weight_cp)
    return weight_cp,VS_L,VS_R,S_arr,max_absW_0,max_now,work_rank


def get_best_sketch_fp16_ret(weights, bits, ratio=0.01, max_sketch_iter = 8, fix_rank = 0):
    row = weights.size(0)
    col = weights.size(1)
    min_rank = min(row,col)
    weight_cp = weights
    if weights.dtype != torch.float64:
        weights = weights.to(torch.float64)
    max_absW_0 = 0
    skethc_L = []
    skethc_R = []
    max_iter = {}

    VS_L = None
    VS_R = None
    VS_L_16 = None
    VS_R_16 = None
    work_rank = 0
    S_arr = []
    work_rank = fix_rank
    # BUG 2 fix: initialise max_now before the loop so it is always defined
    max_now = find_max_abs_value(weight_cp)
    for i in range(0,work_rank):
        r1_L,r1_R,S = compute_r1sketch_fp16_ret(weights,max_sketch_iter)

        # BUG 1/3 fix: skip degenerate rank-1 term (singular value is zero)
        if S.item() == 0.0:
            skethc_L.append(r1_L)
            skethc_R.append(r1_R)
            S_arr.append(0.0)
            continue

        # BUG 3 fix: clamp S to fp16 range before converting to avoid inf overflow
        S_clamped = S.clamp(max=6e4)
        r1_L_FP16 = r1_L.to(torch.float16)
        r1_R_FP16 = r1_R.to(torch.float16)
        r1_L_FP16 = r1_L_FP16 * S_clamped.to(torch.float16)
        r1_matrix = torch.outer(r1_L_FP16, r1_R_FP16)

        # BUG 3 fix: if r1_matrix still contains non-finite values, skip deflation
        if not torch.isfinite(r1_matrix).all():
            skethc_L.append(r1_L)
            skethc_R.append(r1_R)
            S_arr.append(0.0)
            continue

        weights = weights - r1_matrix
        skethc_L.append(r1_L)
        skethc_R.append(r1_R)
        S_arr.append(S.item())
    VS_L = skethc_L[:work_rank]
    VS_R = skethc_R[:work_rank]
    S_arr = S_arr[:work_rank]

    if work_rank != 0:
        max_now = find_max_abs_value(weight_cp)
    return weight_cp,VS_L,VS_R,S_arr,max_absW_0,max_now,work_rank