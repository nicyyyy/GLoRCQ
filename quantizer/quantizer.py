import torch
import torch.nn as nn
import argparse
import os
import json
import numpy as np
import gc
from sketch.r1_sketch import *


import math
import transformers
import torch
from utils import hadamard_utils
import fast_hadamard_transform


qtype = torch.float16


def sketch_pre_svd_split(W, feat_scale, bit=4, fix_rank=0, ratio=0.1, groupsize=128, max_sketch_iter=4, lora_bit = 16):
    dtype = W.dtype
    device = W.device 

    feat_scale = feat_scale.to(device)
    
    W_scale_T = torch.diag(feat_scale) @ W.T
    W_scale_T = W_scale_T.to(torch.float64)

    U, S, Vh = torch.linalg.svd(W_scale_T, full_matrices=False)

    srank = fix_rank

    U_trunc = U[:, :srank].detach().cpu()
    S_trunc = S[:srank].detach().cpu()
    Vh_trunc = Vh[:srank, :].detach().cpu()

    Sa = (torch.tensor(1.0, dtype=torch.float32)/feat_scale.float())

    if lora_bit == 8:
        lora_struct = {
            "U": U_trunc.cuda().to(torch.float8_e4m3fn),
            "V": Vh_trunc.cuda().to(torch.float8_e4m3fn),
            "Si": S_trunc.cuda().to(dtype),
            "Sa": Sa.cuda().to(dtype)
        }
    else:
        lora_struct = {
            "U": U_trunc.cuda().to(dtype),
            "V": Vh_trunc.cuda().to(dtype),
            "Si": S_trunc.cuda().to(dtype),
            "Sa": Sa.cuda().to(dtype)
        }        

    del W_scale_T, U, S, Vh, feat_scale
    del U_trunc, S_trunc, Vh_trunc, Sa

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return lora_struct, srank



def sketch_pre_split(W, feat_scale, bit=4, fix_rank=0, ratio=0.1, groupsize=128, max_sketch_iter=4, lora_bit = 16, lora_iter = 8):
    dtype = W.dtype
    device = W.device

    feat_scale = feat_scale.to(device)
    
    W_scale_T = torch.diag(feat_scale) @ W.T
    W_scale_T = W_scale_T.to(torch.float64)

    W2,U_trunc,Vh_trunc,S_trunc,max_0,max_now,srank = get_best_sketch_fp8_ret(W_scale_T, bit, ratio = ratio, fix_rank = fix_rank, max_sketch_iter = lora_iter)


    U_trunc = [tensor.to(torch.float16) for tensor in U_trunc]
    Vh_trunc = [tensor.to(torch.float16) for tensor in Vh_trunc]

    U_trunc = torch.vstack(U_trunc[:srank])
    Vh_trunc = torch.vstack(Vh_trunc[:srank])
    S_trunc = torch.tensor(S_trunc).cuda()
    Sa = (torch.tensor(1.0, dtype=torch.float32)/feat_scale.float())

    if lora_bit == 8:
        lora_struct = {
            "U": U_trunc.cuda().T,
            "V": Vh_trunc.cuda(),
            "Si": S_trunc.cuda().to(dtype),
            "Sa": Sa.cuda().to(dtype)
        }
    else:
        lora_struct = {
            "U": U_trunc.cuda().to(dtype).T,
            "V": Vh_trunc.cuda().to(dtype),
            "Si": S_trunc.cuda().to(dtype),
            "Sa": Sa.cuda().to(dtype)
        }        

    del W_scale_T,  feat_scale
    del U_trunc, S_trunc, Vh_trunc, Sa

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return lora_struct, srank



def get_minq_maxq(bits, sym):
    if sym:
        maxq = torch.tensor(2**(bits-1)-1)
        minq = -maxq -1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0

    return minq, maxq

def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero

def asym_dequant(q, scale, zero):
    return scale * (q - zero)

def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))

def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq+1), maxq)
    return q, scale
def sym_dequant(q, scale):
    return scale * q

def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))


def two_compl(x, bits: int):
    return torch.where(x < 0, 2 ** bits + x, x)


class WeightQuantizer(torch.nn.Module):
    '''From GPTQ Repo'''

    def __init__(self, shape=1):
        super(WeightQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.9,
    ):
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if sym:
            self.maxq = torch.tensor(2**(bits-1)-1)
        else:
            self.maxq = torch.tensor(2**bits - 1)

    def find_params(self, x):
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:

                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:

            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return
    def quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if self.sym:
                return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
            return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)

class BiWeightQuantizer(torch.nn.Module):
    '''Binary weight quantizer: maps weights to {+scale, -scale} (sym) or {alpha, beta} (asym).
    Interface is kept consistent with WeightQuantizer.'''

    def __init__(self, shape=1):
        super(BiWeightQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(1))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.9,
    ):
        # bits is ignored: binary quantization is always 1-bit
        self.bits = 1
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.maxq = torch.tensor(1)

    def find_params(self, x):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)        # (rows, cols)
        else:
            x = x.flatten().unsqueeze(0)  # (1, all)

        if self.sym:
            # scale = mean(|x|) per row; zero = 0
            # quantize: scale * sign(x)
            self.scale = x.abs().mean(dim=1).clamp(min=1e-5)
            self.zero = torch.zeros_like(self.scale)
        else:
            # Two cluster means: positive cluster (x >= 0) and negative cluster (x < 0)
            # scale = (pos_mean - neg_mean) / 2, zero = midpoint = (pos_mean + neg_mean) / 2
            # quantize: zero + scale * sign(x - zero)
            pos_sum = x.clamp(min=0).sum(dim=1)
            neg_sum = x.clamp(max=0).sum(dim=1)
            pos_cnt = (x >= 0).sum(dim=1).clamp(min=1)
            neg_cnt = (x < 0).sum(dim=1).clamp(min=1)
            pos_mean = pos_sum / pos_cnt
            neg_mean = neg_sum / neg_cnt
            self.scale = ((pos_mean - neg_mean) / 2).clamp(min=1e-5)
            self.zero = (pos_mean + neg_mean) / 2

        if not self.perchannel:
            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape_bc = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape_bc)
        self.zero = self.zero.reshape(shape_bc)

    def quantize(self, x):
        x_dtype = x.dtype
        if not self.ready():
            return x
        if self.sym:
            # {-scale, +scale}: sign(x) * scale, treating 0 as +1
            binary = torch.where(x >= 0,
                                 torch.ones_like(x),
                                 -torch.ones_like(x))
            return (binary * self.scale).to(x_dtype)
        else:
            # {zero - scale, zero + scale}: threshold at zero
            binary = torch.where(x >= self.zero,
                                 torch.ones_like(x),
                                 -torch.ones_like(x))
            return (self.zero + binary * self.scale).to(x_dtype)

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)

def find_qlayers(module, layers=[torch.nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlayers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

