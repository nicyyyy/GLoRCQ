import math
import time
import tqdm
import torch
import torch.nn as nn
import utils.utils
import quantizer
import logging
from utils.hadamard_utils import *
from utils.qlayer_name_utils import *
from utils.moe_utils import *
import torch
from torch import nn
import numpy as np

import math
import time

import transformers
from quantizer.quantizer_v import *
from quantizer.quantizer_v import vq_quantize, quantize_centroids
from transformers.models.bloom.modeling_bloom import BloomBlock, BloomGelu
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.activations import GELUActivation
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2DecoderLayer
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )
DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

def quad_loss(w_q, G, v, offset):
    """
    A generic function for computing the quadratic loss:
    L = 1/2 (G w_q, w_q) + (v, w_q) + offset

    Parameters
    ----------
    w_q : (c_out, m) or (m, 1)
        Quantized weights to be optimized.
    G : (m, m)
        Matrix part.
    v : shape(w_q)
        Linear part.
    offset : ()
        Scalar part.
    """
    # Quadratic loss: 1/2 wGw^T
    loss = 0.5 * (w_q.mm(G) * w_q).sum()
    # Add linear term and offset
    loss += (v * w_q).sum()
    loss += offset
    return loss


def quad_loss_2(W, Q, G):
    Werr = W - Q
    return (Werr.mm(G) * Werr).sum()


class GPTVQ_lora:

    def __init__(self, layer, lora):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.lora = lora
    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        include_m_step=False,
        use_vq=False,
        svd_rank=None,
        hessian_weighted_lookups=False,
        only_init_kmeans=False,
        ha_bsize=256, 
        id_bsize = 256
    ):
        if self.lora["U"] == None:
            return
        if self.lora["U"].shape[1]<16:
            return 
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        lora = (self.lora["U"].to(W.dtype) @ torch.diag(self.lora["Si"])@ self.lora["V"].to(W.dtype))
        lora = (torch.diag(self.lora["Sa"]) @ lora).T

        W = W.float()
        lora = lora.float().to(W.device)

        res = W - lora

        if not self.quantizer.ready() and not use_vq:
            self.quantizer.find_params(W)

        H = self.H
        self.G = self.H.clone()
        del self.H
        if torch.all(H == 0).item():
            print("no inp")
            return
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        res[:, dead] = 0


        # Begin: construct partial permute rotate matrix.
        # Auto-fallback ha_bsize to largest power-of-2 that divides W.shape[1].
        # E.g. W.shape[1]=1408 with ha_bsize=256 → 1408%256=128 ≠ 0, fall back to 128.
        partial_size = id_bsize
        rotate_size = ha_bsize
        while rotate_size > 1 and W.shape[1] % rotate_size != 0:
            rotate_size //= 2
        rot = block_diagonal_walsh_matrix(W.shape[1], rotate_size, W.device)
        abs_tensor = res.abs()
        mean_abs_per_col = abs_tensor.mean(dim=0) 
        #top_values, top_indices = torch.topk(-mean_abs_per_col, W.shape[1])

        perm = torch.log2(torch.diag(H).clamp(min=1e-30))
        perm = mean_abs_per_col/(perm - torch.min(perm) + 1)
        top_values, top_indices = torch.topk(-perm, W.shape[1])

        
        Ppermute = construct_partial_permutation_matrix_upper(top_indices, m = W.shape[1], dtype = W.dtype).to(W.device)
        diagI_rot = create_diagI_matrix_upper(rot, rot.shape[0]-partial_size).to(W.device).to(W.dtype)
        PD_rot = Ppermute @ diagI_rot
        res = res @ PD_rot
        H = PD_rot.T @ H @ PD_rot



        if static_groups:
            raise NotImplementedError("Static groups are not supported in this repo")

        if actorder:
            raise NotImplementedError("Activation (re)-ordering is not supported in this repo")

        vq_dim = self.assignments = None
        S = vq_scaling_blocksize = vq_scaling_n_bits = None
        if use_vq:
            vq_dim = self.quantizer.vq_dim
            groupsize = self.quantizer.get_groupsize(W, groupsize)
            self.assignments = []
            assert blocksize % vq_dim == 0

            vq_scaling_blocksize = self.quantizer.vq_scaling_blocksize
            vq_scaling_n_bits = self.quantizer.vq_scaling_n_bits
            if vq_scaling_blocksize > 0:
                assert vq_scaling_blocksize % vq_dim == 0
                S = torch.ones_like(W)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = res[:, i1:i2].clone()
            if use_vq and vq_scaling_blocksize > 0:
                W1_scaled, S1 = self.quantizer.blockwise_normalize_data(
                    W1,
                    vq_scaling_blocksize,
                    self.quantizer.vq_scaling_norm,
                    vq_scaling_n_bits,
                    self.quantizer.vq_scaling_domain,
                )
                S[:, i1:i2] = S1
            else:
                W1_scaled = W1
                S1 = torch.ones_like(W1)

            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        extra_args = {}
                        if use_vq and vq_dim > 1 and hessian_weighted_lookups:
                            H_inv_diag = torch.diag(Hinv)[i1 + i : i1 + i + groupsize]
                            extra_args["H_inv_diag"] = H_inv_diag

                        W_group = res[:, (i1 + i) : (i1 + i + groupsize)]

                        W_group_scaled = W_group
                        if use_vq:
                            self.assignments.append([])
                            if vq_scaling_blocksize > 0:
                                assert vq_scaling_blocksize % vq_dim == 0
                                W_group_scaled, S_group = self.quantizer.blockwise_normalize_data(
                                    W_group,
                                    vq_scaling_blocksize,
                                    self.quantizer.vq_scaling_norm,
                                    self.quantizer.vq_scaling_n_bits,
                                    self.quantizer.vq_scaling_domain,
                                )

                        self.quantizer.find_params(W_group_scaled, **extra_args)

                if not use_vq:
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    q = quantize(
                        w.unsqueeze(1),
                        self.quantizer.scale,
                        self.quantizer.zero,
                        self.quantizer.maxq,
                    ).flatten()

                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d**2

                    err1 = (w - q) / d
                    # (R x 1).matmul(1 x C') --> R x C' (C': remaining (unquantized) columns)
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

                elif i % vq_dim == 0:
                    w = W1[:, i : i + vq_dim]  # R x D
                    d = torch.diag(Hinv1)[i : i + vq_dim].unsqueeze(0)  # 1 x D
                    w_scaled = W1_scaled[:, i : i + vq_dim]  # R x D
                    s = S1[:, i : i + vq_dim]

                    H_inv_diag = None
                    if vq_dim > 1 and hessian_weighted_lookups:
                        H_inv_diag = 1.0 / d.to(w.device)

                    q, assmt = vq_quantize(
                        w_scaled, self.quantizer, H_inv_diag=H_inv_diag
                    )  # R x 1 x D, R x 1
                    q = torch.mul(q, s)  # de-scaling

                    self.assignments[-1].append(assmt)

                    Q1[:, i : i + vq_dim] = q
                    Losses1[:, i : i + vq_dim] = (w - q) ** 2 / d**2  # R x D / 1 x D

                    err1 = (w - q) / d  # R x D
                    # batch matmul solution: (D x R x 1).matmul(D x 1 x C').sum(0) --> R x C'
                    if not only_init_kmeans:
                        update = torch.bmm(
                            err1.transpose(0, 1).unsqueeze(-1),
                            Hinv1[i : i + vq_dim, i + vq_dim :].unsqueeze(1),
                        ).sum(0)
                        W1[:, i + vq_dim :] -= update
                        Err1[:, i : i + vq_dim] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            if not only_init_kmeans:
                res[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])


        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        Q = Q.reshape(self.layer.weight.shape)

        # ---- Retain VQ state for real-quant export ----
        # We MUST save Q (still in ROTATED space) and derive codes here — after
        # `Q @ PD_rot.T` below, Q is in un-rotated (raw) space and cannot be
        # matched against centroids (which live in the rotated+quantization
        # space). Also save FULL per-group centroids (not just cb[0]) so future
        # non-pool-kmeans configs work too.
        if use_vq and self.assignments is not None:
            try:
                Q_rot = Q.detach().to(torch.float16)  # still in rotated space here
                self.layer.gptvq_Q_rotated = Q_rot.cpu()

                if len(self.quantizer.all_centroids) > 0:
                    # Full centroids per find_params call: (groups_per_column, K, vdim).
                    # With pool_kmeans the codebook was broadcast — cb[0] is fine.
                    # With per-group codebooks (non-pool), cb[i] differs per row-group.
                    cbs = [cb[0].detach().to(torch.float16).cpu()
                           for cb in self.quantizer.all_centroids]
                    self.layer.gptvq_centroids = torch.stack(cbs, dim=0)  # (n_codebooks, K, vdim)

                    # Derive codes NOW from Q_rot + centroids so inference doesn't
                    # need to re-solve nearest-centroid at load time. Uses the same
                    # semantics as the inference dequant path: for each column block
                    # (of codes_per_cb vec-4 vectors), find nearest centroid.
                    _cent_stack = self.layer.gptvq_centroids.to(Q_rot.device).float()
                    n_cb = _cent_stack.shape[0]
                    out_d, in_d = Q_rot.shape
                    n_vecs = in_d // int(vq_dim)
                    assert n_vecs % n_cb == 0, f"n_vecs={n_vecs} not divisible by n_cb={n_cb}"
                    codes_per_cb = n_vecs // n_cb
                    _codes = torch.empty(out_d, n_vecs, dtype=torch.uint8, device=Q_rot.device)
                    Q_f = Q_rot.float()
                    _chunk = 1024 if codes_per_cb <= 128 else 32
                    for _cb_id in range(n_cb):
                        _v_lo = _cb_id * codes_per_cb
                        _v_hi = _v_lo + codes_per_cb
                        _cb = _cent_stack[_cb_id]  # (K, vdim)
                        for _r0 in range(0, out_d, _chunk):
                            _r1 = min(_r0 + _chunk, out_d)
                            _Q_block = Q_f[_r0:_r1, _v_lo * int(vq_dim):_v_hi * int(vq_dim)].reshape(
                                _r1 - _r0, codes_per_cb, int(vq_dim))
                            _d = (_Q_block.unsqueeze(2) - _cb.unsqueeze(0).unsqueeze(0)).pow(2).sum(-1)
                            _codes[_r0:_r1, _v_lo:_v_hi] = _d.argmin(-1).to(torch.uint8)
                    self.layer.gptvq_codes = _codes.cpu()
                self.layer.gptvq_perm       = top_indices.detach().cpu().to(torch.int32)
                # diagI_rot is a (in_d, in_d) diag-style matrix; pull its diagonal
                # to get the (in_d,) sign vector used by Hadamard rotation.
                if diagI_rot.dim() == 2:
                    diag_vec = torch.diagonal(diagI_rot).detach().to(torch.float16).cpu()
                else:
                    diag_vec = diagI_rot.detach().to(torch.float16).cpu()
                self.layer.gptvq_diag_signs = diag_vec
                self.layer.gptvq_vdim       = int(vq_dim) if vq_dim is not None else None
                self.layer.gptvq_groupsize  = int(groupsize)
                self.layer.gptvq_rotate_size = int(rotate_size)
                self.layer.gptvq_partial_size = int(partial_size)
            except Exception as _e:
                print(f"[fasterquant] warning: failed to attach gptvq state: {_e}")

        # Un-rotate Q and add LoRA compensation → set as fake-quant weight
        Q = (Q @ PD_rot.T)
        QR = Q.to(self.layer.weight.data.dtype) + lora.to(self.layer.weight.data.dtype)
        self.layer.weight.data = QR

        W = W.detach().cpu()
        lora = lora.detach().cpu()
        H = H.detach().cpu()
        Q = Q.detach().cpu()
        QR = QR.detach().cpu()
        PD_rot = PD_rot.detach().cpu()
        Ppermute = Ppermute.detach().cpu()
        diagI_rot = diagI_rot.detach().cpu()

        del W,lora, H, Q ,QR, PD_rot, Ppermute, diagI_rot


    
    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()

@torch.no_grad()
def gptvq_fwrd_lora(model, loras, dataloader, dev, args):
    '''
    From GPTQ repo 
    '''
    logging.info('-----GPTQ Quantization-----')
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    layer_kwargs = {}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            layer_kwargs.update(kwargs)
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
        def __getattr__(self, name):
            if name == "module":
                return self._modules["module"]
            try:
                return getattr(self._modules["module"], name)
            except KeyError:
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    layers[0] = Catcher(layers[0])
    #print(dataloader)
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    quantizers = {}


    for i in tqdm.tqdm(range(len(layers)), desc="LoPRo quantization process (vector)..."):

        # Name-based dispatch works for both Qwen (`.experts.` + gate/up/down)
        # and Mixtral (`.experts.` + w1/w2/w3). No isinstance branching needed.
        subset = find_layers(layers[i])
        normal_qlist, shared_qlist, regular_qlist = get_moe_qlayers_name(subset)
        sequential = [regular_qlist]

        lora_layer = loras[i]
        layer = layers[i].to(dev)
        full = quantizer.find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                layer_weight_bits = args.w_bits
                layer_weight_sym = not(args.w_asym)
                if 'lm_head' in name:
                    layer_weight_bits = 16
                    continue
                # Mixtral names down_proj as w2; keep both patterns
                if args.int8_down_proj and ('down_proj' in name or '.w2' in name):
                    layer_weight_bits = 8
                gptq[name] = GPTVQ_lora(subset[name], lora_layer[name])
                

                if args.w_bits == 3:
                    vdim = 2
                    gp = 32768
                else: 
                    vdim = 4
                    gp = 65536
                _pool_km = getattr(args, 'pool_kmeans', False)
                QClass = lambda: VQQuantizer(
                    vq_dim=vdim,
                    columns_per_group=256,
                    vq_scaling_blocksize=0,
                    vq_scaling_norm="max",
                    vq_scaling_n_bits=4,
                    vq_scaling_domain="log",
                    kmeans_init_method="mahalanobis",
                    assignment_chunk_size=None,
                    kmeans_iters=30 if _pool_km else 10,
                    codebook_bitwidth=None if _pool_km else 8,
                    quantize_per_codebook=".",
                    pool_kmeans=_pool_km,
                )

                gptq[name].quantizer = QClass()

                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.w_clip
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize

                include_m_step = False
                use_vq = True
                svd_rank = 0
                hessian_weighted_lookups = True
                only_init_kmeans = False
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=gp,
                    actorder=args.act_order,
                    static_groups=False,
                    include_m_step=include_m_step,
                    use_vq=use_vq,
                    svd_rank=svd_rank,
                    hessian_weighted_lookups=hessian_weighted_lookups,
                    only_init_kmeans=only_init_kmeans,
                    id_bsize = args.id_bsize, 
                    ha_bsize = args.ha_bsize
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        layer = layer.cpu()
        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    cleanup_memory(verbos=True)
    logging.info('-----GPTQ Quantization Done-----\n')
    return quantizers



       
@torch.no_grad()
def rtn_fwrd(model, dev, args):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    assert args.w_groupsize ==-1, "Groupsize not supported in RTN!"
    layers = model.model.layers
    torch.cuda.empty_cache()

    quantizers = {}

    for i in tqdm.tqdm(range(len(layers)), desc="(RtN Quant.) Layers"):
        layer = layers[i].to(dev)

        subset = quantizer.find_qlayers(layer,
                                            layers=[torch.nn.Linear])

        for name in subset:
            layer_weight_bits = args.w_bits
            if 'lm_head' in name:
                layer_weight_bits = 16
                continue
            if args.int8_down_proj and 'down_proj' in name:
                layer_weight_bits = 8

            quantizer = quantizer.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip
            )
            W = subset[name].weight.data
            quantizer.find_params(W)
            subset[name].weight.data = quantizer.quantize(W).to(
                next(iter(layer.parameters())).dtype)
            quantizers['model.layers.%d.%s' % (i, name)] = quantizer.cpu()
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer
            
    cleanup_memory(verbos=True)
    return quantizers
