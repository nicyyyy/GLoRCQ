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


class GPTQ_lora:

    def __init__(self, layer, lora):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.lora = lora
    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())


    def fasterquant_rot_new_perm_split(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False, ha_bsize=256, id_bsize = 256
    ):
        W = self.layer.weight.data.clone()

        if self.lora["U"] == None:
            return
        if self.lora["U"].shape[1]<16:
            return 
        lora = (self.lora["U"].to(W.dtype) @ torch.diag(self.lora["Si"])@ self.lora["V"].to(W.dtype))
        lora = (torch.diag(self.lora["Sa"]) @ lora).T

        W = W.float()
        lora = lora.float().to(W.device)
        res = W - lora
        H = self.H
        if torch.all(H == 0).item():
            print("no inp")
            return

        

        if not self.quantizer.ready():
            self.quantizer.find_params(res)

        
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        res[:, dead] = 0

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        try:
            H = torch.linalg.cholesky(H)
        except torch._C._LinAlgError:
            H += torch.eye(self.columns, device=self.dev) * 1e-6
            H = (H + H.T) / 2
            H = torch.linalg.cholesky(H)

        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = res[:, i1:i2].clone()

            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(res[:, (i1 + i):(i1 + i + groupsize)])

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            res[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        
        Q = Q.reshape(self.layer.weight.shape)

        # Q = (Q @ PD_rot.T)
        QR = Q.to(self.layer.weight.data.dtype) + lora.to(self.layer.weight.data.dtype)
        
        self.layer.weight.data = QR

        W = W.detach().cpu()
        lora = lora.detach().cpu()
        H = H.detach().cpu()
        Q = Q.detach().cpu()
        QR = QR.detach().cpu()

        del W, lora, H, Q, QR


    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        cleanup_memory(verbos=False)
        
        
@torch.no_grad()
def gptq_fwrd_lora(model, loras, dataloader, dev, args):
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
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    quantizers = {}

    for i in tqdm.tqdm(range(len(layers)), desc="LoPRo quantization process (scalar)..."):
        if isinstance(layers[0], MixtralDecoderLayer):
            regular_qlist = MixtralQuantLayerA[1]
            sequential = [regular_qlist]
        else:
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

                gptq[name] = GPTQ_lora(subset[name], lora_layer[name])
                gptq[name].quantizer = quantizer.BiWeightQuantizer()
                gptq[name].quantizer.configure(
                    1, perchannel=True, sym=True, mse=args.w_clip
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
                gptq[name].fasterquant_rot_new_perm_split(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order, static_groups=False, id_bsize = args.id_bsize, ha_bsize = args.ha_bsize
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

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
        if i>3:
            break
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
