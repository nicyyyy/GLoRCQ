from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import transformers
import torch
import torch.nn as nn
import argparse
import os
import json

import tqdm
import gc
import functools
from collections import defaultdict
from typing import List

from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM



@torch.no_grad()
def llama_eval(model, testenc, dev ,seqlen=4096):
    print('Evaluating ...')

    model = model.to(dev)

    for module in model.modules():
        module.to(dev)

    model.seqlen = seqlen
    testenc = testenc.input_ids
    testenc = testenc.to(dev)
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            
            # Handle attention_mask
            if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
                cache['attention_mask'] = torch.as_tensor(kwargs['attention_mask']).to(dev)
            else:
                cache['attention_mask'] = None
            
            # Handle position_ids
            if 'position_ids' in kwargs and kwargs['position_ids'] is not None:
                cache['position_ids'] = torch.as_tensor(kwargs['position_ids']).to(dev)
            else:
                cache['position_ids'] = None
            
            # Move all other kwargs to the device as well
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    kwargs[k] = v.to(dev)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, cache_position = position_ids.squeeze())[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache

@torch.no_grad()
def mixtral_eval(model, testenc, dev ,seqlen=4096):
    print('Evaluating ...')

    model.seqlen = seqlen
    testenc = testenc.to(dev)
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if next(model.model.embed_tokens.parameters()).is_meta:
        model.model.embed_tokens = model.model.embed_tokens.to_empty(device=dev)
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    
    if next(layers[0].parameters()).is_meta:
        layers[0] = layers[0].to_empty(device=dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    for i in range(len(layers)):
        if next(layers[i].parameters()).is_meta:
            layers[i] = layers[i].to_empty(device=dev)
        layer = layers[i].to(dev)
        
        attention_mask = attention_mask.to(next(layer.parameters()).device)
        position_ids = position_ids.to(next(layer.parameters()).device)
        inps = inps.to(next(layer.parameters()).device)
        
        for j in range(nsamples):
            inps[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, cache_position=position_ids.squeeze())[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    if model.model.norm is not None:
        if next(model.model.norm.parameters()).is_meta:
            model.model.norm = model.model.norm.to_empty(device=dev)
        model.model.norm = model.model.norm.to(dev)
    
    if next(model.lm_head.parameters()).is_meta:
        model.lm_head = model.lm_head.to_empty(device=dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache
