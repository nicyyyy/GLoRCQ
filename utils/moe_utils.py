import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import torch
import torch.nn as nn

def is_shared_expert(layer_name):
    """Check if layer is a shared expert."""
    return 'shared_expert' in layer_name


def is_regular_expert(layer_name):
    """Check if layer is a regular expert."""
    return ('mlp.experts' in layer_name) and not is_shared_expert(layer_name)


def extract_expert_info(layer_name):
    """Extract expert index and layer type from layer name."""
    if 'mlp.experts' in layer_name:
        parts = layer_name.split('.')
        expert_idx_pos = parts.index('experts') + 1
        expert_idx = int(parts[expert_idx_pos])
        layer_type = '.'.join(parts[expert_idx_pos + 1:])
        return expert_idx, layer_type
    return None, None


def group_layers_by_type(subset):
    """Group layers by type: normal layers, shared experts, regular experts."""
    normal_layers = {}
    shared_experts = {}
    regular_experts = {}
    

    for name, layer in subset.items():
        if is_shared_expert(name):
            shared_experts[name] = layer
        elif is_regular_expert(name):
            expert_idx, layer_type = extract_expert_info(name)
            if layer_type not in regular_experts:
                regular_experts[layer_type] = {}
            regular_experts[layer_type][expert_idx] = (name, layer)
        else:
            normal_layers[name] = layer
    return normal_layers, shared_experts, regular_experts


def get_moe_qlayers_name(subset):
    
    normal_qlayers = []
    share_qlayers = []
    regular_qlayers = []
    regular_experts = {}
    for name, layer in subset.items():
        if is_shared_expert(name):
            share_qlayers.append(name)
        elif is_regular_expert(name):
            expert_idx, layer_type = extract_expert_info(name)
            if layer_type not in regular_experts:
                regular_experts[layer_type] = {}
            regular_experts[layer_type][expert_idx] = (name, layer)
            regular_qlayers.append(name)
        else:
            normal_qlayers.append(name)
    return normal_qlayers, share_qlayers, regular_qlayers


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

