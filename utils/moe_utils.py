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


def _has_experts_segment(layer_name):
    """Check if name contains a '.experts.' segment (any MoE architecture).

    Matches both Qwen-style ``mlp.experts.0.gate_proj`` and
    Mixtral-style ``block_sparse_moe.experts.0.w1``.
    """
    return '.experts.' in layer_name or layer_name.startswith('experts.')


def is_regular_expert(layer_name):
    """Check if layer is a regular expert."""
    return _has_experts_segment(layer_name) and not is_shared_expert(layer_name)


def extract_expert_info(layer_name):
    """Extract expert index and layer type from layer name."""
    if _has_experts_segment(layer_name):
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


# ---------------------------------------------------------------------------
# Per-architecture MoE config registry.  Add new models by adding an entry.
# ---------------------------------------------------------------------------
_MOE_ARCH_CONFIG = {
    "qwen2_moe": {
        "has_shared_expert": True,     # shared_expert exists, keep FP16
        "expert_container": "mlp",     # module path containing .experts
    },
    "qwen3_moe": {
        # Qwen3-MoE (e.g., Qwen3-30B-A3B): no shared expert, container=`mlp`.
        "has_shared_expert": False,
        "expert_container": "mlp",
    },
    "mixtral": {
        "has_shared_expert": False,
        "expert_container": "block_sparse_moe",
    },
    "deepseek_v2": {
        # DeepSeek-V2-Lite: 64 routed + 2 shared experts, top-6, container=`mlp`.
        # Router `mlp.gate` is a custom MoEGate (not nn.Linear). Layer 0 is a plain
        # dense DeepseekV2MLP (first_k_dense_replace=1) with no `.experts.`.
        "has_shared_expert": True,
        "expert_container": "mlp",
    },
}
_DEFAULT_MOE_CONFIG = {
    "has_shared_expert": False,
    "expert_container": None,
}


def get_moe_config(model_type: str) -> dict:
    """Return MoE arch config for *model_type* (from ``config.model_type``)."""
    return _MOE_ARCH_CONFIG.get(model_type, _DEFAULT_MOE_CONFIG)

