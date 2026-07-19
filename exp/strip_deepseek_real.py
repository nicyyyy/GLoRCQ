#!/usr/bin/env python3
"""Strip fp16 quantized-expert weights from a copied fake-quant DeepSeek-V2-Lite
checkpoint using the already-exported cross_layer_info.pt. Produces the
real-quant checkpoint WITHOUT recomputing Phases 1-3.

The fake-quant export already wrote cross_layer_info.pt (int8 U/SV + Sa +
vq_residuals + attn_gptq_packs). _strip_fp16_quantized_weights only needs
those three structures to know which safetensors tensors are redundant.
"""
import sys, os, torch
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
from run_quantize import _strip_fp16_quantized_weights

REAL = "/mnt/Data/yqy/resource_dir/glorcq_paper_exp/deepseek_v2lite_real"
info = torch.load(os.path.join(REAL, "cross_layer_info.pt"), map_location="cpu")
assignments   = info["assignments"]
vq_residuals  = info["vq_residuals"]
attn_gptq     = info["attn_gptq_packs"]
print(f"assignments wtypes: {[ (wt, len(assignments[wt])) for wt in assignments ]}")
print(f"attn_gptq_packs: {len(attn_gptq)} (expect 0 for attn_bits=16)")
_strip_fp16_quantized_weights(REAL, assignments, attn_gptq, vq_residuals)
print("STRIP_DONE")
