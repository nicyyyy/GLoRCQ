"""
Shared model/tokenizer loading utility for GLoRCQ.

Supports two modes:
  - real_quant=True:  Load packed GLoRCQ quantized model (GLoRCQLinear layers)
  - real_quant=False: Load standard HF fp16 fake-quant model
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def load_model_and_tokenizer(model_path, device="cuda:0", real_quant=False):
    """Load model and tokenizer from a GLoRCQ output directory.

    Args:
        model_path: Path to the model directory (either fake-quant HF checkpoint
                    or real-quant GLoRCQ directory containing glorcq_model.pt).
        device: Target device string (default: "cuda:0").
        real_quant: If True, load packed real-quant model via
                    ``glorcq.inference.model_builder.load_glorcq_model``.
                    If False (default), load standard HF ``AutoModelForCausalLM``.

    Returns:
        (model, tokenizer) tuple.
    """
    if real_quant:
        model = _load_real_quant(model_path, device)
    else:
        model = _load_fake_quant(model_path, device)

    tokenizer = _load_tokenizer(model_path)
    return model, tokenizer


def _load_real_quant(model_path, device):
    """Load packed GLoRCQ real-quantized model."""
    from glorcq.inference.model_builder import load_glorcq_model
    return load_glorcq_model(model_path, device=device)


def _load_fake_quant(model_path, device):
    """Load standard HF model (fake-quant fp16 checkpoint)."""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.use_cache = True
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    if not hasattr(model, "seqlen"):
        model.seqlen = getattr(config, "max_position_embeddings", 4096)
    return model


def _load_tokenizer(model_path):
    """Load tokenizer, falling back to original model path from cross_layer_info.pt."""
    # Try loading directly from model_path first
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True,
        )
        return tokenizer
    except Exception:
        pass

    # Fallback: read original model path from cross_layer_info.pt
    cross_layer_path = os.path.join(model_path, "cross_layer_info.pt")
    if os.path.exists(cross_layer_path):
        try:
            info = torch.load(cross_layer_path, map_location="cpu", weights_only=False)
            original_model = info.get("config", {}).get("model_path", None)
            if original_model:
                return AutoTokenizer.from_pretrained(
                    original_model, use_fast=False, trust_remote_code=True,
                )
        except Exception:
            pass

    raise RuntimeError(
        f"Could not load tokenizer from {model_path} or from cross_layer_info.pt. "
        "Ensure the model directory contains tokenizer files or a valid cross_layer_info.pt."
    )
