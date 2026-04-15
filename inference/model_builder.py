"""
GLoRCQ model builder: load real-quantized model for inference.

Loads packed weights, shared U matrices, per-expert V matrices,
and replaces nn.Linear layers with GLoRCQLinear.
"""

import os
import sys
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from .quantized_linear import GLoRCQLinear

# Ensure glorcq/ is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_GLORCQ_ROOT = os.path.dirname(_HERE)
if _GLORCQ_ROOT not in sys.path:
    sys.path.insert(0, _GLORCQ_ROOT)


# ---------------------------------------------------------------------------
# SharedUCache: manage cross-layer shared U matrices
# ---------------------------------------------------------------------------
class SharedUCache:
    """
    Manage cross-layer shared U matrices.

    Initial implementation: preload all U matrices to GPU.
    Interface reserved for future optimization (LRU, on-demand dequant, etc.).
    """

    def __init__(self, shared_matrices, uv_bits, device="cuda"):
        self._cache = {}
        self._uv_bits = uv_bits
        for wtype, groups in shared_matrices.items():
            for gid, data in groups.items():
                U_fp16 = _dequant_intN(
                    data["U_int8"].to(device),
                    data["U_scale"].to(device),
                    uv_bits,
                ).half()
                S = data["S"].half().to(device)
                self._cache[(wtype, gid)] = (U_fp16, S)

    def get(self, wtype, group_id):
        """Get shared U matrix and S vector for (wtype, group_id)."""
        return self._cache.get((wtype, group_id), (None, None))

    def preload_for_layer(self, layer_idx, active_experts=None):
        """(Interface reserved) Preload U matrices for a specific layer."""
        pass

    def evict(self, wtype=None, group_id=None):
        """(Interface reserved) Release cached matrices."""
        pass


def _dequant_intN(q, scale, nbits):
    """Dequantize intN (stored as int8) → float32 using per-column scale."""
    maxval = 2 ** (nbits - 1) - 1
    return q.float() / maxval * scale.float()


# ---------------------------------------------------------------------------
# RotationCache: manage TurboQuant Pi matrices and centroids
# ---------------------------------------------------------------------------
class RotationCache:
    """
    Manage TurboQuant rotation matrices (Pi) and codebook centroids.

    Each unique (dim, bits, seed) tuple gets one Pi matrix and one centroids
    vector, shared across all experts/layers with that configuration.

    Memory footprint:
      Pi(4096) = 32 MB fp16,  Pi(14336) = 390 MB fp16,
      centroids are negligible (4 × fp32 = 16 bytes per config).
    """

    def __init__(self, device="cuda"):
        self._device = device
        self._pi = {}          # (dim, bits, seed) → Pi (dim, dim) fp16
        self._centroids = {}   # (dim, bits, seed) → centroids (2^bits,) fp32

    def register(self, dim, bits, seed=42):
        """
        Ensure Pi and centroids for (dim, bits, seed) are loaded.

        Uses TurboQuantMSE to generate them deterministically from the seed,
        then discards the quantizer instance and keeps only Pi/centroids.
        """
        key = (dim, bits, seed)
        if key in self._pi:
            return

        # Import here to avoid top-level dependency on turboquant
        _turboquant_path = os.path.join(_GLORCQ_ROOT, "thirdpart", "turboquant")
        if _turboquant_path not in sys.path:
            sys.path.insert(0, _turboquant_path)
        from turboquant.quantizer import TurboQuantMSE

        print(f"[RotationCache] Generating Pi({dim}) and centroids "
              f"(bits={bits}, seed={seed}) ...")
        tq = TurboQuantMSE(
            dim=dim, bits=bits,
            device=self._device, dtype=torch.float32,
            seed=seed,
        )
        self._pi[key] = tq.Pi.half()            # (dim, dim) fp16 on device
        self._centroids[key] = tq.centroids      # (2^bits,) fp32 on device
        # Let the TurboQuantMSE instance be garbage-collected

    def get_pi(self, dim, bits, seed=42):
        """Get rotation matrix Pi (dim, dim) fp16 on device."""
        return self._pi[(dim, bits, seed)]

    def get_centroids(self, dim, bits, seed=42):
        """Get codebook centroids (2^bits,) fp32 on device."""
        return self._centroids[(dim, bits, seed)]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_glorcq_model(model_path, device="cuda:0"):
    """
    Load a GLoRCQ real-quantized model for inference.

    Uses the generic AutoModel + nn.Linear replacement approach.

    Args:
        model_path: directory containing config.json, glorcq_model.pt,
                    cross_layer_info.pt
        device: target device for the model

    Returns:
        model: CausalLM model with GLoRCQLinear layers
    """
    print(f"[GLoRCQ] Loading real-quantized model from {model_path} (generic loader)")

    # 1. Load HF config and create model structure
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.use_cache = True

    # Check if glorcq_model.pt exists (real quant mode)
    glorcq_model_path = os.path.join(model_path, "glorcq_model.pt")
    cross_layer_path = os.path.join(model_path, "cross_layer_info.pt")

    if not os.path.exists(glorcq_model_path):
        raise FileNotFoundError(
            f"glorcq_model.pt not found in {model_path}. "
            "This model was likely saved in fake-quant mode. "
            "Use standard HF loading instead."
        )

    # Create model on CPU with empty weights
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, torch_dtype=torch.float16,
        )
    model.seqlen = 4096

    # 2. Load packed quantized weights
    print("[GLoRCQ] Loading glorcq_model.pt ...")
    model_data = torch.load(glorcq_model_path, map_location="cpu", weights_only=False)
    model_config = model_data["model_config"]
    layers_data = model_data["layers"]

    # 3. Load cross-layer info
    print("[GLoRCQ] Loading cross_layer_info.pt ...")
    cross_layer_info = torch.load(cross_layer_path, map_location="cpu", weights_only=False)
    shared_matrices = cross_layer_info["shared_matrices"]
    per_expert_V = cross_layer_info["per_expert_V"]
    assignments = cross_layer_info["assignments"]
    cl_config = cross_layer_info["config"]

    uv_bits = cl_config.get("uv_bits", model_config.get("uv_bits", 8))

    # 4. Build SharedUCache
    print("[GLoRCQ] Building SharedUCache ...")
    u_cache = SharedUCache(shared_matrices, uv_bits, device=device)

    # 5. Build RotationCache for TurboQuant runtime dequant
    rotation_cache = None
    if model_config.get("use_turboquant", False):
        print("[GLoRCQ] Building RotationCache for TurboQuant ...")
        rotation_cache = RotationCache(device=device)

    # 6. Replace nn.Linear with GLoRCQLinear
    print("[GLoRCQ] Replacing linear layers ...")
    _replace_linear_layers(
        model, layers_data, assignments, per_expert_V,
        u_cache, uv_bits, rotation_cache, device,
    )

    # 7. Move non-linear components to device
    m = getattr(model, "model", model)
    for attr in ("embed_tokens", "norm", "rotary_emb"):
        if hasattr(m, attr):
            obj = getattr(m, attr)
            if obj is not None:
                setattr(m, attr, obj.to(device))
    if hasattr(model, "lm_head"):
        # lm_head shares embed_tokens weight in many models
        if model.lm_head.weight.device.type == "meta":
            # Tied weights: point to embed_tokens
            model.lm_head.weight = m.embed_tokens.weight
        else:
            model.lm_head = model.lm_head.to(device)

    model.eval()
    print(f"[GLoRCQ] Model loaded successfully on {device}")
    return model


def _replace_linear_layers(model, layers_data, assignments, per_expert_V,
                            u_cache, uv_bits, rotation_cache, device):
    """
    Replace nn.Linear layers in the model with GLoRCQLinear.

    Traverses each decoder layer, finds matching modules in layers_data,
    and constructs GLoRCQLinear with the appropriate quantized weights
    and LoRA parameters.

    TurboQuant layers use runtime dequant via RotationCache (weights stay
    compressed on GPU). Falls back to pre-dequant if RotationCache is None.
    """
    # _dequant_intN is defined at module level above

    # Build lookup: (layer_idx, module_name) → assignment info
    assignment_lookup = {}  # (wtype, layer_idx, expert_idx) → (group_id, local_idx)
    for wtype, records in assignments.items():
        for li, rec in enumerate(records):
            key = (wtype, rec["layer"], rec["expert"])
            assignment_lookup[key] = (rec["group_id"], wtype, li)

    # Get model layers
    m = getattr(model, "model", model)
    layers = m.layers

    for layer_idx in range(len(layers)):
        if layer_idx not in layers_data:
            # No quantized data for this layer, move to device as-is
            layers[layer_idx] = layers[layer_idx].to(device)
            continue

        layer = layers[layer_idx]
        layer_data = layers_data[layer_idx]

        # Find all nn.Linear modules in this layer
        linear_modules = _find_linear_modules(layer)

        for module_name, (parent, attr_name, linear) in linear_modules.items():
            if module_name not in layer_data:
                # Not a quantized module, move to device
                setattr(parent, attr_name, linear.to(device))
                continue

            packed = layer_data[module_name]

            # Determine quant type
            if "qweight_int" in packed:
                quant_type = "gptq"
            elif "packed_indices" in packed:
                quant_type = "turbo"
            elif "weight_quant" in packed:
                quant_type = packed.get("quant_method", "unknown")
            else:
                # Unknown format, skip
                setattr(parent, attr_name, linear.to(device))
                continue

            in_f = linear.in_features if hasattr(linear, "in_features") else packed.get("dim", 0)
            out_f = linear.out_features if hasattr(linear, "out_features") else 0
            if in_f == 0 or out_f == 0:
                # Try to infer from packed data
                if "qweight_int" in packed:
                    out_f, in_f = packed["qweight_int"].shape
                elif "weight_quant" in packed:
                    out_f, in_f = packed["weight_quant"].shape

            ql = GLoRCQLinear(in_f, out_f)

            # Load quantized weights
            if quant_type == "gptq":
                ql.load_gptq(packed, device=device)
            elif quant_type == "turbo":
                if rotation_cache is not None:
                    # Runtime dequant: weights stay compressed on GPU
                    ql.load_turbo(packed, rotation_cache, device=device)
                else:
                    # Fallback: pre-dequant if RotationCache unavailable
                    ql.quant_type = "turbo"
                    if "weight_quant" in packed:
                        ql._turbo_dequant_W = packed["weight_quant"].half().to(device)

            # Load LoRA from cross_layer_info
            _load_lora_for_module(
                ql, layer_idx, module_name,
                assignment_lookup, u_cache, per_expert_V,
                uv_bits, device,
            )

            # Load bias if present
            if hasattr(linear, "bias") and linear.bias is not None:
                if linear.bias.device.type != "meta":
                    ql.bias_param = linear.bias.data.to(device)

            # Replace the module
            setattr(parent, attr_name, ql)

        # Move remaining non-linear submodules to device
        layer = layers[layer_idx]
        for name, module in layer.named_modules():
            if isinstance(module, GLoRCQLinear):
                continue
            for pname, param in module.named_parameters(recurse=False):
                if param.device.type == "meta":
                    # Initialize with zeros (these are non-quantized params like norms)
                    new_param = nn.Parameter(
                        torch.zeros_like(param, device=device, dtype=param.dtype)
                    )
                    setattr(module, pname, new_param)
            for bname, buf in module.named_buffers(recurse=False):
                if buf.device.type == "meta":
                    new_buf = torch.zeros_like(buf, device=device, dtype=buf.dtype)
                    module.register_buffer(bname, new_buf)

        layers[layer_idx] = layer.to(device)


def _find_linear_modules(module, prefix=""):
    """Find all nn.Linear modules and return {name: (parent, attr, module)}."""
    result = {}
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            result[full_name] = (module, name, child)
        else:
            result.update(_find_linear_modules(child, full_name))
    return result


def _load_lora_for_module(ql, layer_idx, module_name, assignment_lookup,
                           u_cache, per_expert_V, uv_bits, device):
    """Load LoRA (U, S, V) compensation for a specific module."""
    from utils.moe_utils import is_regular_expert, is_shared_expert, extract_expert_info

    # Determine expert_idx and wtype from module_name
    if is_regular_expert(module_name):
        idx, _ = extract_expert_info(module_name)
        expert_idx = idx if idx is not None else -3
    elif is_shared_expert(module_name):
        expert_idx = -2
    else:
        expert_idx = -1

    # Determine wtype
    wtype = None
    for wt in ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "w1", "w2", "w3"):
        if module_name == wt or module_name.endswith("." + wt):
            # Map Mixtral w1/w2/w3 to gate/up/down
            wtype_map = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
            wtype = wtype_map.get(wt, wt)
            break

    if wtype is None or expert_idx == -2:
        return  # No LoRA for this module

    key = (wtype, layer_idx, expert_idx)
    if key not in assignment_lookup:
        return

    group_id, wtype_key, local_idx = assignment_lookup[key]

    # Get shared U and S from cache
    U, S_shared = u_cache.get(wtype_key, group_id)
    if U is None:
        return

    # Get per-expert V
    if wtype_key in per_expert_V and per_expert_V[wtype_key][local_idx] is not None:
        v_data = per_expert_V[wtype_key][local_idx]
        V = _dequant_intN(
            v_data["V_int8"].to(device),
            v_data["V_scale"].to(device),
            uv_bits,
        ).half()

        # Check if this expert has its own S (from Hessian-weighted per-expert V)
        # The S is part of V's column norms in this case
        V_col_norm = V.norm(dim=0).clamp(min=1e-6)
        # Use shared S since the per-expert S is already baked into V
        S = S_shared
    else:
        return

    ql.load_lora(U, S, V, device=device)
