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
from .moe_block import GraphCompatibleMoeBlock

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

    def __init__(self, shared_matrices, uv_bits, device="cuda", u_bits=None):
        self._cache = {}
        _u_bits = u_bits if u_bits is not None else uv_bits
        for wtype, groups in shared_matrices.items():
            for gid, data in groups.items():
                if "U_fp16" in data:
                    U_fp16 = data["U_fp16"].half().to(device)
                elif "U_int4_packed" in data:
                    orig_rows = int(data["U_orig_rows"].item())
                    U_int8 = _unpack_int4(data["U_int4_packed"].to(device), orig_rows)
                    U_fp16 = _dequant_intN(U_int8, data["U_scale"].to(device), 4).half()
                else:
                    U_fp16 = _dequant_intN(
                        data["U_int8"].to(device),
                        data["U_scale"].to(device),
                        _u_bits,
                    ).half()
                # S is stored in old format only; new format pre-fuses it into SV
                S = data["S"].half().to(device) if "S" in data else None
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


def _unpack_int4(packed, orig_rows):
    """Unpack ((rows+1)//2, cols) uint8 → (orig_rows, cols) int8 in [-7, 7]."""
    lo = (packed & 0xF).to(torch.int8) - 7
    hi = ((packed >> 4) & 0xF).to(torch.int8) - 7
    interleaved = torch.stack([lo, hi], dim=1).reshape(-1, packed.shape[1])
    return interleaved[:orig_rows]


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
# MoE block replacement (CUDA Graph compatibility)
# ---------------------------------------------------------------------------
def _build_global_u_pool(u_cache):
    """Build cross-layer shared U matrix pools from SharedUCache.

    Returns one contiguous tensor per projection type containing all unique
    cluster U matrices.  All MoE layers reference the SAME tensor, so the GPU
    L2 cache sees repeated accesses to identical HBM addresses → cross-layer
    reuse without duplication.

    Returns:
        pools:   {'gate': (hidden_dim, K_total*rank), 'up': ..., 'down': ...}
        id_maps: {'gate': {(wtype_key, group_id): col_index}, ...}
    """
    # Group u_cache entries by wtype
    wtype_map = {'gate_proj': [], 'up_proj': [], 'down_proj': []}
    for (wtype_key, gid), (U_fp16, _S) in u_cache._cache.items():
        if wtype_key in wtype_map:
            wtype_map[wtype_key].append((gid, U_fp16))

    pools = {}
    id_maps = {}
    for wtype_key in ('gate_proj', 'up_proj', 'down_proj'):
        items = wtype_map[wtype_key]
        if not items:
            continue
        # Sort by group_id for deterministic ordering
        items.sort(key=lambda x: x[0])
        pool_key = wtype_key.split('_')[0]  # 'gate', 'up', 'down'
        pools[pool_key]   = torch.cat([u for _, u in items], dim=1)  # (d_in, K*r)
        id_maps[pool_key] = {(wtype_key, gid): i for i, (gid, _) in enumerate(items)}

    return pools, id_maps


def _install_global_u_pool(model, u_cache):
    """Build global U pool and install on all MoE blocks.

    Handles both Qwen (`layer.mlp`) and Mixtral (`layer.block_sparse_moe`)
    container attrs; iterates both and installs on whichever is a wrapped
    GraphCompatibleMoeBlock.
    """
    pools, id_maps = _build_global_u_pool(u_cache)
    m = getattr(model, "model", model)
    for layer in m.layers:
        for attr in ("mlp", "block_sparse_moe"):
            moe = getattr(layer, attr, None)
            if moe is not None and hasattr(moe, "set_global_pool"):
                moe.set_global_pool(pools, id_maps)
                break


def _replace_moe_blocks(model):
    """Replace MoE blocks with our graph-compatible wrapper.

    Handles Qwen2Moe, Qwen3Moe, and Mixtral through a single
    GraphCompatibleMoeBlock — the wrapper uses getattr for arch-specific
    attributes (shared_expert, norm_topk_prob) and aliases w1/w3/w2 to
    gate_proj/up_proj/down_proj so downstream code is uniform.
    """
    qwen_types = []
    mixtral_types = []
    try:
        from transformers.models.qwen2_moe.modeling_qwen2_moe import (
            Qwen2MoeSparseMoeBlock,
        )
        qwen_types.append(Qwen2MoeSparseMoeBlock)
    except ImportError:
        pass
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeSparseMoeBlock,
        )
        qwen_types.append(Qwen3MoeSparseMoeBlock)
    except ImportError:
        pass
    try:
        from transformers.models.mixtral.modeling_mixtral import (
            MixtralSparseMoeBlock,
        )
        mixtral_types.append(MixtralSparseMoeBlock)
    except ImportError:
        pass

    if not qwen_types and not mixtral_types:
        return
    qwen_tuple = tuple(qwen_types)
    mixtral_tuple = tuple(mixtral_types)

    m = getattr(model, "model", model)
    for layer in m.layers:
        for attr in ("mlp", "block_sparse_moe"):
            mod = getattr(layer, attr, None)
            if mod is None:
                continue
            # Qwen wrapper handles BOTH archs (getattr shared_expert/norm_topk_prob
            # defaults, w1/w3/w2 aliasing, dtype-aware LoRA). Routing Mixtral
            # through it inherits all decode-speed opts (CUDA Graph, grouped-GEMV,
            # dual-stream, precompute-share, silu fuse).
            if (qwen_tuple and isinstance(mod, qwen_tuple)) or \
               (mixtral_tuple and isinstance(mod, mixtral_tuple)):
                setattr(layer, attr, GraphCompatibleMoeBlock(mod))
                break


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

    # E11 method writes packed VQ residuals + attention GPTQ directly into
    # cross_layer_info.pt rather than a separate glorcq_model.pt. Detect this
    # and build a synthetic layers_data dict so _replace_linear_layers can
    # be reused unchanged.
    is_e11 = False
    if os.path.exists(cross_layer_path):
        _cli_probe = torch.load(cross_layer_path, map_location="cpu", weights_only=False)
        if _cli_probe.get("config", {}).get("method") == "tileq_glorcq_e11":
            is_e11 = True
            cross_layer_info = _cli_probe
        else:
            del _cli_probe

    if not is_e11 and not os.path.exists(glorcq_model_path):
        raise FileNotFoundError(
            f"glorcq_model.pt not found in {model_path}. "
            "This model was likely saved in fake-quant mode. "
            "Use standard HF loading instead."
        )

    # Create model on CPU with empty weights
    # For E11: load from safetensors so non-quantized params (norms, embeddings,
    # router gates) get their real fp16 values. The quantized linears are still
    # replaced below by GLoRCQLinear.
    #
    # NOTE: explicit device_map='cpu' is critical for large models (Mixtral ~90GB
    # fp16). Without it, `low_cpu_mem_usage=True` places weights on the first
    # visible CUDA device by default → OOM before we can even replace linears.
    if is_e11:
        # If the checkpoint was saved with `--strip_fp16_quantized`, the safetensors
        # only contain non-quantized params (embeddings, norms, router gates,
        # attention when attn_bits=16, etc.). `from_pretrained` will still work —
        # missing keys land in `missing_keys` and quantized-Linear weights stay at
        # whatever `low_cpu_mem_usage` default-inits them to. They get overwritten
        # by GLoRCQLinear in `_replace_linear_layers` below.
        stripped_marker = os.path.join(model_path, ".stripped_real_quant")
        is_stripped = os.path.exists(stripped_marker)
        if is_stripped:
            print(f"[GLoRCQ] Detected stripped real-quant checkpoint: fp16 weights for "
                  f"quantized layers missing (will be reconstructed from cross_layer_info.pt)")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, trust_remote_code=True,
            torch_dtype=torch.float16, low_cpu_mem_usage=True,
            device_map='cpu',
        )
    else:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, torch_dtype=torch.float16,
            )
    model.seqlen = 4096

    if is_e11:
        # ----- E11 path: build layers_data from cross_layer_info -----
        cl_config       = cross_layer_info["config"]
        shared_matrices = cross_layer_info["shared_matrices"]
        per_expert_V    = cross_layer_info["per_expert_V"]
        assignments     = cross_layer_info["assignments"]
        vq_residuals    = cross_layer_info.get("vq_residuals", {})
        attn_gptq_packs = cross_layer_info.get("attn_gptq_packs", {})

        # Container attr depends on MoE architecture (Qwen: `mlp`, Mixtral:
        # `block_sparse_moe`). Look up from utils/moe_utils.
        from utils.moe_utils import get_moe_config
        expert_container = get_moe_config(config.model_type).get(
            'expert_container') or 'mlp'

        layers_data = {}
        # Routing experts → vq4 packed
        for wt in vq_residuals:
            for entry, vq in zip(assignments[wt], vq_residuals[wt]):
                if vq is None:
                    continue
                li, ei = entry['layer'], entry['expert']
                mod_name = f"{expert_container}.experts.{ei}.{wt}"
                layers_data.setdefault(li, {})[mod_name] = {
                    "vq4_Q_rotated":     vq.get('Q_rotated'),
                    "vq4_codes":         vq.get('codes'),
                    "vq4_codes_packed":  vq.get('codes_packed'),
                    "vq4_codes_n_vecs":  vq.get('codes_n_vecs'),
                    "vq4_centroids":     vq['centroids'],
                    "vq4_perm":          vq['perm'],
                    "vq4_diag_signs":    vq['diag_signs'],
                    "vq4_vdim":          vq['vdim'],
                    "vq4_in_d":          vq['in_d'],
                    "vq4_out_d":         vq['out_d'],
                    "vq4_rotate_size":   vq.get('rotate_size', 256),
                    "vq4_partial_size":  vq.get('partial_size', 256),
                }
        # Attention → GPTQ packed
        for (li, an), packed in attn_gptq_packs.items():
            layers_data.setdefault(li, {})[f"self_attn.{an}"] = packed

        # Build model_config-like dict
        model_config = {
            "uv_bits": cl_config.get("u_bits", 8),
            "u_bits":  cl_config.get("u_bits", 8),
            "sv_bits": cl_config.get("sv_bits", 8),
            "use_turboquant": False,
            "method": "tileq_glorcq_e11",
        }
    else:
        # 2. Load packed quantized weights
        print("[GLoRCQ] Loading glorcq_model.pt ...")
        model_data = torch.load(glorcq_model_path, map_location="cpu", weights_only=False)
        model_config = model_data["model_config"]
        layers_data = model_data["layers"]

        # 3. Load cross-layer info
        print("[GLoRCQ] Loading cross_layer_info.pt ...")
        cross_layer_info = torch.load(cross_layer_path, map_location="cpu", weights_only=False)
        shared_matrices = cross_layer_info["shared_matrices"]
        per_expert_V    = cross_layer_info["per_expert_V"]
        assignments     = cross_layer_info["assignments"]
        cl_config       = cross_layer_info["config"]

    uv_bits = cl_config.get("uv_bits", model_config.get("uv_bits", 8))
    u_bits  = cl_config.get("u_bits",  uv_bits)
    sv_bits = cl_config.get("sv_bits", uv_bits)

    # 4. Build SharedUCache
    print("[GLoRCQ] Building SharedUCache ...")
    u_cache = SharedUCache(shared_matrices, uv_bits, device=device, u_bits=u_bits)

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
        sv_bits=sv_bits,
    )

    # 6a. Load RHT signs for Hadamard rotation mode
    rotation_type = model_config.get("rotation_type", "qr")
    if rotation_type == "hadamard":
        from hadamard_rotation import generate_rht_signs
        from inference.quantized_linear import GLoRCQLinear
        print("[GLoRCQ] Loading RHT signs for Hadamard rotation ...")
        for _, m in model.named_modules():
            if isinstance(m, GLoRCQLinear) and m.quant_type == "turbo":
                signs = generate_rht_signs(m.turbo_dim, seed=m.turbo_seed, device=device)
                m._rht_signs = signs

    # 6b. Replace MoE blocks for CUDA Graph compatibility
    _replace_moe_blocks(model)

    # 6c. Install global U pool: all MoE layers share one copy per unique cluster
    #     → GPU L2 cache reuse across layers during decode (Belady-OPT: all hot)
    _install_global_u_pool(model, u_cache)

    # 7. Move non-linear components to device
    m = getattr(model, "model", model)
    for attr in ("embed_tokens", "norm", "rotary_emb"):
        if hasattr(m, attr):
            obj = getattr(m, attr)
            if obj is not None:
                setattr(m, attr, _materialize_meta_module(obj, device))
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


def _materialize_meta_module(module, device):
    """Move a module from meta device to *device*, zero-initializing all
    parameters and buffers.  If already on a real device, equivalent to
    ``module.to(device)``."""
    is_meta = any(
        p.device.type == "meta"
        for p in list(module.parameters()) + list(module.buffers())
    )
    if not is_meta:
        return module.to(device)
    module = module.to_empty(device=device)
    for p in module.parameters():
        if not p.is_meta:
            p.data.zero_()
    for b in module.buffers():
        if not b.is_meta:
            b.data.zero_()
    return module


def _replace_linear_layers(model, layers_data, assignments, per_expert_V,
                            u_cache, uv_bits, rotation_cache, device,
                            sv_bits=None):
    """
    Replace nn.Linear layers in the model with GLoRCQLinear.

    Traverses each decoder layer, finds matching modules in layers_data,
    and constructs GLoRCQLinear with the appropriate quantized weights
    and LoRA parameters.

    TurboQuant layers use runtime dequant via RotationCache (weights stay
    compressed on GPU). Falls back to pre-dequant if RotationCache is None.
    """
    # _dequant_intN is defined at module level above

    # Build lookup: (norm_wtype, layer_idx, expert_idx) → (group_id, raw_wtype, local_idx)
    # Downstream _load_lora_for_module normalizes wt keys via
    #   {w1: gate_proj, w2: down_proj, w3: up_proj}
    # so we key the lookup by the NORMALIZED name, while keeping the raw name
    # (w1/w2/w3 or gate_proj/up_proj/down_proj) inside the tuple so that
    # per_expert_V[raw_wtype] lookups still work.
    _WT_NORMALIZE = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
    assignment_lookup = {}
    for wtype, records in assignments.items():
        norm_wt = _WT_NORMALIZE.get(wtype, wtype)
        for li, rec in enumerate(records):
            key = (norm_wt, rec["layer"], rec["expert"])
            assignment_lookup[key] = (rec["group_id"], wtype, li)

    # Get model layers
    m = getattr(model, "model", model)
    layers = m.layers

    for layer_idx in range(len(layers)):
        if layer_idx not in layers_data:
            # No quantized data for this layer, move to device as-is
            layers[layer_idx] = _materialize_meta_module(layers[layer_idx], device)
            continue

        layer = layers[layer_idx]
        layer_data = layers_data[layer_idx]

        # Find all nn.Linear modules in this layer
        linear_modules = _find_linear_modules(layer)

        for module_name, (parent, attr_name, linear) in linear_modules.items():
            if module_name not in layer_data:
                # Two cases:
                # (a) Non-quantized module (router, norm, etc.) — keep as-is.
                # (b) Quantized MoE expert that was SKIPPED during quant (max_err
                #     over threshold — e.g. Qwen3 down_proj rank=16 tripped 202
                #     skips). Wrap in Fp16LinearShim so moe_block fast paths that
                #     call ``expert.gate_proj(x, precomputed_xU=...)`` don't crash
                #     with TypeError on the unrecognized kwarg.
                from utils.moe_utils import is_regular_expert
                materialized = _materialize_meta_module(linear, device)
                if is_regular_expert(module_name):
                    from inference.quantized_linear import Fp16LinearShim
                    setattr(parent, attr_name, Fp16LinearShim(materialized))
                else:
                    setattr(parent, attr_name, materialized)
                continue

            packed = layer_data[module_name]

            # Determine quant type
            if "vq4_Q_rotated" in packed or "vq4_codes" in packed or "vq4_codes_packed" in packed:
                quant_type = "vq4"
            elif "qweight_int" in packed:
                quant_type = "gptq"
            elif "packed_indices" in packed:
                quant_type = "turbo"
            elif "weight_quant" in packed:
                quant_type = packed.get("quant_method", "unknown")
            else:
                # Unknown format, skip
                setattr(parent, attr_name, _materialize_meta_module(linear, device))
                continue

            in_f = linear.in_features if hasattr(linear, "in_features") else packed.get("dim", 0)
            out_f = linear.out_features if hasattr(linear, "out_features") else 0
            if in_f == 0 or out_f == 0:
                # Try to infer from packed data
                if "qweight_int" in packed:
                    out_f, in_f = packed["qweight_int"].shape
                elif "weight_quant" in packed:
                    out_f, in_f = packed["weight_quant"].shape
                elif "vq4_Q_rotated" in packed or "vq4_codes" in packed or "vq4_codes_packed" in packed:
                    in_f  = packed["vq4_in_d"]
                    out_f = packed["vq4_out_d"]

            ql = GLoRCQLinear(in_f, out_f)

            # Load quantized weights
            if quant_type == "vq4":
                ql.load_vq4(
                    Q_rotated=packed.get("vq4_Q_rotated"),
                    codes=packed.get("vq4_codes"),
                    codes_packed=packed.get("vq4_codes_packed"),
                    codes_n_vecs=packed.get("vq4_codes_n_vecs"),
                    centroids=packed["vq4_centroids"],
                    perm=packed["vq4_perm"],
                    diag_signs=packed["vq4_diag_signs"],
                    vdim=packed["vq4_vdim"],
                    in_d=packed["vq4_in_d"],
                    out_d=packed["vq4_out_d"],
                    rotate_size=packed.get("vq4_rotate_size", 256),
                    partial_size=packed.get("vq4_partial_size", 256),
                    device=device,
                )
            elif quant_type == "gptq":
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
                sv_bits=sv_bits,
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
                           u_cache, per_expert_V, uv_bits, device,
                           sv_bits=None):
    """Load LoRA (U, SV) compensation for a specific module.

    Supports both new format (SV_int8 key, pre-fused SV) and old format
    (V_int8 + S in shared_matrices) for backward compatibility.
    """
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

    # Get per-expert SV (new format) or V+S (old format)
    if wtype_key in per_expert_V and per_expert_V[wtype_key][local_idx] is not None:
        v_data = per_expert_V[wtype_key][local_idx]
        Sa = v_data["Sa"].to(device).half() if "Sa" in v_data else None
        if "SV_int4_packed" in v_data:
            # INT4 packed format: real 4-bit packing, 50% storage vs int8
            orig_rows = int(v_data["SV_orig_rows"].item())
            SV_int8 = _unpack_int4(v_data["SV_int4_packed"].to(device), orig_rows)
            SV = _dequant_intN(SV_int8, v_data["SV_scale"].to(device), 4).half()
            ql.load_sv(U, SV, Sa=Sa, device=device)
        elif "SV_fp16" in v_data:
            # fp16 format: directly stored without quantization
            SV = v_data["SV_fp16"].to(device).half()
            ql.load_sv(U, SV, Sa=Sa, device=device)
        elif "SV_int8" in v_data:
            # int8 (or other nbits stored in int8 container) format
            _sv_bits = sv_bits if sv_bits is not None else uv_bits
            SV = _dequant_intN(
                v_data["SV_int8"].to(device),
                v_data["SV_scale"].to(device),
                _sv_bits,
            ).half()
            ql.load_sv(U, SV, Sa=Sa, device=device)
        else:
            # Old format: V is stored separately; S is in u_cache
            V = _dequant_intN(
                v_data["V_int8"].to(device),
                v_data["V_scale"].to(device),
                uv_bits,
            ).half()
            S = S_shared
            ql.load_lora(U, S, V, Sa=Sa, device=device)
    else:
        return

    # Tag with cluster id for cluster-parallel LoRA in MoE block
    ql.cluster_id = (wtype_key, group_id)
