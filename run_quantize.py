"""
run_quantize.py

Entry point for the GLoRCQ quantization + LoRA pipeline
with cross-layer shared U.

GLoRCQ = Global shared Low-Rank Compensation for Quantization

Pipeline:
  Stage 1 : quantize_joint()            — alternating quantization + Hessian-SVD per layer
  Stage 2 : cluster_residuals()         — Grassmannian clustering
  Stage 3+4: compute_shared_and_reconstruct() — shared U + fake-quant
  Stage 5 : compute_avg_bits()          — bit-width stats

Usage:
  python glorcq/run_quantize.py \\
      --model_path Qwen/Qwen1.5-MoE-A2.7B \\
      --output_path ./output/glorcq_test \\
      --qbit 2 --w_clip --rank 64 \\
      --G_moe 128 --G_attn 4 \\
      --nsamples 128 --n_iter 3
"""

import argparse
import gc
import sys
import os

# Ensure glorcq/ is on sys.path (needed when run as: python glorcq/run_quantize.py)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.qwen2_moe.modeling_qwen2_moe  import Qwen2MoeForCausalLM
from transformers.models.mixtral.modeling_mixtral       import MixtralForCausalLM
from transformers.models.llama.modeling_llama           import LlamaForCausalLM
from transformers.models.opt.modeling_opt               import OPTForCausalLM

from utils.get_calib_data import get_wikitext2_
from cross_layer_share import (
    cluster_residuals,
    compute_shared_and_reconstruct,
    compute_avg_bits,
    save_cross_layer_info,
    save_real_quant,
)

# New Stage 1 implementation
from joint_optim import quantize_joint, requantize_attention_records

DEV   = torch.device("cuda")
qtype = torch.float16


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------
def get_blocks(model):
    """Return the list of transformer decoder layers."""
    if isinstance(model, LlamaForCausalLM) or \
            model.__class__.__name__ == "LlamaForCausalLM":
        return model.model.layers
    elif isinstance(model, OPTForCausalLM):
        return model.model.decoder.layers
    elif isinstance(model, (MixtralForCausalLM, Qwen2MoeForCausalLM)):
        return model.model.layers
    else:
        raise NotImplementedError(f"Unsupported model type: {type(model)}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_joint_quant(args):
    print("\n" + "=" * 60)
    print(f"  GLoRCQ: Quantization + LoRA")
    print(f"  model    : {args.model_path}")
    print(f"  rank     : {args.rank}  n_iter: {args.n_iter}")
    print(f"  G_moe    : {args.G_moe}   G_attn: {args.G_attn}")
    print(f"  share_attn: {args.share_attn}  "
          f"({'G_attn=' + str(args.G_attn) if args.share_attn else 'each layer independent'})")
    print(f"  nbits    : {args.qbit}  sym={args.sym}  mse={args.w_clip}")
    print(f"  groupsize: {args.groupsize}   nsamples: {args.nsamples}")
    print(f"  no_lora  : {args.no_lora}  (skip Stage 2-4 if set)")
    print(f"  act_alpha: {args.act_alpha}  (activation equalization exponent)")
    if args.search_act_alpha:
        print(f"  search_act_alpha: ON  (per-module grid search, "
              f"default_alpha={args.act_alpha} as fallback)")
    u_bits  = args.u_bits  if args.u_bits  is not None else args.uv_bits
    sv_bits = args.sv_bits if args.sv_bits is not None else (args.uv_bits if args.uv_bits != 8 else 4)
    u_bits_attn  = args.u_bits_attn  if args.u_bits_attn  is not None else (u_bits  if u_bits  is not None else 8)
    sv_bits_attn = args.sv_bits_attn if args.sv_bits_attn is not None else (sv_bits if sv_bits is not None else 8)
    print(f"  uv_bits  : {args.uv_bits}  (legacy; u_bits={u_bits}, sv_bits={sv_bits})")
    print(f"  u_bits_attn={u_bits_attn}, sv_bits_attn={sv_bits_attn}")
    print(f"  early_stop_tol: {args.early_stop_tol}  (0=disabled)")
    print(f"  hessian_svd: {args.hessian_svd}  recon_weight: {args.recon_weight}")
    print(f"  use_turboquant: {args.use_turboquant}")
    print(f"  n_lora_iter: {getattr(args, 'n_lora_iter', 1)}  (LoftQ-style E2E iterations)")
    print(f"  real_quant: {args.real_quant}")
    print("=" * 60)

    # -- Load model --
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, config=config, trust_remote_code=True,
        torch_dtype=qtype, low_cpu_mem_usage=True,
    )
    model.eval()
    enc = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=False, trust_remote_code=True,
    )
    model.seqlen = 4096

    layers = get_blocks(model)

    # Calibration dataloader (WikiText-2)
    dataloader, _, _ = get_wikitext2_(args.nsamples, 0, model.seqlen,
                                      args.model_path)

    # -----------------------------------------------------------------------
    # Stage 1: Quantization + LoRA (sequential for TurboQuant, joint for GPTQ)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    quant_method = "TurboQuant" if args.use_turboquant else "GPTQ"
    if args.use_turboquant:
        print(f"  Stage 1: Hybrid GPTQ(attn, n_iter={args.n_iter}) "
              f"+ TurboQuant(MoE) {args.qbit}-bit")
    else:
        print(f"  Stage 1: Joint {quant_method} {args.qbit}-bit + Hessian-weighted LoRA"
              f" (n_iter={args.n_iter})")
    print("=" * 60)
    all_records = quantize_joint(model, layers, dataloader, args,
                                  use_turboquant=args.use_turboquant,
                                  model_type=config.model_type)
    gc.collect()
    torch.cuda.empty_cache()

    if args.no_lora:
        # Save pure quantized model Q(W) without cross-layer sharing
        print("\n[--no_lora] Skipping cross-layer LoRA sharing (Stage 2-4).")
        print(f"\n[save] Saving pure-quantized model → {args.output_path}")
        model.save_pretrained(args.output_path)
        enc.save_pretrained(args.output_path)
        print("\n✓ Done.")
        return

    n_lora_iter = getattr(args, 'n_lora_iter', 1)

    for _lora_round in range(n_lora_iter):
        if _lora_round > 0:
            # -----------------------------------------------------------------------
            # LoftQ Round 2+: re-quantize attention layers with updated LoRA
            # -----------------------------------------------------------------------
            print("\n" + "=" * 60)
            print(f"  LoftQ Round {_lora_round + 1}: re-quantize attention with updated LoRA")
            print("=" * 60)
            requantize_attention_records(all_records, DEV, args)
            gc.collect()
            torch.cuda.empty_cache()

        # -----------------------------------------------------------------------
        # Stage 2: Grassmannian clustering per weight type
        # -----------------------------------------------------------------------
        print("\n" + "=" * 60)
        print(f"  Stage 2: Grassmannian clustering"
              + (f" (LoftQ round {_lora_round + 1}/{n_lora_iter})" if n_lora_iter > 1 else ""))
        print("=" * 60)
        assignments, wtype_indices = cluster_residuals(
            all_records, args.rank, args.G_moe, args.G_attn, seed=args.seed,
            share_attn=args.share_attn, hessian_svd=args.hessian_svd,
            recon_weight=args.recon_weight,
            rank_cluster=args.rank_cluster if args.rank_cluster > 0 else None,
            rank_attn=args.rank_attn, rank_down=args.rank_down,
        )
        gc.collect()

        # -----------------------------------------------------------------------
        # Stage 3+4: Shared U (int8) + fake-quant reconstruction
        # -----------------------------------------------------------------------
        print("\n" + "=" * 60)
        print(f"  Stage 3+4: Shared U + fake-quant reconstruction"
              + (f" (LoftQ round {_lora_round + 1}/{n_lora_iter})" if n_lora_iter > 1 else ""))
        print("=" * 60)
        shared_matrices, per_expert_V = compute_shared_and_reconstruct(
            all_records, assignments, wtype_indices, layers, args.rank,
            analyze=args.analyze, uv_bits=args.uv_bits,
            u_bits=u_bits, sv_bits=sv_bits, sv_topk=args.sv_topk,
            u_fp16=args.u_fp16,
            hessian_svd=args.hessian_svd,
            rank_attn=args.rank_attn, rank_down=args.rank_down,
            u_bits_attn=u_bits_attn, sv_bits_attn=sv_bits_attn,
            sv_bits_down=args.sv_bits_down,
            u_fp16_attn=args.u_fp16_attn,
        )
        gc.collect()
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Stage 5: Average bit-width statistics for the whole model
    # -----------------------------------------------------------------------
    compute_avg_bits(
        all_records, assignments, wtype_indices,
        shared_matrices, args.rank, args.groupsize, nbits=args.qbit,
        uv_bits=args.uv_bits, use_turboquant=args.use_turboquant,
        u_bits=u_bits, sv_bits=sv_bits, u_fp16=args.u_fp16,
        rank_attn=args.rank_attn, rank_down=args.rank_down,
        u_bits_attn=u_bits_attn, sv_bits_attn=sv_bits_attn,
        sv_bits_down=args.sv_bits_down,
        u_fp16_attn=args.u_fp16_attn,
    )

    # -----------------------------------------------------------------------
    # Save model
    # -----------------------------------------------------------------------
    save_config = {
        "model_path": args.model_path,
        "rank":       args.rank,
        "rank_attn":  args.rank_attn,
        "G_moe":      args.G_moe,
        "G_attn":     args.G_attn,
        "share_attn": args.share_attn,
        "nbits":      args.qbit,
        "groupsize":  args.groupsize,
        "n_iter":     args.n_iter,
        "act_alpha":  args.act_alpha,
        "search_act_alpha": args.search_act_alpha,
        "use_turboquant": args.use_turboquant,
        "method":     "hybrid_gptq_attn_turboquant_moe" if args.use_turboquant
                      else "joint_gptq_hessian_svd",
        "uv_bits":      args.uv_bits,
        "u_bits":       u_bits,
        "sv_bits":      sv_bits,
        "u_bits_attn":  u_bits_attn,
        "sv_bits_attn":        sv_bits_attn,
        "sv_topk":             args.sv_topk,
        "u_fp16":       args.u_fp16,
    }

    if args.real_quant:
        # Real quant: save packed weights + LoRA params separately
        print(f"\n[save] Saving real-quantized model → {args.output_path}")
        save_real_quant(
            args.output_path, model, all_records, shared_matrices,
            per_expert_V, assignments, wtype_indices, args,
        )
        save_cross_layer_info(
            args.output_path, save_config,
            assignments, wtype_indices,
            shared_matrices, per_expert_V,
            all_records,
        )
        enc.save_pretrained(args.output_path)
    else:
        # Fake quant: write back W_approx as fp16 + save_pretrained
        print(f"\n[save] Saving model (W_approx = Q(W) + U@S@V.T, fp16)"
              f" → {args.output_path}")
        model.save_pretrained(args.output_path)
        enc.save_pretrained(args.output_path)
        save_cross_layer_info(
            args.output_path, save_config,
            assignments, wtype_indices,
            shared_matrices, per_expert_V,
            all_records,
        )

    print("\n✓ Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="GLoRCQ: Global shared Low-Rank Compensation for Quantization"
    )
    p.add_argument("--model_path",  type=str, required=True,
                   help="HuggingFace model path or local directory")
    p.add_argument("--output_path", type=str, required=True,
                   help="Output directory for fake-quantized model + metadata")
    p.add_argument("--rank",      type=int,   default=64,
                   help="LoRA / SVD rank for MoE expert layers (default: 64)")
    p.add_argument("--rank_attn", type=int,   default=None,
                   help="LoRA / SVD rank for attention layers (q/k/v/o_proj). "
                        "If None, falls back to --rank. Increase for better "
                        "attention quality (e.g. 256 or 512).")
    p.add_argument("--rank_down", type=int,   default=None,
                   help="LoRA / SVD rank for MoE down_proj layers. "
                        "If None, falls back to --rank. Can be set higher than "
                        "gate/up rank since down_proj is harder to compress.")
    p.add_argument("--rank_cluster", type=int, default=0,
                   help="Rank used for Stage 2 Grassmannian clustering only "
                        "(0=auto: min(rank,32)). Stage 3 reconstruction uses --rank.")
    p.add_argument("--G_moe",     type=int,   default=128,
                   help="Cluster count for MoE experts (gate/up/down_proj, "
                        "default: 128)")
    p.add_argument("--G_attn",    type=int,   default=4,
                   help="Cluster count for attention layers (q/k/v/o_proj, "
                        "default: 4)")
    p.add_argument("--groupsize", type=int,   default=128,
                   help="GPTQ column groupsize for scale factors (default: 128)")
    p.add_argument("--nsamples",  type=int,   default=128,
                   help="Number of WikiText-2 calibration samples (default: 128)")
    p.add_argument("--calib_batch_size", type=int, default=8,
                   help="Batch size for calibration forward passes (default: 8). "
                        "Reduce to lower GPU memory usage.")
    p.add_argument("--percdamp",  type=float, default=0.01,
                   help="GPTQ Hessian damping factor (default: 0.01)")
    p.add_argument("--qbit",   type=int,   default=4,
                   help="Weight quantization bits (default: 4)")
    p.add_argument("--sym",    action="store_true", default=False,
                   help="Use symmetric quantization (default: asymmetric)")
    p.add_argument("--w_clip", action="store_true", default=False,
                   help="Use MSE search for quantization clipping")
    p.add_argument("--seed",      type=int,   default=42,
                   help="Random seed for clustering (default: 42)")
    p.add_argument("--n_iter",    type=int,   default=3,
                   help="Alternating optimization iterations per layer "
                        "(default: 3; n_iter=1 ≈ tileq_1d GPTQ)")
    p.add_argument("--no_lora", action="store_true", default=False,
                   help="Skip cross-layer LoRA sharing (Stage 2-4); "
                        "save pure quantized model only")
    p.add_argument("--analyze", action="store_true", default=False,
                   help="Print per-expert residual reconstruction quality stats")
    p.add_argument("--act_alpha", type=float, default=0.6,
                   help="Activation equalization exponent (0=no equalization, "
                        "0.5=AWQ default; default: 0.5)")
    p.add_argument("--search_act_alpha", action="store_true", default=False,
                   help="Per-module grid search for act_alpha (attention only)")
    p.add_argument("--uv_bits", type=int, default=8,
                   help="Quantization bits for shared U and per-expert V "
                        "(default: 8; options: 2/4/8). Legacy: sets both u_bits and sv_bits.")
    p.add_argument("--u_bits", type=int, default=None,
                   help="Quantization bits for shared U (default: uv_bits or 8). "
                        "U is cross-layer shared, so higher precision is preferred.")
    p.add_argument("--sv_bits", type=int, default=None,
                   help="Quantization bits for per-expert SV (default: uv_bits or 4). "
                        "SV is per-expert private; lower precision trades quality for storage.")
    p.add_argument("--u_bits_attn", type=int, default=None,
                   help="U bits for attention layers (default: u_bits or 8). "
                        "Attention U is per-layer; higher precision recommended.")
    p.add_argument("--sv_bits_attn", type=int, default=None,
                   help="SV bits for attention layers (default: sv_bits or 8). "
                        "Attention SV is per-layer; higher precision recommended.")
    p.add_argument("--sv_bits_down", type=int, default=None,
                   help="SV bits for down_proj layers (default: sv_bits). "
                        "down_proj is harder to compress; can use lower bits with higher rank.")
    p.add_argument("--sv_topk", type=int, default=None,
                   help="Keep only top-k singular value columns (e.g. 32 out of rank=64). "
                        "Reduces LoRA storage and GEMV cost by rank/topk ratio.")
    p.add_argument("--u_fp16", action="store_true", default=False,
                   help="Store shared U in fp16 instead of int8 (no quantization error; "
                        "doubles U storage but U is amortized so overhead is small).")
    p.add_argument("--u_fp16_attn", action="store_true", default=False,
                   help="Store attention U in fp16 instead of quantized int (no quant error).")
    p.add_argument("--early_stop_tol", type=float, default=0,
                   help="Relative Frobenius improvement threshold for early stopping "
                        "in alternating optimization (default: 0; 0=disabled)")
    p.add_argument("--hessian_svd", action="store_true", default=True,
                   help="Use Hessian-weighted SVD in Stage 3 (default: True)")
    p.add_argument("--no_hessian_svd", dest="hessian_svd", action="store_false",
                   help="Disable Hessian-weighted SVD in Stage 3")
    p.add_argument("--recon_weight", type=float, default=0.0,
                   help="Cross-reconstruction error weight in clustering "
                        "(0=pure Grassmannian, 0.3 recommended to test)")
    p.add_argument("--use_turboquant", action="store_true", default=False,
                   help="Use TurboQuant vector quantizer instead of GPTQ "
                        "(random rotation + Lloyd-Max optimal codebook)")
    p.add_argument("--rotation_type", type=str, default="hadamard",
                   choices=["qr", "hadamard"],
                   help="Rotation type for TurboQuant (qr=full random orthogonal, "
                        "hadamard=Randomized Hadamard Transform, zero storage)")
    p.add_argument("--turbo_batch_size", type=int, default=0,
                   help="Batch size for TurboQuant MoE experts per wtype "
                        "(0=all at once, reduce if OOM)")
    p.add_argument("--real_quant", action="store_true", default=False,
                   help="Save real quantized weights (packed int + LoRA params) "
                        "instead of fake-quant fp16 W_approx")
    p.add_argument("--n_lora_iter", type=int, default=1,
                   help="LoftQ-style E2E iterations (default: 1 = current behavior; "
                        "2 = one extra re-quantize attention round after Stage 3). "
                        "Only affects attention layers (MoE TurboQuant has no Hessian).")
    attn_share_group = p.add_mutually_exclusive_group()
    attn_share_group.add_argument(
        "--share_attn", dest="share_attn", action="store_true",
        help="Enable cross-layer LoRA sharing for attention (q/k/v/o_proj) via "
             "Grassmannian clustering (G_attn groups). Default: disabled.",
    )
    attn_share_group.add_argument(
        "--no_share_attn", dest="share_attn", action="store_false",
        help="(Default) Each attention layer keeps its own independent LoRA U/V.",
    )
    p.set_defaults(share_attn=False)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_joint_quant(args)
