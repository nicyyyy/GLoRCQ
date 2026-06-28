"""
tileq1d/run_tileq1d.py

TileQ 1D + cross-layer share pipeline.

Key difference from SOTA (run_quantize.py):
  Stage 1 (calibration_only=True): collect Hessian/act_scale, skip MoE quantization.
           MoE weight_quant = weight_orig (placeholder).
  Stage 2 (cluster_on_original=True): cluster on W_orig (TileQ-style).
  Stage 3+4 (fit_on_original=True): SVD of W_orig → lora ≈ W_orig.
           Write-back: model.weight = weight_orig + lora.
  Stage 5: R_k = weight_orig - lora  (genuine low-rank residual, much smaller)
           Q(R_k) = TurboQuant(R_k)  (2-bit of a cleaner signal)
           model.weight = Q(R_k) + lora

Attention layers: still run GPTQ normally (unchanged from SOTA).
"""

import argparse
import gc
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM
from transformers.models.mixtral.modeling_mixtral    import MixtralForCausalLM
from transformers.models.llama.modeling_llama        import LlamaForCausalLM
from transformers.models.opt.modeling_opt            import OPTForCausalLM

from utils.get_calib_data import get_wikitext2_
from cross_layer_share import (
    cluster_residuals,
    compute_shared_and_reconstruct,
    compute_avg_bits,
    save_cross_layer_info,
    assign_down_importance_ranks,
    quantize_lora_residuals,
)
from joint_optim import quantize_joint, requantize_attention_records
from turbo_weight_quantizer import TurboWeightQuantizer

DEV   = torch.device("cuda")
qtype = torch.float16


def get_blocks(model):
    if isinstance(model, LlamaForCausalLM) or \
            model.__class__.__name__ == "LlamaForCausalLM":
        return model.model.layers
    elif isinstance(model, OPTForCausalLM):
        return model.model.decoder.layers
    elif isinstance(model, (MixtralForCausalLM, Qwen2MoeForCausalLM)):
        return model.model.layers
    else:
        raise NotImplementedError(f"Unsupported model type: {type(model)}")


def main(args):
    u_bits     = args.u_bits if args.u_bits is not None else (args.uv_bits if args.uv_bits != 8 else 8)
    sv_bits    = args.sv_bits if args.sv_bits is not None else (args.uv_bits if args.uv_bits != 8 else 4)
    u_bits_attn  = args.u_bits_attn  if args.u_bits_attn  is not None else u_bits
    sv_bits_attn = args.sv_bits_attn if args.sv_bits_attn is not None else sv_bits

    print("=" * 60)
    print("  TileQ 1D + Cross-Layer Share Pipeline")
    print("=" * 60)
    print(f"  model_path:   {args.model_path}")
    print(f"  output_path:  {args.output_path}")
    print(f"  qbit={args.qbit}  groupsize={args.groupsize}  nsamples={args.nsamples}")
    print(f"  rank={args.rank}  rank_down={args.rank_down}  rank_attn={args.rank_attn}")
    print(f"  rank_cluster={args.rank_cluster}  G_moe={args.G_moe}  G_attn={args.G_attn}")
    print(f"  n_iter={args.n_iter}  n_lora_iter={getattr(args, 'n_lora_iter', 1)}")
    print(f"  u_bits={u_bits}  sv_bits={sv_bits}")
    print(f"  u_bits_attn={u_bits_attn}  sv_bits_attn={sv_bits_attn}")
    print(f"  w_clip={args.w_clip}  hessian_svd={args.hessian_svd}  recon_weight={args.recon_weight}")
    print(f"  [TileQ 1D] calibration_only=True, cluster_on_original=True, fit_on_original=True")
    print("=" * 60)

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, config=config, trust_remote_code=True,
        torch_dtype=qtype, low_cpu_mem_usage=True,
    )
    model.eval()
    enc = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    model.seqlen = 4096

    layers = get_blocks(model)

    dataloader, _, _ = get_wikitext2_(args.nsamples, 0, model.seqlen, args.model_path)

    # -----------------------------------------------------------------------
    # Stage 0: Calibration-only quantization
    # Collect Hessian/act_scale; MoE weight_quant = weight_orig (placeholder).
    # Attention layers still run full GPTQ.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  Stage 0: Calibration (GPTQ attn + collect Hessian, skip MoE quant)")
    print("=" * 60)
    all_records = quantize_joint(
        model, layers, dataloader, args,
        use_turboquant=True,
        model_type=config.model_type,
        calibration_only=True,
    )
    gc.collect()
    torch.cuda.empty_cache()

    n_lora_iter = getattr(args, 'n_lora_iter', 1)

    for _lora_round in range(n_lora_iter):
        if _lora_round > 0:
            print("\n" + "=" * 60)
            print(f"  LoftQ Round {_lora_round + 1}: re-quantize attention with updated LoRA")
            print("=" * 60)
            requantize_attention_records(all_records, DEV, args)
            gc.collect()
            torch.cuda.empty_cache()

        # -------------------------------------------------------------------
        # Stage 2: Grassmannian clustering on original weights (TileQ-style)
        # -------------------------------------------------------------------
        _rank_down_eff = args.rank_down
        if getattr(args, 'rank_down_high', None) and getattr(args, 'rank_down_low', None):
            assign_down_importance_ranks(
                all_records,
                rank_down_high=args.rank_down_high,
                rank_down_low=args.rank_down_low,
                topk_frac=getattr(args, 'rank_down_topk', 0.1),
            )

        print("\n" + "=" * 60)
        print(f"  Stage 2: Cluster on W_orig (TileQ-style)"
              + (f" (round {_lora_round + 1}/{n_lora_iter})" if n_lora_iter > 1 else ""))
        print("=" * 60)
        assignments, wtype_indices = cluster_residuals(
            all_records, args.rank, args.G_moe, args.G_attn, seed=args.seed,
            share_attn=args.share_attn, hessian_svd=args.hessian_svd,
            recon_weight=args.recon_weight,
            rank_cluster=args.rank_cluster if args.rank_cluster > 0 else None,
            rank_attn=args.rank_attn, rank_down=_rank_down_eff,
            cluster_on_original=True,
        )
        gc.collect()

        # -------------------------------------------------------------------
        # Stage 3+4: Shared U + V fitted to W_orig (not residual)
        # Write-back: model.weight = weight_orig + lora
        # -------------------------------------------------------------------
        print("\n" + "=" * 60)
        print(f"  Stage 3+4: SVD(W_orig) → shared U + per-expert V"
              + (f" (round {_lora_round + 1}/{n_lora_iter})" if n_lora_iter > 1 else ""))
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
            fit_on_original=True,
        )
        gc.collect()
        torch.cuda.empty_cache()

        # -------------------------------------------------------------------
        # Stage 5: Quantize genuine residual R_k = W_orig - lora
        # model.weight goes from (weight_orig + lora) to (Q(R_k) + lora)
        # -------------------------------------------------------------------
        print("\n" + "=" * 60)
        print(f"  Stage 5: TurboQuant(R_k = W_orig - lora)"
              + (f" (round {_lora_round + 1}/{n_lora_iter})" if n_lora_iter > 1 else ""))
        print("=" * 60)
        turbo_q = TurboWeightQuantizer(
            nbits=args.qbit, device=DEV,
            rotation_type=getattr(args, 'rotation_type', 'hadamard'),
        )
        quantize_lora_residuals(all_records, turbo_q, layers, DEV)
        del turbo_q
        gc.collect()
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Bit-width statistics
    # -----------------------------------------------------------------------
    compute_avg_bits(
        all_records, assignments, wtype_indices,
        shared_matrices, args.rank, args.groupsize, nbits=args.qbit,
        uv_bits=args.uv_bits, use_turboquant=True,
        u_bits=u_bits, sv_bits=sv_bits, u_fp16=args.u_fp16,
        rank_attn=args.rank_attn, rank_down=args.rank_down,
        u_bits_attn=u_bits_attn, sv_bits_attn=sv_bits_attn,
        sv_bits_down=args.sv_bits_down,
        u_fp16_attn=args.u_fp16_attn,
    )

    # -----------------------------------------------------------------------
    # Save (fake-quant: fp16 model + cross-layer metadata)
    # -----------------------------------------------------------------------
    save_config = vars(args)
    save_config["pipeline"] = "tileq1d"
    print(f"\n[save] Saving model → {args.output_path}")
    model.save_pretrained(args.output_path)
    enc.save_pretrained(args.output_path)
    save_cross_layer_info(
        args.output_path, save_config,
        assignments, wtype_indices,
        shared_matrices, per_expert_V,
        all_records,
    )
    print("\n✓ Done.")


def parse_args():
    p = argparse.ArgumentParser(description="TileQ 1D + cross-layer share pipeline")
    p.add_argument("--model_path",  type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--rank",        type=int, default=32)
    p.add_argument("--rank_attn",   type=int, default=None)
    p.add_argument("--rank_down",   type=int, default=None)
    p.add_argument("--rank_down_high", type=int, default=None)
    p.add_argument("--rank_down_low",  type=int, default=None)
    p.add_argument("--rank_down_topk", type=float, default=0.1)
    p.add_argument("--rank_cluster",   type=int, default=0)
    p.add_argument("--qbit",        type=int, default=2)
    p.add_argument("--groupsize",   type=int, default=128)
    p.add_argument("--nsamples",    type=int, default=128)
    p.add_argument("--n_iter",      type=int, default=5)
    p.add_argument("--n_lora_iter", type=int, default=1)
    p.add_argument("--G_moe",       type=int, default=128)
    p.add_argument("--G_attn",      type=int, default=4)
    p.add_argument("--uv_bits",     type=int, default=8)
    p.add_argument("--u_bits",      type=int, default=None)
    p.add_argument("--sv_bits",     type=int, default=None)
    p.add_argument("--sv_bits_down", type=int, default=None)
    p.add_argument("--u_bits_attn",  type=int, default=None)
    p.add_argument("--sv_bits_attn", type=int, default=None)
    p.add_argument("--sv_topk",     type=int, default=None)
    p.add_argument("--u_fp16",      action="store_true", default=False)
    p.add_argument("--u_fp16_attn", action="store_true", default=False)
    p.add_argument("--w_clip",      action="store_true", default=False)
    p.add_argument("--hessian_svd", action="store_true", default=False)
    p.add_argument("--recon_weight", type=float, default=0.7)
    p.add_argument("--percdamp",    type=float, default=0.01)
    p.add_argument("--sym",         action="store_true", default=False)
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--analyze",     action="store_true", default=False)
    p.add_argument("--share_attn",  action="store_true", default=False)
    p.add_argument("--search_act_alpha", action="store_true", default=False)
    p.add_argument("--rotation_type", type=str, default="hadamard",
                   choices=["qr", "hadamard"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.rank_down = args.rank_down or args.rank
    args.rank_attn = args.rank_attn or args.rank
    main(args)
