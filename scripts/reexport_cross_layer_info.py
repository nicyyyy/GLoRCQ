"""Standalone re-export of cross_layer_info.pt for an existing E11 model.

Loads Phase 1 cache, re-runs Phase 2 (cross-layer SVD with int8_lora_v) to
populate WR with the same U/V/Si/Sa tensors as the original run, then calls
_export_real_quant_pack. Phase 3 (VQ residual) is NOT re-run because the
exported file only needs U/V/Si/Sa.

Usage:
    python scripts/reexport_cross_layer_info.py \
        --output_path /home/qyyang/resource_dir/GLoRCQ_out/e11_int8lora_v_realexp \
        --model_path Qwen/Qwen1.5-MoE-A2.7B \
        --phase1_cache_path /home/qyyang/resource_dir/GLoRCQ_out/e10_tileq_glorcq_phase1_cache.pt
"""

import argparse
import os
import sys
import torch

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from collections import defaultdict
from run_quantize import (
    fill_phase2, _export_real_quant_pack, _wr_to_dev, DEV,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output_path', required=True)
    p.add_argument('--model_path', default='Qwen/Qwen1.5-MoE-A2.7B')
    p.add_argument('--phase1_cache_path', required=True)
    p.add_argument('--qbit', type=int, default=2)
    p.add_argument('--fix_rank', type=int, default=32)
    p.add_argument('--G', type=int, default=128)
    p.add_argument('--group_size', type=int, default=128)
    p.add_argument('--lora_bit', type=int, default=16)
    p.add_argument('--lora_iter', type=int, default=8)
    p.add_argument('--attn_bits', type=int, default=4)
    p.add_argument('--int8_lora_v', action='store_true', default=True)
    args = p.parse_args()

    print(f"Loading Phase 1 cache: {args.phase1_cache_path}", flush=True)
    cached = torch.load(args.phase1_cache_path, map_location='cpu', weights_only=False)
    WR              = cached['WR']
    all_expert_recs = cached['all_expert_recs']
    quant_infos     = cached.get('quant_infos', defaultdict(float))
    _wr_to_dev(WR, DEV)
    print(f"  Loaded {len(all_expert_recs)} MoE expert records.", flush=True)

    # Detect wtypes
    first_names = [n for d in WR if d for n in d.keys()][:30]
    if any('w1' in n for n in first_names):
        wtypes = ['w1', 'w2', 'w3']
    else:
        wtypes = ['gate_proj', 'up_proj', 'down_proj']
    print(f"  wtypes: {wtypes}", flush=True)

    print("Running Phase 2 (cross-layer SVD with int8_lora_v) ...", flush=True)
    fill_phase2(
        WR, all_expert_recs, args.fix_rank, args.lora_bit, args.lora_iter,
        args.qbit, args.G, quant_infos, wtypes,
        int8_lora=False, int8_lora_v=args.int8_lora_v,
    )

    print("Exporting cross_layer_info.pt ...", flush=True)
    _export_real_quant_pack(
        WR, args.output_path, args.model_path,
        fix_rank=args.fix_rank, G=args.G, qbit=args.qbit,
        group_size=args.group_size, attn_bits=args.attn_bits,
        int8_lora=False, int8_lora_v=args.int8_lora_v,
    )
    print("✓ Done.")


if __name__ == '__main__':
    main()
