"""
GLoRCQ speed evaluation script.

Compares standard inference vs CUDA Graph accelerated inference
on a GLoRCQ real-quantized model.

Usage:
  uv run python glorcq/inference/eval_speed.py \
      --model_path ./output/...-glorcq-... \
      --batch_size 1 --prompt_len 128 --gen_len 128
"""

import argparse
import sys
import os

# Ensure glorcq is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_GLORCQ_ROOT = os.path.dirname(_HERE)
_PROJ_ROOT = os.path.dirname(_GLORCQ_ROOT)
for p in [_GLORCQ_ROOT, _PROJ_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="GLoRCQ speed benchmark: standard vs CUDA Graph inference"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to GLoRCQ quantized model directory")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for generation (default: 1)")
    parser.add_argument("--prompt_len", type=int, default=128,
                        help="Random prompt length in characters (default: 128)")
    parser.add_argument("--gen_len", type=int, default=128,
                        help="Number of tokens to generate (default: 128)")
    parser.add_argument("--max_seq_len", type=int, default=2048,
                        help="Maximum sequence length for CUDA Graph (default: 2048)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (default: cuda:0)")
    parser.add_argument("--real_quant", action="store_true", default=True,
                        help="Load real-quantized model (default: True)")
    parser.add_argument("--no_real_quant", dest="real_quant", action="store_false",
                        help="Load fake-quantized model (standard HF)")
    parser.add_argument("--graph_only", action="store_true", default=False,
                        help="Only run the CUDA-graph path; skip the standard baseline")
    args = parser.parse_args()

    # Load model
    if args.real_quant:
        from inference.model_builder import load_glorcq_model
        model = load_glorcq_model(args.model_path, device=args.device)
    else:
        from transformers import AutoModelForCausalLM, AutoConfig
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, config=config, trust_remote_code=True,
            torch_dtype=torch.float16, low_cpu_mem_usage=True,
        ).to(args.device)
        model.eval()

    # Load tokenizer. Prefer the local model_path (fake/real quant ckpt dir usually
    # ships tokenizer files) so this works on machines that don't have the original
    # base model at the path stored in cross_layer_info.pt (quant-time recorded).
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, use_fast=False, trust_remote_code=True,
        )
    except Exception:
        try:
            cross_layer_info = torch.load(
                os.path.join(args.model_path, "cross_layer_info.pt"),
                map_location="cpu", weights_only=False,
            )
            original_model = cross_layer_info["config"].get("model_path", args.model_path)
        except (FileNotFoundError, KeyError):
            original_model = args.model_path
        tokenizer = AutoTokenizer.from_pretrained(
            original_model, use_fast=False, trust_remote_code=True,
        )

    # Run benchmark
    from inference.graph_wrapper import run_speed_benchmark
    results = run_speed_benchmark(
        model, tokenizer,
        max_batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        prompt_len=args.prompt_len,
        gen_len=args.gen_len,
        skip_standard=args.graph_only,
    )


if __name__ == "__main__":
    main()
