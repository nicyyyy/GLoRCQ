"""
GLoRCQ speed evaluation: standard inference vs CUDA Graph.

Loads a real-quantized GLoRCQ model and benchmarks generation speed
with and without CUDA Graph acceleration.

Usage:
    uv run python evaluate/eval_speed.py \
        --model_path ./output/model-glorcq \
        --prompt_len 128 --gen_len 128
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# Ensure project root is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)


def _make_random_input(tokenizer, prompt_len, batch_size, device):
    """Create random prompt input_ids of approximately *prompt_len* tokens."""
    import random
    import string

    alphabet = string.ascii_letters + string.digits + " "
    prompt = "".join(random.choice(alphabet) for _ in range(prompt_len * 4))
    inputs = tokenizer(
        prompt, return_tensors="pt", max_length=prompt_len, truncation=True,
    )
    input_ids = inputs.input_ids.to(device)
    if batch_size > 1:
        input_ids = input_ids.expand(batch_size, -1)
    return input_ids


@torch.no_grad()
def benchmark_standard(model, input_ids, gen_len, num_warmup, num_runs):
    """Benchmark standard ``model.generate()``."""
    for _ in range(num_warmup):
        model.generate(input_ids, max_new_tokens=gen_len, do_sample=False)
    torch.cuda.synchronize()

    durations = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(input_ids, max_new_tokens=gen_len, do_sample=False)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - t0)
    return durations


@torch.no_grad()
def benchmark_graph(wrapper, input_ids, gen_len, num_warmup, num_runs):
    """Benchmark CUDA-Graph-accelerated ``wrapper.generate()``."""
    for _ in range(num_warmup):
        wrapper.generate(input_ids, max_new_tokens=gen_len)
    torch.cuda.synchronize()

    durations = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        wrapper.generate(input_ids, max_new_tokens=gen_len)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - t0)
    return durations


def main():
    parser = argparse.ArgumentParser(
        description="GLoRCQ speed benchmark: standard vs CUDA Graph inference",
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to GLoRCQ real-quantized model directory")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prompt_len", type=int, default=128,
                        help="Approximate prompt length in tokens (default: 128)")
    parser.add_argument("--gen_len", type=int, default=128,
                        help="Number of tokens to generate (default: 128)")
    parser.add_argument("--max_seq_len", type=int, default=2048,
                        help="Maximum sequence length for CUDA Graph (default: 2048)")
    parser.add_argument("--num_warmup", type=int, default=2)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--skip_standard", action="store_true",
                        help="Skip standard (non-graph) benchmark")
    parser.add_argument("--skip_graph", action="store_true",
                        help="Skip CUDA Graph benchmark")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save results as JSON")
    args = parser.parse_args()

    # Load model (real-quant)
    from utils.model_loader import load_model_and_tokenizer

    print(f"Loading real-quant model from {args.model_path} ...")
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, device=args.device, real_quant=True,
    )

    input_ids = _make_random_input(
        tokenizer, args.prompt_len, args.batch_size, args.device,
    )
    print(f"Prompt tokens: {input_ids.shape[1]}, Generate: {args.gen_len} tokens, "
          f"Batch size: {args.batch_size}")

    results = {}

    # Phase 1: Standard generation
    if not args.skip_standard:
        print(f"\n[1/2] Standard generation (warmup={args.num_warmup}, "
              f"runs={args.num_runs}) ...")
        durations_std = benchmark_standard(
            model, input_ids, args.gen_len, args.num_warmup, args.num_runs,
        )
        tps_std = [args.gen_len / d for d in durations_std]
        mean_std, std_std = np.mean(tps_std), np.std(tps_std)
        print(f"  Standard: {mean_std:.1f} ± {std_std:.1f} tok/s")
        results["standard"] = {
            "mean_tps": float(mean_std),
            "std_tps": float(std_std),
            "durations": durations_std,
        }

    # Phase 2: CUDA Graph generation
    if not args.skip_graph:
        print(f"\n[2/2] CUDA Graph generation (warmup={args.num_warmup}, "
              f"runs={args.num_runs}) ...")
        from inference.graph_wrapper import GLoRCQGraphWrapper

        wrapper = GLoRCQGraphWrapper(
            model, max_batch_size=args.batch_size, max_seq_len=args.max_seq_len,
        )
        wrapper.capture_graph()

        durations_graph = benchmark_graph(
            wrapper, input_ids, args.gen_len, args.num_warmup, args.num_runs,
        )
        tps_graph = [args.gen_len / d for d in durations_graph]
        mean_graph, std_graph = np.mean(tps_graph), np.std(tps_graph)
        print(f"  Graph:    {mean_graph:.1f} ± {std_graph:.1f} tok/s")
        results["graph"] = {
            "mean_tps": float(mean_graph),
            "std_tps": float(std_graph),
            "durations": durations_graph,
        }

    # Peak GPU memory
    peak_mem_mb = torch.cuda.max_memory_allocated(args.device) / 1024 ** 2
    results["peak_gpu_memory_mb"] = float(peak_mem_mb)

    # Summary
    print(f"\n{'='*50}")
    print(f"  Peak GPU Memory: {peak_mem_mb:.0f} MB")
    if "standard" in results and "graph" in results:
        speedup = results["graph"]["mean_tps"] / results["standard"]["mean_tps"]
        results["speedup"] = float(speedup)
        print(f"  Standard: {results['standard']['mean_tps']:.1f} ± "
              f"{results['standard']['std_tps']:.1f} tok/s")
        print(f"  Graph:    {results['graph']['mean_tps']:.1f} ± "
              f"{results['graph']['std_tps']:.1f} tok/s")
        print(f"  Speedup:  {speedup:.2f}x")
    print(f"{'='*50}")

    # Save results
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
