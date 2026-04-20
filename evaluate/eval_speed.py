"""
GLoRCQ speed evaluation: standard vs CUDA Graph vs torch.compile vs FP16.

Loads a real-quantized GLoRCQ model and benchmarks generation speed
with and without CUDA Graph acceleration, and optionally with
``torch.compile``.  Optionally loads the original FP16 HuggingFace
model as a baseline for comparison.

Usage:
    uv run python evaluate/eval_speed.py \
        --model_path ./output/model-glorcq \
        --prompt_len 128 --gen_len 128

    # With FP16 baseline comparison:
    uv run python evaluate/eval_speed.py \
        --model_path ./output/model-glorcq \
        --hf_model_path Qwen/Qwen1.5-MoE-A2.7B

    # With torch.compile benchmark:
    uv run python evaluate/eval_speed.py \
        --model_path ./output/model-glorcq \
        --skip_graph --use_compile \
        --prompt_len 64 --gen_len 64 --num_runs 3
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


@torch.no_grad()
def benchmark_fp16_baseline(model, input_ids, gen_len, num_warmup, num_runs):
    """Benchmark FP16 model standard generation as baseline."""
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
def benchmark_compile(model, input_ids, gen_len, num_warmup, num_runs):
    """Benchmark ``torch.compile``-d ``model.generate()``."""
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


def main():
    parser = argparse.ArgumentParser(
        description="GLoRCQ speed benchmark: standard vs CUDA Graph inference",
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to GLoRCQ real-quantized model directory")
    parser.add_argument("--hf_model_path", type=str, default=None,
                        help="Original HF model path for FP16 baseline (e.g. Qwen/Qwen1.5-MoE-A2.7B)")
    parser.add_argument("--device", type=str, default="cuda")
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
    parser.add_argument("--use_compile", action="store_true",
                        help="Enable torch.compile benchmark")
    parser.add_argument("--compile_mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode (default: default; avoid reduce-overhead for MoE — "
                             "CUDA Graphs are incompatible with dynamic expert routing)")
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

    # # Phase 1: Standard generation
    # if not args.skip_standard:
    #     print(f"\n[1/2] Standard generation (warmup={args.num_warmup}, "
    #           f"runs={args.num_runs}) ...")
    #     durations_std = benchmark_standard(
    #         model, input_ids, args.gen_len, args.num_warmup, args.num_runs,
    #     )
    #     tps_std = [args.gen_len / d for d in durations_std]
    #     mean_std, std_std = np.mean(tps_std), np.std(tps_std)
    #     print(f"  Standard: {mean_std:.1f} ± {std_std:.1f} tok/s")
    #     results["standard"] = {
    #         "mean_tps": float(mean_std),
    #         "std_tps": float(std_std),
    #         "durations": durations_std,
    #     }

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

    # Phase 2b: torch.compile generation
    if args.use_compile:
        from inference.moe_block import GraphCompatibleMoeBlock

        # Ensure sparse routing (not dense graph_mode)
        for module in model.modules():
            if isinstance(module, GraphCompatibleMoeBlock):
                module.graph_mode = False

        print(f"\n[compile] torch.compile (mode={args.compile_mode}, "
              f"warmup={args.num_warmup}, runs={args.num_runs}) ...")

        # Compile model.forward (not model.generate — generate has Python
        # logic that is not suitable for compilation)
        model.forward = torch.compile(
            model.forward, mode=args.compile_mode, fullgraph=False,
        )

        # Trigger compilation with a single-token generation
        print("  Compiling ...")
        model.generate(input_ids, max_new_tokens=1, do_sample=False)
        torch.cuda.synchronize()

        durations_compile = benchmark_compile(
            model, input_ids, args.gen_len, args.num_warmup, args.num_runs,
        )
        tps_compile = [args.gen_len / d for d in durations_compile]
        mean_compile, std_compile = np.mean(tps_compile), np.std(tps_compile)
        print(f"  Compile:  {mean_compile:.1f} ± {std_compile:.1f} tok/s")
        results["compile"] = {
            "mean_tps": float(mean_compile),
            "std_tps": float(std_compile),
            "mode": args.compile_mode,
            "durations": durations_compile,
        }

    # Phase 3: FP16 baseline generation
    if args.hf_model_path:
        # Free quantized model to save GPU memory
        del model
        if 'wrapper' in locals():
            del wrapper
        torch.cuda.empty_cache()

        print(f"\n[3/3] FP16 baseline generation (warmup={args.num_warmup}, "
              f"runs={args.num_runs}) ...")
        print(f"Loading FP16 model from {args.hf_model_path} ...")
        fp16_model, fp16_tokenizer = load_model_and_tokenizer(
            args.hf_model_path, device=args.device, real_quant=False,
        )
        fp16_input_ids = _make_random_input(
            fp16_tokenizer, args.prompt_len, args.batch_size, args.device,
        )
        print(f"FP16 prompt tokens: {fp16_input_ids.shape[1]}")

        durations_fp16 = benchmark_fp16_baseline(
            fp16_model, fp16_input_ids, args.gen_len,
            args.num_warmup, args.num_runs,
        )
        tps_fp16 = [args.gen_len / d for d in durations_fp16]
        mean_fp16, std_fp16 = np.mean(tps_fp16), np.std(tps_fp16)
        print(f"  FP16:     {mean_fp16:.1f} ± {std_fp16:.1f} tok/s")
        results["fp16_baseline"] = {
            "mean_tps": float(mean_fp16),
            "std_tps": float(std_fp16),
            "durations": durations_fp16,
        }

        del fp16_model
        torch.cuda.empty_cache()

    # Peak GPU memory
    peak_mem_mb = torch.cuda.max_memory_allocated(args.device) / 1024 ** 2
    results["peak_gpu_memory_mb"] = float(peak_mem_mb)

    # Summary
    print(f"\n{'='*50}")
    print(f"  Peak GPU Memory: {peak_mem_mb:.0f} MB")
    if "standard" in results:
        print(f"  Standard: {results['standard']['mean_tps']:.1f} ± "
              f"{results['standard']['std_tps']:.1f} tok/s")
    if "graph" in results:
        print(f"  Graph:    {results['graph']['mean_tps']:.1f} ± "
              f"{results['graph']['std_tps']:.1f} tok/s")
    if "compile" in results:
        print(f"  Compile:  {results['compile']['mean_tps']:.1f} ± "
              f"{results['compile']['std_tps']:.1f} tok/s "
              f"(mode={results['compile']['mode']})")
    if "fp16_baseline" in results:
        print(f"  FP16:     {results['fp16_baseline']['mean_tps']:.1f} ± "
              f"{results['fp16_baseline']['std_tps']:.1f} tok/s")
    if "standard" in results and "graph" in results:
        speedup = results["graph"]["mean_tps"] / results["standard"]["mean_tps"]
        results["speedup_graph_vs_standard"] = float(speedup)
        print(f"  Speedup (Graph vs Standard): {speedup:.2f}x")
    if "graph" in results and "fp16_baseline" in results:
        speedup_vs_fp16 = results["graph"]["mean_tps"] / results["fp16_baseline"]["mean_tps"]
        results["speedup_graph_vs_fp16"] = float(speedup_vs_fp16)
        print(f"  Speedup (Graph vs FP16):     {speedup_vs_fp16:.2f}x")
    if "compile" in results and "fp16_baseline" in results:
        speedup_compile_fp16 = results["compile"]["mean_tps"] / results["fp16_baseline"]["mean_tps"]
        results["speedup_compile_vs_fp16"] = float(speedup_compile_fp16)
        print(f"  Speedup (Compile vs FP16):   {speedup_compile_fp16:.2f}x")
    if "compile" in results and "graph" in results:
        speedup_compile_graph = results["compile"]["mean_tps"] / results["graph"]["mean_tps"]
        results["speedup_compile_vs_graph"] = float(speedup_compile_graph)
        print(f"  Speedup (Compile vs Graph):  {speedup_compile_graph:.2f}x")
    print(f"{'='*50}")

    # Save results
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
