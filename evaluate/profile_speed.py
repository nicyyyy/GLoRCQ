"""
Profile GLoRCQ real-quant vs FP16 inference — per-kernel breakdown.

Runs a single forward pass through the model with torch.profiler,
outputs a breakdown of time spent in each operation category:
  - Attention (q/k/v/o matmul)
  - MoE expert routing
  - MoE expert compute (dequant+matmul, LoRA, activation)
  - Pi rotation (TurboQuant)
  - Other

Usage:
    # Profile real-quant model
    CUDA_VISIBLE_DEVICES=0 uv run python evaluate/profile_speed.py \
        --model_path ./output/Qwen/Qwen1.5-MoE-A2.7B-glorcq-2bit-rank64-realquant

    # Profile FP16 model
    CUDA_VISIBLE_DEVICES=0 uv run python evaluate/profile_speed.py \
        --model_path Qwen/Qwen1.5-MoE-A2.7B --fp16

    # Export Chrome trace for visualization
    CUDA_VISIBLE_DEVICES=0 uv run python evaluate/profile_speed.py \
        --model_path ./output/...-realquant --trace_path ./profile_trace.json
"""
import argparse
import os
import sys
import time

import torch
from torch.profiler import profile, record_function, ProfilerActivity

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)


def _make_input(tokenizer, prompt_len, device):
    """Create a fixed prompt of approximately prompt_len tokens."""
    text = "The quick brown fox jumps over the lazy dog. " * (prompt_len // 8 + 1)
    inputs = tokenizer(text, return_tensors="pt", max_length=prompt_len, truncation=True)
    return inputs.input_ids.to(device)


def _profile_generate(model, input_ids, gen_len, label="model"):
    """Profile model.generate() and return profiler key averages."""
    # Warmup
    model.generate(input_ids, max_new_tokens=2, do_sample=False)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        with record_function(f"{label}_generate"):
            model.generate(input_ids, max_new_tokens=gen_len, do_sample=False)
            torch.cuda.synchronize()

    return prof


def _categorize_event(name):
    """Categorize a CUDA kernel or CPU op into a high-level bucket."""
    name_lower = name.lower()

    # Custom CUDA kernels
    if "turbo_dequant" in name_lower:
        return "TurboQuant kernel"
    if "gptq_dequant" in name_lower:
        return "GPTQ kernel"

    # Rotation
    if "rotate" in name_lower or ("pi" in name_lower and "gemm" in name_lower):
        return "Pi rotation"

    # GEMM/GEMV (cuBLAS)
    if any(k in name_lower for k in ["gemm", "gemv", "cublas", "cutlass"]):
        return "GEMM/GEMV (cuBLAS)"

    # Softmax, layer norm, etc.
    if "softmax" in name_lower:
        return "Softmax"
    if "layernorm" in name_lower or "layer_norm" in name_lower or "rmsnorm" in name_lower:
        return "LayerNorm/RMSNorm"

    # Elementwise
    if any(k in name_lower for k in ["elementwise", "add", "mul", "silu", "gelu", "sigmoid"]):
        return "Elementwise ops"

    # Attention
    if any(k in name_lower for k in ["attention", "sdpa", "flash"]):
        return "Attention"

    # Memory ops
    if any(k in name_lower for k in ["memcpy", "memset", "copy_", "to(", "contiguous"]):
        return "Memory ops"

    # Index / scatter / gather
    if any(k in name_lower for k in ["index", "scatter", "gather", "topk", "sort"]):
        return "Index/Scatter/TopK"

    return "Other"


def _print_summary(prof, label, gen_len, top_n=30):
    """Print categorized kernel time summary."""
    print(f"\n{'='*70}")
    print(f"  {label} — Profile Summary (gen_len={gen_len})")
    print(f"{'='*70}")

    # Top CUDA kernels by time
    events = prof.key_averages()
    print(f"\n--- Top {top_n} CUDA ops by self CUDA time ---")
    print(f"{'Category':<25} {'Op Name':<50} {'Self CUDA (ms)':>14} {'Count':>6}")
    print("-" * 100)

    sorted_events = sorted(events, key=lambda e: e.self_device_time_total, reverse=True)
    for e in sorted_events[:top_n]:
        if e.self_device_time_total == 0:
            continue
        cat = _categorize_event(e.key)
        name = e.key[:48]
        ms = e.self_device_time_total / 1000.0
        print(f"{cat:<25} {name:<50} {ms:>12.2f}  {e.count:>6}")

    # Categorized summary
    cat_totals = {}
    for e in events:
        cat = _categorize_event(e.key)
        cat_totals[cat] = cat_totals.get(cat, 0) + e.self_device_time_total

    total_cuda = sum(cat_totals.values())
    print(f"\n--- Categorized CUDA time ---")
    print(f"{'Category':<30} {'Time (ms)':>12} {'%':>8}")
    print("-" * 55)
    for cat, t in sorted(cat_totals.items(), key=lambda x: -x[1]):
        ms = t / 1000.0
        pct = 100.0 * t / total_cuda if total_cuda > 0 else 0
        print(f"{cat:<30} {ms:>10.2f}   {pct:>6.1f}%")
    print(f"{'TOTAL':<30} {total_cuda/1000.0:>10.2f}   {'100.0':>6}%")

    # Wall time estimate
    wall_ms = total_cuda / 1000.0
    tok_per_sec = gen_len / (wall_ms / 1000.0) if wall_ms > 0 else 0
    print(f"\n  Total CUDA time: {wall_ms:.1f} ms → ~{tok_per_sec:.1f} tok/s (upper bound)")


def main():
    parser = argparse.ArgumentParser(description="Profile GLoRCQ inference")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--fp16", action="store_true",
                        help="Load as FP16 HuggingFace model (not real-quant)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument("--gen_len", type=int, default=16,
                        help="Tokens to generate (keep small for profiling)")
    parser.add_argument("--trace_path", type=str, default=None,
                        help="Export Chrome trace JSON for visualization")
    parser.add_argument("--top_n", type=int, default=30)
    args = parser.parse_args()

    from utils.model_loader import load_model_and_tokenizer

    if args.fp16:
        print(f"Loading FP16 model from {args.model_path} ...")
        model, tokenizer = load_model_and_tokenizer(
            args.model_path, device=args.device, real_quant=False)
        label = "FP16"
    else:
        print(f"Loading real-quant model from {args.model_path} ...")
        model, tokenizer = load_model_and_tokenizer(
            args.model_path, device=args.device, real_quant=True)
        label = "GLoRCQ-realquant"

    input_ids = _make_input(tokenizer, args.prompt_len, args.device)
    print(f"Prompt: {input_ids.shape[1]} tokens, Generate: {args.gen_len} tokens")

    # Check kernel status
    try:
        from inference.kernels import is_cuda_available, is_gptq_cuda_available
        print(f"TurboQuant CUDA kernel: {is_cuda_available()}")
        print(f"GPTQ CUDA kernel: {is_gptq_cuda_available()}")
    except Exception:
        pass

    # Check model layer types
    try:
        from inference.quantized_linear import GLoRCQLinear
        stats = {"gptq": 0, "turbo_kernel": 0, "turbo_predequant": 0, "lora": 0}
        for _, m in model.named_modules():
            if isinstance(m, GLoRCQLinear):
                if m.quant_type == "gptq":
                    stats["gptq"] += 1
                elif m.quant_type == "turbo":
                    if m._rotation_cache is not None:
                        stats["turbo_kernel"] += 1
                    else:
                        stats["turbo_predequant"] += 1
                if m.U is not None:
                    stats["lora"] += 1
        print(f"Layer stats: {stats}")
    except Exception:
        pass

    # Profile
    print(f"\nProfiling {label} ...")
    prof = _profile_generate(model, input_ids, args.gen_len, label=label)
    _print_summary(prof, label, args.gen_len, top_n=args.top_n)

    if args.trace_path:
        prof.export_chrome_trace(args.trace_path)
        print(f"\nChrome trace exported to {args.trace_path}")
        print("Open in chrome://tracing or https://ui.perfetto.dev/")


if __name__ == "__main__":
    main()
