"""
Mixtral-specific speed benchmark for GLoRCQ.

Loads a real-quantized Mixtral checkpoint the same way as inference/eval_speed.py,
then swaps every GraphCompatibleMoeBlock for MixtralFastMoeBlock (pre-baked
expert dispatch — see inference/mixtral_fast.py) BEFORE running the standard
+ CUDA-Graph timing loop.

Usage:
  uv run python glorcq/inference/eval_speed_mixtral.py \
      --model_path ./output/mixtral-8x7b-glorcq-... \
      --batch_size 1 --prompt_len 128 --gen_len 128
"""

import argparse
import os
import sys
import time


# Ensure glorcq is importable regardless of where this script is invoked from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_GLORCQ_ROOT = os.path.dirname(_HERE)
_PROJ_ROOT = os.path.dirname(_GLORCQ_ROOT)
for p in [_GLORCQ_ROOT, _PROJ_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def _apply_fast_dispatch(model, verbose: bool = True):
    """Swap Mixtral MoE blocks for MixtralFastMoeBlock.

    Prints how many layers got the fast path vs fallback so the run log is
    self-describing.
    """
    from inference.mixtral_fast import replace_mixtral_moe_blocks
    n_fast, n_fallback = replace_mixtral_moe_blocks(model, verbose=verbose)
    if verbose:
        mode = "mixtral-fast" if n_fast > 0 else "fallback-only"
        print(f"[eval_speed_mixtral] dispatch-mode: {mode} "
              f"({n_fast} fast + {n_fallback} fallback layers)",
              file=sys.stderr)
    return n_fast, n_fallback


def _random_prompt_inputs(tokenizer, device, prompt_len):
    import random
    import string
    alphabet = string.ascii_letters + string.digits + " "
    prompt = "".join(random.choice(alphabet) for _ in range(prompt_len))
    return tokenizer(
        prompt, return_tensors="pt",
        max_length=prompt_len, truncation=True,
    ).to(device)


def _run_standard(model, inputs, gen_len):
    """Standard HF generate loop with proper cuda.synchronize timing."""
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=gen_len,
            do_sample=False,
        )
    torch.cuda.synchronize()
    dt = time.time() - t0
    return out_ids, dt


def _run_graph(model, inputs, gen_len, batch_size, max_seq_len):
    """CUDA-Graph-captured decode. Uses the project's GLoRCQGraphWrapper."""
    from inference.graph_wrapper import GLoRCQGraphWrapper
    wrapper = GLoRCQGraphWrapper(
        model, max_batch_size=batch_size, max_seq_len=max_seq_len)
    wrapper.capture_graph()
    torch.cuda.synchronize()
    out_ids, dt = wrapper.generate(inputs.input_ids, max_new_tokens=gen_len)
    torch.cuda.synchronize()
    return out_ids, dt


def main():
    parser = argparse.ArgumentParser(
        description=("Mixtral GLoRCQ speed benchmark with pre-baked "
                     "expert-dispatch fast path (Candidate B).")
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to real-quantized Mixtral GLoRCQ ckpt dir")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prompt_len", type=int, default=128)
    parser.add_argument("--gen_len", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no_graph", action="store_true",
                        help="Skip the CUDA-Graph run (standard path only).")
    parser.add_argument("--no_fast_dispatch", action="store_true",
                        help="Skip Mixtral fast-dispatch monkey-patch. Useful "
                             "for A/B baseline vs the fast path.")
    parser.add_argument("--warmup", type=int, default=8,
                        help="Warmup decode tokens before timing (default: 8).")
    args = parser.parse_args()

    # ---- Load real-quant Mixtral model ----
    from inference.model_builder import load_glorcq_model
    print(f"[eval_speed_mixtral] loading {args.model_path} ...", file=sys.stderr)
    model = load_glorcq_model(args.model_path, device=args.device)
    model.eval()

    # ---- Apply the fast-dispatch monkey-patch (unless --no_fast_dispatch) ----
    if args.no_fast_dispatch:
        print("[eval_speed_mixtral] dispatch-mode: baseline (no fast dispatch)",
              file=sys.stderr)
    else:
        _apply_fast_dispatch(model, verbose=True)

    # ---- Tokenizer (mirror eval_speed.py logic) ----
    try:
        cross_layer_info = torch.load(
            os.path.join(args.model_path, "cross_layer_info.pt"),
            map_location="cpu", weights_only=False,
        )
        original_model = cross_layer_info["config"].get(
            "model_path", args.model_path)
    except (FileNotFoundError, KeyError):
        original_model = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(
        original_model, use_fast=False, trust_remote_code=True,
    )

    # ---- Prompt ----
    inputs = _random_prompt_inputs(tokenizer, args.device, args.prompt_len)

    print(f"\n{'='*60}")
    print(f"  GLoRCQ Mixtral Speed Benchmark")
    print(f"  Model:         {args.model_path}")
    print(f"  Fast-dispatch: {'off' if args.no_fast_dispatch else 'on'}")
    print(f"  Prompt tokens: {inputs.input_ids.shape[1]}")
    print(f"  Generate:      {args.gen_len} tokens")
    print(f"  Batch size:    {args.batch_size}")
    print(f"{'='*60}")

    # ---- Warmup (kernel autotune + first-forward compile) ----
    if args.warmup > 0:
        print(f"\n[warmup] generating {args.warmup} tokens ...")
        torch.cuda.synchronize()
        with torch.no_grad():
            _ = model.generate(
                inputs.input_ids,
                max_new_tokens=args.warmup,
                do_sample=False,
            )
        torch.cuda.synchronize()

    # ---- 1) Standard generation ----
    print("\n[1/2] Standard generation (no CUDA Graph) ...")
    _, t_std = _run_standard(model, inputs, args.gen_len)
    std_tps = args.gen_len / t_std if t_std > 0 else 0.0
    print(f"  Time: {t_std:.3f}s   Speed: {std_tps:.2f} tok/s")

    # ---- 2) CUDA-Graph generation ----
    if args.no_graph:
        graph_tps = None
        print("\n[2/2] CUDA Graph generation ... SKIPPED (--no_graph)")
    else:
        print("\n[2/2] CUDA Graph generation ...")
        try:
            _, t_graph = _run_graph(model, inputs, args.gen_len,
                                     args.batch_size, args.max_seq_len)
            graph_tps = args.gen_len / t_graph if t_graph > 0 else 0.0
            print(f"  Time: {t_graph:.3f}s   Speed: {graph_tps:.2f} tok/s")
        except Exception as e:
            graph_tps = None
            print(f"  [warn] CUDA-Graph run failed: {e}", file=sys.stderr)

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  Standard: {std_tps:.2f} tok/s")
    if graph_tps is not None:
        speedup = graph_tps / std_tps if std_tps > 0 else 0.0
        print(f"  Graph:    {graph_tps:.2f} tok/s")
        print(f"  Speedup:  {speedup:.2f}x  (graph / standard)")
    print(f"{'='*60}")

    return {
        "standard_tps": std_tps,
        "graph_tps": graph_tps,
        "fast_dispatch": not args.no_fast_dispatch,
    }


if __name__ == "__main__":
    main()
