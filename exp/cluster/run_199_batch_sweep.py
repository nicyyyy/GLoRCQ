"""Exp 7 (task #199): batch-size scaling for Qwen1.5-MoE-A2.7B real-quant.

Loads the real-quant model ONCE, then benchmarks:
  1. Standard model.generate() at batch sizes {1, 4, 8, 16}
  2. CUDA-Graph decode (GLoRCQGraphWrapper) at the same batch sizes,
     best-effort: capture/replay failures are recorded per batch size
     and do NOT abort the sweep.

Reports tok/s per-sequence (gen_len / duration) AND total (batch x gen_len
/ duration). Results are written incrementally to --output_json after every
config so a crash never loses earlier numbers.

Run inside tmux test:0 (GPU 4):
  CUDA_VISIBLE_DEVICES=4 .venv/bin/python exp/cluster/run_199_batch_sweep.py \
      --model_path /mnt/Data/yqy/resource_dir/hf_dl/GLoRCQ-qwen1.5-moe-a2.7b-fair-grassmann-real \
      --output_json logs/cluster_validity/exp199_batch_sweep.json
"""
import argparse
import gc
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch


def _save(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def _set_graph_mode(model, mode):
    from inference.moe_block import GraphCompatibleMoeBlock
    for m in model.modules():
        if isinstance(m, GraphCompatibleMoeBlock):
            m.graph_mode = mode


def _stats(durations, gen_len, batch_size):
    tps_seq = [gen_len / d for d in durations]
    tps_tot = [batch_size * gen_len / d for d in durations]
    return {
        "durations_s": durations,
        "tok_s_per_seq_mean": float(np.mean(tps_seq)),
        "tok_s_per_seq_std": float(np.std(tps_seq)),
        "tok_s_total_mean": float(np.mean(tps_tot)),
        "tok_s_total_std": float(np.std(tps_tot)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--prompt_len", type=int, default=128)
    ap.add_argument("--gen_len", type=int, default=128)
    ap.add_argument("--max_seq_len", type=int, default=384,
                    help="StaticCache length for graph wrapper "
                         "(must cover prompt_len + gen_len)")
    ap.add_argument("--num_warmup", type=int, default=2)
    ap.add_argument("--num_runs", type=int, default=3)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    from utils.model_loader import load_model_and_tokenizer
    # NOTE: the local evaluate/ dir is shadowed by the HF `evaluate` package
    # (site-packages wins — evaluate/ has no __init__.py). Load by file path.
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "glorcq_eval_speed", os.path.join(_ROOT, "evaluate", "eval_speed.py"))
    _es = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_es)
    _make_random_input = _es._make_random_input
    benchmark_standard = _es.benchmark_standard
    benchmark_graph = _es.benchmark_graph

    dev = args.device
    results = {
        "task": "199_exp7_batch_sweep",
        "model_path": args.model_path,
        "prompt_len": args.prompt_len,
        "gen_len": args.gen_len,
        "num_warmup": args.num_warmup,
        "num_runs": args.num_runs,
        "standard": {},
        "graph": {},
    }

    print(f"[199-exp7] Loading real-quant model from {args.model_path} ...",
          flush=True)
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, device=dev, real_quant=True)
    print(f"[199-exp7] Load done in {time.time()-t0:.1f}s", flush=True)

    # ---- Phase 1: Standard generate at every batch size ----
    for bs in args.batch_sizes:
        _set_graph_mode(model, False)
        torch.cuda.reset_peak_memory_stats(dev)
        input_ids = _make_random_input(tokenizer, args.prompt_len, bs, dev)
        print(f"\n[199-exp7] Standard | batch={bs} prompt={input_ids.shape[1]} "
              f"gen={args.gen_len} (warmup={args.num_warmup}, "
              f"runs={args.num_runs})", flush=True)
        try:
            durs = benchmark_standard(
                model, input_ids, args.gen_len, args.num_warmup, args.num_runs)
            entry = _stats(durs, args.gen_len, bs)
            entry["peak_gpu_alloc_gb"] = float(
                torch.cuda.max_memory_allocated(dev) / 1024**3)
            print(f"  batch={bs}: per-seq {entry['tok_s_per_seq_mean']:.2f} "
                  f"± {entry['tok_s_per_seq_std']:.2f} tok/s | total "
                  f"{entry['tok_s_total_mean']:.2f} tok/s | peak "
                  f"{entry['peak_gpu_alloc_gb']:.2f} GB", flush=True)
        except Exception as e:
            entry = {"error": f"{type(e).__name__}: {e}"}
            print(f"  batch={bs}: FAILED — {entry['error']}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        results["standard"][str(bs)] = entry
        _save(results, args.output_json)

    # ---- Phase 2: CUDA Graph at every batch size (best effort) ----
    from inference.graph_wrapper import GLoRCQGraphWrapper
    for bs in args.batch_sizes:
        torch.cuda.reset_peak_memory_stats(dev)
        input_ids = _make_random_input(tokenizer, args.prompt_len, bs, dev)
        print(f"\n[199-exp7] CUDA-Graph | batch={bs} prompt={input_ids.shape[1]} "
              f"gen={args.gen_len}", flush=True)
        wrapper = None
        try:
            wrapper = GLoRCQGraphWrapper(
                model, max_batch_size=bs, max_seq_len=args.max_seq_len)
            wrapper.capture_graph()
            durs = benchmark_graph(
                wrapper, input_ids, args.gen_len, args.num_warmup,
                args.num_runs)
            entry = _stats(durs, args.gen_len, bs)
            entry["peak_gpu_alloc_gb"] = float(
                torch.cuda.max_memory_allocated(dev) / 1024**3)
            print(f"  batch={bs}: per-seq {entry['tok_s_per_seq_mean']:.2f} "
                  f"± {entry['tok_s_per_seq_std']:.2f} tok/s | total "
                  f"{entry['tok_s_total_mean']:.2f} tok/s | peak "
                  f"{entry['peak_gpu_alloc_gb']:.2f} GB", flush=True)
        except Exception as e:
            entry = {"error": f"{type(e).__name__}: {e}"}
            print(f"  batch={bs}: FAILED — {entry['error']}", flush=True)
        finally:
            _set_graph_mode(model, False)
            if wrapper is not None:
                try:
                    wrapper.graph = None
                    wrapper.static_cache = None
                except Exception:
                    pass
                del wrapper
            gc.collect()
            torch.cuda.empty_cache()
        results["graph"][str(bs)] = entry
        _save(results, args.output_json)

    print("\n[199-exp7] done.", flush=True)


if __name__ == "__main__":
    main()
