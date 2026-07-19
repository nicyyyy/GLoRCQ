#!/usr/bin/env python3
"""bs=1 decode bench for DeepSeek-V2-Lite real-quant. Configs:
  --config moe_graph          : per-MoE-block CUDA graphs, eager attention (current best)
  --config compile_moe_graph  : torch.compile(attention) + MoE-block graphs (new)
  --config eager              : no graphs, no compile (baseline)
Verifies sane text (graphs/compile must not change outputs) and reports dynamo
graph-break stats for the compile config.

fp16 baseline is measured by exp/decode_sanity_deepseek.py --mode fp16 (16.70 tok/s).
"""
import os, sys, time, argparse, torch
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)

REAL = "/mnt/Data/yqy/resource_dir/glorcq_paper_exp/deepseek_v2lite_real"


def bench(model, tok, prompt_len, gen_len, label):
    ids = torch.randint(0, 30000, (1, prompt_len), device="cuda:0")
    with torch.no_grad():                       # warmup (also triggers compile)
        _ = model.generate(ids, max_new_tokens=8, do_sample=False,
                            use_cache=True, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=gen_len, do_sample=False,
                             use_cache=True, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize(); dt = time.time() - t0
    n = out.shape[1] - prompt_len
    print(f"[{label}] {n} tok in {dt:.2f}s = {n/dt:.2f} tok/s", flush=True)
    return n / dt


def sample(model, tok, label):
    pids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda:0")
    with torch.no_grad():
        g = model.generate(pids, max_new_tokens=32, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    print(f"[{label}] SAMPLE: {tok.decode(g[0], skip_special_tokens=True)!r}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    choices=["moe_graph", "compile_moe_graph", "eager", "full_graph"],
                    default="compile_moe_graph")
    ap.add_argument("--max_seq_len", type=int, default=384,
                    help="Fixed KV window for the full_graph decode capture.")
    ap.add_argument("--compile_mode", default="default",
                    choices=["default", "max-autotune"])
    ap.add_argument("--cache_size_limit", type=int, default=None,
                    help="Raise torch._dynamo cache_size_limit (>num_layers) to "
                         "survive per-layer DynamicCache-length recompiles.")
    ap.add_argument("--prompt_len", type=int, default=128)
    ap.add_argument("--gen_len", type=int, default=128)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from inference.model_builder import load_glorcq_model
    from inference import deepseek_support

    ret = load_glorcq_model(REAL, device="cuda:0")
    model = ret[0] if isinstance(ret, tuple) else ret
    tok = AutoTokenizer.from_pretrained(REAL, trust_remote_code=True, use_fast=False)

    if args.config == "full_graph":
        # Reference eager output for byte-identical check BEFORE installing graphs.
        pids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda:0")
        with torch.no_grad():
            ref = model.generate(pids, max_new_tokens=32, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        print(f"[eager-ref] SAMPLE: {tok.decode(ref[0], skip_special_tokens=True)!r}",
              flush=True)

        runner = deepseek_support.install_full_decode_graph(
            model, max_seq_len=args.max_seq_len, device="cuda:0")
        # Correctness: same prompt through the full-graph decode loop.
        g = runner.generate(pids, max_new_tokens=32)
        print(f"[full_graph] SAMPLE: {tok.decode(g[0], skip_special_tokens=True)!r}",
              flush=True)
        match = torch.equal(g[0, :ref.shape[1]].cpu(), ref[0].cpu())
        print(f"[full_graph] byte-identical vs eager (first 32 tok): {match}", flush=True)

        # Speed: prompt_len prefill + gen_len decode via the runner.
        ids = torch.randint(0, 30000, (1, args.prompt_len), device="cuda:0")
        with torch.no_grad():
            _ = runner.generate(ids, max_new_tokens=8)     # warmup (capture happens once)
        torch.cuda.synchronize(); t0 = time.time()
        outg = runner.generate(ids, max_new_tokens=args.gen_len)
        torch.cuda.synchronize(); dt = time.time() - t0
        n = outg.shape[1] - args.prompt_len
        tps = n / dt
        print(f"[full_graph] {n} tok in {dt:.2f}s = {tps:.2f} tok/s", flush=True)
        print(f"RESULT config=full_graph max_seq={args.max_seq_len} "
              f"tok_s={tps:.2f} byte_identical={match}", flush=True)
        return

    if args.config == "compile_moe_graph":
        deepseek_support.install_attention_compile(
            model, mode=args.compile_mode, cache_size_limit=args.cache_size_limit)
    if args.config in ("moe_graph", "compile_moe_graph"):
        deepseek_support.install_moe_block_graphs(model, device="cuda:0")

    sample(model, tok, args.config)            # correctness (also compiles attn)
    tps = bench(model, tok, args.prompt_len, args.gen_len, args.config)

    if args.config == "compile_moe_graph":
        stats = deepseek_support.dynamo_stats()
        gb = stats.get("graph_break", {})
        print(f"[dynamo] graph_break entries: {len(gb)}  total_breaks="
              f"{sum(gb.values()) if gb else 0}", flush=True)
        print(f"[dynamo] counters: {stats}", flush=True)

    print(f"RESULT config={args.config} mode={args.compile_mode} tok_s={tps:.2f}",
          flush=True)


if __name__ == "__main__":
    main()
