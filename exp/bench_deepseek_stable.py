#!/usr/bin/env python3
"""Stable bs=1 decode throughput for DeepSeek-V2-Lite: warmup-discard + N timed
runs, report median + min/max. Identical harness across configs:
  eager      : real-quant, no graphs, eager attention
  moe_graph  : per-MoE-block CUDA graphs + eager attention (current best)
  full_graph : full decode-step CUDA graph (attention + MoE together)
  fp16       : fp16 eager baseline
Each config is run in a fresh process (avoids cross-config model mutation), so the
only thing that differs is the setup — same prompt (seeded), same prefill+decode
timing (prompt_len prefill + gen_len decode, timed together like the prior bench).
"""
import os, sys, time, argparse, statistics, torch
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)

REAL = "/mnt/Data/yqy/resource_dir/glorcq_paper_exp/deepseek_v2lite_real"
FP16 = "/mnt/Data/yqy/resource_dir/deepseek-v2-lite"


def timed_generate(gen_fn, prompt_len, gen_len):
    torch.cuda.synchronize(); t0 = time.time()
    out = gen_fn(gen_len)
    torch.cuda.synchronize(); dt = time.time() - t0
    n = out.shape[1] - prompt_len
    return n / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["eager", "moe_graph", "full_graph", "fp16"],
                    required=True)
    ap.add_argument("--prompt_len", type=int, default=128)
    ap.add_argument("--gen_len", type=int, default=128)
    ap.add_argument("--max_seq_len", type=int, default=384)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from inference import deepseek_support

    if args.config == "fp16":
        from transformers import AutoModelForCausalLM
        deepseek_support.patch_cache_compat()
        tok = AutoTokenizer.from_pretrained(FP16, trust_remote_code=True, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            FP16, trust_remote_code=True, torch_dtype=torch.float16).to("cuda:0")
        model.eval()
    else:
        from inference.model_builder import load_glorcq_model
        ret = load_glorcq_model(REAL, device="cuda:0")
        model = ret[0] if isinstance(ret, tuple) else ret
        tok = AutoTokenizer.from_pretrained(REAL, trust_remote_code=True, use_fast=False)

    runner = None
    if args.config == "moe_graph":
        deepseek_support.install_moe_block_graphs(model, device="cuda:0")
    elif args.config == "full_graph":
        runner = deepseek_support.install_full_decode_graph(
            model, max_seq_len=args.max_seq_len, device="cuda:0")
        # Representative-work sanity: graph_mode=True runs the ALL-experts fixed
        # path (n_routed_experts per layer, inactive weighted 0) across all layers
        # — i.e. >= the eager top-k work, never less. Confirm the runner will drive
        # every MoE block in that mode.
        n_moe = len(runner._moe_blocks)
        n_layers = len(runner.layers)
        n_exp = getattr(model.config, "n_routed_experts", "?")
        print(f"[sanity] full_graph drives {n_layers} decoder layers, {n_moe} MoE "
              f"blocks @ graph_mode=all-{n_exp}-experts (>= eager top-k work)",
              flush=True)

    torch.manual_seed(0)
    ids = torch.randint(0, 30000, (1, args.prompt_len), device="cuda:0")

    if runner is not None:
        gen_fn = lambda g: runner.generate(ids, max_new_tokens=g)
    else:
        gen_fn = lambda g: model.generate(
            ids, max_new_tokens=g, do_sample=False, use_cache=True,
            pad_token_id=tok.eos_token_id)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = gen_fn(args.gen_len)          # discard (also triggers capture)
        vals = []
        for r in range(args.runs):
            tps = timed_generate(gen_fn, args.prompt_len, args.gen_len)
            vals.append(tps)
            print(f"[{args.config}] run{r}: {tps:.2f} tok/s", flush=True)

    med = statistics.median(vals)
    print(f"RESULT config={args.config} median={med:.2f} min={min(vals):.2f} "
          f"max={max(vals):.2f} n={len(vals)} tok_s_all={[round(v,2) for v in vals]}",
          flush=True)


if __name__ == "__main__":
    main()
