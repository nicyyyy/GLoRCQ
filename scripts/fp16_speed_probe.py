"""FP16 baseline decode-speed probe (HF framework, eager, NO CUDA graph).

The honest same-framework denominator for GLoRCQ's speedup claim: the vanilla
fp16 model run through plain HF ``model.generate()`` — no CUDA graph, no
custom kernels, no serving-system tricks (vLLM/PagedAttention are a different
track). This isolates the effect of our method (real-quant + graph) vs the
original model in the same inference stack, the way most weight-quantization
papers report speedup.

Prompt construction mirrors inference/graph_wrapper.py:run_speed_benchmark so
the fp16 and real-quant numbers are apples-to-apples.

Usage:
  python scripts/fp16_speed_probe.py --model_path <fp16 dir or HF id> \
      --prompt_len 128 --gen_len 128 --batch_size 1 --device cuda:0 \
      --output_json fp16_results/<name>.json
"""
import argparse
import json
import random
import string
import time

import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="fp16 model dir or HF id")
    ap.add_argument("--prompt_len", type=int, default=128)
    ap.add_argument("--gen_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_warmup", type=int, default=2)
    ap.add_argument("--num_runs", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()

    print(f"[fp16] loading {args.model_path} (fp16, HF, eager) ...", flush=True)
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.use_cache = True
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, config=config, trust_remote_code=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=False,
                                        trust_remote_code=True)

    # Same random prompt recipe as run_speed_benchmark (alnum+space, truncate).
    alphabet = string.ascii_letters + string.digits + ' '
    prompt = ''.join(random.choice(alphabet) for _ in range(args.prompt_len))
    enc = tok(prompt, return_tensors="pt", max_length=args.prompt_len,
              truncation=True).to(args.device)
    ids = enc.input_ids
    if args.batch_size > 1:
        ids = ids.repeat(args.batch_size, 1)
    n_prompt = ids.shape[1]

    def one_run():
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            model.generate(ids, max_new_tokens=args.gen_len, do_sample=False)
        torch.cuda.synchronize()
        return time.time() - t0

    try:
        for _ in range(args.num_warmup):
            one_run()
        torch.cuda.reset_peak_memory_stats(args.device)
        durs = [one_run() for _ in range(args.num_runs)]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"\n  OOM at batch_size={args.batch_size} — this config does not "
              f"fit; skipping (report as OOM in the table).", flush=True)
        if args.output_json:
            import os
            os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
            json.dump({"model": args.model_path, "batch_size": args.batch_size,
                       "oom": True}, open(args.output_json, "w"), indent=1)
        return

    dur = sum(durs) / len(durs)
    per_seq = args.gen_len / dur
    total = per_seq * args.batch_size
    peak_gb = torch.cuda.max_memory_allocated(args.device) / 1024**3

    print(f"\n{'='*50}")
    print(f"  FP16 baseline (HF eager, no CUDA graph)")
    print(f"  model: {args.model_path}")
    print(f"  prompt_tokens={n_prompt} gen={args.gen_len} batch={args.batch_size}")
    print(f"  fp16 Standard: {per_seq:.1f} tok/s/seq | total {total:.1f} tok/s")
    print(f"  peak_gpu: {peak_gb:.1f} GB | runs(s): {[round(d,2) for d in durs]}")
    print(f"{'='*50}", flush=True)

    if args.output_json:
        import os
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        json.dump({"model": args.model_path, "prompt_tokens": n_prompt,
                   "gen_len": args.gen_len, "batch_size": args.batch_size,
                   "fp16_tok_s_per_seq": per_seq, "fp16_tok_s_total": total,
                   "peak_gpu_gb": peak_gb, "durations_s": durs},
                  open(args.output_json, "w"), indent=1)
        print(f"[fp16] saved -> {args.output_json}")


if __name__ == "__main__":
    main()
