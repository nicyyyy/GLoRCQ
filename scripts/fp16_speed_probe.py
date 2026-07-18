"""FP16 baseline decode-speed probe (HF framework, eager, NO CUDA graph).

The honest same-framework denominator for GLoRCQ's speedup claim: the vanilla
fp16 model run through plain HF ``model.generate()`` — no CUDA graph, no
custom kernels, no serving-system tricks (vLLM/PagedAttention are a different
track). This isolates the effect of our method (real-quant + graph) vs the
original model in the same inference stack, the way most weight-quantization
papers report speedup.

Loads the fp16 model ONCE and sweeps all requested batch sizes in the same
process (fp16 has no fast loader — reloading 27 GB–87 GB per batch is slow).
Prompt construction mirrors inference/graph_wrapper.py:run_speed_benchmark so
the fp16 and real-quant numbers are apples-to-apples.

Usage:
  python scripts/fp16_speed_probe.py --model_path <fp16 dir or HF id> \
      --batch_sizes "1 4 16 64" --prompt_len 128 --gen_len 128 \
      --device cuda:0 --output_prefix fp16_results/<name>
  # writes <prefix>_bs<bs>.json per batch; --batch_size N still works (single).
"""
import argparse
import json
import os
import random
import string
import time

import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="fp16 model dir or HF id")
    ap.add_argument("--batch_sizes", default=None,
                    help='space/comma-separated list, e.g. "1 4 16 64"')
    ap.add_argument("--batch_size", type=int, default=1,
                    help="single batch (used if --batch_sizes omitted)")
    ap.add_argument("--prompt_len", type=int, default=128)
    ap.add_argument("--gen_len", type=int, default=128)
    ap.add_argument("--num_warmup", type=int, default=2)
    ap.add_argument("--num_runs", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output_prefix", default=None,
                    help="writes <prefix>_bs<bs>.json per batch")
    ap.add_argument("--output_json", default=None, help="single-batch json (legacy)")
    args = ap.parse_args()

    if args.batch_sizes:
        batch_sizes = [int(x) for x in args.batch_sizes.replace(",", " ").split()]
    else:
        batch_sizes = [args.batch_size]

    # ---- load ONCE ----
    print(f"[fp16] loading {args.model_path} (fp16, HF, eager) ...", flush=True)
    t0 = time.time()
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.use_cache = True
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, config=config, trust_remote_code=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=False,
                                        trust_remote_code=True)
    print(f"[fp16] loaded in {time.time()-t0:.1f}s; sweeping batches {batch_sizes}", flush=True)

    # Same random prompt recipe as run_speed_benchmark (alnum+space, truncate).
    alphabet = string.ascii_letters + string.digits + ' '
    prompt = ''.join(random.choice(alphabet) for _ in range(args.prompt_len))
    enc = tok(prompt, return_tensors="pt", max_length=args.prompt_len,
              truncation=True).to(args.device)
    base_ids = enc.input_ids
    n_prompt = base_ids.shape[1]

    def save(bs, payload):
        path = None
        if args.output_prefix:
            path = f"{args.output_prefix}_bs{bs}.json"
        elif args.output_json:
            path = args.output_json
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            json.dump(payload, open(path, "w"), indent=1)

    for bs in batch_sizes:
        ids = base_ids.repeat(bs, 1) if bs > 1 else base_ids
        attn = torch.ones_like(ids)

        def one_run():
            torch.cuda.synchronize()
            t = time.time()
            with torch.no_grad():
                model.generate(ids, attention_mask=attn,
                               max_new_tokens=args.gen_len, do_sample=False)
            torch.cuda.synchronize()
            return time.time() - t

        try:
            for _ in range(args.num_warmup):
                one_run()
            torch.cuda.reset_peak_memory_stats(args.device)
            durs = [one_run() for _ in range(args.num_runs)]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  bs={bs}: OOM — does not fit; skipping.", flush=True)
            save(bs, {"model": args.model_path, "batch_size": bs, "oom": True})
            continue

        dur = sum(durs) / len(durs)
        per_seq = args.gen_len / dur
        total = per_seq * bs
        peak_gb = torch.cuda.max_memory_allocated(args.device) / 1024**3
        print(f"  bs={bs}: fp16 {per_seq:.1f} tok/s/seq | total {total:.1f} tok/s "
              f"| peak {peak_gb:.1f} GB", flush=True)
        save(bs, {"model": args.model_path, "prompt_tokens": n_prompt,
                  "gen_len": args.gen_len, "batch_size": bs,
                  "fp16_tok_s_per_seq": per_seq, "fp16_tok_s_total": total,
                  "peak_gpu_gb": peak_gb, "durations_s": durs})


if __name__ == "__main__":
    main()
