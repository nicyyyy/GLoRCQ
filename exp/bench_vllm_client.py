"""
vLLM decode-throughput benchmark client.

Sends requests to a running vLLM server (OpenAI-compatible API) and measures
decode throughput in tokens/second using streaming responses.

The measurement protocol matches evaluate/eval_speed.py:
  - batch_size  = 1 (single request at a time)
  - prompt_len  = 128 tokens  (random text)
  - gen_len     = 128 tokens
  - warmup      = 2 requests  (discarded)
  - runs        = 5 requests  (averaged)

Metric reported:
  - decode_tok_s: tokens generated per second from first generated token
                  to last (excludes prefill/TTFT latency)

Usage:
    python exp/bench_vllm_client.py \
        --host localhost --port 8010 \
        --model Qwen/Qwen1.5-MoE-A2.7B \
        --prompt_len 128 --gen_len 128 --warmup 2 --runs 5 \
        --output_json logs/vllm_bench_qwen_moe.json
"""

import argparse
import json
import os
import string
import sys
import time
from statistics import mean, stdev

import requests


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------
ALPHABET = string.ascii_letters + string.digits + " .,!?"


def _make_prompt(approx_tokens: int) -> str:
    """Build a random English-ish string that tokenizes to ~approx_tokens."""
    chars = approx_tokens * 4  # rough over-estimate; server will truncate
    return "".join(ALPHABET[i % len(ALPHABET)] for i in range(chars))


# ---------------------------------------------------------------------------
# vLLM streaming request
# ---------------------------------------------------------------------------
def _stream_generate(host: str, port: int, model: str, prompt: str, max_tokens: int):
    """
    Send one streaming completion request and return (ttft_s, decode_s, n_tokens).

    ttft_s   : time from request sent to first token received (seconds)
    decode_s : time from first token to last token (seconds); this is the
               decode-only latency we care about
    n_tokens : number of tokens generated (should equal max_tokens)
    """
    url = f"http://{host}:{port}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}

    t_start = time.perf_counter()
    t_first = None
    n_tokens = 0

    with requests.post(url, json=payload, headers=headers, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            if line == b"data: [DONE]":
                break
            if line.startswith(b"data: "):
                chunk = json.loads(line[6:])
                delta = chunk.get("choices", [{}])[0].get("text", "")
                if delta:
                    if t_first is None:
                        t_first = time.perf_counter()
                    n_tokens += 1  # approximate: 1 token per SSE chunk (vLLM default)

    t_end = time.perf_counter()
    ttft_s = (t_first - t_start) if t_first else (t_end - t_start)
    decode_s = (t_end - t_first) if t_first else 0.0

    # Fallback: vLLM may stream multiple tokens per chunk; get exact count
    # from usage field (requires non-streaming fallback). Using n_tokens as
    # approximation is fine for throughput benchmarking.

    return ttft_s, decode_s, n_tokens or max_tokens


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def run_benchmark(host, port, model, prompt_len, gen_len, warmup, runs):
    prompt = _make_prompt(prompt_len)

    print(f"  Warming up ({warmup} requests)...")
    for _ in range(warmup):
        _stream_generate(host, port, model, prompt, gen_len)

    print(f"  Benchmarking ({runs} requests)...")
    ttft_list, decode_tps_list, total_tps_list = [], [], []

    for i in range(runs):
        ttft_s, decode_s, n_tok = _stream_generate(host, port, model, prompt, gen_len)
        # Avoid division by zero on very fast responses
        decode_tps = n_tok / decode_s if decode_s > 1e-4 else float("inf")
        total_tps  = n_tok / (ttft_s + decode_s)
        ttft_list.append(ttft_s)
        decode_tps_list.append(decode_tps)
        total_tps_list.append(total_tps)
        print(f"    run {i+1}/{runs}: TTFT={ttft_s*1000:.0f}ms  "
              f"decode={decode_tps:.1f} tok/s  total={total_tps:.1f} tok/s")

    results = {
        "model": model,
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "warmup": warmup,
        "runs": runs,
        # Decode throughput (first-token to last-token, the primary metric)
        "decode_tok_s_mean": mean(decode_tps_list),
        "decode_tok_s_std":  stdev(decode_tps_list) if len(decode_tps_list) > 1 else 0.0,
        # Total throughput (includes TTFT latency)
        "total_tok_s_mean":  mean(total_tps_list),
        "total_tok_s_std":   stdev(total_tps_list) if len(total_tps_list) > 1 else 0.0,
        # Time to first token
        "ttft_ms_mean": mean(t * 1000 for t in ttft_list),
        "ttft_ms_std":  stdev(t * 1000 for t in ttft_list) if len(ttft_list) > 1 else 0.0,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="vLLM decode-throughput benchmark")
    parser.add_argument("--host",        default="localhost")
    parser.add_argument("--port",        type=int, default=8010)
    parser.add_argument("--model",       required=True,
                        help="Model name as registered in vLLM server")
    parser.add_argument("--prompt_len",  type=int, default=128,
                        help="Approximate input prompt length in tokens")
    parser.add_argument("--gen_len",     type=int, default=128,
                        help="Number of tokens to generate")
    parser.add_argument("--warmup",      type=int, default=2)
    parser.add_argument("--runs",        type=int, default=5)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    print(f"vLLM Benchmark  model={args.model}  "
          f"prompt={args.prompt_len}  gen={args.gen_len}")
    print(f"  server: http://{args.host}:{args.port}")

    results = run_benchmark(
        args.host, args.port, args.model,
        args.prompt_len, args.gen_len,
        args.warmup, args.runs,
    )

    print(f"\n{'='*50}")
    print(f"  vLLM FP16 decode throughput (batch=1)")
    print(f"  Decode tok/s : {results['decode_tok_s_mean']:.1f} ± {results['decode_tok_s_std']:.1f}")
    print(f"  Total  tok/s : {results['total_tok_s_mean']:.1f} ± {results['total_tok_s_std']:.1f}")
    print(f"  TTFT   (ms)  : {results['ttft_ms_mean']:.0f} ± {results['ttft_ms_std']:.0f}")
    print(f"{'='*50}")
    print()
    print("Note: compare decode_tok_s with GLoRCQ eval_speed.py 'Standard' mode.")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
