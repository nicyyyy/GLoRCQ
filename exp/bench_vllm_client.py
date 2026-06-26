"""
vLLM FP16 throughput benchmark for GLoRCQ paper speed comparison.

Measures two metrics so results map cleanly to the paper table:

  generate_tok_s  [PRIMARY — use this for the paper table]
    Non-streaming request.  Measures wall-clock time from request sent to
    full response received, then divides by completion_tokens from the usage
    field.  This matches evaluate/eval_speed.py's model.generate() timing
    exactly, so the numbers are directly comparable.

  decode_tok_s  [SECONDARY — decode-only, higher than generate_tok_s]
    Streaming request.  Time from first token received to last token, i.e.
    pure autoregressive decode latency excluding prefill / TTFT.

  ttft_ms
    Time to first token (prefill latency proxy).

Usage:
    python exp/bench_vllm_client.py \\
        --host localhost --port 8010 \\
        --model Qwen/Qwen1.5-MoE-A2.7B \\
        --prompt_len 128 --gen_len 128 --warmup 2 --runs 5 \\
        --output_json logs/vllm_bench_Qwen1.5-MoE-A2.7B.json
"""

import argparse
import json
import os
import string
import time
from statistics import mean, stdev

import requests

ALPHABET = string.ascii_letters + string.digits + " .,!?"


def _make_prompt(approx_tokens: int) -> str:
    chars = approx_tokens * 4
    return "".join(ALPHABET[i % len(ALPHABET)] for i in range(chars))


# ---------------------------------------------------------------------------
# Non-streaming  →  generate_tok_s  (PRIMARY)
# ---------------------------------------------------------------------------
def _generate_total(host: str, port: int, model: str, prompt: str, max_tokens: int):
    """
    Blocking (non-streaming) completion.
    Returns (total_seconds, n_completion_tokens).
    n_completion_tokens comes from the usage field, so it is exact.
    """
    url = f"http://{host}:{port}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=300)
    t1 = time.perf_counter()
    resp.raise_for_status()
    n_tokens = resp.json().get("usage", {}).get("completion_tokens", max_tokens)
    return t1 - t0, n_tokens


# ---------------------------------------------------------------------------
# Streaming  →  decode_tok_s + ttft  (SECONDARY)
# ---------------------------------------------------------------------------
def _generate_stream(host: str, port: int, model: str, prompt: str, max_tokens: int):
    """
    Streaming completion with include_usage=True so the final SSE chunk
    carries the exact completion_tokens count from vLLM.
    Returns (ttft_seconds, decode_seconds, n_completion_tokens).
    """
    url = f"http://{host}:{port}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},  # vLLM 0.3+
    }

    t_start = time.perf_counter()
    t_first = None
    n_tokens = 0

    with requests.post(url, json=payload, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            if raw == b"data: [DONE]":
                break
            if not raw.startswith(b"data: "):
                continue
            chunk = json.loads(raw[6:])
            # Usage chunk (sent as the last data event when include_usage=True)
            if chunk.get("usage"):
                n_tokens = chunk["usage"].get("completion_tokens", n_tokens)
            # First non-empty text marks end of prefill
            text = chunk.get("choices", [{}])[0].get("text", "")
            if text and t_first is None:
                t_first = time.perf_counter()

    t_end = time.perf_counter()
    ttft_s   = (t_first - t_start) if t_first else (t_end - t_start)
    decode_s = (t_end - t_first)   if t_first else 0.0
    return ttft_s, decode_s, n_tokens or max_tokens


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def run_benchmark(host, port, model, prompt_len, gen_len, warmup, runs):
    prompt = _make_prompt(prompt_len)
    print(f"  prompt ~{prompt_len} tokens, generate {gen_len} tokens, "
          f"warmup={warmup}, runs={runs}")

    # ── Warmup ──────────────────────────────────────────────────────────────
    print(f"  Warmup ({warmup} rounds)...")
    for _ in range(warmup):
        _generate_total(host, port, model, prompt, gen_len)
        _generate_stream(host, port, model, prompt, gen_len)

    # ── Non-streaming: generate_tok_s ────────────────────────────────────────
    print(f"  [1/2] generate_tok_s — non-streaming ({runs} runs)...")
    gen_tps = []
    for i in range(runs):
        total_s, n_tok = _generate_total(host, port, model, prompt, gen_len)
        tps = n_tok / total_s
        gen_tps.append(tps)
        print(f"    run {i+1}/{runs}: {n_tok} tok / {total_s:.3f}s = {tps:.1f} tok/s")

    # ── Streaming: decode_tok_s + TTFT ───────────────────────────────────────
    print(f"  [2/2] decode_tok_s + TTFT — streaming ({runs} runs)...")
    dec_tps, ttft_ms_list = [], []
    for i in range(runs):
        ttft_s, decode_s, n_tok = _generate_stream(host, port, model, prompt, gen_len)
        tps = n_tok / decode_s if decode_s > 1e-4 else float("inf")
        dec_tps.append(tps)
        ttft_ms_list.append(ttft_s * 1000)
        print(f"    run {i+1}/{runs}: TTFT={ttft_s*1000:.0f}ms  "
              f"{n_tok} tok / {decode_s:.3f}s = {tps:.1f} tok/s")

    def _s(lst):
        return mean(lst), (stdev(lst) if len(lst) > 1 else 0.0)

    gm, gs = _s(gen_tps)
    dm, ds = _s(dec_tps)
    tm, ts = _s(ttft_ms_list)

    return {
        "model": model,
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "warmup": warmup,
        "runs": runs,
        # PRIMARY: matches GLoRCQ eval_speed.py model.generate() timing
        "generate_tok_s_mean": round(gm, 2),
        "generate_tok_s_std":  round(gs, 2),
        # SECONDARY: decode-only (excludes prefill, higher number)
        "decode_tok_s_mean": round(dm, 2),
        "decode_tok_s_std":  round(ds, 2),
        # Prefill latency
        "ttft_ms_mean": round(tm, 1),
        "ttft_ms_std":  round(ts, 1),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="vLLM FP16 baseline benchmark for GLoRCQ paper comparison"
    )
    p.add_argument("--host",        default="localhost")
    p.add_argument("--port",        type=int, default=8010)
    p.add_argument("--model",       required=True,
                   help="Model name exactly as served by vLLM")
    p.add_argument("--prompt_len",  type=int, default=128,
                   help="Approximate input length in tokens (default: 128)")
    p.add_argument("--gen_len",     type=int, default=128,
                   help="Number of tokens to generate (default: 128)")
    p.add_argument("--warmup",      type=int, default=2)
    p.add_argument("--runs",        type=int, default=5)
    p.add_argument("--output_json", default=None,
                   help="Save results to this JSON file")
    args = p.parse_args()

    print(f"\nvLLM FP16 Benchmark  ({args.model})")
    print(f"  server : http://{args.host}:{args.port}")
    print()

    results = run_benchmark(
        args.host, args.port, args.model,
        args.prompt_len, args.gen_len,
        args.warmup, args.runs,
    )

    W = 56
    print(f"\n{'='*W}")
    print(f"  vLLM FP16  batch=1  prompt={args.prompt_len}  gen={args.gen_len}")
    print(f"  {'generate_tok_s [PRIMARY]':32s}: "
          f"{results['generate_tok_s_mean']:5.1f} ± {results['generate_tok_s_std']:.1f} tok/s")
    print(f"  {'decode_tok_s  [decode-only]':32s}: "
          f"{results['decode_tok_s_mean']:5.1f} ± {results['decode_tok_s_std']:.1f} tok/s")
    print(f"  {'TTFT':32s}: "
          f"{results['ttft_ms_mean']:5.0f} ± {results['ttft_ms_std']:.0f} ms")
    print(f"{'='*W}")
    print(f"  → Compare generate_tok_s with GLoRCQ eval_speed.py 'Graph' column")
    print(f"{'='*W}")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {args.output_json}")


if __name__ == "__main__":
    main()
