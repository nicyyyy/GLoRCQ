#!/usr/bin/env python3
"""bs=1 decode with per-MoE-block CUDA graphs for the DeepSeek-V2-Lite real-quant
ckpt. Compares eager vs graphed decode; verifies sane text (graphs must not
change outputs)."""
import os, sys, time, argparse, torch
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)

REAL = "/mnt/Data/yqy/resource_dir/glorcq_paper_exp/deepseek_v2lite_real"


def bench(model, tok, prompt_len, gen_len, label):
    ids = torch.randint(0, 30000, (1, prompt_len), device="cuda:0")
    with torch.no_grad():
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
    ap.add_argument("--prompt_len", type=int, default=128)
    ap.add_argument("--gen_len", type=int, default=128)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from inference.model_builder import load_glorcq_model
    from inference import deepseek_support

    ret = load_glorcq_model(REAL, device="cuda:0")
    model = ret[0] if isinstance(ret, tuple) else ret
    tok = AutoTokenizer.from_pretrained(REAL, trust_remote_code=True, use_fast=False)

    # Eager baseline (MoE blocks in sparse/decode mode)
    sample(model, tok, "eager")
    eager = bench(model, tok, args.prompt_len, args.gen_len, "eager")

    # Install per-MoE-block CUDA graphs, then re-measure + re-check text
    runners = deepseek_support.install_moe_block_graphs(model, device="cuda:0")
    sample(model, tok, "moe-graph")
    graphed = bench(model, tok, args.prompt_len, args.gen_len, "moe-graph")

    print(f"RESULT eager={eager:.2f} moe_graph={graphed:.2f} "
          f"speedup={graphed/eager:.2f}x", flush=True)


if __name__ == "__main__":
    main()
