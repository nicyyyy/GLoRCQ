#!/usr/bin/env python3
"""bs=1 eager decode sanity + speed for the DeepSeek-V2-Lite real-quant ckpt
(exercises moe_block non-graph forward + GLoRCQLinear experts). Also measures
the fp16 eager baseline for the ratio."""
import os, sys, time, argparse, torch
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)

# DeepSeek-V2 remote modeling (written for tf ~4.36) calls DynamicCache
# .get_max_length(), removed in tf 4.51 (renamed get_max_cache_shape). For a
# DynamicCache it returned None (unbounded). Add the shim so generate() works.
from transformers.cache_utils import DynamicCache as _DC
if not hasattr(_DC, "get_max_length"):
    _DC.get_max_length = lambda self: None

def bench(model, tok, prompt_len=128, gen_len=128, label=""):
    torch.manual_seed(0)
    ids = torch.randint(0, 30000, (1, prompt_len), device="cuda:0")
    # warmup
    with torch.no_grad():
        _ = model.generate(ids, max_new_tokens=8, do_sample=False,
                            use_cache=True, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=gen_len, do_sample=False,
                             use_cache=True, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    dt = time.time() - t0
    n_new = out.shape[1] - prompt_len
    tps = n_new / dt
    print(f"[{label}] {n_new} new tokens in {dt:.2f}s = {tps:.2f} tok/s", flush=True)
    return tps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "fp16"], required=True)
    ap.add_argument("--prompt_len", type=int, default=128)
    ap.add_argument("--gen_len", type=int, default=128)
    args = ap.parse_args()

    REAL = "/mnt/Data/yqy/resource_dir/glorcq_paper_exp/deepseek_v2lite_real"
    FP16 = "/mnt/Data/yqy/resource_dir/deepseek-v2-lite"
    from transformers import AutoTokenizer

    if args.mode == "real":
        from inference.model_builder import load_glorcq_model
        ret = load_glorcq_model(REAL, device="cuda:0")
        model = ret[0] if isinstance(ret, tuple) else ret
        tok = AutoTokenizer.from_pretrained(REAL, trust_remote_code=True, use_fast=False)
        label = "real-quant eager"
    else:
        from transformers import AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(FP16, trust_remote_code=True, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            FP16, trust_remote_code=True, torch_dtype=torch.float16).to("cuda:0")
        model.eval()
        label = "fp16 eager"

    # Sanity: real text decode
    prompt = "The capital of France is"
    pids = tok(prompt, return_tensors="pt").input_ids.to("cuda:0")
    with torch.no_grad():
        g = model.generate(pids, max_new_tokens=32, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    print(f"[{label}] SAMPLE: {tok.decode(g[0], skip_special_tokens=True)!r}", flush=True)

    tps = bench(model, tok, args.prompt_len, args.gen_len, label)
    print(f"RESULT {args.mode} tok_s={tps:.2f}", flush=True)

if __name__ == "__main__":
    main()
