"""fp16 reference memory probe for task #199 (exp 1 Table 3 fp16 row).

Loads the ORIGINAL fp16 HF model on the same hardware as mem_probe.py,
runs one 128-token prefill + a few decode steps, and records torch
alloc/peak/reserved + nvidia-smi + host RSS at each phase.

Run inside tmux test:0 (GPU 4):
  CUDA_VISIBLE_DEVICES=4 .venv/bin/python exp/cluster/fp16_ref_probe.py \
      --hf_model Qwen/Qwen1.5-MoE-A2.7B --prompt_len 128 --decode_steps 8
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch

from mem_probe import mark  # reuse phase logger (alloc/peak/reserved/smi/RSS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_model", default="Qwen/Qwen1.5-MoE-A2.7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--prompt_len", type=int, default=128)
    ap.add_argument("--decode_steps", type=int, default=8)
    args = ap.parse_args()

    dev = args.device
    mark("fp16 baseline (before load)", dev)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, torch_dtype=torch.float16,
        low_cpu_mem_usage=True, local_files_only=True,
    ).to(dev)
    model.eval()
    # analytic size for cross-check
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[fp16-ref] param count = {n_params/1e9:.3f} B → analytic fp16 = "
          f"{n_params*2/1024**3:.2f} GB", flush=True)
    mark("fp16 after load (RESIDENT model)", dev)

    text = "The history of natural language processing began in the 1950s. " * 20
    input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=args.prompt_len).input_ids.to(dev)
    print(f"[fp16-ref] prompt tokens = {input_ids.shape[1]}, "
          f"decode steps = {args.decode_steps}", flush=True)
    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=args.decode_steps, do_sample=False)
    torch.cuda.synchronize()
    print(f"[fp16-ref] generated {out.shape[1]-input_ids.shape[1]} tokens",
          flush=True)
    mark(f"fp16 after {args.prompt_len}-tok prefill + "
         f"{args.decode_steps} decode steps", dev)

    print("\n[fp16-ref] done.", flush=True)


if __name__ == "__main__":
    main()
