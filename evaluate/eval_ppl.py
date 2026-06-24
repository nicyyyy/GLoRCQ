"""
WikiText-2 perplexity evaluation for GLoRCQ fake-quant models.

Uses a standard sliding-window approach: the model receives overlapping
windows of the full test set and we accumulate the per-token NLL only
for the non-overlapping (stride) portion of each window.

Usage:
    uv run python evaluate/eval_ppl.py \
        --model_path ./output/model-fakequant \
        --max_length 2048 --stride 512
"""

import argparse
import json
import os
import sys

import torch
from datasets import load_dataset

# Ensure project root is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)


@torch.no_grad()
def eval_ppl_sliding_window(model, tokenizer, max_length=2048, stride=512,
                            device="cuda:0"):
    """Evaluate WikiText-2 perplexity with a sliding window.

    Returns:
        ppl (float): perplexity on the test set.
    """
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(testdata["text"])
    encodings = tokenizer(text, return_tensors="pt")
    # When device_map="auto" is used, send input to whichever device holds
    # the first parameter (typically the first visible GPU).
    _input_device = device if device != "auto" else next(model.parameters()).device
    input_ids = encodings.input_ids.to(_input_device)

    seq_len = input_ids.size(1)
    print(f"Dataset tokens: {seq_len}, max_length: {max_length}, stride: {stride}")

    nlls = []
    num_tokens = 0
    prev_end = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end  # number of new tokens to score
        input_window = input_ids[:, begin_loc:end_loc]

        # Build target: mask out tokens already scored
        target_ids = input_window.clone()
        target_ids[:, :-trg_len] = -100

        outputs = model(input_window, labels=target_ids)
        # outputs.loss is already averaged over the valid (non -100) tokens
        neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        num_tokens += trg_len
        prev_end = end_loc

        if end_loc >= seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / num_tokens).item()
    return ppl


def main():
    parser = argparse.ArgumentParser(
        description="WikiText-2 perplexity evaluation (fake-quant fp16 model)",
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fake-quant HF model directory")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Sliding window size in tokens (default: 2048)")
    parser.add_argument("--stride", type=int, default=512,
                        help="Stride between windows (default: 512)")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save results as JSON")
    args = parser.parse_args()

    from utils.model_loader import load_model_and_tokenizer

    print(f"Loading fake-quant model from {args.model_path} ...")
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, device=args.device, real_quant=False,
    )

    ppl = eval_ppl_sliding_window(
        model, tokenizer,
        max_length=args.max_length,
        stride=args.stride,
        device=args.device,
    )
    print(f"\nWikiText-2 PPL: {ppl:.2f}")

    results = {
        "model_path": args.model_path,
        "wikitext2_ppl": ppl,
        "max_length": args.max_length,
        "stride": args.stride,
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
