"""
Zero-shot benchmark evaluation for GLoRCQ fake-quant models.

Uses lm-eval-harness (``lm_eval``) with the HuggingFace LM wrapper (HFLM).
The fake-quant model is a standard HF ``PreTrainedModel`` and can be passed
directly to HFLM without monkey-patching.

Usage:
    uv run python evaluate/eval_zeroshot.py \
        --model_path ./output/model-fakequant \
        --tasks hellaswag,lambada_openai,piqa,winogrande,arc_easy,arc_challenge
"""

import argparse
import json
import os
import sys

import torch

# Ensure project root is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

DEFAULT_TASKS = "hellaswag,lambada_openai,piqa,winogrande,arc_easy,arc_challenge"


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot benchmark evaluation (fake-quant fp16 model)",
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fake-quant HF model directory")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tasks", type=str, default=DEFAULT_TASKS,
                        help=f"Comma-separated task list (default: {DEFAULT_TASKS})")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for lm-eval (default: 64)")
    parser.add_argument("--num_fewshot", type=int, default=0,
                        help="Number of few-shot examples (default: 0)")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save results as JSON")
    parser.add_argument("--metric_mode", default="auto",
                        choices=["auto", "acc", "acc_norm"],
                        help="Metric priority: auto=prefer acc_norm, acc=prefer raw acc, acc_norm=force normalized")
    args = parser.parse_args()

    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    from utils.model_loader import load_model_and_tokenizer

    print(f"Loading fake-quant model from {args.model_path} ...")
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, device=args.device, real_quant=False,
    )

    task_list = [t.strip() for t in args.tasks.split(",")]
    print(f"Tasks: {task_list}")
    print(f"Batch size: {args.batch_size}, num_fewshot: {args.num_fewshot}")

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
    )

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  Zero-Shot Results  (model: {args.model_path})")
    print(f"{'='*60}")
    print(f"  {'Task':<25} {'Metric':<15} {'Value':>10}")
    print(f"  {'-'*50}")

    if args.metric_mode == "acc":
        metric_priority = ("acc,none", "acc", "acc_norm,none", "acc_norm")
    elif args.metric_mode == "acc_norm":
        metric_priority = ("acc_norm,none", "acc_norm", "acc,none", "acc")
    else:
        metric_priority = ("acc_norm,none", "acc,none", "acc_norm", "acc")

    task_results = {}
    for task_name in task_list:
        if task_name in results["results"]:
            task_data = results["results"][task_name]
            for metric_key in metric_priority:
                if metric_key in task_data:
                    val = task_data[metric_key]
                    metric_label = metric_key.replace(",none", "")
                    print(f"  {task_name:<25} {metric_label:<15} {val:>10.4f}")
                    task_results[task_name] = {
                        "metric": metric_label,
                        "value": float(val),
                    }
                    break

    # Compute average
    if task_results:
        avg = sum(v["value"] for v in task_results.values()) / len(task_results)
        print(f"  {'-'*50}")
        print(f"  {'Average':<25} {'acc':<15} {avg:>10.4f}")
        task_results["average"] = {"metric": "acc", "value": float(avg)}

    print(f"{'='*60}")

    # Save results
    output = {
        "model_path": args.model_path,
        "tasks": task_list,
        "num_fewshot": args.num_fewshot,
        "task_results": task_results,
        "full_results": results["results"],
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
