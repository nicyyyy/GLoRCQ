"""
Print a human-readable summary of GLoRCQ evaluation results.

Reads ppl.json, zeroshot_5task.json, and zeroshot_mmlu.json from output_dir
and prints a compact table.  Also writes summary.txt.

Usage:
    python evaluate/print_eval_summary.py --output_dir logs/eval_foo --model_path foo/bar
"""

import argparse
import json
import os


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default="")
    args = parser.parse_args()

    d = args.output_dir
    ppl_data  = load_json(os.path.join(d, "ppl.json"))
    zs5_data  = load_json(os.path.join(d, "zeroshot_5task.json"))
    mmlu_data = load_json(os.path.join(d, "zeroshot_mmlu.json"))

    lines = [
        f"Model   : {args.model_path}",
        "=" * 52,
    ]

    if ppl_data:
        ppl = ppl_data.get("wikitext2_ppl", "N/A")
        lines.append(f"WikiText-2 PPL     : {ppl:.4f}")

    task_order = ["arc_challenge", "arc_easy", "winogrande", "hellaswag", "piqa"]
    task_label = {
        "arc_challenge": "ARC-Challenge",
        "arc_easy":      "ARC-Easy",
        "winogrande":    "WinoGrande",
        "hellaswag":     "HellaSwag",
        "piqa":          "PIQA",
    }

    vals = []
    if zs5_data:
        lines.append("-" * 52)
        for t in task_order:
            r = zs5_data.get("task_results", {}).get(t, {})
            v = r.get("value")
            if v is not None:
                vals.append(v)
                metric = r.get("metric", "acc")
                lines.append(f"  {task_label.get(t, t):<20} {v * 100:6.2f}%  ({metric})")

    if mmlu_data:
        r = mmlu_data.get("task_results", {}).get("mmlu", {})
        v = r.get("value")
        if v is not None:
            vals.append(v)
            metric = r.get("metric", "acc")
            lines.append(f"  {'MMLU (5-shot)':<20} {v * 100:6.2f}%  ({metric})")

    if vals:
        avg = sum(vals) / len(vals)
        lines.append("-" * 52)
        lines.append(f"  {'Average':<20} {avg * 100:6.2f}%  ({len(vals)} tasks)")

    lines.append("=" * 52)
    text = "\n".join(lines)
    print(text)

    out_path = os.path.join(d, "summary.txt")
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    main()
