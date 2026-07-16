"""Exp 5 (task #199): cluster hit-rate of top-k routed experts.

Claim under test (paper C2): "the top-k active experts of a token frequently
fall into few (often one) cross-layer clusters", which justifies same-cluster
wide-GEMM batching and x@U reuse.

Two subcommands:

  trace   — GPU. Load the real-quant model, register forward hooks on every
            MoE router (GraphCompatibleMoeBlock.gate) and record the top-k
            selected expert indices per (layer, token) during real WikiText-2
            prefill+decode. Instrumentation is hook-only: production code is
            NOT modified. Saves a raw npz trace.

  analyze — CPU. Load the npz trace + one or more cross_layer_info.pt
            assignment maps and report, per wtype (gate/up/down assignments
            differ in this checkpoint), the distribution of the number of
            DISTINCT clusters among the top-4 experts, plus a
            random-routing baseline (expert ids permuted within layer).

Usage (trace, inside tmux test:0 / GPU 4):
  CUDA_VISIBLE_DEVICES=4 .venv/bin/python exp/cluster/cluster_hitrate.py trace \
      --model_path /mnt/Data/yqy/resource_dir/hf_dl/GLoRCQ-qwen1.5-moe-a2.7b-fair-grassmann-real \
      --out_npz /mnt/Data/yqy/resource_dir/glorcq_paper_exp/exp5_routing_trace.npz \
      --num_prompts 4 --prompt_len 128 --gen_len 128

Usage (analyze, CPU-only):
  .venv/bin/python exp/cluster/cluster_hitrate.py analyze \
      --trace_npz /mnt/Data/yqy/resource_dir/glorcq_paper_exp/exp5_routing_trace.npz \
      --cli main=/mnt/Data/yqy/resource_dir/hf_dl/GLoRCQ-qwen1.5-moe-a2.7b-fair-grassmann-real/cross_layer_info.pt \
      --out_json logs/cluster_validity/exp199_cluster_hitrate.json
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


# ---------------------------------------------------------------------------
# trace
# ---------------------------------------------------------------------------
def cmd_trace(args):
    import torch
    from utils.model_loader import load_model_and_tokenizer
    from inference.moe_block import GraphCompatibleMoeBlock

    dev = args.device
    print(f"[199-exp5] Loading real-quant model from {args.model_path} ...",
          flush=True)
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, device=dev, real_quant=True)
    model.eval()

    m = getattr(model, "model", model)
    moe_blocks = []
    for li, layer in enumerate(m.layers):
        for attr in ("mlp", "block_sparse_moe"):
            blk = getattr(layer, attr, None)
            if isinstance(blk, GraphCompatibleMoeBlock):
                moe_blocks.append((li, blk))
                break
    assert moe_blocks, "no GraphCompatibleMoeBlock found"
    top_k = moe_blocks[0][1].top_k
    print(f"[199-exp5] {len(moe_blocks)} MoE layers, top_k={top_k}", flush=True)

    # Per-event records (torch.topk on router logits == topk of softmax, since
    # softmax is monotonic; matches selected_experts in moe_block.forward).
    rec_layer, rec_pos, rec_prompt, rec_phase, rec_topk = [], [], [], [], []
    state = {"prompt_id": -1, "pos": {}}  # pos: layer_idx -> next token pos

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            logits = output  # (N, num_experts)
            n = logits.shape[0]
            idx = torch.topk(logits.float(), top_k, dim=-1).indices.cpu().numpy()
            p0 = state["pos"].get(layer_idx, 0)
            phase = 0 if n > 1 else 1  # 0=prefill chunk, 1=decode step
            for j in range(n):
                rec_layer.append(layer_idx)
                rec_pos.append(p0 + j)
                rec_prompt.append(state["prompt_id"])
                rec_phase.append(phase)
                rec_topk.append(idx[j])
            state["pos"][layer_idx] = p0 + n
        return hook

    handles = [blk.gate.register_forward_hook(make_hook(li))
               for li, blk in moe_blocks]

    # Real-text prompts from WikiText-2 test set (local HF cache).
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts, buf = [], ""
    for row in ds["text"]:
        buf += row
        if len(buf) > args.prompt_len * 8:  # ~enough chars for prompt_len toks
            texts.append(buf)
            buf = ""
        if len(texts) >= args.num_prompts:
            break

    for pid, text in enumerate(texts):
        input_ids = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=args.prompt_len).input_ids.to(dev)
        state["prompt_id"] = pid
        state["pos"] = {}
        print(f"[199-exp5] prompt {pid}: {input_ids.shape[1]} tokens prefill "
              f"+ {args.gen_len} decode ...", flush=True)
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=args.gen_len,
                           do_sample=False)

    for h in handles:
        h.remove()

    out = {
        "layer": np.asarray(rec_layer, dtype=np.int16),
        "pos": np.asarray(rec_pos, dtype=np.int32),
        "prompt_id": np.asarray(rec_prompt, dtype=np.int16),
        "phase": np.asarray(rec_phase, dtype=np.int8),
        "topk": np.asarray(rec_topk, dtype=np.int16),  # (n_events, top_k)
    }
    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez_compressed(args.out_npz, **out)
    n_dec = int((out["phase"] == 1).sum())
    print(f"[199-exp5] saved {len(rec_layer)} routing events "
          f"({n_dec} decode, {len(rec_layer)-n_dec} prefill) → {args.out_npz}",
          flush=True)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------
def _load_assignment_maps(cli_path):
    """Return {wtype: (n_layers, n_experts) int array of group_id}."""
    import torch
    cli = torch.load(cli_path, map_location="cpu", weights_only=False,
                     mmap=True)
    maps = {}
    for wt, recs in cli["assignments"].items():
        n_layers = max(r["layer"] for r in recs) + 1
        n_experts = max(r["expert"] for r in recs) + 1
        arr = np.full((n_layers, n_experts), -1, dtype=np.int32)
        for r in recs:
            arr[r["layer"], r["expert"]] = r["group_id"]
        maps[wt] = arr
    return maps


def _distinct_counts(topk, layer, amap):
    """Distinct clusters among top-k experts per event. topk: (n, k)."""
    gids = amap[layer[:, None], topk]          # (n, k) group ids
    g = np.sort(gids, axis=1)
    return 1 + (g[:, 1:] != g[:, :-1]).sum(axis=1)


def _summarize(dc, k):
    n = len(dc)
    out = {"n_events": int(n), "mean_distinct": float(dc.mean())}
    for j in range(1, k + 1):
        out[f"frac_eq_{j}"] = float((dc == j).mean())
        out[f"frac_le_{j}"] = float((dc <= j).mean())
    return out


def cmd_analyze(args):
    tr = np.load(args.trace_npz)
    layer = tr["layer"].astype(np.int64)
    topk = tr["topk"].astype(np.int64)
    phase = tr["phase"]
    k = topk.shape[1]
    print(f"[199-exp5] trace: {len(layer)} events, top_k={k}, "
          f"{int((phase == 1).sum())} decode / {int((phase == 0).sum())} "
          f"prefill", flush=True)

    rng = np.random.default_rng(42)
    results = {"trace_npz": args.trace_npz, "top_k": int(k), "maps": {}}
    for spec in args.cli:
        name, path = spec.split("=", 1)
        maps = _load_assignment_maps(path)
        results["maps"][name] = {"cli_path": path, "wtypes": {}}
        for wt, amap in maps.items():
            entry = {
                "n_clusters_global": int(amap.max() + 1),
                "all": _summarize(_distinct_counts(topk, layer, amap), k),
                "decode_only": _summarize(
                    _distinct_counts(topk[phase == 1], layer[phase == 1],
                                     amap), k),
            }
            # Per-layer mean distinct clusters (decode+prefill)
            dc_all = _distinct_counts(topk, layer, amap)
            entry["per_layer_mean"] = {
                str(li): float(dc_all[layer == li].mean())
                for li in np.unique(layer)
            }
            # Random-routing baseline: permute expert ids within each layer
            # (preserves each layer's cluster-size profile, destroys any
            # router-cluster correlation). 10 permutation rounds.
            rand_means, rand_eq1 = [], []
            for _ in range(10):
                pmap = amap.copy()
                for li in range(amap.shape[0]):
                    pmap[li] = pmap[li][rng.permutation(amap.shape[1])]
                dcr = _distinct_counts(topk, layer, pmap)
                rand_means.append(dcr.mean())
                rand_eq1.append((dcr == 1).mean())
            entry["random_baseline"] = {
                "mean_distinct": float(np.mean(rand_means)),
                "frac_eq_1": float(np.mean(rand_eq1)),
                "n_perms": 10,
            }
            results["maps"][name]["wtypes"][wt] = entry
            a = entry["all"]
            print(f"  [{name}/{wt}] clusters={entry['n_clusters_global']} "
                  f"mean={a['mean_distinct']:.3f} | =1: {a['frac_eq_1']:.3f} "
                  f"<=2: {a['frac_le_2']:.3f} <=3: {a['frac_le_3']:.3f} "
                  f"=4: {a['frac_eq_4']:.3f} | rand mean="
                  f"{entry['random_baseline']['mean_distinct']:.3f}",
                  flush=True)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[199-exp5] wrote {args.out_json}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    tp = sub.add_parser("trace")
    tp.add_argument("--model_path", required=True)
    tp.add_argument("--device", default="cuda:0")
    tp.add_argument("--num_prompts", type=int, default=4)
    tp.add_argument("--prompt_len", type=int, default=128)
    tp.add_argument("--gen_len", type=int, default=128)
    tp.add_argument("--out_npz", required=True)
    tp.set_defaults(func=cmd_trace)

    apz = sub.add_parser("analyze")
    apz.add_argument("--trace_npz", required=True)
    apz.add_argument("--cli", nargs="+", required=True,
                     help="name=path/to/cross_layer_info.pt (repeatable)")
    apz.add_argument("--out_json", default=None)
    apz.set_defaults(func=cmd_analyze)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
