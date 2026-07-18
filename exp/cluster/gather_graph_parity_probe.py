"""
Task #203 parity + speed probe: Mixtral top-k GATHER graph path.

Verifies the new _forward_graph_vq4_gather path in inference/moe_block.py:

  1. MIXTRAL PARITY: fixed 32-tok prompt + 8 greedy decode steps.
       ref    = standard decode (graph_mode=False -> _forward_decode)
       gather = eager gather graph (graph_mode=True -> _forward_graph_vq4_gather)
     Compares the 8 decode-step logits (bit-identical ideal; tiny fp diff ok).
     Also checks greedy token ids match.

  2. MIXTRAL CUDA-GRAPH CAPTURE: capture + replay via GLoRCQGraphWrapper
     (GLORCQ_MIXTRAL_GRAPH=1) and compare its greedy tokens to the eager gather
     tokens -> proves capture is correct.

  3. QWEN ISOLATION: assert num_experts>16 blocks have _use_gather_graph=False and
     the gather method is NEVER entered during a graph decode (instrumented
     counter) -> Qwen takes the byte-identical all-experts path.

Usage:
  python exp/cluster/gather_graph_parity_probe.py <mixtral_dir> [qwen_dir]
"""
import os
import sys
import time

sys.path.insert(0, "/home/qyyang/repo/GLoRCQ")

import torch  # noqa: E402
from transformers import DynamicCache  # noqa: E402


def _p(*a):
    print(*a, flush=True)


DEV = "cuda:0"
MIXTRAL = sys.argv[1] if len(sys.argv) > 1 else None
QWEN = sys.argv[2] if len(sys.argv) > 2 else None


def _set_graph_mode(model, mode):
    from inference.moe_block import GraphCompatibleMoeBlock
    n = 0
    for m in model.modules():
        if isinstance(m, GraphCompatibleMoeBlock):
            m.graph_mode = mode
            n += 1
    return n


def _decode_logits(model, prompt, n_steps, decode_graph_mode):
    """Prefill (graph_mode=False) then n_steps N=1 decode at given graph_mode.
    Returns (n_steps, vocab) stacked decode logits + list of greedy token ids."""
    _set_graph_mode(model, False)
    logits_seq = []
    ids = []
    with torch.no_grad():
        past = DynamicCache()
        out = model(input_ids=prompt, past_key_values=past,
                    use_cache=True, return_dict=True)
        nxt = out.logits[:, -1:].argmax(-1)
        _set_graph_mode(model, decode_graph_mode)
        for _ in range(n_steps):
            out = model(input_ids=nxt, past_key_values=past,
                        use_cache=True, return_dict=True)
            logits_seq.append(out.logits[:, -1, :].clone())
            nxt = out.logits[:, -1:].argmax(-1)
            ids.append(int(nxt.item()))
    _set_graph_mode(model, False)
    return torch.cat(logits_seq), ids


def probe_mixtral():
    from inference.model_builder import load_glorcq_model
    _p(f"\n===== MIXTRAL PARITY =====\n[load] {MIXTRAL}")
    t0 = time.time()
    model = load_glorcq_model(MIXTRAL, device=DEV)
    model.eval()
    torch.cuda.synchronize()
    _p(f"[load] done {time.time()-t0:.0f}s")

    # Report gather gate + shim stats on the MoE blocks.
    from inference.moe_block import GraphCompatibleMoeBlock
    blocks = [m for m in model.modules() if isinstance(m, GraphCompatibleMoeBlock)]
    _p(f"[info] MoE blocks={len(blocks)} num_experts={blocks[0].num_experts} "
       f"top_k={blocks[0].top_k} use_gather={blocks[0]._use_gather_graph}")

    torch.manual_seed(1)
    vocab = getattr(model.config, "vocab_size", 32000)
    prompt = torch.randint(0, min(vocab, 30000), (1, 32), device=DEV)

    L_ref, ids_ref = _decode_logits(model, prompt, 8, decode_graph_mode=False)
    L_gat, ids_gat = _decode_logits(model, prompt, 8, decode_graph_mode=True)

    bit = torch.equal(L_ref, L_gat)
    maxd = (L_ref.float() - L_gat.float()).abs().max().item()
    rel = maxd / (L_ref.float().abs().max().item() + 1e-9)
    tok_match = (ids_ref == ids_gat)
    _p(f"[parity] standard-vs-gather: bitwise={bit} max_abs_diff={maxd:.3e} "
       f"rel={rel:.3e}")
    _p(f"[parity] greedy ids ref   ={ids_ref}")
    _p(f"[parity] greedy ids gather={ids_gat}  tokens_match={tok_match}")

    # ---- CUDA-Graph capture parity ----
    _p("\n===== MIXTRAL CUDA-GRAPH CAPTURE =====")
    os.environ["GLORCQ_MIXTRAL_GRAPH"] = "1"
    from inference.graph_wrapper import GLoRCQGraphWrapper
    try:
        wrapper = GLoRCQGraphWrapper(model, max_batch_size=1, max_seq_len=2048)
        wrapper.capture_graph()
        captured = wrapper.graph is not None
        _p(f"[capture] graph captured = {captured}")
        if captured:
            out_ids, dur = wrapper.generate(prompt, max_new_tokens=9)
            gen_tail = out_ids[0, 32:32 + 8].tolist()
            _p(f"[capture] graph greedy ids = {gen_tail}")
            _p(f"[capture] eager gather ids = {ids_gat}")
            _p(f"[capture] tokens_match(eager gather) = {gen_tail == ids_gat}")
    except Exception as e:
        import traceback
        _p(f"[capture] FAILED: {e}")
        traceback.print_exc()
        captured = False

    ok = tok_match and captured
    _p(f"\n[MIXTRAL RESULT] parity_tokens_match={tok_match} "
       f"maxdiff={maxd:.3e} capture={captured} -> {'PASS' if ok else 'CHECK'}")
    del model
    torch.cuda.empty_cache()
    return ok


def probe_qwen_isolation():
    if QWEN is None:
        _p("\n[QWEN] skipped (no qwen dir arg)")
        return True
    from inference.model_builder import load_glorcq_model
    import inference.moe_block as mb
    _p(f"\n===== QWEN ISOLATION =====\n[load] {QWEN}")
    model = load_glorcq_model(QWEN, device=DEV)
    model.eval()

    blocks = [m for m in model.modules()
              if isinstance(m, mb.GraphCompatibleMoeBlock)]
    all_off = all(not b._use_gather_graph for b in blocks)
    _p(f"[qwen] MoE blocks={len(blocks)} num_experts={blocks[0].num_experts} "
       f"use_gather(all False)={all_off}")

    # Instrument: count entries into the gather method.
    orig = mb.GraphCompatibleMoeBlock._forward_graph_vq4_gather
    counter = {"n": 0}

    def _wrapped(self, *a, **k):
        counter["n"] += 1
        return orig(self, *a, **k)
    mb.GraphCompatibleMoeBlock._forward_graph_vq4_gather = _wrapped

    torch.manual_seed(1)
    vocab = getattr(model.config, "vocab_size", 30000)
    prompt = torch.randint(0, min(vocab, 30000), (1, 32), device=DEV)
    _, ids = _decode_logits(model, prompt, 8, decode_graph_mode=True)
    mb.GraphCompatibleMoeBlock._forward_graph_vq4_gather = orig

    never = (counter["n"] == 0)
    _p(f"[qwen] gather-method entries during graph decode = {counter['n']} "
       f"(must be 0) -> never_entered={never}")
    _p(f"[qwen] greedy ids (graph decode) = {ids}")
    ok = all_off and never
    _p(f"[QWEN RESULT] use_gather_all_off={all_off} gather_never_entered={never} "
       f"-> {'PASS' if ok else 'FAIL'}")
    del model
    torch.cuda.empty_cache()
    return ok


if __name__ == "__main__":
    r1 = probe_mixtral() if MIXTRAL else True
    r2 = probe_qwen_isolation()
    _p(f"\n==== OVERALL: mixtral={r1} qwen_iso={r2} ====")
