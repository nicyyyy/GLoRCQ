#!/usr/bin/env python3
"""Diagnose the self-managed MLA static cache vs eager DynamicCache, per decode
step. Both run MoE graph_mode=False (proven identical to eager) so this isolates
PURE attention-cache correctness (no CUDA graph, no all-experts path)."""
import os, sys, torch
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
REAL = "/mnt/Data/yqy/resource_dir/glorcq_paper_exp/deepseek_v2lite_real"


def main():
    from transformers import AutoTokenizer
    from transformers.cache_utils import DynamicCache
    from inference.model_builder import load_glorcq_model
    from inference import deepseek_support

    ret = load_glorcq_model(REAL, device="cuda:0")
    model = ret[0] if isinstance(ret, tuple) else ret
    tok = AutoTokenizer.from_pretrained(REAL, trust_remote_code=True, use_fast=False)
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda:0")
    seq = ids.shape[1]
    print(f"prompt tokens={seq}", flush=True)

    # ---- Reference: eager manual greedy decode with DynamicCache ----
    ref_logits = []
    ref_ids = ids.clone()
    with torch.no_grad():
        cache = DynamicCache()
        am = torch.ones((1, seq), dtype=torch.long, device="cuda:0")
        out = model(input_ids=ids, attention_mask=am, past_key_values=cache,
                    use_cache=True, return_dict=True)
        logit = out.logits[:, -1, :]
        ref_logits.append(logit.clone())
        nxt = logit.argmax(-1, keepdim=True)
        ref_ids = torch.cat([ref_ids, nxt], 1)
        for _ in range(15):
            am = torch.ones((1, ref_ids.shape[1]), dtype=torch.long, device="cuda:0")
            out = model(input_ids=nxt, attention_mask=am,
                        past_key_values=out.past_key_values,
                        use_cache=True, return_dict=True)
            logit = out.logits[:, -1, :]
            ref_logits.append(logit.clone())
            nxt = logit.argmax(-1, keepdim=True)
            ref_ids = torch.cat([ref_ids, nxt], 1)
    print(f"[eager] {tok.decode(ref_ids[0], skip_special_tokens=True)!r}", flush=True)

    # ---- Mine: static cache, EAGER (no graph), MoE graph_mode=False ----
    runner = deepseek_support.install_full_decode_graph(model, max_seq_len=64,
                                                        device="cuda:0")
    runner._set_moe_graph_mode(False)          # isolate attention (no all-experts)
    with torch.no_grad():
        nxt = runner._prefill(ids)             # prefill logits' argmax
        # compare prefill argmax
        print(f"[mine] prefill next tok = {nxt.item()}  eager = {ref_ids[0,seq].item()}",
              flush=True)
        mine_ids = torch.cat([ids.clone(), nxt], 1)
        runner.cache.mode = "decode"
        runner._set_moe_graph_mode(False)
        for i in range(15):
            pos = seq + i
            runner.static_input_id.copy_(nxt)
            runner.static_position_ids.fill_(pos)
            runner.cache.cache_position.fill_(pos)
            logit = runner._decode_forward()[:, -1, :]   # eager call, no replay
            d = (logit.float() - ref_logits[i + 1].float()).abs().max().item()
            my_arg = logit.argmax(-1).item()
            ref_arg = ref_logits[i + 1].argmax(-1).item()
            flag = "" if my_arg == ref_arg else "  <-- MISMATCH"
            print(f"step {i} pos {pos}: max|dlogit|={d:.4f} mine_arg={my_arg} "
                  f"ref_arg={ref_arg}{flag}", flush=True)
            nxt = logit.argmax(-1, keepdim=True)
            mine_ids = torch.cat([mine_ids, nxt], 1)
    print(f"[mine]  {tok.decode(mine_ids[0], skip_special_tokens=True)!r}", flush=True)


if __name__ == "__main__":
    main()
