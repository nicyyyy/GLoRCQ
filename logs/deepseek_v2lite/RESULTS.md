# DeepSeek-V2-Lite GLoRCQ port — results (2026-07-19)

Model: deepseek-ai/DeepSeek-V2-Lite (deepseek_v2, trust_remote_code)
27 layers; layer 0 dense (first_k_dense_replace=1); 64 routed + 2 shared experts, top-6;
expert dims 2048<->1408 (identical to Qwen1.5-MoE). MLA attention.

Recipe: fair-bit r=32, G=128, qbit=2, group_size=128, lora 16-bit (int8 U/V), pool_kmeans,
cluster_method=grassmannian (seed 42). DE-RISK pass: attn_bits=16 → MLA attention, shared
experts, router, and layer-0 dense MLP all kept FP16; only the 64×26 routing experts are
VQ4 + cross-layer Grassmann-shared LoRA (the paper contribution).

## Bits (quantized MoE routing experts — the contribution)
- weight 2.0000 + LoRA 0.1576 = **2.1576 bits/param** (Extra above qbit=2: +0.1576)
- Grassmannian: 13 clusters/wtype (ceil(1664/128)); 4992 expert LoRAs filled;
  10 down_proj experts exceeded max_err=60 → kept as fp16 shim.
- Attention / shared experts / dense layer-0 MLP: FP16 (de-risk pass).

## WikiText-2 PPL (sliding window, max_len 2048, stride 512)
- GLoRCQ fake-quant : **6.44**
- FP16 baseline     : 5.65
- gap: +0.79 PPL

## Zero-shot — CANONICAL Table-1 protocol (metric=acc, add_bos=True, 0-shot)
Same harness applied to BOTH CLASP (quant) and fp16 for a directly-comparable row.
(batch 16; lm-eval loglikelihood acc is batch-invariant == batch 1.)

| task           | CLASP acc | fp16 acc |
|----------------|-----------|----------|
| ARC-Challenge  | 0.4317    | 0.4386   |
| ARC-Easy       | 0.7471    | 0.7753   |
| WinoGrande     | 0.6882    | 0.7072   |
| HellaSwag      | 0.5314    | 0.5854   |
| PIQA           | 0.7595    | 0.8036   |
| **Average (5)**| **0.6316**| **0.6620** |

CLASP 63.16% vs fp16 66.20% → −3.04 pts (2-bit expert quant).
(Earlier acc_norm/no-add_bos run, for reference: CLASP avg 0.6682.)

## bs=1 decode speed (prompt 128, gen 128, GPU A100)
- real-quant eager             : ~6.4-6.7 tok/s
- **real-quant + MoE-block CUDA graph : 17.70 tok/s  (2.78× over eager; BEATS fp16)**
- fp16 eager                   : 16.70 tok/s  → real-quant graphed ratio **1.06×**
- Full-model CUDA graph (graph_wrapper) does NOT engage: it uses StaticCache +
  cache_position, and DeepSeek-V2's MLA caches asymmetric multi-head K/V (key
  head_dim=192, value head_dim=128) that StaticCache (single head_dim) cannot
  hold; its remote modeling uses the legacy DynamicCache API. So instead we
  capture ONLY the launch-bound MoE-block compute (64 routed + shared expert)
  per layer into 26 per-block CUDA graphs, leaving MLA attention eager (fp16,
  DynamicCache). Isolated in deepseek_support.install_moe_block_graphs(); the
  shared graph_wrapper is untouched. Graphed output is byte-identical to eager.
- GPU memory: real-quant **6.74 GB** resident vs ~31 GB fp16 (~4.6× reduction);
  on-disk checkpoint 6.3 GB (stripped) vs 33 GB fake / ~31 GB original.

## HF upload
https://huggingface.co/Tsingyow/GLoRCQ-deepseek-v2-lite-real  (verified: 19 files incl.
cross_layer_info.pt, .stripped_real_quant, modeling_deepseek.py, index.json)

## Regression check (coordinator requirement)
- Qwen1.5-MoE real-quant graph-only bs=1: 22.5 tok/s (== baseline ~22) across all rounds.
  moe_block.py / model_builder.py have zero diff from the last Qwen-verified commit; the
  MoE-block-graph work is additive in deepseek_support.py + an opt-in eval flag only.
  Sane text: "The capital of France is ... B. Paris ...". No regression.

## Sample decode (real-quant DeepSeek-V2-Lite)
"The capital of France is Paris, and the country's official language is French."
