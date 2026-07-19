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

## Zero-shot (lm-eval-harness, num_fewshot 0, batch 16; acc_norm where applicable)
| task           | metric   | acc    |
|----------------|----------|--------|
| ARC-Challenge  | acc_norm | 0.4369 |
| ARC-Easy       | acc_norm | 0.7302 |
| WinoGrande     | acc      | 0.6882 |
| HellaSwag      | acc_norm | 0.7145 |
| PIQA           | acc_norm | 0.7709 |
| **Average (5)**|          | **0.6682 (66.82%)** |

Note: lm-eval loglikelihood accuracy is batch-invariant; batch 16 used to save wall time
(identical scores to batch 1). Same eval_zeroshot.py/eval_ppl.py used for the other 3 models
(no --add_bos flag exists in the repo harness → matches Qwen/Mixtral/Qwen3 methodology).

## bs=1 decode speed (prompt 128, gen 128, GPU A100)
- real-quant eager : **6.68 tok/s** (0.40× fp16)
- fp16 eager       : **16.70 tok/s**
- CUDA-graph path: **does NOT engage** — the graph wrapper uses StaticCache +
  cache_position, which DeepSeek-V2's MLA remote modeling (legacy DynamicCache API)
  does not support (analogous to Mixtral's graph limitation). So the bs=1 win the
  graph path gives Qwen (~4-5×) is unavailable here; real-quant is a MEMORY win.
- GPU memory: real-quant **6.74 GB** resident vs ~31 GB fp16 (~4.6× reduction);
  on-disk checkpoint 6.3 GB (stripped) vs 33 GB fake / ~31 GB original.

## HF upload
https://huggingface.co/Tsingyow/GLoRCQ-deepseek-v2-lite-real  (verified: 19 files incl.
cross_layer_info.pt, .stripped_real_quant, modeling_deepseek.py, index.json)

## Regression check (coordinator requirement)
- Qwen1.5-MoE real-quant graph-only bs=1: 22.5 → 23.3 tok/s (post-refactor, == baseline ~22).
  Sane text: "The capital of France is ... B. Paris ...". No regression from shared-line edits.

## Sample decode (real-quant DeepSeek-V2-Lite)
"The capital of France is Paris, and the country's official language is French."
