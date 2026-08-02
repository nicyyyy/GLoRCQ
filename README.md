# CLASP: Cross-Layer Active-Subspace Pooling for Quantized Mixture-of-Experts

Anonymous code release for AAAI-2027 submission (double-blind).

CLASP quantizes MoE expert weights to 2 bits with vector quantization and
recovers accuracy with low-rank corrections whose *active-subspace bases are
pooled across layers*: experts from different layers are clustered on the
Grassmannian of their activation-weighted principal subspaces, and each
cluster shares one pooled basis. The shared structure pays twice — it shrinks
storage (one basis per cluster instead of one per expert) and speeds up
decoding (corrections for all activated experts batch onto the shared basis).

*(CLASP is the paper name; the implementation package name in this code is
`glorcq` — internal identifiers such as `GLoRCQLinear` and the `GLORCQ_*`
environment variables refer to the same method.)*

## Requirements / Install

- Python 3.12, a CUDA GPU, and (for kernel builds) a CUDA 12.x `nvcc`
  matching the torch build (torch 2.6.0+cu124).
- Quantization is one-shot on a **single GPU** and calibrates with
  **128 WikiText-2 sequences** (downloaded automatically via `datasets`).
- GPU memory: quantization loads the fp16 model, so Qwen1.5-MoE fits on
  80 GB; Qwen3-30B / Mixtral-8x7B need a larger single GPU (e.g. H200).

```bash
bash install.sh            # venv + torch 2.6.0 cu124 + deps + CUDA kernels
source .venv/bin/activate
```

The fused CUDA kernels (`inference/kernels/`) are only required for the
decode-speed benchmarks; accuracy experiments run without them.

## Quick start

Each script is self-contained: quantize -> WikiText-2 PPL -> 5-task zero-shot
(ARC-C, ARC-E, PIQA, WinoGrande, HellaSwag), with the canonical best
parameters as defaults (all overridable via environment variables). Outputs
go to `./outputs/<model>/`. A commented decode-speed invocation is at the
bottom of each script.

```bash
bash scripts/run_qwen15.sh     # Qwen/Qwen1.5-MoE-A2.7B
bash scripts/run_qwen3.sh      # Qwen/Qwen3-30B-A3B
bash scripts/run_mixtral.sh    # mistralai/Mixtral-8x7B-v0.1
bash scripts/run_deepseek.sh   # deepseek-ai/deepseek-moe-16b-base
```

## Expected results (WikiText-2 PPL @ ~2.16 bits)

| Model               | Bits (attn+FFN) | PPL  |
|---------------------|-----------------|------|
| Qwen1.5-MoE-A2.7B   | ~2.15           | 7.14 |
| Mixtral-8x7B-v0.1   | ~2.17           | 4.50 |
| Qwen3-30B-A3B       | ~2.16           | 8.97 |
| DeepSeek-MoE-16B    | ~2.21           | 6.60 |

Runs are seeded and deterministic: re-running a script reproduces the PPL
bit-identically on the same GPU/software stack.

## Inference-speed switches

The optimized decode paths are opt-in environment variables (all default off;
with them off, outputs are byte-identical to the reference decode path):

| Env var                     | Effect                                                          |
|-----------------------------|-----------------------------------------------------------------|
| `GLORCQ_MIXTRAL_GRAPH=1`    | Mixtral: capture the top-k gather decode step in a CUDA graph   |
| `GLORCQ_MIXTRAL_IDXKERNEL=1`| Mixtral: expert-indexed VQ4 grouped-GEMV kernel (gather in-kernel) |
| `GLORCQ_PREFILL_DEQUANT=1`  | Dequantize experts once for prefill (prefill uses cuBLAS GEMMs) |
| `GLORCQ_GPTQ_ILP=1`         | ILP-optimized GPTQ attention dequant-matmul kernel              |
| `GLORCQ_DEEPSEEK_GATHER=1`  | DeepSeek: compute only the top-k routed experts inside the graph |
| `GLORCQ_DEEPSEEK_IDXKERNEL=1`| DeepSeek: expert-indexed VQ4 grouped-GEMV kernel               |

## Repository layout

```
run_quantize.py          # entry point: Phase 1 collect -> cluster -> pool -> VQ -> export
cross_layer_share.py     # Grassmannian clustering of activation-weighted subspaces
joint_optim.py           # joint GPTQ / low-rank alternating optimization (attention)
get_scale_quant.py       # activation-scaled SVD / low-rank helpers
hadamard_rotation.py     # randomized Hadamard transform utilities
quantizer/               # VQ (GPTVQ-style) + scalar GPTQ quantizers with LoRA hooks
sketch/                  # rank-1 sketch for fast low-rank fitting
utils/                   # calibration data, model loading, MoE/layer naming helpers
inference/               # real-quant runtime: model builder, MoE block, CUDA-graph wrapper
inference/kernels/       # fused CUDA kernels (VQ4 / GPTQ / TurboQuant dequant-matmul)
evaluate/                # eval_ppl.py, eval_zeroshot.py (lm-eval), eval_speed.py, bench_deepseek.py
scripts/                 # per-model reproduction scripts + kernel build
```

## License

MIT (copyright "Anonymous" — placeholder until camera-ready).
