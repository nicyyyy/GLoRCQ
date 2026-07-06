# GLoRCQ

**G**lobal shared **Lo**w-**R**ank **C**ompensation for **Q**uantization

GLoRCQ quantizes MoE LLMs by combining TileQ's activation-scaled SVD with cross-layer
sharing of the LoRA U matrix across groups of experts from different layers. For attention
layers, plain scalar GPTQ (4-bit, no LoRA) is applied.

**Best result on Qwen1.5-MoE-A2.7B: PPL = 6.90 at ~3.47 effective bits**

**Fair-bit results (2-bit + LoRA, Extra +0.16 bits matching TileQ, 2026-07-05):**

| Model | Extra | PPL | vs TileQ | ZS avg | MMLU (5s) | Recipe |
|---|---|---|---|---|---|---|
| Qwen1.5-MoE-A2.7B | +0.1621 | **7.17** | WIN 0.39 vs TileQ_s 7.56 | 65.59% | 57.27% | `bash run_fair_qwen15.sh` |
| Qwen3-30B-A3B | +0.1647 | **9.42** | WIN 0.68 vs TileQ_v 10.1 | 63.00% | 65.52% | `bash run_fair_qwen3.sh` |
| Mixtral-8x7B-v0.1 | +0.1611 | **4.69** | WIN 0.29 vs TileQ_s 4.98 | 69.22% | 49.64% | `bash run_fair_mixtral.sh` |

## Quick Start

```bash
# 1. Setup environment (creates .venv, installs dependencies, builds CUDA kernels)
bash scripts/setup_env.sh

# 2. Run quantization (E11: 2-bit MoE experts + 4-bit attention)
bash run_e11.sh

# Or run directly:
.venv/bin/python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path /path/to/output \
    --qbit 2 --fix_rank 32 --G 128 --group_size 128 \
    --lora_bit 16 --lora_iter 8 \
    --ha_bsize 256 --id_bsize 256 \
    --attn_bits 4

# 3. Evaluate PPL
.venv/bin/python evaluate/eval_ppl.py \
    --model_path /path/to/output \
    --device cuda:0 \
    --output_json logs/result_ppl.json
```

## Environment Setup

### Prerequisites

- Python >= 3.10
- CUDA toolkit (for inference kernel compilation)
- The [TileQ](https://github.com/tilequant/tileq) repo cloned at `../tileq` (sibling directory)

### Setup

```bash
bash scripts/setup_env.sh
```

This creates a `.venv` virtualenv and installs all dependencies from `pyproject.toml`.
The CUDA inference kernels are built separately:

```bash
bash scripts/build_kernels.sh
```

## Pipeline

`run_quantize.py` runs three phases:

| Phase | Description |
|-------|-------------|
| **Phase 1** | Collect activation scales for all MoE routing experts; compute TileQ-style LoRA for attention and shared expert layers. Results cached to `{output}_phase1_cache.pt`. |
| **Phase 2** | Cross-layer group sharing: group G=128 experts of the same type across layers, stack their activation-scaled weights, compute a shared U matrix via rank-1 sketch SVD. |
| **Phase 2.5** | (Optional) 4-bit scalar GPTQ for attention layers (enabled with `--attn_bits 4`). |
| **Phase 3** | TileQ's `gptvq_fwrd_lora`: VQ-quantize the residual for each MoE routing expert. |

### Key Arguments

```
--model_path          HuggingFace model name or local path
--output_path         Output directory for quantized model
--qbit                VQ quantization bits for MoE experts (default: 2)
--fix_rank            LoRA rank per expert (default: 32)
--G                   Experts per cross-layer group (default: 128)
--group_size          VQ group size in columns (default: 128)
--lora_bit            Bits for U/V matrix storage (default: 16)
--lora_iter           Rank-1 sketch iterations (default: 8)
--attn_bits           GPTQ bits for attention layers (default: 16 = disabled)
--no_cache            Force recompute Phase 1 even if cache exists
--phase1_cache_path   Override Phase 1 cache path
```

## Project Structure

```
GLoRCQ/
├── run_quantize.py              # Main entry point (Phases 1–3)
├── run_e11.sh                   # Example run script for Qwen1.5-MoE-A2.7B
├── joint_optim.py               # GPTQJoint class (used by Phase 2.5)
│
├── quantizer/                   # Quantization primitives
│   ├── quantizer.py             #   WeightQuantizer (MSE clipping, scale/zero)
│   ├── quantizer_v.py           #   Vector quantizer variant
│   ├── scalar_quant_utils_lora.py
│   └── vector_quant_utils_lora.py
│
├── sketch/
│   └── r1_sketch.py             # Rank-1 sketch SVD (used in Phase 2)
│
├── inference/                   # Real-quantized model inference
│   ├── model_builder.py         #   load_glorcq_model() — load packed weights
│   ├── quantized_linear.py      #   GLoRCQLinear layer (dequant + matmul + LoRA)
│   ├── graph_wrapper.py         #   CUDA Graph acceleration for decode
│   ├── eval_speed.py            #   Speed benchmark script
│   └── kernels/                 #   Fused CUDA kernels
│
├── evaluate/                    # Evaluation scripts
│   ├── eval_ppl.py              #   WikiText-2 perplexity
│   ├── eval_speed.py            #   Decode throughput
│   └── eval_zeroshot.py         #   Zero-shot benchmarks (lm-eval-harness)
│
├── utils/                       # Shared utilities
│   ├── model_loader.py          #   load_model_and_tokenizer()
│   ├── moe_utils.py             #   MoE expert detection & layer info
│   ├── get_calib_data.py        #   WikiText-2 / C4 calibration data
│   └── ...
│
├── scripts/                     # Setup and build scripts
│   ├── setup_env.sh
│   └── build_kernels.sh
│
└── exp/                         # Paper evaluation scripts
    ├── paper_eval_qwen15moe.sh
    └── ...
```

## Results

| Model | Method | Bits | PPL (WikiText-2) |
|-------|--------|------|-----------------|
| Qwen1.5-MoE-A2.7B | FP16 | 16.0 | ~4.41 |
| Qwen1.5-MoE-A2.7B | TileQ-1D (paper) | ~2.3 | 7.49 |
| Qwen1.5-MoE-A2.7B | GLoRCQ E11 (ours) | 3.47 | **6.90** |

E11 configuration: 2-bit VQ for MoE routing experts (rank=32, G=128), 4-bit GPTQ for
attention, FP16 for shared expert and embeddings.
