# GLoRCQ Experiment Results — Qwen1.5-MoE-A2.7B

Model: `Qwen/Qwen1.5-MoE-A2.7B`  
Date: 2026-06-22  
Hardware: NVIDIA A100 80GB (GPU4, nwonga100), Job 15746

---

## 1. PPL Results (WikiText-2)

| Config | PPL | bits/param | Notes |
|--------|-----|-----------|-------|
| FP16 baseline | 6.79 | 16.00 | From TileQ Table 1 |
| GPTQ+TurboQuant only (no LoRA) | 27.95 | ~2.01 | `logs/gptq_only_ppl.log` |
| GLoRCQ no-sharing (G=1440) | 14.94 | 2.318 | `logs/no_sharing_G1440_ppl.log` |
| **GLoRCQ SOTA** (G=128, n_iter=5) | **8.48** | **2.4787** | `logs/loftq5_niter5_sv8_ppl.log` |
| TileQ_v 2-bit (reference) | 7.35 | ~2.16 | TileQ Table 1 |

### SOTA Bits Breakdown (G=128, rank=32, uv_bits=8, sv_bits=8)

```
2-bit weights:                           2.0000 bits/param
Hybrid quant scale (fp16):               0.0126 bits/param
Shared U (MoE=int8/Attn=int8, amortized): 0.0875 bits/param
Per-expert SV (MoE=int8/Attn=int8):     0.3786 bits/param
Total average:                           2.4787 bits/param
```

### No-Sharing Bits Breakdown (G=1440, each expert independent)

```
2-bit weights:                           2.0000 bits/param
Hybrid quant scale (fp16):               0.0126 bits/param
Shared U (amortized, G=1440):            0.1619 bits/param
Per-expert SV:                           0.1435 bits/param
Total average:                           2.3180 bits/param
```

---

## 2. Zero-Shot Downstream Tasks

Evaluation on 5 tasks (0-shot). Metric: `acc_norm` for ARC/HellaSwag/PIQA, `acc` for WinoGrande.

| Task | GLoRCQ SOTA (2.48 bits) | FP16 Baseline |
|------|------------------------|---------------|
| ARC-Challenge (acc_norm) | 40.19% | 44.54% |
| ARC-Easy (acc_norm) | 65.07% | 69.19% |
| WinoGrande (acc) | 67.32% | 69.06% |
| HellaSwag (acc_norm) | 69.78% | 77.28% |
| PIQA (acc_norm) | 77.04% | 80.52% |
| **Average** | **63.88%** | **68.12%** |

Sources: `logs/zeroshot_sota.log`, `logs/zeroshot_fp16_bs1.log`

---

## 3. Inference Speed

Model: GLoRCQ real_quant (rank_attn=512, rank_down=128, n_iter=1, G_moe=128, G_attn=24)  
Config: batch_size=1, prompt_len=128, generate_len=128, warmup=2, runs=5

| Mode | Throughput | Speedup vs FP16 |
|------|-----------|-----------------|
| FP16 (HF baseline) | 4.6 ± 0.0 tok/s | 1.00× |
| GLoRCQ Standard | 11.3 ± 0.0 tok/s | 2.46× |
| GLoRCQ CUDA Graph | 28.3 ± 0.0 tok/s | **6.17×** |
| Graph vs Standard | — | 2.51× |

Peak GPU Memory: 40,908 MB (GLoRCQ); FP16 baseline: ~40,800 MB  
Source: `logs/e2e_speed_sota_r128.log`

---

## 4. Ablation Summary

| Config | PPL | bits/param | Key Change |
|--------|-----|-----------|-----------|
| FP16 | 6.79 | 16.00 | — |
| GPTQ+TQ, no LoRA | 27.95 | ~2.01 | No LoRA compensation |
| GLoRCQ, G=1440 (no sharing) | 14.94 | 2.318 | Each expert has own U |
| GLoRCQ, G=128, n_lora_iter=2 | ~8.49 | ~2.48 | 2 LoftQ iterations |
| **GLoRCQ, G=128, n_lora_iter=5** | **8.48** | **2.4787** | **5 LoftQ iterations** |

Key finding: LoRA compensation is essential — no LoRA → PPL 27.95 (vs 8.48 with LoRA).  
Grassmannian clustering helps: no-sharing → PPL 14.94 (vs 8.48 with G=128 sharing).

---

## 5. Run Configuration (SOTA)

```bash
python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path /home/qyyang/resource_dir/GLoRCQ_out/loftq5_niter5_sv8 \
    --qbit 2 --groupsize 128 --nsamples 128 \
    --rank 32 --n_iter 5 \
    --G_moe 128 --G_attn 24 \
    --uv_bits 8 --sv_bits 8 \
    --n_lora_iter 5 \
    --use_turboquant --hessian_svd --search_act_alpha
```

---

## 6. Chain2 Run Log Summary

Full chain: `logs/chain2.log`  
Started: 2026-06-22 00:58 HKT | Completed: 2026-06-22 09:41 HKT (≈8.7 hours)

| Step | Description | Duration | Result |
|------|-------------|----------|--------|
| A | GPTQ-only quantization | 29 min | → `GLoRCQ_out/gptq_only_baseline` |
| B | GPTQ-only PPL eval | 4 min | **PPL = 27.95** |
| C | Speed test real_quant (rank_attn=512, rank_down=128) | 92 min | → `GLoRCQ_out/speed_test_r128` |
| D | Speed evaluation | 6 min | **11.3 / 28.3 / 4.6 tok/s** |
| E | No-sharing ablation (G_moe=1440, n_lora_iter=2) | 36 min | → `GLoRCQ_out/no_sharing_G1440` |
| F | No-sharing PPL eval | 4 min | **PPL = 14.94** |
| G | FP16 zero-shot eval (5 tasks, batch_size=1) | 352 min | **Avg = 68.12%** |
