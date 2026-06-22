# GLoRCQ 论文实验设计方案

更新时间：2026-06-22

---

## 1. 测试模型

| 模型 | 专家架构 | 总参数 | 激活参数 | FP16 大小 | 实验状态 |
|------|---------|--------|---------|-----------|---------|
| Qwen1.5-MoE-A2.7B | 60+4 exp/layer, top-4, 24 layers | ~14.3B | ~2.7B | ~28 GB | ✅ 完成（详见 `results_qwen_moe.md`）|
| Mixtral-8x7B-v0.1 | 8 exp/layer, top-2, 32 layers | ~47B | ~13B | ~47 GB | 🔄 运行中（`run_mixtral_chain.sh`）|
| **Qwen3-30B-A3B** | 128 exp/layer, top-8, dim=[512,2048] | ~30B | ~3B | ~60 GB | ⏳ 需要大内存机器（>80 GB）|
| DeepSeek-MoE-16B | 64+2 exp/layer, top-6, 28 layers | ~16B | ~2.8B | ~32 GB | — 可选 |

---

## 2. Evaluation 协议

### 2.1 Benchmark 列表

| 任务 | 指标 | Few-shot | 工具 |
|------|------|---------|------|
| WikiText-2 PPL | perplexity↓ | 0-shot | `evaluate/eval_ppl.py` |
| ARC-Challenge (AC) | acc_norm↑ | 0-shot | lm-eval |
| ARC-Easy (AE) | acc_norm↑ | 0-shot | lm-eval |
| PIQA (PQ) | acc_norm↑ | 0-shot | lm-eval |
| WinoGrande (WI) | acc↑ | 0-shot | lm-eval |
| HellaSwag (HS) | acc_norm↑ | 0-shot | lm-eval |
| **MMLU (MU)** | **acc↑** | **5-shot** | **lm-eval ← 待补跑** |

**注意**：
- TileQ 使用相同的 6 个下游任务（AC/AE/PQ/WI/MU/HS）。我们当前缺少 MMLU。
- MiLo 使用不同任务集（HellaSwag/Lambada/PIQA/MMLU/TriQA），不可直接对比。
- HellaSwag 指标差异：TileQ 可能用 `acc`，我们用 `acc_norm`（约高 5-10%）；论文中需注明。

### 2.2 MMLU 补跑命令

```bash
# Qwen1.5-MoE-A2.7B MMLU (待跑)
CUDA_VISIBLE_DEVICES=4 python evaluate/eval_zeroshot.py \
    --model_path /home/qyyang/resource_dir/GLoRCQ_out/loftq5_niter5_sv8 \
    --device cuda:0 \
    --tasks arc_challenge,arc_easy,winogrande,hellaswag,piqa,mmlu \
    --num_fewshot 0,0,0,0,0,5 \
    --output_json logs/zeroshot_sota_mmlu.json
```

### 2.3 推理速度测试协议

- batch_size=1, prompt_len=128, generate_len=128
- warmup=2, runs=5，取平均值
- 硬件：NVIDIA A100 80GB（nwonga100）
- 模式：FP16（HuggingFace naive）/ GLoRCQ Standard / GLoRCQ CUDA Graph

---

## 3. 对比基线（论文引用数据）

### 3.1 来自 TileQ Table 1（2-bit 量化，ICML 2026）

> 引用：TileQ: Efficient Low-Rank Quantization of Mixture-of-Experts with 2D Tiling  
> Extra Bits (X_v) = 分配给 group-wise 量化 scale 和 LoRA 组件的平均额外位宽  
> 评测：WikiText-2 PPL + 6个下游任务（acc/%），AC=ARC-Challenge, AE=ARC-Easy, PQ=PIQA, WI=WinoGrande, MU=MMLU(5-shot), HS=HellaSwag

#### Qwen1.5-MoE-A2.7B

| 方法 | Extra↓ | PPL↓ | AC↑ | AE↑ | PQ↑ | WI↑ | MU↑ | HS↑ | Avg(6)↑ |
|------|--------|------|-----|-----|-----|-----|-----|-----|---------|
| FP16 | — | 6.79 | 41.6 | 72.9 | 79.6 | 69.1 | 61.2 | 59.3 | 63.95 |
| GPTQ | 0.13 | 7.98 | 29.4 | 42.7 | 50.2 | 53.2 | 25.3 | 37.2 | 39.67 |
| GPTVQ | 0.13 | 8.12 | 34.1 | 68.4 | 71.4 | 62.5 | 53.9 | 49.8 | 56.68 |
| MOEQ | 0.00 | 8.21 | 32.6 | 63.6 | 76.2 | 63.8 | 50.1 | 45.3 | 55.27 |
| LoPRo | 0.43 | 7.52 | 39.9 | 72.7 | 77.6 | 68.2 | 56.8 | 53.4 | 61.43 |
| TileQ_s | 0.16 | 7.56 | 39.6 | 72.5 | 77.8 | 68.9 | 57.5 | 54.1 | 61.73 |
| **TileQ_v** | **0.16** | **7.35** | **40.3** | **73.4** | **78.5** | **68.6** | **58.8** | **55.3** | **62.48** |
| **GLoRCQ (ours)** | **0.48** | 8.48 | 40.19 | 65.07 | 77.04 | 67.32 | *TBD* | 69.78° | 63.88† |

°HellaSwag: acc_norm（TileQ 可能用 acc，差异约 +5-10%）  
†Avg 仅含 5 tasks（不含 MMLU，待补跑）

#### Mixtral-8x7B-v0.1

| 方法 | Extra↓ | PPL↓ | AC↑ | AE↑ | PQ↑ | WI↑ | MU↑ | HS↑ | Avg(6)↑ |
|------|--------|------|-----|-----|-----|-----|-----|-----|---------|
| FP16 | — | 3.87 | 61.9 | 87.3 | 83.7 | 77.1 | 71.2 | 67.3 | 74.75 |
| GPTQ | 0.13 | 15.3 | 26.5 | 35.6 | 53.0 | 49.3 | 24.3 | 28.2 | 36.15 |
| GPTVQ | 0.13 | 5.28 | 42.0 | 71.6 | 75.9 | 66.5 | 58.9 | 55.4 | 61.72 |
| MOEQ | 0.00 | 13.4 | 38.9 | 49.8 | 60.3 | 49.9 | 44.2 | 40.5 | 47.27 |
| LoPRo | 0.21 | 5.01 | 55.3 | 82.5 | 80.6 | 74.9 | 63.7 | 60.3 | 69.55 |
| TileQ_s | 0.16 | 4.98 | 52.5 | 82.8 | 80.9 | 75.1 | 63.8 | 61.4 | 69.42 |
| **TileQ_v** | **0.16** | **4.78** | **56.3** | **83.8** | **80.5** | **74.8** | **64.4** | **61.4** | **70.20** |
| **GLoRCQ (ours)** | TBD | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | — | *TBD* | *TBD* |

#### Qwen3-30B-A3B

> TileQ Table 1 数据（2-bit，基线参考）

| 方法 | Extra↓ | PPL↓ | AC↑ | AE↑ | PQ↑ | WI↑ | MU↑ | HS↑ | Avg(6)↑ |
|------|--------|------|-----|-----|-----|-----|-----|-----|---------|
| FP16 | — | 8.07 | — | — | — | — | — | — | 68.8 |
| **TileQ_v** | **0.16** | **10.1** | — | — | — | — | — | — | **50.8** |
| **GLoRCQ (ours)** | TBD | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | — | *TBD* | *TBD* |

> 注：Qwen3-30B-A3B 需要 >80 GB GPU 进行量化和评测，当前实验待完成。
> TileQ_v PPL=10.1 vs FP16=8.07，劣化幅度大于 Qwen1.5-MoE（8.48 vs 6.79），说明更大模型 2-bit 压缩难度更高。

---

### 3.2 来自 MiLo Table 3（**3-bit** W3A16，MLSys 2025）

> 引用：MiLo: Efficient Quantized MoE Inference with Mixture of Low-Rank Compensators  
> 注：MiLo 使用 3-bit（W3A16），评测集与 TileQ 不同，**不可直接比较**。
> 作为 3-bit 性能上限参考，说明我们 2-bit 方法面对的精度空间。  
> 评测任务：HellaSwag/Lambada/PIQA（0-shot）、MMLU（5-shot）、TriQA（5-shot）

#### Mixtral-8x7B（3-bit W3A16）

| 方法 | Memory | PPL↓ | HellaSwag↑ | Lambada↑ | PIQA↑ | Avg(3)↑ | MMLU(5s)↑ | TriQA(5s)↑ |
|------|--------|------|-----------|---------|-------|--------|----------|----------|
| RTN | 20.5 GB | 4.8133 | 78.40 | 71.18 | 79.10 | 76.23 | 59.36 | 69.41 |
| GPTQ | 18.4 GB | 4.7304 | 77.70 | 74.36 | 79.54 | 77.20 | 63.61 | 68.53 |
| HQQ | 20.5 GB | 4.6119 | 77.88 | 69.74 | 79.16 | 75.59 | 60.93 | 70.66 |
| MiLo-s1 | 20.8 GB | 4.0335 | **82.23** | 75.12 | **81.33** | **79.56** | 67.07 | 75.82 |
| **MiLo-s2** | **21.0 GB** | **3.9076** | 81.60 | **75.72** | 81.12 | 79.48 | **67.69** | **76.42** |
| FP16 参考 | 90 GB | 3.87 | — | — | — | — | 71.2 | — |

MiLo 注：
- MiLo-s1 = Dense-512 + Kurtosis-16 rank 策略
- MiLo-s2 = Dense-1024 + Kurtosis-32 rank 策略
- MiLo 3-bit PPL=3.91，接近 FP16 PPL=3.87（仅 -1% 相对差距）
- MiLo 使用定制 W3A16 内核，比 MARLIN 快 1.2×（batch=1）

---

## 4. 推理速度对比

### 4.1 Qwen1.5-MoE-A2.7B（A100 80GB，batch=1）

> 来源：`logs/e2e_speed_sota_r128.log`（GLoRCQ），TileQ Figure 4（TileQ，A800），MiLo Table 7（MiLo，A100 40GB）

| 方法 | tok/s | Speedup vs FP16 | GPU Memory | 硬件 |
|------|-------|-----------------|-----------|------|
| FP16（HuggingFace） | 4.6 | 1.00× | ~28 GB | A100 80G |
| **vLLM FP16（Docker）** | *TBD* | *TBD* | ~28 GB | A100 80G |
| **GLoRCQ Standard** | **11.3** | **2.46×** | **~10 GB** | A100 80G |
| **GLoRCQ CUDA Graph** | **28.3** | **6.17×** | **~10 GB** | A100 80G |
| TileQ（decode stage） | — | ~1.2× | — | A800（不同硬件）|
| MiLo-s2（Mixtral，3-bit，batch=1） | — | 1.2× vs MARLIN | ~21 GB | A100 40G |

**vLLM 对比说明**：
- 运行：`bash exp/bench_vllm.sh Qwen/Qwen1.5-MoE-A2.7B <gpu_id>`（脚本启动 Docker vLLM server，自动测量）
- 测量协议：与 GLoRCQ eval_speed.py 相同（batch=1, prompt=128, gen=128, warmup=2, runs=5）
- 对比方案：强调 GLoRCQ **内存 footprint** 优势（28 GB → 10 GB），不要求 raw throughput 超越 vLLM

**关键指标**：
- GLoRCQ 内存压缩：**28 GB → 10 GB（约 2.8× 压缩）**
- 这使 Qwen1.5-MoE-A2.7B 从"需要 80GB GPU"变为"可在 24GB GPU（如 RTX 4090）运行"
- Standard 模式 2.46× 加速来自自定义 GPTQ + TurboQuant CUDA 内核
- CUDA Graph 模式额外消除动态 dispatch 开销，进一步提升 2.51×

### 4.2 Mixtral-8x7B（待完成）

Mixtral 当前不支持 CUDA Graph（需扩展 `inference/model_builder.py`），暂无推理速度数据。

---

## 5. Bits 开销分析

| 方法 | 基础权重 | LoRA/Scale 额外开销 | 总 bits/param |
|------|---------|-------------------|-------------|
| GPTQ | 2.00 | 0.13（per-channel scale） | ~2.13 |
| MOEQ | 2.00 | 0.00 | ~2.00 |
| TileQ_v | 2.00 | 0.16（2D-Tile U+Σ+V） | ~2.16 |
| LoPRo | 2.00 | 0.43（per-layer rotation） | ~2.43 |
| **GLoRCQ SOTA** | 2.00 | **0.4787**（shared U + per-expert SV + scale）| **2.4787** |

**GLoRCQ 的 0.4787 bits 分解（Qwen1.5-MoE-A2.7B）**：
```
Hybrid quant scale (fp16):               0.0126 bits/param
Shared U (MoE=int8/Attn=int8, amortized): 0.0875 bits/param
Per-expert SV (MoE=int8/Attn=int8):     0.3786 bits/param
Total overhead:                          0.4787 bits/param
Total average:                           2.4787 bits/param
```

TileQ_v 用更少 bits（0.16 vs 0.48）获得更低 PPL（7.35 vs 8.48）。
这表明 TileQ 的 2D-Tiling 共享在同等 LoRA 开销下效率更高。
GLoRCQ 的优势在于**推理系统**（2.46× / 6.17× 加速）和**架构兼容性**。

---

## 6. GLoRCQ 当前结果汇总

### 6.1 Qwen1.5-MoE-A2.7B

| 指标 | GLoRCQ SOTA | TileQ_v（参考） | GLoRCQ vs TileQ |
|------|------------|----------------|----------------|
| PPL (WikiText-2) | **8.48** | 7.35 | +1.13 (↓) |
| ARC-Challenge | 40.19% | 40.3% | -0.11% |
| ARC-Easy | 65.07% | 73.4% | -8.33% |
| PIQA | 77.04% | 78.5% | -1.46% |
| WinoGrande | 67.32% | 68.6% | -1.28% |
| MMLU (5-shot) | *TBD* | 58.8% | — |
| HellaSwag | 69.78% (acc_norm) | 55.3% (acc?) | — |
| Avg (5 tasks, no MMLU) | **63.88%** | 62.48% (6 tasks) | 需补 MMLU |
| Total bits | 2.4787 | ~2.16 | +0.32 bits |
| Decode tok/s | 11.3 / 28.3 | — | — |

### 6.2 Mixtral-8x7B（运行中）

| 指标 | GLoRCQ | TileQ_v（参考） | 状态 |
|------|--------|----------------|------|
| PPL (WikiText-2) | *TBD* | 4.78 | 🔄 Step M2 |
| Avg 6-task | *TBD* | 70.20% | 🔄 Step M3 |
| Total bits | ~2.47 | ~2.16 | 估计 |

### 6.3 Qwen3-30B-A3B（待运行）

| 指标 | GLoRCQ | TileQ_v（参考） | 状态 |
|------|--------|----------------|------|
| PPL (WikiText-2) | *TBD* | 10.1 | ⏳ 需要 >80 GB GPU |
| Avg 6-task | *TBD* | 50.8% | ⏳ 需要 >80 GB GPU |
| Total bits | ~2.47 | ~2.16 | 估计 |

---

## 7. 待完成实验与 Gaps

| 优先级 | Gap | 影响 | 行动 |
|--------|-----|------|------|
| 🔴 高 | MMLU 缺失（Qwen1.5-MoE + Mixtral） | 无法与 TileQ 6-task 对比 | 补跑 MMLU 5-shot（约 2-4h/模型）|
| 🔴 高 | Mixtral 实验结果 pending | 第二模型数据 | 等待 `run_mixtral_chain.sh`（ETA ~10-20h）|
| 🔴 高 | vLLM 速度对比 pending | 缺少强推理基线 | 运行 `exp/bench_vllm.sh`，补充速度表 |
| 🟡 中 | Qwen3-30B-A3B 需要大内存机器 | 第三个模型数据 | 找 >80 GB GPU 机器，上传 HF 后远程运行 |
| 🟡 中 | HellaSwag acc vs acc_norm 歧义 | 直接数字比较偏差 | 统一协议或在论文中注明 |
| 🟡 中 | Mixtral 无推理速度数据 | 速度表只有一个模型 | 扩展 `_replace_moe_blocks()` 支持 Mixtral |
| 🟢 低 | HuggingFace 模型上传 | 远程评测依赖 | 待用户提供上传脚本后执行 |
| 🟢 低 | DeepSeek-MoE-16B 实验 | 第三个模型（TileQ 有基线）| 视时间决定 |
| 🟢 低 | Ablation：不同 G_moe 值 | 聚类效果的 sensitivity | G=32/64/128/256 各跑一组 |

### 7.1 MMLU 补跑计划

```bash
# Step M6（在 run_mixtral_chain.sh 后附加）
# Qwen1.5-MoE MMLU 补跑
CUDA_VISIBLE_DEVICES=4 python evaluate/eval_zeroshot.py \
    --model_path /home/qyyang/resource_dir/GLoRCQ_out/loftq5_niter5_sv8 \
    --tasks arc_challenge,arc_easy,winogrande,hellaswag,piqa,mmlu \
    --num_fewshot 0,0,0,0,0,5 \
    --output_json logs/zeroshot_sota_6task.json

# FP16 Qwen MMLU 补跑
CUDA_VISIBLE_DEVICES=4 python evaluate/eval_zeroshot.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --tasks arc_challenge,arc_easy,winogrande,hellaswag,piqa,mmlu \
    --num_fewshot 0,0,0,0,0,5 \
    --output_json logs/zeroshot_fp16_6task.json
```

---

## 8. 方法对比总结

| 维度 | GLoRCQ | TileQ | MiLo |
|------|--------|-------|------|
| 量化位宽 | **2-bit** | 2-bit / 3-bit | **3-bit** |
| 共享机制 | Grassmannian 聚类 + 共享 U | 2D-Tiling（行+列聚类，同时共享 U 和 V）| 1D 共享（自适应 rank 策略）|
| 额外 bits | 0.48 | **0.16** | ~0.3-0.5（rank 自适应）|
| 压缩质量 | PPL 8.48（Qwen） | **PPL 7.35**（Qwen）| PPL 3.91（Mixtral，3-bit）|
| 推理速度 | **2.46× / 6.17×** | ~1.2×（prefill/decode）| 1.2× vs MARLIN（batch=1）|
| 内存压缩 | **3×**（28→10 GB）| 减少 90% LoRA 额外内存 | 仅额外 +1.4% 内存 |
| Calibration-free | ✅（TurboQuant 不需要）| ✅ | ✅ |
| 端到端推理系统 | ✅ CUDA Graph | ✅ LoTileMoE fused kernel | ✅ W3A16 custom kernel |
| 硬件目标 | 单 24GB GPU 部署 | 低 latency prefill+decode | batch > 1 推理加速 |
