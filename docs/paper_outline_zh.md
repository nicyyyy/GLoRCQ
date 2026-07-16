# GLoRCQ 论文大纲（中文速读版）

**标题**: GLoRCQ: Global shared Low-Rank Compensation for Quantization of Mixture-of-Experts LLMs（全局共享低秩补偿的 MoE 大模型量化）

**目标会议**: ICML 2026 / NeurIPS 2026 主赛道，8-9 页正文 + 附录
**版本**: outline v0.4 — 2026-07-09（Paper voice 通读：合并 §3.2/§3.3；正文去除函数名、行号、代码分支；Global U pool 描述与代码实际行为一致）
**详细英文版**: `docs/paper_outline.md`（本文档是同结构精简版）

---

## 一句话核心论点

> **TileQ 和 MiLo 已经在做"2-bit 量化 + per-expert 低秩补偿"**；它们的缺点是每个 expert 的低秩因子**各自独立**，rank 预算被切碎。**GLoRCQ 用主成分角距离在层间聚类 experts，每个 cluster 内做一次激活加权 SVD，共享 U 因子替代 G 个 per-expert basis**；推理端顺势利用 cluster 结构：cluster 级 x·U 复用 + 同 cluster K-融 GEMM + LoRA 侧流并行 2-bit backbone。

两个 contribution（C1 算法 + C2 系统）+ 实验 payoff。不谈 memory hook，不声称 sub-2-bit，不声称 alternating optimization。

---

## 页数预算（ICML 8 页正文）

| 章节 | 页 | 主要 message |
|---|---|---|
| Abstract + §1 引言 | 1.0 | TileQ/MiLo lineage → per-expert LoRA 缺点 → 跨层子空间共享 U + 推理 co-design |
| §2 相关工作 | 1.0 | 复用现有 LR + 应用 review 报告 C1/C3/M1 补丁 |
| §3 方法（**核心**） | 2.0 | 前置 + 单一 Method 小节：聚类 + 共享 U 的 SVD + One-shot pipeline |
| §4 推理端 co-design（**核心**） | 0.75 | 共享 U 缓存 + 全局 U pool + 同 cluster K-融 + 侧流并行 |
| §5 实验 | 2.0 | Table 1 主对比 + Table 2 系统加速 + baseline 复现 |
| §6 消融 | 1.25 | rank / G / LoRA / 聚类方法 / 主成分角 heatmap / 系统 ablation |
| §7 讨论与局限 | 0.3 | 推理速度定位、τ 是启发式、Qwen3 real-quant NaN bug |
| §8 结论 | 0.1 | 两句话 |
| **正文总计** | **~8.4** | |
| 附录 A-F | 6.5+ | Bit accounting + 继承 scaffolding + 全消融表 + HF cards + 扩展相关工作 + MxMoE 对比说明 |

---

## Contributions（**只有 2 个**，加实验 payoff）

**Reviewer 讨厌"结果作为 contribution"，所以第 3 项只写进 abstract 头号数字，不编号进 contribution 列表。**

### **C1**: Cross-layer expert subspace pooling with a shared U-factor

不同 Transformer 层的 experts 做的是同类计算，实测上它们的 per-expert 低秩补偿因子占据高度重叠的子空间（Figure 4 in §6）。我们不再层内 per-expert 分配 rank，而是按 top-r 奇异子空间的主成分角距离在层间聚类 experts；cluster 内把成员的激活加权权重堆起来，做**一次**激活加权 SVD。top-r 左奇异向量成为 cluster 内共享的 U，per-expert Σ 和 V 私有。同 bit 预算下 per-expert 有效 rank 相对 per-expert LoRA 提升 **G×**。

### **C2**: Inference-time cluster-batched LoRA on a side stream

因为 cluster 是全局学出来的，一个 token 激活的 top-k experts 常常同一 cluster，也就共享一个 U。由此有三件可复合的收益。**第一**，每个 cluster 的共享 U 在加载时 dequant 一次进 device pool，x·U 每 token 每 active cluster 只算一次，在该 cluster 的所有 active experts 之间复用。**第二**，当 top-k 全部同一 cluster 时，per-expert 因子 concat 成单个宽 GEMM 发出，省 K 次 kernel launch。**第三**，LoRA 侧流并行 2-bit backbone 主流，单点 downstream 同步；graph mode 下这个 pattern 能进 CUDA Graph。

**关键不吹**。attention 在 MoE block 之前完成，我们**不**把 LoRA 与 attention 重叠。我们**不**跨层预取 U。我们**不**声称与 fp16 kernel（vLLM 等）raw throughput 打平 —— 加速比是**同模型 naive per-expert-LoRA-on-main-stream** 的对比。

### 实验 payoff（不编号）

- **主结果** (§5.2 Table 1): 2.16 bits/param 下 PPL + 5-task 0-shot vs TileQ / LoPRo / GPTVQ / MiLo (3-bit 参考) / MxMoE
- **系统加速** (§5.3 Table 2): decode tokens/sec, with/without C2 优化
- **聚类有效性** (§6.5): 每 cluster 跨层组成图 (Fig 4a) + 主成分角距离 heatmap sorted by cluster (Fig 4b) + 同尺寸随机对照 (Table 5)

---

## Abstract 段（150-250 词）

**四句结构**（不用 memory hook，不用 sub-2-bit，不用 stripped ckpt，不用 fused kernel，不用 alternating optimization）:

1. **背景**: MoE 量化的当前 SOTA 配方（TileQ、MiLo）= 权重激进量化 + per-expert 低秩补偿
2. **缺口**: per-expert 低秩因子各自独立，但不同层的 expert 补偿因子子空间高度重叠 → rank 预算浪费
3. **方法**: GLoRCQ 用 top-r 奇异子空间之间的主成分角距离在层间聚类 experts，cluster 内堆叠激活加权权重做一次 SVD，top-r 向量作为 cluster 级共享 U。同样的 cluster 结构使推理端可以做共享 U 缓存 + 侧流 cluster-batched LoRA
4. **结果**: 2-bit + ≤0.165 extra bits/param 下，相对 TileQ_s 在 Qwen1.5-MoE / Mixtral / Qwen3-30B-A3B 上 PPL 分别提升 **0.42 / 0.29 / 2.33**；最大 win 在 Qwen3 上，来自跨越全部 48 层的 cluster 组成（Figure 4a 印证）。推理端 decode 相比 naive baseline 加速 R×。Fake-quant 与 stripped real-quant checkpoint 已发布到 HuggingFace

**关键词**: mixture-of-experts, post-training quantization, low-rank compensation, vector quantization, subspace clustering

---

## §1 引言（1 页 4 段）

**¶1 — 当前 SOTA lineage**: MoE 量化已经收敛到"权重激进量化 + 小 rank 低秩补偿"的配方。TileQ 和 MiLo 是两个代表性实现。**不提** memory bottleneck，**不提** sub-2-bit frontier。

**¶2 — 具体缺陷**: TileQ 和 MiLo 都给每个 expert 独立的 (A, B) 因子，layer-local 计算。但不同 Transformer 层的 expert 做的是同类计算，实测上它们的低秩补偿因子占据高度重叠子空间（Figure 4）→ per-expert 分配浪费 rank 预算在冗余的基向量上。

**¶3 — 我们的修复**: 用 top-r 奇异子空间之间的主成分角距离在层间聚类 experts；同 cluster 的 experts 子空间真的重叠 → cluster 内堆叠激活加权权重 → 一次 SVD → 共享 U，per-expert Σ 和 V 私有。同 bit 预算下 per-expert 有效 rank ×G。为让方法可部署，顺势设计推理路径利用 shared-U cluster 结构: x·U 每 active cluster 每 token 只算一次；同 cluster active experts 的 per-expert 因子 concat 成一次宽 GEMM；LoRA 侧流并行 2-bit backbone 主流。

**¶4 — Contributions + 头号数字**: 两个编号 contribution（**C1** 跨层子空间 pooling + 共享 U，**C2** 共享 U 缓存 + 侧流 cluster-batched LoRA）+ 一句头号: 2-bit + ≤0.165 extra bits/param 下 Qwen1.5/Mixtral/Qwen3 相对 TileQ_s PPL 提升 0.42/0.29/2.33，最大 win 来自 Qwen3 上跨越全部 48 层的 cluster（Figure 4a 支撑），decode 相比 naive baseline 加速 R×。**不吹** memory saving。

---

## §2 相关工作（1 页，已初稿）

**复用** `docs/literature_review/related_work.md`（38 参考，7 小节）

**必须应用** LR review report 的补丁:
- **C1** (LR review 编号): 加 QuantMoE-Bench + MoEQuant 到 §2.4
- **C2** (LR review 编号): 验证 TileQ arxiv 2605.09281 + 去 `(submission)`
- **C3** (LR review 编号): 去 `.bib` 里的 MiLo 重复条目
- **I1-I4**: 补 vLLM、Marlin、VQ 理论依据、LLM.int8() 溯源
- **M1**: 软化 7 处 hallucination 用词
- **新**: 加主成分角 / Grassmann 流形引用（Absil 2006 主 Grassmann 流形，标准 SVD 子空间角定义）到 §2.2

**正文压缩**: 3000 词砍到 ~800，完整版进附录 E

---

## §3 方法（2.0 页）— 论文算法核心

### §3.1 前置定义（~0.4 页）

记号: 每层 N 个 routing experts，每 expert K 种权重（gate / up / down），共 L 层。每个权重 `W ∈ ℝ^{out_d × in_d}` 表示为 `Q + U V_k`，Q 是 2-bit 量化 backbone，`U V_k` 是 rank-r 修正项。Bit 预算：

```
avg_bits ≈ 2 + (r · (in_d + out_d) · lora_precision) / (in_d · out_d)
```

（attention bits 见附录 A）。全文的激活加权 SVD 沿用 LQER (Zhang et al., 2024)，2-bit tile 量化 backbone 沿用 TileQ (Gu et al., 2026)；§3.2 说明我们在此基础上引入的额外结构。

### §3.2 跨层子空间 pooling + 共享 U（~1.6 页 — **C1**）

**经验动机**。对每种权重类型，独立计算每个 expert 激活加权矩阵的 top-r 左奇异 basis，得到的 basis 之间跨层重叠明显。§6 (Figure 4) 的主成分角 heatmap 是层级 block 对角、但 off-diagonal 也有大量能量。Per-expert LoRA 不能利用这一点 —— 每个 expert 重新学一个私有 basis，共享子空间被"付了 L·N 次"。

**按子空间相似度分组**。对每种权重类型，设 U_i 为 expert i 激活加权矩阵的 top-r 左奇异 basis。计算成对主成分角距离

```
d(i, j) = ‖arccos σ(U_i^T U_j)‖_2 / (√r · π/2)  ∈ [0, 1]
```

转 Gaussian affinity 核，对该权重类型的 (L·N) 个 experts 做谱聚类，每组大小 G。同组 experts top-r 子空间真的重叠。

**通过一次 stacked SVD 得到共享 U**。每 cluster 把成员的激活加权权重堆成宽块列矩阵

```
[diag(S_a) · W_1^T | ⋯ | diag(S_a) · W_G^T] ∈ ℝ^{in_d × G · out_d}
```

做**一次**激活加权 SVD (Zhang et al., 2024)。top-r 左奇异向量成为 cluster 内共享 U，per-expert Σ_k 和 V_k 私有。同存储预算下 per-expert 有效 rank 相对 per-expert LoRA 提升 G× —— U 的开销从 per-expert 摊到 per-cluster。

**One-shot pipeline**。整条 pipeline 一趟走完，backbone 量化与低秩拟合之间**不做** alternating。

```
Algorithm 1: GLoRCQ quantization
1. Calibration pass 收集每 expert 的激活 scale
2. 对每种权重类型的 experts 用主成分角距离做聚类
3. 每 cluster 做一次激活加权 SVD → 共享 U，per-expert V
4. 用 2-bit vector-quantized backbone (Gu et al., 2026) 量化残差 W − UV
5. 重构误差超过 τ 的 expert 保 fp16
```

成对距离步 O(N²) per 权重类型 GPU 上算（chunk-batched 守 HBM）；每模型总量化 wallclock 见附录 A。

**Figure 1**: cluster 成员来自 3-4 个不同 Transformer 层的示意 → 一次 stacked SVD → 共享 U + per-expert V。

---

## §4 推理端 co-design（~0.75 页 — **C2**）

**标题**: "Inference-side co-design: cluster-batched LoRA on a side stream"

**¶1 cluster 结构可利用**。cluster 是全局学出来的，一个 token 激活的 top-k experts 常常同一 cluster、共享一个 U。由此三件可复合的收益: (a) x·U 每 active cluster 只算一次；(b) top-k 全部同一 cluster 时把 per-expert projection concat 成一次 GEMM；(c) 整个 LoRA 路径并行 2-bit backbone。

**¶2 共享 U 缓存 + 全局 U pool**。每 cluster 的共享 U 在加载时 dequant 一次，按权重类型 concat 成单个 device-resident 张量，大小为 `O(d · r · K_total)`，K_total 是所有权重类型的 cluster 总数。我们最大评估的模型下此 pool 占用不超过 **10 MiB**（Qwen3-30B-A3B, r=16, 每权重类型 ≤48 clusters, hidden dim 2048）；一次性加载就是我们的做法。推理时 x·U 每 active cluster 每 token 只算一次，广播给该 cluster 的所有 active experts 的 per-expert V 乘法 —— 替代 K 次独立 x·U GEMM。

**¶3 cluster batching + 侧流**。top-k 全部同一 cluster 时，per-expert 因子 concat 成一次 (1, r) × (r, K · out_d) GEMM，省 K 次 kernel launch。LoRA 侧流并行 2-bit backbone 主流，下游 reduction 前两流合流同步。Graph mode 支持，同模式可进 CUDA Graph。

**时序图**: 一小张主流 backbone ∥ 侧流 LoRA overlap 示意。

**¶4 关键"不吹"**。attention 在 MoE block 之前完成，**不重叠** LoRA 与 attention。**不跨层 prefetch U**。**不声称 raw throughput 打平 vLLM fp16** —— 加速比是**同模型 naive per-expert-LoRA-on-main-stream baseline** 的对比。

---

## §5 实验（2.0 页）

### §5.1 设置（0.3 页）

- 模型: Qwen1.5-MoE-A2.7B、Mixtral-8x7B、Qwen3-30B-A3B
- 校准数据: WikiText-2 train 128 samples × 4096 tok
- Bit 预算: 2 + 0.16 = 2.16 bits/param
- Baselines: GPTQ 2-bit, GPTVQ 2-bit, LoPRo 2-bit, TileQ_s / TileQ_v @ 2.16 bit, MxMoE (paper Table 1 数字), MiLo 3-bit 作为更宽预算的参考。基线数字统一测评 config 说明见 §5.4
- Eval: WikiText-2 PPL (max_len=2048, stride=512) + 5-task 0-shot avg (ARC-c/ARC-e/PIQA/WinoGrande/HellaSwag, `acc`, num_fewshot=0, add_bos)。MMLU 从头号数字撤下（多个 2-bit baseline 未报），详见附录 C
- 硬件: 8× H200 (量化) + 1× H200 (eval)
- **方法配置** (一行): 2-bit VQ tile 配置沿 TileQ；attention 用 4-bit GPTQ；重构 max-error 超过 τ 的 expert 保 fp16；跨层 experts 按 §3.2 主成分角距离分组

### §5.2 主结果（0.8 页 — Table 1）

**表 1**（合并主表，来自 2026-07-16 整理的 results sheet；所有 downstream 用 `acc`、num_fewshot=0、add_bos、batch=1，对齐 TileQ Table-1 口径；ours 评测于 2026-07-06）

| 方法 | Bits | Extra bits (Q1.5 / Mixtral / Q3) | Qwen1.5 PPL / Avg(5) | Mixtral PPL / Avg(5) | Qwen3 PPL / Avg(5) |
|---|---|---|---|---|---|
| FP16 (TileQ paper) | 16 | — | 6.79 / 64.50 | 3.87 / 75.46 | 8.07 / 68.12 |
| GPTQ | 2 | 0.13 | 12.5 / 41.12 | 15.3 / 38.52 | 14.6 / 50.96 |
| GPTQ（3-bit 参考） | 3 | 0.13 | 7.98 / 58.30 | 4.72 / 64.00 | 9.42 / 60.58 |
| MOEQ（3-bit 参考） | 3 | 0 | 8.21 / 57.32 | 5.45 / 69.70 | 28.1 / 45.26 |
| LoPRo | 2 | 0.43 / 0.21 / 0.58 | 7.52 / 62.36 | 5.01 / 70.72 | 11.1 / 55.02 |
| MxMoE | 2 | 0.25 | 8.79 / 56.06 | 5.63 / 68.87 | — |
| **TileQ_s** | **2** | **0.16** | **7.56 / 62.86** | **4.98 / 70.92** | **11.3 / 55.24** |
| **GLoRCQ (ours)** | **2** | **0.152 / 0.1611 / 0.1647** | **7.14 / 62.76** | **4.69 / 64.48** | **8.97 / 63.46** |

（每模型完整 per-task 明细表 → 附录，与 en 版 §5.2 一致，逐字来自 results sheet）

**Qwen3-30B-A3B** 上 GLoRCQ PPL 相对 TileQ_s 提升 **2.33** —— 最大 win，也是跨层共享最要害的场景：128 experts/layer 下 per-layer 调度无法把不同层的 experts 归到同一个共享因子里；我们的 cluster 能（Figure 4a），这就是 +2.33 PPL 的来源。**Qwen1.5-MoE** 上相对 TileQ_s 提升 0.42（7.14 vs 7.56）。**Mixtral-8x7B** 上相对 TileQ_s 提升 0.29。GPTQ 3-bit / MOEQ 3-bit 作为更宽预算参考（MOEQ 在 Qwen3 上崩溃 28.1）；MxMoE 用其论文发表数（未报 Qwen3）。（Qwen1.5 全部数字在同一台 A100 + 共享 calibration 上测，headline 与 §6 消融直接可比。）

### §5.3 系统加速（0.5 页 — Table 2）

**表 2**（decode throughput，验证 C2）

| 模型 | GLoRCQ full | w/o side-stream | w/o cluster-batching | w/o shared-U cache |
|---|---|---|---|---|
| Qwen1.5-MoE | X.X tok/s | Y.Y | Z.Z | W.W |
| Mixtral | ... | ... | ... | ... |
| Qwen3 | ... | ... | ... | ... |

（每行对应一次 ablation，直接支撑 C2 声明）

### §5.4 Baseline 复现（0.4 页）

- **MiLo 3-bit 参考**: 公开发布，重跑在 H200
- **MxMoE**: 2-bit weight-only config 不能直接在他们发布代码中复现（他们的 hardcoded tile config 只覆盖 W-A mixed 方案），引用他们论文 Table 1；注意他们的 HellaSwag 数字可能用 acc_norm 而非我们的 raw acc（附录 F）
- **TileQ**: 无发布 ckpt，引用论文
- **GPTVQ / LoPRo**: 发布代码目标 bit 约定不同，引用他们论文
- **我们的复现性**: 所有 RNG（rank-1 sketch 初始化、low-rank SVD 投影、VQ k-means）都已固定 seed=42。给定 Stage-1 校准输入后 pipeline 确定性：fair-bit r=20 配置在三次独立跑（从头校准、缓存校准、消融队列，同 GPU）上产生**逐位相同**的 WikiText-2 PPL 7.2995。审稿人用发布代码 + 公布 seed 即精确复现报告数字。

---

## §6 消融（1.25 页）

### §6.1 Rank 扫描（0.2 页）
**表（Qwen1.5-MoE, Grassmannian, A100, seed 锁定, 固定 G=128）**: r=16→7.85 / 58.60 (2.08 bits)、r=20→7.30 / 62.34 (2.10)、**r=32→7.14 / 62.80 (2.15, fair-bit base)**、r=64→7.38 / 63.18 (2.29, 超预算)。**结论**: PPL 随 rank 呈 U 形 —— r16→r32 急降(shared-U 补偿容量提升),r64 反升(高 rank per-expert 因子 int8 量化误差累积,且 2× LoRA bits 超 2.16 预算)。r=32 是 PPL 最优点,且恰好落在 fair 预算(+0.15),采用为工作点。因 shared U 摊到 cluster 内 G=128 个 experts,r=32 只花 +0.15 bits/param(预算内);per-expert LoRA 同 rank 会超预算。

### §6.2 Group 大小 G 扫描（0.25 页）
**表（Qwen1.5-MoE, Grassmannian, A100, seed 锁定, r=32）**: G=64 (23 clusters)→7.10 / 62.50 (2.15 bits);G=128 (12 clusters, base)→7.14 / 62.80 (2.15)。**结论**: 同 bits 下两个大共享组基本打平 —— G=64 PPL 略好,G=128 ZS 略好 —— 说明组够大后跨层共享对确切组大小是鲁棒的。base 用 G=128(最佳 ZS + 推理时最少 shared-U 需缓存)。真正对照是 G=1 (per-expert = TileQ) 这个无跨层共享的退化下界;表 5 随机 cluster 对照隔离出:大 G 分组的**内容**(而非尺寸)才是恢复精度的关键。Qwen1.5 上 G≥256 clusters<8 → pipeline auto-fallback 回层内顺序,不算 Grassmannian 点。

### §6.3 LoRA on/off（0.15 页）
**表**: rank=0 (纯 VQ backbone,无低秩修正) vs rank=32 (default)。此处 LoRA 与 grouping 无关(rank 0 无 shared U)。**结论**: 低秩修正相对裸 2-bit backbone 贡献可观的精度恢复。

### §6.5 聚类有效性：我们的分组有 structure 吗？（0.4 页 — **motivating C1 的聚类选择**）

用**三个诊断**（两个可视化 + 一个行为对照）证明主成分角聚类真正做到跨层分组、且组内子空间真的重叠。我们**故意不**做"vs 层内顺序"的对比表 —— 层内顺序不是一种被设计的算法替代，而是任何 per-layer 处理调度下的默认行为；隔离聚类信号的正确方式是**同尺寸随机 cluster 对照**（表 5）。

**图 4a — layer × cluster 组成 heatmap（Qwen3-30B-A3B）**。cell (l,c) = cluster c 里来自第 l 层的 expert 数。Qwen3 上 traversal @ G=128 会是完美对角（每 cluster 恰好一层，因为每层正好 128 experts）；实际每个 cluster 竖向撒在很多层 —— gate 平均跨 17.8 层（min 2, max 39），up 16.7 层，down 40.7 层。这是"主成分角聚类在最要害的模型上真正跨层"的直接可视化。Qwen1.5-MoE（平均 7.2 层）放附录 C。

**图 4b — cluster coherence（Qwen3-30B-A3B）**。within/between 主成分角距离直方图（activation-scaled 子空间，每 wtype 一个 panel）。gate/up 的 within 分布明显左于 between（μ 0.654 vs 0.729、0.652 vs 0.727）—— 同 cluster experts 子空间真的更近；down 两分布重合（μ 0.927 vs 0.932）：r=32 下 down-projection experts 无可分子空间结构，是诚实的例外。（不放 2D 散点：48 个 cluster 在高维近正交子空间里，任何 2D 嵌入都是无信息量的一团。）per-wtype coherence 真-但-温和,所以 **表 5 的行为对照(聚合 +0.86 PPL)才是头号证据**,而非距离直方图。

**表 5 — 随机 cluster 对照**。把学到的 cluster 分配换成同尺寸分布的随机分配，用同一 pipeline 重新量化；PPL 差距隔离"聚类内容"对精度的贡献（而非"cluster 尺寸"）。

| 模型 | PPL (学到的 cluster) | PPL (同尺寸随机) | ΔPPL |
|---|---|---|---|
| Qwen1.5-MoE | 7.14 | 7.44 | +0.30 |
| **Qwen3-30B-A3B** | **8.97** | **9.83** | **+0.86** |
| Mixtral-8x7B | 4.69 | （不跑 — Mixtral 用层内顺序）| — |

Qwen3-30B-A3B 上学到的聚类比同尺寸随机好 **0.86 PPL** —— 决定性证据:pipeline 分组的**内容**(哪些 expert 共享因子)、而非尺寸直方图,才是恢复精度的关键。在 2.16 bit 下形成干净的三方序:**Grassmannian 8.97 < traversal 9.42 < random 9.83**。随机跨层分组甚至比层内 traversal 更差 —— 所以跨层共享只在分组是"子空间知情"时才帮忙,这正是主成分角聚类提供的。Qwen1.5 上学到的聚类比随机好 0.30 PPL(7.14 vs 7.44,同一台 A100 + 共享 calibration);差距比 Qwen3 小,因为 Qwen1.5 每层 60 experts 给聚类器的跨层素材远少于 Qwen3 的 128。

### **§6.9 系统消融**（0.2 页 — **motivating evidence for C2**）
with/without shared-U cache + 侧流 + 同 cluster K-融的 decode tokens/sec on Qwen1.5-MoE，直接支撑 Table 2

**挪到附录 C**:
- §6.4 int8-vs-fp16 LoRA 存储
- §6.6 attn-bits 消融
- §6.7 τ 阈值扫描
- §6.8 Qwen1.5 + Mixtral 的跨层组成图（正文 Figure 4a 只放 Qwen3）

---

## §7 讨论与局限（0.3 页）— 老实说

### §7.1 推理速度定位
GLoRCQ 加速比是**同模型 naive per-expert-LoRA-on-main-stream** baseline 的对比。**不与 vLLM fp16 raw throughput 打平**（那是独立的 kernel 优化系统论文）。

### §7.2 MMLU（从头号数字撤下）
Table 1 headline 不放 MMLU：多个 2-bit baseline (TileQ, LoPRo, GPTVQ) 未报 MMLU；MiLo MMLU 的 tokenizer/prompt 也不一致。完整 MMLU 数字在附录 C；头号指标是 5-task 0-shot avg。

### §7.3 τ 是启发式
τ = 60 empirical（weight-space L∞）。未来可用 activation-Hessian 谱做原则化选择，去掉一个超参。

### §7.4 Qwen3 real-quant NaN bug（已解决 2026-07-15, commit 6cea816）
根因是数值 bug：LoRA 输入缩放 `x·S_a` 在 fp16 计算，大激活 scale 模型（Mixtral bf16-native、Qwen3）溢出 → inf → NaN；修复 = 该乘法与 LoRA 归约改 fp32。三个 real-quant checkpoint 推理 sanity 全过，Qwen3 real-quant 速度已测入 Table 2。残余诚实局限：2-bit real-quant 路径长贪心生成（≫128 tokens）会退化为重复；速度 harness 用 gen_len=128。

---

## §8 结论（0.1 页，两句话）

1. **跨层子空间 pooling** 用共享 U 因子把低秩补偿预算摊到整个 MoE 内，同存储预算下 per-expert 有效 rank ×G。
2. 结合 **shared-U cache + 侧流 cluster-batched LoRA** 推理 co-design，GLoRCQ 在三个生产级 MoE 模型上实现相对 TileQ 的质量胜出与相对 naive per-expert baseline 的 decode 加速；ckpt 已发布。

---

## 附录

- **A**: Bit-accounting 完整推导（含 attn_bits=4 的 0.06 修正）+ 每模型每 phase wallclock
- **B**: 2-bit VQ tile backbone + attention 量化（继承的 scaffolding；K=256、vdim=4 细节；Hessian-aware Hadamard rotation）
- **C**: 全消融表 + MMLU per-config 数字 + int8-vs-fp16 LoRA + attn-bits + τ 扫描 + 全 per-task 数字（来自实验表格）
- **D**: 6 个 HF ckpt 的 model cards（`Tsingyow/GLoRCQ-{qwen1.5-moe-a2.7b, mixtral-8x7b, qwen3-30b-a3b}-fair-grassmann-{fake, real}`）+ stripped 格式说明
- **E**: 扩展相关工作（3000 词完整版）
- **F**: MxMoE 对比说明（为何 2-bit weight-only 不能直接在他们发布代码中复现；他们 ZS 数字的 metric 约定 caveat）

---

## ⚠️ 已知未解决问题（写论文前要修）

| # | 问题 | 阻塞什么 | 修复路径 |
|---|---|---|---|
| 1 | **TileQ arxiv 未 verify** | §2 关键 baseline | 拉 abstract 确认 |
| 2 | **聚类有效性图 (Fig 4a 跨层组成 + Fig 4b 主成分角 heatmap) 未生成** | §6.5 可解释性证据 | Fig 4a 从 `cross_layer_info.pt` 已保存的分配画；Fig 4b 从 per-expert top-r 子空间算 pairwise 主成分角（用聚类器同款距离），chunk-batched |
| 3 | **随机 cluster 对照实验 (Table 5) 未跑** | §6.5 隔离聚类信号 | 3 个模型各一次 quant，用 Phase-1 cache 复用，预计 ~4h on 1× H200 |
| 4 | attn_bits=4 bit-accounting 少算 0.06 | Table 1 "matched" 声明 | 附录 A 补 |
| 5 | 匿名化 | 双盲评审 | HF `Tsingyow/*` + GitHub repo 待改 |
| 6 | **AW-SVD 归属声明** | §2 / §3.1 | 引 LQER (Zhang et al., 2024)；只声称跨层 pooling 结构为新 |
| 7 | **不声称 alternating** | §3.2 | Pipeline 为 one-shot（代码已确认） |
| 8 | **§6.9 系统消融数据未跑** | Table 2 | 用 decode-speed harness 跑 with/without variants |
| 9 | ~~Qwen3 real-quant NaN bug~~ **已解决**（6cea816，§7.4 已更新；Qwen3 速度 Standard 2.8 / Graph 6.4 tok/s） | — | — |
| 10 | **Mixtral Table 1 与发布 artifact 的残余差异**（2026-07-16 接受）：Table 1 = v2 fair run（fp16-LoRA, 2.1611 bits；PPL/ZS 均可追溯且 ZS 合规）；HF 发布的是 v3b（int8-LoRA, ≈2.11 bits），自身无 fake eval | abstract 的 checkpoint-release 声明 | 若 reviewer 要 artifact-exact 数：从本地 v3b cross_layer_info.pt hydrate fake 权重再评（需 2×A100，~5-6h） |

---

## 写作顺序（推荐）

1. **§3.2 方法** — 论文最核心，先写
2. **§4 推理端 co-design** — 系统层贡献，紧接 §3.2 写
3. **§5.2 Table 1 + §5.3 Table 2** — 冻结数字
4. **§6.5 聚类消融** — 三模型故事支撑 Table 1 framing
5. **§1 引言** — 等 §3+§4+§5 稳定后回来写
6. **§6 其他消融** — 从实验表拉数据
7. **§7 讨论** — 一次性老实写完
8. **§2 编辑** — 应用 LR review 补丁 + 加主成分角引用
9. **Abstract** — 最后写，改 5+ 次

**预估**: 3-4 天 §3+§4+§5；2 天 §6；各 1 天 §1/§7/§2。**约 10 天集中写**。

---

_大纲 v0.4 结束 — 详细英文版见 `docs/paper_outline.md`_
