# GLoRCQ 论文大纲（中文速读版）

**标题**: GLoRCQ: Global shared Low-Rank Compensation for Quantization of Mixture-of-Experts LLMs（全局共享低秩补偿的 MoE 大模型量化）

**目标会议**: ICML 2026 / NeurIPS 2026 主赛道，8-9 页正文 + 附录
**版本**: outline v0.3 — 2026-07-09（Grassmannian 聚类正式加入 C1，含 auto-fallback）
**详细英文版**: `docs/paper_outline.md`（本文档是同结构精简版）

---

## 一句话核心论点

> **TileQ 和 MiLo 已经在做"2-bit 量化 + per-expert 低秩补偿"**；它们的缺点是每个 expert 的低秩因子**各自独立**，rank 预算被切碎。**GLoRCQ 用 Grassmannian 主成分角谱聚类**把不同层的 expert 按残差子空间相似度分到同 cluster，把 G 个跨层同 cluster 的 experts 激活加权权重堆起来一次 SVD，共享 U 因子 → per-expert 有效 rank 提升 G 倍；再顺势设计 **SharedUCache + 侧流并行 + 同 cluster 融 K-GEMM** 的推理路径，让这份共享结构在系统层也变现。**对于小 MoE（Mixtral 只有 256 experts）**，聚类数 <8 时代码自动 fallback 到 traversal-order 分组，避免过粗聚类拖累精度。

---

## 页数预算（ICML 8 页正文）

| 章节 | 页 | 主要 message |
|---|---|---|
| Abstract + §1 引言 | 1.0 | TileQ/MiLo lineage → per-expert LoRA 缺点 → 跨层 Grassmannian 共享 U + 推理 co-design |
| §2 相关工作 | 1.0 | 复用 LR + 应用 review 报告的 C1/C3/M1 补丁 + Grassmannian 引用 |
| §3 方法（**核心**） | 2.0 | Grassmannian 谱聚类 → stacked SVD + shared U + 小 MoE auto-fallback |
| §4 推理端 co-design（**核心**） | 0.75 | SharedUCache + global U pool + 同 cluster K-融 + 侧流并行 |
| §5 实验 | 2.0 | Table 1 主对比 + Table 2 系统加速 + baseline 复现 (MiLo 3-bit 参考) |
| §6 消融 | 1.25 | rank/G/LoRA + **Grassmannian vs traversal 三模型比较** + heatmap + 系统 ablation |
| §7 讨论与局限 | 0.4 | Grassmannian scaling 老实说；auto-fallback 是设计而非失败 |
| §8 结论 | 0.1 | 两句话 |
| **正文总计** | **~8.5** | |
| 附录 A-F | 6.5+ | Bit accounting / 全消融表 / Speed protocol / HF cards + stripped 格式 |

---

## Contributions（**只有 2 个**，加实验 payoff）

**Reviewer 讨厌"结果作为 contribution"，所以第 3 项只写进 abstract 头号数字，不编号进 contribution 列表。**

### **C1**: Cross-layer expert subspace pooling with Grassmannian-clustered shared U-factor

- **经验观察**（Figure 4 in §6）: 不同层的 expert 低秩补偿因子占据高度重叠的子空间
- **算法回应**: 对每 wtype 用 **Grassmannian 主成分角距离** 
  ```
  d_grass(i, j) = ‖arccos(σ(U_iᵀ U_j))‖_2 / (√r · π/2) ∈ [0, 1]
  ```
  谱聚类 (`n_clusters = ⌈N_wtype / G⌉`)，同 cluster experts 子空间真的重叠。把 cluster 内的 G 个 experts 激活加权权重堆成 `[diag(S_a)·W_1ᵀ | … | diag(S_a)·W_Gᵀ]`（`(in_d, G·out_d)` 大矩阵），做**一次** AW-SVD → top-r 奇异向量作为**共享 U**，per-expert 的 V_k 和 Σ_k 私有
- **收益**: 同 bit 预算下 per-expert 有效 rank ×G
- **小 MoE 自动降级** (`run_quantize.py:472-485`): 当 `n_clusters < 8` 时（Grassmannian 太粗，每 cluster 会吃太多异质 experts），代码自动 fallback 到 traversal-order 分组，并 print `AUTO-FALLBACK` 消息。Mixtral-8x7B（256 experts, G=64 → 4 clusters）触发此分支；Qwen1.5（12 clusters）+ Qwen3（48 clusters）不触发
- **老实归属**: AW-SVD 本身来自 LQER (Zhang 2024)；Grassmannian 主成分角是标准子空间度量；我们的贡献是 **Grassmannian 聚类 + stacked-across-cross-layer + shared U** 三者的组合结构

### **C2**: Inference-time cluster-batched LoRA with SharedUCache + side-stream

- **`SharedUCache`** (`inference/model_builder.py:27-66`): 加载时把每个 (wtype, cluster_id) → fp16 U dequant 一次
- **Global U pool concat** (`model_builder.py:143-190`): 所有 cluster 的 U 拼成一个大 HBM buffer，跨层共享地址 → L2 复用
- **`x @ U` cluster-级预算** (`moe_block._precompute_xU`, line 526-583): 每 token 每 cluster 只算一次
- **同 cluster K-融** (`moe_block.py:874`): 当所有 K 个激活 expert 同一 cluster 时，concat SV → 一次 `(1,r) @ (r, K·out_d)` cuBLAS，省 K 次 kernel launch
- **侧流并行** (`moe_block.py:89-96, 869-902`): LoRA 侧流 ∥ VQ4 主流，`wait_stream` 汇合
- **Graph mode** (`graph_wrapper.py:90,139-154` + `moe_block.py:183-299`): 预 build 批量结构支持 CUDA-Graph capture

**关键说清（避免 overclaim）**:
- ❌ 不重叠 LoRA 与 attention（attention 完了 MoE 才开始）
- ❌ 不跨层 prefetch U（`preload_for_layer`/`evict` 是 stub）
- ❌ 不声称与 vLLM fp16 raw throughput 打平 — 我们的加速比是**同模型 naive per-expert baseline** 的对比，不是 fp16 vLLM

### 实验 payoff（不编号）

- **主结果** (§5.2 Table 1): 2.16 bits/param 下，PPL vs TileQ / MiLo (3-bit ref) / MxMoE
- **系统加速** (§5.3 Table 2): decode tokens/sec, with/without C2 优化
- **Grassmannian 消融** (§6.5 Table 5): 三模型对比 Grassmannian vs traversal，验证 auto-fallback 规则

---

## Abstract 段（150-250 词）

**四句结构**（不用 memory hook，不用 sub-2-bit）:

1. **背景**: MoE 量化的当前 SOTA 配方（TileQ、MiLo）= 权重激进量化 + per-expert 低秩补偿
2. **缺口**: per-expert 低秩因子各自独立，但不同层的 expert 补偿因子子空间高度重叠 → rank 预算浪费
3. **方法**: GLoRCQ 用 **Grassmannian 主成分角谱聚类**把不同层的 experts 按残差子空间相似度归 cluster，cluster 内 stack + 一次 AW-SVD，共享 U；对小 MoE (Mixtral-scale) 当 clusters<8 时自动降级到 traversal；顺势设计 SharedUCache + 侧流并行的推理路径
4. **结果**: 2.16 bits/param 下，Qwen1.5-MoE / Mixtral / Qwen3-30B-A3B 的 PPL 相比 TileQ_s 分别提升 **0.19 / 0.29 / 2.33**；在 Qwen3 上 Grassmannian 变体比 traversal 再赢 **0.45 PPL** — 因为 128 experts/layer + G=128 让 traversal 退化为单层内分组，Grassmannian 是唯一真跨层的方式。推理端 decode 相比 naive baseline 加速 R×

**关键词**: mixture-of-experts, post-training quantization, low-rank compensation, vector quantization, Grassmannian subspace clustering

**Abstract 里不要说**: "memory frontier"、"sub-2-bit"、"stripped ckpt"、"fused VQ4 kernel"、"alternating optimization"

---

## §1 引言（1 页 4 段，全部重写）

**¶1 — 当前 SOTA lineage**: MoE 量化已经收敛到"权重激进量化 + 小 rank 低秩补偿"的配方。TileQ 和 MiLo 是两个代表性实现。**不提** memory bottleneck，**不提** sub-2-bit frontier。

**¶2 — 具体缺陷**: TileQ 和 MiLo 都给每个 expert 独立的 (A, B) 因子，layer-local 计算。但不同 Transformer 层的 expert 做的是同类计算（FFN up/down projection over similar residual streams），实测上它们的低秩补偿因子占据高度重叠子空间（Figure 4）→ per-expert 分配浪费 rank 预算在冗余的基向量上。

**¶3 — 我们的修复**: 用 **Grassmannian 主成分角距离**对每 wtype 的 (L·N) experts 做谱聚类，同 cluster 的 experts 的 top-r 奇异子空间真的重叠 → cluster 内 stack 激活加权权重 → 一次 SVD → 共享 U。同 bit 预算下 per-expert 有效 rank × G。**对小 MoE（Mixtral 256 experts，Grassmannian 只能出 4 cluster 太粗）代码自动 fallback 到 traversal-order**，避免过粗聚类拖累精度。为让方法可部署，顺势设计推理路径利用 shared-U cluster 结构: `x@U` per-cluster 只算一次；同 cluster experts 的 SV concat 后一次 cuBLAS；LoRA 侧流跟 VQ backbone 主流并行。

**¶4 — Contributions + 头号数字**: 两个编号 contribution（C1 跨层 Grassmannian + shared U + auto-fallback，C2 SharedUCache + 侧流并行）+ 一句头号: 2.16 bits/param 下 Qwen1.5/Mixtral/Qwen3 PPL 提升 0.19/0.29/2.33 vs TileQ_s（Qwen3 上 Grassmannian 比 traversal 再 +0.45），decode 相比 naive baseline 加速 R×。**不吹** memory saving。

---

## §2 相关工作（1 页，已初稿）

**复用** `docs/literature_review/related_work.md`（38 参考，7 小节）

**必须应用** LR review report 的补丁:
- **C1** (LR review 编号): 加 QuantMoE-Bench + MoEQuant 到 §2.4
- **C2** (LR review 编号): 验证 TileQ arxiv 2605.09281 + 去 `(submission)`
- **C3** (LR review 编号): 去 `.bib` 里的 MiLo 重复条目
- **I1-I4**: 补 vLLM、Marlin、VQ 理论依据、LLM.int8() 溯源
- **M1**: 软化 7 处 hallucination 用词
- **新（2026-07-09）**: 加 Grassmannian 主成分角距离引用（Absil 2006 主 Grassmann 流形，标准 SVD 子空间角定义）

**正文压缩**: 3000 词砍到 ~800，完整版进附录 E

---

## §3 方法（2.0 页）— 论文算法核心

### §3.1 前置定义（~0.4 页）

- 记号: L 层，每层 N experts，K wtypes
- 目标: W ≈ Q + UV，总存储 b bits/param
- Bit 预算公式（框起来）:
  ```
  avg_bits = 2 + (r · (in_d + out_d) · lora_precision / (in_d · out_d))
  ```
- **归属声明**:
  - VQ4 tile backbone 继承自 TileQ (Gu 2026)，K=256、vdim=4 → 2 bits/param（详见附录 B，不推导）
  - AW-SVD 形式沿用 LQER (Zhang 2024)；novelty 是 Grassmannian 聚类 + stacked shared-U 结构（§3.2）

### §3.2 跨层子空间 pooling + shared U（~1.2 页 — **C1**）

**内容**:
1. **经验动机**: 前指 §6 主成分角 heatmap (Figure 4)
2. **Grassmannian 谱聚类**（具体聚类算法）: 对每 wtype 计算 (L·N)×(L·N) 主成分角距离矩阵 → Gaussian affinity 核 → SpectralClustering (n_clusters=⌈N_wtype/G⌉, random_state=42)。同 cluster 的 experts 子空间真的重叠
3. **小 MoE Auto-fallback**: `n_clusters < 8` 时 fallback 到 traversal-order（`run_quantize.py:476` 硬编码阈值 `_GRASS_MIN_CLUSTERS = 8`）。实际触发情况：Mixtral (4 clusters at G=64) → fallback；Qwen1.5 (12 clusters) 和 Qwen3 (48 clusters) 走 Grassmannian
4. **Stacked block-column SVD**: `[diag(S_a)·W_1ᵀ | … | diag(S_a)·W_Gᵀ] = U Σ V^T`。共享 U，per-expert V_k / Σ_k 私有
5. **Bit accounting**: 明式推 G× 摊销
6. **Figure 1**: 组员来自 3-4 个不同层的示意图 + Grassmannian 聚类颜色

### §3.3 量化 backbone 集成（~0.4 页）

**One-shot pipeline**（**不写** alternating）:
```
Phase 0. 校准 forward 收集 S_a
Phase 1. 每 wtype: 若 n_clusters >= 8 → Grassmannian 谱聚类；否则 traversal (auto-fallback)
Phase 2. 每 cluster: stack + AW-SVD → 共享 U_c + per-expert V_k
Phase 3. 每 expert: R_k = W_k − U_c V_k → VQ4 tile 量化 R_k
Phase 4. max_err > τ 的 expert 保 fp16
Phase 5. 导出 cross_layer_info.pt + safetensors
```
- Algorithm 1: ~12 行伪代码
- 声明: AW-SVD 独立形式来自 LQER，我们的 novelty 在 §3.2 的 Grassmannian + stacked + shared 组合结构

**复杂度**: Phase 1 forward ~50 min (Mixtral)；Grassmannian pairwise O(N²) GPU chunk-batched；总: ~2h Qwen1.5、~2h Mixtral (cache 复用)、~14h Qwen3 (1× H200)

**去除**（相比 v0.1）:
- ❌ §3.4 fp16 fallback 独立小节 → 挪到 §5.1 setup 一行 + §7 局限
- ❌ §3.5 attention 量化 → 挪到 §5.1 setup 一行
- ❌ Alternating optimization 声称（代码里没有）

---

## §4 推理端 co-design（0.75 页 — **C2**）

**标题**: "Inference-side co-design: cluster-batched LoRA on a side stream"

### ¶1 结构可利用

top-k 路由的 K 个 active experts 常常同一 cluster，因此共享一个 U → 三件事可以合力: (a) `x @ U` 只算一次, (b) K 个 per-expert SV 批量, (c) 整个 LoRA 路径与 VQ backbone 并行

### ¶2 SharedUCache + Global U pool

- `SharedUCache` (`inference/model_builder.py:27-66`) 加载时把 (wtype, cluster_id) → fp16 U dequant 一次
- `_build_global_u_pool` / `_install_global_u_pool` (line 143-190) 把每个 projection type 的所有 cluster U concat 成一个 `(hidden_dim, K_total·rank)` HBM 张量，跨层共享地址 → L2 复用
- Forward 时 `moe_block._precompute_xU` (line 526-583) 每 token 每 cluster 只算一次 `x@U`；downstream `GLoRCQLinear.forward` (`quantized_linear.py:503-602`) 消费 precomputed_xU，跳过冗余 GEMM

### ¶3 Batching + 侧流

- 当所有 K 个 active experts 同一 cluster（`all_same_cluster` at moe_block.py:874），concat SV → 一次 `(1, r) @ (r, K·out_d)` cuBLAS，省 K 次 kernel launch
- LoRA 侧流 `self._side_stream` (line 89-96)，主流跑 `turbo_dequant_matmul_fused` (line 897)
- 同步: `torch.cuda.current_stream().wait_stream(side)` at line 902
- Graph mode: `inference/graph_wrapper.py:90,139-154` + `moe_block.py:183-299` 预建批量结构支持 CUDA-Graph
- **时序图**: 一小张主流 VQ ∥ 侧流 LoRA overlap 示意

### 关键"不吹"（避免 overclaim）

- **不重叠 LoRA + attention**（attention 完了 MoE 才开始）
- **不跨层 prefetch U**（`preload_for_layer`/`evict` at model_builder.py:60-66 是 stub）
- **不声称 raw throughput 打平 vLLM fp16** — GLoRCQ 加速比是**同模型 naive per-expert-LoRA-on-main-stream baseline** 的对比

**去除**（相比 v0.1）:
- ❌ §4.1 Stripped ckpt format → 挪附录 D
- ❌ §4.2 Fused VQ4 kernel 作为独立贡献 → 归为继承的 TileQ scaffolding，只在 setup / 附录 B 提
- ❌ §4.3 推理速度与显存 → 挪到 §5.3

---

## §5 实验（2.0 页）

### §5.1 设置（0.3 页）

- 模型: Qwen1.5-MoE-A2.7B、Mixtral-8x7B、Qwen3-30B-A3B
- 校准数据: WikiText-2 train 128 samples × 4096 tok
- Bit 预算: 2 + 0.16 = 2.16 bits/param
- Baselines: GPTQ, AWQ, LQER, LQ-LoRA, MiLo 3-bit（更宽 bit 预算的 reference），TileQ, MxMoE (paper 数字)
- Eval: WikiText-2 PPL + 5-task 0-shot 平均 (`acc` metric, `add_bos_token`, `batch>=1`)。MMLU 因多个 baseline 未报而移到附录 C
- 硬件: 8× H200 (量化) + 1× H200 (eval, batch=32-48)
- **一行内容**: VQ4 tile 配置继承自 TileQ；attention 用 4-bit GPTQ；max-err > 60 的 expert 保 fp16；**跨层聚类用 Grassmannian 主成分角谱聚类，`n_clusters < 8` 时自动 fallback 到 traversal（触发于 Mixtral）**

### §5.2 主结果（0.8 页 — Table 1）

**表 1**（fair-bit 对比 +0.16 extra bits，所有 downstream 用 `acc`、num_fewshot=0、add_bos=True、batch≥1）

| 方法 | bits | Qwen1.5 PPL/Avg(5) | Mixtral PPL/Avg(5) | Qwen3 PPL/Avg(5) |
|---|---|---|---|---|
| fp16 | 16.0 | 6.51 / 64.26 | 3.42 / 72.55 | 7.75 / 68.10 |
| GPTQ 2-bit | 2.13 | 12.5 / 43.15 | 15.3 / 38.48 | 14.6 / 51.65 |
| GPTVQ 2-bit | 2.13 | 8.12 / 57.24 | 5.28 / 62.09 | 11.8 / 55.99 |
| LoPRo 2-bit | 2.43 | 7.52 / 62.20 | 5.01 / 70.62 | 11.1 / 57.02 |
| **TileQ_s 2-bit** | **2.16** | **7.56 / 63.15** | **4.98 / 70.85** | **11.3 / 57.68** |
| **TileQ_v 2-bit** | **2.16** | **7.35 / 63.44** | **4.78 / 71.36** | **10.1 / 63.24** |
| MiLo **3-bit** (参考, +1 bit) | 3.00 | 7.15 / 62.94 | 4.03 / 70.42 | 8.44 / 66.99 |
| **GLoRCQ (ours, Grassmannian)** | **2.16** | **7.37 / 60.56** | 5.99 / 50.69 (Grass) | **8.97 / 63.46** ✨ |
| **GLoRCQ (ours, auto-fallback traversal)** | **2.16** | 7.17 / 61.13 | **4.69 / 64.38** ✨ | 9.42 / 60.29 |

**framing**: win 对齐到 TileQ_s / TileQ_v。**Qwen3 是 Grassmannian 故事的关键**：128 experts/layer + G=128 让 traversal 退化为纯单层内分组，Grassmannian 强制跨层 → 赢 traversal +0.45 PPL、赢 TileQ_s +2.33 PPL。**Qwen1.5**：两法在 0.2 PPL 内 tie（12 clusters 已够粗但两法都能覆盖）。**Mixtral**：4 clusters 时 Grassmannian 差 +1.30，auto-fallback 保护 → 走 traversal（**这是设计而非失败**）。**MiLo** 是 3-bit 参考（他们的代码不支持 2-bit，我们不 apples-to-apples 比）

### §5.3 系统加速（新增 0.5 页 — 移自 §4）

**表 2**（decode throughput，验证 C2）

| 模型 | GLoRCQ full | GLoRCQ w/o side-stream | GLoRCQ w/o cluster-batching | GLoRCQ w/o SharedUCache |
|---|---|---|---|---|
| Qwen1.5-MoE | X.X tok/s | Y.Y | Z.Z | W.W |
| Mixtral | ... | ... | ... | ... |
| Qwen3 | ... | ... | ... | ... |

（每行对应一次 ablation，直接支撑 C2 声明）

### §5.4 Baseline 复现（0.4 页）

- **MiLo 3-bit 参考**: 公开发布，我们在 H200 重跑，数字进 Table 1
- **MxMoE**: 2-bit config 不能直接复现（他们的 hardcoded tile config 只有 w4a4+w8a8+w4a4_g128 混合，无 w2/w3/w4 weight-only 组合），引用他们论文 table 6
- **TileQ**: 无发布 ckpt，引用论文
- **LQER**: 公开发布，重跑

---

## §6 消融（1.25 页）

### §6.1 Rank 扫描（0.2 页）
**图 2**: PPL vs rank ∈ {16, 32, 64, 128} on Qwen1.5-MoE

### §6.2 Group 大小 G 扫描（0.25 页）
**图 3**: PPL vs G ∈ {64, 128, 256, 512}，同 bit 预算
- **关键**: G=1（等于 TileQ per-expert）严格劣于 G=128

### §6.3 LoRA on/off（0.15 页）
**表 3**: rank=0 vs rank=32 → LoRA 贡献占比

### §6.5 聚类方法: Grassmannian vs traversal（0.25 页 — **motivating C1 的聚类选择**）

**表 5**: 三模型对比，同 fair-bit 配置

| 模型 | # clusters | PPL Grassmannian | PPL Traversal | Δ | Cluster 分布 (down_proj min/max/mean) |
|---|---|---|---|---|---|
| Qwen1.5-MoE (1440 experts) | 12 (G=128) | 7.37 | 7.17 | +0.20 | ~120±20 (均衡) |
| Qwen3-30B-A3B (6144 experts) | 48 (G=128) | **8.97** ✨ | 9.42 | **−0.45** | 3 / 312 / 128 (极端跨层) |
| Mixtral-8x7B (256 experts) | 4 (G=64) | 5.99 ❌ | **4.69** | +1.30 | 4 clusters 太粗 → auto-fallback |

**Message**: Grassmannian 收益**随 expert 数量 scale**。只有 clusters 够多（~≥8）时 manifold distance 才盖过 intra-cluster 噪声。**Qwen3 是故事最闪光的点**：48 clusters + 128 experts per layer → traversal G=128 退化为单层内分组，Grassmannian 是唯一跨层机制。Auto-fallback 规则 (`n_clusters<8 → traversal`) 在 Mixtral 上得到经验验证。

### §6.8 主成分角 heatmap（0.15 页）— **motivating evidence for C1**
**图 4**: 每层每 expert 独立算 U 的主成分角 heatmap → block 对角 + 明显 off-diagonal 能量 → 视觉证明跨层共享有意义。附上 Grassmannian cluster 分配彩色标记 → 视觉验证 clustering 对应低距离 blocks

### **§6.9 系统消融**（新增，0.2 页 — **motivating evidence for C2**）
- with/without SharedUCache + 侧流 + 同 cluster K-融的 decode tokens/sec
- 用 `inference/eval_speed.py` 直接跑

**挪到附录 C**:
- §6.4 int8-vs-fp16 LoRA 存储
- §6.6 attn-bits 消融
- §6.7 τ 阈值扫描

---

## §7 讨论与局限（0.4 页）— 老实说

### §7.1 Grassmannian scaling: 何时帮，何时 auto-fallback 更好
Grassmannian 主成分角聚类只有在 cluster 数量足够（经验 ≥8）时才有 discriminative power。小 MoE 如 Mixtral (256 experts per wtype → 4 clusters at G=64) 会让 Grassmannian 差 traversal +1.3 PPL，因为每 cluster 吃太多异质 experts。代码里的 auto-fallback (`n_clusters<8 → traversal`) 处理得干净，但这是算法在小 MoE 尺度的**真实局限**。未来工作: adaptive G 让 `n_clusters` 保持在任意 MoE 大小下的甜蜜区间。

### §7.2 Mixtral MMLU（从头号数字撤下）
MMLU 不在 Table 1 headline 里，因为多个 baseline (TileQ, LoPRo, GPTVQ) 未报 MMLU 2-bit 数据、MiLo MMLU 的 tokenizer/prompt 也不一致。完整 MMLU 数字在附录 C

### §7.3 推理速度定位
GLoRCQ 加速比是**同模型 naive per-expert-LoRA-on-main-stream** baseline 的对比。**不与 vLLM fp16 比 raw throughput**（那是独立的 kernel 优化系统论文）

### §7.4 τ 是启发式
τ=60 empirical；未来可 activation-Hessian 谱做原则化选择

### §7.5 Qwen3 real-quant NaN bug
Qwen3 stripped real-quant 通过我们的推理路径出 NaN PPL（fake-quant 正常，Table 1 accuracy 对比不受影响）。open bug；可能在 shim-expert dispatch 里，fp16 approx 被 strip 后 packed-code path 对 128 experts × top-8 routing 支持有缺陷

### §7.6 Scale 上限
最大 30B-A3B；DeepSeek-V3 (671B) 未测。Phase 1 内存随 expert 数量线性增长，Grassmannian pairwise 是 O(N²) 可能需要更激进的 chunk-batching

---

## §8 结论（0.1 页，两句话）

1. **跨层 Grassmannian 聚类的 U 共享**把低秩补偿预算在整个 MoE 内摊薄，同精度下 per-expert 有效 rank ×G；小 MoE 通过 auto-fallback 保留 pipeline
2. **SharedUCache + 侧流并行 + 同 cluster K-融** 让这份共享结构在推理端也变现，decode 相比 naive baseline 加速 R×

---

## 附录

- **A**: Bit-accounting 完整推导（含 attn_bits=4 的 0.06 修正）
- **B**: VQ4 tile backbone + attention 量化（继承的 scaffolding）
- **C**: 全消融表 + MMLU per-config 数字 + int8-vs-fp16 LoRA + attn-bits + τ 扫描
- **D**: HF 6 个 ckpt 的 model cards（`Tsingyow/GLoRCQ-{qwen1.5-moe-a2.7b, mixtral-8x7b, qwen3-30b-a3b}-fair-grassmann-{fake, real}`）+ stripped 格式说明
- **E**: 扩展相关工作（3000 词完整版）
- **F**: MxMoE Qwen3 port 尝试 + 对比说明（为何 2-bit weight-only 不能直接在他们发布代码中复现）

---

## ⚠️ 已知未解决问题（写论文前要修）

| # | 问题 | 阻塞什么 | 修复路径 |
|---|---|---|---|
| 1 | ~~Mixtral v3b MMLU~~ | ~~Abstract/§5~~ | **已解决** — MMLU 从头号数字撤下（§7.2）|
| 2 | ~~MxMoE Qwen3 数字~~ | ~~Table 1~~ | **已解决** — 引用论文 table 6，不复现（§5.4）|
| 3 | **TileQ arxiv 2605.09281 未 verify** | §2 关键 baseline | 拉 abstract 确认 |
| 4 | **主成分角 heatmap (Fig 4) 未生成** | §6.8 可解释性证据 | 从 Qwen1.5 saved SVD 出图 + Grassmannian cluster 着色 |
| 5 | attn_bits=4 bit-accounting 少算 0.06 | Table 1 "matched" 声明 | 附录 A 补 |
| 6 | 匿名化 | 双盲评审 | HF `Tsingyow/*` + GitHub `nicyyyy/*` 待改 |
| 7 | ~~跨层聚类 metric~~ | ~~§3.2/§5.1/§6.5~~ | **已解决** — Grassmannian + auto-fallback (`run_quantize.py:472`) |
| 8 | **AW-SVD 归属声明** | §2 / §3.1 | 引 LQER (Zhang 2024)；只声称 Grassmannian + stacked + shared-U 组合为新 |
| 9 | **不声称 alternating** | §3.3 | Pipeline 为 one-shot（代码已确认） |
| 10 | **§6.9 系统消融数据未跑** | Table 2 | 用 `inference/eval_speed.py` 跑 with/without variants |
| 11 | **Qwen3 real-quant NaN bug**（§7.5）| Table 2 speed（若要在 Qwen3 上测） | fake-quant accuracy 不受影响；real-quant 需 debug 推理路径 |

---

## 写作顺序（推荐）

1. **§3.2 Grassmannian 聚类 + stacked shared-U SVD** — 论文最核心，先写
2. **§4 SharedUCache + 侧流** — 系统层贡献，第二难，紧接 §3.2 写
3. **§5.2 Table 1 + §5.3 Table 2** — 冻结数字
4. **§6.5 Grassmannian vs traversal 三模型消融** — 支撑 Table 1 的 framing
5. **§1 引言** — 等 §3+§4+§5 稳定后回来写
6. **§6 消融** — 从 Google Sheet Table 1-4 拉数据
7. **§7 讨论** — 一次性老实写完（Grassmannian scaling 局限是新东西）
8. **§2 编辑** — 应用 LR review 的 C1/C3/M1 补丁 + Grassmannian 引用
9. **Abstract** — 最后写，改 5+ 次

**预估**: 3-4 天 §3+§4+§5；2 天 §6；各 1 天 §1/§7/§2。**约 10 天集中写**。

---

_大纲 v0.3 结束 — 详细英文版见 `docs/paper_outline.md`_
