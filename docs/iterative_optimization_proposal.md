# 端到端迭代优化方案

## 背景：当前 Pipeline 的局限

当前 GLoRCQ 是线性单向的：

```
Stage 1: Q(W)           → Stage 2: 聚类 → Stage 3: U@SV^T → Stage 4: 写回
         ↑ 量化时不知道          ↑ 分组时不知道
           后续 LoRA 会补偿什么     重建质量如何
```

最终近似：`W_approx = Q(W) + U @ SV^T`

**问题**：Stage 1 量化的目标是最小化 `||W - Q(W)||_H`，但实际上 LoRA 会补偿一部分误差，真正的目标应该是最小化 `||W - Q(W) - U@SV^T||_H`。两者不一致导致次优解。

---

## 方案一：LoftQ 式完整迭代（效果最好）

### 原理

交替优化量化参数和 LoRA 参数：

```
初始化: W^(0) = W

Round k:
  1. 固定 U^(k-1)@SV^(k-1)，量化残差:
     Q^(k) = GPTQ(W - U^(k-1)@SV^(k-1))

  2. 固定 Q^(k)，更新 LoRA:
     E^(k) = W - Q^(k)(W - U^(k-1)@SV^(k-1))
     U^(k), SV^(k) = Grassmannian_SVD(E^(k))

最终: W_approx = Q^(K)(W - U^(K-1)@SV^(K-1)) + U^(K)@SV^(K)^T
```

### 实现关键

- **GPTQ 复用 Hessian**：Stage 1 结束后把 `H_inv` 保留在 CPU 内存，无需重新采样数据
  - 内存估算：attention 层 H_inv ≈ 1.5 GB（24层 × 4proj × 2048²× 4B），可接受
  - MoE 层 TurboQuant 无 Hessian，只能重新量化（或跳过迭代）
- **迭代轮数**：通常 2-3 轮收敛，收益递减

### 优缺点

| 优点 | 缺点 |
|------|------|
| 理论最优，attention 误差可大幅降低 | MoE（TurboQuant）无 Hessian，不易迭代 |
| 复用现有 GPTQ 代码 | 需要修改 Stage 1 保存 H_inv |
| 2-3 轮后收敛快 | 每轮多约 20% 时间（重量化 attention）|

---

## 方案二：多层 LoRA 叠加（Quick Win）

### 原理

不重新量化，只对残差多做几次 SVD 分解：

```
E^(1) = W - Q(W)                      ← Stage 3 第一层
U^(1)@SV^(1) = SVD(E^(1))

E^(2) = E^(1) - U^(1)@SV^(1)          ← 一阶残差
U^(2)@SV^(2) = SVD(E^(2))             ← Stage 3 第二层

最终: W_approx = Q(W) + U^(1)@SV^(1)^T + U^(2)@SV^(2)^T
```

### 实现

代码改动极小：Stage 3 多循环一次，对残差再做一次 SVD + 共享 U 分解。

### 优缺点

| 优点 | 缺点 |
|------|------|
| 代码改动 < 50 行 | 量化 Q(W) 没有被优化，天花板低 |
| 无需保存 H_inv | extra bits 翻倍（两套 U + SV）|
| MoE/Attention 都适用 | 第二层 LoRA 的 U 分摊效果有限 |

---

## 方案三：TurboQuant + LoRA 联合优化（最有潜力）

### 原理

当前 TurboQuant 对每个 MoE expert 独立量化，完全不考虑后续的 LoRA 补偿。更理想的做法是在量化时就联合考虑 LoRA：

```
当前:  min_{Q} ||W - Q(W)||²
目标:  min_{Q, U, SV_i} ||W_i - Q(W_i) - U @ SV_i^T||_H  (跨 expert 联合)
```

### 实现思路

**轻量版**（可行性高）：

1. Stage 1 TurboQuant 正常跑，得到 `Q(W)` 和残差 `E`
2. Stage 3 得到 `U@SV^T`，写回 `W_approx = Q(W) + U@SV^T`
3. **新增 Stage 1b**：对每个 expert，在 `W_approx` 基础上做一次 TurboQuant 微调：
   - 目标：`Q'(W_i - U@SV_i^T)`，即对"去掉 LoRA 后的残差"重新量化
   - 由于 TurboQuant 是 row-wise 的，可以快速重跑（无需 Hessian）
4. 更新残差，重新做一次 Stage 3

**完整版**（效果最好）：

引入 Hessian 到 TurboQuant：目前 TurboQuant 没有用 Hessian，加入 Hessian 权重后联合优化效果会更好，但实现复杂度大幅增加。

### 优缺点

| 优点 | 缺点 |
|------|------|
| 直接优化 MoE（模型主体） | TurboQuant 原本无 Hessian，联合优化需要新设计 |
| 轻量版实现可行 | 轻量版理论保证弱 |
| 针对性强 | 完整版改动大 |

---

## 实验优先级建议

| 优先级 | 方案 | 预期 PPL 改善 | 实现难度 | 时间成本 |
|--------|------|--------------|---------|---------|
| ★★★ | 方案一（2轮迭代，仅 attention） | ~0.1-0.2 | 中 | +30% 时间 |
| ★★★ | 方案三轻量版（TurboQuant 迭代） | ~0.3-0.5 | 中高 | +50% 时间 |
| ★★ | 方案二（双层 LoRA） | ~0.1-0.2 | 低 | +10% 时间 |
| ★ | 方案三完整版（联合 Hessian） | ~0.5+ | 高 | 研究级 |

---

## 当前实验状态参考

| 配置 | PPL | Total bits |
|------|-----|-----------|
| SOTA (rank32, rank_attn512) | 8.94 | 2.29 |
| +rank_down=128, iter=20 | **8.53** | 2.47 |
| +rank_down=512, iter=5 | 7.99 | 3.19 |
| TileQ (target) | 7.35 | — |
| LoPRo (target) | 7.52 | — |

当前最佳：`--rank 32 --rank_attn 512 --rank_down 128 --n_iter 20`
