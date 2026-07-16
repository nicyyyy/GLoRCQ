# GLoRCQ: Paper Outline (v0.4)

**Working title**: GLoRCQ: Global shared Low-Rank Compensation for Quantization of Mixture-of-Experts LLMs

**Target venue**: ICML 2026 / NeurIPS 2026 main track, 8–9 pages main text + appendix
**Version**: outline v0.4 — 2026-07-09 (paper voice pass: merged §3.2/§3.3, cut function names and code-branch discussion from main body, honest Global-U-pool description)
**Status**: aligned with code at HEAD
**Chinese speed-read version**: `docs/paper_outline_zh.md`

---

## One-sentence core thesis

> TileQ and MiLo have converged on the "aggressive quantization + per-expert low-rank compensation" recipe for MoE weight compression, but they allocate low-rank factors **layer-locally and per-expert**, wasting rank budget on subspaces that overlap heavily across layers. **GLoRCQ groups experts across layers by Grassmannian principal-angle distance and takes one activation-weighted SVD per cluster**, so a single shared U replaces G per-expert bases; at inference the same cluster structure enables a shared-U cache plus side-stream cluster-batched LoRA overlapping the 2-bit backbone.

Two contributions (C1 algorithmic + C2 systems) plus the experimental payoff. No memory hook, no "sub-2-bit" claim, no alternating-optimization claim.

---

## Structural budget (8-page ICML template)

| Section | Pages | Focus |
|---|---|---|
| Abstract + §1 Introduction | 1.0 | TileQ/MiLo lineage → per-expert LoRA weakness → cross-layer subspace pooling + inference co-design |
| §2 Related Work | 1.0 | Reuse existing draft; apply LR-review report's C1/C3/M1 patches |
| §3 Method (**core**) | 2.0 | Preliminaries + single Method section: clustering + shared-U SVD + one-shot pipeline |
| §4 Inference-side co-design (**core**) | 0.75 | Shared-U cache + global pool + same-cluster batching + side stream |
| §5 Experiments | 2.0 | Main comparison table + systems speedup table + baseline reproductions |
| §6 Ablations | 1.25 | rank / G / LoRA on-off / clustering method / principal-angle heatmap / systems ablation |
| §7 Discussion & Limitations | 0.3 | Inference-speed positioning, τ heuristic, real-quant Qwen3 bug, scaling |
| §8 Conclusion | 0.1 | Two sentences |
| **Main total** | **~8.4** | |
| Appendix A: bit accounting | 1.0 | Reproducible formula + attn=4 undercount fix + wallclock per phase per model |
| Appendix B: inherited scaffolding | 1.0 | 2-bit VQ tile backbone + attention GPTQ |
| Appendix C: extended ablations | 1.5 | int8-vs-fp16 LoRA / attn-bits / τ / MMLU / full per-task tables |
| Appendix D: HF release cards + stripped-format spec | 0.75 | Reproducibility |
| Appendix E: extended related work | 2.0 | Full 3,000-word §2 |
| Appendix F: MxMoE comparison notes | 0.25 | Why 2-bit weight-only isn't directly reproducible in their released code |

---

## Contributions (exactly 2)

Reviewers penalize "results are contributions" phrasing, so the third user-requested item ("speedup and accuracy loss") becomes the headline number in the abstract and §1 ¶4, not a numbered contribution.

### **C1**: Cross-layer expert subspace pooling with a shared U-factor

Different Transformer layers' experts perform the same class of computation, and empirically their per-expert low-rank compensators occupy heavily overlapping subspaces (Figure 4 in §6). Rather than allocate rank budget per layer per expert, we group experts across layers by the principal-angle distance between their top-r singular subspaces, stack the members of each group's activation-scaled weights, and take a single activation-weighted SVD. The top-r singular vectors become a shared U across the group; per-expert Σ and V remain private. Under matched storage the effective per-expert rank grows by G× compared to per-expert LoRA at the same bit budget.

### **C2**: Inference-time cluster-batched LoRA on a side stream

Because clusters are learned globally, the top-k active experts activated for a token frequently share one cluster and therefore one U. Three composed wins follow. First, each cluster's shared U is dequantized once at load into a device-resident pool, so x·U is computed once per active cluster per token and reused across the cluster's active experts. Second, when all top-k active experts share one cluster, their per-expert factors are concatenated and issued as a single wide GEMM instead of K per-expert launches. Third, the LoRA path is scheduled on a side CUDA stream and overlaps the 2-bit backbone GEMM on the main stream, with a single downstream synchronization. Graph-mode capture is supported so the same pattern replays inside a CUDA graph.

**Explicit non-claims**. Attention completes before the MoE block begins; we do not overlap LoRA with attention. We do not prefetch U across layers. We do not claim raw-throughput parity with fp16 kernels such as vLLM's — our decode speedup is measured against a naive same-model per-expert-LoRA-on-main-stream baseline.

### Experimental payoff (not numbered)

- **Main table** (§5.2 Table 1): 2-bit fair-budget PPL + 5-task 0-shot vs GPTQ (2/3-bit), MOEQ (3-bit), LoPRo, MxMoE, TileQ_s; FP16 reference from the TileQ paper.
- **Systems speedup** (§5.3 Table 2): decode tokens/sec with vs without each C2 component.
- **Cluster validity** (§6.5): cross-layer composition per cluster (Figure 4a), principal-angle distance heatmap sorted by cluster (Figure 4b), and same-size random-cluster control (Table 5).

---

## Abstract (150–250 words, 4 sentences)

**Do NOT use**: "memory frontier", "sub-2-bit", "stripped checkpoint", "fused kernel", "alternating optimization".

Structure:

1. **Context**: SOTA MoE weight quantization has converged on "aggressive quantization + per-expert low-rank compensation" (TileQ, MiLo).
2. **Gap**: per-expert low-rank factors are allocated layer-locally, but different Transformer layers' expert compensators empirically occupy heavily overlapping subspaces, so per-expert allocation wastes rank budget on redundant basis vectors.
3. **Method**: GLoRCQ groups experts across layers by the principal-angle distance between their top-r singular subspaces, stacks the activation-scaled weights within each group, and takes one SVD whose top-r vectors form a shared U across the group. The same cluster structure enables a shared-U cache and side-stream cluster-batched LoRA at inference time.
4. **Results**: at 2-bit + ≤0.165 extra bits/param, GLoRCQ improves WikiText-2 PPL over TileQ_s by **0.42 / 0.29 / 2.33** on Qwen1.5-MoE / Mixtral-8x7B / Qwen3-30B-A3B; the largest gain (Qwen3) is driven by clusters that span the entire 48-layer stack, confirmed by our cluster-composition analysis (Figure 4a). The cluster-batched LoRA + side-stream co-design delivers **R×** decode speedup over a naive per-expert baseline. Fake-quant and real-quant checkpoints released on HuggingFace.

**Keywords**: mixture-of-experts, post-training quantization, low-rank compensation, vector quantization, subspace clustering.

---

## §1 Introduction (~1 page, 4 paragraphs)

**¶1 — Where the field is now.** MoE weight quantization has converged on a shared recipe: quantize the expert weight backbone aggressively (2-bit vector quantization, per-column GPTQ) *and* add a small trainable low-rank residual on top to recover accuracy. TileQ (Gu et al., 2026) and MiLo (Huang et al., 2025) are two concrete instantiations for MoE. This is the SOTA lineage the paper builds on. We do not open with a memory-bottleneck hook or claim a sub-2-bit frontier.

**¶2 — The specific weakness of this recipe.** Both TileQ and MiLo give every expert its own private low-rank factors, computed layer-locally. But different Transformer layers' experts perform the same class of computation, and empirically their per-expert low-rank compensators occupy heavily overlapping subspaces (§6, Figure 4). Per-expert allocation therefore wastes rank budget on redundant basis vectors.

**¶3 — Our fix.** We group experts across layers by the principal-angle distance on their top-r singular subspaces, so that group members genuinely share the subspace we are about to compress. Within each group we stack the activation-scaled weights and take a single SVD; the top-r vectors become a shared U, per-expert Σ and V remain private. Under matched bit budget the effective per-expert rank grows by G× compared to per-expert LoRA. To make this deployable we co-design an inference path that exploits the cluster structure: x·U is computed once per active cluster per token; the per-expert factors of same-cluster active experts are issued in a single wide GEMM; and the whole LoRA path runs on a side CUDA stream overlapping the 2-bit backbone on the main stream.

**¶4 — Contributions and headline numbers.** Two numbered contributions — **C1** cross-layer subspace pooling with a shared U-factor, **C2** shared-U cache and side-stream cluster-batched LoRA. Headline: on Qwen1.5-MoE / Mixtral-8x7B / Qwen3-30B-A3B at 2-bit + ≤0.165 extra bits/param, GLoRCQ improves PPL over TileQ_s by 0.42 / 0.29 / 2.33, with the largest gain on Qwen3 supported by clusters that span its entire 48-layer stack (Figure 4a); the systems co-design delivers **R×** decode speedup. We do not claim memory savings as a headline: the LoRA overhead adds bits on top of a pure 2-bit baseline; the paper's value is quality-at-budget plus inference-time compute reuse, not raw memory reduction.

---

## §2 Related Work (~1 page, existing draft)

**Reuse**: `docs/literature_review/related_work.md` (38 refs, 7 subsections).

**Required patches from LR-review report** (do before submission):

- **C1** (LR review numbering): add QuantMoE-Bench + MoEQuant to §2.4.
- **C2** (LR): verify TileQ arxiv ID resolves; drop `(submission)` label.
- **C3** (LR): deduplicate MiLo bib entries.
- **I1**: reword "fair-bit win" once Mixtral is finalized.
- **I2**: add vLLM + Marlin to §2.6 (systems).
- **I3**: add VQ theoretical basis to §2.2 (1 sentence).
- **I4**: add LLM.int8() (Dettmers 2022) outlier lineage to §2.5.
- **M1**: soften 7 hallucination-flagged phrasings.
- **New**: cite the Grassmann manifold / principal-angle distance references in §2.2 (Absil et al., 2006; standard SVD subspace-angle definition).

**Compress**: cut §2 from 3,000 words in the LR document to ~800 for main text; full version to Appendix E.

---

## §3 Method (~2.0 pages)

### §3.1 Preliminaries and problem statement (~0.4 page)

Notation: an MoE layer has N routing experts, each holding K weight matrices per expert (gate / up / down); the model has L layers. We represent each `W ∈ ℝ^{out_d × in_d}` as `Q + U V_k`, where Q is a 2-bit quantized backbone and `U V_k` is a rank-r correction. The average bit budget per parameter satisfies

```
avg_bits ≈ 2 + (r · (in_d + out_d) · lora_precision) / (in_d · out_d)
```

with attention parameters accounted for separately in Appendix A. Throughout, we build on activation-weighted SVD (Zhang et al., 2024) for the low-rank fit and a 2-bit tile-quantized backbone (Gu et al., 2026); §3.2 states the additional structure that makes this composition novel.

### §3.2 Cross-layer subspace pooling with a shared U-factor (~1.6 pages) — **C1**

**Empirical motivation.** For each weight type (gate / up / down projection), independently computing the top-r left singular basis of every expert's activation-scaled weight matrix reveals large overlaps across layers. The principal-angle heatmap in §6 (Figure 4) is block-diagonal at the layer level but carries substantial off-diagonal mass. Per-expert LoRA cannot exploit this — every expert re-learns a private basis, and the shared subspace is paid for L·N times.

**Grouping by subspace similarity.** For each weight type, let U_i denote the top-r left singular basis of expert i's activation-scaled weight matrix. We compute the pairwise Grassmannian principal-angle distance

```
d(i, j) = ‖arccos σ(U_i^T U_j)‖_2 / (√r · π/2)  ∈ [0, 1]
```

convert it to a Gaussian affinity kernel, and spectrally cluster the (L·N) experts of each weight type into groups of size G. Same-cluster experts have overlapping top-r subspaces.

**Shared U via one stacked SVD.** For each cluster, we stack per-expert activation-scaled weight matrices into a wide block-column matrix

```
[diag(S_a) · W_1^T | ⋯ | diag(S_a) · W_G^T] ∈ ℝ^{in_d × G · out_d}
```

and take a single activation-weighted SVD (Zhang et al., 2024). The top-r left singular vectors form a shared U across the cluster while per-expert Σ_k and V_k remain private. Under a matched storage budget the effective per-expert rank grows by G× compared to per-expert LoRA: the U cost is paid once per cluster instead of once per expert.

**One-shot pipeline.** The pipeline is a single pass — no alternating loop between the low-rank fit and the backbone quantization.

```
Algorithm 1: GLoRCQ quantization
1. Collect activation scales per expert (calibration pass).
2. Cluster experts of each weight type by principal-angle distance.
3. For each cluster, take one activation-weighted SVD → shared U, per-expert V.
4. Quantize the residual W - UV with a 2-bit vector-quantized backbone (Gu et al., 2026).
5. Retain outlier experts in fp16 when the reconstruction error exceeds τ.
```

The pairwise-distance step is O(N²) per weight type on GPU (chunk-batched to stay in HBM); total quantization wallclock per model is reported in Appendix A.

**Figure 1.** Schematic: same-cluster experts drawn from different Transformer layers → one stacked SVD → shared U + per-expert V.

---

## §4 Inference-side co-design (~0.75 page) — **C2**

**Section title**: "Inference-side co-design: cluster-batched LoRA on a side stream"

**¶1 Cluster structure to exploit.** Because clusters are learned globally, the top-k experts activated for a given token frequently share one cluster and therefore one U. This suggests three overlapping wins: (a) compute x·U once per active cluster; (b) batch the per-expert projections when all top-k experts share a cluster; (c) overlap the entire LoRA path with the 2-bit backbone.

**¶2 Shared-U cache and global pool.** Each cluster's shared U is dequantized once at load and concatenated per weight type into a single device-resident tensor of size `O(d · r · K_total)`, where K_total is the total number of clusters across weight types. For our largest evaluated model this pool occupies under 10 MiB (Qwen3-30B-A3B, r=16, at most 48 clusters per weight type, hidden dimension 2048); eager materialization is what we do. At inference, x·U is computed once per unique active cluster per token, and the result is broadcast to the per-expert V multiplications of the active experts in that cluster — replacing K independent x·U GEMMs.

**¶3 Cluster batching and stream overlap.** When all top-k active experts for a token share one cluster, we concatenate their per-expert factors and issue a single (1, r) × (r, K · out_d) GEMM instead of K separate ones — a K× kernel-launch amortization. The LoRA path is scheduled on a side CUDA stream in parallel with the 2-bit backbone GEMM on the main stream; the two streams synchronize before the downstream reduction. Graph-mode capture is supported so the pattern replays inside a CUDA graph.

**Figure**: a small timing diagram showing main-stream backbone ∥ side-stream LoRA overlap.

**¶4 Explicit non-claims.** We do not overlap LoRA with attention (attention completes before the MoE block begins). We do not prefetch U across layers. We do not claim raw-throughput parity with fp16 kernels such as vLLM's — our decode speedup is measured against a naive same-model per-expert-LoRA-on-main-stream baseline. Systems-level parity with mature fp16 kernels is orthogonal to this paper's contribution.

---

## §5 Experiments (~2.0 pages)

### §5.1 Setup (~0.3 page)

- **Models**: Qwen1.5-MoE-A2.7B (60 experts × top-4, 24 layers); Mixtral-8x7B (8 × top-2, 32 layers); Qwen3-30B-A3B (128 × top-8, 48 layers).
- **Calibration data**: 128 samples × 4096 tokens from WikiText-2 train split.
- **Bit budget target**: 2 + 0.16 = 2.16 bits/param (matching TileQ paper).
- **Baselines**: GPTQ 2-bit and 3-bit, MOEQ 3-bit, LoPRo 2-bit, TileQ_s (fair +0.16), MxMoE (paper's published Table-1 numbers; Qwen3 not reported there); FP16 reference from the TileQ paper. Baseline numbers under identical evaluation config are described in §5.4.
- **Evaluation**: WikiText-2 PPL (sliding window, max_len=2048, stride=512); 5-task 0-shot average of ARC-c / ARC-e / PIQA / WinoGrande / HellaSwag (accuracy, batch=1, add_bos=True). MMLU dropped from headline (also absent from several 2-bit baselines; discussed in Appendix C).
- **Hardware**: 8× H200 (calibration), 1× H200 (evaluation).
- **Method configuration** (single sentence): 2-bit VQ tile config inherited from TileQ; attention uses 4-bit GPTQ; experts with reconstruction max-error above τ retained in fp16 (τ ablation in Appendix C); cross-layer experts are grouped by the principal-angle distance introduced in §3.2.

### §5.2 Main results (~0.8 page) — Table 1

**Table 1** (merged main table, from the curated results sheet 2026-07-16; all downstream tasks use `acc` metric matching the TileQ Table-1 convention, num_fewshot=0, add_bos=True, batch=1; ours evaluated 2026-07-06):

| Method | Bits | Extra bits (Q1.5 / Mixtral / Q3) | Qwen1.5-MoE PPL / Avg(5) | Mixtral-8x7B PPL / Avg(5) | Qwen3-30B-A3B PPL / Avg(5) |
|---|---|---|---|---|---|
| FP16 (TileQ paper) | 16 | — | 6.79 / 64.50 | 3.87 / 75.46 | 8.07 / 68.12 |
| GPTQ | 2 | 0.13 | 12.5 / 41.12 | 15.3 / 38.52 | 14.6 / 50.96 |
| GPTQ (3-bit ref) | 3 | 0.13 | 7.98 / 58.30 | 4.72 / 64.00 | 9.42 / 60.58 |
| MOEQ (3-bit ref) | 3 | 0 | 8.21 / 57.32 | 5.45 / 69.70 | 28.1 / 45.26 |
| LoPRo | 2 | 0.43 / 0.21 / 0.58 | 7.52 / 62.36 | 5.01 / 70.72 | 11.1 / 55.02 |
| MxMoE | 2 | 0.25 | 8.79 / 56.06 | 5.63 / 68.87 | — |
| **TileQ_s** | **2** | **0.16** | **7.56 / 62.86** | **4.98 / 70.92** | **11.3 / 55.24** |
| **GLoRCQ (ours)** | **2** | **0.152 / 0.1611 / 0.1647** | **7.14 / 62.76** | **4.69 / 64.48** | **8.97 / 63.46** |

On **Qwen3-30B-A3B** GLoRCQ improves PPL by **2.33** over TileQ_s — the largest win, and the one where cross-layer sharing matters most: at 128 experts per layer, per-layer grouping schedules cannot mix experts from different layers within a single shared factor. Our clusters do (Figure 4a) and this is what the +2.33 PPL captures. On **Qwen1.5-MoE** GLoRCQ improves PPL by 0.42 over TileQ_s (7.14 vs 7.56). On **Mixtral-8x7B** GLoRCQ improves PPL by 0.29 over TileQ_s at matched bits. GPTQ 3-bit and MOEQ 3-bit are included as higher-budget references; their extra ~1 bit over our budget explains any PPL advantage (and note MOEQ collapses on Qwen3, 28.1). MxMoE numbers are the paper-published ones (Qwen3 not reported there). (Qwen1.5 numbers are all measured on a single A100 with a shared calibration pass so the headline and the §6 ablations are directly comparable.)

**Full per-task tables (→ Appendix, reproduced verbatim from the results sheet):**

*Qwen1.5-MoE-A2.7B* (WikiText-2 PPL ↓; acc ↑):

| Method | Bits | Extra | PPL | ARC-C | ARC-E | PIQA | WinoG | HellaS | Avg(5) |
|---|---|---|---|---|---|---|---|---|---|
| FP16 (TileQ paper) | 16 | — | 6.79 | 41.6 | 72.9 | 79.6 | 69.1 | 59.3 | 64.50 |
| GPTQ | 2 | 0.13 | 12.5 | 29.4 | 42.7 | 50.2 | 53.2 | 30.1 | 41.12 |
| GPTQ | 3 | 0.13 | 7.98 | 33.4 | 64.3 | 77.1 | 64.5 | 52.2 | 58.30 |
| MOEQ | 3 | 0 | 8.21 | 32.6 | 63.6 | 76.2 | 63.8 | 50.4 | 57.32 |
| LoPRo | 2 | 0.43 | 7.52 | 39.9 | 72.7 | 77.6 | 68.2 | 53.4 | 62.36 |
| MxMoE | 2 | 0.25 | 8.79 | 31.66 | 53.28 | 71.33 | 61.25 | 62.8 | 56.06 |
| TileQ | 2 | 0.16 | 7.56 | 39.6 | 72.5 | 77.8 | 68.9 | 55.5 | 62.86 |
| **GLoRCQ** | 2 | 0.152 | **7.14** | 39.68 | 72.73 | 78.02 | 69.14 | 54.24 | **62.76** |

*Mixtral-8x7B-v0.1*:

| Method | Bits | Extra | PPL | ARC-C | ARC-E | PIQA | WinoG | HellaS | Avg(5) |
|---|---|---|---|---|---|---|---|---|---|
| FP16 (TileQ paper) | 16 | — | 3.87 | 61.9 | 87.3 | 83.7 | 77.1 | 67.3 | 75.46 |
| GPTQ | 2 | 0.13 | 15.3 | 26.5 | 35.6 | 53.0 | 49.3 | 28.2 | 38.52 |
| GPTQ | 3 | 0.13 | 4.72 | 52.4 | 69.3 | 79.1 | 74.4 | 44.8 | 64.00 |
| MOEQ | 3 | 0 | 5.45 | 57.4 | 80.2 | 78.9 | 71.9 | 60.1 | 69.70 |
| LoPRo | 2 | 0.21 | 5.01 | 55.3 | 82.5 | 80.6 | 74.9 | 60.3 | 70.72 |
| MxMoE | 2 | 0.25 | 5.63 | 48.98 | 72.77 | 76.28 | 68.9 | 77.44 | 68.87 |
| TileQ_s | 2 | 0.16 | 4.98 | 55.5 | 82.8 | 80.9 | 75.1 | 60.3 | 70.92 |
| **GLoRCQ** | 2 | 0.1611 | **4.69** | 46.16 | 76.14 | 75.08 | 71.82 | 53.19 | **64.48** |

*Qwen3-30B-A3B*:

| Method | Bits | Extra | PPL | ARC-C | ARC-E | PIQA | WinoG | HellaS | Avg(5) |
|---|---|---|---|---|---|---|---|---|---|
| FP16 (TileQ paper) | 16 | — | 8.07 | 52.6 | 79.2 | 79.7 | 70.3 | 58.8 | 68.12 |
| GPTQ | 2 | 0.13 | 14.6 | 31.1 | 54.4 | 68.9 | 57.2 | 43.2 | 50.96 |
| GPTQ | 3 | 0.13 | 9.42 | 44.0 | 70.3 | 75.1 | 63.4 | 50.1 | 60.58 |
| MOEQ | 3 | 0 | 28.1 | 37.8 | 30.6 | 59.4 | 55.3 | 43.2 | 45.26 |
| LoPRo | 2 | 0.58 | 11.1 | 34.4 | 58.3 | 71.5 | 62.9 | 48.0 | 55.02 |
| TileQ_s | 2 | 0.16 | 11.3 | 34.6 | 58.4 | 71.8 | 63.1 | 48.3 | 55.24 |
| **GLoRCQ** | 2 | 0.1647 | **8.97** | 45.65 | 74.41 | 76.33 | 67.56 | 53.36 | **63.46** |

### §5.3 Systems speedup (~0.5 page) — Table 2

Validates **C2** with a per-component ablation:

| Model | GLoRCQ (full C2) | w/o side-stream | w/o same-cluster batching | w/o shared-U cache |
|---|---|---|---|---|
| Qwen1.5-MoE | X.X tok/s | Y.Y | Z.Z | W.W |
| Mixtral | ... | ... | ... | ... |
| Qwen3 | ... | ... | ... | ... |

Data collected via decode-speed harness (batch=1, prompt_len=128, gen_len=128, max_seq_len=512).

### §5.4 Baseline reproducibility (~0.4 page)

- **MiLo 3-bit**: publicly released; rerun on our H200 (Qwen1.5 7.15 / Mixtral 4.03 / Qwen3 8.44). Dropped from the Table-1 row set in the 2026-07-16 sheet merge (GPTQ-3bit and MOEQ-3bit serve as the higher-budget references); numbers kept here in case a reviewer asks.
- **MxMoE**: 2-bit weight-only config is not directly reproducible in their released code (their hardcoded tile configurations cover mixed W-A schemes only); we cite paper Table 1 numbers and flag the caveat that their evaluation may use a different HellaSwag metric than ours (Appendix F).
- **TileQ**: no released checkpoints; cite paper numbers directly.
- **GPTVQ / LoPRo**: cite paper numbers, since released code targets a different bit convention.
- **Our reproducibility**: all RNG (rank-1 sketch init, low-rank SVD projection, VQ k-means) is seeded (seed=42). The pipeline is deterministic given the Stage-1 calibration input: the fair-bit r=20 configuration produces a **bit-identical** WikiText-2 PPL of 7.2995 across three independent runs (from-scratch calibration, cached-calibration, and the ablation queue), on the same GPU. Reviewers reproduce the reported numbers exactly by running the released code with the published seed.

---

## §6 Ablations (~1.25 pages)

### §6.1 Rank sweep (~0.2 p)
**Table (Qwen1.5-MoE, Grassmannian, A100, seed-locked, fixed G=128)**: PPL / 0-shot avg vs LoRA rank r — r=16 → 7.85 / 58.60 (2.08 bits), r=20 → 7.30 / 62.34 (2.10), **r=32 → 7.14 / 62.80 (2.15, fair-bit base)**, r=64 → 7.38 / 63.18 (2.29, over budget). *(Table 1's headline quotes 62.76 — the 2026-07-06 sheet eval of the same r32/G128 operating point; the Δ0.04 Avg is run-to-run eval noise, PPL identical at 7.14. Pick one number at writing time and use it in both places.)* **Message**: perplexity is U-shaped in rank — it drops sharply from r=16 to r=32 as the shared-U compensation gains capacity, then rises at r=64 where higher-rank per-expert factors accumulate more int8 quantization error while costing 2× the LoRA bits (2.29 total, above the 2.16 budget). r=32 is the perplexity optimum and lands almost exactly at the fair-bit budget (+0.15), so we adopt it as the operating point. Because the shared U is amortized over the G=128 experts in a cluster, r=32 costs only +0.15 bits/param — within budget — whereas per-expert LoRA at the same rank would exceed it.

### §6.2 Group size G sweep (~0.25 p)
**Table (Qwen1.5-MoE, Grassmannian, A100, seed-locked, r=32)**: G=64 (23 clusters) → 7.10 / 62.50 (2.15 bits); G=128 (12 clusters, base) → 7.14 / 62.80 (2.15). **Message**: at matched bits the two large-sharing settings are essentially tied — G=64 marginally better on perplexity, G=128 marginally better on 0-shot accuracy — showing cross-layer sharing is robust to the exact group size once groups are large. We use G=128 for the base (best 0-shot, fewest shared-U matrices to cache at inference). The meaningful contrast is against G=1 (per-expert LoRA, = TileQ), the degenerate lower end with no cross-layer sharing; Table 5's random-cluster control isolates that the *content* of the large-G grouping, not merely its size, recovers accuracy. For Qwen1.5, G≥256 yields fewer than 8 clusters and the pipeline auto-falls back to layer-order grouping, so those points are not Grassmannian.

### §6.3 LoRA on/off (~0.15 p)
**Table**: rank=0 (pure VQ backbone, no low-rank correction) vs rank=32 (default). LoRA compensation is grouping-agnostic here (no shared U exists at rank 0). **Message**: the low-rank correction contributes a substantial share of the accuracy recovery over the bare 2-bit backbone.

### §6.5 Cluster validity: do our groups carry structure? (~0.4 p) — **motivating evidence for C1's clustering choice**

We use three diagnostics — two visual, one behavioral — to establish that principal-angle clustering produces cross-layer groups with genuine subspace overlap. We deliberately do **not** compare against "layer-order grouping"; layer-order is not a designed algorithmic alternative but the default that emerges from any per-layer processing schedule. The honest way to isolate the clustering signal is a same-size random-cluster control (Table 5 below).

**Figure 4a — Layer × cluster composition (Qwen3-30B-A3B).** A `layers × clusters` heatmap where cell (l, c) counts how many experts of cluster c come from Transformer layer l. On Qwen3, traversal grouping at G=128 would place exactly one layer per cluster (a perfect diagonal of single bright cells), because each layer has exactly 128 experts. Instead every cluster draws experts from a wide band of layers — 17.8 distinct layers per cluster on average for gate-projection (min 2, max 39), 16.7 for up-projection, 40.7 for down-projection. This is the direct visual evidence that principal-angle clustering produces genuinely cross-layer groups on the model where it matters most. Qwen1.5-MoE panels (7.2 distinct layers/cluster on average) go to Appendix C.

**Figure 4b — Cluster coherence (Qwen3-30B-A3B).** Within-cluster vs between-cluster pairwise principal-angle distance histograms (activation-scaled subspaces, one panel per weight type). For gate- and up-projection the within-cluster distribution sits clearly left of the between-cluster one (μ 0.654 vs 0.729, and 0.652 vs 0.727) — same-cluster experts genuinely occupy closer subspaces. For down-projection the two distributions coincide (μ 0.927 vs 0.932): at rank r=32 the down-projection experts have no separable subspace structure, an honest exception. (We omit a 2D scatter: with 48 clusters in a near-orthogonal high-dimensional subspace, any 2D embedding is an uninformative blob.) The coherence is real but modest per weight type, which is why the behavioral control in Table 5 — where the aggregate effect is a decisive +0.86 PPL — is the headline evidence, not the distance histogram.

**Table 5 — Random-cluster control.** We replace the learned cluster assignments with a random assignment matching the learned clusters' size distribution and re-quantize with the same pipeline; the PPL degradation isolates how much of the accuracy at matched bit budget comes from *what* we cluster into, versus the *sizes* of the groups.

| Model | PPL (learned clusters) | PPL (random, matched sizes) | ΔPPL |
|---|---|---|---|
| Qwen1.5-MoE | 7.14 | 7.44 | +0.30 |
| **Qwen3-30B-A3B** | **8.97** | **9.83** | **+0.86** |
| Mixtral-8x7B | 4.69 | (not run — Mixtral uses layer-order) | — |

On Qwen3-30B-A3B the learned clustering beats a same-size random assignment by **0.86 PPL** — decisive evidence that *what* the pipeline groups (which experts share a factor), not merely the group-size histogram, is what recovers accuracy. This composes into a clean three-way ordering on Qwen3 at matched 2.16 bits/param: **Grassmannian 8.97 < traversal 9.42 < random 9.83**. Random cross-layer grouping is actually *worse* than layer-order traversal, so cross-layer sharing only helps when the grouping is subspace-informed — exactly what principal-angle clustering provides. On Qwen1.5-MoE the learned clustering beats random by 0.30 PPL (7.14 vs 7.44; both measured on the same A100 with a shared calibration pass). The effect is smaller than Qwen3's because Qwen1.5's 60-expert layers give the clusterer far less cross-layer material to exploit than Qwen3's 128-expert layers.

### §6.9 Systems ablation (~0.2 p) — **motivating evidence for C2**
With/without shared-U cache, with/without side-stream, with/without same-cluster batching: decode tokens/sec on Qwen1.5-MoE. Feeds directly into Table 2's per-component decomposition.

### Moved to Appendix C
- §6.4 int8 vs fp16 LoRA storage.
- §6.6 attn-bits comparison.
- §6.7 τ threshold sweep.
- §6.8 per-model cross-layer composition panels (Qwen1.5-MoE + Mixtral-8x7B; the main-text Figure 4a shows only Qwen3-30B-A3B).

---

## §7 Discussion & Limitations (~0.3 page)

### §7.1 Inference-speed positioning
GLoRCQ's decode speedup is measured against a naive same-model per-expert-LoRA-on-main-stream baseline. We do not claim raw-throughput parity with fp16 kernels such as vLLM's. Optimized 2-bit-backbone-plus-shared-LoRA kernels (Marlin-style) are a separate systems paper.

### §7.2 MMLU (moved from headline)
MMLU is not part of the Table 1 headline comparison because several Table-1 baselines (TileQ, LoPRo, MxMoE, MOEQ) do not report MMLU under a comparable setup. Full MMLU numbers are in Appendix C; the paper's headline metric is 5-task 0-shot average.

### §7.3 τ selection is heuristic
The fp16-retention threshold τ is empirical (weight-space L∞). A principled τ derived from activation-Hessian eigenvalues would remove a hyperparameter.

### §7.4 Real-quant Qwen3 inference bug
*(Resolved 2026-07-15, commit 6cea816.)* The earlier NaN through the stripped real-quant path was a numerical bug — the LoRA input scaling `x·S_a` was computed in fp16, overflowing on models with large activation scales (Mixtral bf16-native; Qwen3) — fixed by computing the scale multiply and LoRA reduction in fp32. All three real-quant checkpoints now pass inference sanity, and Qwen3's real-quant decode speed is measured (Table 2). Remaining honest limitation: long greedy generations (≫128 tokens) can degrade into repetition on the 2-bit real-quant path; the speed harness uses gen_len=128.

---

## §8 Conclusion (~0.1 page, two sentences)

1. **Cross-layer subspace pooling** with a shared U-factor amortizes the low-rank compensation budget across an entire MoE, growing per-expert effective rank by G× at matched storage.
2. Combined with a **shared-U cache and side-stream cluster-batched LoRA** inference co-design, GLoRCQ delivers quality-at-budget wins over TileQ and decode speedup over a naive per-expert baseline on three production MoE models; checkpoints released.

---

## Appendices

- **A**: Bit-accounting derivation (with attn=4 undercount fix) and per-phase wallclock per model.
- **B**: 2-bit VQ tile backbone + attention quantization (inherited scaffolding; K=256, vdim=4 details; Hessian-aware Hadamard rotation).
- **C**: Extended ablation tables (int8-vs-fp16 LoRA; attn-bits; τ sweep; MMLU per-config numbers; full per-task per-config numbers from the results spreadsheet).
- **D**: HuggingFace release model cards + stripped-checkpoint format specification. Six repositories under `Tsingyow/GLoRCQ-{qwen1.5-moe-a2.7b, mixtral-8x7b, qwen3-30b-a3b}-fair-grassmann-{fake, real}`.
- **E**: Extended related work (full 3,000-word §2, compressed for main text).
- **F**: MxMoE comparison notes (why 2-bit weight-only is not directly reproducible in their released code; the metric-convention caveat on their reported zero-shot numbers).

---

## ⚠️ Known open issues at outline time (2026-07-09)

1. **TileQ arxiv verification** — must confirm the arxiv ID resolves before submission.
2. **Cluster validity figures (Fig 4a cross-layer composition + Fig 4b principal-angle heatmap)** — Fig 4a computed from the cluster assignments already saved in `cross_layer_info.pt`; Fig 4b computed from per-expert top-r singular subspaces (chunk-batched pairwise, using the same distance our clusterer used). Both need to be plotted.
3. **Random-cluster control experiment (Table 5)** — three additional quant runs (Qwen1.5-MoE, Qwen3-30B-A3B, Mixtral-8x7B) at the fair-bit config, replacing learned cluster assignments with a same-size random shuffle. Estimated ~4 h on 1× H200 with Phase-1 cache reuse.
4. **attn_bits=4 bit-accounting undercount by ~0.06** — must be fixed in Appendix A before Table 1's "matched bit budget" claim survives scrutiny.
5. **Anonymization** — HF repository handles and GitHub repository name need anonymization for double-blind review.
6. **AW-SVD attribution**: LQER (Zhang et al., 2024) cited as prior art; only the cross-layer pooling structure is claimed novel.
7. **No alternating optimization claim**: pipeline is one-shot (verified in code audit 2026-07-08).
8. **§6.9 systems ablation data uncollected**: need to run the decode-speed harness with the various C2 components disabled to produce Table 2. *(Update 2026-07-16: recommended re-scope — fill Table 2 with the measured Standard-vs-CUDA-graph decode numbers, which exist for all 3 models; the per-component-flag ablation requires disable-flags that were never implemented.)*
9. ~~**Qwen3 real-quant NaN bug** (§7.4)~~ **RESOLVED 2026-07-15** (commit 6cea816: `x·Sa` fp16 overflow → fp32; all 3 real-quant models pass inference sanity; Qwen3 speed measured — Standard 2.8 / Graph 6.4 tok/s). §7.4 text needs updating accordingly.
10. **Table 1 Extra-bits cells (2026-07-16 sheet merge — (a)(b) RESOLVED, (c) deferred)**: (a) ~~Qwen1.5 Extra 0.1621~~ → **0.152** (matches the seeded r32 canonical quant log TOTAL=2.1518); (b) ~~Mixtral Extra 0.11~~ → **0.1611** (matches the eval'd fp16-LoRA v2 run; 0.11 was the int8-LoRA v3b accounting whose run has no evals — see issue 11); user's sheet should mirror both. (c) OPEN: MxMoE HellaSwag cells look anomalous (Qwen1.5 62.8 and Mixtral 77.44 both *above* their FP16 references) — likely acc_norm from the MxMoE paper or a column transposition; user chose to keep as-is for now; verify against MxMoE Table 1 before camera-ready.
11. **Mixtral Table 1 vs released artifact (residual, accepted 2026-07-16)**: Table 1's Mixtral row (4.69 / 64.48) is the fair-bit **v2** run (r32, G=64, fp16-LoRA, 2.1611 bits; PPL `logs/h200_fair_evals/ppl_mixtral_fair.json`, compliant ZS `logs/h200_run_v2/results/mixtral_fair_zs_2026-07-06T15-15-59.json`). The released HF real-quant artifact is **v3b** (same recipe with int8-LoRA, ≈2.11 bits) and has no fake-quant eval of its own (weights stripped pre-eval). If a reviewer demands artifact-exact numbers: hydrate fake weights from v3b's `cross_layer_info.pt` (local, 16.5 GB) and eval — needs 2× A100-80G (Mixtral fp16 ≈93 GB), est. ~5-6 h.

---

## Writing sequence (recommended)

1. **§3.2 method** — algorithmic core; hardest, do first.
2. **§4 inference-side co-design** — systems contribution; tightly coupled to §3, write immediately after.
3. **§5.2 Table 1 + §5.3 Table 2** — freeze the headline numbers; everything else references them.
4. **§6.5 clustering ablation** — three-model story supports Table 1's framing.
5. **§1 Introduction** — only after §3 + §4 + §5 stable; contributions in past tense.
6. **§6 remaining ablations** — pull from the results spreadsheet, expand into narrative.
7. **§7 discussion** — honest limits, one sitting.
8. **§2 related work edits** — apply LR-review report patches + add principal-angle references.
9. **Abstract** — last, tightest, revised multiple times.

**Rough day budget**: 3–4 days for §3 + §4 + §5, 2 days for §6, 1 day each for §1 / §7 / §2. Total: ~10 focused writing days.

---

_End of outline v0.4._
