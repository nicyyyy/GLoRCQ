# `exp/cluster/` — Qwen1.5-MoE cluster validity experiment

Artifacts and scripts backing §6.5 of `docs/paper_outline.md` (paper outline v0.4):
**Figure 4a** (per-cluster cross-layer composition), **Figure 4b** (principal-angle
distance heatmap sorted by cluster), and **Table 5** (same-size random-cluster PPL
control). All artifacts target Qwen1.5-MoE-A2.7B; other models are Appendix C.

## Files

Paper uses **Qwen3-30B-A3B** for the main Fig 4a / 4b / Table 5 (strongest cross-layer
case: 128 experts/layer means traversal at G=128 is strictly one-layer-per-cluster,
so any cross-layer grouping is a clear departure). Qwen1.5-MoE versions are Appendix C.

Figure scheme (v2, chosen 2026-07-09 for readability):
- **Fig 4a** = layer × cluster composition heatmap (cell = # experts from layer l in cluster c).
  Wide vertical spread per column = cross-layer. Replaces the earlier hard-to-read stacked bars.
- **Fig 4b** = 2D MDS scatter (colored by cluster) + within/between distance histogram.
  Replaces the earlier dense 1440² heatmap.

| File | Purpose |
|---|---|
| `plot_fig4a.py` / `plot_fig4a_qwen3.py` | Layer × cluster heatmap from `cross_layer_info.pt` assignments (Qwen1.5 / Qwen3). |
| `compute_D_grass.py` / `compute_D_grass_qwen3.py` | Pairwise Grassmannian distance matrix. Qwen1.5 reads the Phase-1 cache (activation-scaled); Qwen3 reads the fp16 base directly (raw weights — Phase-1 cache was deleted; a faithful activation-scaled recompute uses the cache saved by the overnight random-quant run). |
| `plot_fig4b.py` / `plot_fig4b_qwen3.py` | MDS scatter + distance histogram (Qwen1.5 / Qwen3). |
| `D_grass_qwen{15,3}_{wtype}.npz` | Distance matrix (float32 `(N,N)`) + per-expert `group_id` (int32 `(N,)`). |
| `fig4a_qwen{15,3}_{wtype}.{png,pdf}` | Figure 4a per weight type. |
| `fig4b_qwen{15,3}_{wtype}.{png,pdf}` | Figure 4b per weight type. |
| `qwen3_assignments.pkl` | Small pickle of Qwen3 `{layer, expert, group_id}` extracted from the 11.6 GB `cross_layer_info.pt` on H2. |
| `results/qwen15_random_cluster_ppl.json` | Wikitext-2 PPL of the Qwen1.5 random-cluster ckpt (Table 5). |

## Table 5 results so far

| Model | learned clusters PPL | random matched-size PPL | ΔPPL |
|---|---|---|---|
| Qwen1.5-MoE | 7.14 | 7.44 | +0.30 (learned better; both A100+same cache) |
| Qwen3-30B-A3B | 8.97 | 9.83 | **+0.86** (learned better) |

Three-way at matched 2.16 bits on Qwen3: Grassmannian 8.97 < traversal 9.42 < random 9.83.
Random cross-layer grouping is *worse* than layer-order traversal — cross-layer sharing
only helps when the grouping is subspace-informed. This is the decisive cluster-validity
result. Fig 4b (activation-scaled `D_grass_qwen3_scaled_*.npz`): gate/up show clear
within<between separation (Δμ≈0.075); down_proj has no separable structure at r=32.

## Cross-layer composition (Fig 4a) — distinct layers per cluster

| Model | gate_proj | up_proj | down_proj | traversal degenerate |
|---|---|---|---|---|
| Qwen1.5-MoE (24 layers) | 7.2 | 6.4 | 22.5 | 1 |
| Qwen3-30B-A3B (48 layers) | 17.8 | 16.7 | 40.7 | 1 |

## Reference cluster assignment

The learned Grassmannian assignments used throughout come from the winning fair-bit
Qwen1.5 fake-quant checkpoint at
`/mnt/Data/yqy/resource_dir/glorcq_grassmann/qwen15_fair_grassmann_v3a_r32_recon0/`
(A100 canonical PPL = 7.14; the v3a ckpt itself scored 7.22 due to cross-run kmeans variance; +0.16 extra bits, `fix_rank=20`, `G=128`, `cluster_rank=32`,
`cluster_recon_weight=0.0`, `seed=42`). `cross_layer_info.pt` in that directory
stores `assignments[wtype]` as a list of `{layer, expert, group_id}` records.
`group_id` here is the SVD-sharing group (size-G slabs of the sorted spectral
labels) — exactly the grouping the model uses at inference.

## Random-cluster control (Table 5)

The random-cluster fake-quant checkpoint reuses the same fair-bit config with one
change (`--cluster_method random_matched`, seed 42), which shuffles each weight
type's expert list before the size-G slice loop. This yields the exact same cluster
size distribution as Grassmannian (`[128]*11 + [32]`) with random membership.

Command used to produce the random ckpt (~2 h on 1× A100 with Phase-1 cache reuse):

```
CUDA_VISIBLE_DEVICES=<free_gpu> .venv/bin/python run_quantize.py \
    --model_path <Qwen1.5-MoE fp16 snapshot> \
    --output_path /mnt/Data/yqy/resource_dir/glorcq_grassmann/qwen15_random_cluster_r32 \
    --qbit 2 --fix_rank 20 --G 128 --group_size 128 --attn_bits 4 \
    --int8_lora --int8_lora_v --pool_kmeans \
    --cluster_method random_matched --cluster_seed 42 \
    --export_real_quant \
    --phase1_cache_path /mnt/Data/yqy/resource_dir/glorcq_smoketest/qwen1.5-moe_stripped_v1_phase1_cache.pt
```

PPL is measured with the same evaluator (`evaluate/eval_ppl.py`, `--max_length 2048
--stride 512`) used everywhere else, and lives at `results/qwen15_random_cluster_ppl.json`.

## Regenerating

```
# Distance matrices (~30 min on 1 A100 GPU, all three wtypes)
CUDA_VISIBLE_DEVICES=<free_gpu> python exp/cluster/compute_D_grass.py

# Plots (CPU, <5 min total)
python exp/cluster/plot_fig4a.py
python exp/cluster/plot_fig4b.py
```

## §6.1/§6.2 ablation table (Qwen1.5-MoE, Grassmannian, A100, **seed-locked**)

Seed fix (commit b1e06b0) pins all RNG (rank-1 sketch init, low-rank SVD projection, VQ
k-means) to seed=42, so every run is deterministic given the Stage-1 calibration input.
**These seeded numbers replace all earlier pre-seed values** (which were RNG-lucky draws,
not reproducible). Result JSONs under `results/ablation_seeded/` (gitignored; numbers below).
Reproduction: `exp/cluster/run_seeded_ablation.sh`.

| config | fix_rank | G | bits | PPL | ZS avg(5) |
|---|---|---|---|---|---|
| rank sweep | 16 | 128 | 2.081 | 7.845 | 58.60 |
| rank sweep | 20 | 128 | 2.099 | 7.2995 | 62.34 |
| **base (fair-bit)** | **32** | **128** | **2.152** | **7.138** | **62.80** |
| rank sweep | 64 | 128 | 2.293 | 7.377 | 63.18 |
| G sweep | 32 | 64 | 2.153 | 7.103 | 62.50 |
| G sweep | 20 | 256/512 | — | auto-fallback (n_clusters<8 → traversal) | — |
| LoRA off | 0 | — | 2.000 | grouping-agnostic (reuse traversal) | — |

**Reproducibility proof**: the r=20 config produces a bit-identical PPL of 7.2995 across
three independent runs (from-scratch calibration, cached-calibration, ablation queue),
same GPU.

Findings: (1) **Rank sweep is U-shaped**: PPL drops sharply r16→r32 (shared-U gains
capacity), then rises at r64 (higher-rank per-expert factors accumulate int8 error, +2×
LoRA bits over budget). r=32 is the PPL optimum and lands at the fair-bit budget (+0.15),
so it is the new base. (2) **G=64 ≈ G=128** at matched bits (G64 marginally better PPL,
G128 marginally better ZS) — cross-layer sharing is robust to group size once groups are
large; base uses G=128 for best ZS + fewest shared-U caches. base 7.138/62.80 is the
seed-locked A100 canonical Qwen1.5 headline (replaces the earlier pre-seed 7.142 and the
H2 cross-machine 7.37).
