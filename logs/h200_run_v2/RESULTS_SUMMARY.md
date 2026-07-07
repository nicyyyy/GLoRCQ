# H200 v2 Eval Rerun — Results Summary (2026-07-07)

## Config change vs previous run
- `lm_eval` CLI (not HFLM(pretrained=model))
- `add_bos_token=True, dtype=float16, trust_remote_code=True`
- `batch_size=1` for ZS, `batch_size=auto` for MMLU 5-shot
- Report **acc** metric (matches TileQ Table 1 convention)

## Key finding — Mixtral MMLU mystery persists

- Old (batch=32 + HFLM pre-loaded): Mixtral fair MMLU = **49.64**
- New (batch=1 + lm_eval CLI + add_bos): Mixtral fair MMLU = **49.63**
- **Basically unchanged.** My hypothesis about `add_bos_token` was WRONG.
- But: FP16 baselines are now well-aligned with TileQ paper (Mixtral fp16 MMLU 70.42 vs TileQ 71.2, Qwen1.5 fp16 MMLU 61.17 vs TileQ 61.2, Qwen3 fp16 MMLU 79.55 vs TileQ 79.6). So config fix helps fp16 but doesn't help our fair-bit Mixtral.
- The Mixtral fair MMLU gap vs TileQ_s (63.8) is a genuine quantization quality issue on our side. Need to investigate quantizer, not eval.

## Table 1 — Qwen1.5-MoE-A2.7B (all values = acc metric)

| Method | Bits | Extra | PPL↓ | ARC-C↑ | ARC-E↑ | PIQA↑ | WinoG↑ | MMLU↑ | HellaS↑ | Avg(6)↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| FP16 (TileQ paper) | 16 | — | 6.79 | 41.6 | 72.9 | 79.6 | 69.1 | 61.2 | 59.3 | 63.95 |
| **FP16 (ours v2)** | 16 | — | **6.51** | **41.98** | **73.15** | **80.03** | **68.75** | **61.17** | **57.96** | **63.84** |
| GPTQ (TileQ) | 2 | 0.13 | 12.50 | 29.4 | 42.7 | 50.2 | 53.2 | 25.2 | 30.1 | 38.47 |
| GPTVQ (TileQ) | 2 | 0.13 | 8.12 | 34.1 | 68.4 | 71.4 | 62.5 | 53.9 | 49.8 | 56.68 |
| MOEQ (TileQ) | 2 | 0.00 | 5e5 | 34.9 | 34.8 | 59.0 | 58.2 | 48.4 | 38.5 | 45.63 |
| LoPRo (TileQ) | 2 | 0.43 | 7.52 | 39.9 | 72.7 | 77.6 | 68.2 | 56.8 | 53.4 | 61.43 |
| TileQ_s | 2 | 0.16 | 7.56 | 39.6 | 72.5 | 77.8 | 68.9 | 58.8 | 55.5 | 62.18 |
| TileQ_v | 2 | 0.16 | 7.35 | 40.2 | 73.4 | 78.5 | 68.6 | 58.8 | 55.5 | 62.50 |
| **GLoRCQ (fair, ours v2)** | **2** | **0.1621** | **7.17** ★ | 39.51 | 71.51 | 77.26 | 67.09 | 57.25 | 54.17 | **61.13** |

**PPL WIN 0.39 vs TileQ_s. Avg 61.13 vs TileQ_s 62.18 (−1.05). Basically tied on ZS.**

## Table 2 — Mixtral-8x7B (all values = acc metric)

| Method | Bits | Extra | PPL↓ | ARC-C↑ | ARC-E↑ | PIQA↑ | WinoG↑ | MMLU↑ | HellaS↑ | Avg(6)↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| FP16 (TileQ paper) | 16 | — | 3.87 | 61.9 | 87.3 | 83.7 | 77.1 | 71.2 | 67.3 | 74.75 |
| **FP16 (ours v2)** | 16 | — | **3.42** | **56.74** | **83.67** | **82.59** | **77.11** | **70.42** | **65.08** | **72.60** |
| GPTQ (TileQ) | 2 | 0.13 | 15.30 | 26.5 | 35.6 | 53.0 | 49.3 | 24.3 | 28.2 | 36.15 |
| GPTVQ (TileQ) | 2 | 0.13 | 5.28 | 42.0 | 71.6 | 75.9 | 66.5 | 58.9 | 55.4 | 61.72 |
| MOEQ (TileQ) | 2 | 0.00 | 13.40 | 38.9 | 49.8 | 60.3 | 49.9 | 44.2 | 40.5 | 47.27 |
| LoPRo (TileQ) | 2 | 0.21 | 5.01 | 55.3 | 82.5 | 80.6 | 74.9 | 63.7 | 60.3 | 69.55 |
| TileQ_s | 2 | 0.16 | 4.98 | 55.5 | 82.8 | 80.9 | 75.1 | 63.8 | 60.3 | 69.73 |
| TileQ_v | 2 | 0.16 | 4.78 | 56.3 | 83.8 | 80.5 | 74.8 | 64.4 | 61.4 | 70.20 |
| **GLoRCQ (fair, ours v2)** | **2** | **0.1611** | **4.69** ★ | 46.16 | 76.14 | 75.08 | 71.82 | **49.63** ⚠️ | 53.19 | **62.00** |

**PPL WIN 0.29 vs TileQ_s. Avg 62.00 vs TileQ_s 69.73 (−7.73). MMLU 49.63 anomalously low (was 49.64 before, unchanged by config fix — real quantization issue, needs investigation).**

## Table 3 — Qwen3-30B-A3B (all values = acc metric)

| Method | Bits | Extra | PPL↓ | ARC-C↑ | ARC-E↑ | PIQA↑ | WinoG↑ | MMLU↑ | HellaS↑ | Avg(6)↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| FP16 (TileQ paper) | 16 | — | 8.07 | 52.6 | 79.2 | 79.7 | 70.3 | 79.6 | 58.8 | 70.03 |
| **FP16 (ours v2)** | 16 | — | **7.75** | **52.13** | **79.34** | **79.22** | **70.09** | **79.55** | **59.62** | **69.99** |
| GPTQ (TileQ) | 2 | 0.13 | 14.60 | 31.1 | 54.4 | 68.9 | 57.2 | 55.1 | 43.2 | 51.65 |
| GPTVQ (TileQ) | 2 | 0.13 | 11.80 | 34.1 | 58.5 | 70.7 | 61.2 | 61.5 | 47.9 | 55.65 |
| MOEQ (TileQ) | 2 | 0.00 | 3e4 | 26.6 | 29.8 | 50.5 | 49.9 | 41.4 | 25.7 | 37.32 |
| LoPRo (TileQ) | 2 | 0.58 | 11.10 | 34.4 | 58.3 | 71.5 | 62.9 | 62.9 | 48.0 | 56.33 |
| TileQ_s | 2 | 0.16 | 11.30 | 34.6 | 58.4 | 71.8 | 63.1 | 62.9 | 48.3 | 56.52 |
| TileQ_v | 2 | 0.16 | 10.10 | 42.2 | 70.4 | 74.1 | 65.7 | 71.3 | 50.8 | 62.42 |
| **GLoRCQ (fair, ours v2)** | **2** | **0.1647** | **9.42** ★ | 38.05 | 63.09 | 75.08 | 67.88 | 65.52 | 52.11 | **60.29** |

**PPL WIN 0.68 vs TileQ_v (biggest win). Avg 60.29 vs TileQ_s 56.52 (+3.77 WIN). vs TileQ_v 62.42 (−2.13).**

## Summary of fair-bit story

| Model | PPL vs TileQ_s | ZS Avg vs TileQ_s | Verdict |
|---|---|---|---|
| Qwen1.5-MoE | 7.17 vs 7.56 (**WIN 0.39**) | 61.13 vs 62.18 (LOSE 1.05) | PPL win, ZS ~tie |
| Mixtral | 4.69 vs 4.98 (**WIN 0.29**) | 62.00 vs 69.73 (LOSE 7.73) | PPL win, ZS bad (MMLU pathological) |
| Qwen3 | 9.42 vs 11.30 (**WIN 1.88**) | 60.29 vs 56.52 (**WIN 3.77**) | **DOUBLE WIN** — best story |

## FP16 alignment vs TileQ paper (validates our new config)

| Model × Metric | TileQ paper | Ours v2 | Diff |
|---|---|---|---|
| Qwen1.5 fp16 MMLU | 61.2 | 61.17 | −0.03 ✅ |
| Qwen1.5 fp16 HellaS | 59.3 | 57.96 | −1.34 |
| Mixtral fp16 MMLU | 71.2 | 70.42 | −0.78 ✅ |
| Mixtral fp16 HellaS | 67.3 | 65.08 | −2.22 |
| Qwen3 fp16 MMLU | 79.6 | 79.55 | −0.05 ✅ |
| Qwen3 fp16 HellaS | 58.8 | 59.62 | +0.82 ✅ |

Config fix (add_bos + batch=1 + acc) works. FP16 baselines aligned.
