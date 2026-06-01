"""
Phase 2: Belady OPT U 矩阵访存调度分析

分析 Qwen1.5-MoE 模型中：
  1. 每个 cluster (wtype, group_id) 的 U 矩阵大小和跨层访问频率 f_g
  2. 全局池总大小 vs A100/H100 L2 容量
  3. 0-1 背包最优常驻集 S*，以及 C*(M) vs M 曲线
  4. 当前（per-layer copy）vs 全局池的 HBM 读取量对比

访问模型：_forward_graph() 每层执行：
  1. hidden @ _U_cat_gate  → 读取该层所有 gate 唯一 cluster U 矩阵
  2. hidden @ _U_cat_up    → 读取该层所有 up 唯一 cluster U 矩阵
  3. bmm(h_per_expert, _U_per_expert_down) → 每个 expert 读取其 down cluster U

"f_g = 多少层用到了 cluster g" → 跨层共享 → L2 复用机会

Usage:
    CUDA_VISIBLE_DEVICES=3 uv run python exp/analyze_u_schedule.py \
        --model_path /path/to/glorcq-realquant
"""

import argparse
import os
import sys
import collections

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def load_model(model_path, device="cuda"):
    from inference.model_builder import load_glorcq_model
    return load_glorcq_model(model_path, device=device)


def collect_cluster_info(model):
    """
    遍历所有 MoE 层，收集：
      - 每个 (wtype_key, group_id) 的 U 矩阵大小（bytes）
      - 访问序列 T：每层每个唯一 cluster 出现一次（对应一次 GEMV 读取）
        - gate: 该层所有唯一 gate cluster 各出现一次
        - up:   同上
        - down: 该层所有唯一 down cluster 各出现一次（per-expert BMM）

    返回：
      access_seq: [(wtype_key, group_id), ...]  # 一次 decode step 的访问序列
      u_sizes:    {(wtype_key, group_id): bytes}
      layer_info: [{'layer_idx': i, 'gate': {cids}, 'up': {cids}, 'down': {cids}}, ...]
    """
    from inference.quantized_linear import GLoRCQLinear

    m = getattr(model, "model", model)
    layers = m.layers

    access_seq = []
    u_sizes = {}
    layer_info = []

    for layer_idx, layer in enumerate(layers):
        moe = getattr(layer, "mlp", None)
        if moe is None or not hasattr(moe, "experts"):
            continue

        # 收集该层各 projection 的唯一 cluster 集合
        proj_clusters = {"gate_proj": set(), "up_proj": set(), "down_proj": set()}

        for expert in moe.experts:
            for wtype_attr in ("gate_proj", "up_proj", "down_proj"):
                proj = getattr(expert, wtype_attr, None)
                if not isinstance(proj, GLoRCQLinear):
                    continue
                if not hasattr(proj, "cluster_id") or proj.cluster_id is None:
                    continue
                key = (proj.cluster_id[0], proj.cluster_id[1])
                proj_clusters[wtype_attr].add(key)
                if key not in u_sizes and proj.U is not None:
                    u_sizes[key] = proj.U.numel() * proj.U.element_size()

        # 访问序列：每个 unique cluster 在该层出现一次
        # 顺序：gate, up, down（与 _forward_graph 执行顺序一致）
        for wtype_attr in ("gate_proj", "up_proj", "down_proj"):
            for key in sorted(proj_clusters[wtype_attr], key=lambda x: str(x)):
                access_seq.append(key)

        layer_info.append({
            "layer_idx": layer_idx,
            "gate": proj_clusters["gate_proj"],
            "up": proj_clusters["up_proj"],
            "down": proj_clusters["down_proj"],
        })

    return access_seq, u_sizes, layer_info


def compute_freq(access_seq):
    """计算每个 cluster 在一个 decode step 中的访问频率 f_g（出现在多少层中）。"""
    return collections.Counter(access_seq)


def belady_knapsack(u_sizes, freq, capacity_bytes):
    """
    贪心背包（fractional knapsack 近似，按 f_g 降序）。
    value_g = f_g * d_g（常驻可节省的 HBM 读取量）
    weight_g = d_g
    capacity = capacity_bytes

    返回 (resident_set, total_saved_bytes, pool_used_bytes)
    """
    items = sorted(
        [(key, d_g, freq.get(key, 0)) for key, d_g in u_sizes.items()],
        key=lambda x: x[2], reverse=True  # 按 f_g 降序
    )

    resident = []
    pool_used = 0
    total_saved = 0
    for key, d_g, f_g in items:
        if pool_used + d_g <= capacity_bytes:
            resident.append(key)
            pool_used += d_g
            total_saved += f_g * d_g

    return resident, total_saved, pool_used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("=" * 65)
    print("Phase 2: U 矩阵 Belady-OPT 访存调度分析")
    print("=" * 65)

    print(f"\n[1/4] 加载模型 ...")
    model = load_model(args.model_path, device=args.device)

    print("\n[2/4] 收集 cluster 分配 ...")
    access_seq, u_sizes, layer_info = collect_cluster_info(model)
    freq = compute_freq(access_seq)

    num_layers = len(layer_info)
    G = len(u_sizes)
    total_u_mb = sum(u_sizes.values()) / 1e6
    seq_len = len(access_seq)

    # 按 wtype 分开统计
    gate_clusters = {k: v for k, v in u_sizes.items() if k[0] == "gate_proj"}
    up_clusters   = {k: v for k, v in u_sizes.items() if k[0] == "up_proj"}
    down_clusters = {k: v for k, v in u_sizes.items() if k[0] == "down_proj"}

    print(f"\n  MoE 层数:           {num_layers}")
    print(f"  每 step 访问序列长: {seq_len}")
    print(f"  unique cluster 数:  {G}  (gate:{len(gate_clusters)}, up:{len(up_clusters)}, down:{len(down_clusters)})")
    print(f"  gate U 总大小:      {sum(gate_clusters.values())/1e6:.2f} MB")
    print(f"  up   U 总大小:      {sum(up_clusters.values())/1e6:.2f} MB")
    print(f"  down U 总大小:      {sum(down_clusters.values())/1e6:.2f} MB")
    print(f"  全部 U 矩阵总大小:  {total_u_mb:.2f} MB")

    f_vals = [freq.get(k, 0) for k in u_sizes]
    cold_count = sum(1 for f in f_vals if f <= 1)
    print(f"\n  f_g 分布: min={min(f_vals)}, max={max(f_vals)}, "
          f"mean={sum(f_vals)/len(f_vals):.2f}")
    print(f"  cold (f_g=1): {sum(1 for f in f_vals if f==1)},  "
          f"f_g=0(unused): {sum(1 for f in f_vals if f==0)},  "
          f"hot (f_g≥2): {sum(1 for f in f_vals if f>=2)}")

    # Top-10 hot clusters
    print("\n  Top-10 最热 cluster:")
    top10 = sorted(u_sizes.items(), key=lambda kv: freq.get(kv[0], 0), reverse=True)[:10]
    print(f"  {'key':<35} {'size(KB)':>10} {'f_g':>6} {'savings(MB)':>12}")
    print("  " + "-" * 68)
    for key, d_g in top10:
        f_g = freq.get(key, 0)
        print(f"  {str(key):<35} {d_g/1024:>10.1f} {f_g:>6d} {f_g*d_g/1e6:>12.2f}")

    # ── HBM 读取量分析 ──
    print("\n[3/4] HBM 读取量分析:")

    # 当前模式：每层有自己的 per-layer copy，访问时读取该层 unique clusters 总大小
    # 等价于每层读一次各自的 _U_cat_gate / _U_cat_up / _U_per_expert_down
    per_layer_read_mb = 0
    for info in layer_info:
        for wtype_attr, cluster_set in [("gate_proj", info["gate"]),
                                         ("up_proj",   info["up"]),
                                         ("down_proj",  info["down"])]:
            for key in cluster_set:
                per_layer_read_mb += u_sizes.get(key, 0)
    per_layer_read_mb /= 1e6

    # 全局池模式（理想 L2 warm）：总大小读一次（仅首次访问各 cluster 需 HBM）
    global_pool_first_read_mb = total_u_mb

    # 考虑 L2 大小限制的实际分析
    print(f"  当前 per-layer copy: {per_layer_read_mb:.1f} MB/step")
    print(f"  全局池（全在 L2）:   {global_pool_first_read_mb:.1f} MB/step")
    print(f"  最大可节省:          {per_layer_read_mb - global_pool_first_read_mb:.1f} MB/step")

    hbm_bw_tbs = 2.0  # A100 HBM BW in TB/s
    # 1 MB = 1e6 bytes = 1e-6 TB → time(ms) = saved_mb / bw_tbs × 1e-3
    max_saved_ms = (per_layer_read_mb - global_pool_first_read_mb) / hbm_bw_tbs / 1e3
    step_ms = 1000 / 28.6
    print(f"  (A100 @2TB/s) 全在 L2 节省时间: {max_saved_ms:.3f} ms/step")

    # ── Belady 背包分析 ──
    print("\n[4/4] Belady-OPT 背包分析（按 L2 容量）:")
    l2_configs = [
        ("A100 L2",     40 * 1024 * 1024),
        ("H100 L2",     50 * 1024 * 1024),
        ("H100 SXM L2", 60 * 1024 * 1024),
    ]
    baseline_mb = sum(freq.get(k, 0) * d_g for k, d_g in u_sizes.items()) / 1e6

    for name, cap in l2_configs:
        resident, saved_bytes, pool_used = belady_knapsack(u_sizes, freq, cap)
        saved_mb = saved_bytes / 1e6
        actual_cost_mb = baseline_mb - saved_mb
        delta_tok_s = 0.0
        if saved_mb > 0:
            saved_time_ms = saved_mb / hbm_bw_tbs / 1e3  # MB / (TB/s) × 1e-3 = ms
            delta_tok_s = 1000 / (step_ms - saved_time_ms) - 28.6

        fits = "✓ 全部装入" if sum(u_sizes.values()) <= cap else f"✗ 超出（实际 {total_u_mb:.1f} MB）"
        print(f"\n  [{name}] capacity={cap/1e6:.0f} MB, {fits}")
        print(f"    常驻矩阵: {len(resident)}/{G}, pool used: {pool_used/1e6:.1f} MB")
        print(f"    全 miss 基准:  {baseline_mb:.1f} MB/step")
        print(f"    节省:          {saved_mb:.1f} MB/step ({saved_mb/baseline_mb*100:.1f}%)")
        print(f"    剩余 HBM 读取: {actual_cost_mb:.1f} MB/step")
        print(f"    估算速度提升:  +{delta_tok_s:.2f} tok/s")

    print("\n" + "=" * 65)
    print("结论:")
    if total_u_mb <= 40:
        print("  ✓ 全部 U 矩阵 ({:.1f} MB) 可装入 A100 L2（40MB）".format(total_u_mb))
        print("  ✓ Phase 1（Global U Pool）即可实现最优 L2 复用，无需 Phase 3")
    elif total_u_mb <= 60:
        print(f"  ~ 全部 U 矩阵 ({total_u_mb:.1f} MB) 不完全适合 A100 L2，但 H100 上完全适合")
        print(f"  ✓ Phase 1 仍有效，A100 上需 Phase 3 的选择性 warm-up")
    else:
        print(f"  ✗ U 矩阵总大小 ({total_u_mb:.1f} MB) 超出 H100 L2，Phase 3 选择性 warm-up 有价值")
        hot_size = sum(d_g for k, d_g in u_sizes.items() if freq.get(k, 0) >= 3) / 1e6
        print(f"  → f_g≥3 的 hot cluster 总大小: {hot_size:.1f} MB（优先常驻）")
    print("=" * 65)


if __name__ == "__main__":
    main()
