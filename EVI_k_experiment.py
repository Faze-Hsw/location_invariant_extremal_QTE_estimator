# -*- coding: utf-8 -*-
"""EVI 估计量的 k 敏感性分析：不同 k 下的 EVI 估计量箱线图。

仿照 Hill EVI 的 k 敏感性模拟图：固定 (模型, 样本量, 组)，
改变估计使用的 top-k 观测数（中间水平 α_n = k/n），重复 R 次估计
Causal Fraga 与 Causal Hill 的 EVI（处理组/对照组），绘制 EVI 估计量
随 k 的变化箱线图（展示 R 次重复的估计分布），并画真实 EVI 参考线。

估计设置：
  - Hill  : estimate_evi_causal_hill(data, alpha_n)，α_n = k/n
  - Fraga : estimate_evi_causal_fraga(data, beta_n, alpha_n)，
            β_n = k^(2/3)/n（统一辅助水平，随 k 变化）

每个模型一张图（仅 EVI 估计量）：行 = 样本量 n、列 = 组 (γ1/γ0)，
子图横轴 = k（在 (0, kmax] 等距取 6 个点），纵轴 = EVI 估计量，
每个 k 处 fraga / hill 各一个箱线图，真实 EVI 参考线。
横纵坐标说明放在图最外层。

CLI 参数：
  --replications R        重复次数（默认读配置）
  --sample-sizes n1 n2..  样本量列表
  --k-grid k1 k2 ..       指定 k 序列（默认 (0, n/2] 等距 6 点）
  --workers W             并行进程数

运行:
  D:\\Miniconda\\python.exe EVI_k_experiment.py
"""
import os
import sys
from pathlib import Path

# 限制 worker 内 BLAS/OpenMP 线程数为 1（须在 import numpy 前）
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
})

import numpy as np

_BASE = Path(__file__).resolve().parent
for _sub in ("data", "estimate", "EVI"):
    sys.path.insert(0, str(_BASE / _sub))

from data_generation import load_config, generate_dataset, tau_levels  # noqa: E402
from estimate_propensity_sieve import estimate_propensity_sieve  # noqa: E402
from causal_fraga import estimate_evi_causal_fraga  # noqa: E402
from causal_hill import estimate_evi_causal_hill  # noqa: E402

ESTIMATORS = ("fraga", "hill")
COLORS = {"fraga": "#5b9bd5", "hill": "#ed7d31"}
LABELS = {"fraga": "Causal Fraga", "hill": "Causal Hill"}
GROUPS = ("gamma_treated", "gamma_control")
GROUP_LABELS = {"gamma_treated": "γ₁", "gamma_control": "γ₀"}


def default_k_grid(cfg, n):
    """默认 k 网格：[50, n/2] 等距取 10 个点。"""
    if cfg["design"].get("k_grid"):
        return [float(k) for k in cfg["design"]["k_grid"]]
    k_min = 50
    k_max = n // 2
    return np.linspace(k_min, k_max, 10)


def estimate_evi_by_method(data, k, n):
    """给定 k（top observations），估计各方法的 EVI（处理组/对照组）。

    data: 含 Y, D, pi_estimate 的数据 dict
    返回 {estimator: {group: gamma}}（估计失败可为 nan）。
    """
    alpha_n = float(k) / n                 # 中间水平锚点
    beta_n = float(k) ** (2.0 / 3.0) / n   # Fraga 辅助水平（随 k 变化）
    out = {}
    try:
        h = estimate_evi_causal_hill(data, alpha_n)
        out["hill"] = {"gamma_treated": h["gamma_treated"],
                       "gamma_control": h["gamma_control"]}
    except Exception:
        out["hill"] = {"gamma_treated": np.nan, "gamma_control": np.nan}
    try:
        f = estimate_evi_causal_fraga(data, beta_n, alpha_n)
        out["fraga"] = {"gamma_treated": f["gamma_treated"],
                        "gamma_control": f["gamma_control"]}
    except Exception:
        out["fraga"] = {"gamma_treated": np.nan, "gamma_control": np.nan}
    return out


def run_experiment(cfg, model, n, k_grid, replications, base_seed):
    """对单个 (模型, 样本量) 跑 R 次重复、全部 k。

    返回 {group: {k: {estimator: ndarray}}}（R 次重复的 EVI 估计）。
    """
    results = {g: {k: {name: [] for name in ESTIMATORS} for k in k_grid}
               for g in GROUPS}
    for rep in range(replications):
        seed = base_seed + rep
        data = generate_dataset(cfg, model, n, seed)
        data, _h_n, _info = estimate_propensity_sieve(data)
        for k in k_grid:
            est = estimate_evi_by_method(data, k, n)
            for g in GROUPS:
                for name in ESTIMATORS:
                    results[g][k][name].append(est[name][g])
    for g in GROUPS:
        for k in k_grid:
            for name in ESTIMATORS:
                results[g][k][name] = np.asarray(results[g][k][name], dtype=float)
    return results


def plot_k_curves(all_results, truth_by_n, model, sample_sizes, k_grid_by_n,
                  out_dir):
    """每个模型一张图：行 = 样本量 n、列 = 组 (γ1/γ0)，
    子图横轴 = k（top observations），纵轴 = EVI 估计量，
    每估计量一条折线（各 k 处 R 次重复的均值），真实 EVI 参考线。
    横纵坐标说明放在图的最外层。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    n_g = len(GROUPS)
    n_n = len(sample_sizes)
    ks_by_n = {n: np.asarray(k_grid_by_n[n], dtype=float) for n in sample_sizes}

    fig, axes = plt.subplots(n_n, n_g, figsize=(4.4 * n_g, 3.4 * n_n), squeeze=False)
    for r, n in enumerate(sample_sizes):
        ks = ks_by_n[n]
        for c, g in enumerate(GROUPS):
            ax = axes[r][c]
            truth = truth_by_n[n][g]
            for name in ESTIMATORS:
                # 各 k 处 R 次重复的均值（忽略 nan）
                means = [all_results[n][g][k][name][np.isfinite(
                    all_results[n][g][k][name])].mean() for k in ks]
                ax.plot(ks, means, marker="o", markersize=2.5, linewidth=1.2,
                        color=COLORS[name], label=LABELS[name])
            # 真实 EVI 水平参考线
            ax.axhline(truth, color="black", linestyle="--", linewidth=1.2,
                       alpha=0.7)
            if r == 0:
                ax.set_title(rf"{GROUP_LABELS[g]} = {truth:.3f}")
            # 横轴刻度固定为 (0, kmax] 等距 6 个点（数据点仍为 10 个 k 值）
            ticks = np.linspace(0.0, ks.max(), 6)
            ax.set_xticks(ticks)
            ax.set_xticklabels([int(t) for t in ticks], fontsize=7)
            ax.grid(alpha=0.3, which="both", axis="y")
            if c == n_g - 1:
                ax.set_ylabel(f"n = {n}", fontsize=11)
                ax.yaxis.set_label_position("right")
    # 最外层横纵坐标说明
    fig.supxlabel("k-number of top observations", fontsize=13, y=0.04)
    fig.supylabel("EVI estimate", fontsize=13, x=0.05)
    handles = [plt.Line2D([0], [0], color=COLORS[name], label=LABELS[name])
               for name in ESTIMATORS]
    handles.append(plt.Line2D([0], [0], color="black", linestyle="--",
                              label="true EVI"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               fontsize=9, frameon=False)
    fig.suptitle(model, fontsize=13, y=0.93)
    fig.tight_layout(rect=(0.03, 0.05, 0.97, 0.94))
    out_path = out_dir / f"EVI_k_{model}.png"
    fig.savefig(out_path, dpi=150)
    print(f"K-sensitivity plot saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="EVI k-sensitivity analysis")
    parser.add_argument("--replications", type=int, default=None)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--k-grid", type=float, nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    seed_base = cfg["experiment"]["random_seed"]
    sample_sizes = args.sample_sizes if args.sample_sizes else list(cfg["design"]["sample_sizes"])
    replications = args.replications if args.replications else cfg["design"].get("replications", 1000)
    workers = args.workers if args.workers is not None else int(cfg["design"].get("num_workers", 0))
    models = list(cfg["outcome_models"])
    theory = {m: cfg["outcome_models"][m]["evi"] for m in models}

    k_grid_by_n = {}
    for n in sample_sizes:
        k_grid_by_n[n] = args.k_grid if args.k_grid else default_k_grid(cfg, n)

    print("=" * 72)
    print("EVI 估计量 k 敏感性分析（k = top observations 数量）")
    print(f"  模型: {models}")
    print(f"  样本量列表: {sample_sizes}")
    print(f"  重复次数 R = {replications}")
    for n in sample_sizes:
        print(f"  n={n}: k 网格 = {[int(k) for k in k_grid_by_n[n]]}")
    print("=" * 72)

    tasks = [(model, n) for model in models for n in sample_sizes]
    if workers > 0:
        max_workers = min(len(tasks), workers)
    else:
        max_workers = min(len(tasks), os.cpu_count() or 1)
    print(f"  任务数: {len(tasks)}，并行进程数: {max_workers}")

    all_results = {model: {n: None for n in sample_sizes} for model in models}
    from tqdm import tqdm

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(run_experiment, cfg, model, n, k_grid_by_n[n],
                        replications, seed_base): (model, n)
            for model, n in tasks
        }
        with tqdm(total=len(futures), desc="Running k-sensitivity", unit="task",
                  ncols=100, dynamic_ncols=True, mininterval=0.0, miniters=1) as pbar:
            for fut in as_completed(futures):
                model, n = futures[fut]
                try:
                    all_results[model][n] = fut.result()
                except Exception as e:
                    print(f"[模型 {model}, n={n}] 失败: {e}")
                    raise
                pbar.set_postfix_str(f"{model} n={n}")
                pbar.update(1)

    out_dir = Path(__file__).resolve().parent / "results"
    for model in models:
        truth_by_n = {}
        for n in sample_sizes:
            truth_by_n[n] = {"gamma_treated": theory[model]["gamma_1"],
                             "gamma_control": theory[model]["gamma_0"]}
        plot_k_curves(all_results[model], truth_by_n, model, sample_sizes,
                      k_grid_by_n, out_dir)

    print("\n实验结束。")
