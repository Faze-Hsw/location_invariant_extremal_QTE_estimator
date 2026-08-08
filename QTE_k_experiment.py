# -*- coding: utf-8 -*-
"""QTE 估计量的 k 敏感性分析：不同 k 下的均值与 MSE 折线图。

仿照 Hill EVI 的 k 敏感性模拟图：固定 (模型, 样本量, 目标水平 τ)，
改变 Hill 系列估计使用的 top-k 观测数（锚点水平 α_n = k/n），
重复 R 次估计 QTE，绘制各方法 QTE 估计的均值（应接近真实 QTE）
与 MSE（相对真实 QTE）随 k 的变化折线图。

方法：
  - Deuber      : Hill EVI + Weissman 外推（锚点 α_n = k/n）
  - Deuber_diff : Hill EVI + 差分外推（锚点 α_n = k/n，β_n = k^(2/3)/n 统一）
  - Fraga_alpha : Fraga EVI + Weissman 外推（锚点 α_n = k/n，k0 随 k 分组自适应）
  - Fraga_diff  : Fraga EVI + 差分外推（k0 随 k 分组自适应）

对每个 (模型, 样本量) 一张图：行 = (均值, MSE)，列 = tau_n 水平，
横轴 = k（top observations），每方法一条折线，真实 QTE 参考线。

CLI 参数：
  --replications R        重复次数（默认读配置）
  --sample-sizes n1 n2..  样本量列表
  --k-grid k1 k2 ..       指定 k 序列（默认配置 design.k_grid 或对数网格）
  --workers W             并行进程数

运行:
  D:\\Miniconda\\python.exe QTE_k_experiment.py
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
for _sub in ("data", "estimate", "EVI", "QTE"):
    sys.path.insert(0, str(_BASE / _sub))

from data_generation import load_config, generate_dataset, tau_levels  # noqa: E402
from estimate_propensity_sieve import estimate_propensity_sieve  # noqa: E402
from estimate_k0 import estimate_k0_by_group, fallback_beta  # noqa: E402
from Deuber import estimate_qte_extrapolation  # noqa: E402
from Deuber_diff import estimate_qte_diff  # noqa: E402
from QTE_Fraga import estimate_qte_extrapolation_fraga  # noqa: E402
from QTE_Fraga_diff import estimate_qte_diff_fraga  # noqa: E402
from QTE_real import real_qte  # noqa: E402

METHODS = ["Deuber", "Deuber_diff", "Fraga_alpha", "Fraga_diff"]
COLORS = {
    "Deuber": "#ed7d31",
    "Deuber_diff": "#5b9bd5",
    "Fraga_alpha": "#70ad47",
    "Fraga_diff": "#7030a0",
}
LABELS = {
    "Deuber": "Causal Hill (mult extrapolation)",
    "Deuber_diff": "Causal Hill (diff extrapolation)",
    "Fraga_alpha": "Causal Fraga (mult extrapolation)",
    "Fraga_diff": "Causal Fraga (diff extrapolation)",
}


def default_k_grid(cfg, n):
    """默认 k 网格：[50, n/2] 等距取 10 个点（与 EVI_k_experiment 一致）。"""
    if cfg["design"].get("k_grid"):
        return [float(k) for k in cfg["design"]["k_grid"]]
    k_min = 50
    k_max = n // 2
    return np.linspace(k_min, k_max, 10)


def estimate_qte_by_method(cfg, data, tau, k, n, truth):
    """给定 k（top observations），估计各方法 QTE（tau 为目标上尾概率）。

    cfg : 实验配置（k0 自适应估计用）
    data: 含 Y, D, pi_estimate 的数据 dict
    返回 {method: qte}（估计失败/超尾越界时可为 nan）。
    """
    alpha_n = float(k) / n                 # 锚点水平
    beta_dd = float(k) ** (2.0 / 3.0) / n  # Deuber_diff 统一辅助水平
    beta_fb = fallback_beta(cfg, n)        # Fraga 兜底辅助水平
    out = {}
    try:
        out["Deuber"] = estimate_qte_extrapolation(data, alpha_n, tau)["qte_ext"]
    except Exception:
        out["Deuber"] = np.nan
    try:
        out["Deuber_diff"] = estimate_qte_diff(data, alpha_n, beta_dd, tau)["qte_ext"]
    except Exception:
        out["Deuber_diff"] = np.nan
    try:
        # Fraga 的 k0 随 k 变化：分组自适应估计 β_n（k0 = k^m，k = n·α_n）
        k0res = estimate_k0_by_group(cfg, data, n, k=k)
        beta_t = k0res["beta_treated"]
        beta_c = k0res["beta_control"]
        out["Fraga_alpha"] = estimate_qte_extrapolation_fraga(
            data, beta_fb, alpha_n, tau,
            beta_treated=beta_t, beta_control=beta_c)["qte_ext"]
    except Exception:
        out["Fraga_alpha"] = np.nan
    try:
        k0res = estimate_k0_by_group(cfg, data, n, k=k)
        beta_t = k0res["beta_treated"]
        beta_c = k0res["beta_control"]
        out["Fraga_diff"] = estimate_qte_diff_fraga(
            data, alpha_n, beta_fb, tau,
            beta_treated=beta_t, beta_control=beta_c)["qte_ext"]
    except Exception:
        out["Fraga_diff"] = np.nan
    return out


def run_experiment(cfg, model, n, k_grid, replications, base_seed):
    """对单个 (模型, 样本量) 跑 R 次重复、全部 k，返回 {tau_name: {k: {method: ndarray}}}。"""
    levels = dict(tau_levels(cfg, n))
    tau_names = [name for name in levels if name.startswith("tau_n")]
    truth = {name: real_qte(cfg, model, 1.0 - levels[name])["qte"]
             for name in tau_names}
    results = {name: {k: {m: [] for m in METHODS} for k in k_grid}
               for name in tau_names}
    for rep in range(replications):
        seed = base_seed + rep
        data = generate_dataset(cfg, model, n, seed)
        data, _h_n, _info = estimate_propensity_sieve(data)
        for k in k_grid:
            for name in tau_names:
                est = estimate_qte_by_method(cfg, data, levels[name], k, n, truth[name])
                for m in METHODS:
                    results[name][k][m].append(est[m])
    for name in tau_names:
        for k in k_grid:
            for m in METHODS:
                results[name][k][m] = np.asarray(results[name][k][m], dtype=float)
    return results, truth


def summarize(results, truth):
    """从重复数组汇总均值与 MSE：返回 {tau_name: {k: {method: (mean, mse)}}}。"""
    summ = {}
    for name, by_k in results.items():
        summ[name] = {}
        for k, by_m in by_k.items():
            summ[name][k] = {}
            for m, arr in by_m.items():
                v = arr[np.isfinite(arr)]
                mean = float(v.mean()) if v.size else np.nan
                mse = float(np.mean((v - truth[name]) ** 2)) if v.size else np.nan
                summ[name][k][m] = (mean, mse)
    return summ


def plot_k_curves(summ, truth, tau_names, tau_formulas, model, sample_sizes,
                  k_grid_by_n, out_dir):
    """一个模型两张图（均值、MSE 分开）：行 = 样本量 n、列 = tau_n 水平，
    子图横轴 = k（top observations），每方法一条折线。MSE 图与 H1 的均值图
    用对数刻度（主刻度 10 倍、无小刻度），H2/H3 的均值图用线性刻度。
    横纵坐标说明放在图的最外层；行标签 n 在最右列 y 轴右侧；
    τ_n 标识只在第一行顶部显示。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullLocator

    out_dir.mkdir(parents=True, exist_ok=True)
    n_tau = len(tau_names)
    n_n = len(sample_sizes)
    ks_by_n = {n: np.asarray(k_grid_by_n[n], dtype=float) for n in sample_sizes}
    for metric, ylabel, suffix in (("mean", "mean of QTE estimates", "mean"),
                                   ("mse", "MSE", "mse")):
        fig, axes = plt.subplots(n_n, n_tau, figsize=(4.2 * n_tau, 3.4 * n_n),
                                 squeeze=False)
        for r, n in enumerate(sample_sizes):
            ks = ks_by_n[n]
            for c, name in enumerate(tau_names):
                ax = axes[r][c]
                for m in METHODS:
                    vals = [summ[n][name][k][m][0 if metric == "mean" else 1]
                            for k in ks]
                    ax.plot(ks, vals, marker="o", markersize=2.5, linewidth=1.2,
                            color=COLORS[m], label=LABELS[m])
                # H1 均值图与全部 MSE 图用对数刻度（主刻度固定为 10 的整数
                # 次幂、10 倍间距，关闭小刻度）；H2/H3 均值图用线性刻度
                if metric == "mse" or (metric == "mean" and model == "H1"):
                    ax.set_yscale("log")
                    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
                    ax.yaxis.set_minor_locator(NullLocator())
                if metric == "mean":
                    ax.axhline(truth[n][name], color="black", linestyle="--",
                               linewidth=1.2, alpha=0.7, label="true QTE")
                if r == 0:
                    ax.set_title(rf"$\tau_n = {tau_formulas[name]}$")
                # 横轴固定 5 个等距刻度：0, max_k/4, ..., max_k（如 n=1000 显示 0,200,400,600,800,1000）
                ticks = np.linspace(0, ks.max(), 6)
                ax.set_xticks(ticks)
                ax.set_xticklabels([int(t) for t in ticks], fontsize=7)
                ax.grid(alpha=0.3, which="major", axis="y")
                # 行标签（样本量）：放在最右列子图的 y 轴右侧
                if c == n_tau - 1:
                    ax.set_ylabel(f"n = {n}", fontsize=11)
                    ax.yaxis.set_label_position("right")
        # 最外层横纵坐标说明
        fig.supxlabel("k-number of top observations", fontsize=13, y=0.04)
        fig.supylabel(ylabel, fontsize=13)
        handles = [plt.Line2D([0], [0], color=COLORS[m], label=LABELS[m])
                   for m in METHODS]
        if metric == "mean":
            handles.append(plt.Line2D([0], [0], color="black", linestyle="--",
                                      label="true QTE"))
        fig.legend(handles=handles, loc="upper center", ncol=len(handles),
                   fontsize=9, frameon=False)
        fig.suptitle(model, fontsize=13, y=0.93)
        fig.tight_layout(rect=(0.03, 0.05, 0.97, 0.94))
        out_path = out_dir / f"QTE_k_{suffix}_{model}.png"
        fig.savefig(out_path, dpi=150)
        print(f"K-sensitivity plot saved: {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    import argparse
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="QTE k-sensitivity analysis")
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

    # 每个 (模型, 样本量) 可能有不同的 k 网格
    k_grid_by_n = {}
    for n in sample_sizes:
        k_grid_by_n[n] = args.k_grid if args.k_grid else default_k_grid(cfg, n)

    print("=" * 72)
    print("QTE 估计量 k 敏感性分析（k = top observations 数量）")
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
    all_truth = {model: {n: None for n in sample_sizes} for model in models}
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
                    all_results[model][n], all_truth[model][n] = fut.result()
                except Exception as e:
                    print(f"[模型 {model}, n={n}] 失败: {e}")
                    raise
                pbar.set_postfix_str(f"{model} n={n}")
                pbar.update(1)

    tau_formulas = {q["name"]: q["formula"] for q in cfg["design"]["quantile_levels"]
                    if q["name"].startswith("tau_n")}
    out_dir = Path(__file__).resolve().parent / "results"
    for model in models:
        summ_by_n = {}
        truth_by_n = {}
        for n in sample_sizes:
            summ_by_n[n] = summarize(all_results[model][n], all_truth[model][n])
            truth_by_n[n] = all_truth[model][n]
        tau_names = list(next(iter(summ_by_n.values())).keys())
        plot_k_curves(summ_by_n, truth_by_n, tau_names, tau_formulas,
                      model, sample_sizes, k_grid_by_n, out_dir)

    print("\n实验结束。")
