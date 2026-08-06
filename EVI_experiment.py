# -*- coding: utf-8 -*-
"""EVI 估计量平移敏感性 Monte Carlo 实验：relative shift -> MSE 折线图。

平移采用相对尺度: 实际平移量 c = lambda · q_{Y1}(1-α_n)（与 QTE_experiment 一致），
配置 shifts 中的值是相对比例 λ，避免绝对平移在不同模型间不可比。

流程：
  1. 对每个模型 / 样本量 / λ，依配置生成数据，并把结果指标 Y 统一平移 +λ·q_{Y1}(1-α_n)；
  2. 分别用 Causal Fraga 与 Causal Hill 估计量估计处理组/对照组 EVI；
  3. 重复 R 次，对每个估计量计算相对真值的 MSE（一个 λ 对应一个 MSE）；
  4. 画 fraga / hill 折线图：横轴 λ，纵轴 MSE。

CLI 参数（均可选，默认读配置文件）：
  --replications R        重复次数，如 --replications 500
  --sample-sizes n1 n2..  样本量列表，如 --sample-sizes 1000 2000
  --shifts s1 s2 ..       平移量列表，如 --shifts 0 1 5 10

运行:
  D:\\Miniconda\\python.exe EVI_experiment.py
  D:\\Miniconda\\python.exe EVI_experiment.py --replications 500 --shifts 0 1 2 5 10
"""
import os
import sys
from pathlib import Path

# 限制每个 worker 进程内部的 BLAS/OpenMP 线程数为 1，避免 ProcessPoolExecutor
# 的多个子进程与 OpenBLAS/MKL 的多线程互相竞争，导致 CPU 利用率反而极低。
# 必须在 import numpy 之前设置。
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
})

import numpy as np

# 依赖仓库内脚本
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "estimate"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "EVI"))

from data_generation import load_config, generate_dataset, tau_levels  # noqa: E402
from estimate_propensity_sieve import estimate_propensity_sieve  # noqa: E402
from estimate_quantile_real import real_quantile  # noqa: E402
from causal_fraga import estimate_evi_causal_fraga  # noqa: E402
from causal_hill import estimate_evi_for_config as hill_for_config  # noqa: E402
from estimate_k0 import estimate_k0_by_group, fallback_beta  # noqa: E402

ESTIMATORS = ("fraga", "hill")


def run_experiment(cfg, model, n, shifts, replications, base_seed):
    """对单个 (模型, 样本量) 跑 R 次重复。

    筛方法倾向得分只依赖 X、D，与结果 Y（及 shift）无关，因此每个重复只做
    一次数据生成 + 倾向得分估计，然后对所有 shift 复用 pi_estimate 计算 EVI。

    Fraga 估计量使用自适应 k0（k0 = k^m）：对每个 (模型, n) 的处理组/对照组
    分别估计 k0（不取统一值），对应各自的 β_n。Hill 估计量锚点 α_n 不变。

    平移采用相对尺度: 实际平移量 c = lambda · q_{Y1}(1-α_n)（与 QTE_experiment 一致），
    shifts 列表中的值是相对比例 λ。

    返回 {λ: {estimator: {group: np.ndarray}}}。
    """
    alpha_n = dict(tau_levels(cfg, n))["alpha_n"]
    beta_fb = fallback_beta(cfg, n)   # 兜底 β_n（自适应失败时回退；分组值优先）
    scale = real_quantile(cfg, model, 1, 1.0 - alpha_n)  # 处理组锚点理论分位数

    results = {s: {name: {"gamma_treated": [], "gamma_control": []}
                   for name in ESTIMATORS} for s in shifts}
    for rep in range(replications):
        seed = base_seed + rep
        data = generate_dataset(cfg, model, n, seed)
        data, _h_n, _info = estimate_propensity_sieve(data)
        base_Y = np.asarray(data["Y"])
        # 分组 k0：处理组/对照组各自的 β_n（Fraga 对平移不变，用未平移数据估计即可）
        k0res = estimate_k0_by_group(cfg, data, n)
        beta_t, beta_c = k0res["beta_treated"], k0res["beta_control"]
        for s in shifts:
            data_s = dict(data)                 # 浅拷贝，仅替换 Y 字段
            data_s["Y"] = base_Y + s * scale    # 相对平移 c = λ · q_{Y1}(1-α_n)
            # Fraga（分组 β_n，兜底 beta_fb）与 Hill（统一 α_n）
            res_f = estimate_evi_causal_fraga(data_s, beta_fb, alpha_n, beta_t, beta_c)
            res_h = hill_for_config(cfg, data_s, n)
            for tag, res in (("fraga", res_f), ("hill", res_h)):
                results[s][tag]["gamma_treated"].append(res["gamma_treated"])
                results[s][tag]["gamma_control"].append(res["gamma_control"])
    for s in shifts:
        for name in ESTIMATORS:
            for group in ("gamma_treated", "gamma_control"):
                results[s][name][group] = np.asarray(results[s][name][group], dtype=float)
    return results


def mse(values, truth):
    """估计值与真值的均方误差（忽略 nan）。"""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.nan
    return float(np.mean((v - truth) ** 2))


def summarize(values, truth):
    """返回 (mean, bias, std, rmse, n_valid)。"""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    mean = v.mean()
    bias = mean - truth
    std = v.std(ddof=1)
    rmse = np.sqrt(np.mean((v - truth) ** 2))
    return mean, bias, std, rmse, v.size


def plot_shift_lines(mse_by, theory, models, sample_sizes, shifts, out_dir):
    """每个 (模型, 样本量) 一张折线图：两子图（gamma_1 / gamma_0），
    横轴 shift，纵轴 MSE，两条线分别对应 fraga / hill。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"[Warning] matplotlib unavailable, skip plotting: {e}")
        return

    est_names = list(ESTIMATORS)
    colors = {"fraga": "#5b9bd5", "hill": "#ed7d31"}
    est_labels = {"fraga": "Causal Fraga", "hill": "Causal Hill"}
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        for n in sample_sizes:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            for c, group in enumerate(("gamma_treated", "gamma_control")):
                ax = axes[c]
                label = "γ₁" if group == "gamma_treated" else "γ₀"
                truth = theory[model]["gamma_1"] if group == "gamma_treated" else theory[model]["gamma_0"]
                for name in est_names:
                    vals = [mse_by[model][n][name][group][s] for s in shifts]
                    ax.plot(shifts, vals, marker="o", linewidth=1.8, linestyle="-",
                            label=est_labels[name], color=colors[name])
                ax.set_xlabel("relative shift λ")
                ax.set_ylabel("MSE")
                ax.set_title(f"{model} (n = {n}, {label} = {truth:.3f})")
                ax.grid(alpha=0.3)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=len(est_names), fontsize=10, frameon=False)
            fig.tight_layout(rect=(0, 0, 1, 0.92))

            out_path = out_dir / f"EVI_shift_{model}_n{n}.png"
            fig.savefig(out_path, dpi=150)
            print(f"Line plot saved: {out_path}")
            plt.close(fig)


if __name__ == "__main__":
    import argparse
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="EVI shift-sensitivity Monte Carlo experiment")
    parser.add_argument("--replications", type=int, default=None,
                        help="repeat count per scenario (default: config design.replications)")
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=None,
                        help="list of sample sizes n (default: config design.sample_sizes)")
    parser.add_argument("--shifts", type=float, nargs="+", default=None,
                        help="list of outcome shifts (default: config design.shifts)")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel worker processes (default: config design.num_workers, <=0 = auto)")
    args = parser.parse_args()

    cfg = load_config()
    seed_base = cfg["experiment"]["random_seed"]
    sample_sizes = args.sample_sizes if args.sample_sizes else list(cfg["design"]["sample_sizes"])
    shifts = args.shifts if args.shifts else [float(s) for s in cfg["design"].get("shifts", [0.0])]
    replications = args.replications if args.replications else cfg["design"].get("replications", 1000)
    workers = args.workers if args.workers is not None else int(cfg["design"].get("num_workers", 0))
    models = list(cfg["outcome_models"])
    theory = {m: cfg["outcome_models"][m]["evi"] for m in models}

    print("=" * 72)
    print("EVI 估计量平移敏感性 Monte Carlo 实验")
    print("  （并行粒度：模型 x 样本量组合，每个组合一个进程）")
    print(f"  模型: {models}")
    print(f"  样本量列表: {sample_sizes}")
    print(f"  相对平移比例 λ 列表: {shifts}")
    print(f"  重复次数 R = {replications}")
    print("=" * 72)

    # 任务 = 每个 (模型, 样本量) 组合一个进程，进程内对所有 shift 复用倾向得分
    tasks = [(model, n) for model in models for n in sample_sizes]
    if workers > 0:
        max_workers = min(len(tasks), workers)
    else:  # 自动：min(任务数, CPU 逻辑核数)
        max_workers = min(len(tasks), os.cpu_count() or 1)
    print(f"  任务数: {len(tasks)}，并行进程数: {max_workers}（配置 num_workers={workers}）")

    all_results = {model: {n: None for n in sample_sizes} for model in models}
    from tqdm import tqdm

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(run_experiment, cfg, model, n, shifts, replications, seed_base): (model, n)
            for model, n in tasks
        }
        with tqdm(total=len(futures), desc="Running scenarios", unit="task",
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

    # 汇总统计（MSE 主结果 + 均值/偏差/std 参考）
    print("\n" + "=" * 72)
    print("汇总统计（MSE = 估计值与真值的均方误差）")
    print(f"{'模型':<5}{'n':>6}{'shift':>8}{'组':<6}{'估计量':<8}{'均值':>9}{'偏差':>9}{'std':>9}{'MSE':>9}")
    print("-" * 72)
    mse_by = {model: {n: {name: {group: {} for group in ("gamma_treated", "gamma_control")}
                          for name in ESTIMATORS} for n in sample_sizes} for model in models}
    for model in models:
        for n in sample_sizes:
            for s in shifts:
                for group, gname in (("gamma_treated", "γ1"), ("gamma_control", "γ0")):
                    truth = theory[model]["gamma_1"] if group == "gamma_treated" else theory[model]["gamma_0"]
                    for name in ESTIMATORS:
                        mean, bias, std, _rmse, _nv = summarize(all_results[model][n][s][name][group], truth)
                        m = mse(all_results[model][n][s][name][group], truth)
                        mse_by[model][n][name][group][s] = m
                        print(f"{model:<5}{n:>6}{s:>8.2f}{gname:<6}{name:<8}{mean:9.4f}{bias:9.4f}"
                              f"{std:9.4f}{m:9.4f}")

    # 折线图（每个模型 x 样本量一张）
    out_dir = Path(__file__).resolve().parent / "results"
    plot_shift_lines(mse_by, theory, models, sample_sizes, shifts, out_dir)

    print("\n实验结束。")
