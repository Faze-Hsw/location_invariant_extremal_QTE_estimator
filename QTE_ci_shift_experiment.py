# -*- coding: utf-8 -*-
"""QTE 置信区间覆盖率随位置偏移量 u 的变化（固定模型 H2、固定分位水平 5/(n·log n)）。

固定 (模型 = H2, 分位水平 τ = 5/(n·log n) = tau_n_3)，对位置偏移量列表
shifts 中每个 u：
  1. 生成带偏移的原始样本（Y = 原分布 + u）并筛倾向得分；
  2. Bootstrap 重采样 B 次（重估倾向得分与 Fraga 分组 k0，二者只依赖 X/D，
     与 u 无关），得到 q̂*_b，标准误 se = std(q̂*_b)；
  3. 正态近似置信区间 CI = q̂ ± z_{1-α/2}·se，名义置信水平 1-α 读自
     配置 inference.confidence_level（默认 0.9）；
  4. 覆盖率 = 1{真实 QTE ∈ CI} 的 Monte Carlo 均值（忽略点估计或 se 非有限的重复）。

真实 QTE 平移不变、不随 u 变化（由 QTE_real.py 给出）。

方法（与 QTE_ci_experiment.py 一致）：
  - Deuber            : Hill EVI + Weissman 外推（锚点 alpha_n）
  - Deuber_diff       : Hill EVI + 差分外推（双锚点 alpha_n / beta_n）
  - Fraga_alpha       : Fraga EVI + Weissman 幂外推（锚点 alpha_n）
  - Fraga_diff        : Fraga EVI + 差分外推（双锚点 alpha_n / beta_n）
  - Fraga_diff_asymp  : Fraga EVI + 差分外推 + 解析标准误（论文公式 18-22，CI 公式 22）

绘图：每个估计量单独一张图（模型固定 H2、样本量固定、方法固定），
子图横轴 = 位置偏移量 u、纵轴 = 覆盖率（固定 [0,1]），该方法的折线 +
Wilson 误差棒，并画名义置信水平参考线。

CLI 参数（均可选，默认读配置文件）：
  --replications R          Monte Carlo 重复次数
  --sample-sizes n1 n2..    样本量列表
  --bootstrap B             bootstrap 重采样次数
  --shifts s1 s2 ..         位置偏移量列表
  --workers W               并行进程数

运行:
  D:\\Miniconda\\python.exe QTE_ci_shift_experiment.py
  D:\\Miniconda\\python.exe QTE_ci_shift_experiment.py --replications 500 --bootstrap 200 --shifts 0 1 2 5 10
"""
import os
import sys
from pathlib import Path

# 限制每个 worker 进程内部的 BLAS/OpenMP 线程数为 2（须在 import numpy 前设置）
os.environ.update({
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "VECLIB_MAXIMUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
})

import numpy as np

# 依赖仓库内脚本
_BASE = Path(__file__).resolve().parent
for _sub in ("data", "estimate", "EVI", "QTE"):
    sys.path.insert(0, str(_BASE / _sub))

from data_generation import load_config, generate_dataset, tau_levels  # noqa: E402
from estimate_propensity_sieve import estimate_propensity_sieve  # noqa: E402
from QTE_real import real_qte  # noqa: E402
from estimate_k0 import estimate_k0_by_group, fallback_beta  # noqa: E402
from QTE_ci_experiment import (estimate_qte_by_method, fraga_diff_asymp_ci,  # noqa: E402
                               bootstrap_prep, METHODS, COLORS, LABELS)

# 固定模型与分位水平（用户指定）
MODEL = "H2"
TAU_NAME = "tau_n_3"      # τ = 5/(n·log n)


def run_experiment(cfg, n, s, replications, bootstrap_B, conf_level, base_seed):
    """对单个 (样本量 n, 偏移量 s) 跑 R 次重复、B 次 bootstrap，计算各方法覆盖率。

    并行粒度 = 每个 (n, s) 组合一个进程，互不共享。

    返回 (coverage, truth, tau)：
      coverage: {method: (覆盖率, Wilson 下界, 上界)}
      truth   : 真实 QTE（平移不变，与 s 无关）
      tau     : 上尾概率
    """
    import copy
    import warnings
    from scipy import stats

    levels = dict(tau_levels(cfg, n))
    alpha_n = levels["alpha_n"]
    tau = levels[TAU_NAME]
    beta_fb = fallback_beta(cfg, n)
    k0_dd = eval(cfg["design"]["k0_deuber_diff_formula"], {"n": n, "log": np.log})
    beta_deuber_diff = float(k0_dd) / n
    truth = real_qte(cfg, MODEL, 1.0 - tau)["qte"]

    z = stats.norm.ppf(1.0 - (1.0 - conf_level) / 2.0)
    cov_hits = {m: [] for m in METHODS}

    for rep in range(replications):
        seed = base_seed + rep
        data0 = generate_dataset(cfg, MODEL, n, seed)   # μ=0 原始样本
        data0, _h_n, _info = estimate_propensity_sieve(data0)
        pi0 = data0["pi_estimate"]
        rng = np.random.default_rng(seed + 987654321)   # bootstrap 子种子
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                k0_orig = estimate_k0_by_group(cfg, data0, n)
            except Exception:
                k0_orig = None
        # bootstrap 单位只依赖 X/D 与 μ=0 数据
        boot_idx, boot_D, boot_pi, boot_k0 = bootstrap_prep(
            data0, n, bootstrap_B, rng, cfg)

        # 带偏移的样本（X/D 相同 → 倾向得分复用）
        cfg_s = copy.deepcopy(cfg)
        cfg_s["design"]["mu"] = s
        data_s = generate_dataset(cfg_s, MODEL, n, seed)
        data_s["pi_estimate"] = pi0
        # 点估计（偏移样本）
        point = estimate_qte_by_method(
            data_s, tau, alpha_n, beta_fb, beta_deuber_diff, k0_orig)
        # bootstrap 分布：同一组重采样索引/倾向得分/k0 应用到偏移样本的 Y
        boot_ests = {m: [] for m in METHODS if m != "Fraga_diff_asymp"}
        for idx, D_b, pi_b, k0_b in zip(boot_idx, boot_D, boot_pi, boot_k0):
            data_b = {"Y": np.asarray(data_s["Y"])[idx], "D": D_b,
                      "pi_estimate": pi_b}
            if k0_b is None:
                k0_b = {"beta_treated": beta_fb, "beta_control": beta_fb}
            b_est = estimate_qte_by_method(
                data_b, tau, alpha_n, beta_fb, beta_deuber_diff, k0_b)
            for m in boot_ests:
                boot_ests[m].append(b_est[m])
        for m in METHODS:
            if m == "Fraga_diff_asymp":
                k0_asym = k0_orig if k0_orig is not None else {
                    "beta_treated": beta_fb, "beta_control": beta_fb}
                asymp = fraga_diff_asymp_ci(
                    data_s, alpha_n, beta_fb, tau, k0_asym, n)
                q_asym, se_asym = asymp["qte"], asymp["se"]
                if not (np.isfinite(q_asym) and np.isfinite(se_asym)
                        and se_asym > 0):
                    cov_hits[m].append(np.nan)
                    continue
                lo, hi = q_asym - z * se_asym, q_asym + z * se_asym
                cov_hits[m].append(
                    1.0 if (lo <= truth <= hi) else 0.0)
                continue
            q = float(point[m])
            arr = np.asarray(boot_ests[m], dtype=float)
            arr = arr[np.isfinite(arr)]
            if not np.isfinite(q) or arr.size < 2:
                cov_hits[m].append(np.nan)
                continue
            se = float(np.std(arr, ddof=1))
            if not (np.isfinite(se) and se > 0):
                cov_hits[m].append(np.nan)
                continue
            lo, hi = q - z * se, q + z * se
            cov_hits[m].append(1.0 if (lo <= truth <= hi) else 0.0)

    # 覆盖率点估计 + Wilson 区间
    wilson_conf = 0.95
    coverage = {}
    for m in METHODS:
        arr = np.asarray(cov_hits[m], dtype=float)
        arr = arr[np.isfinite(arr)]
        r = arr.size
        if r == 0:
            coverage[m] = (np.nan, np.nan, np.nan)
            continue
        x = int(arr.sum())
        ci = stats.binomtest(x, r).proportion_ci(
            confidence_level=wilson_conf, method="wilson")
        coverage[m] = (x / r, float(ci.low), float(ci.high))
    return coverage, truth, tau


def plot_coverage_shift(coverage_by, shifts, n, tau_formula, conf_level, out_dir):
    """每个估计量单独一张图：横轴 = 位置偏移量 u、纵轴 = 覆盖率（[0,1]），
    该方法的折线 + Wilson 误差棒，并画名义置信水平参考线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    for m in METHODS:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        xs, ys, los, his = [], [], [], []
        for s in shifts:
            cov, lo, hi = coverage_by[s][m]
            if not np.isfinite(cov):
                continue
            xs.append(s)
            ys.append(cov)
            los.append(cov - lo)
            his.append(hi - cov)
        ax.errorbar(xs, ys, yerr=[los, his], fmt="o", color=COLORS[m],
                    markersize=4, linewidth=1.0, capsize=2,
                    label=LABELS[m])
        ax.axhline(conf_level, color="black", linestyle="--", linewidth=1.2,
                   alpha=0.7, label=f"nominal level {conf_level:.0%}")
        # 误差棒说明：模拟为短横线（Line2D）放在图例第二行
        from matplotlib.lines import Line2D
        err_bar_proxy = Line2D([0], [0], color="gray", lw=1.0, marker="_",
                               markersize=8, label="error bar: 95% Wilson CI")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(shifts)
        ax.set_xticklabels([f"{s:.1f}" for s in shifts])
        ax.set_xlabel(r"location shift $u$", fontsize=12)
        ax.set_ylabel("coverage", fontsize=12)
        ax.grid(alpha=0.3, which="major", axis="y")
        # 图例放到图片最顶部居中（标题上方，3 列容纳 3 条说明）
        handles, labels = ax.get_legend_handles_labels()
        handles.append(err_bar_proxy)
        labels.append("error bar: 95% Wilson CI")
        ax.legend(handles, labels, fontsize=8, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, 1.20), ncol=3)
        # 标题不含方法名（方法名由图例说明），放在 axes 上方默认位置
        ax.set_title(rf"{MODEL}, n={n}, $\tau_n = {tau_formula}$",
                     fontsize=13)
        fig.tight_layout(rect=(0.03, 0.05, 0.97, 0.88))
        # 文件名用方法 key（短名），避免空格/斜杠
        out_path = out_dir / f"QTE_ci_shift_{MODEL}_n{n}_{m}.png"
        fig.savefig(out_path, dpi=150)
        print(f"Coverage-vs-shift plot saved: {out_path}")
        plt.close(fig)


def main():
    import argparse
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm

    cfg = load_config()
    parser = argparse.ArgumentParser(
        description="QTE CI-coverage vs location shift (H2, tau=5/(n log n))")
    parser.add_argument("--replications", type=int, default=None)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument("--shifts", type=float, nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    seed_base = cfg["experiment"]["random_seed"]
    sample_sizes = (args.sample_sizes if args.sample_sizes
                    else [int(x) for x in cfg["design"]["sample_sizes"]])
    replications = (args.replications if args.replications
                    else int(cfg["design"].get("replications", 1000)))
    workers = (args.workers if args.workers is not None
               else int(cfg["design"].get("num_workers", 0)))
    conf_level = float(cfg["inference"]["confidence_level"])
    bootstrap_B = (args.bootstrap if args.bootstrap
                   else int(cfg["inference"].get("bootstrap_replications", 200)))
    shifts = (args.shifts if args.shifts
              else [float(s) for s in cfg["design"].get("shifts", [0.0])])
    tau_formula = next(q["formula"] for q in cfg["design"]["quantile_levels"]
                       if q["name"] == TAU_NAME)

    from scipy import stats
    z = stats.norm.ppf(1.0 - (1.0 - conf_level) / 2.0)

    print("=" * 72)
    print("QTE 置信区间覆盖率随位置偏移量 u 的变化")
    print(f"  模型: {MODEL}（固定），分位水平: {TAU_NAME} = {tau_formula}")
    print(f"  样本量列表: {sample_sizes}")
    print(f"  位置偏移列表: {shifts}")
    print(f"  Monte Carlo 重复 R = {replications}，bootstrap 重采样 B = {bootstrap_B}")
    print(f"  名义置信水平 = {conf_level}（正态临界值 z = {z:.4f}）")
    print("=" * 72)

    tasks = [(n, s) for n in sample_sizes for s in shifts]
    max_workers = min(len(tasks), workers) if workers > 0 else min(len(tasks), os.cpu_count() or 1)
    print(f"  任务数: {len(tasks)}（{len(sample_sizes)} 个样本量 × {len(shifts)} 个偏移量），"
          f"并行进程数: {max_workers}")

    all_cov = {n: {s: None for s in shifts} for n in sample_sizes}
    all_tau = {n: None for n in sample_sizes}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_experiment, cfg, n, s, replications,
                               bootstrap_B, conf_level, seed_base): (n, s)
                   for n in sample_sizes for s in shifts}
        with tqdm(total=len(futures), desc="Running CI-shift experiments", unit="task",
                  ncols=100, dynamic_ncols=True, mininterval=0.0, miniters=1) as pbar:
            for fut in as_completed(futures):
                n, s = futures[fut]
                try:
                    coverage, truth, tau = fut.result()
                except Exception as e:
                    print(f"[n={n}, s={s}] 失败: {e}")
                    raise
                all_cov[n][s] = coverage
                all_tau[n] = (truth, tau)
                pbar.set_postfix_str(f"n={n}, s={s}")
                pbar.update(1)

    out_dir = Path(__file__).resolve().parent / "results"
    for n in sample_sizes:
        # 从 (n, s) 分片汇总成 {s: {method: (cov, lo, hi)}}
        coverage_by = {s: all_cov[n][s] for s in shifts}
        plot_coverage_shift(coverage_by, shifts, n, tau_formula, conf_level, out_dir)

    print("\n实验结束。")


if __name__ == "__main__":
    main()
