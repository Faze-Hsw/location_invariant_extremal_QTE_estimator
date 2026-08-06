# -*- coding: utf-8 -*-
"""QTE 估计量位置敏感性 Monte Carlo 实验：mu -> 箱线图。

对比各 QTE 估计方法在不同共享位置偏移 μ 下对分位数水平 1-tau_n 处 QTE 估计的
MSE（真实 QTE 由 QTE_real.py 给出），覆盖配置中全部 tau_n_* 水平：

  - Deuber      : Hill EVI + Weissman 外推（锚点 alpha_n）
  - Deuber_diff : Hill EVI + 差分外推（双锚点 alpha_n / beta_n）
  - Fraga_alpha : Fraga EVI + Weissman 幂外推（锚点 alpha_n）
  - Fraga_diff  : Fraga EVI + 差分外推（双锚点 alpha_n / beta_n）

横坐标为共享位置偏移 μ（模型层外加 Y(j) = μ + 原分布，H1/H2/H3 一致）。
真实 QTE = q_{Y1}(tau) - q_{Y0}(tau) 对 μ 平移不变（两端抵消），
因此 MSE 随 μ 的变化完全反映各估计量的位置敏感性。

流程：
  1. 对每个模型 / 样本量 / μ，依配置生成不同位置分布的样本；
  2. 各方法估计各分位数水平 1-tau_n 处的 QTE；
  3. 重复 R 次，对每个方法计算相对真实 QTE 的 MSE（一个 μ 对应一个 MSE）；
  4. 画箱线图：一个模型一张图，行 = 样本量 n、列 = tau_n 水平。
     子图内横轴 μ、纵轴 Squared Error（对数刻度），
     每个 μ 处各方法一个箱线图（展示 R 次重复的平方误差分布）；
     横纵坐标说明（location shift / Squared Error）放在图最外层。

CLI 参数（均可选，默认读配置文件）：
  --replications R        重复次数，如 --replications 500
  --sample-sizes n1 n2..  样本量列表
  --shifts s1 s2 ..       μ 位置偏移列表
  --workers W             并行进程数

运行:
  D:\\Miniconda\\python.exe QTE_experiment.py
  D:\\Miniconda\\python.exe QTE_experiment.py --replications 500 --shifts 0 1 2 5 10
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
_BASE = Path(__file__).resolve().parent
for _sub in ("data", "estimate", "EVI", "QTE"):
    sys.path.insert(0, str(_BASE / _sub))

from data_generation import load_config, generate_dataset, tau_levels  # noqa: E402
from estimate_propensity_sieve import estimate_propensity_sieve  # noqa: E402
from Deuber import estimate_qte_extrapolation  # noqa: E402
from Deuber_diff import estimate_qte_diff  # noqa: E402
from QTE_Fraga import estimate_qte_extrapolation_fraga  # noqa: E402
from QTE_Fraga_diff import estimate_qte_diff_fraga  # noqa: E402
from QTE_real import real_qte  # noqa: E402
from estimate_k0 import estimate_k0_by_group, fallback_beta  # noqa: E402

METHODS = ["Deuber", "Deuber_diff", "Fraga_alpha", "Fraga_diff"]
COLORS = {
    "Deuber": "#ed7d31",
    "Deuber_diff": "#5b9bd5",
    "Fraga_alpha": "#70ad47",
    "Fraga_diff": "#7030a0",
}
LABELS = {
    "Deuber": "Deuber (Hill)",
    "Deuber_diff": "Deuber (diff extrapolation)",
    "Fraga_alpha": "Fraga (alpha anchor)",
    "Fraga_diff": "Fraga (diff extrapolation)",
}


def estimate_qte_by_method(data_s, tau, alpha_n, beta_n,
                           beta_treated=None, beta_control=None,
                           beta_deuber_diff=None):
    """各方法估计分位数水平 1-tau 处的 QTE（tau 为上尾概率）。

    data_s          : 平移后的数据 dict（含 Y, D, pi_estimate）
    beta_treated    : 可选，处理组自适应 k0 对应的 β_n（Fraga 系列使用）
    beta_control    : 可选，对照组自适应 k0 对应的 β_n（Fraga 系列使用）
    beta_deuber_diff: 可选，Deuber_diff 统一的 β_n（k0 = k^(2/3) 固定，缺省回退 beta_n）
    返回 {method: qte}。
    """
    beta_dd = float(beta_deuber_diff) if beta_deuber_diff is not None else float(beta_n)
    return {
        "Deuber": estimate_qte_extrapolation(data_s, alpha_n, tau)["qte_ext"],
        "Deuber_diff": estimate_qte_diff(data_s, alpha_n, beta_dd, tau)["qte_ext"],
        "Fraga_alpha": estimate_qte_extrapolation_fraga(
            data_s, beta_n, alpha_n, tau,
            beta_treated=beta_treated, beta_control=beta_control)["qte_ext"],
        "Fraga_diff": estimate_qte_diff_fraga(
            data_s, alpha_n, beta_n, tau,
            beta_treated=beta_treated, beta_control=beta_control)["qte_ext"],
    }


def run_experiment(cfg, model, n, shifts, replications, base_seed):
    """对单个 (模型, 样本量) 跑 R 次重复，覆盖配置中全部 tau_n_* 水平。

    横坐标为共享位置偏移 μ（模型层外加 Y(j) = μ + 原分布）：shifts 中的值直接
    作为 μ（设 cfg["design"]["mu"]），深拷贝配置并重新生成样本。同 seed 下
    X、U、D 不变，仅 Y 随 μ 变。真实 QTE 平移不变（μ 两端抵消），truth_by 与 μ 无关。

    筛方法倾向得分只依赖 X、D，与结果 Y（及 μ）无关，因此每重复只筛估计一次，
    所有 μ 场景复用 pi_estimate。Fraga 系列使用分组自适应 k0（基于 μ=0 数据）。

    返回 (results, truth_by, tau_vals)。
    results : {tau_name: {μ: {method: np.ndarray}}}（R 次重复的 QTE 估计）
    truth_by: {tau_name: 真实 QTE（理论值，μ 平移不变）}
    tau_vals: {tau_name: 上尾概率 tau}
    """
    import copy

    levels = dict(tau_levels(cfg, n))
    tau_names = [name for name in levels if name.startswith("tau_n")]
    alpha_n = levels["alpha_n"]
    beta_fb = fallback_beta(cfg, n)   # 兜底 β_n（自适应失败时回退；分组值优先）
    # Deuber_diff 统一 β_n：k0 取配置公式（k0_deuber_diff_formula，默认 k^(2/3)，
    # k = n^0.65），不随 Fraga 分组估计；β = k0 / n
    k0_dd = eval(cfg["design"]["k0_deuber_diff_formula"], {"n": n, "log": np.log})
    beta_deuber_diff = float(k0_dd) / n
    tau_vals = {name: levels[name] for name in tau_names}
    # μ 平移不变：真实 QTE 与 μ 无关
    truth_by = {name: real_qte(cfg, model, 1.0 - levels[name])["qte"]
                for name in tau_names}

    results = {name: {s: {m: [] for m in METHODS} for s in shifts}
               for name in tau_names}
    for rep in range(replications):
        seed = base_seed + rep
        data0 = generate_dataset(cfg, model, n, seed)
        data0, _h_n, _info = estimate_propensity_sieve(data0)
        pi0 = data0["pi_estimate"]
        # 分组 k0：处理组/对照组各自的 β_n（Fraga 位置不变，用 μ=0 数据估计即可）
        k0res = estimate_k0_by_group(cfg, data0, n)
        beta_t = k0res["beta_treated"]
        beta_c = k0res["beta_control"]
        for s in shifts:
            cfg_s = copy.deepcopy(cfg)
            cfg_s["design"]["mu"] = s                 # 横坐标：共享位置偏移 μ
            data_s = generate_dataset(cfg_s, model, n, seed)
            data_s["pi_estimate"] = pi0              # X、D 相同 → 倾向得分复用
            for name in tau_names:
                est = estimate_qte_by_method(data_s, levels[name], alpha_n, beta_fb,
                                             beta_t, beta_c,
                                             beta_deuber_diff=beta_deuber_diff)
                for m in METHODS:
                    results[name][s][m].append(est[m])
    for name in tau_names:
        for s in shifts:
            for m in METHODS:
                results[name][s][m] = np.asarray(results[name][s][m], dtype=float)
    return results, truth_by, tau_vals


def mse(values, truth):
    """估计值与真值的均方误差（忽略 nan）。"""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.nan
    return float(np.mean((v - truth) ** 2))


def formula_to_latex(formula):
    r"""把配置里的分位数公式（Python 表达式）转成 LaTeX 分数，供 mathtext 渲染。

    示例: "5 / n" -> r"\frac{5}{n}";  "5 / (n * log(n))" -> r"\frac{5}{n\log(n)}"
    """
    s = formula.replace(" ", "")
    s = s.replace("log(n)", r"\log(n)")
    s = s.replace("*", r"\, ")
    if "/" in s:
        num, den = s.split("/", 1)
        den = den.strip()
        if den.startswith("(") and den.endswith(")"):
            den = den[1:-1]
        return rf"\frac{{{num}}}{{{den}}}"
    return s


def plot_shift_boxplots(all_results, truth_by, tau_names, tau_formulas,
                        models, sample_sizes, shifts, out_dir):
    """一个模型一张图：行 = 样本量 n，列 = tau_n 水平。
    子图内横轴 μ、纵轴平方误差 (QTE 估计 - 真实 QTE)^2（对数刻度），
    每个 μ 处各方法一个箱线图（展示 R 次重复的平方误差分布）。
    横纵坐标说明（location shift / squared error）放在图的最外层。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import LogLocator
        from matplotlib.patches import Patch
    except ImportError as e:
        print(f"[Warning] matplotlib unavailable, skip plotting: {e}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    n_methods = len(METHODS)
    width = 0.8 / n_methods
    n_tau = len(tau_names)
    n_n = len(sample_sizes)
    for model in models:
        fig, axes = plt.subplots(n_n, n_tau, figsize=(4.6 * n_tau, 3.6 * n_n),
                                 squeeze=False)
        for r, n in enumerate(sample_sizes):
            for c, name in enumerate(tau_names):
                ax = axes[r][c]
                _tau, truth_qte = truth_by[model][n][name]
                results, _truth, _tau_vals = all_results[model][n]
                x = np.arange(len(shifts))
                for i, m in enumerate(METHODS):
                    positions = x + (i - (n_methods - 1) / 2) * width
                    # 平方误差恒非负；对数刻度只能展示正值，过滤零值（及 nan）
                    data = [((results[name][s][m] - truth_qte) ** 2)
                            [np.isfinite(results[name][s][m])] for s in shifts]
                    data = [d[d > 0] for d in data]
                    bp = ax.boxplot(data, positions=positions, widths=width * 0.8,
                                    patch_artist=True, showfliers=False,
                                    manage_ticks=False)
                    for patch in bp["boxes"]:
                        patch.set_facecolor(COLORS[m])
                        patch.set_alpha(0.7)
                    # 每个 μ 处用黑色圆点标注均值（忽略空箱）
                    means = [float(d.mean()) if d.size else np.nan for d in data]
                    ax.plot(positions, means, "ko", markersize=2)
                ax.set_xticks(x)
                ax.set_xticklabels([f"{s:.1f}" for s in shifts])
                # τ_n 标识只在第一行子图顶部显示
                if r == 0:
                    ax.set_title(rf"$\tau_n = {tau_formulas[name]}$")
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))  # 10 的幂刻度
                ax.grid(alpha=0.3, which="both", axis="y")
                # 行标签（样本量）：放在最右列子图的 y 轴右侧
                if c == n_tau - 1:
                    ax.set_ylabel(f"n = {n}", fontsize=11)
                    ax.yaxis.set_label_position("right")
        # 最外层横纵坐标说明
        fig.supxlabel(r"location shift $\mu$", fontsize=13, y=0.08)
        fig.supylabel("Squared Error", fontsize=13)
        handles = [Patch(facecolor=COLORS[m], label=LABELS[m]) for m in METHODS]
        fig.legend(handles=handles, loc="upper center", ncol=len(METHODS),
                   fontsize=10, frameon=False)
        # suptitle 下移，避免与顶部图例重合
        fig.suptitle(model, fontsize=13, y=0.90)
        fig.tight_layout(rect=(0, 0.04, 1, 0.94))

        out_path = out_dir / f"QTE_shift_{model}.png"
        fig.savefig(out_path, dpi=150)
        print(f"Line plot saved: {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    import argparse
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="QTE shift-sensitivity Monte Carlo experiment")
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

    # 目标分位数水平：配置中全部 tau_n_*（上尾概率，分位数水平 1-tau_n）
    tau_names = [name for name, _ in tau_levels(cfg, sample_sizes[0])
                 if name.startswith("tau_n")]

    print("=" * 72)
    print("QTE 估计量位置敏感性 Monte Carlo 实验")
    print("  （并行粒度：模型 x 样本量组合，每个组合一个进程）")
    print(f"  模型: {models}")
    print(f"  样本量列表: {sample_sizes}")
    print(f"  μ 位置偏移列表: {shifts}")
    print(f"  目标分位数水平: {tau_names}（对应分位数水平 1-tau_n）")
    print(f"  重复次数 R = {replications}")
    print(f"  对比方法: {LABELS}")
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
            pool.submit(run_experiment, cfg, model, n, shifts, replications,
                        seed_base): (model, n)
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

    # 汇总统计：构造 truth_by 供绘图使用（不打印 MSE）
    truth_by = {model: {n: {name: None for name in tau_names} for n in sample_sizes}
                for model in models}
    for model in models:
        for n in sample_sizes:
            results, truth, tau_vals = all_results[model][n]
            for name in tau_names:
                truth_by[model][n][name] = (tau_vals[name], truth[name])

    # 箱线图（每个模型 x 样本量一张，每张包含全部 tau_n 水平子图）
    tau_formulas = {q["name"]: q["formula"] for q in cfg["design"]["quantile_levels"]
                    if q["name"].startswith("tau_n")}
    out_dir = Path(__file__).resolve().parent / "results"
    plot_shift_boxplots(all_results, truth_by, tau_names, tau_formulas,
                        models, sample_sizes, shifts, out_dir)

    print("\n实验结束。")
