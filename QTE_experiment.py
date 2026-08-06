# -*- coding: utf-8 -*-
"""QTE 估计量平移敏感性 Monte Carlo 实验：relative shift -> MSE 折线图。

对比各 QTE 估计方法在样本平移下对分位数水平 1-tau_n 处 QTE 估计的 MSE
（真实 QTE 由 QTE_real.py 给出），覆盖配置中全部 tau_n_* 水平：

  - Zhang       : 经验 IPW 加权 QTE（q1 - q0，对平移完全不变）
  - Deuber      : Hill EVI + Weissman 外推（锚点 alpha_n）
  - Fraga_alpha : Fraga EVI + Weissman 幂外推（锚点 alpha_n）
  - Fraga_diff  : Fraga EVI + 差分外推（双锚点 alpha_n / beta_n）

平移采用相对尺度: 实际平移量 c = lambda · q_{Y1}(1-α_n)（λ 为配置 shifts 中的
相对比例），与分布自身尾部位置成比例。避免绝对平移在不同模型间不可比；也避免
绝对 c 下 Hill 的 γ̂ 随 c 收缩使外推因子 (α/τ)^γ→1、外推退化导致的 MSE 虚假下降。

注意：真实 QTE = q_{Y1}(tau) - q_{Y0}(tau) 对样本平移不变（两端 shift 抵消），
因此 MSE 随 λ 的变化完全反映各估计量的平移敏感性。

流程：
  1. 对每个模型 / 样本量 / λ，生成数据并把结果指标 Y 统一平移 +λ·q_{Y1}(1-α_n)；
  2. 各方法估计各分位数水平 1-tau_n 处的 QTE；
  3. 重复 R 次，对每个方法计算相对真实 QTE 的 MSE（一个 λ 对应一个 MSE）；
  4. 画折线图：每个 (模型, 样本量) 一张图，每个 tau_n 水平一个子图，
     子图内横轴 λ、纵轴 MSE（对数刻度），每方法一条线。

CLI 参数（均可选，默认读配置文件）：
  --replications R        重复次数，如 --replications 500
  --sample-sizes n1 n2..  样本量列表
  --shifts s1 s2 ..       平移量列表
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
from Zhang import estimate_qte  # noqa: E402
from Deuber import estimate_qte_extrapolation  # noqa: E402
from QTE_Fraga import estimate_qte_extrapolation_fraga  # noqa: E402
from QTE_Fraga_diff import estimate_qte_diff_fraga  # noqa: E402
from QTE_real import real_qte  # noqa: E402
from estimate_k0 import estimate_k0_by_group, fallback_beta  # noqa: E402

METHODS = ["Zhang", "Deuber", "Fraga_alpha", "Fraga_diff"]
COLORS = {
    "Zhang": "#5b9bd5",
    "Deuber": "#ed7d31",
    "Fraga_alpha": "#70ad47",
    "Fraga_diff": "#7030a0",
}
LABELS = {
    "Zhang": "Zhang (empirical)",
    "Deuber": "Deuber (Hill)",
    "Fraga_alpha": "Fraga (alpha anchor)",
    "Fraga_diff": "Fraga (diff extrapolation)",
}


def estimate_qte_by_method(data_s, tau, alpha_n, beta_n,
                           beta_treated=None, beta_control=None):
    """各方法估计分位数水平 1-tau 处的 QTE（tau 为上尾概率）。

    data_s      : 平移后的数据 dict（含 Y, D, pi_estimate）
    beta_treated: 可选，处理组自适应 k0 对应的 β_n（Fraga 系列使用）
    beta_control: 可选，对照组自适应 k0 对应的 β_n（Fraga 系列使用）
    返回 {method: qte}。
    """
    return {
        "Zhang": estimate_qte(data_s, 1.0 - tau)["qte"],
        "Deuber": estimate_qte_extrapolation(data_s, alpha_n, tau)["qte_ext"],
        "Fraga_alpha": estimate_qte_extrapolation_fraga(
            data_s, beta_n, alpha_n, tau,
            beta_treated=beta_treated, beta_control=beta_control)["qte_ext"],
        "Fraga_diff": estimate_qte_diff_fraga(
            data_s, alpha_n, beta_n, tau,
            beta_treated=beta_treated, beta_control=beta_control)["qte_ext"],
    }


def run_experiment(cfg, model, n, shifts, replications, base_seed):
    """对单个 (模型, 样本量) 跑 R 次重复，覆盖配置中全部 tau_n_* 水平。

    筛方法倾向得分只依赖 X、D，与结果 Y（及 shift）无关，因此每个重复只做
    一次数据生成 + 倾向得分估计，然后对所有 shift 复用 pi_estimate 计算 QTE。

    Fraga 系列（Fraga_alpha / Fraga_diff）使用自适应 k0（k0 = k^m）：对每个
    (模型, n) 的处理组/对照组分别估计 k0（不取统一值），对应各自的 β_n；
    Zhang / Deuber（Hill）不受影响。

    平移采用相对尺度: 实际平移量 c = lambda · q_{Y1}(1-α_n)，即 shifts 列表
    中的值是相对比例 λ，与分布自身尾部位置成比例，避免绝对平移在不同模型间
    不可比、以及绝对 c 下 Hill 外推因子失效导致的虚假趋势。

    返回 (results, truth_by, tau_vals)。
    results : {tau_name: {λ: {method: np.ndarray}}}（R 次重复的 QTE 估计）
    truth_by: {tau_name: 真实 QTE（理论值，平移不变）}
    tau_vals: {tau_name: 上尾概率 tau}
    """
    levels = dict(tau_levels(cfg, n))
    tau_names = [name for name in levels if name.startswith("tau_n")]
    alpha_n = levels["alpha_n"]
    beta_fb = fallback_beta(cfg, n)   # 兜底 β_n（自适应失败时回退；分组值优先）
    tau_vals = {name: levels[name] for name in tau_names}
    truth_by = {name: real_qte(cfg, model, 1.0 - levels[name])["qte"]
                for name in tau_names}

    # 相对平移尺度：处理组锚点理论分位数 q_{Y1}(1-α_n)（不依赖样本与重复）
    scale = real_qte(cfg, model, 1.0 - alpha_n)["q_treated"]

    results = {name: {s: {m: [] for m in METHODS} for s in shifts}
               for name in tau_names}
    for rep in range(replications):
        seed = base_seed + rep
        data = generate_dataset(cfg, model, n, seed)
        data, _h_n, _info = estimate_propensity_sieve(data)
        base_Y = np.asarray(data["Y"])
        # 分组 k0：处理组/对照组各自的 β_n（Fraga 对平移不变，用未平移数据估计即可）
        k0res = estimate_k0_by_group(cfg, data, n)
        beta_t = k0res["beta_treated"]
        beta_c = k0res["beta_control"]
        for s in shifts:
            data_s = dict(data)                 # 浅拷贝，仅替换 Y 字段
            data_s["Y"] = base_Y + s * scale    # 相对平移 c = λ · q_{Y1}(1-α_n)
            for name in tau_names:
                est = estimate_qte_by_method(data_s, levels[name], alpha_n, beta_fb,
                                             beta_t, beta_c)
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
    """每个 (模型, 样本量) 一张图，每个 tau_n 水平一个子图：
    子图内横轴 λ、纵轴 QTE 估计量（对数刻度），每个 λ 处各方法一个箱线图
    （展示 R 次重复的估计分布），并画真实 QTE 水平参考线。"""
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
    for model in models:
        for n in sample_sizes:
            fig, axes = plt.subplots(1, n_tau, figsize=(5.0 * n_tau, 4.5), squeeze=False)
            for c, name in enumerate(tau_names):
                ax = axes[0][c]
                _tau, truth_qte = truth_by[model][n][name]
                results, _truth, _tau_vals = all_results[model][n]
                x = np.arange(len(shifts))
                for i, m in enumerate(METHODS):
                    positions = x + (i - (n_methods - 1) / 2) * width
                    # 对数刻度只能展示正值，过滤非正值
                    data = [results[name][s][m][results[name][s][m] > 0] for s in shifts]
                    bp = ax.boxplot(data, positions=positions, widths=width * 0.8,
                                    patch_artist=True, showfliers=False,
                                    manage_ticks=False)
                    for patch in bp["boxes"]:
                        patch.set_facecolor(COLORS[m])
                        patch.set_alpha(0.7)
                ax.axhline(truth_qte, color="black", linestyle="--", linewidth=1.2,
                           alpha=0.7)
                ax.set_xticks(x)
                ax.set_xticklabels([f"{s:.1f}" for s in shifts])
                ax.set_xlabel("relative shift λ")
                ax.set_ylabel("QTE estimate")
                ax.set_title(rf"$\tau_n = {tau_formulas[name]}$")
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))  # 10 的幂刻度
                ax.grid(alpha=0.3, which="both", axis="y")
            handles = [Patch(facecolor=COLORS[m], label=LABELS[m]) for m in METHODS]
            handles.append(plt.Line2D([0], [0], color="black", linestyle="--",
                                      label="true QTE"))
            fig.legend(handles=handles, loc="upper center", ncol=len(METHODS) + 1,
                       fontsize=10, frameon=False)
            # suptitle 下移，避免与顶部图例重合
            fig.suptitle(f"{model} (n = {n})", fontsize=13, y=0.90)
            fig.tight_layout(rect=(0, 0, 1, 0.90))

            out_path = out_dir / f"QTE_shift_{model}_n{n}.png"
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
    print("QTE 估计量平移敏感性 Monte Carlo 实验")
    print("  （并行粒度：模型 x 样本量组合，每个组合一个进程）")
    print(f"  模型: {models}")
    print(f"  样本量列表: {sample_sizes}")
    print(f"  相对平移比例 λ 列表: {shifts}")
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

    # 汇总统计：MSE 主结果
    print("\n" + "=" * 110)
    print("汇总统计（MSE = 估计 QTE 与真实 QTE 的均方误差）")
    print(f"{'模型':<5}{'n':>6}{'水平':<10}{'shift':>8}{'方法':<20}{'MSE':>14}")
    print("-" * 110)
    mse_by = {model: {n: {name: {m: {} for m in METHODS} for name in tau_names}
                      for n in sample_sizes} for model in models}
    truth_by = {model: {n: {name: None for name in tau_names} for n in sample_sizes}
                for model in models}
    for model in models:
        for n in sample_sizes:
            results, truth, tau_vals = all_results[model][n]
            for name in tau_names:
                truth_by[model][n][name] = (tau_vals[name], truth[name])
                for s in shifts:
                    for m in METHODS:
                        mse_val = mse(results[name][s][m], truth[name])
                        mse_by[model][n][name][m][s] = mse_val
                        print(f"{model:<5}{n:>6}{name:<10}{s:>8.2f}"
                              f"{LABELS[m]:<20}{mse_val:14.6f}")

    # 箱线图（每个模型 x 样本量一张，每张包含全部 tau_n 水平子图）
    tau_formulas = {q["name"]: q["formula"] for q in cfg["design"]["quantile_levels"]
                    if q["name"].startswith("tau_n")}
    out_dir = Path(__file__).resolve().parent / "results"
    plot_shift_boxplots(all_results, truth_by, tau_names, tau_formulas,
                        models, sample_sizes, shifts, out_dir)

    print("\n实验结束。")
