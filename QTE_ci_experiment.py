# -*- coding: utf-8 -*-
"""四种 QTE 估计量的置信区间覆盖率 Monte Carlo 实验（固定 location shift u = 0）。

固定 u = 0，重复 R 次：
  1. 生成原始样本并筛倾向得分，用原始样本得到四种方法的 QTE 点估计 q̂；
  2. Bootstrap 重采样 B 次（每次重估倾向得分与 Fraga 分组 k0，二者只依赖
     X/D 与 μ=0 数据，对所有场景复用），得到 q̂*_b，标准误 se = std(q̂*_b)；
  3. 正态近似置信区间 CI = q̂ ± z_{1-α/2}·se，名义置信水平 1-α 读自
     配置 inference.confidence_level（默认 0.9）；
  4. 覆盖率 = 1{真实 QTE ∈ CI} 的 Monte Carlo 均值（忽略点估计或 se 非有限的重复）。

覆盖率估计本身的不确定性：用 Wilson 区间（默认 95%）报告，绘图误差棒 = 区间。

方法（与 QTE_experiment.py 一致）：
  - Deuber            : Hill EVI + Weissman 外推（锚点 alpha_n）
  - Deuber_diff       : Hill EVI + 差分外推（双锚点 alpha_n / beta_n）
  - Fraga_alpha       : Fraga EVI + Weissman 幂外推（锚点 alpha_n）
  - Fraga_diff        : Fraga EVI + 差分外推（双锚点 alpha_n / beta_n）
  - Fraga_diff_asymp  : Fraga EVI + 差分外推 + 解析标准误（论文公式 18-22，CI 公式 22）

真实 QTE = q_{Y1}(1-τ) - q_{Y0}(1-τ) 由 QTE_real.py 给出，平移不变、不随 u 变化。

绘图：每个模型一张图，行 = 样本量 n、列 = tau_n 水平，子图横轴 = 各方法、
纵轴 = 覆盖率（固定 [0,1]），每种方法一个点估计 + Wilson 误差棒，并画名义
置信水平参考线。

CLI 参数（均可选，默认读配置文件）：
  --replications R         Monte Carlo 重复次数
  --sample-sizes n1 n2..   样本量列表
  --bootstrap B            bootstrap 重采样次数
  --workers W              并行进程数

location shift 固定为常值 u = 0（不提供 --shifts 参数）。

运行:
  D:\\Miniconda\\python.exe QTE_ci_experiment.py
  D:\\Miniconda\\python.exe QTE_ci_experiment.py --replications 200 --bootstrap 200
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
from causal_fraga import estimate_evi_causal_fraga  # noqa: E402
from Deuber import estimate_qte_extrapolation  # noqa: E402
from Deuber_diff import estimate_qte_diff  # noqa: E402
from QTE_Fraga import estimate_qte_extrapolation_fraga  # noqa: E402
from QTE_Fraga_diff import estimate_qte_diff_fraga  # noqa: E402
from QTE_Fraga_diff import difference_extrapolate  # noqa: E402
from QTE_real import real_qte  # noqa: E402
from estimate_k0 import estimate_k0_by_group, fallback_beta  # noqa: E402
from causal_fraga import _weighted_quantile as fraga_wquant  # noqa: E402

METHODS = ["Deuber", "Deuber_diff", "Fraga_alpha", "Fraga_diff", "Fraga_diff_asymp"]
COLORS = {
    "Deuber": "#ed7d31",
    "Deuber_diff": "#5b9bd5",
    "Fraga_alpha": "#70ad47",
    "Fraga_diff": "#7030a0",
    "Fraga_diff_asymp": "#c00000",
}
LABELS = {
    "Deuber": "Causal Hill (mult extrapolation)",
    "Deuber_diff": "Causal Hill (diff extrapolation)",
    "Fraga_alpha": "Causal Fraga (mult extrapolation)",
    "Fraga_diff": "Causal Fraga (diff extrapolation)",
    "Fraga_diff_asymp": "Causal Fraga (asymp CI)",
}


def estimate_qte_by_method(data, tau, alpha_n, beta_fb, beta_deuber_diff, k0res):
    """单个样本上估计四种方法在目标水平 tau 处的 QTE（标量 tau）。

    data : dict，含 Y, D, pi_estimate
    k0res: estimate_k0_by_group 的结果（Fraga 分组自适应 β_t/β_c）
    返回 {method: qte}（估计失败时可为 nan）。
    """
    beta_t = k0res["beta_treated"]
    beta_c = k0res["beta_control"]
    return {
        "Deuber": estimate_qte_extrapolation(data, alpha_n, tau)["qte_ext"],
        "Deuber_diff": estimate_qte_diff(data, alpha_n, beta_deuber_diff, tau)["qte_ext"],
        "Fraga_alpha": estimate_qte_extrapolation_fraga(
            data, beta_fb, alpha_n, tau,
            beta_treated=beta_t, beta_control=beta_c)["qte_ext"],
        "Fraga_diff": estimate_qte_diff_fraga(
            data, alpha_n, beta_fb, tau,
            beta_treated=beta_t, beta_control=beta_c)["qte_ext"],
    }


def fraga_diff_asymp_ci(data, alpha_n, beta_fb, tau_target, k0res, n):
    """Fraga 差分外推 + 解析 CI（基于论文公式 (18)-(22)）。

    对极端分位数 QTE 用 Fraga EVI + 差分外推得到点估计 Δ̂(1-τ)；
    解析标准误按公式 (22)：se = σ̂ / φ̂_n，其中

        σ̂² = min{1,κ̂}² (γ̂₁^F)² â_{n,1}² + min{1,1/κ̂}² (γ̂₀^F)² â_{n,0}²   (公式 21)
        φ̂_n = √k0 / [ log(β_n/τ) · max{Q̂₁(1-τ), Q̂₀(1-τ)} ]                (公式 18)
        κ̂   = Q̂₁(1-τ) / Q̂₀(1-τ)                                             (公式 19)
        â²_{n,j} = (1/k0) Σ_i R̂²_{n,j,i}                                    (公式 20)
        R̂_{n,j,i} = (D_i/π̂)^j ((1-D_i)/(1-π̂))^{1-j} 1{Y_i>q̂_j(1-β_n)}
                     · (1/γ̂_j^F) · log[(Y_i - q̂_j(1-α_n)) / (q̂_j(1-β_n) - q̂_j(1-α_n))]
                     - β_n                                                   (公式 20 内的 R)

    解析 CI: Δ̂(1-τ) ± z_{1-α/2} · σ̂ / φ̂_n                                  (公式 22)

    返回 dict {qte, q_treated, q_control, sigma, phi, se, ci}；
    任一环节出现非有限值则返回 nan（让外层用 nan 标记跳过本次重复）。
    """
    Y = np.asarray(data["Y"]).ravel()
    D = np.asarray(data["D"]).ravel()
    pi = np.asarray(data["pi_estimate"]).ravel()
    if not (0.0 < tau_target < alpha_n) or tau_target <= 0:
        return {"qte": np.nan, "se": np.nan, "sigma": np.nan, "phi": np.nan,
                "q_treated": np.nan, "q_control": np.nan, "ci": (np.nan, np.nan)}
    beta_t = float(k0res["beta_treated"])
    beta_c = float(k0res["beta_control"])
    if not (0.0 < beta_t < alpha_n) or not (0.0 < beta_c < alpha_n):
        return {"qte": np.nan, "se": np.nan, "sigma": np.nan, "phi": np.nan,
                "q_treated": np.nan, "q_control": np.nan, "ci": (np.nan, np.nan)}

    eps = 1e-6
    pi_c = np.clip(pi, eps, 1.0 - eps)

    mask_t = (D == 1)
    mask_c = (D == 0)
    w_t = 1.0 / pi_c[mask_t]
    w_c = 1.0 / (1.0 - pi_c[mask_c])

    # 锚点分位数（与 Fraga_diff 一致；β 锚点按组使用各自的 β_j）
    q_alpha_t = fraga_wquant(Y[mask_t], w_t, 1.0 - alpha_n)
    q_alpha_c = fraga_wquant(Y[mask_c], w_c, 1.0 - alpha_n)
    q_beta_t = fraga_wquant(Y[mask_t], w_t, 1.0 - beta_t)
    q_beta_c = fraga_wquant(Y[mask_c], w_c, 1.0 - beta_c)

    # Fraga EVI：缺省按 fraga_diff 的方式（β 锚点用 β_t / β_c 各自的 β_n）
    fraga = estimate_evi_causal_fraga(data, beta_fb, alpha_n, beta_t, beta_c)
    gamma_t = fraga["gamma_treated"]
    gamma_c = fraga["gamma_control"]
    if not (np.isfinite(gamma_t) and np.isfinite(gamma_c)) \
            or abs(gamma_t) < 1e-12 or abs(gamma_c) < 1e-12:
        return {"qte": np.nan, "se": np.nan, "sigma": np.nan, "phi": np.nan,
                "q_treated": np.nan, "q_control": np.nan, "ci": (np.nan, np.nan)}

    # 差分外推点估计（与 Fraga_diff 相同实现）
    q_t = difference_extrapolate(q_alpha_t, q_beta_t, alpha_n, beta_t, tau_target, gamma_t)
    q_c = difference_extrapolate(q_alpha_c, q_beta_c, alpha_n, beta_c, tau_target, gamma_c)
    qte = q_t - q_c

    # === 解析标准误（公式 18–22） ===
    # k0 用 n · β_n 处理组（与分组 k0 公式保持一致；公式 (20) 用处理组 β_n
    # 推导时的常数；这里取 n·beta_t，与 fraga 系列 β_t 来自同一自适应估计）。
    k0 = float(n) * beta_t
    if not (k0 > 0):
        return {"qte": np.nan, "se": np.nan, "sigma": np.nan, "phi": np.nan,
                "q_treated": q_t, "q_control": q_c, "ci": (np.nan, np.nan)}

    # 公式 (20)：按 j 分别算 â²_{n,1} 与 â²_{n,0}。
    # 处理组 (j=1): 权重 w_t = D/π̂
    denom1 = q_beta_t - q_alpha_t
    denom0 = q_beta_c - q_alpha_c
    if not (denom1 > 0 and denom0 > 0):
        return {"qte": qte, "se": np.nan, "sigma": np.nan, "phi": np.nan,
                "q_treated": q_t, "q_control": q_c, "ci": (np.nan, np.nan)}

    # j=1：处理组（使用 β_t）
    ind1 = (Y > q_beta_t) & (D == 1)
    if ind1.any():
        log_term1 = np.log((Y[ind1] - q_alpha_t) / denom1)
        R1 = (D[ind1] / pi_c[ind1]) * (1.0 / gamma_t) * log_term1 - beta_t
        a_sq_t = float(np.sum(R1 ** 2)) / k0
    else:
        a_sq_t = np.nan
    # j=0：对照组（使用 β_c）
    ind0 = (Y > q_beta_c) & (D == 0)
    if ind0.any():
        log_term0 = np.log((Y[ind0] - q_alpha_c) / denom0)
        R0 = ((1.0 - D[ind0]) / (1.0 - pi_c[ind0])) * (1.0 / gamma_c) * log_term0 - beta_c
        a_sq_c = float(np.sum(R0 ** 2)) / k0
    else:
        a_sq_c = np.nan

    if not (np.isfinite(a_sq_t) and np.isfinite(a_sq_c)):
        return {"qte": qte, "se": np.nan, "sigma": np.nan, "phi": np.nan,
                "q_treated": q_t, "q_control": q_c, "ci": (np.nan, np.nan)}

    # κ̂ = Q̂₁(1-τ) / Q̂₀(1-τ) （公式 19）
    if not (np.isfinite(q_t) and np.isfinite(q_c) and q_c > 0):
        return {"qte": qte, "se": np.nan, "sigma": np.nan, "phi": np.nan,
                "q_treated": q_t, "q_control": q_c, "ci": (np.nan, np.nan)}
    kappa = q_t / q_c
    kappa = float(np.clip(kappa, 1e-12, None))   # 避免 1/κ 下溢

    # σ̂² = min{1,κ̂}² (γ̂₁^F)² â²_{n,1} + min{1,1/κ̂}² (γ̂₀^F)² â²_{n,0}  (公式 21)
    m1 = min(1.0, kappa)
    m0 = min(1.0, 1.0 / kappa)
    sigma2 = (m1 * gamma_t) ** 2 * a_sq_t + (m0 * gamma_c) ** 2 * a_sq_c
    if not (sigma2 > 0):
        return {"qte": qte, "se": np.nan, "sigma": np.nan, "phi": np.nan,
                "q_treated": q_t, "q_control": q_c, "ci": (np.nan, np.nan)}
    sigma = float(np.sqrt(sigma2))

    # φ̂_n = √k0 / [ log(β_n/τ) · max{Q̂₁(1-τ), Q̂₀(1-τ)} ]           (公式 18)
    # 注：公式 (18) 的 β_n 与 k0 来自同一处理组辅助水平（k0 = n·β_n），故
    # 取 min{β_t, β_c} 作"最深的辅助水平"以保持 log(β_n/τ) > 0。
    beta_phi = float(min(beta_t, beta_c))
    if not (beta_phi > tau_target > 0) or beta_phi <= 0:
        return {"qte": qte, "se": np.nan, "sigma": np.nan, "phi": np.nan,
                "q_treated": q_t, "q_control": q_c, "ci": (np.nan, np.nan)}
    log_term_phi = float(np.log(beta_phi / tau_target))
    if abs(log_term_phi) < 1e-12 or not (q_t > 0 and q_c > 0):
        return {"qte": qte, "se": np.nan, "sigma": np.nan, "phi": np.nan,
                "q_treated": q_t, "q_control": q_c, "ci": (np.nan, np.nan)}
    phi = float(np.sqrt(k0) / (log_term_phi * max(q_t, q_c)))
    if not (phi > 0):
        return {"qte": qte, "se": np.nan, "sigma": np.nan, "phi": np.nan,
                "q_treated": q_t, "q_control": q_c, "ci": (np.nan, np.nan)}

    se = sigma / phi
    return {"qte": float(qte), "se": float(se), "sigma": float(sigma),
            "phi": float(phi), "q_treated": float(q_t), "q_control": float(q_c),
            "ci": (float(qte - 1.96 * se), float(qte + 1.96 * se))}


def bootstrap_prep(data0, n, B, rng, cfg):
    """对 μ=0 原始样本做 B 次有放回重采样，预计算每组 bootstrap 单位。

    返回 (boot_idx, boot_D, boot_pi, boot_k0)：
      boot_idx : list of (n,) 重采样索引
      boot_D   : list of (n,) 重采样后的处理指示
      boot_pi  : list of (n,) 重采样后重新筛估计的倾向得分
      boot_k0  : list of estimate_k0_by_group 结果（bootstrap 样本上自适应 k0）
    """
    import warnings

    X, D, Y = data0["X"], data0["D"], data0["Y"]
    boot_idx, boot_D, boot_pi, boot_k0 = [], [], [], []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        bs = {"X": np.asarray(X)[idx], "D": np.asarray(D)[idx]}
        bs, _h, _info = estimate_propensity_sieve(bs)
        bs["Y"] = np.asarray(Y)[idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                k0 = estimate_k0_by_group(cfg, bs, n)
            except Exception:
                k0 = None  # 失败：后续按兜底 β_n 处理
        boot_idx.append(idx)
        boot_D.append(bs["D"])
        boot_pi.append(bs["pi_estimate"])
        boot_k0.append(k0)
    return boot_idx, boot_D, boot_pi, boot_k0


def run_experiment(cfg, model, n, replications, bootstrap_B,
                   conf_level, base_seed):
    """对单个 (模型, 样本量) 跑 R 次重复、B 次 bootstrap，计算覆盖率。

    location shift 固定为 u = 0。

    返回 (coverage_by, truth_by, tau_vals)。
    coverage_by: {tau_name: {method: (覆盖率, Wilson 下界, 上界)}}
    truth_by   : {tau_name: 真实 QTE}
    tau_vals   : {tau_name: 上尾概率 tau}
    """
    import warnings
    from scipy import stats

    levels = dict(tau_levels(cfg, n))
    tau_names = [name for name in levels if name.startswith("tau_n")]
    alpha_n = levels["alpha_n"]
    beta_fb = fallback_beta(cfg, n)
    k0_dd = eval(cfg["design"]["k0_deuber_diff_formula"], {"n": n, "log": np.log})
    beta_deuber_diff = float(k0_dd) / n
    tau_vals = {name: levels[name] for name in tau_names}
    truth_by = {name: real_qte(cfg, model, 1.0 - levels[name])["qte"]
                for name in tau_names}

    z = stats.norm.ppf(1.0 - (1.0 - conf_level) / 2.0)
    # 记录每方法 R 次重复的覆盖指示（非有限估计记为 nan）
    cov_hits = {name: {m: [] for m in METHODS} for name in tau_names}

    for rep in range(replications):
        seed = base_seed + rep
        data0 = generate_dataset(cfg, model, n, seed)   # μ 缺省 = 0（u 固定为 0）
        data0, _h_n, _info = estimate_propensity_sieve(data0)
        rng = np.random.default_rng(seed + 987654321)   # bootstrap 子种子
        # 原始样本的分组 k0（Fraga；μ=0 数据）
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                k0_orig = estimate_k0_by_group(cfg, data0, n)
            except Exception:
                k0_orig = None
        # 预计算 bootstrap 单位（μ=0 样本重采样 + 重估倾向得分 + 分组 k0）
        boot_idx, boot_D, boot_pi, boot_k0 = bootstrap_prep(
            data0, n, bootstrap_B, rng, cfg)
        Y = data0["Y"]

        # 原始样本点估计（每个 tau 水平四种方法）
        point = {name: estimate_qte_by_method(
            data0, levels[name], alpha_n, beta_fb, beta_deuber_diff, k0_orig)
            for name in tau_names}

        # bootstrap 分布：同一组重采样索引/倾向得分/k0 应用到当前样本的 Y
        # （每个 tau 水平分别记录 bootstrap 估计，se 按 tau 独立计算；
        #  Fraga_diff_asymp 不依赖 bootstrap，跳过）
        boot_methods = [m for m in METHODS if m != "Fraga_diff_asymp"]
        boot_ests = {name: {m: [] for m in boot_methods} for name in tau_names}
        for idx, D_b, pi_b, k0_b in zip(boot_idx, boot_D, boot_pi, boot_k0):
            data_b = {"Y": np.asarray(Y)[idx], "D": D_b, "pi_estimate": pi_b}
            if k0_b is None:
                k0_b = {"beta_treated": beta_fb, "beta_control": beta_fb}
            for name in tau_names:
                b_est = estimate_qte_by_method(
                    data_b, levels[name], alpha_n, beta_fb,
                    beta_deuber_diff, k0_b)
                for m in boot_methods:
                    boot_ests[name][m].append(b_est[m])

        for name in tau_names:
            # 点估计对每个 tau 水平是独立的标量
            est_m = point[name]
            for m in METHODS:
                if m == "Fraga_diff_asymp":
                    # 解析 CI：不依赖 bootstrap，按公式 (22) 直接由 σ̂/φ̂ 构造
                    k0_asym = k0_orig if k0_orig is not None else {
                        "beta_treated": beta_fb, "beta_control": beta_fb}
                    asymp = fraga_diff_asymp_ci(
                        data0, alpha_n, beta_fb, levels[name], k0_asym, n)
                    q_asym, se_asym = asymp["qte"], asymp["se"]
                    if not (np.isfinite(q_asym) and np.isfinite(se_asym)
                            and se_asym > 0):
                        cov_hits[name][m].append(np.nan)
                        continue
                    lo, hi = q_asym - z * se_asym, q_asym + z * se_asym
                    cov_hits[name][m].append(
                        1.0 if (lo <= truth_by[name] <= hi) else 0.0)
                    continue
                q = float(est_m[m])
                arr = np.asarray(boot_ests[name][m], dtype=float)
                arr = arr[np.isfinite(arr)]
                if not np.isfinite(q) or arr.size < 2:
                    cov_hits[name][m].append(np.nan)
                    continue
                se = float(np.std(arr, ddof=1))
                if not (np.isfinite(se) and se > 0):
                    cov_hits[name][m].append(np.nan)
                    continue
                lo, hi = q - z * se, q + z * se
                cov_hits[name][m].append(
                    1.0 if (lo <= truth_by[name] <= hi) else 0.0)

    # 覆盖率点估计 + 覆盖率估计本身的 Wilson 置信区间（估计不确定性）
    # （binomtest 的 wilson 方法即 Wilson score interval）
    wilson_conf = 0.95
    coverage_by = {}
    for name in tau_names:
        coverage_by[name] = {}
        for m in METHODS:
            arr = np.asarray(cov_hits[name][m], dtype=float)
            arr = arr[np.isfinite(arr)]
            r = arr.size
            if r == 0:
                coverage_by[name][m] = (np.nan, np.nan, np.nan)
                continue
            x = int(arr.sum())
            ci = stats.binomtest(x, r).proportion_ci(
                confidence_level=wilson_conf, method="wilson")
            coverage_by[name][m] = (x / r, float(ci.low), float(ci.high))
    return coverage_by, truth_by, tau_vals


def plot_coverage(coverage_by, tau_names, tau_formulas,
                  model, sample_sizes, conf_level, out_dir):
    """一个模型一张图（覆盖率）：location shift 固定 u = 0，行 = 样本量 n、
    列 = tau_n 水平，子图横轴不标方法名、纵轴 = 覆盖率（[0,1]），每种方法一个
    点估计 + Wilson 置信区间误差棒（方法名由图例说明），并画名义置信水平
    参考线。行标签 n 放在最右列 y 轴右侧。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    n_tau = len(tau_names)
    n_n = len(sample_sizes)
    x = np.arange(len(METHODS))
    fig, axes = plt.subplots(n_n, n_tau, figsize=(4.2 * n_tau, 3.4 * n_n),
                             squeeze=False)
    for r, n in enumerate(sample_sizes):
        cov_n = coverage_by[n]          # {tau_name: {method: (cov, lo, hi)}}
        for c, name in enumerate(tau_names):
            ax = axes[r][c]
            for i, m in enumerate(METHODS):
                cov, lo, hi = cov_n[name][m]
                if not np.isfinite(cov):
                    continue
                ax.errorbar(i, cov, yerr=[[cov - lo], [hi - cov]],
                            fmt="o", color=COLORS[m], markersize=5,
                            linewidth=1.2, capsize=3)
            ax.axhline(conf_level, color="black", linestyle="--", linewidth=1.2,
                       alpha=0.7)
            ax.set_ylim(0.0, 1.0)
            # 横坐标不做方法标识（方法名由图例说明）
            ax.set_xticks(x)
            ax.set_xticklabels([])
            if r == 0:
                ax.set_title(rf"$\tau_n = {tau_formulas[name]}$")
            ax.grid(alpha=0.3, which="major", axis="y")
            if c == n_tau - 1:
                ax.set_ylabel(f"n = {n}", fontsize=11)
                ax.yaxis.set_label_position("right")
    fig.supylabel("coverage", fontsize=13)
    # 图例显示各方法与参考线/误差棒说明
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=COLORS[m],
                          markersize=5, label=LABELS[m]) for m in METHODS]
    handles.append(plt.Line2D([0], [0], color="black", linestyle="--",
                              label=f"nominal level {conf_level:.0%}"))
    handles.append(plt.Line2D([0], [0], color="gray", marker="_", ls="",
                              label="error bar: 95% Wilson CI"))
    # 图例放图片最上方；模型标识放在图例下方、τ_n 标题上方
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=5, fontsize=8, frameon=False, alignment="center")
    fig.text(0.5, 0.87, model, ha="center", va="center", fontsize=13)
    fig.tight_layout(rect=(0.03, 0.05, 0.97, 0.84))
    out_path = out_dir / f"QTE_ci_cov_{model}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Coverage plot saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="QTE CI-coverage Monte Carlo experiment")
    parser.add_argument("--replications", type=int, default=None)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    seed_base = cfg["experiment"]["random_seed"]
    sample_sizes = args.sample_sizes if args.sample_sizes else list(cfg["design"]["sample_sizes"])
    replications = args.replications if args.replications else int(
        cfg["design"].get("replications", 1000))
    workers = args.workers if args.workers is not None else int(cfg["design"].get("num_workers", 0))
    conf_level = float(cfg["inference"]["confidence_level"])
    bootstrap_B = args.bootstrap if args.bootstrap else int(
        cfg["inference"].get("bootstrap_replications", 200))
    models = list(cfg["outcome_models"])

    tau_names = [name for name, _ in tau_levels(cfg, sample_sizes[0])
                 if name.startswith("tau_n")]
    tau_formulas = {q["name"]: q["formula"] for q in cfg["design"]["quantile_levels"]
                    if q["name"].startswith("tau_n")}

    from scipy import stats
    z = stats.norm.ppf(1.0 - (1.0 - conf_level) / 2.0)

    print("=" * 72)
    print("QTE 置信区间覆盖率 Monte Carlo 实验")
    print("  （点估计用原始样本；标准误用 bootstrap，正态近似 CI）")
    print(f"  模型: {models}")
    print(f"  样本量列表: {sample_sizes}")
    print("  location shift u = 0（固定）")
    print(f"  Monte Carlo 重复 R = {replications}，bootstrap 重采样 B = {bootstrap_B}")
    print(f"  名义置信水平 = {conf_level}（正态临界值 z = {z:.4f}）")
    print("=" * 72)

    tasks = [(model, n) for model in models for n in sample_sizes]
    if workers > 0:
        max_workers = min(len(tasks), workers)
    else:
        max_workers = min(len(tasks), os.cpu_count() or 1)
    print(f"  任务数: {len(tasks)}，并行进程数: {max_workers}")

    all_cov = {model: {n: None for n in sample_sizes} for model in models}
    all_truth = {model: {n: None for n in sample_sizes} for model in models}
    from tqdm import tqdm

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(run_experiment, cfg, model, n, replications,
                        bootstrap_B, conf_level, seed_base): (model, n)
            for model, n in tasks
        }
        with tqdm(total=len(futures), desc="Running CI experiments", unit="task",
                  ncols=100, dynamic_ncols=True, mininterval=0.0, miniters=1) as pbar:
            for fut in as_completed(futures):
                model, n = futures[fut]
                try:
                    all_cov[model][n], all_truth[model][n], _ = fut.result()
                except Exception as e:
                    print(f"[模型 {model}, n={n}] 失败: {e}")
                    raise
                pbar.set_postfix_str(f"{model} n={n}")
                pbar.update(1)

    out_dir = Path(__file__).resolve().parent / "results"
    for model in models:
        cov_by_n = {n: all_cov[model][n] for n in sample_sizes}
        plot_coverage(cov_by_n, tau_names, tau_formulas,
                      model, sample_sizes, conf_level, out_dir)

    print("\n实验结束。")
