# -*- coding: utf-8 -*-
"""差分外推法（difference extrapolation）极端分位数处理效应 (QTE) 估计量。

新估计量（用户提出）：利用两个中间水平锚点 q_j(1-α_n)、q_j(1-β_n) 的差分
按极值指数 γ_j 做外推:

    Q_j(1-τ_n) ≈ q_j(1-α_n)
                 + [q_j(1-β_n) - q_j(1-α_n)] · ( (α_n/τ_n)^{γ_j} - 1 )
                                                 -------------------------
                                                 ( (α_n/β_n)^{γ_j} - 1 )
γ→0 时按 L'Hôpital 取极限，斜率 → log(α/τ) / log(α/β)。

锚点分位数: IPW 加权经验分位数（分位数水平 1-α_n 与 1-β_n）。
极值指数:   Candal–Fraga 估计量 γ̂_j^F（EVI/causal_fraga.py 的
            estimate_evi_causal_fraga(data, beta_n, alpha_n)）。

输入: 含 Y, D, pi_estimate 字段的 dict（一维数组）
输出: 目标极端水平 τ（分位数水平 1-τ）下外推的处理组/对照组分位数及 QTE
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "EVI"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from causal_fraga import estimate_evi_causal_fraga  # noqa: E402
from estimate_k0 import fallback_beta  # noqa: E402


def weighted_quantile(Y, weights, tau):
    """加权经验分位数：找 q 使 cum_w(q)/total_w >= tau 的最小 Y 排序值。"""
    Y = np.asarray(Y).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    if Y.size == 0:
        return np.nan
    order = np.argsort(Y, kind="mergesort")
    Y_sorted = Y[order]
    w_sorted = weights[order]
    cum_w = np.cumsum(w_sorted)
    total_w = cum_w[-1]
    if total_w <= 0:
        return np.nan
    target = tau * total_w
    idx = int(np.searchsorted(cum_w, target, side="left"))
    if idx >= Y_sorted.size:
        return float(Y_sorted[-1])
    return float(Y_sorted[idx])


def difference_extrapolate(q_alpha, q_beta, alpha_n, beta_n, tau_target, gamma):
    """差分外推: q̂(1-τ) = q̂(1-α) + [q̂(1-β) - q̂(1-α)] · ((α/τ)^γ - 1)/((α/β)^γ - 1)。

    gamma ≈ 0 时用极限 log(α/τ) / log(α/β)。
    """
    rt = (alpha_n / tau_target) ** gamma
    rb = (alpha_n / beta_n) ** gamma
    if abs(rb - 1.0) < 1e-12:            # γ ≈ 0，0/0 → L'Hôpital
        slope = np.log(alpha_n / tau_target) / np.log(alpha_n / beta_n)
    else:
        slope = (rt - 1.0) / (rb - 1.0)
    return q_alpha + (q_beta - q_alpha) * slope


def estimate_qte_diff_fraga(data, alpha_n, beta_n, tau_target, gamma=None,
                            beta_treated=None, beta_control=None):
    """差分外推法估计极端分位数处理效应（Fraga 极值指数版）。

    data          : dict，需含 Y, D, pi_estimate 三个一维字段
    alpha_n       : 锚点水平一（中间，上尾概率），锚点分位数为 q̂_j(1-α_n)
    beta_n        : 锚点水平二（更深，上尾概率，需 beta_n < alpha_n），锚点分位数为 q̂_j(1-β_n)
    tau_target    : 目标极端水平（上尾概率），标量或数组，需为正（tau > 0）；
                    一般用于 tau_target < beta_n（外推到比两锚点更极端的尾部）
    gamma         : 可选的极值指数 dict {gamma_treated, gamma_control}；
                    缺省时用 Candal–Fraga 估计量（estimate_evi_causal_fraga）计算
    beta_treated  : 可选，处理组各自的 β_n（分组 k0 时传入，锚点 q̂_j(1-β) 也分组）
    beta_control  : 可选，对照组各自的 β_n（分组 k0 时传入）

    返回 dict {alpha_n, beta_n, tau, q_anchor_alpha_treated, q_anchor_alpha_control,
              q_anchor_beta_treated, q_anchor_beta_control,
              gamma_treated, gamma_control,
              q_treated_ext, q_control_ext, qte_ext}。
    tau 为数组时 q_*_ext 与 qte_ext 也返回数组。
    """
    Y = np.asarray(data["Y"]).ravel()
    D = np.asarray(data["D"]).ravel()
    pi = np.asarray(data["pi_estimate"]).ravel()
    taus = np.atleast_1d(np.asarray(tau_target, dtype=float))
    if np.any(taus <= 0):
        raise ValueError("tau_target 需为正（tau > 0）")
    beta_t = float(beta_treated) if beta_treated is not None else float(beta_n)
    beta_c = float(beta_control) if beta_control is not None else float(beta_n)
    if not (0.0 < beta_t < alpha_n) or not (0.0 < beta_c < alpha_n):
        raise ValueError("需满足 0 < beta_n < alpha_n")

    eps = 1e-6
    pi_c = np.clip(pi, eps, 1.0 - eps)

    mask_t = (D == 1)
    mask_c = (D == 0)
    w_t = 1.0 / pi_c[mask_t]
    w_c = 1.0 / (1.0 - pi_c[mask_c])

    # 两个锚点中间水平分位数（IPW 加权；β 锚点按组取 q̂_j(1-β_j)）
    q_alpha_t = weighted_quantile(Y[mask_t], w_t, 1.0 - alpha_n)
    q_alpha_c = weighted_quantile(Y[mask_c], w_c, 1.0 - alpha_n)
    q_beta_t = weighted_quantile(Y[mask_t], w_t, 1.0 - beta_t)
    q_beta_c = weighted_quantile(Y[mask_c], w_c, 1.0 - beta_c)

    # 极值指数：缺省用 Candal–Fraga（可分组 β_n）
    if gamma is None:
        fraga = estimate_evi_causal_fraga(data, beta_n, alpha_n,
                                          beta_treated, beta_control)
        gamma_t = fraga["gamma_treated"]
        gamma_c = fraga["gamma_control"]
    else:
        gamma_t = float(gamma["gamma_treated"])
        gamma_c = float(gamma["gamma_control"])

    q_ext_t = np.array([difference_extrapolate(
        q_alpha_t, q_beta_t, alpha_n, beta_t, t, gamma_t) for t in taus])
    q_ext_c = np.array([difference_extrapolate(
        q_alpha_c, q_beta_c, alpha_n, beta_c, t, gamma_c) for t in taus])
    qte_ext = q_ext_t - q_ext_c

    def _scalar_or_array(arr):
        return arr[0] if arr.size == 1 else arr

    return {
        "alpha_n": float(alpha_n),
        "beta_n": float(beta_n),
        "tau": _scalar_or_array(taus),
        "q_anchor_alpha_treated": float(q_alpha_t),
        "q_anchor_alpha_control": float(q_alpha_c),
        "q_anchor_beta_treated": float(q_beta_t),
        "q_anchor_beta_control": float(q_beta_c),
        "gamma_treated": gamma_t,
        "gamma_control": gamma_c,
        "q_treated_ext": _scalar_or_array(q_ext_t),
        "q_control_ext": _scalar_or_array(q_ext_c),
        "qte_ext": _scalar_or_array(qte_ext),
    }


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "estimate"))
    from data_generation import load_config, generate_dataset, tau_levels
    from estimate_propensity_sieve import estimate_propensity_sieve

    cfg = load_config()
    seed = cfg["experiment"]["random_seed"]
    first_model = list(cfg["outcome_models"])[0]
    first_n = cfg["design"]["sample_sizes"][0]

    # 锚点水平 α_n、β_n 取自配置；目标极端水平 τ 取自配置中所有 tau_n_* 水平
    levels = dict(tau_levels(cfg, first_n))
    alpha_n = levels["alpha_n"]
    beta_n = fallback_beta(cfg, first_n)   # 兜底 β_n（锚点水平二）
    target_levels = [(name, levels[name]) for name in levels if name.startswith("tau_n")]
    target_taus = [t for _, t in target_levels]

    data = generate_dataset(cfg, first_model, first_n, seed)
    data, h_n, info = estimate_propensity_sieve(data)

    print("=" * 92)
    print(f"[测试] 模型={first_model}, n={first_n}, h_n={h_n}")
    print(f"  锚点水平 alpha_n = {alpha_n:.4e}, beta_n = {beta_n:.4e}")

    res = estimate_qte_diff_fraga(data, alpha_n, beta_n, target_taus)
    print(f"  锚点分位数: q1(1-a)={res['q_anchor_alpha_treated']:12.3f}, "
          f"q1(1-b)={res['q_anchor_beta_treated']:12.3f}")
    print(f"  Fraga EVI:  gamma_1^F={res['gamma_treated']:.4f}, "
          f"gamma_0^F={res['gamma_control']:.4f}")

    print("\n  [差分外推 QTE (Fraga)]（目标水平取自配置文件，显示 1-tau 对应的分位数水平）")
    print(f"  {'name':<12}{'tau(上尾)':<14}{'1-tau':<16}"
          f"{'q_treated':>16}{'q_control':>16}{'QTE':>16}")
    print(f"  {'-' * 90}")
    for i, (name, t) in enumerate(target_levels):
        q_level = 1.0 - t
        print(f"  {name:<12}{t:<14.3e}{q_level:<16.6f}"
              f"{np.asarray(res['q_treated_ext'])[i]:16.4f}"
              f"{np.asarray(res['q_control_ext'])[i]:16.4f}"
              f"{np.asarray(res['qte_ext'])[i]:16.4f}")
