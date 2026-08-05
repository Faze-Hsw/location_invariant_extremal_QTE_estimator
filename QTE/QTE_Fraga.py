# -*- coding: utf-8 -*-
"""外推法（Weissman 型）极端分位数处理效应 (QTE) 估计量（Fraga 极值指数版）。

锚点: 中间水平分位数 q̂_j(1-α_n)（IPW 加权经验分位数，α_n 来自配置，
     公式 α_n = k/n，k = n^{0.65}，对应分位数水平 1-α_n）。
极值指数: Candal–Fraga 估计量 γ̂_j^F（EVI/causal_fraga.py 的
     estimate_evi_causal_fraga，内部还需辅助水平 β_n 构造阈值差）。

对更极端的尾部水平 τ < α_n（对应分位数水平 1-τ）用 Weissman 型外推:
    q̂_j^ext(1-τ) = q̂_j(1-α_n) · (α_n / τ)^{γ̂_j^F}
    QTE^ext(1-τ)  = q̂_1^ext(1-τ) - q̂_0^ext(1-τ)

原理: 若尾部分布近似 Pareto，则超过大阈值 u 的对数超出量服从指数分布，
其尺度由 EVI γ 刻画；q̂(1-α_n) 随水平按幂律 (α_n/τ)^γ 向外推移。

输入: 含 Y, D, pi_estimate 字段的 dict（一维数组）
输出: 目标极端水平 τ（分位数水平 1-τ）下外推的处理组/对照组分位数及 QTE
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "EVI"))
from causal_fraga import estimate_evi_causal_fraga  # noqa: E402


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


def extrapolate_quantile(q_anchor, anchor, tau_target, gamma):
    """Weissman 型外推: q̂(1-τ) ≈ q̂(1-anchor) · (anchor/τ)^γ。"""
    return q_anchor * (anchor / tau_target) ** gamma


def estimate_qte_extrapolation_fraga(data, beta_n, alpha_n, tau_target, gamma=None):
    """外推法估计极端分位数处理效应（Fraga 极值指数版，锚点 α_n）。

    data      : dict，需含 Y, D, pi_estimate 三个一维字段
    beta_n    : Fraga 估计量的辅助中间水平（构造阈值差，需 beta_n < alpha_n）
    alpha_n   : 锚点中间水平（上尾概率），锚点分位数为 q̂_j(1-α_n)
    tau_target: 目标极端水平（上尾概率），标量或数组，需为正（tau > 0）；
                一般用于 tau_target < alpha_n（向比锚点更极端的尾部外推）
    gamma     : 可选的极值指数 dict {gamma_treated, gamma_control}；
                缺省时用 Candal–Fraga 估计量（estimate_evi_causal_fraga）计算

    返回 dict {beta_n, alpha_n, tau, q_anchor_treated, q_anchor_control,
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

    eps = 1e-6
    pi_c = np.clip(pi, eps, 1.0 - eps)

    # 锚点中间水平分位数 q̂_j(1-α_n)（IPW 加权）
    mask_t = (D == 1)
    mask_c = (D == 0)
    w_t = 1.0 / pi_c[mask_t]
    w_c = 1.0 / (1.0 - pi_c[mask_c])
    q_anchor_t = weighted_quantile(Y[mask_t], w_t, 1.0 - alpha_n)
    q_anchor_c = weighted_quantile(Y[mask_c], w_c, 1.0 - alpha_n)

    # 极值指数：缺省用 Candal–Fraga
    if gamma is None:
        fraga = estimate_evi_causal_fraga(data, beta_n, alpha_n)
        gamma_t = fraga["gamma_treated"]
        gamma_c = fraga["gamma_control"]
    else:
        gamma_t = float(gamma["gamma_treated"])
        gamma_c = float(gamma["gamma_control"])

    q_ext_t = np.array([extrapolate_quantile(q_anchor_t, alpha_n, t, gamma_t)
                        for t in taus])
    q_ext_c = np.array([extrapolate_quantile(q_anchor_c, alpha_n, t, gamma_c)
                        for t in taus])
    qte_ext = q_ext_t - q_ext_c

    def _scalar_or_array(arr):
        return arr[0] if arr.size == 1 else arr

    return {
        "beta_n": float(beta_n),
        "alpha_n": float(alpha_n),
        "tau": _scalar_or_array(taus),
        "q_anchor_treated": float(q_anchor_t),
        "q_anchor_control": float(q_anchor_c),
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

    # 锚点水平 β_n 与辅助水平 α_n 取自配置；目标极端水平 τ 取自配置中所有 tau_n_* 水平
    levels = dict(tau_levels(cfg, first_n))
    beta_n = levels["beta_n"]
    alpha_n = levels["alpha_n"]
    target_levels = [(name, levels[name]) for name in levels if name.startswith("tau_n")]
    target_taus = [t for _, t in target_levels]

    data = generate_dataset(cfg, first_model, first_n, seed)
    data, h_n, info = estimate_propensity_sieve(data)

    print("=" * 92)
    print(f"[测试] 模型={first_model}, n={first_n}, h_n={h_n}")
    print(f"  锚点水平 alpha_n = {alpha_n:.4e}, 分位数水平 1-alpha_n = {1-alpha_n:.6f}")
    print(f"  Fraga 辅助水平 beta_n = {beta_n:.4e}")

    res = estimate_qte_extrapolation_fraga(data, beta_n, alpha_n, target_taus)
    print(f"  锚点分位数: q1(1-a)={res['q_anchor_treated']:12.3f}, "
          f"q0(1-a)={res['q_anchor_control']:12.3f}")
    print(f"  Fraga EVI:  gamma_1^F={res['gamma_treated']:.4f}, "
          f"gamma_0^F={res['gamma_control']:.4f}")

    print("\n  [Weissman 外推 QTE (Fraga)]（目标水平取自配置文件，显示 1-tau 对应的分位数水平）")
    print(f"  {'name':<12}{'tau(上尾)':<14}{'1-tau':<16}"
          f"{'q_treated':>16}{'q_control':>16}{'QTE':>16}")
    print(f"  {'-' * 90}")
    for i, (name, t) in enumerate(target_levels):
        q_level = 1.0 - t
        print(f"  {name:<12}{t:<14.3e}{q_level:<16.6f}"
              f"{np.asarray(res['q_treated_ext'])[i]:16.4f}"
              f"{np.asarray(res['q_control_ext'])[i]:16.4f}"
              f"{np.asarray(res['qte_ext'])[i]:16.4f}")
