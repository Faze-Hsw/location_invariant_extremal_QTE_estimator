# -*- coding: utf-8 -*-
"""Causal Hill 极值指数（EVI）估计量。

基于论文公式 (7)：

    γ̂_1^H := (1/(n·τ_n)) Σ_i [log(Y_i) - log(q̂_1(1-τ_n))]
             · [D_i / π̂(X_i)] · 1{Y_i > q̂_1(1-τ_n)}

    γ̂_0^H := (1/(n·τ_n)) Σ_i [log(Y_i) - log(q̂_0(1-τ_n))]
             · [(1-D_i) / (1-π̂(X_i))] · 1{Y_i > q̂_0(1-τ_n)}

图里的 τ_n 对应本仓库配置中的 α_n（中间分位数水平）。

输入:  含 Y, D, pi_estimate 字段的 dict（一维数组）
输出:  处理组与对照组的 Hill EVI 估计 γ̂_1, γ̂_0
"""
from pathlib import Path
import sys

import numpy as np


def _weighted_quantile(Y, weights, tau):
    """加权经验分位数（与 estimate_quantile_empirical 保持一致）。"""
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


def estimate_evi_causal_hill(data, alpha_n):
    """估计处理组与对照组的 Causal Hill EVI。

    data   : dict，含 Y, D, pi_estimate
    alpha_n: 分位数水平 τ_n（对应上尾概率），图里记为 τ_n

    返回 dict {alpha_n, q_treated, q_control, gamma_treated, gamma_control}。
    """
    Y = np.asarray(data["Y"]).ravel()
    D = np.asarray(data["D"]).ravel()
    pi = np.asarray(data["pi_estimate"]).ravel()
    n = Y.size

    eps = 1e-6
    pi_c = np.clip(pi, eps, 1.0 - eps)

    tau = 1.0 - alpha_n

    # 处理组阈值 q̂_1(1-τ_n)
    mask_t = (D == 1)
    w_t = 1.0 / pi_c[mask_t]
    q1 = _weighted_quantile(Y[mask_t], w_t, tau)

    # 对照组阈值 q̂_0(1-τ_n)
    mask_c = (D == 0)
    w_c = 1.0 / (1.0 - pi_c[mask_c])
    q0 = _weighted_quantile(Y[mask_c], w_c, tau)

    gamma_treated = np.nan
    gamma_control = np.nan

    # 处理组 Hill
    if q1 > 0 and not np.isnan(q1):
        indicator1 = (Y > q1) & mask_t
        # 上尾观测必须为正才能取 log；若 q1>0 则这些 Y_i>q1>0
        log_term1 = np.log(Y[indicator1]) - np.log(q1)
        weights1 = D[indicator1] / pi_c[indicator1]
        gamma_treated = float(np.sum(weights1 * log_term1) / (n * alpha_n))

    # 对照组 Hill
    if q0 > 0 and not np.isnan(q0):
        indicator0 = (Y > q0) & (~mask_t)
        log_term0 = np.log(Y[indicator0]) - np.log(q0)
        weights0 = (1 - D[indicator0]) / (1.0 - pi_c[indicator0])
        gamma_control = float(np.sum(weights0 * log_term0) / (n * alpha_n))

    return {
        "alpha_n": float(alpha_n),
        "q_treated": float(q1),
        "q_control": float(q0),
        "gamma_treated": gamma_treated,
        "gamma_control": gamma_control,
    }


def estimate_evi_for_config(cfg, data, n):
    """根据配置中 alpha_n 公式计算 Hill EVI。"""
    alpha_n = None
    for q in cfg["design"]["quantile_levels"]:
        if q["name"] == "alpha_n":
            alpha_n = eval(q["formula"], {"n": n, "log": np.log})
            break
    if alpha_n is None:
        raise ValueError("配置中缺少 alpha_n 分位数水平")
    return estimate_evi_causal_hill(data, alpha_n)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "estimate"))

    from data_generation import load_config, generate_dataset
    from estimate_propensity_sieve import estimate_propensity_sieve

    cfg = load_config()
    seed = cfg["experiment"]["random_seed"]

    print("=" * 72)
    print("Causal Hill EVI 估计（公式 7，tau_n = alpha_n）")
    print("=" * 72)

    for model in cfg["outcome_models"]:
        for n in cfg["design"]["sample_sizes"]:
            data = generate_dataset(cfg, model, n, seed)
            data, h_n, _ = estimate_propensity_sieve(data)
            res = estimate_evi_for_config(cfg, data, n)

            theory = cfg["outcome_models"][model]["evi"]
            print(f"\n[模型={model}, n={n}, h_n={h_n}]")
            print(f"  alpha_n(tau_n)={res['alpha_n']:.4e}")
            print(f"  q_hat_1(1-alpha)={res['q_treated']:12.3f}, "
                  f"q_hat_0(1-alpha)={res['q_control']:12.3f}")
            print(f"  gamma_hat_1^H = {res['gamma_treated']:8.4f}  "
                  f"(理论 gamma_1 = {theory['gamma_1']:.4f})")
            print(f"  gamma_hat_0^H = {res['gamma_control']:8.4f}  "
                  f"(理论 gamma_0 = {theory['gamma_0']:.4f})")
