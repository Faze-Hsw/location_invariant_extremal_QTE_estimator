# -*- coding: utf-8 -*-
"""Candal–Fraga 极值指数（EVI）估计量。

基于论文公式 (8) 与 (9)：

    γ̂_1^F(β_n, α_n) = (1/(n·β_n)) Σ_i [D_i / π̂(X_i)]
                      · 1{Y_i > q̂_1(1-β_n)}
                      · log[(Y_i - q̂_1(1-α_n)) / (q̂_1(1-β_n) - q̂_1(1-α_n))]

    γ̂_0^F(β_n, α_n) = (1/(n·β_n)) Σ_i [(1-D_i) / (1-π̂(X_i))]
                      · 1{Y_i > q̂_0(1-β_n)}
                      · log[(Y_i - q̂_0(1-α_n)) / (q̂_0(1-β_n) - q̂_0(1-α_n))]

输入:  含 Y, D, pi_estimate 字段的 dict（一维数组）
输出:  处理组与对照组的 EVI 估计 γ̂_1, γ̂_0
"""
from pathlib import Path
import sys

import numpy as np


def _weighted_quantile(Y, weights, tau):
    """加权经验分位数（与 estimate_quantile_empirical 保持一致，避免循环导入）。"""
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


def estimate_evi_causal_fraga(data, beta_n, alpha_n, beta_treated=None, beta_control=None):
    """估计处理组与对照组的 Candal–Fraga EVI。

    data          : dict，含 Y, D, pi_estimate（与经验分位数脚本同结构）
    beta_n        : 辅助分位数水平（β_n），对应上尾概率；缺省用于两组
    alpha_n       : 中间分位数水平（α_n），用于构造阈值差
    beta_treated  : 可选，处理组各自的 β_n（分组 k0 时传入）
    beta_control  : 可选，对照组各自的 β_n（分组 k0 时传入）

    返回 dict {beta_n, alpha_n, beta_treated, beta_control,
              q_treated_beta, q_treated_alpha, q_control_beta, q_control_alpha,
              gamma_treated, gamma_control}。
    """
    Y = np.asarray(data["Y"]).ravel()
    D = np.asarray(data["D"]).ravel()
    pi = np.asarray(data["pi_estimate"]).ravel()
    n = Y.size

    beta_t = float(beta_treated) if beta_treated is not None else float(beta_n)
    beta_c = float(beta_control) if beta_control is not None else float(beta_n)

    eps = 1e-6
    pi_c = np.clip(pi, eps, 1.0 - eps)

    # 上尾分位数阈值（处理组/对照组可分别用各自的 β_n）
    tau_beta_t = 1.0 - beta_t
    tau_beta_c = 1.0 - beta_c
    tau_alpha = 1.0 - alpha_n

    # 处理组 (j=1)
    mask_t = (D == 1)
    w_t = 1.0 / pi_c[mask_t]
    q1_beta = _weighted_quantile(Y[mask_t], w_t, tau_beta_t)
    q1_alpha = _weighted_quantile(Y[mask_t], w_t, tau_alpha)

    # 对照组 (j=0)
    mask_c = (D == 0)
    w_c = 1.0 / (1.0 - pi_c[mask_c])
    q0_beta = _weighted_quantile(Y[mask_c], w_c, tau_beta_c)
    q0_alpha = _weighted_quantile(Y[mask_c], w_c, tau_alpha)

    denom1 = q1_beta - q1_alpha
    denom0 = q0_beta - q0_alpha

    gamma_treated = np.nan
    gamma_control = np.nan

    # 处理组 EVI
    if denom1 > 0 and not np.isnan(q1_beta):
        indicator1 = (Y > q1_beta) & mask_t
        weights1 = D[indicator1] / pi_c[indicator1]
        log_term1 = np.log((Y[indicator1] - q1_alpha) / denom1)
        gamma_treated = float(np.sum(weights1 * log_term1) / (n * beta_t))

    # 对照组 EVI
    if denom0 > 0 and not np.isnan(q0_beta):
        indicator0 = (Y > q0_beta) & (~mask_t)
        weights0 = (1 - D[indicator0]) / (1.0 - pi_c[indicator0])
        log_term0 = np.log((Y[indicator0] - q0_alpha) / denom0)
        gamma_control = float(np.sum(weights0 * log_term0) / (n * beta_c))

    return {
        "beta_n": float(beta_n),
        "alpha_n": float(alpha_n),
        "beta_treated": float(beta_t),
        "beta_control": float(beta_c),
        "q_treated_beta": float(q1_beta),
        "q_treated_alpha": float(q1_alpha),
        "q_control_beta": float(q0_beta),
        "q_control_alpha": float(q0_alpha),
        "gamma_treated": gamma_treated,
        "gamma_control": gamma_control,
    }


def estimate_evi_for_config(cfg, data, n):
    """根据配置中 beta_n / alpha_n 公式，一次性计算 EVI 估计。"""
    beta_n = None
    alpha_n = None
    for q in cfg["design"]["quantile_levels"]:
        if q["name"] == "beta_n":
            beta_n = eval(q["formula"], {"n": n, "log": np.log})
        if q["name"] == "alpha_n":
            alpha_n = eval(q["formula"], {"n": n, "log": np.log})
    if beta_n is None or alpha_n is None:
        raise ValueError("配置中缺少 beta_n 或 alpha_n 分位数水平")
    return estimate_evi_causal_fraga(data, beta_n, alpha_n)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "estimate"))

    from data_generation import load_config, generate_dataset
    from estimate_propensity_sieve import estimate_propensity_sieve

    cfg = load_config()
    seed = cfg["experiment"]["random_seed"]

    print("=" * 72)
    print("Candal-Fraga EVI 估计（公式 8 & 9）")
    print("=" * 72)

    for model in cfg["outcome_models"]:
        for n in cfg["design"]["sample_sizes"]:
            data = generate_dataset(cfg, model, n, seed)
            data, h_n, _ = estimate_propensity_sieve(data)
            res = estimate_evi_for_config(cfg, data, n)

            theory = cfg["outcome_models"][model]["evi"]
            print(f"\n[模型={model}, n={n}, h_n={h_n}]")
            print(f"  β_n={res['beta_n']:.4e}, α_n={res['alpha_n']:.4e}")
            print(f"  q_hat_1(1-beta)={res['q_treated_beta']:12.3f}, "
                  f"q_hat_1(1-alpha)={res['q_treated_alpha']:12.3f}")
            print(f"  q_hat_0(1-beta)={res['q_control_beta']:12.3f}, "
                  f"q_hat_0(1-alpha)={res['q_control_alpha']:12.3f}")
            print(f"  gamma_hat_1^F = {res['gamma_treated']:8.4f}  "
                  f"(理论 gamma_1 = {theory['gamma_1']:.4f})")
            print(f"  gamma_hat_0^F = {res['gamma_control']:8.4f}  "
                  f"(理论 gamma_0 = {theory['gamma_0']:.4f})")
