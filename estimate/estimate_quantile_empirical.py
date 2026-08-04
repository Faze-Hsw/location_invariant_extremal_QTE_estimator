# -*- coding: utf-8 -*-
"""经验分位数估计量（IPW 加权）。

公式 (来自论文):
    q_hat_j(tau) = argmin_q Σ_i [D_i/π̂(X_i)]^j [(1-D_i)/(1-π̂(X_i))]^(1-j)
                                  · (Y_i - q)(tau - 1{Y_i ≤ q})
    j=1: 处理组 IPW 分位数,  权重 w_i = D_i / π̂(X_i)
    j=0: 对照组 IPW 分位数,  权重 w_i = (1-D_i) / (1-π̂(X_i))

argmin 等价于求解加权经验 CDF:  找 q 使  Σ w_i 1{Y_i ≤ q} = tau · Σ w_i
实现: 排序 Y 后累积权重, 用 searchsorted 找第一个累积权重 ≥ tau·total 的位置。

输入:  含 Y, D, pi_estimate 字段的 dict（一维数组）
输出:  给定 tau 下的处理组和对照组经验分位数估计
"""
from pathlib import Path
import sys

import numpy as np


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


def estimate_quantile_ipw(data, tau):
    """估计处理组 (j=1) 和对照组 (j=0) 在分位数水平 tau 下的 IPW 分位数。

    data: 需包含 Y, D, pi_estimate 三个一维字段
    tau:  目标分位数水平，标量，取值 (0, 1)

    返回 dict {tau, q_treated, q_control, n_treated, n_control, qte}。
    """
    Y = np.asarray(data["Y"]).ravel()
    D = np.asarray(data["D"]).ravel()
    pi = np.asarray(data["pi_estimate"]).ravel()

    # 截断避免除零
    eps = 1e-6
    pi_c = np.clip(pi, eps, 1.0 - eps)

    # 处理组 (j=1)
    mask_t = (D == 1)
    w_t = 1.0 / pi_c[mask_t]
    q_treated = weighted_quantile(Y[mask_t], w_t, tau)

    # 对照组 (j=0)
    mask_c = (D == 0)
    w_c = 1.0 / (1.0 - pi_c[mask_c])
    q_control = weighted_quantile(Y[mask_c], w_c, tau)

    return {
        "tau": float(tau),
        "q_treated": q_treated,
        "q_control": q_control,
        "n_treated": int(mask_t.sum()),
        "n_control": int(mask_c.sum()),
        "qte": q_treated - q_control,
    }


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
    from data_generation import load_config, generate_dataset, tau_levels
    from estimate_propensity_sieve import estimate_propensity_sieve

    cfg = load_config()
    seed = cfg["experiment"]["random_seed"]
    first_model = list(cfg["outcome_models"])[0]
    first_n = cfg["design"]["sample_sizes"][0]

    data = generate_dataset(cfg, first_model, first_n, seed)
    data, h_n, info = estimate_propensity_sieve(data)

    print("=" * 72)
    print(f"[测试] 模型={first_model}, n={first_n}")
    print(f"  筛基 h_n={h_n}, 处理组={int((data['D']==1).sum())}, "
          f"对照组={int((data['D']==0).sum())}")

    print(f"\n  {'τ_n':<22}{'q_treated':>14}{'q_control':>14}{'QTE':>14}")
    print(f"  {'-' * 64}")
    for name, tau in tau_levels(cfg, first_n):
        res = estimate_quantile_ipw(data, tau)
        print(f"  {name} (τ={tau:.2e})  {res['q_treated']:12.3f}  "
              f"{res['q_control']:12.3f}  {res['qte']:12.3f}")
