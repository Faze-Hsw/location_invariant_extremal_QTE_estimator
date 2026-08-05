# -*- coding: utf-8 -*-
"""一般经验分位数处理效应 (QTE) 估计量（IPW 形式）。

对任意分位数水平 tau ∈ (0, 1)，分别估计处理组与对照组的经验分位数，
QTE(tau) = q_hat_1(tau) - q_hat_0(tau)。

IPW 加权经验分位数（与 estimate/estimate_quantile_empirical.py 一致）:
    处理组权重 w_i = D_i / pi_hat(X_i)，对照组权重 w_i = (1-D_i) / (1-pi_hat(X_i))

加权分位数的 argmin 等价于求解加权经验 CDF: 找 q 使 Σ w_i 1{Y_i <= q} = tau · Σ w_i。
实现: 排序 Y 后累积权重，用 searchsorted 找第一个累积权重 >= tau·total 的位置。

输入:  含 Y, D, pi_estimate 字段的 dict（一维数组）
输出:  给定 tau 下的处理组/对照组经验分位数及 QTE
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


def estimate_qte(data, tau):
    """估计一般经验分位数处理效应 QTE(tau) = q_hat_1(tau) - q_hat_0(tau)。

    data: dict，需含 Y, D, pi_estimate 三个一维字段
    tau : 分位数水平，标量或一维数组，取值 (0, 1)

    返回 dict {tau, q_treated, q_control, n_treated, n_control, qte}。
    tau 为数组时 q_treated/q_control/qte 也返回数组。
    """
    Y = np.asarray(data["Y"]).ravel()
    D = np.asarray(data["D"]).ravel()
    pi = np.asarray(data["pi_estimate"]).ravel()
    taus = np.atleast_1d(np.asarray(tau, dtype=float))

    # 截断避免除零
    eps = 1e-6
    pi_c = np.clip(pi, eps, 1.0 - eps)

    # 处理组 (D=1)
    mask_t = (D == 1)
    w_t = 1.0 / pi_c[mask_t]
    q_t = np.array([weighted_quantile(Y[mask_t], w_t, t) for t in taus])

    # 对照组 (D=0)
    mask_c = (D == 0)
    w_c = 1.0 / (1.0 - pi_c[mask_c])
    q_c = np.array([weighted_quantile(Y[mask_c], w_c, t) for t in taus])

    qte = q_t - q_c

    def _scalar_or_array(arr):
        return arr[0] if arr.size == 1 else arr

    return {
        "tau": _scalar_or_array(taus),
        "q_treated": _scalar_or_array(q_t),
        "q_control": _scalar_or_array(q_c),
        "n_treated": int(mask_t.sum()),
        "n_control": int(mask_c.sum()),
        "qte": _scalar_or_array(qte),
    }


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
    from data_generation import load_config, generate_dataset, tau_levels
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "estimate"))
    from estimate_propensity_sieve import estimate_propensity_sieve

    cfg = load_config()
    seed = cfg["experiment"]["random_seed"]
    first_model = list(cfg["outcome_models"])[0]
    first_n = cfg["design"]["sample_sizes"][0]

    data = generate_dataset(cfg, first_model, first_n, seed)
    data, h_n, info = estimate_propensity_sieve(data)

    # 分位数水平统一取自配置文件，仅保留极端水平 tau_n_*（alpha_n/beta_n 为辅助水平，不在此估计）
    # 配置文件中的 tau_n 是上尾概率，对应分位数水平 1-tau_n
    levels = [(name, tau) for name, tau in tau_levels(cfg, first_n)
              if name.startswith("tau_n")]
    quantile_levels_used = [1.0 - tau for _, tau in levels]

    print("=" * 84)
    print(f"[测试] 模型={first_model}, n={first_n}, 筛基 h_n={h_n}")
    print(f"  处理组 n1={int((data['D'] == 1).sum())}, "
          f"对照组 n0={int((data['D'] == 0).sum())}")

    print("\n  [IPW 加权经验分位数]（水平取自配置文件，显示 1-tau 对应的分位数水平）")
    print(f"  {'name':<12}{'tau(上尾)':<14}{'1-tau':<16}"
          f"{'q_treated':>16}{'q_control':>16}{'QTE':>16}")
    print(f"  {'-' * 90}")
    for (name, t), q_level in zip(levels, quantile_levels_used):
        res = estimate_qte(data, q_level)
        print(f"  {name:<12}{t:<14.3e}{q_level:<16.6f}"
              f"{res['q_treated']:16.4f}{res['q_control']:16.4f}{res['qte']:16.4f}")

    # 数组输入验证
    res_arr = estimate_qte(data, quantile_levels_used)
    print(f"\n  数组输入验证: 分位数水平形状={np.shape(res_arr['tau'])}, "
          f"QTE 数组={np.array2string(np.asarray(res_arr['qte']), precision=4)}")
