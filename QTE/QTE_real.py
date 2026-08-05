# -*- coding: utf-8 -*-
"""真实（理论）分位数处理效应 QTE。

复用 estimate/estimate_quantile_real.py：对潜在结果 Y_j 的边际分布
F_{Y_j}(q) = ∫_0^1 F_{Y_j|X=x}(q) dx 做数值积分，再用 brentq 解
F(q) = τ，得到真实分位数 q_{Y_j}(τ)。

    真实 QTE(τ) = q_{Y1}(τ) - q_{Y0}(τ)

配置中的 tau_n 是上尾概率，对应分位数水平 1-tau_n，与
Zhang.py（经验）/ Deuber.py / QTE_Fraga*.py（外推）在相同水平上可比。

输入: 配置 + 分位数水平 τ（标量或数组）
输出: 各模型在水平 τ 下的真实分位数与真实 QTE
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "estimate"))
from estimate_quantile_real import load_config, real_quantile  # noqa: E402


def real_qte(cfg, model, tau):
    """真实 QTE(τ) = q_{Y1}(τ) - q_{Y0}(τ)。

    cfg  : 配置 dict
    model: 结果模型名（H1/H2/H3）
    tau  : 分位数水平，标量或一维数组，取值 (0, 1)

    返回 dict {tau, q_treated, q_control, qte}。
    tau 为数组时 q_treated/q_control/qte 也返回数组。
    """
    taus = np.atleast_1d(np.asarray(tau, dtype=float))
    q_t = np.array([real_quantile(cfg, model, 1, t) for t in taus])
    q_c = np.array([real_quantile(cfg, model, 0, t) for t in taus])
    qte = q_t - q_c

    def _scalar_or_array(arr):
        return arr[0] if arr.size == 1 else arr

    return {
        "tau": _scalar_or_array(taus),
        "q_treated": _scalar_or_array(q_t),
        "q_control": _scalar_or_array(q_c),
        "qte": _scalar_or_array(qte),
    }


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
    from data_generation import tau_levels

    cfg = load_config()
    first_n = cfg["design"]["sample_sizes"][0]

    # 分位数水平统一取自配置文件，仅保留极端水平 tau_n_*（alpha_n/beta_n 为辅助水平）
    # 配置中的 tau_n 是上尾概率，对应分位数水平 1-tau_n
    levels = [(name, tau) for name, tau in tau_levels(cfg, first_n)
              if name.startswith("tau_n")]

    print("=" * 92)
    print(f"真实（理论）QTE  n={first_n}（水平取自配置文件，显示 1-tau 对应的分位数水平）")
    print("=" * 92)

    for model in cfg["outcome_models"]:
        print(f"\n[{model}]")
        print(f"  {'name':<12}{'tau(上尾)':<14}{'1-tau':<16}"
              f"{'q_treated':>16}{'q_control':>16}{'QTE':>16}")
        print(f"  {'-' * 90}")
        for name, t in levels:
            q_level = 1.0 - t
            res = real_qte(cfg, model, q_level)
            print(f"  {name:<12}{t:<14.3e}{q_level:<16.6f}"
                  f"{res['q_treated']:16.4f}{res['q_control']:16.4f}"
                  f"{res['qte']:16.4f}")
