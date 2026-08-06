# -*- coding: utf-8 -*-
"""自适应估计辅助分位数水平 k0（k0 = k^m）。

对每个 (模型, n) 的**处理组与对照组分别**估计 k0（不取统一值）：

流程：
  1. 初始取定 k0* = 2·k^{2/3}（k 来自配置 alpha_n，k = n^{0.65}）；
  2. 用 β_n = k0*/n 作辅助水平，估计 Causal Fraga EVI γ̂_j^F（处理组/对照组）；
  3. m*_j = 2·γ̂_j^F / (1 + 2·γ̂_j^F)；
  4. m_j = m*_j - σ（σ 为超参数，配置 design.k0_sigma）；
  5. k0_j = k^{m_j}，β_{n,j} = k0_j / n（处理组 j=1、对照组 j=0 各自取值）。

estimate_k0_by_group() 可作为库被 EVI_experiment.py / QTE_experiment.py 调用，
失败时的兜底 β_n 由 fallback_beta(cfg, n)（初始 β* = k0*/n）内置提供，
配置中不再需要手动指定 beta_n。独立运行时 __main__ 打印各 (模型, n, 组) 的
m、k0 与 β_n。

配置字段：
  design.k0_init_formula : 初始 k0* 的公式（默认 "2 * (n ** 0.65) ** (2 / 3)"）
  design.k0_sigma        : m = m* - σ 中的 σ（默认 0.01）
"""
from pathlib import Path
import sys
import warnings

import numpy as np

BASE = Path(__file__).resolve().parent
CONFIG_PATH = BASE / "configs" / "data_generation.yaml"

sys.path.insert(0, str(BASE / "data"))
sys.path.insert(0, str(BASE / "estimate"))
sys.path.insert(0, str(BASE / "EVI"))

from data_generation import load_config, generate_dataset, tau_levels  # noqa: E402
from estimate_propensity_sieve import estimate_propensity_sieve  # noqa: E402
from causal_fraga import estimate_evi_causal_fraga  # noqa: E402


def k0_init_star(cfg, n):
    """初始取定的 k0*（默认 2·k^{2/3}）。"""
    return eval(cfg["design"]["k0_init_formula"], {"n": n, "log": np.log})


def fallback_beta(cfg, n):
    """自适应估计失败（γ̂ 为 nan）时的兜底 β_n：取初始 β* = k0*/n。"""
    return float(k0_init_star(cfg, n)) / n


def estimate_k0_by_group(cfg, data, n):
    """基于含 pi_estimate 的数据，估计处理组/对照组各自的 m、k0 与 β_n。

    data: 生成的数据 dict（需已含 pi_estimate 字段）
    n   : 样本量

    某组 γ̂ 估计失败时，该组的 m/k0 为 nan、β_n 回退到 fallback_beta(cfg, n)。

    返回 dict {alpha_n, k0_star, sigma, gamma_treated, gamma_control,
              m_treated, m_control, k0_treated, k0_control,
              beta_treated, beta_control}。
    """
    alpha_n = dict(tau_levels(cfg, n))["alpha_n"]
    k = n * alpha_n                      # k = n^{0.65}
    k0_star = float(k0_init_star(cfg, n))
    sigma = float(cfg["design"]["k0_sigma"])
    m_lower = float(cfg["design"].get("k0_m_lower", 0.05))   # m 下界兜底
    beta_star = k0_star / n
    fb = fallback_beta(cfg, n)           # 兜底 β_n

    # 用初始 k0* 估计两组的 Causal Fraga EVI
    res = estimate_evi_causal_fraga(data, beta_star, alpha_n)
    g1 = float(res["gamma_treated"])
    g0 = float(res["gamma_control"])

    def _m_k0(gamma, tag):
        """m* = 2γ/(1+2γ)，m = m* - σ；m 低于下界 k0_m_lower 时截断到该下界。"""
        if not np.isfinite(gamma):
            warnings.warn(
                f"k0 自适应估计失败（gamma_{tag} 非有限，n={n}），"
                f"该组 β_n 回退到初始 β* = {fb:.4e}",
                RuntimeWarning,
            )
            return np.nan, np.nan
        m = 2.0 * gamma / (1.0 + 2.0 * gamma) - sigma
        if m < m_lower:
            m = m_lower                      # 下界兜底
        return m, k ** m

    m1, k0_1 = _m_k0(g1, "treated")
    m0, k0_0 = _m_k0(g0, "control")

    def _beta(k0_j):
        return float(k0_j) / n if np.isfinite(k0_j) else fb

    return {
        "alpha_n": float(alpha_n),
        "k0_star": k0_star,
        "sigma": sigma,
        "gamma_treated": g1,
        "gamma_control": g0,
        "m_treated": m1,
        "m_control": m0,
        "k0_treated": k0_1,
        "k0_control": k0_0,
        "beta_treated": _beta(k0_1),
        "beta_control": _beta(k0_0),
    }


if __name__ == "__main__":
    cfg = load_config()
    seed = cfg["experiment"]["random_seed"]
    sigma = float(cfg["design"]["k0_sigma"])

    print("=" * 88)
    print("k0 自适应估计（处理组/对照组分别估计）：k0 = k^m，m = m* - σ")
    print(f"  超参数 σ = {sigma}")
    print(f"  初始 k0* = {cfg['design']['k0_init_formula']}")
    print("=" * 88)

    results = {}
    for model in cfg["outcome_models"]:
        for n in cfg["design"]["sample_sizes"]:
            data = generate_dataset(cfg, model, n, seed)
            data, _h_n, _info = estimate_propensity_sieve(data)
            r = estimate_k0_by_group(cfg, data, n)
            results[(model, n)] = r
            m_star1 = (2 * r["gamma_treated"] / (1 + 2 * r["gamma_treated"])
                       if np.isfinite(r["gamma_treated"]) else np.nan)
            m_star0 = (2 * r["gamma_control"] / (1 + 2 * r["gamma_control"])
                       if np.isfinite(r["gamma_control"]) else np.nan)
            print(f"\n[{model}] n={n}")
            print(f"  k0* = {r['k0_star']:.3f}  (β_n* = {r['k0_star'] / n:.4e}, α_n = {r['alpha_n']:.4e})")
            print(f"  处理组: gamma_1 = {r['gamma_treated']:.4f} -> m*_1 = {m_star1:.4f} "
                  f"-> m_1 = {r['m_treated']:.4f} -> k0_1 = {r['k0_treated']:.3f} "
                  f"-> beta_n1 = {r['beta_treated']:.4e}")
            print(f"  对照组: gamma_0 = {r['gamma_control']:.4f} -> m*_0 = {m_star0:.4f} "
                  f"-> m_0 = {r['m_control']:.4f} -> k0_0 = {r['k0_control']:.3f} "
                  f"-> beta_n0 = {r['beta_control']:.4e}")

    print("\n" + "=" * 88)
    print(f"  兜底 β_n（自适应失败时回退）= fallback_beta(cfg, n) = 初始 β* = k0*/n")
    print("=" * 88)
