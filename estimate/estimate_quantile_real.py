# -*- coding: utf-8 -*-
"""计算各模型下潜在结果 Y1(处理组) / Y0(对照组) 的真实（理论）分位数。

真实分位数 q_{Y_j}(τ) 满足 P(Y_j <= q) = τ。Y_j 的边际分布是 X~U[0,1]
上的条件分布混合:

    F_{Y_j}(q) = ∫_0^1 F_{Y_j|X=x}(q) dx

用 scipy.integrate.quad 数值积分得到边际 CDF，再用 brentq 对 F(q) - τ = 0
求根。可用于与 IPW 经验分位数估计 (estimate_quantile_empirical.py) 对比，
衡量偏差、方差与覆盖率。

输入: 目标分位数 τ（标量或数组）
输出: H1/H2/H3 模型下对照组 (j=0) 与处理组 (j=1) 的真实分位数
"""
from pathlib import Path

import numpy as np
import yaml
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy import stats

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "data_generation.yaml"


def load_config(path: str = CONFIG_PATH) -> dict:
    """读取数据生成配置文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def marginal_cdf(cfg: dict, model: str, j: int, q: float) -> float:
    """边际 CDF F_{Y_j}(q) = ∫_0^1 F_{Y_j|X=x}(q) dx。

    j=1 处理组 (Y1), j=0 对照组 (Y0)。
    """
    model_cfg = cfg["outcome_models"][model]
    mu = float(cfg.get("design", {}).get("mu", 0.0))   # 共享位置偏移 μ
    q0 = q - mu                                        # F_{Y}(q) = F_{原Y}(q - μ)

    if model == "H1":
        # Y(j) = μ + coef·S(1+X), S ~ t(df)
        coef = 5.0 if j == 1 else 1.0
        df = model_cfg["noise"]["df"]
        integrand = lambda x: stats.t.cdf(q0 / (coef * (1 + x)), df=df)

    elif model == "H2":
        # Y(j) = μ + C_s·exp(X), C_s~Fréchet(shape)
        shape = (model_cfg["Y1"]["noise"]["shape"] if j == 1
                 else model_cfg["Y0"]["noise"]["shape"])
        if q0 <= 0:
            return 0.0
        integrand = lambda x: stats.invweibull.cdf(q0 / np.exp(x), c=shape)

    elif model == "H3":
        # Y(j) = μ + P_{shape,scale}, Pareto(0, scale)
        if j == 1:
            shape_formula, scale = model_cfg["Y1"]["shape_formula"], model_cfg["Y1"]["scale"]
        else:
            shape_formula, scale = model_cfg["Y0"]["shape_formula"], model_cfg["Y0"]["scale"]
        if q0 <= scale:
            return 0.0
        integrand = lambda x: stats.pareto.cdf(
            q0, b=eval(shape_formula, {"X": x}), loc=0.0, scale=scale)

    else:
        raise ValueError(f"未知模型: {model}")

    val, _ = quad(integrand, 0.0, 1.0)
    return float(np.clip(val, 0.0, 1.0))


def real_quantile(cfg: dict, model: str, j: int, tau: float) -> float:
    """解 F_{Y_j}(q) = tau，返回真实分位数 q。"""
    def f(q):
        return marginal_cdf(cfg, model, j, q) - tau

    # 从 q=0 出发自动扩展 bracket
    if abs(f(0.0)) < 1e-14:
        return 0.0
    if f(0.0) > 0:  # 需向负方向扩展
        a, step = 0.0, 1.0
        while f(a) > 0 and a > -1e9:
            a -= step
            step *= 2.0
        return brentq(f, a, 0.0, xtol=1e-12, rtol=1e-12)
    # 需向正方向扩展
    b, step = 0.0, 1.0
    while f(b) < 0 and b < 1e9:
        b += step
        step *= 2.0
    return brentq(f, 0.0, b, xtol=1e-12, rtol=1e-12)


def compute_real_quantiles(cfg: dict, taus):
    """对全部模型与 j=0/1 计算真实分位数。

    返回 dict: result[model][j][tau] = q
    """
    result = {}
    for model in cfg["outcome_models"]:
        result[model] = {}
        for j, label in ((1, "treated"), (0, "control")):
            result[model][label] = {}
            for tau in taus:
                result[model][label][tau] = real_quantile(cfg, model, j, float(tau))
    return result


if __name__ == "__main__":
    cfg = load_config()
    n = cfg["design"]["sample_sizes"][0]

    # 目标分位数：由配置文件中的公式代入 n 得到（也可在此修改为自定义数组）
    taus = []
    for q in cfg["design"]["quantile_levels"]:
        taus.append(eval(q["formula"], {"n": n, "log": np.log}))

    print("=" * 76)
    print(f"真实（理论）分位数  n={n}")
    print("=" * 76)

    for model in cfg["outcome_models"]:
        print(f"\n[{model}]")
        print(f"  {'tau':<12}{'q_treated(真实)':>18}{'q_control(真实)':>18}{'真实QTE':>16}")
        print(f"  {'-' * 64}")
        for tau in taus:
            q1 = real_quantile(cfg, model, 1, tau)
            q0 = real_quantile(cfg, model, 0, tau)
            print(f"  {tau:.4e}    {q1:>14.4f}    {q0:>14.4f}    {q1 - q0:>12.4f}")
