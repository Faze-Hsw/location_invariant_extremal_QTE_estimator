# -*- coding: utf-8 -*-
"""Compute the true (theoretical) quantiles of the potential outcomes Y1 (treated) / Y0 (control)
under each model.

The true quantile q_{Y_j}(τ) satisfies P(Y_j <= q) = τ. The marginal distribution of Y_j is the
mixture of the conditional distributions over X~U[0,1]:

    F_{Y_j}(q) = ∫_0^1 F_{Y_j|X=x}(q) dx

Numerically integrate with scipy.integrate.quad to get the marginal CDF, then find the root of
F(q) - τ = 0 with brentq. Can be used to compare against the IPW empirical quantile estimator
(estimate_quantile_empirical.py) to measure bias, variance and coverage.

Input:  target quantile τ (scalar or array)
Output: true quantiles of the control group (j=0) and treated group (j=1) under models H1/H2/H3
"""
from pathlib import Path

import numpy as np
import yaml
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy import stats

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "data_generation.yaml"


def load_config(path: str = CONFIG_PATH) -> dict:
    """Load the data generation config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def marginal_cdf(cfg: dict, model: str, j: int, q: float) -> float:
    """Marginal CDF F_{Y_j}(q) = ∫_0^1 F_{Y_j|X=x}(q) dx.

    j=1 treated group (Y1), j=0 control group (Y0).
    """
    model_cfg = cfg["outcome_models"][model]
    mu = float(cfg.get("design", {}).get("mu", 0.0))   # shared location shift μ
    q0 = q - mu                                        # F_{Y}(q) = F_{original Y}(q - μ)

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
        raise ValueError(f"Unknown model: {model}")

    val, _ = quad(integrand, 0.0, 1.0)
    return float(np.clip(val, 0.0, 1.0))


def real_quantile(cfg: dict, model: str, j: int, tau: float) -> float:
    """Solve F_{Y_j}(q) = tau and return the true quantile q."""
    def f(q):
        return marginal_cdf(cfg, model, j, q) - tau

    # automatically expand the bracket starting from q=0
    if abs(f(0.0)) < 1e-14:
        return 0.0
    if f(0.0) > 0:  # need to expand in the negative direction
        a, step = 0.0, 1.0
        while f(a) > 0 and a > -1e9:
            a -= step
            step *= 2.0
        return brentq(f, a, 0.0, xtol=1e-12, rtol=1e-12)
    # need to expand in the positive direction
    b, step = 0.0, 1.0
    while f(b) < 0 and b < 1e9:
        b += step
        step *= 2.0
    return brentq(f, 0.0, b, xtol=1e-12, rtol=1e-12)


def compute_real_quantiles(cfg: dict, taus):
    """Compute the true quantiles for all models and j=0/1.

    Returns dict: result[model][j][tau] = q
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

    # target quantiles: obtained by substituting n into the config formulas (may also be changed
    # to a custom array here)
    taus = []
    for q in cfg["design"]["quantile_levels"]:
        taus.append(eval(q["formula"], {"n": n, "log": np.log}))

    print("=" * 76)
    print(f"true (theoretical) quantiles  n={n}")
    print("=" * 76)

    for model in cfg["outcome_models"]:
        print(f"\n[{model}]")
        print(f"  {'tau':<12}{'q_treated(true)':>18}{'q_control(true)':>18}{'true QTE':>16}")
        print(f"  {'-' * 64}")
        for tau in taus:
            q1 = real_quantile(cfg, model, 1, tau)
            q0 = real_quantile(cfg, model, 0, tau)
            print(f"  {tau:.4e}    {q1:>14.4f}    {q0:>14.4f}    {q1 - q0:>12.4f}")
