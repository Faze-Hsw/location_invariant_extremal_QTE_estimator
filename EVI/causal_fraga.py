# -*- coding: utf-8 -*-
"""Candal–Fraga extreme value index (EVI) estimator.

Based on equations (8) and (9) in the paper:

    γ̂_1^F(β_n, α_n) = (1/(n·β_n)) Σ_i [D_i / π̂(X_i)]
                      · 1{Y_i > q̂_1(1-β_n)}
                      · log[(Y_i - q̂_1(1-α_n)) / (q̂_1(1-β_n) - q̂_1(1-α_n))]

    γ̂_0^F(β_n, α_n) = (1/(n·β_n)) Σ_i [(1-D_i) / (1-π̂(X_i))]
                      · 1{Y_i > q̂_0(1-β_n)}
                      · log[(Y_i - q̂_0(1-α_n)) / (q̂_0(1-β_n) - q̂_0(1-α_n))]

Input:  dict with fields Y, D, pi_estimate (1-D arrays)
Output: EVI estimates γ̂_1, γ̂_0 for the treated and control groups
"""
from pathlib import Path
import sys

import numpy as np


def _weighted_quantile(Y, weights, tau):
    """Weighted empirical quantile (consistent with estimate_quantile_empirical, avoiding circular imports)."""
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
    """Estimate the Candal–Fraga EVI for the treated and control groups.

    data          : dict containing Y, D, pi_estimate (same structure as the empirical quantile script)
    beta_n        : auxiliary quantile level (β_n), an upper-tail probability; used for both groups by default
    alpha_n       : intermediate quantile level (α_n), used to construct the threshold difference
    beta_treated  : optional, treated-group β_n (passed in for group-specific k0)
    beta_control  : optional, control-group β_n (passed in for group-specific k0)

    Returns dict {beta_n, alpha_n, beta_treated, beta_control,
              q_treated_beta, q_treated_alpha, q_control_beta, q_control_alpha,
              gamma_treated, gamma_control}.
    """
    Y = np.asarray(data["Y"]).ravel()
    D = np.asarray(data["D"]).ravel()
    pi = np.asarray(data["pi_estimate"]).ravel()
    n = Y.size

    beta_t = float(beta_treated) if beta_treated is not None else float(beta_n)
    beta_c = float(beta_control) if beta_control is not None else float(beta_n)

    eps = 1e-6
    pi_c = np.clip(pi, eps, 1.0 - eps)

    # upper-tail quantile thresholds (the treated/control groups may use their own β_n)
    tau_beta_t = 1.0 - beta_t
    tau_beta_c = 1.0 - beta_c
    tau_alpha = 1.0 - alpha_n

    # treated group (j=1)
    mask_t = (D == 1)
    w_t = 1.0 / pi_c[mask_t]
    q1_beta = _weighted_quantile(Y[mask_t], w_t, tau_beta_t)
    q1_alpha = _weighted_quantile(Y[mask_t], w_t, tau_alpha)

    # control group (j=0)
    mask_c = (D == 0)
    w_c = 1.0 / (1.0 - pi_c[mask_c])
    q0_beta = _weighted_quantile(Y[mask_c], w_c, tau_beta_c)
    q0_alpha = _weighted_quantile(Y[mask_c], w_c, tau_alpha)

    denom1 = q1_beta - q1_alpha
    denom0 = q0_beta - q0_alpha

    gamma_treated = np.nan
    gamma_control = np.nan

    # treated-group EVI
    if denom1 > 0 and not np.isnan(q1_beta):
        indicator1 = (Y > q1_beta) & mask_t
        weights1 = D[indicator1] / pi_c[indicator1]
        log_term1 = np.log((Y[indicator1] - q1_alpha) / denom1)
        gamma_treated = float(np.sum(weights1 * log_term1) / (n * beta_t))

    # control-group EVI
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
    """Compute the EVI estimate in one go using fallback_beta as β_n and the alpha_n formula from the config."""
    # deferred import to avoid a circular dependency with estimate_k0
    from estimate_k0 import fallback_beta

    alpha_n = None
    for q in cfg["design"]["quantile_levels"]:
        if q["name"] == "alpha_n":
            alpha_n = eval(q["formula"], {"n": n, "log": np.log})
            break
    if alpha_n is None:
        raise ValueError("Missing alpha_n quantile level in config")
    beta_n = fallback_beta(cfg, n)
    return estimate_evi_causal_fraga(data, beta_n, alpha_n)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "estimate"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from data_generation import load_config, generate_dataset
    from estimate_propensity_sieve import estimate_propensity_sieve

    cfg = load_config()
    seed = cfg["experiment"]["random_seed"]

    print("=" * 72)
    print("Candal-Fraga EVI estimation (equations 8 & 9)")
    print("=" * 72)

    for model in cfg["outcome_models"]:
        for n in cfg["design"]["sample_sizes"]:
            data = generate_dataset(cfg, model, n, seed)
            data, h_n, _ = estimate_propensity_sieve(data)
            res = estimate_evi_for_config(cfg, data, n)

            theory = cfg["outcome_models"][model]["evi"]
            print(f"\n[model={model}, n={n}, h_n={h_n}]")
            print(f"  β_n={res['beta_n']:.4e}, α_n={res['alpha_n']:.4e}")
            print(f"  q_hat_1(1-beta)={res['q_treated_beta']:12.3f}, "
                  f"q_hat_1(1-alpha)={res['q_treated_alpha']:12.3f}")
            print(f"  q_hat_0(1-beta)={res['q_control_beta']:12.3f}, "
                  f"q_hat_0(1-alpha)={res['q_control_alpha']:12.3f}")
            print(f"  gamma_hat_1^F = {res['gamma_treated']:8.4f}  "
                  f"(theoretical gamma_1 = {theory['gamma_1']:.4f})")
            print(f"  gamma_hat_0^F = {res['gamma_control']:8.4f}  "
                  f"(theoretical gamma_0 = {theory['gamma_0']:.4f})")
