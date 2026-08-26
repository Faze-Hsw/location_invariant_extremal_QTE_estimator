# -*- coding: utf-8 -*-
"""Causal Hill extreme value index (EVI) estimator.

Based on equation (7) in the paper:

    γ̂_1^H := (1/(n·α_n)) Σ_i [log(Y_i) - log(q̂_1(1-α_n))]
             · [D_i / π̂(X_i)] · 1{Y_i > q̂_1(1-α_n)}

    γ̂_0^H := (1/(n·α_n)) Σ_i [log(Y_i) - log(q̂_0(1-α_n))]
             · [(1-D_i) / (1-π̂(X_i))] · 1{Y_i > q̂_0(1-α_n)}

α_n is the intermediate quantile level (config design.alpha_n, α_n = n^{0.65}/n).

Input:  dict with fields Y, D, pi_estimate (1-D arrays)
Output: Hill EVI estimates γ̂_1, γ̂_0 for the treated and control groups
"""
from pathlib import Path
import sys

import numpy as np


def _weighted_quantile(Y, weights, tau):
    """Weighted empirical quantile (consistent with estimate_quantile_empirical)."""
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
    """Estimate the Causal Hill EVI for the treated and control groups.

    data   : dict containing Y, D, pi_estimate
    alpha_n: intermediate quantile level α_n (upper-tail probability, config design.alpha_n)

    Returns dict {alpha_n, q_treated, q_control, gamma_treated, gamma_control}.
    """
    Y = np.asarray(data["Y"]).ravel()
    D = np.asarray(data["D"]).ravel()
    pi = np.asarray(data["pi_estimate"]).ravel()
    n = Y.size

    eps = 1e-6
    pi_c = np.clip(pi, eps, 1.0 - eps)

    tau = 1.0 - alpha_n

    # treated-group threshold q̂_1(1-τ_n)
    mask_t = (D == 1)
    w_t = 1.0 / pi_c[mask_t]
    q1 = _weighted_quantile(Y[mask_t], w_t, tau)

    # control-group threshold q̂_0(1-τ_n)
    mask_c = (D == 0)
    w_c = 1.0 / (1.0 - pi_c[mask_c])
    q0 = _weighted_quantile(Y[mask_c], w_c, tau)

    gamma_treated = np.nan
    gamma_control = np.nan

    # treated-group Hill
    if q1 > 0 and not np.isnan(q1):
        indicator1 = (Y > q1) & mask_t
        # upper-tail observations must be positive to take log; if q1>0 then these Y_i>q1>0
        log_term1 = np.log(Y[indicator1]) - np.log(q1)
        weights1 = D[indicator1] / pi_c[indicator1]
        gamma_treated = float(np.sum(weights1 * log_term1) / (n * alpha_n))

    # control-group Hill
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
    """Compute the Hill EVI using the alpha_n formula from the config."""
    alpha_n = None
    for q in cfg["design"]["quantile_levels"]:
        if q["name"] == "alpha_n":
            alpha_n = eval(q["formula"], {"n": n, "log": np.log})
            break
    if alpha_n is None:
        raise ValueError("Missing alpha_n quantile level in config")
    return estimate_evi_causal_hill(data, alpha_n)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "estimate"))

    from data_generation import load_config, generate_dataset
    from estimate_propensity_sieve import estimate_propensity_sieve

    cfg = load_config()
    seed = cfg["experiment"]["random_seed"]

    print("=" * 72)
    print("Causal Hill EVI estimation (equation 7, tau_n = alpha_n)")
    print("=" * 72)

    for model in cfg["outcome_models"]:
        for n in cfg["design"]["sample_sizes"]:
            data = generate_dataset(cfg, model, n, seed)
            data, h_n, _ = estimate_propensity_sieve(data)
            res = estimate_evi_for_config(cfg, data, n)

            theory = cfg["outcome_models"][model]["evi"]
            print(f"\n[model={model}, n={n}, h_n={h_n}]")
            print(f"  alpha_n(tau_n)={res['alpha_n']:.4e}")
            print(f"  q_hat_1(1-alpha)={res['q_treated']:12.3f}, "
                  f"q_hat_0(1-alpha)={res['q_control']:12.3f}")
            print(f"  gamma_hat_1^H = {res['gamma_treated']:8.4f}  "
                  f"(theoretical gamma_1 = {theory['gamma_1']:.4f})")
            print(f"  gamma_hat_0^H = {res['gamma_control']:8.4f}  "
                  f"(theoretical gamma_0 = {theory['gamma_0']:.4f})")
