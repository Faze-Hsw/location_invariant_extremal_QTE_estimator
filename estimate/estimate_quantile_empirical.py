# -*- coding: utf-8 -*-
"""Empirical quantile estimator (IPW-weighted).

Formula (from the paper):
    q_hat_j(tau) = argmin_q Σ_i [D_i/π̂(X_i)]^j [(1-D_i)/(1-π̂(X_i))]^(1-j)
                                  · (Y_i - q)(tau - 1{Y_i ≤ q})
    j=1: treated-group IPW quantile,  weight w_i = D_i / π̂(X_i)
    j=0: control-group IPW quantile,  weight w_i = (1-D_i) / (1-π̂(X_i))

The argmin is equivalent to solving the weighted empirical CDF: find q such that
Σ w_i 1{Y_i ≤ q} = tau · Σ w_i.
Implementation: sort Y and accumulate weights, then use searchsorted to find the first position
where the cumulative weight ≥ tau·total.

Input:  dict containing Y, D, pi_estimate fields (1-D arrays)
Output: empirical quantile estimates of the treated and control groups at the given tau
"""
from pathlib import Path
import sys

import numpy as np


def weighted_quantile(Y, weights, tau):
    """Weighted empirical quantile: the smallest sorted Y value with cum_w(q)/total_w >= tau."""
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
    """Estimate the IPW quantiles of the treated group (j=1) and control group (j=0) at quantile level tau.

    data: must contain the three 1-D fields Y, D, pi_estimate
    tau:  target quantile level, scalar, in (0, 1)

    Returns dict {tau, q_treated, q_control, n_treated, n_control, qte}.
    """
    Y = np.asarray(data["Y"]).ravel()
    D = np.asarray(data["D"]).ravel()
    pi = np.asarray(data["pi_estimate"]).ravel()

    # truncate to avoid division by zero
    eps = 1e-6
    pi_c = np.clip(pi, eps, 1.0 - eps)

    # treated group (j=1)
    mask_t = (D == 1)
    w_t = 1.0 / pi_c[mask_t]
    q_treated = weighted_quantile(Y[mask_t], w_t, tau)

    # control group (j=0)
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
    print(f"[test] model={first_model}, n={first_n}")
    print(f"  sieve basis h_n={h_n}, treated group={int((data['D']==1).sum())}, "
          f"control group={int((data['D']==0).sum())}")

    print(f"\n  {'τ_n':<22}{'q_treated':>14}{'q_control':>14}{'QTE':>14}")
    print(f"  {'-' * 64}")
    for name, tau in tau_levels(cfg, first_n):
        res = estimate_quantile_ipw(data, tau)
        print(f"  {name} (τ={tau:.2e})  {res['q_treated']:12.3f}  "
              f"{res['q_control']:12.3f}  {res['qte']:12.3f}")
