# -*- coding: utf-8 -*-
"""Plain empirical quantile treatment effect (QTE) estimator (IPW form).

For any quantile level tau ∈ (0, 1), estimate the empirical quantiles of the
treated and control groups separately, QTE(tau) = q_hat_1(tau) - q_hat_0(tau).

IPW-weighted empirical quantile (consistent with estimate/estimate_quantile_empirical.py):
    treated-group weight w_i = D_i / pi_hat(X_i), control-group weight w_i = (1-D_i) / (1-pi_hat(X_i))

The argmin of the weighted quantile is equivalent to solving the weighted empirical CDF:
find q such that Σ w_i 1{Y_i <= q} = tau · Σ w_i.
Implementation: sort Y and accumulate weights, then use searchsorted to find the first
position where the cumulative weight >= tau·total.

Input:  dict with fields Y, D, pi_estimate (1-D arrays)
Output: empirical quantiles of the treated/control groups and the QTE at the given tau
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


def estimate_qte(data, tau):
    """Estimate the plain empirical quantile treatment effect QTE(tau) = q_hat_1(tau) - q_hat_0(tau).

    data: dict containing Y, D, pi_estimate (three 1-D fields)
    tau : quantile level, scalar or 1-D array, in (0, 1)

    Returns dict {tau, q_treated, q_control, n_treated, n_control, qte}.
    When tau is an array, q_treated/q_control/qte are also arrays.
    """
    Y = np.asarray(data["Y"]).ravel()
    D = np.asarray(data["D"]).ravel()
    pi = np.asarray(data["pi_estimate"]).ravel()
    taus = np.atleast_1d(np.asarray(tau, dtype=float))

    # truncate to avoid division by zero
    eps = 1e-6
    pi_c = np.clip(pi, eps, 1.0 - eps)

    # treated group (D=1)
    mask_t = (D == 1)
    w_t = 1.0 / pi_c[mask_t]
    q_t = np.array([weighted_quantile(Y[mask_t], w_t, t) for t in taus])

    # control group (D=0)
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

    # quantile levels are taken uniformly from the config, keeping only the extreme levels
    # tau_n_* (alpha_n/beta_n are auxiliary levels and are not estimated here)
    # tau_n in the config is an upper-tail probability, corresponding to quantile level 1-tau_n
    levels = [(name, tau) for name, tau in tau_levels(cfg, first_n)
              if name.startswith("tau_n")]
    quantile_levels_used = [1.0 - tau for _, tau in levels]

    print("=" * 84)
    print(f"[test] model={first_model}, n={first_n}, sieve basis h_n={h_n}")
    print(f"  treated group n1={int((data['D'] == 1).sum())}, "
          f"control group n0={int((data['D'] == 0).sum())}")

    print("\n  [IPW-weighted empirical quantile] (levels from the config, showing the quantile level 1-tau)")
    print(f"  {'name':<12}{'tau (upper tail)':<16}{'1-tau':<16}"
          f"{'q_treated':>16}{'q_control':>16}{'QTE':>16}")
    print(f"  {'-' * 92}")
    for (name, t), q_level in zip(levels, quantile_levels_used):
        res = estimate_qte(data, q_level)
        print(f"  {name:<12}{t:<16.3e}{q_level:<16.6f}"
              f"{res['q_treated']:16.4f}{res['q_control']:16.4f}{res['qte']:16.4f}")

    # array input check
    res_arr = estimate_qte(data, quantile_levels_used)
    print(f"\n  array input check: quantile level shape={np.shape(res_arr['tau'])}, "
          f"QTE array={np.array2string(np.asarray(res_arr['qte']), precision=4)}")
