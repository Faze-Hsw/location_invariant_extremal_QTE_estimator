# -*- coding: utf-8 -*-
"""Extrapolation (Weissman-type) extreme quantile treatment effect (QTE) estimator (Fraga EVI version).

Anchor: intermediate-level quantile q̂_j(1-α_n) (IPW-weighted empirical quantile, α_n from the
     config, formula α_n = k/n, k = n^{0.65}, corresponding to quantile level 1-α_n).
Extreme value index: Candal–Fraga estimator γ̂_j^F (estimate_evi_causal_fraga in
     EVI/causal_fraga.py, which internally needs an auxiliary level β_n to construct the
     threshold difference).

For more extreme tail levels τ < α_n (corresponding to quantile level 1-τ), use Weissman-type
extrapolation:
    q̂_j^ext(1-τ) = q̂_j(1-α_n) · (α_n / τ)^{γ̂_j^F}
    QTE^ext(1-τ)  = q̂_1^ext(1-τ) - q̂_0^ext(1-τ)

Principle: if the tail distribution is approximately Pareto, the log-exceedances above a
large threshold u follow an exponential distribution whose scale is characterized by the EVI
γ; q̂(1-α_n) is pushed outward by the level according to the power law (α_n/τ)^γ.

Input:  dict with fields Y, D, pi_estimate (1-D arrays)
Output: extrapolated treated/control-group quantiles and the QTE at the target extreme level
        τ (quantile level 1-τ)
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "EVI"))
from causal_fraga import estimate_evi_causal_fraga  # noqa: E402


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


def extrapolate_quantile(q_anchor, anchor_level, tau_target, gamma):
    """Weissman-type extrapolation: q̂(1-τ) ≈ q̂(1-anchor) · (anchor/τ)^γ."""
    return q_anchor * (anchor_level / tau_target) ** gamma


def estimate_qte_extrapolation_fraga(data, beta_n, alpha_n, tau_target, gamma=None,
                                     beta_treated=None, beta_control=None):
    """Estimate the extreme quantile treatment effect by extrapolation (Fraga EVI version).

    data      : dict containing Y, D, pi_estimate (three 1-D fields)
    beta_n    : auxiliary intermediate level for the Fraga estimator (used to build the threshold
                difference; requires beta_n < alpha_n)
    alpha_n   : intermediate anchor level (upper-tail probability); the anchor quantile is q̂_j(1-α_n)
    tau_target: target extreme level (upper-tail probability), scalar or array, must be positive
                (tau > 0); usually used with tau_target < alpha_n (extrapolating to a more extreme
                tail than the anchor)
    gamma     : optional extreme value index dict {gamma_treated, gamma_control};
                by default computed with the Candal–Fraga estimator (estimate_evi_causal_fraga)
    beta_treated  : optional, treated-group β_n (passed in for group-specific k0)
    beta_control  : optional, control-group β_n (passed in for group-specific k0)

    Returns dict {beta_n, alpha_n, tau, q_anchor_treated, q_anchor_control,
              gamma_treated, gamma_control, q_treated_ext, q_control_ext, qte_ext}.
    When tau is an array, q_*_ext and qte_ext are also arrays.
    """
    Y = np.asarray(data["Y"]).ravel()
    D = np.asarray(data["D"]).ravel()
    pi = np.asarray(data["pi_estimate"]).ravel()
    n = Y.size
    taus = np.atleast_1d(np.asarray(tau_target, dtype=float))
    if np.any(taus <= 0):
        raise ValueError("tau_target must be positive (tau > 0)")

    eps = 1e-6
    pi_c = np.clip(pi, eps, 1.0 - eps)

    # intermediate anchor-level quantile q̂_j(1-α_n) (IPW-weighted)
    mask_t = (D == 1)
    mask_c = (D == 0)
    w_t = 1.0 / pi_c[mask_t]
    w_c = 1.0 / (1.0 - pi_c[mask_c])
    q_anchor_t = weighted_quantile(Y[mask_t], w_t, 1.0 - alpha_n)
    q_anchor_c = weighted_quantile(Y[mask_c], w_c, 1.0 - alpha_n)

    # extreme value index: use Candal–Fraga by default (with group-specific β_n)
    if gamma is None:
        fraga = estimate_evi_causal_fraga(data, beta_n, alpha_n,
                                          beta_treated=beta_treated,
                                          beta_control=beta_control)
        gamma_t = fraga["gamma_treated"]
        gamma_c = fraga["gamma_control"]
    else:
        gamma_t = float(gamma["gamma_treated"])
        gamma_c = float(gamma["gamma_control"])

    q_ext_t = np.array([extrapolate_quantile(q_anchor_t, alpha_n, t, gamma_t)
                        for t in taus])
    q_ext_c = np.array([extrapolate_quantile(q_anchor_c, alpha_n, t, gamma_c)
                        for t in taus])
    qte_ext = q_ext_t - q_ext_c

    def _scalar_or_array(arr):
        return arr[0] if arr.size == 1 else arr

    return {
        "beta_n": float(beta_n),
        "alpha_n": float(alpha_n),
        "tau": _scalar_or_array(taus),
        "q_anchor_treated": float(q_anchor_t),
        "q_anchor_control": float(q_anchor_c),
        "gamma_treated": gamma_t,
        "gamma_control": gamma_c,
        "q_treated_ext": _scalar_or_array(q_ext_t),
        "q_control_ext": _scalar_or_array(q_ext_c),
        "qte_ext": _scalar_or_array(qte_ext),
    }


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "estimate"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data_generation import load_config, generate_dataset, tau_levels
    from estimate_propensity_sieve import estimate_propensity_sieve
    from estimate_k0 import fallback_beta

    cfg = load_config()
    seed = cfg["experiment"]["random_seed"]
    first_model = list(cfg["outcome_models"])[0]
    first_n = cfg["design"]["sample_sizes"][0]

    # anchor level α_n and auxiliary level β_n from the config; target extreme level τ
    # from all tau_n_* levels in the config
    levels = dict(tau_levels(cfg, first_n))
    alpha_n = levels["alpha_n"]
    beta_n = fallback_beta(cfg, first_n)   # fallback β_n (auxiliary level)
    target_levels = [(name, levels[name]) for name in levels if name.startswith("tau_n")]
    target_taus = [t for _, t in target_levels]

    data = generate_dataset(cfg, first_model, first_n, seed)
    data, h_n, info = estimate_propensity_sieve(data)

    print("=" * 92)
    print(f"[test] model={first_model}, n={first_n}, h_n={h_n}")
    print(f"  anchor level alpha_n = {alpha_n:.4e}, beta_n = {beta_n:.4e}")

    res = estimate_qte_extrapolation_fraga(data, beta_n, alpha_n, target_taus)
    print(f"  anchor quantile: q1(1-a)={res['q_anchor_treated']:12.3f}, "
          f"q0(1-a)={res['q_anchor_control']:12.3f}")
    print(f"  Fraga EVI:   gamma_1^F={res['gamma_treated']:.4f}, "
          f"gamma_0^F={res['gamma_control']:.4f}")

    print("\n  [Weissman extrapolation QTE (Fraga)] (target levels from the config, showing the quantile level 1-tau)")
    print(f"  {'name':<12}{'tau (upper tail)':<16}{'1-tau':<16}"
          f"{'q_treated':>16}{'q_control':>16}{'QTE':>16}")
    print(f"  {'-' * 92}")
    for i, (name, t) in enumerate(target_levels):
        q_level = 1.0 - t
        print(f"  {name:<12}{t:<16.3e}{q_level:<16.6f}"
              f"{np.asarray(res['q_treated_ext'])[i]:16.4f}"
              f"{np.asarray(res['q_control_ext'])[i]:16.4f}"
              f"{np.asarray(res['qte_ext'])[i]:16.4f}")
