# -*- coding: utf-8 -*-
"""Adaptively estimate the auxiliary quantile level k0 (k0 = k^m).

For each (model, n), a single k0 is estimated **uniformly** (shared by the treated and
control groups, no longer group-specific):

Procedure:
  1. Initial k0* = 2·k^{2/3} (k from the config alpha_n, k = n^{0.65});
  2. Use β_n = k0*/n as the auxiliary level to estimate the Causal Fraga EVI γ̂_j^F
     (treated/control groups);
  3. m*_j = 2·γ̂_j^F / (1 + 2·γ̂_j^F) (j=1 treated, j=0 control);
  4. Uniform m = min(m*_1, m*_0) - σ (σ is a hyperparameter, config design.k0_sigma),
     truncated to the lower bound k0_m_lower if it falls below it;
  5. Uniform k0 = k^{m}, β_n = k0 / n (both groups share the same k0 and β_n).

estimate_k0_by_group() can be used as a library by the experiment scripts; the fallback
β_n is provided by fallback_beta(cfg, n) (the initial β* = k0*/n), so beta_n no longer
needs to be specified manually in the config. Running standalone, __main__ prints m*,
m, k0 and β_n for each (model, n).

Config fields:
  design.k0_init_formula : formula for the initial k0* (default "2 * (n ** 0.65) ** (2 / 3)")
  design.k0_sigma        : the σ in m = m* - σ (default 0.05)
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
    """Initial k0* (default 2·k^{2/3})."""
    return eval(cfg["design"]["k0_init_formula"], {"n": n, "log": np.log})


def fallback_beta(cfg, n):
    """Fallback β_n when adaptive estimation fails (γ̂ is nan): take the initial β* = k0*/n."""
    return float(k0_init_star(cfg, n)) / n


def estimate_k0_by_group(cfg, data, n, k=None):
    """Uniformly estimate the shared m, k0 and β_n for the treated/control groups.

    data: the generated data dict (must already contain the pi_estimate field)
    n   : sample size
    k   : optional, the top-k observation count of the intermediate level (used for the
          k-sensitivity analysis; by default taken from the config k = n·α_n = n^{0.65}).
          When k is passed, α_n = k/n and k0* = k^{2/3}, and the adaptive k0 = k^m varies
          with this k.

    Algorithm (uniform k0, no longer group-specific):
      1. Estimate the Causal Fraga EVI γ̂_j^F (j=1/0) for both groups with the initial
         β* = k0*/n;
      2. For each group, m*_j = 2·γ̂_j^F / (1 + 2·γ̂_j^F);
      3. Uniform m = min(m*_1, m*_0) - σ (σ is config design.k0_sigma), truncated to the
         lower bound k0_m_lower if it falls below it;
      4. Uniform k0 = k^{m}, β_n = k0 / n (both groups share the same k0 and β_n).

    If the γ̂ estimate fails in either group, m/k0 become nan and β_n falls back to
    fallback_beta(cfg, n).

    Returns dict {alpha_n, k0_star, sigma, gamma_treated, gamma_control,
              m_star_treated, m_star_control, m, k0, beta,
              beta_treated, beta_control} (beta_treated == beta_control == beta).
    """
    if k is None:
        alpha_n = dict(tau_levels(cfg, n))["alpha_n"]
        k = n * alpha_n                  # k = n^{0.65}
        k0_star = float(k0_init_star(cfg, n))
    else:
        k = float(k)
        alpha_n = k / n
        k0_star = float(k) ** (2.0 / 3.0)   # varies with k: k0* = k^{2/3}
    sigma = float(cfg["design"]["k0_sigma"])
    m_lower = float(cfg["design"].get("k0_m_lower", 0.05))   # lower bound for m
    beta_star = k0_star / n
    fb = fallback_beta(cfg, n)           # fallback β_n

    # estimate the Causal Fraga EVI for both groups with the initial k0*
    res = estimate_evi_causal_fraga(data, beta_star, alpha_n)
    g1 = float(res["gamma_treated"])
    g0 = float(res["gamma_control"])

    def _m_star(gamma, tag):
        """m* = 2γ/(1+2γ) (without subtracting σ). Returns nan and warns if γ̂ is not finite."""
        if not np.isfinite(gamma):
            warnings.warn(
                f"k0 adaptive estimation failed (gamma_{tag} not finite, n={n}), "
                f"β_n falls back to the initial β* = {fb:.4e}",
                RuntimeWarning,
            )
            return np.nan
        return 2.0 * gamma / (1.0 + 2.0 * gamma)

    m_star_1 = _m_star(g1, "treated")
    m_star_0 = _m_star(g0, "control")

    # uniform m = min(m*_1, m*_0) - σ; a failure in either group fails the whole estimate
    if not (np.isfinite(m_star_1) and np.isfinite(m_star_0)):
        m = np.nan
        k0 = np.nan
    else:
        m = min(m_star_1, m_star_0) - sigma
        if m < m_lower:
            m = m_lower                      # lower-bound fallback
        k0 = k ** m
    beta = float(k0) / n if np.isfinite(k0) else fb

    return {
        "alpha_n": float(alpha_n),
        "k0_star": k0_star,
        "sigma": sigma,
        "gamma_treated": g1,
        "gamma_control": g0,
        "m_star_treated": m_star_1,
        "m_star_control": m_star_0,
        "m": m,
        "k0": k0,
        "beta": beta,
        "beta_treated": beta,
        "beta_control": beta,
    }


if __name__ == "__main__":
    cfg = load_config()
    seed = cfg["experiment"]["random_seed"]
    sigma = float(cfg["design"]["k0_sigma"])

    print("=" * 88)
    print("k0 adaptive estimation (uniform k0 across treated/control groups): k0 = k^m, m = min(m*_1, m*_0) - σ")
    print(f"  hyperparameter σ = {sigma}")
    print(f"  initial k0* = {cfg['design']['k0_init_formula']}")
    print("=" * 88)

    results = {}
    for model in cfg["outcome_models"]:
        for n in cfg["design"]["sample_sizes"]:
            data = generate_dataset(cfg, model, n, seed)
            data, _h_n, _info = estimate_propensity_sieve(data)
            r = estimate_k0_by_group(cfg, data, n)
            results[(model, n)] = r
            print(f"\n[{model}] n={n}")
            print(f"  k0* = {r['k0_star']:.3f}  (β_n* = {r['k0_star'] / n:.4e}, α_n = {r['alpha_n']:.4e})")
            print(f"  treated: gamma_1 = {r['gamma_treated']:.4f} -> m*_1 = {r['m_star_treated']:.4f}")
            print(f"  control: gamma_0 = {r['gamma_control']:.4f} -> m*_0 = {r['m_star_control']:.4f}")
            print(f"  uniform: m = min(m*_1, m*_0) - σ = {r['m']:.4f} -> k0 = {r['k0']:.3f} "
                  f"-> β_n = {r['beta']:.4e} (shared by both groups)")

    print("\n" + "=" * 88)
    print(f"  fallback β_n (on adaptive failure) = fallback_beta(cfg, n) = initial β* = k0*/n")
    print("=" * 88)
