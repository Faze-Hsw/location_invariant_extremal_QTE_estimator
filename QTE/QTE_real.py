# -*- coding: utf-8 -*-
"""True (theoretical) quantile treatment effect QTE.

Reuses estimate/estimate_quantile_real.py: numerically integrate the marginal distribution
F_{Y_j}(q) = ∫_0^1 F_{Y_j|X=x}(q) dx of the potential outcome Y_j, then solve F(q) = τ with
brentq to obtain the true quantile q_{Y_j}(τ).

    true QTE(τ) = q_{Y1}(τ) - q_{Y0}(τ)

tau_n in the config is an upper-tail probability, corresponding to quantile level 1-tau_n,
comparable at the same level with Zhang.py (empirical) / Deuber.py / QTE_Fraga*.py (extrapolation).

Input:  config + quantile level τ (scalar or array)
Output: true quantiles and true QTE at level τ for each model
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "estimate"))
from estimate_quantile_real import real_quantile  # noqa: E402


def real_qte(cfg, model, tau):
    """True QTE(τ) = q_{Y1}(τ) - q_{Y0}(τ).

    cfg  : config dict
    model: outcome model name (H1/H2/H3)
    tau  : quantile level, scalar or 1-D array, in (0, 1)

    Returns dict {tau, q_treated, q_control, qte}.
    When tau is an array, q_treated/q_control/qte are also arrays.
    """
    q_t = real_quantile(cfg, model, 1, tau)
    q_c = real_quantile(cfg, model, 0, tau)
    q_t_arr = np.atleast_1d(np.asarray(q_t, dtype=float))
    q_c_arr = np.atleast_1d(np.asarray(q_c, dtype=float))
    qte = q_t_arr - q_c_arr
    taus = np.atleast_1d(np.asarray(tau, dtype=float))

    def _scalar_or_array(arr):
        return arr[0] if arr.size == 1 else arr

    return {
        "tau": _scalar_or_array(taus),
        "q_treated": _scalar_or_array(q_t_arr),
        "q_control": _scalar_or_array(q_c_arr),
        "qte": _scalar_or_array(qte),
    }


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
    from data_generation import load_config, tau_levels

    cfg = load_config()
    first_model = list(cfg["outcome_models"])[0]
    first_n = cfg["design"]["sample_sizes"][0]

    # quantile levels are taken uniformly from the config, keeping only the extreme levels
    # tau_n_* (alpha_n/beta_n are auxiliary levels)
    # tau_n in the config is an upper-tail probability, corresponding to quantile level 1-tau_n
    levels = [(name, tau) for name, tau in tau_levels(cfg, first_n)
              if name.startswith("tau_n")]
    quantile_levels_used = [1.0 - tau for _, tau in levels]

    print("=" * 84)
    print(f"true (theoretical) QTE  n={first_n} (levels from the config, showing the quantile level 1-tau)")
    print(f"  {'name':<12}{'tau (upper tail)':<16}{'1-tau':<16}"
          f"{'q_treated':>16}{'q_control':>16}{'QTE':>16}")
    print(f"  {'-' * 92}")
    for model in cfg["outcome_models"]:
        for (name, t), q_level in zip(levels, quantile_levels_used):
            res = real_qte(cfg, model, q_level)
            print(f"  {model:<12}{name:<8}{t:<16.3e}{q_level:<16.6f}"
                  f"{res['q_treated']:16.4f}{res['q_control']:16.4f}{res['qte']:16.4f}")
