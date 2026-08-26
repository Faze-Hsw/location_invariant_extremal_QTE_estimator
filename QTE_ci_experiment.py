# -*- coding: utf-8 -*-
"""Confidence interval coverage Monte Carlo experiment for the QTE estimators (fixed location shift u = 0).

With u fixed at 0, repeat R times:
  1. Generate the original sample and estimate the propensity score by sieves, then obtain the
     QTE point estimates q̂ of the methods from the original sample;
  2. Bootstrap-resample B times (each time re-estimating the propensity score and the Fraga group
     k0, both of which depend only on X/D and the μ=0 data and are reused across all scenarios),
     yielding q̂*_b with standard error se = std(q̂*_b);
  3. Normal-approximation confidence interval CI = q̂ ± z_{1-α/2}·se, with the nominal confidence
     level 1-α read from the config inference.confidence_level (default 0.9);
  4. Coverage = the Monte Carlo mean of 1{true QTE ∈ CI} (ignoring repetitions whose point
     estimate or se is not finite).

The uncertainty of the coverage estimate itself is reported with a Wilson interval (default 95%),
and the plot error bars are that interval.

Methods (consistent with QTE_experiment.py):
  - Deuber            : Hill EVI + Weissman extrapolation (anchor alpha_n)
  - Deuber_diff       : Hill EVI + difference extrapolation (two anchors alpha_n / beta_n)
  - Fraga_alpha       : Fraga EVI + Weissman power extrapolation (anchor alpha_n)
  - Fraga_diff        : Fraga EVI + difference extrapolation (two anchors alpha_n / beta_n)
  - Fraga_diff_asymp  : Fraga EVI + difference extrapolation + analytic standard error
                        (equations 18-22 in the paper, CI equation 22)

The true QTE = q_{Y1}(1-τ) - q_{Y0}(1-τ) is given by QTE_real.py; it is translation-invariant
and does not depend on u.

Plotting: one figure per model, rows = sample size n, columns = tau_n levels, subplot x-axis =
the methods, y-axis = coverage (fixed [0,1]), one point estimate + Wilson error bar per method,
plus the nominal confidence level reference line.

CLI args (all optional, defaults from the config):
  --replications R         number of Monte Carlo repetitions
  --sample-sizes n1 n2..   list of sample sizes
  --bootstrap B            number of bootstrap resamples
  --workers W              number of parallel processes

The location shift is fixed at the constant u = 0 (no --shifts argument is provided).

Run:
  D:\\Miniconda\\python.exe QTE_ci_experiment.py
  D:\\Miniconda\\python.exe QTE_ci_experiment.py --replications 200 --bootstrap 200
"""
import os
import sys
from pathlib import Path

# Limit the BLAS/OpenMP threads inside each worker process to 2 (must be set before importing numpy)
os.environ.update({
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "VECLIB_MAXIMUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
})

import numpy as np

# import repo-local scripts
_BASE = Path(__file__).resolve().parent
for _sub in ("data", "estimate", "EVI", "QTE"):
    sys.path.insert(0, str(_BASE / _sub))

from data_generation import load_config, generate_dataset, tau_levels  # noqa: E402
from estimate_propensity_sieve import estimate_propensity_sieve  # noqa: E402
from causal_fraga import estimate_evi_causal_fraga  # noqa: E402
from Deuber import estimate_qte_extrapolation  # noqa: E402
from Deuber_diff import estimate_qte_diff  # noqa: E402
from QTE_Fraga import estimate_qte_extrapolation_fraga  # noqa: E402
from QTE_Fraga_diff import estimate_qte_diff_fraga  # noqa: E402
from QTE_Fraga_diff import difference_extrapolate  # noqa: E402
from QTE_real import real_qte  # noqa: E402
from estimate_k0 import estimate_k0_by_group, fallback_beta  # noqa: E402
from causal_fraga import _weighted_quantile as fraga_wquant  # noqa: E402

METHODS = ["Deuber", "Deuber_diff", "Fraga_alpha", "Fraga_diff", "Fraga_diff_asymp"]
COLORS = {
    "Deuber": "#ed7d31",
    "Deuber_diff": "#5b9bd5",
    "Fraga_alpha": "#70ad47",
    "Fraga_diff": "#7030a0",
    "Fraga_diff_asymp": "#c00000",
}
LABELS = {
    "Deuber": "Causal Hill (mult extrapolation)",
    "Deuber_diff": "Causal Hill (diff extrapolation)",
    "Fraga_alpha": "Causal Fraga (mult extrapolation)",
    "Fraga_diff": "Causal Fraga (diff extrapolation)",
    "Fraga_diff_asymp": "Causal Fraga (asymp CI)",
}


def estimate_qte_by_method(data, tau, alpha_n, beta_fb, beta_deuber_diff, k0res):
    """Estimate the QTE of the four methods at target level tau on a single sample (scalar tau).

    data : dict containing Y, D, pi_estimate
    k0res: the result of estimate_k0_by_group (Fraga group-adaptive β_t/β_c)
    Returns {method: qte} (may be nan when the estimate fails).
    """
    beta_t = k0res["beta_treated"]
    beta_c = k0res["beta_control"]
    return {
        "Deuber": estimate_qte_extrapolation(data, alpha_n, tau)["qte_ext"],
        "Deuber_diff": estimate_qte_diff(data, alpha_n, beta_deuber_diff, tau)["qte_ext"],
        "Fraga_alpha": estimate_qte_extrapolation_fraga(
            data, beta_fb, alpha_n, tau,
            beta_treated=beta_t, beta_control=beta_c)["qte_ext"],
        "Fraga_diff": estimate_qte_diff_fraga(
            data, alpha_n, beta_fb, tau,
            beta_treated=beta_t, beta_control=beta_c)["qte_ext"],
    }


def fraga_diff_asymp_ci(data, alpha_n, beta_fb, tau_target, k0res, n):
    """Fraga difference extrapolation + analytic CI (based on equations (18)-(22) in the paper).

    For the extreme-quantile QTE, use Fraga EVI + difference extrapolation to get the point
    estimate Δ̂(1-τ); the analytic standard error follows equation (22): se = σ̂ / φ̂_n, where

        σ̂² = min{1,κ̂}² (γ̂₁^F)² â_{n,1}² + min{1,1/κ̂}² (γ̂₀^F)² â_{n,0}²   (equation 21)
        φ̂_n = √k0 / [ log(β_n/τ) · max{Q̂₁(1-τ), Q̂₀(1-τ)} ]                (equation 18)
        κ̂   = Q̂₁(1-τ) / Q̂₀(1-τ)                                             (equation 19)
        â²_{n,j} = (1/k0) Σ_i R̂²_{n,j,i}                                    (equation 20)
        R̂_{n,j,i} = (D_i/π̂)^j ((1-D_i)/(1-π̂))^{1-j} 1{Y_i>q̂_j(1-β_n)}
                     · (1/γ̂_j^F) · log[(Y_i - q̂_j(1-α_n)) / (q̂_j(1-β_n) - q̂_j(1-α_n))]
                     - β_n                                                   (R in equation 20)

    Analytic CI: Δ̂(1-τ) ± z_{1-α/2} · σ̂ / φ̂_n                                  (equation 22)

    Returns dict {qte, q_treated, q_control, sigma, phi, se, ci};
    if any quantity becomes non-finite, returns nan (so the caller marks this repetition with nan).
    """
    Y = np.asarray(data["Y"]).ravel()
    D = np.asarray(data["D"]).ravel()
    pi = np.asarray(data["pi_estimate"]).ravel()
    if not (0.0 < tau_target < alpha_n) or tau_target <= 0:
        return {"qte": np.nan, "se": np.nan, "sigma": np.nan, "phi": np.nan,
                "q_treated": np.nan, "q_control": np.nan, "ci": (np.nan, np.nan)}
    beta_t = float(k0res["beta_treated"])
    beta_c = float(k0res["beta_control"])
    if not (0.0 < beta_t < alpha_n) or not (0.0 < beta_c < alpha_n):
        return {"qte": np.nan, "se": np.nan, "sigma": np.nan, "phi": np.nan,
                "q_treated": np.nan, "q_control": np.nan, "ci": (np.nan, np.nan)}

    eps = 1e-6
    pi_c = np.clip(pi, eps, 1.0 - eps)

    mask_t = (D == 1)
    mask_c = (D == 0)
    w_t = 1.0 / pi_c[mask_t]
    w_c = 1.0 / (1.0 - pi_c[mask_c])

    # anchor quantiles (consistent with Fraga_diff; the β anchor uses the group-specific β_j)
    q_alpha_t = fraga_wquant(Y[mask_t], w_t, 1.0 - alpha_n)
    q_alpha_c = fraga_wquant(Y[mask_c], w_c, 1.0 - alpha_n)
    q_beta_t = fraga_wquant(Y[mask_t], w_t, 1.0 - beta_t)
    q_beta_c = fraga_wquant(Y[mask_c], w_c, 1.0 - beta_c)

    # Fraga EVI: by default as in fraga_diff (the β anchor uses the respective β_n of β_t / β_c)
    fraga = estimate_evi_causal_fraga(data, beta_fb, alpha_n, beta_t, beta_c)
    gamma_t = fraga["gamma_treated"]
    gamma_c = fraga["gamma_control"]
    if not (np.isfinite(gamma_t) and np.isfinite(gamma_c)) \
            or abs(gamma_t) < 1e-12 or abs(gamma_c) < 1e-12:
        return {"qte": np.nan, "se": np.nan, "sigma": np.nan, "phi": np.nan,
                "q_treated": np.nan, "q_control": np.nan, "ci": (np.nan, np.nan)}

    # difference-extrapolation point estimate (same implementation as Fraga_diff)
    q_t = difference_extrapolate(q_alpha_t, q_beta_t, alpha_n, beta_t, tau_target, gamma_t)
    q_c = difference_extrapolate(q_alpha_c, q_beta_c, alpha_n, beta_c, tau_target, gamma_c)
    qte = q_t - q_c

    # === analytic standard error (equations 18-22) ===
    # k0 uses n · β_n of the treated group (consistent with the group k0 formula; equation (20)
    # uses the treated-group β_n as the constant in the derivation; here n·beta_t is taken, with
    # beta_t coming from the same adaptive estimate as in the Fraga family).
    k0 = float(n) * beta_t
    if not (k0 > 0):
        return {"qte": np.nan, "se": np.nan, "sigma": np.nan, "phi": np.nan,
                "q_treated": q_t, "q_control": q_c, "ci": (np.nan, np.nan)}

    # equation (20): compute â²_{n,1} and â²_{n,0} for j separately.
    # treated group (j=1): weight w_t = D/π̂
    denom1 = q_beta_t - q_alpha_t
    denom0 = q_beta_c - q_alpha_c
    if not (denom1 > 0 and denom0 > 0):
        return {"qte": qte, "se": np.nan, "sigma": np.nan, "phi": np.nan,
                "q_treated": q_t, "q_control": q_c, "ci": (np.nan, np.nan)}

    # j=1: treated group (using β_t)
    ind1 = (Y > q_beta_t) & (D == 1)
    if ind1.any():
        log_term1 = np.log((Y[ind1] - q_alpha_t) / denom1)
        R1 = (D[ind1] / pi_c[ind1]) * (1.0 / gamma_t) * log_term1 - beta_t
        a_sq_t = float(np.sum(R1 ** 2)) / k0
    else:
        a_sq_t = np.nan
    # j=0: control group (using β_c)
    ind0 = (Y > q_beta_c) & (D == 0)
    if ind0.any():
        log_term0 = np.log((Y[ind0] - q_alpha_c) / denom0)
        R0 = ((1.0 - D[ind0]) / (1.0 - pi_c[ind0])) * (1.0 / gamma_c) * log_term0 - beta_c
        a_sq_c = float(np.sum(R0 ** 2)) / k0
    else:
        a_sq_c = np.nan

    if not (np.isfinite(a_sq_t) and np.isfinite(a_sq_c)):
        return {"qte": qte, "se": np.nan, "sigma": np.nan, "phi": np.nan,
                "q_treated": q_t, "q_control": q_c, "ci": (np.nan, np.nan)}

    # κ̂ = Q̂₁(1-τ) / Q̂₀(1-τ)  (equation 19)
    if not (np.isfinite(q_t) and np.isfinite(q_c) and q_c > 0):
        return {"qte": qte, "se": np.nan, "sigma": np.nan, "phi": np.nan,
                "q_treated": q_t, "q_control": q_c, "ci": (np.nan, np.nan)}
    kappa = q_t / q_c
    kappa = float(np.clip(kappa, 1e-12, None))   # avoid underflow of 1/κ

    # σ̂² = min{1,κ̂}² (γ̂₁^F)² â²_{n,1} + min{1,1/κ̂}² (γ̂₀^F)² â²_{n,0}  (equation 21)
    m1 = min(1.0, kappa)
    m0 = min(1.0, 1.0 / kappa)
    sigma2 = (m1 * gamma_t) ** 2 * a_sq_t + (m0 * gamma_c) ** 2 * a_sq_c
    if not (sigma2 > 0):
        return {"qte": qte, "se": np.nan, "sigma": np.nan, "phi": np.nan,
                "q_treated": q_t, "q_control": q_c, "ci": (np.nan, np.nan)}
    sigma = float(np.sqrt(sigma2))

    # φ̂_n = √k0 / [ log(β_n/τ) · max{Q̂₁(1-τ), Q̂₀(1-τ)} ]           (equation 18)
    # Note: the β_n and k0 in equation (18) come from the same treated-group auxiliary level
    # (k0 = n·β_n), so min{β_t, β_c} is taken as the "deepest auxiliary level" to keep
    # log(β_n/τ) > 0.
    beta_phi = float(min(beta_t, beta_c))
    if not (beta_phi > tau_target > 0) or beta_phi <= 0:
        return {"qte": qte, "se": np.nan, "sigma": np.nan, "phi": np.nan,
                "q_treated": q_t, "q_control": q_c, "ci": (np.nan, np.nan)}
    log_term_phi = float(np.log(beta_phi / tau_target))
    if abs(log_term_phi) < 1e-12 or not (q_t > 0 and q_c > 0):
        return {"qte": qte, "se": np.nan, "sigma": np.nan, "phi": np.nan,
                "q_treated": q_t, "q_control": q_c, "ci": (np.nan, np.nan)}
    phi = float(np.sqrt(k0) / (log_term_phi * max(q_t, q_c)))
    if not (phi > 0):
        return {"qte": qte, "se": np.nan, "sigma": np.nan, "phi": np.nan,
                "q_treated": q_t, "q_control": q_c, "ci": (np.nan, np.nan)}

    se = sigma / phi
    return {"qte": float(qte), "se": float(se), "sigma": float(sigma),
            "phi": float(phi), "q_treated": float(q_t), "q_control": float(q_c),
            "ci": (float(qte - 1.96 * se), float(qte + 1.96 * se))}


def bootstrap_prep(data0, n, B, rng, cfg):
    """Do B resamples with replacement of the μ=0 original sample, precomputing each bootstrap unit.

    Returns (boot_idx, boot_D, boot_pi, boot_k0):
      boot_idx : list of (n,) resample indices
      boot_D   : list of (n,) resampled treatment indicators
      boot_pi  : list of (n,) propensity scores re-estimated by sieves on the resample
      boot_k0  : list of estimate_k0_by_group results (adaptive k0 on the bootstrap sample)
    """
    import warnings

    X, D, Y = data0["X"], data0["D"], data0["Y"]
    boot_idx, boot_D, boot_pi, boot_k0 = [], [], [], []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        bs = {"X": np.asarray(X)[idx], "D": np.asarray(D)[idx]}
        bs, _h, _info = estimate_propensity_sieve(bs)
        bs["Y"] = np.asarray(Y)[idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                k0 = estimate_k0_by_group(cfg, bs, n)
            except Exception:
                k0 = None  # failure: treated with the fallback β_n later
        boot_idx.append(idx)
        boot_D.append(bs["D"])
        boot_pi.append(bs["pi_estimate"])
        boot_k0.append(k0)
    return boot_idx, boot_D, boot_pi, boot_k0


def run_experiment(cfg, model, n, replications, bootstrap_B,
                   conf_level, base_seed):
    """Run R repetitions and B bootstrap resamples for a single (model, sample size) to compute coverage.

    The location shift is fixed at u = 0.

    Returns (coverage_by, truth_by, tau_vals).
    coverage_by: {tau_name: {method: (coverage, Wilson lower bound, upper bound)}}
    truth_by   : {tau_name: true QTE}
    tau_vals   : {tau_name: upper-tail probability tau}
    """
    import warnings
    from scipy import stats

    levels = dict(tau_levels(cfg, n))
    tau_names = [name for name in levels if name.startswith("tau_n")]
    alpha_n = levels["alpha_n"]
    beta_fb = fallback_beta(cfg, n)
    k0_dd = eval(cfg["design"]["k0_deuber_diff_formula"], {"n": n, "log": np.log})
    beta_deuber_diff = float(k0_dd) / n
    tau_vals = {name: levels[name] for name in tau_names}
    truth_by = {name: real_qte(cfg, model, 1.0 - levels[name])["qte"]
                for name in tau_names}

    z = stats.norm.ppf(1.0 - (1.0 - conf_level) / 2.0)
    # record the coverage indicator over R repetitions for each method (non-finite estimates are nan)
    cov_hits = {name: {m: [] for m in METHODS} for name in tau_names}

    for rep in range(replications):
        seed = base_seed + rep
        data0 = generate_dataset(cfg, model, n, seed)   # μ defaults to 0 (u is fixed at 0)
        data0, _h_n, _info = estimate_propensity_sieve(data0)
        rng = np.random.default_rng(seed + 987654321)   # bootstrap sub-seed
        # group k0 of the original sample (Fraga; μ=0 data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                k0_orig = estimate_k0_by_group(cfg, data0, n)
            except Exception:
                k0_orig = None
        # precompute the bootstrap units (μ=0 sample resampling + re-estimating the propensity
        # score + group k0)
        boot_idx, boot_D, boot_pi, boot_k0 = bootstrap_prep(
            data0, n, bootstrap_B, rng, cfg)
        Y = data0["Y"]

        # point estimates on the original sample (four methods for each tau level)
        point = {name: estimate_qte_by_method(
            data0, levels[name], alpha_n, beta_fb, beta_deuber_diff, k0_orig)
            for name in tau_names}

        # bootstrap distribution: the same set of resample indices / propensity score / k0 is
        # applied to the current sample's Y (the bootstrap estimates are recorded separately per
        # tau level, and se is computed independently per tau; Fraga_diff_asymp does not depend
        # on bootstrap and is skipped)
        boot_methods = [m for m in METHODS if m != "Fraga_diff_asymp"]
        boot_ests = {name: {m: [] for m in boot_methods} for name in tau_names}
        for idx, D_b, pi_b, k0_b in zip(boot_idx, boot_D, boot_pi, boot_k0):
            data_b = {"Y": np.asarray(Y)[idx], "D": D_b, "pi_estimate": pi_b}
            if k0_b is None:
                k0_b = {"beta_treated": beta_fb, "beta_control": beta_fb}
            for name in tau_names:
                b_est = estimate_qte_by_method(
                    data_b, levels[name], alpha_n, beta_fb,
                    beta_deuber_diff, k0_b)
                for m in boot_methods:
                    boot_ests[name][m].append(b_est[m])

        for name in tau_names:
            # the point estimate is an independent scalar for each tau level
            est_m = point[name]
            for m in METHODS:
                if m == "Fraga_diff_asymp":
                    # analytic CI: does not depend on bootstrap; constructed directly from σ̂/φ̂
                    # by equation (22)
                    k0_asym = k0_orig if k0_orig is not None else {
                        "beta_treated": beta_fb, "beta_control": beta_fb}
                    asymp = fraga_diff_asymp_ci(
                        data0, alpha_n, beta_fb, levels[name], k0_asym, n)
                    q_asym, se_asym = asymp["qte"], asymp["se"]
                    if not (np.isfinite(q_asym) and np.isfinite(se_asym)
                            and se_asym > 0):
                        cov_hits[name][m].append(np.nan)
                        continue
                    lo, hi = q_asym - z * se_asym, q_asym + z * se_asym
                    cov_hits[name][m].append(
                        1.0 if (lo <= truth_by[name] <= hi) else 0.0)
                    continue
                q = float(est_m[m])
                arr = np.asarray(boot_ests[name][m], dtype=float)
                arr = arr[np.isfinite(arr)]
                if not np.isfinite(q) or arr.size < 2:
                    cov_hits[name][m].append(np.nan)
                    continue
                se = float(np.std(arr, ddof=1))
                if not (np.isfinite(se) and se > 0):
                    cov_hits[name][m].append(np.nan)
                    continue
                lo, hi = q - z * se, q + z * se
                cov_hits[name][m].append(
                    1.0 if (lo <= truth_by[name] <= hi) else 0.0)

    # coverage point estimate + the Wilson confidence interval for the coverage estimate itself
    # (estimation uncertainty); binomtest's wilson method is the Wilson score interval
    wilson_conf = 0.95
    coverage_by = {}
    for name in tau_names:
        coverage_by[name] = {}
        for m in METHODS:
            arr = np.asarray(cov_hits[name][m], dtype=float)
            arr = arr[np.isfinite(arr)]
            r = arr.size
            if r == 0:
                coverage_by[name][m] = (np.nan, np.nan, np.nan)
                continue
            x = int(arr.sum())
            ci = stats.binomtest(x, r).proportion_ci(
                confidence_level=wilson_conf, method="wilson")
            coverage_by[name][m] = (x / r, float(ci.low), float(ci.high))
    return coverage_by, truth_by, tau_vals


def plot_coverage(coverage_by, tau_names, tau_formulas,
                  model, sample_sizes, conf_level, out_dir):
    """One figure per model (coverage): the location shift is fixed at u = 0, rows = sample size n,
    columns = tau_n levels, the subplot x-axis does not label the method names, the y-axis =
    coverage ([0,1]), one point estimate + Wilson CI error bar per method (the method names are
    explained by the legend), plus the nominal confidence level reference line. The row label n
    is placed to the right of the y-axis of the rightmost column."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    n_tau = len(tau_names)
    n_n = len(sample_sizes)
    x = np.arange(len(METHODS))
    fig, axes = plt.subplots(n_n, n_tau, figsize=(4.2 * n_tau, 3.4 * n_n),
                             squeeze=False)
    for r, n in enumerate(sample_sizes):
        cov_n = coverage_by[n]          # {tau_name: {method: (cov, lo, hi)}}
        for c, name in enumerate(tau_names):
            ax = axes[r][c]
            for i, m in enumerate(METHODS):
                cov, lo, hi = cov_n[name][m]
                if not np.isfinite(cov):
                    continue
                ax.errorbar(i, cov, yerr=[[cov - lo], [hi - cov]],
                            fmt="o", color=COLORS[m], markersize=5,
                            linewidth=1.2, capsize=3)
            ax.axhline(conf_level, color="black", linestyle="--", linewidth=1.2,
                       alpha=0.7)
            ax.set_ylim(0.0, 1.0)
            # the x-axis does not identify the methods (the method names are explained by the legend)
            ax.set_xticks(x)
            ax.set_xticklabels([])
            if r == 0:
                ax.set_title(rf"$\tau_n = {tau_formulas[name]}$")
            ax.grid(alpha=0.3, which="major", axis="y")
            if c == n_tau - 1:
                ax.set_ylabel(f"n = {n}", fontsize=11)
                ax.yaxis.set_label_position("right")
    fig.supylabel("coverage", fontsize=13)
    # the legend shows the methods and the reference line / error bar descriptions
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=COLORS[m],
                          markersize=5, label=LABELS[m]) for m in METHODS]
    handles.append(plt.Line2D([0], [0], color="black", linestyle="--",
                              label=f"nominal level {conf_level:.0%}"))
    handles.append(plt.Line2D([0], [0], color="gray", marker="_", ls="",
                              label="error bar: 95% Wilson CI"))
    # the legend is at the very top of the figure; the model label is placed below the legend
    # and above the τ_n titles
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=5, fontsize=8, frameon=False, alignment="center")
    fig.text(0.5, 0.87, model, ha="center", va="center", fontsize=13)
    fig.tight_layout(rect=(0.03, 0.05, 0.97, 0.84))
    out_path = out_dir / f"QTE_ci_cov_{model}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Coverage plot saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="QTE CI-coverage Monte Carlo experiment")
    parser.add_argument("--replications", type=int, default=None)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    seed_base = cfg["experiment"]["random_seed"]
    sample_sizes = args.sample_sizes if args.sample_sizes else list(cfg["design"]["sample_sizes"])
    replications = args.replications if args.replications else int(
        cfg["design"].get("replications", 1000))
    workers = args.workers if args.workers is not None else int(cfg["design"].get("num_workers", 0))
    conf_level = float(cfg["inference"]["confidence_level"])
    bootstrap_B = args.bootstrap if args.bootstrap else int(
        cfg["inference"].get("bootstrap_replications", 200))
    models = list(cfg["outcome_models"])

    tau_names = [name for name, _ in tau_levels(cfg, sample_sizes[0])
                 if name.startswith("tau_n")]
    tau_formulas = {q["name"]: q["formula"] for q in cfg["design"]["quantile_levels"]
                    if q["name"].startswith("tau_n")}

    from scipy import stats
    z = stats.norm.ppf(1.0 - (1.0 - conf_level) / 2.0)

    print("=" * 72)
    print("QTE confidence interval coverage Monte Carlo experiment")
    print("  (point estimate from the original sample; standard error from bootstrap, normal-approximation CI)")
    print(f"  models: {models}")
    print(f"  sample sizes: {sample_sizes}")
    print("  location shift u = 0 (fixed)")
    print(f"  Monte Carlo repetitions R = {replications}, bootstrap resamples B = {bootstrap_B}")
    print(f"  nominal confidence level = {conf_level} (normal critical value z = {z:.4f})")
    print("=" * 72)

    tasks = [(model, n) for model in models for n in sample_sizes]
    if workers > 0:
        max_workers = min(len(tasks), workers)
    else:
        max_workers = min(len(tasks), os.cpu_count() or 1)
    print(f"  tasks: {len(tasks)}, parallel processes: {max_workers}")

    all_cov = {model: {n: None for n in sample_sizes} for model in models}
    all_truth = {model: {n: None for n in sample_sizes} for model in models}
    from tqdm import tqdm

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(run_experiment, cfg, model, n, replications,
                        bootstrap_B, conf_level, seed_base): (model, n)
            for model, n in tasks
        }
        with tqdm(total=len(futures), desc="Running CI experiments", unit="task",
                  ncols=100, dynamic_ncols=True, mininterval=0.0, miniters=1) as pbar:
            for fut in as_completed(futures):
                model, n = futures[fut]
                try:
                    all_cov[model][n], all_truth[model][n], _ = fut.result()
                except Exception as e:
                    print(f"[model {model}, n={n}] failed: {e}")
                    raise
                pbar.set_postfix_str(f"{model} n={n}")
                pbar.update(1)

    out_dir = Path(__file__).resolve().parent / "results"
    for model in models:
        cov_by_n = {n: all_cov[model][n] for n in sample_sizes}
        plot_coverage(cov_by_n, tau_names, tau_formulas,
                      model, sample_sizes, conf_level, out_dir)

    print("\nExperiment finished.")
