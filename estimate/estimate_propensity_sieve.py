# -*- coding: utf-8 -*-
"""Estimate the propensity score using a nonparametric sieve method.

Model:  logit(pi(x)) = H_{h_n}(x)^T * pi_n
        pi(x)   = 1 / (1 + exp(-H_{h_n}(x)^T * pi_n))

where H_{h_n} is a one-dimensional polynomial basis H(x) = (1, x, x^2, ..., x^{h_n-1}), and the
parameters are estimated by
    pi_n = argmax Σ [D_i log L(H(X_i)^T pi) + (1-D_i) log(1 - L(H(X_i)^T pi))]
(L is the sigmoid). This script uses Newton-Raphson + Armijo backtracking line search to do the
MLE of the **unregularized** negative log-likelihood; the basis matrix is column-standardized to
improve the conditioning.

Input:  dict containing X, D, pi fields (1-D arrays)
Output: adds a pi_estimate field to the table (the estimated propensity score of the same length)
"""
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit


def sieve_basis(X, h_n):
    """One-dimensional polynomial sieve basis H(x) = (1, x, x^2, ..., x^{h_n-1}), returning the (n, h_n) design matrix."""
    return np.vander(X, N=h_n, increasing=True)


def default_sieve_dim(n):
    """Default sieve basis dimension h_n = ⌊2·n^{1/11}⌋ (user-specified formula).

    The exponent 1/11 makes the basis dimension grow very slowly with n; for common sample sizes:
    n=1000 → ⌊2·1000^{1/11}⌋ = ⌊3.73⌋ = 3; n=5000 → ⌊2·5000^{1/11}⌋ = ⌊4.66⌋ = 4.
    """
    return int(np.floor(2.0 * n ** (1.0 / 11.0)))


def _newton_logistic(H, D, max_iter=200, tol=1e-8):
    """Newton-Raphson for the unregularized logistic MLE; with Armijo backtracking line search.

    Objective: negative log-likelihood NLL(theta) = -Σ [D_i eta_i - log(1+e^{eta_i})], eta = H theta
    Gradient:  ∇NLL = H^T (sigma - D)
    Hessian:   ∇^2 NLL = H^T diag(sigma(1-sigma)) H
    """
    n, p = H.shape
    p0 = np.clip(D.mean(), 1e-3, 1 - 1e-3)
    theta = np.zeros(p)
    theta[0] = np.log(p0 / (1 - p0))  # initial: constant term logit(D mean), the rest 0

    for it in range(max_iter):
        eta = H @ theta
        sigma = expit(eta)
        g = H.T @ (sigma - D)
        W = sigma * (1 - sigma)
        Hmat = H.T @ (W[:, None] * H)

        # Newton step: Hmat * step = g
        try:
            step = np.linalg.solve(Hmat, g)
        except np.linalg.LinAlgError:
            break

        # Armijo backtracking line search
        nll_old = -np.sum(D * eta - np.logaddexp(0, eta))
        g_dot_step = g @ step
        if g_dot_step >= 0:  # the step is not a descent direction (rare), fall back to steepest descent
            step = -g
            g_dot_step = g @ step

        alpha = 1.0
        accepted = False
        for _ in range(50):
            new_eta = eta + alpha * (H @ step)
            nll_new = -np.sum(D * new_eta - np.logaddexp(0, new_eta))
            if nll_new <= nll_old + 1e-4 * alpha * g_dot_step:
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break

        theta = theta + alpha * step
        if np.linalg.norm(g, ord=np.inf) < tol:
            return theta, it + 1

    return theta, it + 1


def estimate_propensity_sieve(data, h_n=None):
    """Estimate the propensity score by the sieve method, writing the result to data['pi_estimate'].

    Returns (data, h_n, info_dict).
    """
    X = np.asarray(data["X"]).ravel()
    D = np.asarray(data["D"]).astype(float)
    n = X.size
    if h_n is None:
        h_n = default_sieve_dim(n)

    H = sieve_basis(X, h_n).astype(float)

    # column standardization (the constant column is kept as 1), improving the numerical
    # conditioning of the Newton step
    mu = H.mean(axis=0)
    std = H.std(axis=0)
    mu[0] = 0.0     # the constant column is not shifted
    std[0] = 1.0    # the constant column is not scaled
    std[std < 1e-12] = 1.0
    Hn = (H - mu) / std

    theta, n_iter = _newton_logistic(Hn, D)
    pi_hat = expit(Hn @ theta)
    data["pi_estimate"] = pi_hat

    # map back to the original parameter space (for printing the coefficients)
    theta_orig = theta / std
    intercept = -mu @ theta_orig
    info = {
        "success": True,
        "n_iter": n_iter,
        "neg_log_lik": float(-np.sum(D * (Hn @ theta) - np.logaddexp(0, Hn @ theta))),
        "theta_orig": theta_orig,
        "intercept": float(intercept),
    }
    return data, h_n, info


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
    from data_generation import load_config, generate_dataset, describe_data

    cfg = load_config()
    seed = cfg["experiment"]["random_seed"]
    first_model = list(cfg["outcome_models"])[0]
    first_n = cfg["design"]["sample_sizes"][0]

    data = generate_dataset(cfg, first_model, first_n, seed)

    print("=" * 72)
    print(f"[test] model={first_model}, n={first_n}")
    print(f"  true pi:      min={data['pi'].min():.3f}  max={data['pi'].max():.3f}  "
          f"mean={data['pi'].mean():.3f}")
    print(f"  D distribution: D=1: {int((data['D']==1).sum())}, "
          f"D=0: {int((data['D']==0).sum())}")

    data, h_n, info = estimate_propensity_sieve(data)
    print(f"  sieve basis dimension h_n = {h_n}  (default floor(2*n^(1/11)) = {default_sieve_dim(first_n)})")
    print(f"  Newton optimization: n_iter={info['n_iter']}, neg log L = {info['neg_log_lik']:.3f}")
    print(f"  intercept a0 = {info['intercept']:.4f}; "
          f"first 3 sieve coefficients = {np.array2string(info['theta_orig'][1:4], precision=4)}")

    pi = data["pi"]
    pe = data["pi_estimate"]
    err = pe - pi
    print(f"  estimated pi_est: min={pe.min():.3f}  max={pe.max():.3f}  mean={pe.mean():.3f}")
    print(f"  error:        MAE={np.abs(err).mean():.4f}  "
          f"RMSE={np.sqrt(np.mean(err**2)):.4f}  "
          f"corr={np.corrcoef(pi, pe)[0, 1]:.4f}")

    print("\nUpdated table field structure:")
    describe_data(data, label="after sieve estimation")
