# -*- coding: utf-8 -*-
"""QTE estimator k-sensitivity analysis: mean and MSE line plots across k.

Modeled on the Hill EVI k-sensitivity plots: fix (model, sample size, target level τ),
vary the number of top-k observations used by the Hill-family estimators (anchor level
α_n = k/n), repeat R times to estimate the QTE, and plot the mean of the QTE estimates
(which should be close to the true QTE) and the MSE (relative to the true QTE) as
functions of k.

Methods:
  - Deuber      : Hill EVI + Weissman extrapolation (anchor α_n = k/n)
  - Deuber_diff : Hill EVI + difference extrapolation (anchor α_n = k/n, β_n = k^(2/3)/n uniform)
  - Fraga_alpha : Fraga EVI + Weissman extrapolation (anchor α_n = k/n, k0 adapts with k per group)
  - Fraga_diff  : Fraga EVI + difference extrapolation (k0 adapts with k per group)

One figure per model per metric (mean and MSE drawn separately): rows = sample size n,
columns = tau_n levels, x-axis = k (top observations), one line per method, with the true
QTE reference line.

CLI args:
  --replications R        number of repetitions (default: from config)
  --sample-sizes n1 n2..  list of sample sizes
  --k-grid k1 k2 ..       explicit k sequence (default: config design.k_grid or a log grid)
  --workers W             number of parallel processes

Run:
  D:\\Miniconda\\python.exe QTE_k_experiment.py
"""
import os
import sys
from pathlib import Path

# Limit the BLAS/OpenMP threads inside each worker to 1 (must be set before importing numpy)
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
})

import numpy as np

_BASE = Path(__file__).resolve().parent
for _sub in ("data", "estimate", "EVI", "QTE"):
    sys.path.insert(0, str(_BASE / _sub))

from data_generation import load_config, generate_dataset, tau_levels  # noqa: E402
from estimate_propensity_sieve import estimate_propensity_sieve  # noqa: E402
from estimate_k0 import estimate_k0_by_group, fallback_beta  # noqa: E402
from Deuber import estimate_qte_extrapolation  # noqa: E402
from Deuber_diff import estimate_qte_diff  # noqa: E402
from QTE_Fraga import estimate_qte_extrapolation_fraga  # noqa: E402
from QTE_Fraga_diff import estimate_qte_diff_fraga  # noqa: E402
from QTE_real import real_qte  # noqa: E402

METHODS = ["Deuber", "Deuber_diff", "Fraga_alpha", "Fraga_diff"]
COLORS = {
    "Deuber": "#ed7d31",
    "Deuber_diff": "#5b9bd5",
    "Fraga_alpha": "#70ad47",
    "Fraga_diff": "#7030a0",
}
LABELS = {
    "Deuber": "Causal Hill (mult extrapolation)",
    "Deuber_diff": "Causal Hill (diff extrapolation)",
    "Fraga_alpha": "Causal Fraga (mult extrapolation)",
    "Fraga_diff": "Causal Fraga (diff extrapolation)",
}


def default_k_grid(cfg, n):
    """Default k grid: 10 equally spaced points in [50, n/2]."""
    if cfg["design"].get("k_grid"):
        return [float(k) for k in cfg["design"]["k_grid"]]
    k_min = 50
    k_max = n // 2
    return np.linspace(k_min, k_max, 10)


def estimate_qte_by_method(cfg, data, tau, k, n, truth):
    """Given k (top observations), estimate the QTE of each method (tau is the target upper-tail probability).

    cfg : experiment config (used for the k0 adaptive estimation)
    data: data dict containing Y, D, pi_estimate
    Returns {method: qte} (may be nan when the estimate fails or goes out of the tail range).
    """
    alpha_n = float(k) / n                 # anchor level
    beta_dd = float(k) ** (2.0 / 3.0) / n  # uniform auxiliary level for Deuber_diff
    beta_fb = fallback_beta(cfg, n)        # fallback auxiliary level for Fraga
    out = {}
    try:
        out["Deuber"] = estimate_qte_extrapolation(data, alpha_n, tau)["qte_ext"]
    except Exception:
        out["Deuber"] = np.nan
    try:
        out["Deuber_diff"] = estimate_qte_diff(data, alpha_n, beta_dd, tau)["qte_ext"]
    except Exception:
        out["Deuber_diff"] = np.nan
    try:
        # Fraga's k0 varies with k: group-adaptive β_n (k0 = k^m, k = n·α_n)
        k0res = estimate_k0_by_group(cfg, data, n, k=k)
        beta_t = k0res["beta_treated"]
        beta_c = k0res["beta_control"]
        out["Fraga_alpha"] = estimate_qte_extrapolation_fraga(
            data, beta_fb, alpha_n, tau,
            beta_treated=beta_t, beta_control=beta_c)["qte_ext"]
    except Exception:
        out["Fraga_alpha"] = np.nan
    try:
        k0res = estimate_k0_by_group(cfg, data, n, k=k)
        beta_t = k0res["beta_treated"]
        beta_c = k0res["beta_control"]
        out["Fraga_diff"] = estimate_qte_diff_fraga(
            data, alpha_n, beta_fb, tau,
            beta_treated=beta_t, beta_control=beta_c)["qte_ext"]
    except Exception:
        out["Fraga_diff"] = np.nan
    return out


def run_experiment(cfg, model, n, k_grid, replications, base_seed):
    """Run R repetitions over all k for a single (model, sample size), returning
    {tau_name: {k: {method: ndarray}}}."""
    levels = dict(tau_levels(cfg, n))
    tau_names = [name for name in levels if name.startswith("tau_n")]
    truth = {name: real_qte(cfg, model, 1.0 - levels[name])["qte"]
             for name in tau_names}
    results = {name: {k: {m: [] for m in METHODS} for k in k_grid}
               for name in tau_names}
    for rep in range(replications):
        seed = base_seed + rep
        data = generate_dataset(cfg, model, n, seed)
        data, _h_n, _info = estimate_propensity_sieve(data)
        for k in k_grid:
            for name in tau_names:
                est = estimate_qte_by_method(cfg, data, levels[name], k, n, truth[name])
                for m in METHODS:
                    results[name][k][m].append(est[m])
    for name in tau_names:
        for k in k_grid:
            for m in METHODS:
                results[name][k][m] = np.asarray(results[name][k][m], dtype=float)
    return results, truth


def summarize(results, truth):
    """Aggregate the repetition arrays into mean and MSE: returns {tau_name: {k: {method: (mean, mse)}}}."""
    summ = {}
    for name, by_k in results.items():
        summ[name] = {}
        for k, by_m in by_k.items():
            summ[name][k] = {}
            for m, arr in by_m.items():
                v = arr[np.isfinite(arr)]
                mean = float(v.mean()) if v.size else np.nan
                mse = float(np.mean((v - truth[name]) ** 2)) if v.size else np.nan
                summ[name][k][m] = (mean, mse)
    return summ


def plot_k_curves(summ, truth, tau_names, tau_formulas, model, sample_sizes,
                  k_grid_by_n, out_dir):
    """Two figures per model (mean and MSE drawn separately): rows = sample size n,
    columns = tau_n levels, subplot x-axis = k (top observations), one line per method.
    The MSE figure and the mean figure of H1 use log scale (major ticks at powers of 10,
    no minor ticks); the mean figures of H2/H3 use linear scale.
    The axis captions are placed at the outermost edges; the row label n is to the right
    of the y-axis of the rightmost column; the τ_n label is shown only at the top of the
    first row."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullLocator

    out_dir.mkdir(parents=True, exist_ok=True)
    n_tau = len(tau_names)
    n_n = len(sample_sizes)
    ks_by_n = {n: np.asarray(k_grid_by_n[n], dtype=float) for n in sample_sizes}
    for metric, ylabel, suffix in (("mean", "mean of QTE estimates", "mean"),
                                   ("mse", "MSE", "mse")):
        fig, axes = plt.subplots(n_n, n_tau, figsize=(4.2 * n_tau, 3.4 * n_n),
                                 squeeze=False)
        for r, n in enumerate(sample_sizes):
            ks = ks_by_n[n]
            for c, name in enumerate(tau_names):
                ax = axes[r][c]
                for m in METHODS:
                    vals = [summ[n][name][k][m][0 if metric == "mean" else 1]
                            for k in ks]
                    ax.plot(ks, vals, marker="o", markersize=2.5, linewidth=1.2,
                            color=COLORS[m], label=LABELS[m])
                # H1's mean figure and all MSE figures use log scale (major ticks fixed to
                # integer powers of 10 with 10x spacing, minor ticks disabled); H2/H3's mean
                # figures use linear scale
                if metric == "mse" or (metric == "mean" and model == "H1"):
                    ax.set_yscale("log")
                    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
                    ax.yaxis.set_minor_locator(NullLocator())
                if metric == "mean":
                    ax.axhline(truth[n][name], color="black", linestyle="--",
                               linewidth=1.2, alpha=0.7, label="true QTE")
                if r == 0:
                    ax.set_title(rf"$\tau_n = {tau_formulas[name]}$")
                # x-axis fixed to 5 equally spaced ticks: 0, max_k/4, ..., max_k
                # (e.g. n=1000 shows 0,200,400,600,800,1000)
                ticks = np.linspace(0, ks.max(), 6)
                ax.set_xticks(ticks)
                ax.set_xticklabels([int(t) for t in ticks], fontsize=7)
                ax.grid(alpha=0.3, which="major", axis="y")
                # row label (sample size): placed to the right of the y-axis of the
                # rightmost-column subplot
                if c == n_tau - 1:
                    ax.set_ylabel(f"n = {n}", fontsize=11)
                    ax.yaxis.set_label_position("right")
        # outermost axis captions
        fig.supxlabel("k-number of top observations", fontsize=13, y=0.04)
        fig.supylabel(ylabel, fontsize=13)
        handles = [plt.Line2D([0], [0], color=COLORS[m], label=LABELS[m])
                   for m in METHODS]
        if metric == "mean":
            handles.append(plt.Line2D([0], [0], color="black", linestyle="--",
                                      label="true QTE"))
        fig.legend(handles=handles, loc="upper center", ncol=len(handles),
                   fontsize=9, frameon=False)
        fig.suptitle(model, fontsize=13, y=0.93)
        fig.tight_layout(rect=(0.03, 0.05, 0.97, 0.94))
        out_path = out_dir / f"QTE_k_{suffix}_{model}.png"
        fig.savefig(out_path, dpi=150)
        print(f"K-sensitivity plot saved: {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    import argparse
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="QTE k-sensitivity analysis")
    parser.add_argument("--replications", type=int, default=None)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--k-grid", type=float, nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    seed_base = cfg["experiment"]["random_seed"]
    sample_sizes = args.sample_sizes if args.sample_sizes else list(cfg["design"]["sample_sizes"])
    replications = args.replications if args.replications else cfg["design"].get("replications", 1000)
    workers = args.workers if args.workers is not None else int(cfg["design"].get("num_workers", 0))
    models = list(cfg["outcome_models"])

    # each (model, sample size) may have a different k grid
    k_grid_by_n = {}
    for n in sample_sizes:
        k_grid_by_n[n] = args.k_grid if args.k_grid else default_k_grid(cfg, n)

    print("=" * 72)
    print("QTE estimator k-sensitivity analysis (k = number of top observations)")
    print(f"  models: {models}")
    print(f"  sample sizes: {sample_sizes}")
    print(f"  repetitions R = {replications}")
    for n in sample_sizes:
        print(f"  n={n}: k grid = {[int(k) for k in k_grid_by_n[n]]}")
    print("=" * 72)

    tasks = [(model, n) for model in models for n in sample_sizes]
    if workers > 0:
        max_workers = min(len(tasks), workers)
    else:
        max_workers = min(len(tasks), os.cpu_count() or 1)
    print(f"  tasks: {len(tasks)}, parallel processes: {max_workers}")

    all_results = {model: {n: None for n in sample_sizes} for model in models}
    all_truth = {model: {n: None for n in sample_sizes} for model in models}
    from tqdm import tqdm

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(run_experiment, cfg, model, n, k_grid_by_n[n],
                        replications, seed_base): (model, n)
            for model, n in tasks
        }
        with tqdm(total=len(futures), desc="Running k-sensitivity", unit="task",
                  ncols=100, dynamic_ncols=True, mininterval=0.0, miniters=1) as pbar:
            for fut in as_completed(futures):
                model, n = futures[fut]
                try:
                    all_results[model][n], all_truth[model][n] = fut.result()
                except Exception as e:
                    print(f"[model {model}, n={n}] failed: {e}")
                    raise
                pbar.set_postfix_str(f"{model} n={n}")
                pbar.update(1)

    tau_formulas = {q["name"]: q["formula"] for q in cfg["design"]["quantile_levels"]
                    if q["name"].startswith("tau_n")}
    out_dir = Path(__file__).resolve().parent / "results"
    for model in models:
        summ_by_n = {}
        truth_by_n = {}
        for n in sample_sizes:
            summ_by_n[n] = summarize(all_results[model][n], all_truth[model][n])
            truth_by_n[n] = all_truth[model][n]
        tau_names = list(next(iter(summ_by_n.values())).keys())
        plot_k_curves(summ_by_n, truth_by_n, tau_names, tau_formulas,
                      model, sample_sizes, k_grid_by_n, out_dir)

    print("\nExperiment finished.")
