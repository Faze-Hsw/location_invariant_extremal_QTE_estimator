# -*- coding: utf-8 -*-
"""QTE estimator location-sensitivity Monte Carlo experiment: mu -> boxplots.

Compare the QTE estimates of the various methods at quantile level 1-tau_n under different
shared location shifts μ (the true QTE is given by QTE_real.py), covering all tau_n_* levels
in the config:

  - Deuber      : Hill EVI + Weissman extrapolation (anchor alpha_n)
  - Deuber_diff : Hill EVI + difference extrapolation (two anchors alpha_n / beta_n)
  - Fraga_alpha : Fraga EVI + Weissman power extrapolation (anchor alpha_n)
  - Fraga_diff  : Fraga EVI + difference extrapolation (two anchors alpha_n / beta_n)

The x-axis is the shared location shift μ (added at the model level as Y(j) = μ + original
distribution, identical across H1/H2/H3). The true QTE = q_{Y1}(tau) - q_{Y0}(tau) is
translation-invariant in μ (the two ends cancel), so the change of the MSE with μ reflects
purely the location sensitivity of each estimator.

Procedure:
  1. For each model / sample size / μ, generate samples with different location distributions;
  2. Each method estimates the QTE at each quantile level 1-tau_n;
  3. Repeat R times and collect the estimates of each estimator;
  4. Draw boxplots: two figures per model (estimate / squared error), rows = sample size n,
     columns = tau_n levels, with μ on the subplot x-axis and one boxplot per method at each μ.
     The estimate figure's y-axis = QTE estimate (log scale, with the true QTE reference line);
     the squared-error figure's y-axis = Squared Error (log scale);
     the axis captions (location shift / metric) are placed at the outermost edges.

CLI args (all optional, defaults from the config):
  --replications R        number of repetitions, e.g. --replications 500
  --sample-sizes n1 n2..  list of sample sizes
  --shifts s1 s2 ..       list of μ location shifts
  --workers W             number of parallel processes

Run:
  D:\\Miniconda\\python.exe QTE_experiment.py
  D:\\Miniconda\\python.exe QTE_experiment.py --replications 500 --shifts 0 1 2 5 10
"""
import os
import sys
from pathlib import Path

# Limit the BLAS/OpenMP threads inside each worker process to 1 to avoid the multiple
# subprocesses of ProcessPoolExecutor competing with OpenBLAS/MKL multithreading, which
# would otherwise make the CPU utilization extremely low.
# Must be set before importing numpy.
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
})

import numpy as np

# import repo-local scripts
_BASE = Path(__file__).resolve().parent
for _sub in ("data", "estimate", "EVI", "QTE"):
    sys.path.insert(0, str(_BASE / _sub))

from data_generation import load_config, generate_dataset, tau_levels  # noqa: E402
from estimate_propensity_sieve import estimate_propensity_sieve  # noqa: E402
from Deuber import estimate_qte_extrapolation  # noqa: E402
from Deuber_diff import estimate_qte_diff  # noqa: E402
from QTE_Fraga import estimate_qte_extrapolation_fraga  # noqa: E402
from QTE_Fraga_diff import estimate_qte_diff_fraga  # noqa: E402
from QTE_real import real_qte  # noqa: E402
from estimate_k0 import estimate_k0_by_group, fallback_beta  # noqa: E402

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


def estimate_qte_by_method(data_s, tau, alpha_n, beta_n,
                           beta_treated=None, beta_control=None,
                           beta_deuber_diff=None):
    """Estimate the QTE at quantile level 1-tau for each method (tau is an upper-tail probability).

    data_s          : the shifted data dict (containing Y, D, pi_estimate)
    beta_treated    : optional, treated-group β_n from the adaptive k0 (used by the Fraga methods)
    beta_control    : optional, control-group β_n from the adaptive k0 (used by the Fraga methods)
    beta_deuber_diff: optional, uniform β_n for Deuber_diff (k0 = k^(2/3) fixed, defaults to beta_n)
    Returns {method: qte}.
    """
    beta_dd = float(beta_deuber_diff) if beta_deuber_diff is not None else float(beta_n)
    return {
        "Deuber": estimate_qte_extrapolation(data_s, alpha_n, tau)["qte_ext"],
        "Deuber_diff": estimate_qte_diff(data_s, alpha_n, beta_dd, tau)["qte_ext"],
        "Fraga_alpha": estimate_qte_extrapolation_fraga(
            data_s, beta_n, alpha_n, tau,
            beta_treated=beta_treated, beta_control=beta_control)["qte_ext"],
        "Fraga_diff": estimate_qte_diff_fraga(
            data_s, alpha_n, beta_n, tau,
            beta_treated=beta_treated, beta_control=beta_control)["qte_ext"],
    }


def run_experiment(cfg, model, n, shifts, replications, base_seed):
    """Run R repetitions for a single (model, sample size), covering all tau_n_* levels.

    The x-axis is the shared location shift μ (added at the model level as Y(j) = μ + original
    distribution): the values in shifts are used directly as μ (by setting cfg["design"]["mu"]),
    deep-copying the config and regenerating the samples. Under the same seed, X, U, D are
    unchanged and only Y varies with μ. The true QTE is translation-invariant (μ cancels on both
    ends), so truth_by is independent of μ.

    The sieve propensity score depends only on X and D, not on the outcome Y (nor μ), so the
    propensity score is estimated only once per repetition and reused across all μ scenarios.
    The Fraga methods use the group-adaptive k0 (estimated from the μ=0 data).

    Returns (results, truth_by, tau_vals).
    results : {tau_name: {μ: {method: np.ndarray}}} (QTE estimates over R repetitions)
    truth_by: {tau_name: true QTE (theoretical value, translation-invariant in μ)}
    tau_vals: {tau_name: upper-tail probability tau}
    """
    import copy

    levels = dict(tau_levels(cfg, n))
    tau_names = [name for name in levels if name.startswith("tau_n")]
    alpha_n = levels["alpha_n"]
    beta_fb = fallback_beta(cfg, n)   # fallback β_n (on adaptive failure; group value takes priority)
    # uniform β_n for Deuber_diff: k0 from the config formula (k0_deuber_diff_formula, default
    # k^(2/3), k = n^0.65), not following the Fraga group estimate; β = k0 / n
    k0_dd = eval(cfg["design"]["k0_deuber_diff_formula"], {"n": n, "log": np.log})
    beta_deuber_diff = float(k0_dd) / n
    tau_vals = {name: levels[name] for name in tau_names}
    # translation-invariant in μ: the true QTE is independent of μ
    truth_by = {name: real_qte(cfg, model, 1.0 - levels[name])["qte"]
                for name in tau_names}

    results = {name: {s: {m: [] for m in METHODS} for s in shifts}
               for name in tau_names}
    for rep in range(replications):
        seed = base_seed + rep
        data0 = generate_dataset(cfg, model, n, seed)
        data0, _h_n, _info = estimate_propensity_sieve(data0)
        pi0 = data0["pi_estimate"]
        # group k0: treated/control-group β_n (the Fraga location is unchanged, so estimating
        # from the μ=0 data suffices)
        k0res = estimate_k0_by_group(cfg, data0, n)
        beta_t = k0res["beta_treated"]
        beta_c = k0res["beta_control"]
        for s in shifts:
            cfg_s = copy.deepcopy(cfg)
            cfg_s["design"]["mu"] = s                 # x-axis: shared location shift μ
            data_s = generate_dataset(cfg_s, model, n, seed)
            data_s["pi_estimate"] = pi0              # X, D unchanged -> reuse the propensity score
            for name in tau_names:
                est = estimate_qte_by_method(data_s, levels[name], alpha_n, beta_fb,
                                             beta_t, beta_c,
                                             beta_deuber_diff=beta_deuber_diff)
                for m in METHODS:
                    results[name][s][m].append(est[m])
    for name in tau_names:
        for s in shifts:
            for m in METHODS:
                results[name][s][m] = np.asarray(results[name][s][m], dtype=float)
    return results, truth_by, tau_vals


def mse(values, truth):
    """Mean squared error between the estimates and the truth (ignoring nan)."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.nan
    return float(np.mean((v - truth) ** 2))


def formula_to_latex(formula):
    r"""Convert the quantile formula in the config (a Python expression) to a LaTeX fraction for mathtext.

    Example: "5 / n" -> r"\frac{5}{n}";  "5 / (n * log(n))" -> r"\frac{5}{n\log(n)}"
    """
    s = formula.replace(" ", "")
    s = s.replace("log(n)", r"\log(n)")
    s = s.replace("*", r"\, ")
    if "/" in s:
        num, den = s.split("/", 1)
        den = den.strip()
        if den.startswith("(") and den.endswith(")"):
            den = den[1:-1]
        return rf"\frac{{{num}}}{{{den}}}"
    return s


def plot_shift_boxplots(all_results, truth_by, tau_names, tau_formulas,
                        models, sample_sizes, shifts, out_dir):
    """Two figures per model (estimate / squared error): rows = sample size n, columns = tau_n levels.
    Inside each subplot the x-axis is μ, with one boxplot per method at each μ (showing the
    distribution of the R estimates). Figure 1's y-axis = QTE estimate (log scale, with the true
    QTE reference line); figure 2's y-axis = squared error (estimate - truth)^2 (log scale).
    The axis captions (location shift / metric) are placed at the outermost edges."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import LogLocator, NullLocator
        from matplotlib.patches import Patch
    except ImportError as e:
        print(f"[Warning] matplotlib unavailable, skip plotting: {e}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    n_methods = len(METHODS)
    width = 0.8 / n_methods
    n_tau = len(tau_names)
    n_n = len(sample_sizes)
    # metric: (y-axis label, whether it is the estimate figure, filename suffix)
    metrics = (("QTE estimate", True, "est"),
               ("Squared Error", False, "sqerr"))
    for model in models:
        for metric, is_est, suffix in metrics:
            fig, axes = plt.subplots(n_n, n_tau, figsize=(4.6 * n_tau, 3.6 * n_n),
                                     squeeze=False)
            for r, n in enumerate(sample_sizes):
                for c, name in enumerate(tau_names):
                    ax = axes[r][c]
                    _tau, truth_qte = truth_by[model][n][name]
                    results, _truth, _tau_vals = all_results[model][n]
                    x = np.arange(len(shifts))
                    for i, m in enumerate(METHODS):
                        positions = x + (i - (n_methods - 1) / 2) * width
                        if is_est:
                            # estimates (filter nan; log scale can only show positive values,
                            # so also filter non-positive ones)
                            data = [results[name][s][m]
                                    [np.isfinite(results[name][s][m])] for s in shifts]
                            data = [d[d > 0] for d in data]
                        else:
                            # squared error is always non-negative; log scale can only show
                            # positive values, so filter zeros (and nan)
                            data = [((results[name][s][m] - truth_qte) ** 2)
                                    [np.isfinite(results[name][s][m])] for s in shifts]
                            data = [d[d > 0] for d in data]
                        bp = ax.boxplot(data, positions=positions, widths=width * 0.8,
                                        patch_artist=True, showfliers=False,
                                        manage_ticks=False, whis=(10, 90))
                        for patch in bp["boxes"]:
                            patch.set_facecolor(COLORS[m])
                            patch.set_alpha(0.7)
                        # mark the mean with a black dot at each μ (skip empty boxes)
                        means = [float(d.mean()) if d.size else np.nan for d in data]
                        ax.plot(positions, means, "ko", markersize=2)
                    ax.set_xticks(x)
                    ax.set_xticklabels([f"{s:.1f}" for s in shifts])
                    # the τ_n label is shown only at the top of the first row of subplots
                    if r == 0:
                        ax.set_title(rf"$\tau_n = {tau_formulas[name]}$")
                    # both figures use log scale: major ticks fixed to integer powers of 10
                    # (10x spacing), minor ticks disabled; the estimate figure adds the true
                    # QTE reference line (must be positive; skipped automatically if negative)
                    ax.set_yscale("log")
                    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
                    ax.yaxis.set_minor_locator(NullLocator())
                    if is_est and truth_qte > 0:
                        ax.axhline(truth_qte, color="black", linestyle="--",
                                   linewidth=1.2, alpha=0.7)
                    ax.grid(alpha=0.3, which="major", axis="y")
                    # row label (sample size): placed to the right of the y-axis of the
                    # rightmost-column subplot
                    if c == n_tau - 1:
                        ax.set_ylabel(f"n = {n}", fontsize=11)
                        ax.yaxis.set_label_position("right")
            # outermost axis captions
            fig.supxlabel(r"location shift $u$", fontsize=13, y=0.04)
            fig.supylabel(metric, fontsize=13)
            handles = [Patch(facecolor=COLORS[m], label=LABELS[m]) for m in METHODS]
            if is_est:
                handles.append(plt.Line2D([0], [0], color="black", linestyle="--",
                                          label="true QTE"))
            fig.legend(handles=handles, loc="upper center", ncol=len(handles),
                       fontsize=10, frameon=False)
            # move the suptitle down to avoid overlapping the top legend
            fig.suptitle(model, fontsize=13, y=0.93)
            fig.tight_layout(rect=(0, 0.04, 1, 0.94))

            out_path = out_dir / f"QTE_shift_{suffix}_{model}.png"
            fig.savefig(out_path, dpi=150)
            print(f"Boxplot saved: {out_path}")
            plt.close(fig)


if __name__ == "__main__":
    import argparse
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="QTE shift-sensitivity Monte Carlo experiment")
    parser.add_argument("--replications", type=int, default=None,
                        help="repeat count per scenario (default: config design.replications)")
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=None,
                        help="list of sample sizes n (default: config design.sample_sizes)")
    parser.add_argument("--shifts", type=float, nargs="+", default=None,
                        help="list of outcome shifts (default: config design.shifts)")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel worker processes (default: config design.num_workers, <=0 = auto)")
    args = parser.parse_args()

    cfg = load_config()
    seed_base = cfg["experiment"]["random_seed"]
    sample_sizes = args.sample_sizes if args.sample_sizes else list(cfg["design"]["sample_sizes"])
    shifts = args.shifts if args.shifts else [float(s) for s in cfg["design"].get("shifts", [0.0])]
    replications = args.replications if args.replications else cfg["design"].get("replications", 1000)
    workers = args.workers if args.workers is not None else int(cfg["design"].get("num_workers", 0))
    models = list(cfg["outcome_models"])

    # target quantile levels: all tau_n_* in the config (upper-tail probabilities,
    # quantile level 1-tau_n)
    tau_names = [name for name, _ in tau_levels(cfg, sample_sizes[0])
                 if name.startswith("tau_n")]

    print("=" * 72)
    print("QTE estimator location-sensitivity Monte Carlo experiment")
    print("  (parallel granularity: model x sample size, one process per combination)")
    print(f"  models: {models}")
    print(f"  sample sizes: {sample_sizes}")
    print(f"  μ location shifts: {shifts}")
    print(f"  target quantile levels: {tau_names} (corresponding quantile level 1-tau_n)")
    print(f"  repetitions R = {replications}")
    print(f"  compared methods: {LABELS}")
    print("=" * 72)

    # a task = one process per (model, sample size) combination; within a process the
    # propensity score is reused across all shifts
    tasks = [(model, n) for model in models for n in sample_sizes]
    if workers > 0:
        max_workers = min(len(tasks), workers)
    else:  # auto: min(number of tasks, number of logical CPU cores)
        max_workers = min(len(tasks), os.cpu_count() or 1)
    print(f"  tasks: {len(tasks)}, parallel processes: {max_workers} (config num_workers={workers})")

    all_results = {model: {n: None for n in sample_sizes} for model in models}
    from tqdm import tqdm

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(run_experiment, cfg, model, n, shifts, replications,
                        seed_base): (model, n)
            for model, n in tasks
        }
        with tqdm(total=len(futures), desc="Running scenarios", unit="task",
                  ncols=100, dynamic_ncols=True, mininterval=0.0, miniters=1) as pbar:
            for fut in as_completed(futures):
                model, n = futures[fut]
                try:
                    all_results[model][n] = fut.result()
                except Exception as e:
                    print(f"[model {model}, n={n}] failed: {e}")
                    raise
                pbar.set_postfix_str(f"{model} n={n}")
                pbar.update(1)

    # aggregate statistics: build truth_by for plotting (MSE is not printed)
    truth_by = {model: {n: {name: None for name in tau_names} for n in sample_sizes}
                for model in models}
    for model in models:
        for n in sample_sizes:
            results, truth, tau_vals = all_results[model][n]
            for name in tau_names:
                truth_by[model][n][name] = (tau_vals[name], truth[name])

    # boxplots (one per model x sample size, each containing subplots for all tau_n levels)
    tau_formulas = {q["name"]: q["formula"] for q in cfg["design"]["quantile_levels"]
                    if q["name"].startswith("tau_n")}
    out_dir = Path(__file__).resolve().parent / "results"
    plot_shift_boxplots(all_results, truth_by, tau_names, tau_formulas,
                        models, sample_sizes, shifts, out_dir)

    print("\nExperiment finished.")
