# Location-Invariant Extremal QTE Estimator

<p align="center">
  <a href="https://arxiv.org/abs/2609.04018"><img src="https://img.shields.io/badge/arXiv-2609.04018-b31b1b.svg" alt="arXiv"></a>
</p>

Simulation code accompanying the paper **《A Location-Invariant Estimator of Extremal Quantile Treatment Effects for Heavy-Tailed Distributions》**.

The paper addresses estimation of extremal quantile treatment effects (QTE) under heavy-tailed potential outcomes, with two contributions: (1) the location-invariant Fraga extreme value index (EVI) estimator is adapted to the causal setting via inverse propensity score weighting (the Causal Fraga estimator); (2) the original multiplicative extrapolation formula is replaced by a difference-based scheme, under which the location parameter cancels when quantile differences are taken. The resulting QTE estimator is location invariant, and the paper establishes its consistency, asymptotic normality, and a consistent variance estimator (valid confidence intervals).

## Repository Layout

```
├── configs/
│   └── data_generation.yaml        # Global config: DGP (H1/H2/H3), propensity score, sample sizes,
│                                   #   quantile levels, k0 hyperparameters, MC replications,
│                                   #   location shifts μ, parallel workers, inference settings
├── data/
│   └── data_generation.py          # Data generation module (with __main__ self-test: prints data overview)
├── estimate/
│   ├── estimate_propensity_sieve.py# Nonparametric sieve logistic regression for the propensity score
│   │                               #   (Newton-Raphson + Armijo backtracking line search)
│   ├── estimate_quantile_empirical.py # IPW-weighted empirical quantile estimator (Firpo-style)
│   └── estimate_quantile_real.py   # True (theoretical) quantiles: numerical integration + brentq root finding
├── EVI/
│   ├── causal_hill.py              # Causal Hill EVI estimator (Deuber baseline, Eq. (7) of the paper)
│   └── causal_fraga.py             # Causal Fraga EVI estimator (paper's contribution, Eqs. (8)(9))
├── QTE/
│   ├── Zhang.py                    # Intermediate-level IPW empirical quantile QTE (Firpo/Zhang-style)
│   ├── Deuber.py                   # Hill EVI + Weissman multiplicative extrapolation QTE (Deuber baseline)
│   ├── Deuber_diff.py              # Hill EVI + difference extrapolation QTE
│   ├── QTE_Fraga.py                # Fraga EVI + Weissman multiplicative extrapolation QTE (paper's Appendix B)
│   ├── QTE_Fraga_diff.py           # Fraga EVI + difference extrapolation QTE (proposed location-invariant estimator)
│   └── QTE_real.py                 # True (theoretical) QTE: q_{Y1}(τ) − q_{Y0}(τ)
├── estimate_k0.py                  # Adaptive auxiliary level k0 estimation (k0 = k^m, m = min(m*_1,m*_0) − σ)
├── QTE_experiment.py               # Experiment 1: location-shift sensitivity Monte Carlo (reproduces Fig. 1)
├── QTE_k_experiment.py             # Experiment 2: threshold k sensitivity analysis
├── QTE_ci_experiment.py            # Experiment 3: confidence interval coverage (bootstrap + asymptotic CI)
└── results/                        # Output figures (generated automatically)
```

## Requirements

Python 3.10+ with: `numpy`, `scipy`, `pyyaml`, `matplotlib`, `tqdm`.

```bash
pip install numpy scipy pyyaml matplotlib tqdm
```

## Experiments and How to Run

All three experiment scripts are run from the repository root. All CLI arguments are optional (defaults are read from `configs/data_generation.yaml`). Execution is parallelized over processes.

### Experiment 1: Location-Shift Sensitivity (paper Fig. 1 / Section 5)

For each model H1/H2/H3, under a shared location shift μ (the true QTE is translation-invariant), repeat the Monte Carlo R times, compare the QTE estimates and squared errors of 4 methods, and output boxplots `results/QTE_shift_{est,sqerr}_H*.png`.

```bash
python QTE_experiment.py
python QTE_experiment.py --replications 500 --shifts 0 1 2 5 10
# Optional arguments:
#   --replications R     replications per scenario (default: design.replications = 1000)
#   --sample-sizes n1..  list of sample sizes (default: [1000, 5000])
#   --shifts s1 s2 ..    list of location shifts μ (default: [0,10,20,30])
#   --workers W          number of parallel worker processes (default: design.num_workers = 10, <=0 = auto)
```

### Experiment 2: Threshold k Sensitivity Analysis

Fix (model, n, τ) and sweep the number of top-k observations k (anchor level α_n = k/n); plot the mean of the QTE estimates and the MSE of each method as functions of k (`results/QTE_k_{mean,mse}_H*.png`). For the Fraga-family methods, k0 adapts group-wise with k (k0 = k^m).

```bash
python QTE_k_experiment.py
# Optional arguments:
#   --replications R     number of replications
#   --sample-sizes n1..  list of sample sizes
#   --k-grid k1 k2 ..    explicit k sequence (default: 10 equally spaced points in [50, n/2])
#   --workers W          number of parallel worker processes
```

### Experiment 3: Confidence Interval Coverage (paper Section 5 coverage experiment)

With μ fixed at 0, repeat R times: point estimation on the original sample → B bootstrap resamples to estimate the standard error → normal-approximation CI; additionally, the `Fraga_diff_asymp` method constructs an analytic asymptotic CI from the analytic standard error of Eqs. (18)–(22) of the paper. Outputs coverage with Wilson intervals `results/QTE_ci_cov_H*.png`.

```bash
python QTE_ci_experiment.py
python QTE_ci_experiment.py --replications 200 --bootstrap 200
# Optional arguments:
#   --replications R     number of Monte Carlo replications
#   --sample-sizes n1..  list of sample sizes
#   --bootstrap B        bootstrap resamples per replication (default: inference.bootstrap_replications)
#   --workers W          number of parallel worker processes
# (Nominal confidence level is controlled by inference.confidence_level in the config, default 0.9)
```

### Method Reference

| Method name in experiments | EVI estimator | Extrapolation | Implementation |
|---|---|---|---|
| `Deuber` | Causal Hill | Multiplicative (Weissman) | `QTE/Deuber.py` |
| `Deuber_diff` | Causal Hill | Difference extrapolation | `QTE/Deuber_diff.py` |
| `Fraga_alpha` | Causal Fraga | Multiplicative (Weissman) | `QTE/QTE_Fraga.py` |
| `Fraga_diff` (proposed method) | Causal Fraga | Difference extrapolation | `QTE/QTE_Fraga_diff.py` |
| `Fraga_diff_asymp` (Experiment 3 only) | Causal Fraga | Difference extrapolation + analytic SE of Eqs. (18)–(22) | `fraga_diff_asymp_ci` in `QTE_ci_experiment.py` |

## Module Self-Tests

Each library script can be run standalone for quick verification (defaults to the first model and sample size in the config):

```bash
python data/data_generation.py                      # Data overview: group sizes, quantile exceedance sanity check
python estimate/estimate_propensity_sieve.py        # Propensity score estimation errors (MAE/RMSE/correlation)
python estimate/estimate_quantile_empirical.py      # IPW empirical quantiles and QTE at each τ level
python estimate/estimate_quantile_real.py           # True quantiles and true QTE at each τ level
python estimate_k0.py                               # Adaptive k0 / m / β_n for each (model, n)
python QTE/QTE_real.py                              # True QTE
```

## Configuration (configs/data_generation.yaml)

Main fields: `outcome_models` (H1: linear + Student-t noise; H2: exponential + Fréchet noise; H3: Pareto with covariate-dependent shape; each with theoretical EVI γ); `design.sample_sizes`, `design.quantile_levels` (extreme levels τ_n_1..3 and intermediate level α_n = n^{-0.35}); `design.k0_init_formula / k0_sigma / k0_m_lower` (adaptive k0 hyperparameters); `design.shifts` (list of location shifts); `design.replications`; `design.num_workers`; and `inference.confidence_level`, `inference.bootstrap_replications` (CI experiment settings).

## Output

All experiment figures are written to `results/` (dpi=150) with the following naming scheme:

- `QTE_shift_est_{model}.png` / `QTE_shift_sqerr_{model}.png` — Experiment 1 estimate / squared-error boxplots (rows = n, columns = τ_n, subplot x-axis = μ)
- `QTE_k_mean_{model}.png` / `QTE_k_mse_{model}.png` — Experiment 2 mean / MSE curves over k
- `QTE_ci_cov_{model}.png` — Experiment 3 coverage (point estimate + 95% Wilson error bars + nominal level reference line)
