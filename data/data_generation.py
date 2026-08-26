# -*- coding: utf-8 -*-
"""Generate simulated experiment data from configs/data_generation.yaml.

Covers the three outcome models H1/H2/H3, multiple sample sizes and extreme
quantile levels τ_n.
Usage:
    python data/data_generation.py
"""
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "data_generation.yaml"


def load_config(path: str = CONFIG_PATH) -> dict:
    """Load the data generation config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sample_noise(spec: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate an (n,) array according to the noise distribution."""
    dist = spec["distribution"]
    if dist == "student_t":
        return stats.t.rvs(df=spec["df"], size=n, random_state=rng)
    if dist == "frechet":
        # Fréchet(shape, loc, scale) corresponds to scipy.stats.invweibull(c=shape, loc, scale)
        return stats.invweibull.rvs(
            c=spec["shape"], loc=spec["loc"], scale=spec["scale"],
            size=n, random_state=rng,
        )
    raise ValueError(f"Unknown noise distribution: {dist}")


def _generate_potential(
    spec: dict, X: np.ndarray, rng: np.random.Generator, shared_noise: dict = None
) -> np.ndarray:
    """Generate a single potential outcome (Y1 or Y0).

    Supports three config structures:
      - spec carries its own noise  (H2: Y1/Y0 have independent noise)
      - use model-level shared noise (H1: Y1/Y0 share the same S)
      - directly specify the distribution (H3: Pareto, shape depends on covariates)
    """
    noise = None
    var = None
    if "noise" in spec:
        noise = _sample_noise(spec["noise"], X.size, rng)
        var = spec["noise"]["variable"]
    elif shared_noise is not None:
        noise = _sample_noise(shared_noise, X.size, rng)
        var = shared_noise["variable"]
    if noise is not None:
        # formula string (e.g. "5 * S * (1 + X)" or "C2 * exp(X)"), evaluated after
        # injecting the noise variable name
        env = {"X": X, var: noise, "np": np, "exp": np.exp}
        return eval(spec["formula"], env)
    if spec["distribution"] == "pareto":
        # shape parameter may depend on covariates, e.g. "1.75 + X"
        shape = eval(spec["shape_formula"], {"X": X})
        return stats.pareto.rvs(b=shape, scale=spec["scale"], random_state=rng)
    raise ValueError(f"Unknown outcome distribution: {spec.get('distribution')}")


def generate_dataset(cfg: dict, model: str, n: int, seed: int) -> dict:
    """Generate one dataset (containing X, U, D, Y1, Y0, Y, pi)."""
    rng = np.random.default_rng(seed)

    dg = cfg["data_generation"]
    X = rng.uniform(dg["covariate"]["low"], dg["covariate"]["high"], n)
    U = rng.uniform(dg["unobserved"]["low"], dg["unobserved"]["high"], n)

    coef = cfg["treatment"]["propensity_score"]["coefficients"]
    pi = coef["constant"] + coef["linear"] * X + coef["quadratic"] * X**2
    D = (U <= pi).astype(int)

    model_cfg = cfg["outcome_models"][model]
    shared_noise = model_cfg.get("noise")  # S in H1 is the model-level shared noise
    Y1 = _generate_potential(model_cfg["Y1"], X, rng, shared_noise)
    Y0 = _generate_potential(model_cfg["Y0"], X, rng, shared_noise)
    # shared location shift μ (added at the model level; the true QTE is translation-invariant)
    mu = float(cfg.get("design", {}).get("mu", 0.0))
    Y1 = Y1 + mu
    Y0 = Y0 + mu
    Y = np.where(D == 1, Y1, Y0)

    return {"X": X, "U": U, "D": D, "Y1": Y1, "Y0": Y0, "Y": Y, "pi": pi}


def tau_levels(cfg: dict, n: int) -> list:
    """Evaluate the quantile level τ_n formulas at a specific n, returning [(name, tau), ...]."""
    out = []
    for q in cfg["design"]["quantile_levels"]:
        tau = eval(q["formula"], {"n": n, "log": np.log})
        out.append((q.get("name", q["formula"]), tau))
    return out


def save_dataset(data: dict, path: str) -> None:
    """Save a single dataset as .npz."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)


def describe_data(data: dict, label: str = "") -> None:
    """Print the table structure of the dataset: field name, type, shape, sample values."""
    print(f"\nTable structure {label}")
    print(f"  {'field':<5}{'type':<10}{'shape':<11}sample values (first 3)")
    print(f"  {'-' * 52}")
    for key, value in data.items():
        arr = np.asarray(value)
        samples = np.array2string(arr[:3], precision=4, separator=", ")
        print(f"  {key:<5}{str(arr.dtype):<10}{str(arr.shape):<11}{samples}")


if __name__ == "__main__":
    cfg = load_config()
    seed = cfg["experiment"]["random_seed"]

    print("=" * 72)
    print("Config overview")
    print(f"  random seed:   {seed}")
    print(f"  outcome models: {list(cfg['outcome_models'])}")
    print(f"  sample sizes:   {cfg['design']['sample_sizes']}")
    print(f"  quantile formulas: {[q['formula'] for q in cfg['design']['quantile_levels']]}")
    print("=" * 72)

    first_data = True
    for model in cfg["outcome_models"]:
        for n in cfg["design"]["sample_sizes"]:
            data = generate_dataset(cfg, model, n, seed)
            if first_data:
                describe_data(data, label=f"[{model}] n={n}")
                first_data = False
            n1, n0 = int((data["D"] == 1).sum()), int((data["D"] == 0).sum())

            print(f"\n[{model}] n={n}")
            print(f"  treated/control: D=1: {n1}, D=0: {n0}  (Pi mean={data['pi'].mean():.3f})")
            for tag, y in (("Y1", data["Y1"]), ("Y0", data["Y0"]), ("Y ", data["Y"])):
                print(f"    {tag}  mean={y.mean():9.3f}  median={np.median(y):9.3f}  max={y.max():12.3f}")

            # sanity check for the extreme quantile levels τ_n (self-consistency, not a theoretical check)
            for name, tau in tau_levels(cfg, n):
                q = np.quantile(data["Y"], 1 - tau)
                n_exceed = int((data["Y"] > q).sum())
                print(f"    τ_n={tau:.2e} ({name}): theoretical exceedances={tau * n:.1f}, actual exceedances={n_exceed}")
