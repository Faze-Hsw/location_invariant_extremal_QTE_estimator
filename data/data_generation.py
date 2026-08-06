# -*- coding: utf-8 -*-
"""根据 configs/data_generation.yaml 生成模拟实验数据。

覆盖三种结果模型 H1/H2/H3、多种样本量与极端分位数水平 τ_n。
用法:
    python data/data_generation.py
"""
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "data_generation.yaml"


def load_config(path: str = CONFIG_PATH) -> dict:
    """读取数据生成配置文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sample_noise(spec: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """按噪声分布生成 (n,) 数组。"""
    dist = spec["distribution"]
    if dist == "student_t":
        return stats.t.rvs(df=spec["df"], size=n, random_state=rng)
    if dist == "frechet":
        # Fréchet(shape, loc, scale) 对应 scipy.stats.invweibull(c=shape, loc, scale)
        return stats.invweibull.rvs(
            c=spec["shape"], loc=spec["loc"], scale=spec["scale"],
            size=n, random_state=rng,
        )
    raise ValueError(f"未知噪声分布: {dist}")


def _generate_potential(
    spec: dict, X: np.ndarray, rng: np.random.Generator, shared_noise: dict = None
) -> np.ndarray:
    """生成单个潜在结果 (Y1 或 Y0)。

    支持三种配置结构:
      - spec 自带 noise  (H2: Y1/Y0 各自独立噪声)
      - 使用模型级共享 noise (H1: Y1/Y0 共用同一 S)
      - 直接指定分布   (H3: Pareto，形状依赖协变量)
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
        # 公式字符串（如 "5 * S * (1 + X)" 或 "C2 * exp(X)"），注入噪声变量名后求值
        env = {"X": X, var: noise, "np": np, "exp": np.exp}
        return eval(spec["formula"], env)
    if spec["distribution"] == "pareto":
        # 形状参数可依赖协变量，如 "1.75 + X"
        shape = eval(spec["shape_formula"], {"X": X})
        return stats.pareto.rvs(b=shape, scale=spec["scale"], random_state=rng)
    raise ValueError(f"未知结果分布: {spec.get('distribution')}")


def generate_dataset(cfg: dict, model: str, n: int, seed: int) -> dict:
    """生成一份数据集（含 X, U, D, Y1, Y0, Y, pi）。"""
    rng = np.random.default_rng(seed)

    dg = cfg["data_generation"]
    X = rng.uniform(dg["covariate"]["low"], dg["covariate"]["high"], n)
    U = rng.uniform(dg["unobserved"]["low"], dg["unobserved"]["high"], n)

    coef = cfg["treatment"]["propensity_score"]["coefficients"]
    pi = coef["constant"] + coef["linear"] * X + coef["quadratic"] * X**2
    D = (U <= pi).astype(int)

    model_cfg = cfg["outcome_models"][model]
    shared_noise = model_cfg.get("noise")  # H1 的 S 为模型级共享噪声
    Y1 = _generate_potential(model_cfg["Y1"], X, rng, shared_noise)
    Y0 = _generate_potential(model_cfg["Y0"], X, rng, shared_noise)
    # 共享位置偏移 μ（模型层外加；真实 QTE 平移不变）
    mu = float(cfg.get("design", {}).get("mu", 0.0))
    Y1 = Y1 + mu
    Y0 = Y0 + mu
    Y = np.where(D == 1, Y1, Y0)

    return {"X": X, "U": U, "D": D, "Y1": Y1, "Y0": Y0, "Y": Y, "pi": pi}


def tau_levels(cfg: dict, n: int) -> list:
    """把分位数水平 τ_n 公式代入具体 n 求值，返回 [(name, tau), ...]。"""
    out = []
    for q in cfg["design"]["quantile_levels"]:
        tau = eval(q["formula"], {"n": n, "log": np.log})
        out.append((q.get("name", q["formula"]), tau))
    return out


def save_dataset(data: dict, path: str) -> None:
    """将单份数据集保存为 .npz。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)


def describe_data(data: dict, label: str = "") -> None:
    """打印数据集的表结构：字段名、类型、形状、示例值。"""
    print(f"\n表字段结构 {label}")
    print(f"  {'字段':<5}{'类型':<10}{'形状':<11}示例值(前3)")
    print(f"  {'-' * 52}")
    for key, value in data.items():
        arr = np.asarray(value)
        samples = np.array2string(arr[:3], precision=4, separator=", ")
        print(f"  {key:<5}{str(arr.dtype):<10}{str(arr.shape):<11}{samples}")


if __name__ == "__main__":
    cfg = load_config()
    seed = cfg["experiment"]["random_seed"]

    print("=" * 72)
    print("配置概览")
    print(f"  随机种子:   {seed}")
    print(f"  结果模型:   {list(cfg['outcome_models'])}")
    print(f"  样本量:     {cfg['design']['sample_sizes']}")
    print(f"  分位数公式: {[q['formula'] for q in cfg['design']['quantile_levels']]}")
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
            print(f"  处理/对照: D=1: {n1}, D=0: {n0}  (Pi均值={data['pi'].mean():.3f})")
            for tag, y in (("Y1", data["Y1"]), ("Y0", data["Y0"]), ("Y ", data["Y"])):
                print(f"    {tag}  均值={y.mean():9.3f}  中位数={np.median(y):9.3f}  最大值={y.max():12.3f}")

            # 极端分位数水平 τ_n 的 sanity check（自洽性，非理论校验）
            for name, tau in tau_levels(cfg, n):
                q = np.quantile(data["Y"], 1 - tau)
                n_exceed = int((data["Y"] > q).sum())
                print(f"    τ_n={tau:.2e} ({name}): 理论超出数={tau * n:.1f}, 实际超出数={n_exceed}")
