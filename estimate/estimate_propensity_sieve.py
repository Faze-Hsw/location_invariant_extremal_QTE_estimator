# -*- coding: utf-8 -*-
"""使用非参数筛方法估计倾向得分。

模型:  logit(pi(x)) = H_{h_n}(x)^T * pi_n
      pi(x)   = 1 / (1 + exp(-H_{h_n}(x)^T * pi_n))

其中 H_{h_n} 为一维多项式基 H(x) = (1, x, x^2, ..., x^{h_n-1})，参数由
    pi_n = argmax Σ [D_i log L(H(X_i)^T pi) + (1-D_i) log(1 - L(H(X_i)^T pi))]
估计 (L 为 sigmoid)。本脚本用 Newton-Raphson + Armijo 回溯线搜索
对**无正则**负对数似然做 MLE；对基矩阵做列标准化以改善条件数。

输入: 含 X, D, pi 字段的 dict (一维数组)
输出: 在表上新增 pi_estimate 字段（同等长度估计倾向得分）
"""
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit


def sieve_basis(X, h_n):
    """一维多项式筛基 H(x) = (1, x, x^2, ..., x^{h_n-1})，返回 (n, h_n) 设计矩阵。"""
    return np.vander(X, N=h_n, increasing=True)


def default_sieve_dim(n, r=1):
    """默认筛基维度 h_n = max(2, round(n^{1/(2r+1)})) (Hirano, Imbens, Ridder 2003)。"""
    return max(2, int(round(n ** (1 / (2 * r + 1)))))


def _newton_logistic(H, D, max_iter=200, tol=1e-8):
    """Newton-Raphson 求解无正则 logistic MLE；带 Armijo 回溯线搜索。

    优化目标: 负对数似然 NLL(theta) = -Σ [D_i eta_i - log(1+e^{eta_i})], eta = H theta
    梯度:      ∇NLL = H^T (sigma - D)
    Hessian:   ∇^2 NLL = H^T diag(sigma(1-sigma)) H
    """
    n, p = H.shape
    p0 = np.clip(D.mean(), 1e-3, 1 - 1e-3)
    theta = np.zeros(p)
    theta[0] = np.log(p0 / (1 - p0))  # 初始：常数项 logit(D 均值)，其余 0

    for it in range(max_iter):
        eta = H @ theta
        sigma = expit(eta)
        g = H.T @ (sigma - D)
        W = sigma * (1 - sigma)
        Hmat = H.T @ (W[:, None] * H)

        # Newton 步: Hmat * step = g
        try:
            step = np.linalg.solve(Hmat, g)
        except np.linalg.LinAlgError:
            break

        # Armijo 回溯线搜索
        nll_old = -np.sum(D * eta - np.logaddexp(0, eta))
        g_dot_step = g @ step
        if g_dot_step >= 0:  # 步子非下降方向（极少见），退化为最速下降
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
    """用筛方法估计倾向得分，结果写入 data['pi_estimate']。

    返回 (data, h_n, info_dict)。
    """
    X = np.asarray(data["X"]).ravel()
    D = np.asarray(data["D"]).astype(float)
    n = X.size
    if h_n is None:
        h_n = default_sieve_dim(n)

    H = sieve_basis(X, h_n).astype(float)

    # 列标准化（常数项保留为 1），改善 Newton 步的数值条件
    mu = H.mean(axis=0)
    std = H.std(axis=0)
    mu[0] = 0.0     # 常数列不平移
    std[0] = 1.0    # 常数列不缩放
    std[std < 1e-12] = 1.0
    Hn = (H - mu) / std

    theta, n_iter = _newton_logistic(Hn, D)
    pi_hat = expit(Hn @ theta)
    data["pi_estimate"] = pi_hat

    # 还原到原参数空间（用于打印系数）
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
    print(f"[测试] 模型={first_model}, n={first_n}")
    print(f"  真实 pi:     min={data['pi'].min():.3f}  max={data['pi'].max():.3f}  "
          f"均值={data['pi'].mean():.3f}")
    print(f"  D 分布:      D=1: {int((data['D']==1).sum())}, "
          f"D=0: {int((data['D']==0).sum())}")

    data, h_n, info = estimate_propensity_sieve(data)
    print(f"  筛基维度 h_n = {h_n}  (默认 round(n^(1/3)) = {default_sieve_dim(first_n)})")
    print(f"  Newton 优化: n_iter={info['n_iter']}, neg log L = {info['neg_log_lik']:.3f}")
    print(f"  截距 a0 = {info['intercept']:.4f}; "
          f"前 3 个筛基系数 = {np.array2string(info['theta_orig'][1:4], precision=4)}")

    pi = data["pi"]
    pe = data["pi_estimate"]
    err = pe - pi
    print(f"  估计 pi_est: min={pe.min():.3f}  max={pe.max():.3f}  均值={pe.mean():.3f}")
    print(f"  误差:        MAE={np.abs(err).mean():.4f}  "
          f"RMSE={np.sqrt(np.mean(err**2)):.4f}  "
          f"corr={np.corrcoef(pi, pe)[0, 1]:.4f}")

    print("\n更新后表字段结构:")
    describe_data(data, label="after sieve estimation")
