from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from riskbound.settings import LOOKBACK_WINDOW, GLOBAL_SEED
from riskbound.data.dataset import load_data
from riskbound.data.market_env import MarketEnv
from riskbound.data.source import DataSource
from riskbound.metrics import calculate_pf_metrics


def enforce_simplex(v: np.ndarray, z: float = 1.0) -> np.ndarray:
    if z <= 0.0:
        raise ValueError("Simplex sum z must be > 0")
    v = v.astype(np.float64, copy=False)
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    rho = -1
    for j in range(n):
        t = u[j] - cssv[j] / (j + 1)
        if t > 0:
            rho = j
    if rho == -1:
        return np.full((n,), z / n, dtype=np.float64)
    theta = cssv[rho] / (rho + 1)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return np.full((n,), z / n, dtype=np.float64)
    w *= (z / s)
    return w

def estimate_lipschitz(Sigma: np.ndarray, lam: float, iters: int = 30, eps: float = 1e-12) -> float:
    n = Sigma.shape[0]
    if n == 1:
        return float(lam * abs(Sigma[0, 0]) + eps)
    x = np.random.randn(n).astype(np.float64)
    x /= (np.linalg.norm(x) + eps)
    for _ in range(iters):
        x = Sigma @ x
        x_norm = np.linalg.norm(x)
        if x_norm <= eps:
            return float(lam * eps)
        x /= x_norm
    sx = Sigma @ x
    norm2 = float(np.dot(x, sx))
    norm2 = max(norm2, eps)
    return float(lam * norm2 + eps)

class BaselinePolicy:
    def __init__(self, n_assets: int):
        self.n = int(n_assets)
        self.L = LOOKBACK_WINDOW
        self.eps: float = 1e-8
        self.mv_ridge: float = 1e-3
        self.mv_clip: float = 0.0
        self.mv_use_shrinkage: bool = True
        self.mv_shrink_alpha: float = 0.9
        self.mv_risk_aversion: float = 5.0
        self.mv_max_iter: int = 1000
        self.mv_tol: float = 1e-10
        self.mv_step_size: Optional[float] = None
        self.rp_ridge: float = 1e-6
        self.rp_use_shrinkage: bool = True
        self.rp_shrink_alpha: float = 0.5
        self.rp_max_iter: int = 1000
        self.rp_tol: float = 1e-8
        self.rp_step: float = 0.05

    def reset(self, obs: Dict) -> None:
        return

    def act(self, obs: Dict) -> np.ndarray:
        raise NotImplementedError


class InverseVolPolicy(BaselinePolicy):
    def act(self, obs: Dict) -> np.ndarray:
        X = np.asarray(obs["asset_window"], dtype=np.float64)
        r = X[:, :, 4]
        W = int(self.L)
        W = max(2, min(W, r.shape[1]))
        rW = r[:, -W:]
        sigma = rW.std(axis=1, ddof=0)
        inv = 1.0 / (sigma + self.eps)
        inv = np.clip(inv, self.eps, np.inf)
        s = float(inv.sum())
        if not np.isfinite(s) or s <= 0.0:
            return np.full((self.n,), 1.0 / self.n, dtype=np.float32)
        w = (inv / s).astype(np.float32)
        w = w / float(w.sum())
        return w


class RiskParityPolicy(BaselinePolicy):
    def act(self, obs: Dict) -> np.ndarray:
        X = np.asarray(obs["asset_window"], dtype=np.float64)
        r = X[:, :, 4]
        W = int(self.L)
        W = max(3, min(W, r.shape[1]))
        rW = r[:, -W:]
        Sigma = np.cov(rW, bias=True)
        if Sigma.ndim == 0:
            Sigma = np.array([[float(Sigma)]], dtype=np.float64)
        if self.rp_use_shrinkage and self.n > 1:
            diag = np.diag(np.diag(Sigma))
            a = float(self.rp_shrink_alpha)
            a = float(np.clip(a, 0.0, 1.0))
            Sigma = (1.0 - a) * Sigma + a * diag
        Sigma = Sigma + float(self.rp_ridge) * np.eye(self.n, dtype=np.float64)
        if not np.isfinite(Sigma).all():
            return np.full((self.n,), 1.0 / self.n, dtype=np.float32)
        var = np.diag(Sigma).copy()
        var = np.clip(var, self.eps, np.inf)
        w = 1.0 / np.sqrt(var)
        w = np.clip(w, self.eps, np.inf)
        w = w / float(w.sum())
        b = np.full((self.n,), 1.0 / self.n, dtype=np.float64)
        step = float(self.rp_step)
        tol = float(self.rp_tol)
        for _ in range(int(self.rp_max_iter)):
            Sw = Sigma @ w
            port_var = float(w @ Sw)
            if not np.isfinite(port_var) or port_var <= 0.0:
                break
            RC = w * Sw
            target = b * port_var
            rel = (RC - target) / (target + self.eps)
            max_err = float(np.max(np.abs(rel)))
            if max_err < tol:
                break
            w = w * np.exp(-step * rel)
            w = np.clip(w, self.eps, np.inf)
            s = float(w.sum())
            if not np.isfinite(s) or s <= 0.0:
                w = np.full((self.n,), 1.0 / self.n, dtype=np.float64)
                break
            w = w / s
        w = w.astype(np.float32)
        w = np.clip(w, 0.0, 1.0)
        w = w / float(w.sum())
        return w

class MeanVariancePolicy(BaselinePolicy):
    def act(self, obs: Dict) -> np.ndarray:
        X = np.asarray(obs["asset_window"], dtype=np.float64)
        r = X[:, :, 4]
        W = int(self.L)
        W = max(3, min(W, r.shape[1]))
        rW = r[:, -W:]
        n = rW.shape[0]
        if n == 1:
            return np.array([1.0], dtype=np.float32)
        mu = rW.mean(axis=1)
        Sigma = np.cov(rW, bias=True)
        if Sigma.ndim == 0:
            Sigma = np.array([[float(Sigma)]], dtype=np.float64)
        if self.mv_use_shrinkage and n > 1:
            alpha = float(self.mv_shrink_alpha)
            alpha = min(max(alpha, 0.0), 1.0)
            diag = np.diag(np.diag(Sigma))
            Sigma = (1.0 - alpha) * Sigma + alpha * diag
        Sigma = Sigma + float(self.mv_ridge) * np.eye(n, dtype=np.float64)
        lam = float(self.mv_risk_aversion)
        if not np.isfinite(lam) or lam <= 0.0:
            return np.full((n,), 1.0 / n, dtype=np.float32)
        w = np.full((n,), 1.0 / n, dtype=np.float64)
        if self.mv_step_size is None:
            Lg = estimate_lipschitz(Sigma, lam, iters=30, eps=self.eps)
            step = 1.0 / Lg
        else:
            step = float(self.mv_step_size)
        step = max(step, 1e-8)
        prev_obj = -np.inf
        for _ in range(int(self.mv_max_iter)):
            grad = mu - lam * (Sigma @ w)
            a_new = enforce_simplex(w + step * grad, z=1.0)
            obj = float(mu @ a_new - 0.5 * lam * (a_new @ (Sigma @ a_new)))
            if not np.isfinite(obj):
                break
            if abs(obj - prev_obj) <= float(self.mv_tol):
                w = a_new
                break
            if obj < prev_obj:
                step *= 0.5
                if step < 1e-8:
                    w = a_new
                    break
                continue
            w = a_new
            prev_obj = obj
        w = np.maximum(w, 0.0)
        s = float(w.sum())
        if not np.isfinite(s) or s <= 0.0:
            w = np.full((n,), 1.0 / n, dtype=np.float64)
        else:
            w = w / s
        return w.astype(np.float32)

def make_policy(name, *, n_assets: int) -> BaselinePolicy:
    if name == "inverse_vol":
        return InverseVolPolicy(n_assets)
    if name == "risk_parity":
        return RiskParityPolicy(n_assets)
    if name == "mean_variance":
        return MeanVariancePolicy(n_assets)
    raise ValueError(f"Unknown baseline: {name}")

def run_baseline(
    *,
    market: str,
    baseline: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    dates, market_features, symbols, asset_features, ror_array = load_data(market=market)

    src = DataSource(
        asset_features=asset_features,
        market_features=market_features,
        rate_of_returns=ror_array,
        dates=dates,
    ).set_mode("test")

    env = MarketEnv(data_source=src)

    policy = make_policy(baseline, n_assets=src.N)

    obs, info = env.reset(seed=GLOBAL_SEED)

    done = False
    steps = 0
    ep_reward = 0.0
    log_ret_sum = 0.0

    policy.reset(obs)

    while not done:
        steps += 1

        action = policy.act(obs)

        action = np.clip(action.astype(np.float32), 0.0, 1.0)
        s = float(action.sum())
        if not np.isfinite(s) or s <= 0.0:
            action = np.full((src.N,), 1.0 / src.N, dtype=np.float32)
        else:
            action = action / s

        obs, reward, terminated, truncated, info_next = env.step(action)
        done = bool(terminated or truncated)

        ep_reward += float(reward)
        log_ret_sum += float(info_next["log_return"])

    metrics = calculate_pf_metrics(np.asarray(env.returns, dtype=np.float64))
    extra = {
        "steps": float(steps),
        "episode_reward": float(ep_reward),
        "cumulative_log_return": float(log_ret_sum),
        "final_pv": float(env.portfolio_value),
    }
    return metrics, extra
