from __future__ import annotations
import random

from typing import Dict, Any, Union

import torch

import numpy as np
from riskbound.data.market_env import MarketEnv
from riskbound.metrics import calculate_pf_metrics
from riskbound.core.adapter import ActionAdapter

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def normalize_weights(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if w.ndim != 2:
        raise ValueError(f"a_prev must be (B,N), got {tuple(w.shape)}")
    w = w.clamp_min(0.0)
    s = w.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(w, 1.0 / w.shape[-1])
    w = torch.where(s > 0, w / (s + eps), uniform)
    return w

def to_tensor(x: np.ndarray, *, device: Union[str, torch.device], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x).copy(), device=device, dtype=dtype)

@torch.no_grad()
def evaluate_agent(
    eval_env: MarketEnv,
    actor: torch.nn.Module,
    adapter: ActionAdapter,
    *,
    device: str,
) -> Dict[str, Any]:
    actor.eval()

    obs, info = eval_env.reset()
    done = False

    total_reward = 0.0
    steps = 0

    while not done:
        steps += 1
        asset = to_tensor(obs["asset_window"], device=device).unsqueeze(0)   
        market = to_tensor(obs["market_window"], device=device).unsqueeze(0) 
        wprev = to_tensor(obs["a_prev"], device=device).unsqueeze(0)

        a_min, a_max = adapter.get_bounds(asset, market)

        a_out = actor(asset, market, wprev)
        a_raw = torch.softmax(a_out, dim=-1)

        action = adapter.project(a_raw, a_min, a_max)[0]

        action = action.cpu().numpy().astype(np.float32)

        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = bool(terminated or truncated)

        total_reward += float(reward)

    policy_metric = calculate_pf_metrics(eval_env.returns)

    actor.train()
    return {
        "pv": float(info["pv"]),
        "steps": int(steps),
        "reward": float(total_reward),
        "metric": policy_metric,
    }

def report_pf_metrics(market: str, method: str, metrics: Dict[str, float], header=True) -> None:
    if header:
        print("Market:", market.upper())
        print(f"{'Method':<20} | {'ARR':<8} | {'ASR':<8} | {'SoR':<8} | {'CR':<8}")
        print("-" * 60)
    print(f"{method:<20} | {metrics['ARR']:.3f}   | {metrics['ASR']:.3f}   | {metrics['SoR']:.3f}     | {metrics['CR']:.3f}")


def calculate_rollvol(Xs: torch.Tensor, Xm: torch.Tensor) -> torch.Tensor:
    r = Xs[..., 4]
    return r.std(dim=-1, unbiased=False).clamp_min(1e-8)


def calculate_rolldownvol(Xs: torch.Tensor, Xm: torch.Tensor) -> torch.Tensor:
    r = Xs[..., 4]
    neg = torch.clamp(r, max=0.0)
    return torch.sqrt((neg * neg).mean(dim=-1) + 1e-8)


def calculate_rollmdd(Xs: torch.Tensor, Xm: torch.Tensor) -> torch.Tensor:
    r = Xs[..., 4]
    v = torch.cumprod(1.0 + r, dim=-1)
    peak = torch.cummax(v, dim=-1).values
    dd = 1.0 - (v / (peak + 1e-8))
    mdd = dd.max(dim=-1).values
    return mdd.clamp(min=0.0, max=1.0)


def calculate_garch(
    Xs: torch.Tensor,
    Xm: torch.Tensor,
    omega: float | str = "auto",
    alpha: float = 0.05,
    beta: float = 0.90,
    init: str = "sample",
    use_mean_sigma: bool = False,
    eps: float = 1e-8,
):
    r = Xs[..., 4]
    r2 = r.pow(2)
    B, N, W = r.shape
    device, dtype = r.device, r.dtype

    if init == "sample":
        sigma2 = r.var(dim=2, unbiased=False).clamp_min(eps)
    elif init == "uncond":
        denom = max(1.0 - alpha - beta, eps)

        sigma2 = torch.full((B, N), 1.0, device=device, dtype=dtype)
    else:
        raise ValueError("init must be 'sample' or 'uncond'")

    if omega == "auto":
        target_var = r.var(dim=2, unbiased=False).clamp_min(eps)
        omega_t = (1.0 - alpha - beta) * target_var
        omega_t = omega_t.clamp_min(eps)
    else:
        omega_t = torch.full((B, N), float(omega), device=device, dtype=dtype)

    if init == "uncond":
        denom = max(1.0 - alpha - beta, eps)
        sigma2 = (omega_t / denom).clamp_min(eps)

    if use_mean_sigma:
        sigma_hist = torch.empty((B, N, W), device=device, dtype=dtype)

    for t in range(W):
        sigma2 = omega_t + alpha * r2[..., t] + beta * sigma2
        if use_mean_sigma:
            sigma_hist[..., t] = torch.sqrt(sigma2.clamp_min(eps))

    if use_mean_sigma:
        return sigma_hist.mean(dim=2).clamp_min(eps)
    else:
        return torch.sqrt(sigma2.clamp_min(eps))
