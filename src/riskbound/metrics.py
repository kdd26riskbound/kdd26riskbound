from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

def calculate_pf_metrics(
    returns: np.ndarray,
    *,
    periods_per_year: int = 12,
    eps: float = 1e-12,
    vol_floor: float = 1e-6,
    zero_ret_tol: float = 1e-6,
    dd_floor: float = 1e-6,
    down_floor: float = 1e-6,
):
    r = np.asarray(returns, dtype=np.float64)
    T = len(r)
    assert T > 1, "Need at least 2 periods"
    if np.all(np.abs(r) < zero_ret_tol):
        return {
            "ARR": 0.0,
            "ASR": 0.0,
            "SoR": 0.0,
            "CR": 0.0,
        }
    ac = np.cumprod(1.0 + r)

    mean_r = float(r.mean())
    ARR = mean_r * float(periods_per_year)

    vol = float(np.sqrt(np.mean((r - mean_r) ** 2)))
    AVol = vol * float(np.sqrt(periods_per_year))

    downside = np.minimum(r, 0.0)
    downside_mean = float(downside.mean())
    downside_var = float(np.mean((downside - downside_mean) ** 2))
    downside_std = float(np.sqrt(downside_var)) * float(np.sqrt(periods_per_year))

    peak = np.maximum.accumulate(ac)
    drawdown = (peak - ac) / (peak + eps)
    MDD = float(drawdown.max())

    if AVol < vol_floor:
        ASR = 0.0
    else:
        ASR = ARR / (AVol + eps)

    if downside_std < down_floor:
        SoR = 0.0
    else:
        SoR = ARR / (downside_std + eps)

    if MDD < dd_floor:
        CR = 0.0
    else:
        CR = ARR / (MDD + eps)

    return {
        "ARR": ARR,
        "ASR": ASR,
        "SoR": SoR,
        "CR": CR,
    }

def pearson_series(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if not torch.isfinite(pred).all():
        raise RuntimeError("pred has NaN/Inf in evaluation")
    x = pred - pred.mean(dim=1, keepdim=True)
    y = true - true.mean(dim=1, keepdim=True)
    num = (x * y).sum(dim=1)
    den = torch.sqrt((x * x).sum(dim=1) * (y * y).sum(dim=1) + eps)
    return num / den.clamp_min(eps)

def rankic_series(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if not torch.isfinite(pred).all():
        raise RuntimeError("pred has NaN/Inf in evaluation")
    rp = torch.argsort(torch.argsort(pred, dim=1), dim=1).float()
    rt = torch.argsort(torch.argsort(true, dim=1), dim=1).float()
    return pearson_series(rp, rt, eps=eps)

def mean_ir(x: torch.Tensor, eps: float = 1e-8) -> Tuple[float, float]:
    mu = x.mean().item()
    sd = x.std(unbiased=False).item()
    return mu, mu / (sd + eps)