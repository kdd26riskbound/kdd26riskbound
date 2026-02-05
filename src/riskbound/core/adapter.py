from __future__ import annotations

from riskbound.core.rabg import RiskAwareBoundaryGenerator
from riskbound.core.fpsg import FPSG
from typing import Optional, Tuple
import torch

class ActionAdapter:
    def __init__(
        self,
        *,
        action_dim: int,
        device: str,
        rabg: RiskAwareBoundaryGenerator,
        fpsg_leak=0.1,
        fpsg_beta=10.0,
    ):
        self.action_dim = action_dim
        self.device = device
        self.rabg = rabg
        self.leak = fpsg_leak
        self.beta = fpsg_beta

    def get_bounds(self, asset_t: torch.Tensor, market_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.rabg.generate(asset_t, market_t)

    def project(self, a_raw: torch.Tensor, a_min: torch.Tensor, a_max: torch.Tensor) -> torch.Tensor:
        a_min = a_min.detach()
        a_max = a_max.detach()
        return FPSG.apply(
            a_raw,
            a_min,
            a_max,
            self.beta,
            self.leak,
        )