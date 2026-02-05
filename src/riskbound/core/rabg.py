from __future__ import annotations

from riskbound.settings import LOOKBACK_WINDOW
from torch.nn.utils.parametrizations import weight_norm
from typing import Tuple
from pathlib import Path
from typing import List

import json
import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        Conv = nn.Conv1d
        Drop = nn.Dropout

        self.conv1 = weight_norm(
            Conv(in_ch, out_ch, kernel_size, stride=1, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = Drop(dropout)

        self.conv2 = weight_norm(
            Conv(out_ch, out_ch, kernel_size, stride=1, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = Drop(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.drop1,
            self.conv2, self.chomp2, self.relu2, self.drop2,
        )

        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.conv1, self.conv2]:
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, mean=0.0, std=0.01)
            if self.downsample.bias is not None:
                nn.init.zeros_(self.downsample.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(
        self,
        in_ch: int,
        channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
        dilation: List[int] | None = None
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_ch
        if dilation is None:
            dilation = [2 ** i for i in range(len(channels))]
        assert len(dilation) == len(channels)

        for out_ch, d in zip(channels, dilation):
            layers.append(TemporalBlock(prev, out_ch, kernel_size, d, dropout))
            prev = out_ch
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class RiskScorer(nn.Module):
    def __init__(self, asset_in, market_in, hidden_dim=128, tcn_levels=3, kernel_size=3):
        super().__init__()

        self.asset_mlp = nn.Sequential(
            nn.Linear(asset_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.tcn = TemporalConvNet(
            in_ch=hidden_dim,
            channels=[hidden_dim] * tcn_levels,
            dropout=0.1,
            kernel_size=kernel_size,
        )

        self.asset_ln = nn.LayerNorm(hidden_dim)

        self.market_mlp = nn.Sequential(
            nn.Linear(market_in * LOOKBACK_WINDOW, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )

    def forward(self, X_s: torch.Tensor, X_m: torch.Tensor) -> torch.Tensor:
        B, N, L, F_a = X_s.shape

        x_s_flat = X_s.reshape(B * N, L, F_a)

        emb_s = self.asset_mlp(x_s_flat)

        tcn_in = emb_s.transpose(1, 2)
        tcn_out = self.tcn(tcn_in)

        res_out = tcn_out.transpose(1, 2) + emb_s

        asset_feat = res_out.mean(dim=1).reshape(B, N, -1)

        x_m_flat = X_m.reshape(B, -1)
        market_feat = self.market_mlp(x_m_flat)
        market_feat = market_feat.unsqueeze(1).expand(-1, N, -1)

        combined = torch.cat([asset_feat, market_feat], dim=-1)

        risk_score = self.head(combined).squeeze(-1)

        return risk_score

    def save_model(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        json_path = path.with_suffix(".json")
        with json_path.open("w", encoding="utf-8") as f:
            json.dump({
                "asset_in": self.asset_mlp[0].in_features,
                "market_in": self.market_mlp[0].in_features // LOOKBACK_WINDOW,
                "hidden_dim": self.asset_mlp[0].out_features,
                "tcn_levels": len(self.tcn.network),
                "kernel_size": self.tcn.network[0].conv1.kernel_size[0],
            }, f, indent=4, ensure_ascii=False)

    @staticmethod
    def load_model(path: str | Path, device: str) -> RiskScorer:
        path = Path(path)
        json_path = path.with_suffix(".json")

        model = RiskScorer(**json.load(json_path.open("r", encoding="utf-8")))
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

class InverseRiskBoundary:
    def __init__(self, global_cap: float = 1.5, eps=1e-8):
        self.global_cap = global_cap
        self.eps = eps

    def generate(self, s_raw: torch.Tensor) -> torch.Tensor:
        eps = float(self.eps)
        if s_raw.dim() != 2:
            raise ValueError(f"s_raw must be (B,N), got {tuple(s_raw.shape)}")

        s = s_raw.clamp_min(0.0)
        u = 1.0 / (s + eps)

        u_sum = u.sum(dim=1, keepdim=True).clamp_min(eps)
        a_inv = u / u_sum

        a_max = float(self.global_cap) * a_inv

        a_max = a_max.clamp_max(1.0)

        return a_max

class RiskAwareBoundaryGenerator:
    def __init__(self, global_cap: float=1.5):
        self.scorer = None
        self.boundary = InverseRiskBoundary(global_cap=global_cap)
        self.a_min = 0.0

    def generate(self, Xs: torch.Tensor, Xm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.scorer is None:
            raise RuntimeError("RiskScorer model is not loaded.")

        with torch.no_grad():
            s_raw = self.scorer(Xs, Xm)

        a_max_all = self.boundary.generate(s_raw)
        a_min_all = torch.full_like(a_max_all, float(self.a_min))
        return a_min_all, a_max_all

    def load_model(self, path: str | Path, device: str):
        self.scorer = RiskScorer.load_model(path, device)