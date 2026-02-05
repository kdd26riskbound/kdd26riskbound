from __future__ import annotations

from riskbound.utils import normalize_weights
from typing import Tuple

import torch
import torch.nn as nn

class ALSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.u_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        u = torch.tanh(self.W_a(out))
        scores = self.u_a(u).squeeze(-1)
        alpha = torch.softmax(scores, dim=1)
        ctx = (alpha.unsqueeze(-1) * out).sum(dim=1)
        return ctx

class ActorNet(nn.Module):
    def __init__(self, *, num_assets: int, hidden_dim: int, asset_feat_dim: int = 5, market_feat_dim: int = 4):
        super().__init__()
        self.eps = 1e-8
        self.num_assets = num_assets

        self.asset_feat_dim = asset_feat_dim
        self.market_feat_dim = market_feat_dim

        self.asset_encoder = ALSTMEncoder(self.asset_feat_dim, hidden_dim)
        self.market_encoder = ALSTMEncoder(self.market_feat_dim, hidden_dim // 2)

        self.market_to_asset = nn.Linear(hidden_dim // 2, hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, X_s: torch.Tensor, X_m: torch.Tensor, a_prev: torch.Tensor) -> torch.Tensor:
        B, N, L, _ = X_s.shape

        a_prev_n = normalize_weights(a_prev, eps=self.eps)

        x_asset = X_s.reshape(B * N, L, self.asset_feat_dim)
        asset_emb = self.asset_encoder(x_asset).reshape(B, N, -1)

        market_emb = self.market_encoder(X_m)
        market_emb = self.market_to_asset(market_emb)
        market_exp = market_emb.unsqueeze(1).expand(-1, N, -1)

        a_prev_i = a_prev_n.unsqueeze(-1)
        feat_i = torch.cat([asset_emb, market_exp, a_prev_i], dim=-1)

        logits = self.head(feat_i).squeeze(-1)
        return logits

class CriticNet(nn.Module):
    def __init__(self, *, num_assets: int, hidden_dim: int, asset_feat_dim: int = 5, market_feat_dim: int = 4):
        super().__init__()
        self.eps = 1e-8
        self.num_assets = num_assets

        self.asset_feat_dim = asset_feat_dim
        self.market_feat_dim = market_feat_dim

        self.asset_encoder = ALSTMEncoder(self.asset_feat_dim, hidden_dim)
        self.market_encoder = ALSTMEncoder(self.market_feat_dim, hidden_dim // 2)
        self.market_to_asset = nn.Linear(hidden_dim // 2, hidden_dim)

        self.phi = nn.Sequential(
            nn.Linear(2 * hidden_dim + 3, hidden_dim // 2),
            nn.ReLU(),
        )

        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, X_s: torch.Tensor, X_m: torch.Tensor, a_prev: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        B, N, L, _ = X_s.shape
        a_prev_n = normalize_weights(a_prev, eps=self.eps)

        x_asset = X_s.reshape(B * N, L, self.asset_feat_dim)
        asset_emb = self.asset_encoder(x_asset).reshape(B, N, -1)

        market_emb = self.market_to_asset(self.market_encoder(X_m))
        market_exp = market_emb.unsqueeze(1).expand(-1, N, -1)

        a_i = action.unsqueeze(-1)
        a_prev_i = a_prev_n.unsqueeze(-1)
        d_i = (action - a_prev_n).abs().unsqueeze(-1)

        feat_i = torch.cat([asset_emb, market_exp, a_prev_i, a_i, d_i], dim=-1)

        z_i = self.phi(feat_i)
        z = z_i.mean(dim=1)
        q = self.q_head(z)
        return q

def build_actor_critic(
        action_dim: int,
        hidden_dim: int = 128,
) -> Tuple[nn.Module, nn.Module]:
    N = action_dim

    actor = ActorNet(num_assets=N, hidden_dim=hidden_dim)
    critic = CriticNet(num_assets=N, hidden_dim=hidden_dim)
    return actor, critic