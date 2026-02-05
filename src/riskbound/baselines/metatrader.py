from __future__ import annotations

import copy
import math
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from riskbound.settings import (GLOBAL_SEED, EPISODES, HIDDEN_DIM, BATCH_SIZE,
                                ACTOR_LR, CRITIC_LR, BUFFER_SIZE, TAU, GAMMA)
from riskbound.data.market_env import MarketEnv
from riskbound.rl.noise import OUNoise
from riskbound.metrics import calculate_pf_metrics
from riskbound.utils import to_tensor

"""
Implementation of MetaTrader

Reference:
Hui Niu, Siyuan Li, and Jian Li. 2022. MetaTrader: An Reinforcement Learning Approach Integrating Diverse Policies for Portfolio Optimization. 
In Proceedings of the 31st ACM International Conference on Information & Knowledge Management (CIKM '22). 
Association for Computing Machinery, New York, NY, USA, 1573–1583. https://doi.org/10.1145/3511808.3557363

Note:
This is a from-scratch reimplementation based on the paper (not affiliated with the authors).
"""

DSR_ETA = 0.01
DSR_CLIP = 10.0
EPS = 1e-12

def normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(w)):
        return np.full_like(w, 1.0 / w.shape[0], dtype=np.float32)
    w = np.maximum(w, 0.0)
    s = float(w.sum())
    if s <= 0.0:
        return np.full_like(w, 1.0 / w.shape[0], dtype=np.float32)
    w = w / s
    w[-1] = 1.0 - float(w[:-1].sum())
    if w[-1] < 0.0:
        w = np.maximum(w, 0.0)
        w = w / (float(w.sum()) + EPS)

    return w.astype(np.float32, copy=False)


class DifferentialSharpeReward:
    def __init__(self, eta: float = DSR_ETA, eps: float = 1e-12, clip: Optional[float] = DSR_CLIP):
        self.eta = float(eta)
        self.eps = float(eps)
        self.clip = None if clip is None else float(clip)
        self.reset()

    def reset(self) -> None:
        self.alpha = 0.0
        self.beta = 0.0

    def update(self, r_t: float) -> float:
        r = float(r_t)
        alpha_prev = self.alpha
        beta_prev = self.beta
        eta = self.eta

        self.alpha = alpha_prev + eta * (r - alpha_prev)
        self.beta = beta_prev + eta * ((r * r) - beta_prev)

        var = beta_prev - alpha_prev * alpha_prev
        if var < self.eps:
            return 0.0

        delta_alpha = eta * (r - alpha_prev)
        delta_beta = eta * ((r * r) - beta_prev)
        numerator = beta_prev * delta_alpha - 0.5 * alpha_prev * delta_beta
        denom = math.pow(max(var, self.eps), 1.5)
        dsr = numerator / denom

        if self.clip is not None:
            dsr = float(np.clip(dsr, -self.clip, self.clip))
        return float(dsr)

def get_topk(score: np.ndarray, k: int) -> np.ndarray:
    k = int(k)
    if k <= 0:
        return np.array([], dtype=np.int64)
    k = min(k, score.shape[0])
    return np.argsort(score, kind="stable")[-k:]

class ExpertGenerator:
    def __init__(self, n_assets: int, *, topk: int):
        self.n_assets = int(n_assets)
        self.topk = int(topk)

    def allocate_topk(self, score: np.ndarray) -> np.ndarray:
        w = np.zeros((self.n_assets,), dtype=np.float32)
        idx = get_topk(score, self.topk)
        if idx.size == 0:
            w[:] = 1.0 / self.n_assets
        else:
            w[idx] = 1.0 / float(idx.size)
        return w

    def csm_momentum(self, asset_window: np.ndarray, *, close_rel_idx: int = 3) -> np.ndarray:
        close_rel = asset_window[..., close_rel_idx]
        gross = np.prod(np.clip(close_rel, 1e-6, None), axis=1)
        score = gross - 1.0
        return self.allocate_topk(score)

    def blsw_lowvol(self, asset_window: np.ndarray, *, close_rel_idx: int = 3) -> np.ndarray:
        close_rel = asset_window[..., close_rel_idx]
        r = close_rel - 1.0
        vol = r.std(axis=1)
        score = -vol
        return self.allocate_topk(score)

    def empty_uniform(self) -> np.ndarray:
        return np.full((self.n_assets,), 1.0 / self.n_assets, dtype=np.float32)

    def hindsight(self, y_t: np.ndarray) -> np.ndarray:
        score = np.asarray(y_t, dtype=np.float32).reshape(-1)
        return self.allocate_topk(score)

class SubBuffer:
    def __init__(self, capacity: int, device: torch.device, seed: int):
        self.capacity = int(capacity)
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.ptr = 0
        self.size = 0
        self._allocated = False

    def __len__(self) -> int:
        return self.size

    def _alloc(self, asset_window: np.ndarray, market_window: np.ndarray) -> None:
        N, L, Fs = asset_window.shape
        Lm, Fm = market_window.shape
        assert Lm == L

        C = self.capacity
        self.asset_w = np.empty((C, N, L, Fs), dtype=np.float32)
        self.market_w = np.empty((C, L, Fm), dtype=np.float32)
        self.a_prev = np.empty((C, N), dtype=np.float32)
        self.action = np.empty((C, N), dtype=np.float32)
        self.reward = np.empty((C, 1), dtype=np.float32)
        self.next_asset_w = np.empty((C, N, L, Fs), dtype=np.float32)
        self.next_market_w = np.empty((C, L, Fm), dtype=np.float32)
        self.next_a_prev = np.empty((C, N), dtype=np.float32)
        self.done = np.empty((C, 1), dtype=np.float32)
        self.expert_action = np.empty((C, N), dtype=np.float32)
        self._allocated = True

    def push(
            self,
            asset_window: np.ndarray,
            market_window: np.ndarray,
            a_prev: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_asset_window: np.ndarray,
            next_market_window: np.ndarray,
            next_a_prev: np.ndarray,
            done: float,
            expert_action: np.ndarray,
    ) -> None:
        if not self._allocated:
            self._alloc(asset_window, market_window)

        i = self.ptr
        self.asset_w[i] = asset_window.astype(np.float32, copy=False)
        self.market_w[i] = market_window.astype(np.float32, copy=False)
        self.a_prev[i] = np.asarray(a_prev, dtype=np.float32).reshape(-1)
        self.action[i] = np.asarray(action, dtype=np.float32).reshape(-1)
        self.reward[i, 0] = float(reward)
        self.next_asset_w[i] = next_asset_window.astype(np.float32, copy=False)
        self.next_market_w[i] = next_market_window.astype(np.float32, copy=False)
        self.next_a_prev[i] = np.asarray(next_a_prev, dtype=np.float32).reshape(-1)
        self.done[i, 0] = float(done)
        self.expert_action[i] = np.asarray(expert_action, dtype=np.float32).reshape(-1)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _to_torch(self, x: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(x)
        if self.device.type != "cpu":
            t = t.to(self.device, non_blocking=True)
        return t

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=int(batch_size))

        asset_b = self._to_torch(self.asset_w[idx])
        market_b = self._to_torch(self.market_w[idx])
        wprev_b = self._to_torch(self.a_prev[idx])
        act_b = self._to_torch(self.action[idx])
        rew_b = self._to_torch(self.reward[idx])
        next_asset_b = self._to_torch(self.next_asset_w[idx])
        next_market_b = self._to_torch(self.next_market_w[idx])
        next_wprev_b = self._to_torch(self.next_a_prev[idx])
        done_b = self._to_torch(self.done[idx])
        exp_b = self._to_torch(self.expert_action[idx])

        return asset_b, market_b, wprev_b, act_b, rew_b, next_asset_b, next_market_b, next_wprev_b, done_b, exp_b

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :-self.chomp_size]

class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1, bias: bool = True):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.left_pad = (self.kernel_size - 1) * self.dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.left_pad,
            bias=bias,
        )
        self.chomp = Chomp1d(self.left_pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.chomp(self.conv(x))

class TemporalConvBlock(nn.Module):
    def __init__(self, fin: int, fhidden: int, k: int, dilation: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = CausalConv1d(fin, fhidden, kernel_size=k, dilation=dilation)
        self.conv2 = CausalConv1d(fhidden, fhidden, kernel_size=k, dilation=dilation)
        self.proj = nn.Conv1d(fin, fhidden, kernel_size=1) if fin != fhidden else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.proj(x)
        y = self.dropout(F.relu(self.conv1(x)))
        y = self.dropout(F.relu(self.conv2(y)))
        return y + res

class TemporalConvNet(nn.Module):
    def __init__(self, fin: int, fhidden: int, k: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_ch = fin if i == 0 else fhidden
            layers.append(TemporalConvBlock(in_ch, fhidden, k=k, dilation=dilation, dropout=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SpatialAttention(nn.Module):
    def __init__(self, fhidden: int):
        super().__init__()
        self.q = nn.Linear(fhidden, fhidden)
        self.k = nn.Linear(fhidden, fhidden)
        self.v = nn.Linear(fhidden, fhidden)
        self.scale = float(math.sqrt(fhidden))

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        B, N, Fh, T = H.shape
        X = H.permute(0, 3, 1, 2).contiguous().view(B * T, N, Fh)
        Q = self.q(X)
        K = self.k(X)
        V = self.v(X)
        att = torch.matmul(Q, K.transpose(1, 2)) / (self.scale + 1e-8)
        att = torch.softmax(att, dim=-1)
        Y = torch.matmul(att, V)
        return Y.view(B, T, N, Fh).permute(0, 2, 3, 1).contiguous()


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


class MTActor(nn.Module):
    def __init__(self, n_assets: int, f_asset: int, f_market: int, hidden: int):
        super().__init__()
        self.n_assets = int(n_assets)
        self.hidden = int(hidden)

        self.tcn = TemporalConvNet(fin=int(f_asset), fhidden=self.hidden, k=3, num_layers=4, dropout=0.0)
        self.satt = SpatialAttention(fhidden=self.hidden)
        self.market_enc = ALSTMEncoder(input_dim=int(f_market), hidden_dim=self.hidden)
        self.prev_proj = nn.Linear(1, self.hidden)

        self.fuse = nn.Sequential(
            nn.Linear(3 * self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )

    def forward(self, asset_window: torch.Tensor, market_window: torch.Tensor, a_prev: torch.Tensor) -> torch.Tensor:
        B, N, L, Fs = asset_window.shape
        assert N == self.n_assets

        X = asset_window.reshape(B * N, L, Fs).transpose(1, 2).contiguous()
        H = self.tcn(X)
        H = H.view(B, N, self.hidden, L)
        H = H + self.satt(H)
        Hp = H.mean(dim=-1)

        m = self.market_enc(market_window).unsqueeze(1).expand(B, N, self.hidden)
        wp = self.prev_proj(a_prev.unsqueeze(-1))

        feat = torch.cat([Hp, m, wp], dim=-1)
        logits = self.fuse(feat).squeeze(-1)
        return logits


class MTCritic(nn.Module):
    def __init__(self, n_assets: int, f_asset: int, f_market: int, hidden: int):
        super().__init__()
        self.n_assets = int(n_assets)
        self.emb_dim = int(hidden)

        self.asset_enc = ALSTMEncoder(input_dim=int(f_asset), hidden_dim=self.emb_dim)
        self.market_enc = ALSTMEncoder(input_dim=int(f_market), hidden_dim=self.emb_dim)

        self.phi = nn.Sequential(
            nn.Linear(2 * self.emb_dim + 3, hidden),
            nn.ReLU(),
        )
        self.q_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, asset_window: torch.Tensor, market_window: torch.Tensor, a_prev: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        B, N, L, Fs = asset_window.shape
        assert N == self.n_assets

        x_asset = asset_window.reshape(B * N, L, Fs)
        asset_emb = self.asset_enc(x_asset).view(B, N, self.emb_dim)

        market_emb = self.market_enc(market_window).unsqueeze(1).expand(B, N, self.emb_dim)

        a_prev_i = a_prev.unsqueeze(-1)
        a_i = action.unsqueeze(-1)
        d_i = (action - a_prev).abs().unsqueeze(-1)

        feat_i = torch.cat([asset_emb, market_emb, a_prev_i, a_i, d_i], dim=-1)
        z_i = self.phi(feat_i)
        z = z_i.mean(dim=1)
        q = self.q_head(z)
        return q

def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)

class DDPG:
    def __init__(self, *, n_assets: int, f_asset: int, f_market: int, device: torch.device, sub_lambda: float,
                 seed: int):
        self.n_assets = int(n_assets)
        self.device = device
        self.sub_lambda = float(sub_lambda)

        self.actor = MTActor(n_assets=n_assets, f_asset=f_asset, f_market=f_market, hidden=HIDDEN_DIM).to(device)
        self.critic = MTCritic(n_assets=n_assets, f_asset=f_asset, f_market=f_market, hidden=HIDDEN_DIM).to(device)

        self.actor_t = copy.deepcopy(self.actor).to(device).eval()
        self.critic_t = copy.deepcopy(self.critic).to(device).eval()
        for p in self.actor_t.parameters():
            p.requires_grad_(False)
        for p in self.critic_t.parameters():
            p.requires_grad_(False)

        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.rb = SubBuffer(BUFFER_SIZE, device=device, seed=seed)
        self.ou = OUNoise(action_dim=n_assets)

        self.train_step_count = 0

    @torch.no_grad()
    def act(self, asset_w: torch.Tensor, market_w: torch.Tensor, a_prev: torch.Tensor, *,
            explore: bool) -> torch.Tensor:
        logits = self.actor(asset_w, market_w, a_prev)
        if explore:
            n = torch.as_tensor(self.ou().astype(np.float32), device=self.device).unsqueeze(0)
            logits = logits + n
        return torch.softmax(logits, dim=-1)

    def push(
            self,
            *,
            obs: Dict[str, Any],
            action: np.ndarray,
            reward: float,
            next_obs: Optional[Dict[str, Any]],
            done: bool,
            expert_action: np.ndarray,
    ) -> None:
        if next_obs is None:
            next_asset = obs["asset_window"]
            next_market = obs["market_window"]
            next_wprev = obs["a_prev"]
        else:
            next_asset = next_obs["asset_window"]
            next_market = next_obs["market_window"]
            next_wprev = next_obs["a_prev"]

        self.rb.push(
            obs["asset_window"],
            obs["market_window"],
            obs["a_prev"],
            action,
            float(reward),
            next_asset,
            next_market,
            next_wprev,
            1.0 if done else 0.0,
            expert_action,
        )

    def update(self) -> Dict[str, float]:
        self.train_step_count += 1
        if len(self.rb) < BATCH_SIZE:
            return {}

        asset_b, market_b, wprev_b, act_b, rew_b, next_asset_b, next_market_b, next_wprev_b, done_b, exp_b = self.rb.sample(
            BATCH_SIZE)

        with torch.no_grad():
            next_logits = self.actor_t(next_asset_b, next_market_b, next_wprev_b)
            next_w = torch.softmax(next_logits, dim=-1)
            next_q = self.critic_t(next_asset_b, next_market_b, next_wprev_b, next_w)
            y = rew_b + GAMMA * (1.0 - done_b) * next_q

        q = self.critic(asset_b, market_b, wprev_b, act_b)
        loss_c = F.mse_loss(q, y)

        self.opt_c.zero_grad(set_to_none=True)
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 20.0)
        self.opt_c.step()

        logits = self.actor(asset_b, market_b, wprev_b)
        pred_w = torch.softmax(logits, dim=-1)

        loss_a_rl = -self.critic(asset_b, market_b, wprev_b, pred_w).mean()
        loss_sub = torch.tensor(0.0, device=self.device)
        loss_a = loss_a_rl
        if self.sub_lambda > 0.0:
            loss_sub = F.mse_loss(pred_w, exp_b)
            loss_a = loss_a + self.sub_lambda * loss_sub

        self.opt_a.zero_grad(set_to_none=True)
        loss_a.backward()
        gn = 0.0
        gn = float(torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 20.0).item())
        self.opt_a.step()

        soft_update(self.actor_t, self.actor, TAU)
        soft_update(self.critic_t, self.critic, TAU)

        return {
            "critic_loss": float(loss_c.item()),
            "actor_loss": float(loss_a.item()),
            "actor_loss_rl": float(loss_a_rl.item()),
            "actor_loss_sub": float(loss_sub.item()),
            "actor_grad_norm": float(gn),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path)

    def load(self, path: Path) -> None:
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(state_dict)
        self.actor_t.load_state_dict(state_dict)

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, att_dim: int = 128):
        super().__init__()
        self.W = nn.Linear(hidden_dim, att_dim)
        self.v = nn.Linear(att_dim, 1, bias=False)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        u = torch.tanh(self.W(H))
        e = self.v(u).squeeze(-1)
        att = torch.softmax(e, dim=1)
        c = torch.sum(H * att.unsqueeze(-1), dim=1)
        return c

class MetaQ(nn.Module):
    def __init__(self, seq_len: int, perf_dim: int, K: int):
        super().__init__()
        self.seq_len = int(seq_len)
        self.perf_dim = int(perf_dim)
        self.K = int(K)

        self.lstm = nn.LSTM(input_size=1, hidden_size=HIDDEN_DIM, num_layers=1, batch_first=True)
        self.att = TemporalAttention(hidden_dim=HIDDEN_DIM, att_dim=HIDDEN_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN_DIM + perf_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, K),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x[:, : self.seq_len]
        perf = x[:, self.seq_len: self.seq_len + self.perf_dim]
        H, _ = self.lstm(seq.unsqueeze(-1))
        z_m = self.att(H)
        z = torch.cat([z_m, perf], dim=1)
        return self.mlp(z)

class MetaReplay:
    def __init__(self, capacity: int, batch_size: int, seed: int):
        self.capacity = int(capacity)
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)
        self.ptr = 0
        self.size = 0
        self._allocated = False

    def __len__(self) -> int:
        return self.size

    def _alloc(self, s: np.ndarray) -> None:
        D = int(np.asarray(s).reshape(-1).shape[0])
        C = self.capacity
        self.S = np.empty((C, D), dtype=np.float32)
        self.A = np.empty((C,), dtype=np.int64)
        self.R = np.empty((C, 1), dtype=np.float32)
        self.S2 = np.empty((C, D), dtype=np.float32)
        self.D = np.empty((C, 1), dtype=np.float32)
        self._allocated = True

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, d: float) -> None:
        if not self._allocated:
            self._alloc(s)

        i = self.ptr
        self.S[i] = np.asarray(s, dtype=np.float32).reshape(-1)
        self.A[i] = int(a)
        self.R[i, 0] = float(r)
        self.S2[i] = np.asarray(s2, dtype=np.float32).reshape(-1)
        self.D[i, 0] = float(d)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self):
        idx = self.rng.integers(0, self.size, size=self.batch_size)
        return self.S[idx], self.A[idx], self.R[idx], self.S2[idx], self.D[idx]

class DQN:
    def __init__(self, *, seq_len: int, perf_dim: int, K: int, device: torch.device, seed: int):
        self.K = int(K)
        self.device = device
        self.rng = np.random.default_rng(seed)

        self.q = MetaQ(seq_len=seq_len, perf_dim=perf_dim, K=K).to(device)
        self.qt = copy.deepcopy(self.q).to(device).eval()
        for p in self.qt.parameters():
            p.requires_grad_(False)

        self.opt = torch.optim.Adam(self.q.parameters(), lr=3e-4)
        self.rb = MetaReplay(capacity=BUFFER_SIZE, batch_size=BATCH_SIZE, seed=seed)

        self.step = 0
        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_decay_steps = 200_000
        self.target_update = 2_000

    def _eps(self) -> float:
        t = self.step
        if t >= self.eps_decay_steps:
            return self.eps_end
        frac = t / float(self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    @torch.no_grad()
    def act(self, s: np.ndarray, *, deterministic: bool = False) -> int:
        eps = 0.0 if deterministic else self._eps()
        if (not deterministic) and (self.rng.random() < eps):
            return int(self.rng.integers(0, self.K))
        x = torch.as_tensor(np.asarray(s, dtype=np.float32), device=self.device).unsqueeze(0)
        q = self.q(x)[0]
        return int(torch.argmax(q).item())

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        self.rb.push(s, a, r, s2, 1.0 if done else 0.0)

    def update(self) -> Dict[str, float]:
        if len(self.rb) < BATCH_SIZE:
            return {}

        s, a, r, s2, d = self.rb.sample()
        st = torch.as_tensor(s, device=self.device, dtype=torch.float32)
        at = torch.as_tensor(a, device=self.device, dtype=torch.long).unsqueeze(1)
        rt = torch.as_tensor(r, device=self.device, dtype=torch.float32)
        s2t = torch.as_tensor(s2, device=self.device, dtype=torch.float32)
        dt = torch.as_tensor(d, device=self.device, dtype=torch.float32)

        q_cur = self.q(st).gather(1, at)
        with torch.no_grad():
            a2 = torch.argmax(self.q(s2t), dim=1, keepdim=True)
            q2 = self.qt(s2t).gather(1, a2)
            y = rt + (1.0 - dt) * GAMMA * q2

        loss = F.smooth_l1_loss(q_cur, y)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        if (self.step % self.target_update) == 0:
            self.qt.load_state_dict(self.q.state_dict())

        return {"meta_loss": float(loss.item())}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.q.state_dict(), path)

    def load(self, path: Path) -> None:
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.q.load_state_dict(state_dict)
        self.qt.load_state_dict(state_dict)

def policy_step(*, a_prev: np.ndarray, a_t: np.ndarray, y_t: np.ndarray, fee: float) -> Tuple[np.ndarray, float]:
    a_prev = normalize_weights(a_prev)
    a_t = normalize_weights(a_t)
    y_t = np.asarray(y_t, dtype=np.float64).reshape(-1)

    delta = float(np.abs(a_t - a_prev).sum())
    mu_t = max(1.0 - float(fee) * delta, EPS)

    port_gain = float(np.dot(y_t, a_t))
    port_gain = max(port_gain, EPS)
    rho_t = (port_gain * mu_t) - 1.0

    a_prev_next = (y_t * a_t) / port_gain
    a_prev_next = a_prev_next.astype(np.float32)
    a_prev_next = normalize_weights(a_prev_next)
    return a_prev_next, float(rho_t)


def perf_features(rets: deque) -> np.ndarray:
    if len(rets) == 0:
        return np.zeros((4,), dtype=np.float32)
    arr = np.asarray(rets, dtype=np.float32)
    mu = float(arr.mean())
    vol = float(arr.std())
    ac = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(ac)
    dd = (peak - ac) / (peak + EPS)
    mdd = float(dd.max()) if dd.size else 0.0
    last = float(arr[-1])
    return np.asarray([mu, vol, mdd, last], dtype=np.float32)

def build_meta_state(*, market_window: np.ndarray, shadow_rets: List[deque]) -> np.ndarray:
    seq = np.asarray(market_window[:, 0], dtype=np.float32).reshape(-1)
    denom = float(np.abs(seq).mean() + 1e-8)
    seq = seq / denom
    perf = np.concatenate([perf_features(d) for d in shadow_rets], axis=0).astype(np.float32)
    return np.concatenate([seq, perf], axis=0).astype(np.float32)

@torch.no_grad()
def eval_sub_policy(env: MarketEnv, agent: DDPG, *, device: torch.device) -> Dict[str, Any]:
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        steps += 1
        asset_t = to_tensor(obs["asset_window"], device=device).unsqueeze(0)
        market_t = to_tensor(obs["market_window"], device=device).unsqueeze(0)
        wprev_t = to_tensor(obs["a_prev"], device=device).unsqueeze(0)

        w = agent.act(asset_t, market_t, wprev_t, explore=False)[0].cpu().numpy()
        w = normalize_weights(w)

        obs, reward, terminated, truncated, info = env.step(w)
        total_reward += float(reward)
        done = bool(terminated or truncated)

    pm = calculate_pf_metrics(np.asarray(env.returns, dtype=np.float64))
    return {"pv": float(info["pv"]), "steps": int(steps), "metric": pm, "reward": float(total_reward)}

@torch.no_grad()
def eval_meta_policy(env: MarketEnv, sub_agents: List[DDPG], meta: DQN, *, device: torch.device) -> Dict[str, Any]:
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    K = len(sub_agents)
    N = env.action_dim
    shadow_wprev = [np.full((N,), 1.0 / N, dtype=np.float32) for _ in range(K)]
    shadow_rets = [deque(maxlen=12) for _ in range(K)]

    while not done:
        steps += 1
        s = build_meta_state(market_window=obs["market_window"], shadow_rets=shadow_rets)
        a = meta.act(s, deterministic=True)

        asset_t = to_tensor(obs["asset_window"], device=device).unsqueeze(0)
        market_t = to_tensor(obs["market_window"], device=device).unsqueeze(0)

        a_list: List[np.ndarray] = []
        for k in range(K):
            wprev_k = torch.as_tensor(shadow_wprev[k], device=device, dtype=torch.float32).unsqueeze(0)
            w_k = sub_agents[k].act(asset_t, market_t, wprev_k, explore=False)[0].cpu().numpy()
            a_list.append(normalize_weights(w_k))

        w = a_list[a]
        obs2, reward, terminated, truncated, info2 = env.step(w)
        total_reward += float(reward)
        done = bool(terminated or truncated)

        y_t = np.asarray(info2.get("y_t"), dtype=np.float32).reshape(-1)
        for k in range(K):
            shadow_wprev[k], rho_k = policy_step(a_prev=shadow_wprev[k], a_t=a_list[k], y_t=y_t, fee=env.fee)
            shadow_rets[k].append(rho_k)

        obs = obs2 if obs2 is not None else obs

    pm = calculate_pf_metrics(np.asarray(env.returns, dtype=np.float64))
    return {"pv": float(info2["pv"]), "steps": int(steps), "metric": pm, "reward": float(total_reward)}

def train_sub_policies(*, train_env: MarketEnv, val_env: MarketEnv, device: torch.device, out_dir: Path) -> List[DDPG]:
    obs0, _ = train_env.reset()
    N = train_env.action_dim
    Fs = obs0["asset_window"].shape[-1]
    Fm = obs0["market_window"].shape[-1]

    topk = N
    expert = ExpertGenerator(n_assets=N, topk=topk)

    sub_cfgs = [
        dict(name="csm", sub_lambda=1.0, expert_rule="csm"),
        dict(name="blsw", sub_lambda=1.0, expert_rule="blsw"),
        dict(name="hindsight", sub_lambda=1.0, expert_rule="hindsight"),
        dict(name="empty", sub_lambda=0.0, expert_rule="empty"),
    ]

    agents: List[DDPG] = []
    for i, cfg in enumerate(sub_cfgs):
        agents.append(
            DDPG(n_assets=N, f_asset=Fs, f_market=Fm, device=device, sub_lambda=cfg["sub_lambda"], seed=GLOBAL_SEED)
        )

    for ep in range(1, EPISODES + 1):
        for k, cfg in enumerate(sub_cfgs):
            obs, _ = train_env.reset()
            done = False

            dsr = DifferentialSharpeReward(eta=DSR_ETA, clip=DSR_CLIP)
            dsr.reset()
            agents[k].ou.reset()

            while not done:
                asset_t = to_tensor(obs["asset_window"], device=device).unsqueeze(0)
                market_t = to_tensor(obs["market_window"], device=device).unsqueeze(0)
                wprev_t = to_tensor(obs["a_prev"], device=device).unsqueeze(0)

                w_t = agents[k].act(asset_t, market_t, wprev_t, explore=True)[0].cpu().numpy()
                w_t = normalize_weights(w_t)

                obs2, _, terminated, truncated, info2 = train_env.step(w_t)
                done = bool(terminated or truncated)

                logret = float(info2.get("log_return", 0.0))
                r_port = float(math.expm1(logret))
                r = dsr.update(r_port)

                if cfg["expert_rule"] == "csm":
                    a_exp = expert.csm_momentum(obs["asset_window"])
                elif cfg["expert_rule"] == "blsw":
                    a_exp = expert.blsw_lowvol(obs["asset_window"])
                elif cfg["expert_rule"] == "hindsight":
                    y_t = np.asarray(info2.get("y_t"), dtype=np.float32).reshape(-1)
                    a_exp = expert.hindsight(y_t)
                elif cfg["expert_rule"] == "empty":
                    a_exp = expert.empty_uniform()
                else:
                    raise ValueError(f"Unknown expert_rule: {cfg['expert_rule']}")

                agents[k].push(obs=obs, action=w_t, reward=float(r), next_obs=obs2, done=done, expert_action=a_exp)
                agents[k].update()

                obs = obs2 if obs2 is not None else obs

        if (ep % 20) == 0:
            for k, cfg in enumerate(sub_cfgs):
                out = eval_sub_policy(val_env, agents[k], device=device)
                print(f"[train sub {cfg['name']}] episode={ep}/{EPISODES} reward={out['reward']:.6f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for k, cfg in enumerate(sub_cfgs):
        agents[k].save(out_dir / f"sub_{cfg['name']}.pth")

    return agents


def train_meta_policy(*, train_env: MarketEnv, val_env: MarketEnv, test_env: MarketEnv,
                      sub_agents: List[DDPG], device: torch.device, out_dir: Path) -> DQN:
    for a in sub_agents:
        a.actor.eval()
        a.critic.eval()
        for p in a.actor.parameters():
            p.requires_grad_(False)
        for p in a.critic.parameters():
            p.requires_grad_(False)

    obs0, _ = train_env.reset()
    seq_len = obs0["market_window"].shape[0]
    K = len(sub_agents)
    perf_dim = K * 4

    meta = DQN(seq_len=seq_len, perf_dim=perf_dim, K=K, device=device, seed=GLOBAL_SEED)

    for ep in range(1, EPISODES + 1):
        obs, _ = train_env.reset()
        done = False

        N = train_env.action_dim
        shadow_wprev = [np.full((N,), 1.0 / N, dtype=np.float32) for _ in range(K)]
        shadow_rets = [deque(maxlen=12) for _ in range(K)]

        dsr = DifferentialSharpeReward(eta=DSR_ETA, clip=DSR_CLIP)
        dsr.reset()

        while not done:
            s = build_meta_state(market_window=obs["market_window"], shadow_rets=shadow_rets)
            a_idx = meta.act(s, deterministic=False)

            asset_t = to_tensor(obs["asset_window"], device=device).unsqueeze(0)
            market_t = to_tensor(obs["market_window"], device=device).unsqueeze(0)

            w_list: List[np.ndarray] = []
            for k in range(K):
                wprev_k = torch.as_tensor(shadow_wprev[k], device=device, dtype=torch.float32).unsqueeze(0)
                w_k = sub_agents[k].act(asset_t, market_t, wprev_k, explore=False)[0].cpu().numpy()
                w_list.append(normalize_weights(w_k))

            w = w_list[a_idx]
            obs2, _, terminated, truncated, info2 = train_env.step(w)
            done = bool(terminated or truncated)

            logret = float(info2.get("log_return", 0.0))
            r_port = float(math.expm1(logret))
            r = dsr.update(r_port)

            y_t = np.asarray(info2.get("y_t"), dtype=np.float32).reshape(-1)
            for k in range(K):
                shadow_wprev[k], rho_k = policy_step(a_prev=shadow_wprev[k], a_t=w_list[k], y_t=y_t, fee=train_env.fee)
                shadow_rets[k].append(rho_k)

            obs_next = obs if obs2 is None else obs2
            s2 = build_meta_state(market_window=obs_next["market_window"], shadow_rets=shadow_rets)

            meta.push(s, a_idx, float(r), s2, done)
            meta.update()
            meta.step += 1

            obs = obs2 if obs2 is not None else obs

        if (ep % 20) == 0:
            val_out = eval_meta_policy(val_env, sub_agents, meta, device=device)
            print(f"[train meta] episode={ep}/{EPISODES} reward={val_out['reward']:.6f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    meta.save(out_dir / "meta.pth")

    test_out = eval_meta_policy(test_env, sub_agents, meta, device=device)
    print("\n=== DONE ===")
    print("Test Metrics:")
    for k, v in test_out["metric"].items():
        print(f"  {k}: {v:.3f}")
    print(f"Test PV: {test_out['pv']:.4f}")

    return meta


def test_meta_policy(N, Fs, Fm, L, test_env: MarketEnv, device: torch.device, out_dir: Path) -> Dict[str, float]:
    sub_agents = []
    sub_names = ["csm", "blsw", "hindsight", "empty"]
    for i, name in enumerate(sub_names):
        agent = DDPG(
            n_assets=N,
            f_asset=Fs,
            f_market=Fm,
            device=device,
            sub_lambda=1.0 if name != "empty" else 0.0,
            seed=GLOBAL_SEED,
        )
        agent.load(out_dir / f"sub_{name}.pth")
        sub_agents.append(agent)

    meta = DQN(
        seq_len=L,
        perf_dim=len(sub_agents) * 4,
        K=len(sub_agents),
        device=device,
        seed=GLOBAL_SEED,
    )
    meta.load(out_dir / "meta.pth")

    test_out = eval_meta_policy(test_env, sub_agents, meta, device=device)
    metrics = test_out["metric"]
    return metrics