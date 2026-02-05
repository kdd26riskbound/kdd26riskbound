from __future__ import annotations

import gymnasium as gym
import numpy as np

from riskbound.settings import TRADING_COST
from riskbound.data.source import DataSource

class MarketEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, data_source: DataSource):
        super().__init__()
        self.src = data_source

        self.N = data_source.N
        self.action_dim = self.N

        self.fee = TRADING_COST

        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        self.returns = []

        self.evolved_weight = self._init_weights()
        self.portfolio_value = 1.0

        self.reset()

    def _init_weights(self) -> np.ndarray:
        return np.full((self.action_dim,), 1.0 / self.N, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.src.reset(seed=seed)
        self.evolved_weight = self._init_weights()
        self.portfolio_value = 1.0

        self.returns = []

        if obs is not None:
            obs.update({
                "a_prev": self.evolved_weight.copy(),
            })
        info = {
            "pv": self.portfolio_value
        }
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        assert action.shape[0] == self.action_dim, f"action shape must be ({self.action_dim},)"
        assert abs(float(action.sum()) - 1.0) < 1e-5, "Action weights must sum to 1."

        a_t = action
        a_prime_t = self.evolved_weight
        p_t_prev = self.portfolio_value

        delta = np.abs(a_t - a_prime_t).sum()
        mu_t = max(1.0 - self.fee * delta, 0.0)

        if mu_t <= 0.0:
            raise ValueError("Non-positive shrink factor encountered (mu_t <= 0).")

        obs, terminated, info = self.src.step()
        returns = np.asarray(info["label"], dtype=np.float32)

        g_assets = 1.0 + returns

        if np.any(g_assets <= 0.0):
            raise ValueError("Non-positive asset gross return encountered (g_assets <= 0).")
        g_all = g_assets

        y_t = np.prod(g_all, axis=1).astype(np.float32)

        portfolio_gain = float(y_t @ a_t)
        if portfolio_gain <= 0.0:
            raise ValueError("Non-positive portfolio gain encountered (y_t @ a_t <= 0).")
        p_prime_t = p_t_prev * portfolio_gain
        p_t = p_prime_t * mu_t
        rho_t = p_t / p_t_prev - 1.0
        self.portfolio_value = p_t

        log_return = float(np.log(p_t / p_t_prev))
        reward = log_return

        a_prime_t_next = (y_t * a_t) / portfolio_gain

        self.evolved_weight = a_prime_t_next

        fee_rate = 1.0 - mu_t
        self.returns.append(rho_t)

        truncated = False
        if obs is not None:
            obs.update({
                "a_prev": self.evolved_weight.copy(),
            })
        info = {
            "pv": self.portfolio_value,
            "y_t": y_t,
            "delta": delta,
            "reward": reward,
            "log_return": log_return,
            "fee_rate": fee_rate
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        return