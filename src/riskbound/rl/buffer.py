from __future__ import annotations

from collections import deque
from typing import Deque, Tuple

import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity: int, device: str):
        self.buffer: Deque[Tuple] = deque(maxlen=capacity)
        self.device = device

    def __len__(self) -> int:
        return len(self.buffer)

    @staticmethod
    def _ensure_1d(x: np.ndarray, name: str) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError(f"{name} must be 1D, got shape={x.shape}")
        return x

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
    ) -> None:
        a_prev = self._ensure_1d(a_prev, "current_weight").astype(np.float32, copy=False)
        action = self._ensure_1d(action, "action").astype(np.float32, copy=False)
        next_a_prev = self._ensure_1d(next_a_prev, "next_a_prev").astype(np.float32, copy=False)

        done_f = float(done)
        if done_f not in (0.0, 1.0):
            raise ValueError(f"done must be 0.0 or 1.0, got {done_f}")

        self.buffer.append(
            (
                asset_window.astype(np.float32, copy=False),
                market_window.astype(np.float32, copy=False),
                a_prev,
                action,
                float(reward),
                next_asset_window.astype(np.float32, copy=False),
                next_market_window.astype(np.float32, copy=False),
                next_a_prev,
                done_f,
            )
        )

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        asset_w, market_w, a_prev, action, reward, next_asset_w, next_market_w, next_a_prev, done = zip(*batch)
        asset_b = torch.as_tensor(np.stack(asset_w), device=self.device, dtype=torch.float32)            
        market_b = torch.as_tensor(np.stack(market_w), device=self.device, dtype=torch.float32)          
        wprev_b = torch.as_tensor(np.stack(a_prev), device=self.device, dtype=torch.float32)
        action_b = torch.as_tensor(np.stack(action), device=self.device, dtype=torch.float32)            
        reward_b = torch.as_tensor(np.array(reward, dtype=np.float32), device=self.device).unsqueeze(-1)
        next_asset_b = torch.as_tensor(np.stack(next_asset_w), device=self.device, dtype=torch.float32)  
        next_market_b = torch.as_tensor(np.stack(next_market_w), device=self.device, dtype=torch.float32)
        next_wprev_b = torch.as_tensor(np.stack(next_a_prev), device=self.device, dtype=torch.float32)
        done_b = torch.as_tensor(np.array(done, dtype=np.float32), device=self.device).unsqueeze(-1)
        return asset_b, market_b, wprev_b, action_b, reward_b, next_asset_b, next_market_b, next_wprev_b, done_b
