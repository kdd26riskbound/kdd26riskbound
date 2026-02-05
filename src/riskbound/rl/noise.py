from __future__ import annotations

import numpy as np


class OUNoise:
    def __init__(self, action_dim, x0=None, theta=0.15, mu=0.0, sigma=0.3, dt=1e-2):
        self.theta = theta
        self.mu = np.zeros(action_dim)
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
