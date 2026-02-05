from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from typing import Optional, Dict

from torch.nn import functional as F

from riskbound.core.adapter import ActionAdapter
from riskbound.rl.models import build_actor_critic
from riskbound.rl.buffer import ReplayBuffer
import torch
import json

class DDPGAgent:
    def __init__(
            self,
            action_dim,
            actor_lr: float=1e-4,
            critic_lr: float=3e-4,
            batch_size: int=64,
            buffer_size: int=10_000,
            tau: float=0.005,
            gamma: float=0.99,
            grad_clip: float=20.0,
            device: str=None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.actor, self.critic = build_actor_critic(
            action_dim=action_dim,
        )
        self.actor = self.actor.to(self.device).train()
        self.critic = self.critic.to(self.device).train()
        self.actor_target = deepcopy(self.actor).to(self.device).eval()
        self.critic_target = deepcopy(self.critic).to(self.device).eval()

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size, device=self.device)

        self.tau = tau
        self.gamma = gamma
        self.grad_clip = grad_clip

    def transition(self, asset, market, a_prev, action, reward, next_asset, next_market, next_a_prev, done) -> None:
        self.buffer.push(
            asset,
            market,
            a_prev,
            action,
            float(reward),
            next_asset,
            next_market,
            next_a_prev,
            1.0 if done else 0.0,
        )

    def update(self, adapter: ActionAdapter) -> Optional[Dict[str, float]]:
        if len(self.buffer) < self.batch_size:
            return None

        (assets, markets, a_prevs, actions, rewards,
         next_assets, next_markets, next_a_prevs, dones) = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            next_a_min, next_a_max = adapter.get_bounds(next_assets, next_markets)

            next_z_action = self.actor_target(next_assets, next_markets, next_a_prevs)
            next_a_raw = torch.softmax(next_z_action, dim=-1)
            next_actions = adapter.project(next_a_raw, next_a_min, next_a_max)
            next_q = self.critic_target(next_assets, next_markets, next_a_prevs, next_actions)

            target_q = rewards + self.gamma * (1.0 - dones) * next_q

        q = self.critic(assets, markets, a_prevs, actions)
        critic_loss = F.mse_loss(q, target_q)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip).item()

        self.critic_opt.step()

        self.actor_opt.zero_grad(set_to_none=True)
        a_min, a_max = adapter.get_bounds(assets, markets)

        pred_z_action = self.actor(assets, markets, a_prevs)
        pred_a_raw = torch.softmax(pred_z_action, dim=-1)
        pred_action = adapter.project(pred_a_raw, a_min, a_max)

        actor_loss = -self.critic(assets, markets, a_prevs, pred_action).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip).item()
        self.actor_opt.step()
        self.soft_update()

    def soft_update(self) -> None:
        with torch.no_grad():
            for tp, sp in zip(self.critic_target.parameters(), self.critic.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(sp.data, alpha=self.tau)

    def save_model(self, path: str) -> None:
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), save_dir / "ddpg_actor.pth")
        torch.save(self.critic.state_dict(), save_dir / "ddpg_critic.pth")
        config = {
            "action_dim": self.action_dim,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "gamma": self.gamma,
            "grad_clip": self.grad_clip,
        }
        with open(save_dir / "ddpg_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

    @classmethod
    def load_model(cls, path: str, device: str=None) -> DDPGAgent:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        load_dir = Path(path)
        with open(load_dir / "ddpg_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        agent = cls(
            action_dim=config["action_dim"],
            actor_lr=1e-4,
            critic_lr=3e-4,
            batch_size=config["batch_size"],
            tau=config["tau"],
            gamma=config["gamma"],
            grad_clip=config["grad_clip"],
            device=device,
        )
        agent.actor.load_state_dict(torch.load(load_dir / "ddpg_actor.pth", map_location="cpu", weights_only=True))
        agent.critic.load_state_dict(torch.load(load_dir / "ddpg_critic.pth", map_location="cpu", weights_only=True))
        agent.actor_target = deepcopy(agent.actor).to(agent.device).eval()
        agent.critic_target = deepcopy(agent.critic).to(agent.device).eval()
        return agent