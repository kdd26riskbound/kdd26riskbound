from __future__ import annotations
import torch

import argparse
from typing import Optional

import numpy as np

from riskbound.core.rabg import RiskAwareBoundaryGenerator
from riskbound.settings import MODEL_DIR, GLOBAL_SEED, EPISODES

from riskbound.data.dataset import load_data
from riskbound.data.market_env import MarketEnv
from riskbound.data.source import DataSource
from riskbound.core.adapter import ActionAdapter
from riskbound.rl.ddpg import DDPGAgent
from riskbound.rl.models import build_actor_critic
from riskbound.rl.noise import OUNoise
from riskbound.utils import set_seed, to_tensor, evaluate_agent

def train_rl(
    market: str,
    fpsg_beta: float,
    fpsg_leak: float,
    global_cap: float,
    device: Optional[str] = None,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(GLOBAL_SEED)
    market_dates, market_features, asset_symbols, asset_features, rate_of_returns = load_data(market=market)

    train_ds = DataSource(
        asset_features=asset_features,
        market_features=market_features,
        rate_of_returns=rate_of_returns,
        dates=market_dates,
        seed=GLOBAL_SEED,
    )
    val_ds = DataSource(
        asset_features=asset_features,
        market_features=market_features,
        rate_of_returns=rate_of_returns,
        dates=market_dates,
        seed=GLOBAL_SEED,
    )
    test_ds = DataSource(
        asset_features=asset_features,
        market_features=market_features,
        rate_of_returns=rate_of_returns,
        dates=market_dates,
        seed=GLOBAL_SEED,
    )
    train_ds.set_mode("train")
    val_ds.set_mode("val")
    test_ds.set_mode("test")

    train_env = MarketEnv(train_ds)
    val_env = MarketEnv(val_ds)
    test_env = MarketEnv(test_ds)

    rabg = RiskAwareBoundaryGenerator(
        global_cap=global_cap,
    )
    ou = OUNoise(action_dim=train_env.action_dim)
    model_path = MODEL_DIR / "riskbound" / market / "rabg" / f"risk_scorer.pth"
    rabg.load_model(model_path, device=device)
    build_actor_critic(action_dim=train_env.action_dim)
    agent = DDPGAgent(action_dim=train_env.action_dim, device=device)
    adapter = ActionAdapter(
        action_dim=train_env.action_dim,
        device=device,
        rabg=rabg,
        fpsg_beta=fpsg_beta,
        fpsg_leak=fpsg_leak,
    )
    for ep in range(EPISODES):
        obs, _ = train_env.reset()
        ou.reset()
        done = False
        total_reward = 0.0
        step = 0

        while not done:
            x_s = to_tensor(obs["asset_window"], device=device).unsqueeze(0)
            x_m = to_tensor(obs["market_window"], device=device).unsqueeze(0)
            a_prev = to_tensor(obs["a_prev"], device=device).unsqueeze(0)

            with torch.no_grad():
                a_min, a_max = adapter.get_bounds(x_s, x_m)
                z_action = agent.actor(x_s, x_m, a_prev)
                z_np = z_action[0].cpu().numpy().astype(np.float32)
                z_noisy = z_np + ou()
                a_raw = torch.softmax(torch.as_tensor(z_noisy, device=device, dtype=torch.float32).unsqueeze(0), dim=-1)
                action = adapter.project(a_raw, a_min, a_max)[0].cpu().numpy().astype(np.float32)

            obs_next, reward, terminated, truncated, info_next = train_env.step(action)
            done = bool(terminated or truncated)

            total_reward += float(reward)
            step += 1

            if obs_next is None:
                next_asset_np = obs["asset_window"]
                next_market_np = obs["market_window"]
                next_a_prev_np = obs["a_prev"]
            else:
                next_asset_np = obs_next["asset_window"]
                next_market_np = obs_next["market_window"]
                next_a_prev_np = obs_next["a_prev"]

            agent.transition(
                obs["asset_window"],
                obs["market_window"],
                obs["a_prev"],
                action,
                float(reward),
                next_asset_np,
                next_market_np,
                next_a_prev_np,
                1.0 if done else 0.0,
            )

            obs = obs_next

            agent.update(adapter)

        if (ep + 1) % 20 == 0 or ep == 0:
            avg_reward = total_reward / max(step, 1)
            print(f"[train] episode {ep + 1}/{EPISODES} reward={avg_reward:.6f}")

    test_out = evaluate_agent(
        test_env,
        agent.actor,
        adapter,
        device=device,
    )
    pm = test_out["metric"]

    print("=== DONE ===")
    print("Test Metrics:")
    for k, v in pm.items():
        print(f"  {k}: {v:.3f}")

    model_save_dir = MODEL_DIR / "riskbound" / market / "agent"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    agent.save_model(model_save_dir)

def main():
    p = argparse.ArgumentParser("Train RL with RiskBound")

    p.add_argument("--market", type=str, default="us")

    p.add_argument("--fpsg_beta", type=float, default=10.0)
    p.add_argument("--fpsg_leak", type=float, default=0.1)
    p.add_argument("--global_cap", type=float, default=1.5)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))

    args = p.parse_args()
    train_rl(
        market=args.market,
        fpsg_beta=args.fpsg_beta,
        fpsg_leak=args.fpsg_leak,
        global_cap=args.global_cap,
        device=args.device,
    )

if __name__ == "__main__":
    main()