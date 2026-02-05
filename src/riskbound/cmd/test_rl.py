from __future__ import annotations
import torch

import argparse
from typing import Optional, Dict

from riskbound.core.rabg import RiskAwareBoundaryGenerator
from riskbound.settings import MODEL_DIR, GLOBAL_SEED

from riskbound.data.dataset import load_data
from riskbound.data.market_env import MarketEnv
from riskbound.data.source import DataSource
from riskbound.core.adapter import ActionAdapter
from riskbound.rl.ddpg import DDPGAgent
from riskbound.utils import set_seed, evaluate_agent, report_pf_metrics


def test_rl(
    market: str,
    fpsg_beta: float,
    fpsg_leak: float,
    global_cap: float,
    device: Optional[str] = None,
) -> Dict[str, float]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(GLOBAL_SEED)

    market_dates, market_features, asset_symbols, asset_features, rate_of_returns = load_data(market=market)

    test_ds = DataSource(
        asset_features=asset_features,
        market_features=market_features,
        rate_of_returns=rate_of_returns,
        dates=market_dates,
        seed=GLOBAL_SEED,
    )
    test_ds.set_mode("test")
    test_env = MarketEnv(test_ds)

    rabg = RiskAwareBoundaryGenerator(
        global_cap=global_cap,
    )
    model_path = MODEL_DIR / "riskbound" / market / "rabg" / f"risk_scorer.pth"
    rabg.load_model(model_path, device=device)

    agent = DDPGAgent.load_model(
        MODEL_DIR / "riskbound" / market / "agent",
    )
    adapter = ActionAdapter(
        action_dim=agent.action_dim,
        device=device,
        rabg=rabg,
        fpsg_beta=fpsg_beta,
        fpsg_leak=fpsg_leak,
    )
    test_out = evaluate_agent(
        test_env,
        agent.actor,
        adapter,
        device=device,
    )
    metrics = test_out["metric"]

    return metrics

def main():
    p = argparse.ArgumentParser("Test RL Agent")

    p.add_argument("--market", type=str, default="us")
    p.add_argument("--no-header", action="store_true", help="Do not print header for metrics")
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--fpsg_beta", type=float, default=10.0)
    p.add_argument("--fpsg_leak", type=float, default=0.1)
    p.add_argument("--global_cap", type=float, default=1.5)


    args = p.parse_args()
    metrics = test_rl(
        market=args.market,
        fpsg_beta=args.fpsg_beta,
        fpsg_leak=args.fpsg_leak,
        global_cap=args.global_cap,
        device=args.device,
    )
    report_pf_metrics(args.market, 'RiskBound (Proposed)', metrics, header=not args.no_header)

if __name__ == "__main__":
    main()