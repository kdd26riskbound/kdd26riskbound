import argparse
import torch
from riskbound.settings import MODEL_DIR, GLOBAL_SEED
from riskbound.data.dataset import load_data
from riskbound.data.source import DataSource
from riskbound.data.market_env import MarketEnv
from riskbound.baselines.metatrader import train_sub_policies, train_meta_policy, test_meta_policy
from riskbound.utils import set_seed, report_pf_metrics


def main() -> None:
    p = argparse.ArgumentParser("MetaTrader runner")

    p.add_argument("--market", type=str, default="us")
    p.add_argument("--no-header", action="store_true", help="Do not print header for metrics")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    set_seed(GLOBAL_SEED)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    dates, market_features, symbols, asset_features, ror_array = load_data(market=args.market)

    train_src = DataSource(asset_features=asset_features, market_features=market_features,
                           rate_of_returns=ror_array, dates=dates, seed=GLOBAL_SEED)
    val_src = DataSource(asset_features=asset_features, market_features=market_features,
                         rate_of_returns=ror_array, dates=dates, seed=GLOBAL_SEED)
    test_src = DataSource(asset_features=asset_features, market_features=market_features,
                          rate_of_returns=ror_array, dates=dates, seed=GLOBAL_SEED)

    train_src.set_mode("train")
    val_src.set_mode("val")
    test_src.set_mode("test")

    train_env = MarketEnv(train_src)
    val_env = MarketEnv(val_src)
    test_env = MarketEnv(test_src)

    out_dir = MODEL_DIR / "metatrader" / f"{args.market}"
    model_exists = all((out_dir / fname).exists() for fname in [
        "sub_csm.pth",
        "sub_blsw.pth",
        "sub_hindsight.pth",
        "sub_empty.pth",
        "meta.pth",
    ])

    if not model_exists:
        sub_agents = train_sub_policies(train_env=train_env, val_env=val_env, device=device, out_dir=out_dir)
        train_meta_policy(train_env=train_env, val_env=val_env, test_env=test_env,
                          sub_agents=sub_agents, device=device, out_dir=out_dir)
    else:
        obs0, _ = train_env.reset()
        N = train_env.action_dim
        Fs = obs0["asset_window"].shape[-1]
        Fm = obs0["market_window"].shape[-1]
        L = obs0["market_window"].shape[0]

        metrics = test_meta_policy(N=N, Fs=Fs, Fm=Fm, L=L, test_env=test_env, device=device, out_dir=out_dir)
        report_pf_metrics(args.market, 'MetaTrader', metrics, header=not args.no_header)

if __name__ == "__main__":
    main()
