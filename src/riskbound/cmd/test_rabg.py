import argparse
import torch

from riskbound.cmd.pretrain_rabg import create_rabg_dataset
from riskbound.data.dataset import load_data
from riskbound.data.source import DataSource
from riskbound.core.rabg import RiskScorer
from riskbound.settings import MODEL_DIR, LOOKBACK_WINDOW, HOLDING_PERIOD
from riskbound.metrics import pearson_series, rankic_series, mean_ir
from riskbound.utils import calculate_rollvol, calculate_rolldownvol, calculate_rollmdd, calculate_garch

ROLLING_WINDOW = LOOKBACK_WINDOW


def main():
    parser = argparse.ArgumentParser("Test Risk-Aware Boundary Generator")
    parser.add_argument("--market", type=str, default="us")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()


    dates, market_features, asset_symbols, asset_features, ror_array = load_data(market=args.market)
    model_path = MODEL_DIR / "riskbound" / args.market / "rabg" / f"risk_scorer.pth"

    try:
        rabg_model = RiskScorer.load_model(model_path, args.device)
        rabg_model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    ds = DataSource(
        asset_features=asset_features,
        market_features=market_features,
        rate_of_returns=ror_array,
        dates=dates,
    )

    ds.set_mode("test")

    Xs, Xm, Yf = create_rabg_dataset(ds, stride=HOLDING_PERIOD, device=args.device)
    baselines = {
        "Roll DownVol": calculate_rolldownvol,
        "Roll Vol": calculate_rollvol,
        "Roll MDD": calculate_rollmdd,
        "GARCH": calculate_garch,
        "RiskBound (Proposed)": lambda Xs, Xm: rabg_model(Xs, Xm)
    }
    print()
    print("Market:", args.market.upper())
    print(f"{'Method':<20} | {'IC':<8} | {'ICIR':<8} | {'RankIC':<8} | {'RankICIR':<8}")
    print("-" * 60)

    for name, model in baselines.items():
        with torch.no_grad():
            pred = model(Xs, Xm)
        assert pred.shape == Yf.shape, f"Prediction shape mismatch for {name}"
        assert pred.isnan().sum().item() == 0, f"NaN values found in predictions of {name}"
        ic = pearson_series(pred, Yf)
        ric = rankic_series(pred, Yf)

        IC, ICIR = mean_ir(ic)
        RIC, RICIR = mean_ir(ric)

        print(f"{name:<20} | {IC:.3f}   | {ICIR:.3f}   | {RIC:.3f}     | {RICIR:.3f}")


if __name__ == "__main__":
    main()