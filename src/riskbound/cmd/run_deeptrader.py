from __future__ import annotations

from riskbound.baselines.deeptrader.environment.portfolio_env import PortfolioEnv
from riskbound.baselines.deeptrader.agent import RLActor, RLAgent
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import argparse
import json
import numpy as np
import pandas as pd
import torch

from riskbound.settings import LOOKBACK_WINDOW, HOLDING_PERIOD, TRAIN_END_DATE, VAL_END_DATE, TEST_END_DATE, \
    DATA_DIR, MODEL_DIR, GLOBAL_SEED, EPISODES, HIDDEN_DIM, BATCH_SIZE, TRADING_COST
from riskbound.metrics import calculate_pf_metrics
from riskbound.utils import set_seed, report_pf_metrics

@dataclass
class SplitInfo:
    end_idx: int
    val_idx: int
    test_idx: int
    date_from: str
    train_end: str
    val_end: str
    test_end: str
    date_to: str

def load_np(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return np.load(path, allow_pickle=True)

def load_preprocessed_dir(data_dir: Path) -> Dict[str, object]:
    data_dir = data_dir.expanduser().resolve()
    dates = load_np(data_dir / "dates.npy")
    symbols = load_np(data_dir / "symbols.npy")
    asset_features = load_np(data_dir / "asset_features.npy")
    market_features = load_np(data_dir / "market_features.npy")

    meta_path = data_dir / "meta.json"
    meta = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    out = {
        "data_dir": str(data_dir),
        "dates": np.asarray(dates, dtype=str),
        "symbols": np.asarray(symbols, dtype=str),
        "asset_features": np.asarray(asset_features, dtype=np.float32),
        "market_features": np.asarray(market_features, dtype=np.float32),
        "meta": meta,
    }

    ror_path = data_dir / "ror_array.npy"
    if ror_path.exists():
        out["ror_array"] = np.asarray(np.load(ror_path, allow_pickle=True), dtype=np.float32)
    else:
        out["ror_array"] = None

    return out

def get_returns(asset_features: np.ndarray, meta: Optional[dict]) -> Tuple[np.ndarray, str]:
    if meta and "asset_feature_labels" in meta:
        labels = list(meta["asset_feature_labels"])
        if "r_adj_close" in labels:
            idx = labels.index("r_adj_close")
            return asset_features[:, :, idx], "asset_features:r_adj_close"
    return asset_features[:, :, -1], "asset_features:last_feature"

def compute_splits(
    dates: np.ndarray,
    train_end_date: str,
    val_end_date: str,
    test_end_date: str,
    trade_len: int,
    window_len: int = 1,
) -> SplitInfo:
    dt = pd.to_datetime(pd.Series(dates), utc=False).values
    T = len(dates)

    def right_idx(end_date: str) -> int:
        ts = np.datetime64(pd.Timestamp(end_date))
        return int(np.searchsorted(dt, ts, side="right"))

    val_idx  = right_idx(train_end_date)
    test_idx = right_idx(val_end_date)
    end_idx  = right_idx(test_end_date)
    end_idx = min(end_idx, T)
    val_idx = min(val_idx, end_idx)
    test_idx = min(test_idx, end_idx)
    feasible_end = max(0, end_idx - trade_len)

    end_idx = feasible_end
    val_idx = min(val_idx, end_idx)
    test_idx = min(test_idx, end_idx)
    min_need = window_len + trade_len + 1
    if val_idx < min_need:
        raise ValueError(f"Train too short after clamping: val_idx={val_idx}, need >= {min_need}")
    if (test_idx - val_idx) < min_need:
        raise ValueError(f"Val too short after clamping: val span={test_idx - val_idx}, need >= {min_need}")
    if (end_idx - test_idx) < min_need:
        raise ValueError(f"Test too short after clamping: test span={end_idx - test_idx}, need >= {min_need}")

    date_from = str(dates[0])
    date_to = str(dates[end_idx - 1]) if end_idx > 0 else str(dates[-1])

    return SplitInfo(
        end_idx=end_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        date_from=date_from,
        train_end=train_end_date,
        val_end=val_end_date,
        test_end=test_end_date,
        date_to=date_to,
    )

@torch.no_grad()
def evaluate_agent(env: PortfolioEnv, actor: RLActor, mode: str, deterministic: bool = True) -> Tuple[pd.DataFrame, Dict[str, float]]:
    actor.eval()
    mode = mode.lower()
    if mode == "val":
        env.set_eval()
    elif mode == "test":
        env.set_test()
    else:
        raise ValueError("mode must be one of {'val','test'}")

    (xa, xm), mask = env.reset()
    xa_t = torch.from_numpy(xa).to(actor.args.device)
    xm_t = torch.from_numpy(xm).to(actor.args.device) if xm is not None else None
    mask_t = torch.from_numpy(mask).to(actor.args.device)

    rows = []
    next_end_idx = int(env.src.cursor[0]) if hasattr(env.src, "cursor") else None

    done = False
    while not done:
        weights, rho, _, _ = actor(xa_t, xm_t, mask_t, deterministic=deterministic)

        end_idx = next_end_idx
        (xa_n, xm_n), _, _, mask_n, done, info = env.step(weights, rho)

        xa_t = torch.from_numpy(xa_n).to(actor.args.device) if xa_n is not None else None
        xm_t = torch.from_numpy(xm_n).to(actor.args.device) if xm_n is not None else None
        mask_t = torch.from_numpy(mask_n).to(actor.args.device) if mask_n is not None else None

        next_end_idx = int(env.src.cursor[0]) if hasattr(env.src, "cursor") else None

        r = float(np.asarray(info["rate_of_return"]).reshape(-1)[0])
        v = float(np.asarray(info["total_value"]).reshape(-1)[0])
        p = info.get("p", None)
        p0 = float(np.asarray(p).reshape(-1)[0]) if p is not None else float("nan")

        end_date = None
        if end_idx is not None and hasattr(env.src, "dates") and 0 <= end_idx < len(env.src.dates):
            end_date = str(env.src.dates[end_idx])

        rows.append({
            "step": len(rows),
            "end_idx": end_idx,
            "end_date": end_date,
            "period_return": r,
            "portfolio_value": v,
            "rho_p": p0,
        })

    df = pd.DataFrame(rows)

    metrics = calculate_pf_metrics(returns=df["period_return"].values)
    return df, metrics

def main() -> None:
    p = argparse.ArgumentParser("DeepTrader runner")
    p.add_argument("--market", type=str, default="us")
    p.add_argument("--norm_type", type=str, default="standard", choices=["standard", "min-max", "div-last"])
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--num_blocks", type=int, default=3)
    p.add_argument("--G", type=int, default=None)
    p.add_argument("--no_spatial_bool", action="store_true")
    p.add_argument("--no_addaptiveadj", action="store_true")
    p.add_argument("--no_msu_bool", action="store_true")
    p.add_argument("--no-header", action="store_true", help="Do not print header for metrics")


    args_cli = p.parse_args()

    allow_short = False

    spatial_bool = True
    if args_cli.no_spatial_bool:
        spatial_bool = False

    addaptiveadj = True
    if args_cli.no_addaptiveadj:
        addaptiveadj = False

    msu_bool = True
    if args_cli.no_msu_bool:
        msu_bool = False

    set_seed(GLOBAL_SEED)

    data_dir = DATA_DIR / args_cli.market
    data = load_preprocessed_dir(data_dir)
    dates = data["dates"]
    symbols = data["symbols"]
    asset_features = data["asset_features"]
    market_features = data["market_features"]
    meta = data["meta"]

    rtns_data, rtn_source = get_returns(asset_features, meta)

    split = compute_splits(
        dates,
        train_end_date=TRAIN_END_DATE,
        val_end_date=VAL_END_DATE,
        test_end_date=TEST_END_DATE,
        trade_len=HOLDING_PERIOD,
        window_len=LOOKBACK_WINDOW,
    )

    end_idx = split.end_idx
    dates = dates[:end_idx]
    asset_features = asset_features[:, :end_idx]
    market_features = market_features[:end_idx]
    rtns_data = rtns_data[:, :end_idx]

    val_idx, test_idx = split.val_idx, split.test_idx
    if not (0 < val_idx < test_idx <= end_idx):
        raise ValueError(
            f"Invalid split indices: val_idx={val_idx}, test_idx={test_idx}, end_idx={end_idx}. "
            "Check your end dates and available date range."
        )

    if val_idx <= LOOKBACK_WINDOW - 1:
        raise ValueError(
            f"val_idx={val_idx} is too early for window_len={LOOKBACK_WINDOW}. "
            "Need at least window_len-1 history before validation start."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = SimpleNamespace(
        num_assets=int(asset_features.shape[0]),
        in_features=[int(asset_features.shape[2]), int(market_features.shape[1])],
        hidden_dim=int(HIDDEN_DIM),
        window_len=int(LOOKBACK_WINDOW),
        dropout=float(args_cli.dropout),
        kernel_size=int(args_cli.kernel_size),
        num_blocks=int(args_cli.num_blocks),
        spatial_bool=bool(spatial_bool),
        addaptiveadj=bool(addaptiveadj),
        msu_bool=bool(msu_bool),
        G=int(args_cli.G or asset_features.shape[0]),
        lr=float(args_cli.lr),
        device=device,
    )


    supports = []

    env_train = PortfolioEnv(
        assets_data=asset_features,
        market_data=market_features,
        rtns_data=rtns_data,
        in_features=args.in_features,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=int(BATCH_SIZE),
        fee=TRADING_COST,
        time_cost=0.0,
        window_len=int(LOOKBACK_WINDOW),
        trade_len=int(HOLDING_PERIOD),
        max_steps=int(HOLDING_PERIOD),
        norm_type=str(args_cli.norm_type),
        is_norm=True,
        allow_short=bool(allow_short),
        mode="train",
        assets_name=symbols,
        dates=dates,
    )

    env_eval = PortfolioEnv(
        assets_data=asset_features,
        market_data=market_features,
        rtns_data=rtns_data,
        in_features=args.in_features,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=1,
        fee=TRADING_COST,
        time_cost=0.0,
        window_len=int(LOOKBACK_WINDOW),
        trade_len=int(HOLDING_PERIOD),
        max_steps=int(HOLDING_PERIOD),
        norm_type=str(args_cli.norm_type),
        is_norm=True,
        allow_short=bool(allow_short),
        mode="eval",
        assets_name=symbols,
        dates=dates,
    )

    actor = RLActor(supports=supports, args=args).to(device)
    actor.args = args
    agent = RLAgent(env_train, actor, args)
    out_dir = MODEL_DIR / "deeptrader" / args_cli.market
    model_exists = (out_dir / "deeptrader.pth").exists()

    if not model_exists:
        out_dir.mkdir(parents=True, exist_ok=True)

        losses = []
        env_train.set_train()
        for ep in range(int(EPISODES)):
            loss = agent.train_episode()
            losses.append({"episode": ep, "loss": float(loss)})
            if (ep + 1) % 20 == 0 or ep == 0:
                print(f"[train] episode={ep+1}/{EPISODES} loss={loss:.6f}")

        agent.save(out_dir / "deeptrader.pth")
        test_df, test_metrics = evaluate_agent(env_eval, actor, mode="test", deterministic=True)

        print("\n=== DONE ===")
        print("Test Metrics:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.3f}")
    else:
        agent.load(out_dir / "deeptrader.pth")
        test_df, test_metrics = evaluate_agent(env_eval, actor, mode="test", deterministic=True)
        report_pf_metrics(args_cli.market, 'DeepTrader', test_metrics, header=not args_cli.no_header)

if __name__ == "__main__":
    main()
