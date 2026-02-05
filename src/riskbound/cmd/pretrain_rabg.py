from __future__ import annotations

import argparse
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from riskbound.settings import HOLDING_PERIOD, MODEL_DIR

from riskbound.core.rabg import RiskScorer


from riskbound.data.dataset import load_data
from riskbound.data.source import DataSource
from riskbound.utils import set_seed

from riskbound.metrics import pearson_series, rankic_series, mean_ir


def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        opt: torch.optim.Optimizer,
        *,
        device: str,
        lambda_corr: float,
) -> Dict[str, float]:
    model.train()

    loss_sum = 0.0
    n = 0

    for Xs, Xm, y in loader:
        Xs = Xs.to(device, non_blocking=True)
        Xm = Xm.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(Xs, Xm)

        loss_reg = F.smooth_l1_loss(pred, y)

        corr = pearson_series(pred, y).mean()
        loss = loss_reg - lambda_corr * corr

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            loss_sum += loss.item()
            n += 1

    return {
        "train_loss": loss_sum / max(1, n),
    }

@torch.no_grad()
def evaluate(
        model: nn.Module,
        loader: DataLoader,
        *,
        device: str,
) -> Dict[str, float]:
    model.eval()
    ic_all, ric_all = [], []
    preds_all, y_all = [], []

    for Xs, Xm, y in loader:
        Xs = Xs.to(device, non_blocking=True)
        Xm = Xm.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(Xs, Xm)

        ic_all.append(pearson_series(pred, y).cpu())
        ric_all.append(rankic_series(pred, y).cpu())

        preds_all.append(pred.cpu())
        y_all.append(y.cpu())

    if len(ic_all) > 0:
        ic = torch.cat(ic_all, dim=0)
        ric = torch.cat(ric_all, dim=0)
        IC, ICIR = mean_ir(ic)
        RIC, RICIR = mean_ir(ric)
    else:
        IC, ICIR, RIC, RICIR = 0.0, 0.0, 0.0, 0.0

    return {
        "IC": IC,
        "ICIR": ICIR,
        "RankIC": RIC,
        "RankICIR": RICIR,
    }

def label_transform_downvol(Y_future: torch.Tensor) -> torch.Tensor:
    neg = torch.clamp(Y_future, max=0.0)
    return torch.sqrt((neg * neg).mean(dim=-1) + 1e-8)

@torch.no_grad()
def create_rabg_dataset(src: DataSource, stride: int, device: str):
    Xs, Xm, Yf = src.dataset(stride=stride, device=device)

    y = label_transform_downvol(Yf)

    return Xs, Xm, y

def create_data_loader(
        Xs: torch.Tensor,
        Xm: torch.Tensor,
        y: torch.Tensor,
        *,
        batch_size: int,
        shuffle: bool,
        device: str,
) -> DataLoader:
    ds = TensorDataset(Xs, Xm, y)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device == "cuda" or device.startswith("cuda")),
    )

def main():
    ap = argparse.ArgumentParser("Pretrain Risk-Aware Boundary Generator")
    ap.add_argument("--market", type=str, default="us")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))

    ap.add_argument("--epochs", type=int, default=30)

    ap.add_argument("--lambda_corr", type=float, default=0.1)

    args = ap.parse_args()

    set_seed(args.seed)

    market_dates, market_features, asset_symbols, asset_features, rate_of_returns = load_data(market=args.market)

    train_ds = DataSource(
        asset_features=asset_features,
        market_features=market_features,
        rate_of_returns=rate_of_returns,
        dates=market_dates,
        seed=args.seed,
    )
    val_ds = DataSource(
        asset_features=asset_features,
        market_features=market_features,
        rate_of_returns=rate_of_returns,
        dates=market_dates,
        seed=args.seed,
    )
    test_ds = DataSource(
        asset_features=asset_features,
        market_features=market_features,
        rate_of_returns=rate_of_returns,
        dates=market_dates,
        seed=args.seed,
    )

    train_ds.set_mode("train")
    val_ds.set_mode("val")
    test_ds.set_mode("test")

    Xs_tr, Xm_tr, y_tr = create_rabg_dataset(train_ds, stride=1, device="cpu")
    Xs_va, Xm_va, y_va = create_rabg_dataset(val_ds, stride=HOLDING_PERIOD, device="cpu")
    Xs_te, Xm_te, y_te = create_rabg_dataset(test_ds, stride=HOLDING_PERIOD, device="cpu")

    train_loader = create_data_loader(Xs_tr, Xm_tr, y_tr, batch_size=64, shuffle=True, device=args.device)
    val_loader = create_data_loader(Xs_va, Xm_va, y_va, batch_size=64, shuffle=False, device=args.device)
    test_loader = create_data_loader(Xs_te, Xm_te, y_te, batch_size=64, shuffle=False, device=args.device)

    asset_in = Xs_tr.shape[-1]
    market_in = Xm_tr.shape[-1]

    model_cfg = {
        "asset_in": asset_in,
        "market_in": market_in,
    }
    model = RiskScorer(**model_cfg).to(args.device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_combined_score = -1e9
    best_state = None

    for epoch in range(1, args.epochs + 1):
        lam = args.lambda_corr

        tr = train_one_epoch(model, train_loader, opt, device=args.device, lambda_corr=lam)
        va = evaluate(model, val_loader, device=args.device)

        print(
            f"[train] epoch={epoch}/{args.epochs} loss={tr['train_loss']:.6f}"
        )

        combined_val_score = va["RankIC"] + va["IC"]
        if combined_val_score > best_combined_score:
            best_combined_score = combined_val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    te = evaluate(model, test_loader, device=args.device)

    print("=== DONE ===")
    for key, value in te.items():
        print(f"Test {key}: {value:.3f}")

    save_path = MODEL_DIR / "riskbound" / args.market / "rabg" / f"risk_scorer.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(save_path)


if __name__ == "__main__":
    main()