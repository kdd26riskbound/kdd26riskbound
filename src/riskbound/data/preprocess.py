import argparse
import pandas as pd
import numpy as np
import json

from riskbound.settings import DATA_DIR, RAW_DATA_DIR

# momentum features for market index
MARKET_FEATURE_LABELS = ["mkt_m5", "mkt_m10", "mkt_m20", "mkt_m60"]

# price ratio based features for individual assets
ASSET_FEATURE_LABELS = ["r_open", "r_high", "r_low", "r_close", "r_adj_close"]


def process_market_data(index_path):
    df = pd.read_csv(index_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')

    feature_dict = {}
    price = df['adj_close']

    for d in [5, 10, 20, 60]:
        feature_dict[f'mkt_m{d}'] = price.pct_change(d)

    df_features = pd.DataFrame(feature_dict)

    df_features = df_features[MARKET_FEATURE_LABELS]
    return df_features.dropna()


def process_asset_data(stocks_path):
    df = pd.read_csv(stocks_path)
    df['date'] = pd.to_datetime(df['date'])

    df_pivot = df.pivot(index='date', columns='symbol')
    df_pivot = df_pivot.dropna(axis=0, how='any')

    opens = df_pivot['open']
    highs = df_pivot['high']
    lows = df_pivot['low']
    closes = df_pivot['close']
    adj_closes = df_pivot['adj_close']

    ror = adj_closes.pct_change().shift(-1)

    r_open = (opens / closes) - 1.0
    r_high = (highs / closes) - 1.0
    r_low = (lows / closes) - 1.0

    r_close = closes.pct_change()
    r_adj = adj_closes.pct_change()

    feat_map = {
        'r_open': r_open,
        'r_high': r_high,
        'r_low': r_low,
        'r_close': r_close,
        'r_adj_close': r_adj
    }

    return feat_map, ror, df_pivot.columns.get_level_values(1).unique()


def main():
    p = argparse.ArgumentParser("Data Preprocessing")
    p.add_argument("--market", type=str, default="uk", help="Market name (e.g., us, uk, cn)")
    args = p.parse_args()

    index_file = RAW_DATA_DIR / f"{args.market}_index.csv"
    stocks_file = RAW_DATA_DIR / f"{args.market}_stocks.csv"

    mkt_df = process_market_data(index_file)
    asset_feats, ror_df, symbols = process_asset_data(stocks_file)

    common_dates = mkt_df.index.intersection(ror_df.index)

    valid_mask = asset_feats['r_adj_close'].loc[common_dates].notna().all(axis=1)
    common_dates = common_dates[valid_mask]

    valid_mask_mkt = mkt_df.loc[common_dates].notna().all(axis=1)
    common_dates = common_dates[valid_mask_mkt]

    valid_mask_ror = ror_df.loc[common_dates].notna().all(axis=1)
    common_dates = common_dates[valid_mask_ror]

    final_dates = common_dates.strftime('%Y-%m-%d').to_numpy()
    final_symbols = symbols.to_numpy()

    market_features = mkt_df.loc[common_dates].values.astype(np.float32)

    ror_array = ror_df.loc[common_dates].values.T.astype(np.float32)

    feat_list = [asset_feats[label] for label in ASSET_FEATURE_LABELS]
    asset_arrays = [f.loc[common_dates].values for f in feat_list]

    temp_asset = np.dstack(asset_arrays)

    asset_features = temp_asset.transpose(1, 0, 2).astype(np.float32)

    out_dir = DATA_DIR / args.market
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "dates.npy", final_dates)
    np.save(out_dir / "symbols.npy", final_symbols)
    np.save(out_dir / "market_features.npy", market_features)
    np.save(out_dir / "asset_features.npy", asset_features)
    np.save(out_dir / "ror_array.npy", ror_array)

    meta = {
        "market": args.market,
        "n_dates": int(len(final_dates)),
        "n_symbols": int(len(final_symbols)),
        "date_from": str(final_dates[0]),
        "date_to": str(final_dates[-1]),
        "shapes": {
            "dates": list(final_dates.shape),
            "market_features": list(market_features.shape),
            "asset_features": list(asset_features.shape),
            "ror_array": list(ror_array.shape),
        },
        "market_feature_labels": MARKET_FEATURE_LABELS,
        "asset_feature_labels": ASSET_FEATURE_LABELS,
        "description": "Ratio-based features. Shapes: Asset(N,T,F), Market(T,F), ROR(N,T)."
    }

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    print(f"Done. Saved to {out_dir}")


if __name__ == "__main__":
    main()