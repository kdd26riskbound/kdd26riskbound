# Dataset Description

This directory contains preprocessed datasets used in the paper  
"RiskBound: Risk-Aware Boundary-Guided Portfolio Optimization via Action Space Reshaping"
(submitted to KDD 2026).

The preprocessed data is provided for review and reproducibility purposes only.

## Dataset
We evaluate RiskBound on three stock market datasets (U.S./CN/U.K.).

| Market | # Assets | Train Period | Validation Period | Test Period |
|---|---:|---|---|---|
| U.S. stock market | 28 | 2008.03-2018.09 | 2018.10-2019.09 | 2019.10-2025.09 |
| CN stock market | 35 | 2008.04-2018.09 | 2018.10-2019.09 | 2019.10-2025.09 |
| U.K. stock market | 39 | 2008.03-2018.09 | 2018.10-2019.09 | 2019.10-2025.09 |

## Data Sources

The original raw data is obtained from public sources:

- **U.S. / CN markets**: TradeMaster  
  https://github.com/TradeMaster-NTU/TradeMaster

- **U.K. market**: DTML  
  https://datalab.snu.ac.kr/dtml/#datasets

Following the original sources, we adapt the asset universe and extend the time horizon
to enable a more comprehensive evaluation.

## Preprocessing Pipeline

Only preprocessed data is included in this repository.  
Raw data files are **not** included.

The preprocessing pipeline is publicly available at:
```
src/riskbound/data/preprocess.py
```

### Raw Data Format

To reproduce the preprocessing, place the following CSV files under `raw_data/`:

- `{market}_index.csv`  
- `{market}_stocks.csv`

Expected columns:
- **Index file**: `date`, `adj_close`
- **Stock file**: `date`, `symbol`, `open`, `high`, `low`, `close`, `adj_close`

### Running Preprocessing

```bash
python src/riskbound/data/preprocess.py --market <market>
# e.g., python src/riskbound/data/preprocess.py --market uk
```

### Feature Construction
#### Market features

- `mkt_m5`, `mkt_m10`, `mkt_m20`, `mkt_m60`
  (price momentum over 5, 10, 20, 60 trading days)

#### Asset features

- r_open = open / close - 1
- r_high = high / close - 1
- r_low = low / close - 1
- r_close = close / prev_close - 1
- r_adj_close = adj_close / prev_adj_close - 1

#### Target (label):

- ror = next_adj_close / adj_close - 1

Only dates with complete observations across all assets and market features are retained.

### File Structure

The preprocessed data for each market is organized as follows:

```
data/
  us/
    asset_features.npy  # (N, T, F_s)
    market_features.npy # (T, F_m)
    ror_array.npy       # (N, T)
    dates.npy           # (T,)
    symbols.npy         # (N,)
    meta.json           # dataset metadata
  cn/
    ...
  uk/
    ...
```

### Metadata

The `meta.json` file contains following:

- Market name
- Number of assets
- Date range
- Feature dimensions
- Feature labels