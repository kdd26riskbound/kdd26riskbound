# RiskBound: Risk-Aware Boundary-Guided Portfolio Optimization via Action Space Reshaping

Implementation of the paper "RiskBound: Risk-Aware Boundary-Guided Portfolio Optimization via Action Space Reshaping" (submitted to KDD 2026).

## Prerequisites

### System
- Linux (tested on Ubuntu 20.04)
- NVIDIA GPU with CUDA support (tested on RTX 3090)

### Software
- Python 3.11
- PyTorch 2.5.1
- CUDA 12.1 (toolkit) + NVIDIA driver compatible with CUDA 12.x

> Note: Please install a PyTorch build that matches your CUDA version.

### Python Packages
Install required packages via pip:
```bash
pip install -r requirements.txt
```

## Dataset
We evaluate RiskBound on three stock market datasets (U.S./CN/U.K.).

| Dataset | # Assets | Train Period    | Validation Period | Test Period     | Source          |
|---|---:|-----------------|-------------------|-----------------|-----------------|
| U.S. stock market | 28 | 2008.03-2018.09 | 2018.10-2019.09   | 2019.10-2025.09 | [TradeMaster](https://github.com/TradeMaster-NTU/TradeMaster) |
| CN stock market | 35 | 2008.04-2018.09 | 2018.10-2019.09   | 2019.10-2025.09 | [TradeMaster](https://github.com/TradeMaster-NTU/TradeMaster) |
| U.K. stock market | 39 | 2008.03-2018.09 | 2018.10-2019.09   | 2019.10-2025.09 | [DTML](https://datalab.snu.ac.kr/dtml/#datasets)        |

> Note: Following the original data sources, we adapt the asset universe and extend the time horizon for a more comprehensive evaluation.

Preprocessed data is provided under `data/`. See `data/README.md` for details.

## Reallocation Protocol
- Lookback window: L=40 trading days
- Reallocation horizon: H=20 trading days
- Transaction cost: 0.1%

## Training RL with RiskBound
Run the following script to train RL agents with RiskBound on a specified market:
```bash
bash scripts/train_riskbound.sh <market>
# e.g., bash scripts/train_riskbound.sh us
# supported markets: us, cn, uk
```

## Evaluating RL with RiskBound
Run the following script to evaluate RL agents with RiskBound on a specified market:
```bash
bash scripts/evaluate_riskbound.sh <market>
# e.g., bash scripts/evaluate_riskbound.sh us
```

## Reproducing Results
To reproduce the results in the paper, we provide saved model weights in `models/`.
Run the following script:
```bash
bash scripts/reproduce_results.sh
```
