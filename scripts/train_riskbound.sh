#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd)/src
market=$1

if [ -z "${market}" ]; then
  echo "Usage: $0 {us|cn|uk}"
  exit 1
fi

echo "RiskBound Training for market: ${market}"
echo "[1/2] Pretraining RABG..."
python -m riskbound.cmd.pretrain_rabg --market "${market}"
echo "[2/2] Training RL with RiskBound..."
python -m riskbound.cmd.train_rl --market "${market}"
echo "RiskBound Training completed for market: ${market}"