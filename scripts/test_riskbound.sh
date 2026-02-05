#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd)/src
market=$1

if [ -z "${market}" ]; then
  echo "Usage: $0 {us|cn|uk}"
  exit 1
fi

echo "RiskBound Test for market: ${market}"
echo "[1/2] RiskScorer test"
python -m riskbound.cmd.test_rabg --market "${market}"
echo "[2/2] RiskBound out-of-sample test:"
python -m riskbound.cmd.test_rl --market "${market}"