#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd)/src

echo "Reproducing all results for US, CN, and UK markets..."

echo "Portfolio Performance:"

for market in us cn uk; do
  python -m riskbound.cmd.run_traditional --market "${market}"
  python -m riskbound.cmd.run_deeptrader --market "${market}" --no-header
  python -m riskbound.cmd.run_metatrader --market "${market}" --no-header
  python -m riskbound.cmd.test_rl --market "${market}" --no-header
  echo ""
done

echo "Risk Scoring Performance:"

for market in us cn uk; do
  python -m riskbound.cmd.test_rabg --market "${market}"
  echo ""
done