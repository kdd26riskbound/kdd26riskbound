import argparse
from riskbound.baselines.traditional import run_baseline
from riskbound.utils import report_pf_metrics

def main():
    p = argparse.ArgumentParser("Traditional baselines")
    p.add_argument("--market", type=str, default="us")
    p.add_argument("--no-header", action="store_true", help="Do not print header for metrics")
    args = p.parse_args()
    markets = [args.market] if args.market != "all" else ["us", "cn", "uk"]
    baselines = ["mean_variance", "risk_parity", "inverse_vol"]

    for m in markets:
        for i, b in enumerate(baselines):
            met, extra = run_baseline(
                market=m,
                baseline=b,
            )
            report_pf_metrics(m, b, met, header=(i == 0 and not args.no_header))

if __name__ == "__main__":
    main()
