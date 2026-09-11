"""Command-line entry point for the complete experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backtest import run_backtest, save_equity_chart
from .data import load_prices, synthetic_prices
from .features import build_research_frame
from .model import walk_forward_variance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["synthetic", "stooq"], default="synthetic")
    parser.add_argument("--symbol", default="spy.us", help="Stooq symbol when source=stooq")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--target-vol", type=float, default=0.12)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prices = synthetic_prices() if args.source == "synthetic" else load_prices(args.symbol)
    frame = build_research_frame(prices)
    forecast = walk_forward_variance(frame)
    result = run_backtest(
        frame,
        forecast,
        target_vol=args.target_vol,
        cost_bps=args.cost_bps,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    save_equity_chart(result, args.output / "equity_curve.png")
    metrics = {
        strategy: {metric: round(float(value), 6) for metric, value in row.items()}
        for strategy, row in result.metrics.iterrows()
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(result.metrics.round(3).to_string())

