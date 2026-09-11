"""Leakage-aware walk-forward quantitative research utilities."""

from .backtest import BacktestResult, run_backtest
from .data import load_prices, synthetic_prices
from .features import build_research_frame
from .model import walk_forward_variance

__all__ = [
    "BacktestResult",
    "build_research_frame",
    "load_prices",
    "run_backtest",
    "synthetic_prices",
    "walk_forward_variance",
]

