"""Feature construction with explicit close-to-next-close alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "return_1",
    "momentum_5",
    "momentum_20",
    "realized_vol_5",
    "realized_vol_20",
    "downside_vol_20",
    "range_5",
]


def build_research_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Build end-of-day features and a target realized after that close."""
    required = {"close", "high", "low"}
    if not required.issubset(prices.columns):
        raise ValueError(f"prices must include {sorted(required)}")
    if not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise ValueError("price index must be unique and increasing")

    close = prices["close"].astype(float)
    returns = close.pct_change()
    downside = returns.clip(upper=0.0)
    log_range = np.log(prices["high"].astype(float) / prices["low"].astype(float))

    frame = pd.DataFrame(index=prices.index)
    frame["return_1"] = returns
    frame["momentum_5"] = close.pct_change(5)
    frame["momentum_20"] = close.pct_change(20)
    frame["realized_vol_5"] = returns.rolling(5).std(ddof=0) * np.sqrt(252)
    frame["realized_vol_20"] = returns.rolling(20).std(ddof=0) * np.sqrt(252)
    frame["downside_vol_20"] = downside.rolling(20).std(ddof=0) * np.sqrt(252)
    frame["range_5"] = log_range.rolling(5).mean() * np.sqrt(252)

    # The feature row at t predicts the return observed from close t to close t+1.
    frame["next_return"] = returns.shift(-1)
    frame["target_log_variance"] = np.log(frame["next_return"].pow(2) + 1e-8)
    return frame.dropna().copy()

