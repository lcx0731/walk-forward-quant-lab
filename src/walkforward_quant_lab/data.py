"""Data loaders for deterministic experiments and optional public prices."""

from __future__ import annotations

from io import StringIO
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


def synthetic_prices(n: int = 1800, seed: int = 7) -> pd.DataFrame:
    """Generate a deterministic price path with calm, stress, and recovery regimes."""
    if n < 400:
        raise ValueError("n must be at least 400 for the walk-forward experiment")

    rng = np.random.default_rng(seed)
    cut_1, cut_2 = int(n * 0.45), int(n * 0.70)
    drift = np.r_[
        np.full(cut_1, 0.00035),
        np.full(cut_2 - cut_1, -0.00010),
        np.full(n - cut_2, 0.00045),
    ]
    volatility = np.r_[
        np.full(cut_1, 0.008),
        np.full(cut_2 - cut_1, 0.024),
        np.full(n - cut_2, 0.012),
    ]

    shocks = rng.normal(drift, volatility)
    stress_idx = rng.choice(np.arange(cut_1, cut_2), size=max(3, n // 180), replace=False)
    shocks[stress_idx] += rng.normal(-0.045, 0.015, size=len(stress_idx))
    close = 100.0 * np.exp(np.cumsum(shocks))

    intraday_width = np.abs(rng.normal(0.008, 0.003, n))
    high = close * (1.0 + intraday_width)
    low = close * (1.0 - intraday_width)
    index = pd.bdate_range("2018-01-02", periods=n)
    return pd.DataFrame({"close": close, "high": high, "low": low}, index=index)


def load_prices(symbol: str = "spy.us", timeout: int = 20) -> pd.DataFrame:
    """Download daily OHLC observations from Stooq."""
    query = urlencode({"s": symbol.lower(), "i": "d"})
    request = Request(
        f"https://stooq.com/q/d/l/?{query}",
        headers={"User-Agent": "walk-forward-quant-lab/0.1"},
    )
    with urlopen(request, timeout=timeout) as response:
        text = response.read().decode("utf-8")

    frame = pd.read_csv(StringIO(text), parse_dates=["Date"])
    required = {"Date", "Close", "High", "Low"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"Stooq returned no usable daily data for {symbol!r}")

    frame = frame.rename(columns=str.lower).set_index("date").sort_index()
    frame = frame[["close", "high", "low"]].astype(float)
    if len(frame) < 400:
        raise ValueError("at least 400 daily observations are required")
    return frame

