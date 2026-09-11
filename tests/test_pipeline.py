import numpy as np
import pandas as pd

from walkforward_quant_lab.backtest import run_backtest
from walkforward_quant_lab.data import synthetic_prices
from walkforward_quant_lab.features import FEATURE_COLUMNS, build_research_frame
from walkforward_quant_lab.model import walk_forward_variance


def test_features_do_not_change_when_only_future_price_changes():
    prices = synthetic_prices(n=500)
    cutoff = prices.index[300]
    original = build_research_frame(prices)

    changed = prices.copy()
    changed.loc[changed.index > cutoff, ["close", "high", "low"]] *= 1.8
    revised = build_research_frame(changed)

    pd.testing.assert_frame_equal(
        original.loc[:cutoff, FEATURE_COLUMNS],
        revised.loc[:cutoff, FEATURE_COLUMNS],
    )


def test_target_is_next_close_return():
    prices = synthetic_prices(n=500)
    frame = build_research_frame(prices)
    date = frame.index[25]
    location = prices.index.get_loc(date)
    expected = prices["close"].iloc[location + 1] / prices["close"].iloc[location] - 1.0
    assert np.isclose(frame.loc[date, "next_return"], expected)


def test_walk_forward_output_starts_after_minimum_training_window():
    frame = build_research_frame(synthetic_prices(n=520))
    forecast = walk_forward_variance(frame, min_train=120, train_window=240, refit_every=10)
    assert forecast.iloc[:120].isna().all()
    assert forecast.iloc[120:].notna().all()
    assert forecast.iloc[120:].between(0.03, 2.0).all()


def test_costs_cannot_improve_same_positions():
    frame = build_research_frame(synthetic_prices(n=520))
    forecast = walk_forward_variance(frame, min_train=120, train_window=240, refit_every=20)
    free = run_backtest(frame, forecast, cost_bps=0.0)
    costly = run_backtest(frame, forecast, cost_bps=10.0)
    difference = free.returns - costly.returns
    assert (difference >= -1e-15).all().all()
    assert difference.to_numpy().sum() > 0

