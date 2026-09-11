"""Walk-forward variance forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS


def walk_forward_variance(
    frame: pd.DataFrame,
    min_train: int = 252,
    train_window: int = 756,
    refit_every: int = 20,
    alpha: float = 10.0,
) -> pd.Series:
    """Forecast annualized volatility using only outcomes known before each row."""
    if min_train < 60:
        raise ValueError("min_train must be at least 60")
    if train_window < min_train:
        raise ValueError("train_window must be greater than or equal to min_train")
    if refit_every < 1:
        raise ValueError("refit_every must be positive")

    x = frame[FEATURE_COLUMNS]
    y = frame["target_log_variance"]
    predictions = pd.Series(np.nan, index=frame.index, name="predicted_vol")

    for block_start in range(min_train, len(frame), refit_every):
        # At this close, y[block_start - 1] is observable. The fitted model is
        # frozen until the next scheduled refit, so predicting a block at once
        # is both faster and identical to daily inference with the same model.
        train_start = max(0, block_start - train_window)
        block_end = min(block_start + refit_every, len(frame))
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(x.iloc[train_start:block_start], y.iloc[train_start:block_start])
        log_daily_variance = model.predict(x.iloc[block_start:block_end])
        predictions.iloc[block_start:block_end] = np.sqrt(
            np.exp(log_daily_variance) * 252
        )

    return predictions.clip(lower=0.03, upper=2.0)
