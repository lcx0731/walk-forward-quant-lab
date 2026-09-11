"""Risk-scaled backtesting and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    returns: pd.DataFrame
    positions: pd.DataFrame
    metrics: pd.DataFrame


def _metrics(returns: pd.Series, position: pd.Series) -> dict[str, float]:
    clean = returns.dropna()
    equity = (1.0 + clean).cumprod()
    years = max(len(clean) / 252.0, 1.0 / 252.0)
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0
    volatility = clean.std(ddof=0) * np.sqrt(252)
    sharpe = clean.mean() / clean.std(ddof=0) * np.sqrt(252) if volatility > 0 else np.nan
    drawdown = equity / equity.cummax() - 1.0
    turnover = position.diff().abs().mean() * 252
    return {
        "cagr": float(cagr),
        "annualized_volatility": float(volatility),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "annualized_turnover": float(turnover),
    }


def run_backtest(
    frame: pd.DataFrame,
    predicted_vol: pd.Series,
    target_vol: float = 0.12,
    leverage_cap: float = 1.5,
    cost_bps: float = 5.0,
) -> BacktestResult:
    """Compare risk-scaled momentum with the same signal at fixed exposure."""
    if target_vol <= 0 or leverage_cap <= 0 or cost_bps < 0:
        raise ValueError("target_vol and leverage_cap must be positive; cost_bps non-negative")

    common = frame.join(predicted_vol).dropna()
    direction = np.sign(common["momentum_20"]).replace(0.0, np.nan).ffill().fillna(0.0)
    scaled_position = direction * (target_vol / common["predicted_vol"]).clip(0.0, leverage_cap)
    baseline_position = direction
    positions = pd.DataFrame({"risk_scaled": scaled_position, "baseline": baseline_position})

    cost_rate = cost_bps / 10_000.0
    strategy_returns = positions.mul(common["next_return"], axis=0)
    strategy_returns -= positions.diff().abs().fillna(positions.abs()) * cost_rate

    metrics = pd.DataFrame(
        {
            column: _metrics(strategy_returns[column], positions[column])
            for column in strategy_returns.columns
        }
    ).T
    return BacktestResult(strategy_returns, positions, metrics)


def save_equity_chart(result: BacktestResult, path: Path) -> None:
    """Save a compact cumulative-growth comparison."""
    path.parent.mkdir(parents=True, exist_ok=True)
    equity = (1.0 + result.returns).cumprod()
    fig, ax = plt.subplots(figsize=(9, 5))
    equity.rename(columns={"risk_scaled": "Risk-scaled", "baseline": "Fixed exposure"}).plot(
        ax=ax,
        linewidth=1.8,
    )
    ax.set_title("Walk-forward momentum: growth of $1 (synthetic regimes)")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

