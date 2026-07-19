"""Baseline forecasters - the benchmarks every real model must beat.

* naive-24h        : tomorrow's hour h = today's hour h  (lag 24 h)
* seasonal naive-168h : tomorrow's hour h = same hour last week (lag 168 h)

Both are valid day-ahead forecasts: at the origin (end of day D) every value
they reference is already observed.

Run:  python -m src.models.baselines        (quick sanity check)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import TARGET
from src.data_ingestion import load_processed


def naive_24h(y: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Predict y(t) = y(t - 24h) for the requested timestamps."""
    return y.shift(24).reindex(index)


def seasonal_naive_168h(y: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Predict y(t) = y(t - 168h) for the requested timestamps."""
    return y.shift(168).reindex(index)


def main() -> None:
    df = load_processed()
    y = df[TARGET]
    test = y.index[-14 * 24:]
    for name, fn in [("naive-24h", naive_24h), ("seasonal-naive-168h", seasonal_naive_168h)]:
        pred = fn(y, test)
        mask = pred.notna()
        mape = float(np.mean(np.abs((y[test][mask] - pred[mask]) / y[test][mask]))) * 100
        print(f"{name:>20s} | MAPE over last 14 days: {mape:.2f}%")


if __name__ == "__main__":
    main()
