"""Produce the next-24h day-ahead forecast with a P10-P90 band.

Trains the LightGBM point + quantile models on *all* available history,
builds the feature matrix for the 24 hours after the last observation and
predicts. Future temperature uses 24 h persistence (the value observed at
the same hour one day earlier) - in production this would be replaced by a
numerical weather forecast.

Outputs
-------
reports/forecast_next24h.csv
reports/figures/forecast_next24h.png

Run:  python -m src.forecast        (from the repo root)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import FIGURES, REPORTS, TARGET, ensure_dirs
from src.data_ingestion import load_processed
from src.features import FEATURES, build_features
from src.models.lgbm_model import predict_quantiles, save, train_point, train_quantiles

FORECAST_CSV = REPORTS / "forecast_next24h.csv"
FORECAST_PNG = FIGURES / "forecast_next24h.png"


def extend_with_future(df: pd.DataFrame, horizon: int = 24) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Append `horizon` future rows; weather = 24 h persistence."""
    future_index = pd.date_range(df.index.max() + pd.Timedelta(hours=1),
                                 periods=horizon, freq="h", tz="UTC")
    future = pd.DataFrame(index=future_index, columns=df.columns, dtype=float)
    weather_cols = [c for c in df.columns if c != TARGET]
    persisted = df.loc[future_index - pd.Timedelta(hours=24), weather_cols]
    future[weather_cols] = persisted.to_numpy()
    return pd.concat([df, future]), future_index


def main() -> None:
    ensure_dirs()
    df = load_processed()

    extended, future_index = extend_with_future(df)
    feats = build_features(extended, dropna=False)
    X_future = feats.loc[future_index, FEATURES]
    assert not X_future.isna().any().any(), "future feature matrix contains NaNs"

    print("Training LightGBM point + quantile models on full history ...")
    train_feats = feats.dropna()
    X, y = train_feats[FEATURES], train_feats[TARGET]
    point = train_point(X, y)
    quantiles = train_quantiles(X, y)
    save(point, quantiles, str(X.index.max()))

    result = predict_quantiles(quantiles, X_future)
    result["point"] = point.predict(X_future)
    result["point"] = result[["p10", "point", "p90"]].apply(
        lambda r: min(max(r["point"], r["p10"]), r["p90"]), axis=1)
    result.index.name = "time"
    result.to_csv(FORECAST_CSV)

    # ---------------------------------------------------------------- plot ----
    history = df[TARGET].iloc[-7 * 24 :]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(history.index, history.values, color="black", lw=1.4, label="Observed load")
    ax.fill_between(result.index, result["p10"], result["p90"], color="tab:blue",
                    alpha=0.25, label="P10-P90 interval")
    ax.plot(result.index, result["point"], color="tab:blue", lw=2.2, label="Point forecast")
    ax.axvline(df.index.max(), color="grey", ls="--", lw=1)
    ax.set(title="Next-24h day-ahead load forecast", ylabel="Load (MW)")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.savefig(FORECAST_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(result.round(0).to_string())
    print(f"Saved {FORECAST_CSV}")
    print(f"Saved {FORECAST_PNG}")


if __name__ == "__main__":
    main()
