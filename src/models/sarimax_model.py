"""SARIMAX on daily-aggregated load with temperature as exogenous regressor.

The classical-statistics benchmark. Hourly load has three interacting
seasonalities (24 h / 168 h / annual) which plain SARIMA handles poorly, so
we model *daily mean load* with weekly seasonality (s=7) and daily mean
temperature (+ its square) as exogenous drivers of the annual/weather
component.

Order selection: small grid over (p,d,q)x(P,D,Q,7) ranked by AIC on the
training window; the full table is written to
``reports/sarimax_order_selection.csv``.

To compare against hourly models, the daily-mean forecast is disaggregated
to hourly using the average hour-of-week profile estimated on training data
(each week-hour weight = mean multiplicative deviation from the daily mean).

Run:  python -m src.models.sarimax_model        (fit + AIC table)
"""

from __future__ import annotations

import json
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import LOCAL_TZ, REPORTS, TARGET, TEST_WEEKS, ensure_dirs
from src.data_ingestion import load_processed

warnings.filterwarnings("ignore")

ORDER_FILE = REPORTS / "sarimax_order.json"

P_RANGE = (0, 1, 2)
Q_RANGE = (0, 1, 2)
D = 1
SEASONAL_CANDIDATES = [(1, 1, 1, 7), (0, 1, 1, 7)]
SELECTION_WINDOW_DAYS = 365  # AIC selection on the most recent training year (speed)


def make_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Daily mean load + exogenous temperature terms (naive tz-free index)."""
    daily = df[[TARGET, "temp"]].resample("D").mean().dropna()
    daily["temp_sq"] = (daily["temp"] - 17.0) ** 2  # centred square = U-shape
    daily.index = daily.index.tz_localize(None)
    daily.index.freq = pd.infer_freq(daily.index)
    return daily


def exog_matrix(daily: pd.DataFrame) -> pd.DataFrame:
    return daily[["temp", "temp_sq"]]


def select_order(y: pd.Series, exog: pd.DataFrame) -> tuple[tuple, tuple]:
    """Grid-search (p,1,q)x(P,D,Q,7) by AIC; persist the ranked table.

    Selection uses the most recent SELECTION_WINDOW_DAYS of the training
    window (AIC comparisons are valid on a common sample; the final model is
    refit on the full training data).
    """
    y, exog = y.iloc[-SELECTION_WINDOW_DAYS:], exog.iloc[-SELECTION_WINDOW_DAYS:]
    rows = []
    for (p, q), seasonal in product(product(P_RANGE, Q_RANGE), SEASONAL_CANDIDATES):
        order = (p, D, q)
        try:
            res = SARIMAX(
                y, exog=exog, order=order, seasonal_order=seasonal,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False, maxiter=80)
            rows.append({"order": str(order), "seasonal_order": str(seasonal), "aic": res.aic})
        except Exception as exc:  # noqa: BLE001 - log-and-skip is intended
            rows.append({"order": str(order), "seasonal_order": str(seasonal),
                         "aic": np.nan, "error": str(exc)[:80]})

    table = pd.DataFrame(rows).sort_values("aic")
    table.to_csv(REPORTS / "sarimax_order_selection.csv", index=False)
    best = table.dropna(subset=["aic"]).iloc[0]
    best_order = eval(best["order"])  # noqa: S307 - our own serialised tuples
    best_seasonal = eval(best["seasonal_order"])  # noqa: S307
    ORDER_FILE.write_text(json.dumps(
        {"order": best_order, "seasonal_order": best_seasonal, "aic": float(best["aic"])}
    ))
    return best_order, best_seasonal


def load_or_select_order(y: pd.Series, exog: pd.DataFrame) -> tuple[tuple, tuple]:
    if ORDER_FILE.exists():
        try:
            saved = json.loads(ORDER_FILE.read_text())
            return tuple(saved["order"]), tuple(saved["seasonal_order"])
        except (json.JSONDecodeError, KeyError):
            pass  # stale/corrupt cache -> re-select
    return select_order(y, exog)


def fit(y: pd.Series, exog: pd.DataFrame, order: tuple, seasonal: tuple):
    return SARIMAX(
        y, exog=exog, order=order, seasonal_order=seasonal,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False, maxiter=200)


def hour_of_week_profile(df: pd.DataFrame) -> pd.Series:
    """Multiplicative hour-of-week weights (mean 1) to disaggregate daily means."""
    local = df.index.tz_convert(LOCAL_TZ)
    frame = pd.DataFrame({
        "y": df[TARGET].to_numpy(),
        "how": local.weekday * 24 + local.hour,
        "date": df.index.date,  # UTC day, matching the daily aggregation
    })
    daily_mean = frame.groupby("date")["y"].transform("mean")
    frame["ratio"] = frame["y"] / daily_mean
    return frame.groupby("how")["ratio"].mean()


def disaggregate(daily_pred: float, day_index: pd.DatetimeIndex, profile: pd.Series) -> pd.Series:
    """Turn one daily-mean prediction into 24 hourly values via the profile."""
    local = day_index.tz_convert(LOCAL_TZ)
    how = local.weekday * 24 + local.hour
    weights = profile.reindex(how).to_numpy()
    weights = np.where(np.isnan(weights), 1.0, weights)
    return pd.Series(daily_pred * weights, index=day_index)


def main() -> None:
    ensure_dirs()
    df = load_processed()
    daily = make_daily(df)

    train = daily.iloc[: -TEST_WEEKS * 7]  # keep the backtest window unseen
    order, seasonal = select_order(train[TARGET], exog_matrix(train))
    res = fit(train[TARGET], exog_matrix(train), order, seasonal)

    print(f"Selected order {order} x {seasonal}  (AIC={res.aic:.1f})")
    print(f"Full AIC table: {REPORTS / 'sarimax_order_selection.csv'}")
    print(res.summary().tables[1])


if __name__ == "__main__":
    main()
