"""Feature engineering for day-ahead (24 h horizon) load forecasting.

All lag/rolling features are shifted by >= 24 hours, so every feature is
known at forecast time (the origin is the end of day D, targets are the 24
hours of day D+1). Calendar features are derived in local time
(Europe/Madrid) because human routines, and Spanish holidays, follow the
local clock, not UTC.

Feature groups
--------------
calendar : hour, day-of-week, month, weekend flag, Spanish holidays
           (via the `holidays` package), cyclical sin/cos encodings
lags     : t-24 h, t-48 h, t-168 h load
rolling  : 24 h and 168 h rolling means (+ 24 h rolling std), each shifted
           24 h so they only use information available day-ahead
weather  : city-averaged temperature and its square (captures the U-shaped
           heating/cooling response)

Run:  python -m src.features        (from the repo root)
"""

from __future__ import annotations

import sys
from pathlib import Path

import holidays as holidays_lib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import (
    FEATURES_FILE,
    LAG_HOURS,
    LOCAL_TZ,
    ROLL_WINDOWS,
    TARGET,
    ensure_dirs,
)
from src.data_ingestion import load_processed

CALENDAR_FEATURES = [
    "hour", "weekday", "month", "is_weekend", "is_holiday",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]
LAG_FEATURES = [f"lag_{h}h" for h in LAG_HOURS]
ROLL_FEATURES = [f"roll_mean_{w}h" for w in ROLL_WINDOWS] + ["roll_std_24h"]
WEATHER_FEATURES = ["temp", "temp_sq"]

FEATURES: list[str] = CALENDAR_FEATURES + LAG_FEATURES + ROLL_FEATURES + WEATHER_FEATURES


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    local = df.index.tz_convert(LOCAL_TZ)
    years = range(local.year.min(), local.year.max() + 2)
    es_holidays = holidays_lib.country_holidays("ES", years=years)
    holiday_dates = set(es_holidays.keys())

    df["hour"] = local.hour
    df["weekday"] = local.weekday
    df["month"] = local.month
    df["is_weekend"] = (local.weekday >= 5).astype(int)
    df["is_holiday"] = np.fromiter((d in holiday_dates for d in local.date), dtype=int)

    df["hour_sin"] = np.sin(2 * np.pi * local.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * local.hour / 24)
    doy = local.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def add_lags_and_rolls(df: pd.DataFrame) -> pd.DataFrame:
    y = df[TARGET]
    for h in LAG_HOURS:
        df[f"lag_{h}h"] = y.shift(h)
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}h"] = y.shift(24).rolling(w, min_periods=w).mean()
    df["roll_std_24h"] = y.shift(24).rolling(24, min_periods=24).std()
    return df


def add_weather(df: pd.DataFrame) -> pd.DataFrame:
    df["temp_sq"] = df["temp"] ** 2
    return df


def build_features(df: pd.DataFrame | None = None, dropna: bool = True) -> pd.DataFrame:
    """Return a frame with [TARGET] + FEATURES, indexed by UTC hour."""
    if df is None:
        df = load_processed()
    df = df.copy()
    df = add_calendar(df)
    df = add_lags_and_rolls(df)
    df = add_weather(df)
    out = df[[TARGET] + FEATURES]
    if dropna:
        out = out.dropna()
    return out


def load_features() -> pd.DataFrame:
    """Load the persisted feature table (build it if missing)."""
    if FEATURES_FILE.exists():
        df = pd.read_csv(FEATURES_FILE, index_col="time", parse_dates=["time"])
        df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
        return df
    return build_features()


def main() -> None:
    ensure_dirs()
    feats = build_features()
    feats.to_csv(FEATURES_FILE)
    print(f"Saved {FEATURES_FILE}  shape={feats.shape}")
    print(f"Features ({len(FEATURES)}): {FEATURES}")
    print(f"Rows dropped in lag warm-up: expected ~{max(LAG_HOURS) + 24}")


if __name__ == "__main__":
    main()
