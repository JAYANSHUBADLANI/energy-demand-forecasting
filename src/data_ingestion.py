"""Ingestion + cleaning: raw Kaggle CSVs -> one tidy hourly table.

Steps
-----
1. (Optional) download the dataset via the Kaggle API if credentials exist,
   otherwise expect ``energy_dataset.csv`` and ``weather_features.csv`` in
   ``data/raw/`` (placed manually).
2. Energy file: parse tz-aware timestamps to UTC, drop duplicate hours,
   reindex to a complete hourly range, interpolate short gaps in
   ``total load actual`` (counts are logged and written to
   ``reports/data_quality.json``).
3. Weather file: strip whitespace from city names, de-duplicate
   (city, timestamp) rows, convert Kelvin -> °C, average the five cities
   into a single national weather signal.
4. Merge on the UTC timestamp and save ``data/processed/energy_weather_hourly.csv``.

Run:  python -m src.data_ingestion        (from the repo root)
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import (
    DATA_RAW,
    ENERGY_CSV,
    KAGGLE_DATASET,
    PROCESSED_FILE,
    REPORTS,
    TARGET,
    WEATHER_CSV,
    WEATHER_NUMERIC,
    ensure_dirs,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

MAX_GAP_HOURS = 24  # longest gap we are willing to interpolate


def maybe_download_from_kaggle() -> None:
    """Download the dataset with the Kaggle CLI if files are absent and credentials exist."""
    energy_path = DATA_RAW / ENERGY_CSV
    weather_path = DATA_RAW / WEATHER_CSV
    if energy_path.exists() and weather_path.exists():
        return

    has_creds = (Path.home() / ".kaggle" / "kaggle.json").exists()
    if not has_creds:
        raise FileNotFoundError(
            f"Expected {energy_path} and {weather_path}.\n"
            "Download 'Hourly energy demand generation and weather' from Kaggle "
            f"({KAGGLE_DATASET}) and place both CSVs in data/raw/, or configure "
            "Kaggle API credentials (~/.kaggle/kaggle.json). For a quick smoke "
            "test without the real data run:  python -m src.synthetic_data"
        )

    log.info("Downloading %s via Kaggle API ...", KAGGLE_DATASET)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
         "-p", str(DATA_RAW), "--unzip"],
        check=True,
    )


def load_energy() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(DATA_RAW / ENERGY_CSV)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time")

    n_dupes = int(df.duplicated(subset="time").sum())
    df = df.drop_duplicates(subset="time", keep="first").set_index("time")

    keep = [c for c in (TARGET, "total load forecast", "price actual") if c in df.columns]
    df = df[keep].apply(pd.to_numeric, errors="coerce")

    quality = {"energy_duplicate_timestamps_dropped": n_dupes}
    return df, quality


def load_weather() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(DATA_RAW / WEATHER_CSV)
    df["city_name"] = df["city_name"].str.strip()
    df["dt_iso"] = pd.to_datetime(df["dt_iso"], utc=True)

    n_dupes = int(df.duplicated(subset=["city_name", "dt_iso"]).sum())
    df = df.drop_duplicates(subset=["city_name", "dt_iso"], keep="first")

    cols = [c for c in WEATHER_NUMERIC if c in df.columns]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

    # Kelvin -> Celsius (auto-detect so already-converted files pass through)
    if df["temp"].median() > 100:
        df["temp"] = df["temp"] - 273.15

    weather = df.groupby("dt_iso")[cols].mean()  # average over the 5 cities
    weather.index.name = "time"

    quality = {
        "weather_duplicate_rows_dropped": n_dupes,
        "weather_cities": sorted(df["city_name"].unique().tolist()),
    }
    return weather, quality


def clean_and_merge(energy: pd.DataFrame, weather: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    full_range = pd.date_range(energy.index.min(), energy.index.max(), freq="h", tz="UTC")

    n_missing_hours = int(len(full_range) - len(energy.index))
    energy = energy.reindex(full_range)
    n_nan_target = int(energy[TARGET].isna().sum())  # missing rows + native NaNs

    energy[TARGET] = energy[TARGET].interpolate(method="time", limit=MAX_GAP_HOURS)
    for col in energy.columns.drop(TARGET):
        energy[col] = energy[col].interpolate(method="time", limit=MAX_GAP_HOURS)

    weather = weather.reindex(full_range).interpolate(method="time", limit=MAX_GAP_HOURS)

    merged = energy.join(weather)
    n_unfixable = int(merged[TARGET].isna().sum())
    merged = merged.dropna(subset=[TARGET])
    merged.index.name = "time"

    quality = {
        "missing_hours_reindexed": n_missing_hours,
        "target_nans_interpolated": n_nan_target - n_unfixable,
        "rows_dropped_gap_too_long": n_unfixable,
        "final_rows": int(len(merged)),
        "start": str(merged.index.min()),
        "end": str(merged.index.max()),
    }
    return merged, quality


def main() -> pd.DataFrame:
    ensure_dirs()
    maybe_download_from_kaggle()

    energy, q1 = load_energy()
    weather, q2 = load_weather()
    merged, q3 = clean_and_merge(energy, weather)

    quality = {**q1, **q2, **q3}
    (REPORTS / "data_quality.json").write_text(json.dumps(quality, indent=2))
    merged.to_csv(PROCESSED_FILE)

    log.info("Data quality summary: %s", json.dumps(quality, indent=2))
    log.info("Saved %s  (%s rows, %s cols)", PROCESSED_FILE, len(merged), merged.shape[1])
    return merged


def load_processed() -> pd.DataFrame:
    """Convenience loader used by every downstream script."""
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(
            f"{PROCESSED_FILE} not found - run `python -m src.data_ingestion` first."
        )
    df = pd.read_csv(PROCESSED_FILE, index_col="time", parse_dates=["time"])
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    return df


if __name__ == "__main__":
    main()
