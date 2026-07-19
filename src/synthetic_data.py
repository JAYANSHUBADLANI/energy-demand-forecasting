"""Generate a synthetic sample of the Kaggle dataset schema for testing.

Writes ``energy_dataset.csv`` and ``weather_features.csv`` to ``data/raw/``
with the exact column layout of the real Kaggle files
("Hourly energy demand generation and weather", nicholasjhana), including
the known quirks the ingestion step must survive:

* tz-aware local timestamps (``2015-01-01 00:00:00+01:00``),
* duplicated rows in the weather file,
* a handful of missing hours and NaN targets in the energy file,
* temperatures in Kelvin,
* a stray leading space in one ``city_name``.

The synthetic load embeds realistic structure (daily double peak, weekend
dip, annual cycle, U-shaped temperature response) so EDA plots and models
behave sensibly. It is a development utility only; real analysis uses the
Kaggle data.

Run:  python -m src.synthetic_data [--years 4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import DATA_RAW, ENERGY_CSV, LOCAL_TZ, WEATHER_CSV, ensure_dirs

RNG = np.random.default_rng(42)

CITY_OFFSETS = {  # mean °C offset, annual amplitude
    "Valencia": (2.0, 7.5),
    "Madrid": (0.0, 10.0),
    "Bilbao": (-1.5, 6.5),
    " Barcelona": (1.0, 7.0),  # leading space mirrors the real file
    "Seville": (4.0, 9.0),
}

ENERGY_COLUMNS = [
    "time",
    "generation biomass",
    "generation fossil brown coal/lignite",
    "generation fossil coal-derived gas",
    "generation fossil gas",
    "generation fossil hard coal",
    "generation fossil oil",
    "generation fossil oil shale",
    "generation fossil peat",
    "generation geothermal",
    "generation hydro pumped storage aggregated",
    "generation hydro pumped storage consumption",
    "generation hydro run-of-river and poundage",
    "generation hydro water reservoir",
    "generation marine",
    "generation nuclear",
    "generation other",
    "generation other renewable",
    "generation solar",
    "generation waste",
    "generation wind offshore",
    "generation wind onshore",
    "forecast solar day ahead",
    "forecast wind offshore eday ahead",
    "forecast wind onshore day ahead",
    "total load forecast",
    "total load actual",
    "price day ahead",
    "price actual",
]


def _ar1(n: int, phi: float, sigma: float) -> np.ndarray:
    eps = RNG.normal(0, sigma, n)
    out = np.empty(n)
    out[0] = eps[0]
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return out


def _hour_shapes() -> tuple[np.ndarray, np.ndarray]:
    """Season-dependent double-peak intraday profiles (multiplicative, mean≈1).

    Real load shape drifts with the season (sharper evening peak in winter,
    fatter air-conditioning midday in summer), so the profile is a seasonal
    blend rather than a single fixed curve.
    """
    h = np.arange(24)
    winter = (0.84
              + 0.18 * np.exp(-0.5 * ((h - 12.0) / 3.0) ** 2)
              + 0.32 * np.exp(-0.5 * ((h - 20.0) / 1.7) ** 2)
              - 0.10 * np.exp(-0.5 * ((h - 4.0) / 2.2) ** 2))
    summer = (0.88
              + 0.26 * np.exp(-0.5 * ((h - 14.0) / 3.6) ** 2)
              + 0.16 * np.exp(-0.5 * ((h - 21.5) / 2.3) ** 2)
              - 0.12 * np.exp(-0.5 * ((h - 4.5) / 2.4) ** 2))
    return winter / winter.mean(), summer / summer.mean()


def make_temperature(idx_local: pd.DatetimeIndex, mean_off: float, amp: float) -> np.ndarray:
    doy = idx_local.dayofyear.to_numpy()
    hour = idx_local.hour.to_numpy()
    annual = -amp * np.cos(2 * np.pi * (doy - 15) / 365.25)
    diurnal = (2.5 + 2.0 * (annual > 0)) * -np.cos(2 * np.pi * (hour - 5) / 24)
    noise = _ar1(len(idx_local), 0.95, 0.55)
    return 16.0 + mean_off + annual + diurnal + noise


def build(years: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp("2015-01-01 00:00", tz=LOCAL_TZ)
    idx_local = pd.date_range(start, periods=years * 365 * 24, freq="h")
    n = len(idx_local)

    # ---- city temperatures ------------------------------------------------
    temps = {c: make_temperature(idx_local, off, amp) for c, (off, amp) in CITY_OFFSETS.items()}
    temp_avg = np.mean(list(temps.values()), axis=0)

    # ---- load -------------------------------------------------------------
    daily_temp = pd.Series(temp_avg, index=idx_local).resample("D").mean()
    daily_temp_h = daily_temp.reindex(idx_local, method="ffill").to_numpy()

    base = 28000.0
    trend = 60.0 * (idx_local.year - idx_local.year.min()).to_numpy()
    temp_effect = 26.0 * np.clip(daily_temp_h - 17.0, -14, 16) ** 2

    winter_shape, summer_shape = _hour_shapes()
    hours = idx_local.hour.to_numpy()
    winter_w = 0.5 + 0.5 * np.cos(2 * np.pi * (idx_local.dayofyear.to_numpy() - 15) / 365.25)
    hour_factor = winter_w * winter_shape[hours] + (1 - winter_w) * summer_shape[hours]
    dow = idx_local.dayofweek.to_numpy()
    dow_factor = np.where(dow == 5, 0.93, np.where(dow == 6, 0.88, 1.0))

    load = (base + trend + temp_effect) * hour_factor * dow_factor
    load += _ar1(n, 0.75, 320.0)

    # ---- energy dataframe (full real schema) -------------------------------
    hour = idx_local.hour.to_numpy()
    solar_shape = np.clip(np.sin(np.pi * (hour - 7) / 12), 0, None)
    solar_season = 1.0 + 0.45 * -np.cos(2 * np.pi * (idx_local.dayofyear.to_numpy() - 15) / 365.25)
    solar = 4200 * solar_shape * solar_season * (1 + RNG.normal(0, 0.10, n)).clip(0.2)
    wind = np.clip(5400 + _ar1(n, 0.97, 260.0) * 4.0, 300, None)

    energy = pd.DataFrame(index=idx_local, columns=ENERGY_COLUMNS[1:], dtype=float)
    energy["total load actual"] = load
    energy["total load forecast"] = load * (1 + RNG.normal(0, 0.012, n))
    energy["generation solar"] = solar
    energy["generation wind onshore"] = wind
    energy["forecast solar day ahead"] = solar * (1 + RNG.normal(0, 0.08, n))
    energy["forecast wind onshore day ahead"] = wind * (1 + RNG.normal(0, 0.10, n))
    energy["generation nuclear"] = 6900 + _ar1(n, 0.99, 25.0)
    energy["generation fossil gas"] = np.clip(load * 0.18 - solar * 0.3 + _ar1(n, 0.9, 180), 0, None)
    energy["generation fossil hard coal"] = np.clip(4200 + _ar1(n, 0.98, 140.0), 0, None)
    energy["generation hydro water reservoir"] = np.clip(2400 + _ar1(n, 0.97, 220.0), 0, None)
    energy["generation biomass"] = 380 + _ar1(n, 0.95, 12.0)
    energy["generation waste"] = 270 + _ar1(n, 0.95, 8.0)
    energy["generation other"] = 60.0
    energy["generation other renewable"] = 85.0
    energy["generation fossil oil"] = 300 + _ar1(n, 0.9, 15.0)
    energy["generation hydro run-of-river and poundage"] = np.clip(960 + _ar1(n, 0.98, 60.0), 0, None)
    energy["generation hydro pumped storage consumption"] = np.clip(-500 * (hour_factor - 1) * 3 + 400, 0, None)
    energy["price actual"] = 12 + load / 900 + RNG.normal(0, 3.5, n)
    energy["price day ahead"] = energy["price actual"] * (1 + RNG.normal(0, 0.05, n))
    for col in ("generation fossil brown coal/lignite", "generation fossil coal-derived gas",
                "generation fossil oil shale", "generation fossil peat", "generation geothermal",
                "generation marine", "generation wind offshore",
                "generation hydro pumped storage aggregated", "forecast wind offshore eday ahead"):
        energy[col] = 0.0

    # quirks: missing hours + NaN targets
    drop_idx = RNG.choice(np.arange(200, n - 200), size=40, replace=False)
    nan_idx = RNG.choice(np.setdiff1d(np.arange(200, n - 200), drop_idx), size=36, replace=False)
    energy.iloc[nan_idx, energy.columns.get_loc("total load actual")] = np.nan
    energy = energy.drop(energy.index[drop_idx])

    energy = energy.reset_index(names="time")
    energy["time"] = energy["time"].dt.strftime("%Y-%m-%d %H:%M:%S%z").str.replace(
        r"(\+|-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True)

    # ---- weather dataframe --------------------------------------------------
    frames = []
    for city, series in temps.items():
        w = pd.DataFrame({
            "dt_iso": idx_local,
            "city_name": city,
            "temp": series + 273.15,  # Kelvin, like the real file
            "temp_min": series - 1.6 + 273.15,
            "temp_max": series + 1.6 + 273.15,
            "pressure": (1014 + _ar1(n, 0.98, 1.2)).round(0),
            "humidity": np.clip(62 - 1.4 * (series - 16) + _ar1(n, 0.9, 4.0), 8, 100).round(0),
            "wind_speed": np.clip(3 + _ar1(n, 0.9, 1.1), 0, None).round(0),
            "wind_deg": RNG.integers(0, 360, n),
            "rain_1h": np.where(RNG.random(n) < 0.045, RNG.exponential(0.9, n), 0.0),
            "rain_3h": 0.0,
            "snow_3h": 0.0,
            "clouds_all": np.clip(38 + _ar1(n, 0.9, 14.0), 0, 100).round(0),
            "weather_id": 800,
            "weather_main": "clear",
            "weather_description": "sky is clear",
            "weather_icon": "01n",
        })
        frames.append(w)
    weather = pd.concat(frames, ignore_index=True)

    # quirk: duplicated rows (the real file has thousands)
    dupes = weather.sample(n=1500, random_state=7)
    weather = pd.concat([weather, dupes], ignore_index=True)
    weather = weather.sort_values(["city_name", "dt_iso"]).reset_index(drop=True)
    weather["dt_iso"] = weather["dt_iso"].dt.strftime("%Y-%m-%d %H:%M:%S%z").str.replace(
        r"(\+|-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True)

    return energy, weather


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, default=4)
    args = parser.parse_args()

    ensure_dirs()
    energy, weather = build(args.years)
    energy.to_csv(DATA_RAW / ENERGY_CSV, index=False)
    weather.to_csv(DATA_RAW / WEATHER_CSV, index=False)
    print(f"Wrote {DATA_RAW / ENERGY_CSV}  ({len(energy):,} rows)")
    print(f"Wrote {DATA_RAW / WEATHER_CSV}  ({len(weather):,} rows)")


if __name__ == "__main__":
    main()
