"""Exploratory data analysis: seasonal structure of Spanish electricity load.

Generates and saves to ``reports/figures/``:

* ``heatmap_hour_weekday.png``  - mean load, hour x weekday
* ``profile_daily.png``         - average intraday profile (weekday vs weekend)
* ``profile_weekly.png``        - average hour-of-week profile
* ``profile_annual.png``        - monthly distribution + daily mean series
* ``stl_decomposition.png``     - STL on daily means (weekly seasonality)
* ``acf_pacf.png``              - ACF/PACF of hourly load
* ``load_vs_temperature.png``   - the U-shape: heating + cooling demand

Run:  python -m src.eda        (from the repo root)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import FIGURES, LOCAL_TZ, TARGET, ensure_dirs
from src.data_ingestion import load_processed

sns.set_theme(style="whitegrid", context="talk")
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _save(fig: plt.Figure, name: str) -> None:
    path = FIGURES / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def heatmap_hour_weekday(df: pd.DataFrame) -> None:
    pivot = df.pivot_table(index="hour", columns="weekday", values=TARGET, aggfunc="mean")
    pivot.columns = [WEEKDAYS[c] for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, cmap="rocket_r", ax=ax, cbar_kws={"label": "Mean load (MW)"})
    ax.set(title="Mean load by hour and weekday", xlabel="", ylabel="Hour of day (local)")
    _save(fig, "heatmap_hour_weekday.png")


def daily_profile(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, sub in (("Weekday", df[df.weekday < 5]), ("Weekend", df[df.weekday >= 5])):
        prof = sub.groupby("hour")[TARGET]
        ax.plot(prof.mean(), label=label, lw=2.5)
        ax.fill_between(prof.mean().index, prof.quantile(0.1), prof.quantile(0.9), alpha=0.15)
    ax.set(title="Average intraday load profile", xlabel="Hour of day (local)", ylabel="Load (MW)")
    ax.legend()
    _save(fig, "profile_daily.png")


def weekly_profile(df: pd.DataFrame) -> None:
    how = df.weekday * 24 + df.hour
    prof = df.groupby(how)[TARGET].mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(prof.index, prof.values, lw=1.8)
    ax.set_xticks(range(0, 168, 24), WEEKDAYS)
    for x in range(0, 169, 24):
        ax.axvline(x, color="grey", lw=0.5, alpha=0.5)
    ax.set(title="Average hour-of-week load profile", ylabel="Load (MW)")
    _save(fig, "profile_weekly.png")


def annual_profile(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    sns.boxplot(data=df, x="month", y=TARGET, ax=axes[0], color="#4c72b0", fliersize=1)
    axes[0].set(title="Load distribution by month", xlabel="", ylabel="Load (MW)")
    daily = df[TARGET].resample("D").mean()
    axes[1].plot(daily.index, daily.values, lw=0.8)
    axes[1].set(title="Daily mean load", ylabel="Load (MW)")
    fig.tight_layout()
    _save(fig, "profile_annual.png")


def stl_daily(df: pd.DataFrame) -> None:
    daily = df[TARGET].resample("D").mean().interpolate()
    res = STL(daily, period=7, robust=True).fit()
    fig = res.plot()
    fig.set_size_inches(11, 9)
    fig.suptitle("STL decomposition of daily mean load (weekly period)", y=1.02)
    _save(fig, "stl_decomposition.png")


def acf_pacf(df: pd.DataFrame) -> None:
    y = df[TARGET].dropna()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(y, lags=192, ax=axes[0])
    axes[0].set_title("ACF of hourly load (8 days of lags)")
    plot_pacf(y[-5000:], lags=72, method="ywm", ax=axes[1])
    axes[1].set_title("PACF of hourly load")
    fig.tight_layout()
    _save(fig, "acf_pacf.png")


def load_vs_temperature(df: pd.DataFrame) -> None:
    daily = df.resample("D").mean(numeric_only=True).dropna(subset=[TARGET, "temp"])
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(daily["temp"], daily[TARGET], s=12, alpha=0.35, label="Daily means")
    bins = pd.cut(daily["temp"], np.arange(daily["temp"].min() // 1, daily["temp"].max() + 1, 1))
    binned = daily.groupby(bins, observed=True)[TARGET].mean()
    centers = [b.mid for b in binned.index]
    ax.plot(centers, binned.values, color="crimson", lw=3, label="Binned mean")
    ax.set(title="Load vs temperature: heating + cooling U-shape",
           xlabel="Mean temperature across 5 cities (°C)", ylabel="Daily mean load (MW)")
    ax.legend()
    _save(fig, "load_vs_temperature.png")


def main() -> None:
    ensure_dirs()
    df = load_processed()
    local = df.index.tz_convert(LOCAL_TZ)
    df = df.assign(hour=local.hour, weekday=local.weekday, month=local.month)

    heatmap_hour_weekday(df)
    daily_profile(df)
    weekly_profile(df)
    annual_profile(df)
    stl_daily(df)
    acf_pacf(df)
    load_vs_temperature(df)
    print(f"All EDA figures written to {FIGURES}")


if __name__ == "__main__":
    main()
