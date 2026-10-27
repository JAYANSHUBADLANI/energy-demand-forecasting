"""Cleaning and merge behaviour on the seeded sample."""

from __future__ import annotations

import pandas as pd

from src.config import TARGET


def test_hourly_index_is_complete_and_utc(frame):
    assert str(frame.index.tz) == "UTC"
    assert frame.index.is_monotonic_increasing
    steps = frame.index.to_series().diff().dropna()
    assert (steps == pd.Timedelta(hours=1)).all()


def test_no_missing_target_or_temperature(frame):
    assert frame[TARGET].notna().all()
    assert frame["temp"].notna().all()


def test_temperature_is_in_celsius(frame):
    assert frame["temp"].between(-20, 50).all()


def test_weather_duplicates_are_dropped_and_city_names_cleaned(ingested):
    _, quality = ingested
    assert quality["weather_duplicate_rows_dropped"] == 1500
    assert quality["weather_cities"] == ["Barcelona", "Bilbao", "Madrid", "Seville", "Valencia"]


def test_every_gap_in_the_sample_is_short_enough_to_interpolate(ingested):
    _, quality = ingested
    assert quality["missing_hours_reindexed"] > 0
    assert quality["rows_dropped_gap_too_long"] == 0
