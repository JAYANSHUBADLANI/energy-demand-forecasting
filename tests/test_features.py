"""Leakage checks for the day-ahead feature set.

The claim under test: every load-derived feature at hour t uses only load up
to t - 24 h, so the feature matrix for day D can be built at the end of day D-1.
The check changes the load from a cut-off onward and asks which feature cells
move. Cells in the 24 h after the cut-off must not.

Weather is the exception and is pinned down here too: ``temp`` and ``temp_sq``
are the observed value at the target hour, so the backtest assumes a perfect
weather forecast.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import HORIZON, TARGET
from src.features import FEATURES, WEATHER_FEATURES, build_features

SHIFT_MW = 12345.0
POSITIONS = [1500, 6007, 11113, -400]  # different hours of day, early to late


def build_all(df: pd.DataFrame) -> pd.DataFrame:
    return build_features(df, dropna=False)


def leaked_cells(builder, frame: pd.DataFrame, cut: pd.Timestamp, columns: list[str]) -> int:
    """Feature cells that change in the 24 h after `cut` when load from `cut` on is altered."""
    perturbed = frame.copy()
    perturbed.loc[perturbed.index >= cut, TARGET] += SHIFT_MW
    base, other = builder(frame), builder(perturbed)
    early = base.index < cut + pd.Timedelta(hours=HORIZON)
    moved = base.loc[early, columns].fillna(-999.0) != other.loc[early, columns].fillna(-999.0)
    return int(moved.to_numpy().sum())


@pytest.mark.parametrize("position", POSITIONS)
def test_features_never_use_load_from_the_last_24_hours(frame, position):
    cut = frame.index[position]
    assert leaked_cells(build_all, frame, cut, FEATURES) == 0


@pytest.mark.parametrize("position", POSITIONS)
def test_the_check_is_sensitive(frame, position):
    """Features further than 24 h past the cut-off do move, so a zero above is not vacuous."""
    cut = frame.index[position]
    perturbed = frame.copy()
    perturbed.loc[perturbed.index >= cut, TARGET] += SHIFT_MW
    base, other = build_all(frame), build_all(perturbed)
    late = base.index >= cut + pd.Timedelta(hours=HORIZON)
    assert (base.loc[late, "lag_24h"] != other.loc[late, "lag_24h"]).any()


def test_the_check_catches_a_planted_leak(frame):
    def leaky(df: pd.DataFrame) -> pd.DataFrame:
        return build_all(df).assign(lag_1h=df[TARGET].shift(1))

    cut = frame.index[6007]
    assert leaked_cells(leaky, frame, cut, FEATURES + ["lag_1h"]) > 0


def test_weather_features_are_the_observed_value_at_the_target_hour(frame):
    hour = frame.index[5000]
    perturbed = frame.copy()
    perturbed.loc[hour, "temp"] += 5.0
    base, other = build_all(frame), build_all(perturbed)
    moved = base[FEATURES].fillna(-999.0) != other[FEATURES].fillna(-999.0)
    assert set(moved.columns[moved.any()]) == set(WEATHER_FEATURES)
    assert list(moved.index[moved.any(axis=1)]) == [hour]


def test_feature_set_has_no_raw_forecast_or_price_columns():
    assert not {"total load forecast", "price actual"} & set(FEATURES)


def test_calendar_features_use_local_time(frame):
    feats = build_all(frame)
    row = feats.loc[pd.Timestamp("2015-01-05 23:00", tz="UTC")]  # 00:00 on 6 Jan in Madrid
    assert row["hour"] == 0
    assert row["is_holiday"] == 1  # Epiphany
    assert row["weekday"] == 1
