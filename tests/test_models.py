"""Baselines, SARIMAX disaggregation and the LightGBM forecaster."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.evaluate as ev
from src.config import HORIZON, TARGET
from src.features import FEATURES, build_features
from src.models import sarimax_model as sm
from src.models.baselines import naive_24h, seasonal_naive_168h
from src.models.lgbm_model import predict_quantiles, train_point

HOLDOUT_DAYS = 28


@pytest.fixture(scope="module")
def split(frame):
    feats = build_features(frame)
    cut = feats.index[-HOLDOUT_DAYS * 24]
    return cut, feats[feats.index < cut], feats[feats.index >= cut]


@pytest.fixture(scope="module")
def point_model(split):
    _, train, _ = split
    return train_point(train[FEATURES], train[TARGET])


def test_baselines_are_plain_lags(frame):
    y = frame[TARGET]
    idx = y.index[1000:1010]
    assert naive_24h(y, idx).to_numpy() == pytest.approx(y.shift(24).loc[idx].to_numpy())
    assert seasonal_naive_168h(y, idx).to_numpy() == pytest.approx(y.shift(168).loc[idx].to_numpy())
    assert y.loc[idx[0] - pd.Timedelta(hours=24)] == naive_24h(y, idx).iloc[0]


def test_lightgbm_beats_both_naive_baselines_on_a_holdout(frame, split, point_model):
    _, _, test = split
    pred = point_model.predict(test[FEATURES])
    actual = test[TARGET].to_numpy()
    lgbm = ev.mape(actual, pred)
    assert lgbm < ev.mape(actual, naive_24h(frame[TARGET], test.index).to_numpy())
    assert lgbm < ev.mape(actual, seasonal_naive_168h(frame[TARGET], test.index).to_numpy())


def test_first_forecast_day_ignores_the_load_it_is_scored_on(frame, split, point_model):
    """End to end: corrupting the holdout load cannot change the first day's forecast."""
    cut, _, test = split
    corrupted = frame.copy()
    corrupted.loc[corrupted.index >= cut, TARGET] += 9999.0
    other = build_features(corrupted, dropna=False)
    first_day = test.index[test.index < cut + pd.Timedelta(hours=HORIZON)]
    clean_pred = point_model.predict(test.loc[first_day, FEATURES])
    corrupt_pred = point_model.predict(other.loc[first_day, FEATURES])
    assert np.array_equal(clean_pred, corrupt_pred)


def test_quantile_predictions_are_ordered():
    class Stub:
        def __init__(self, value: float):
            self.value = value

        def predict(self, X):
            return np.full(len(X), self.value)

    X = pd.DataFrame({"a": [0.0, 1.0, 2.0]})
    out = predict_quantiles({0.1: Stub(30.0), 0.5: Stub(10.0), 0.9: Stub(20.0)}, X)
    assert (out["p10"] <= out["p50"]).all() and (out["p50"] <= out["p90"]).all()


def test_hour_of_week_profile_has_mean_close_to_one(frame):
    profile = sm.hour_of_week_profile(frame)
    assert len(profile) == 7 * 24
    assert profile.mean() == pytest.approx(1.0, abs=0.01)


def test_disaggregated_day_averages_back_to_the_daily_forecast(frame):
    profile = sm.hour_of_week_profile(frame)
    day = pd.date_range("2016-06-01 00:00", periods=24, freq="h", tz="UTC")
    hourly = sm.disaggregate(30000.0, day, profile)
    assert len(hourly) == 24
    assert hourly.mean() == pytest.approx(30000.0, rel=0.02)
