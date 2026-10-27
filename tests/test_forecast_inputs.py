"""Inputs of the live next-24h forecast (src/forecast.py)."""

from __future__ import annotations

import numpy as np

from src.config import TARGET
from src.features import FEATURES, build_features
from src.forecast import extend_with_future


def test_future_weather_is_24h_persistence(frame):
    extended, future_index = extend_with_future(frame)
    assert len(future_index) == 24
    weather_cols = [c for c in frame.columns if c != TARGET]
    for col in weather_cols:
        np.testing.assert_allclose(
            extended.loc[future_index, col].to_numpy(), frame[col].iloc[-24:].to_numpy()
        )
    assert extended.loc[future_index, TARGET].isna().all()


def test_future_feature_matrix_is_complete(frame):
    extended, future_index = extend_with_future(frame)
    feats = build_features(extended, dropna=False)
    assert not feats.loc[future_index, FEATURES].isna().any().any()
