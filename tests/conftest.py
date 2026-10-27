"""Shared fixtures.

The tests never need the Kaggle files. They build a two year sample with the
schema-exact generator in ``src/synthetic_data.py`` and run it through the real
ingestion code, so cleaning and feature logic is exercised on the same path as
the full pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import data_ingestion as di
from src import synthetic_data as sd
from src.config import ENERGY_CSV, WEATHER_CSV

YEARS = 2


@pytest.fixture(scope="session")
def ingested(tmp_path_factory) -> tuple[pd.DataFrame, dict]:
    """Merged hourly frame and the quality dict, built from a seeded sample."""
    raw = tmp_path_factory.mktemp("raw")
    with pytest.MonkeyPatch.context() as mp:
        # build() draws from a module level generator, so reseed for a fixed sample
        mp.setattr(sd, "RNG", np.random.default_rng(42))
        energy, weather = sd.build(YEARS)
        energy.to_csv(raw / ENERGY_CSV, index=False)
        weather.to_csv(raw / WEATHER_CSV, index=False)

        mp.setattr(di, "DATA_RAW", raw)
        energy_df, q_energy = di.load_energy()
        weather_df, q_weather = di.load_weather()
        merged, q_merge = di.clean_and_merge(energy_df, weather_df)
    return merged, {**q_energy, **q_weather, **q_merge}


@pytest.fixture(scope="session")
def frame(ingested) -> pd.DataFrame:
    """Hourly load + weather frame. Treat as read-only; copy before editing."""
    return ingested[0]
