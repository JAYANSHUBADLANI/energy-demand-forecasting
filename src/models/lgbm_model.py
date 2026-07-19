"""LightGBM models on hourly data - the main forecaster.

* Point model: LGBMRegressor (L2) on the full feature set.
* Quantile models: three LGBMRegressor(objective="quantile") fits at
  alpha = 0.10 / 0.50 / 0.90 for day-ahead prediction intervals.

Because every lag/rolling feature is shifted >= 24 h (see src/features.py),
the model produces a *direct* day-ahead forecast - no recursive feedback,
no leakage.

Running this module trains on all data except the evaluation window, prints
a quick holdout check, and pickles the models + feature list to ``models/``
for reuse by the forecast script and the Streamlit app.

Run:  python -m src.models.lgbm_model
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import (
    LGBM_PARAMS,
    MODELS_DIR,
    QUANTILES,
    TARGET,
    TEST_WEEKS,
    ensure_dirs,
)
from src.features import FEATURES, load_features

POINT_MODEL_FILE = MODELS_DIR / "lgbm_point.pkl"
QUANTILE_MODEL_FILE = MODELS_DIR / "lgbm_quantiles.pkl"
META_FILE = MODELS_DIR / "lgbm_meta.json"


def train_point(X: pd.DataFrame, y: pd.Series) -> LGBMRegressor:
    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(X, y)
    return model


def train_quantiles(X: pd.DataFrame, y: pd.Series) -> dict[float, LGBMRegressor]:
    models = {}
    for q in QUANTILES:
        params = {**LGBM_PARAMS, "objective": "quantile", "alpha": q, "n_estimators": 400}
        models[q] = LGBMRegressor(**params).fit(X, y)
    return models


def predict_quantiles(models: dict[float, LGBMRegressor], X: pd.DataFrame) -> pd.DataFrame:
    """Predict all quantiles and enforce monotonicity (P10 <= P50 <= P90)."""
    preds = pd.DataFrame({f"p{int(q * 100)}": m.predict(X) for q, m in models.items()},
                         index=X.index)
    preds[:] = np.sort(preds.to_numpy(), axis=1)
    return preds


def save(point: LGBMRegressor, quantiles: dict[float, LGBMRegressor], train_end: str) -> None:
    with open(POINT_MODEL_FILE, "wb") as fh:
        pickle.dump(point, fh)
    with open(QUANTILE_MODEL_FILE, "wb") as fh:
        pickle.dump(quantiles, fh)
    META_FILE.write_text(json.dumps({"features": FEATURES, "train_end": train_end}))


def load() -> tuple[LGBMRegressor, dict[float, LGBMRegressor]]:
    with open(POINT_MODEL_FILE, "rb") as fh:
        point = pickle.load(fh)
    with open(QUANTILE_MODEL_FILE, "rb") as fh:
        quantiles = pickle.load(fh)
    return point, quantiles


def main() -> None:
    ensure_dirs()
    feats = load_features()
    X, y = feats[FEATURES], feats[TARGET]

    cutoff = feats.index.max() - pd.Timedelta(weeks=TEST_WEEKS)
    X_train, y_train = X[X.index <= cutoff], y[y.index <= cutoff]

    point = train_point(X_train, y_train)
    quantiles = train_quantiles(X_train, y_train)
    save(point, quantiles, str(cutoff))

    # quick internal check on the first unseen week (full backtest lives in evaluate.py)
    week = X[(X.index > cutoff) & (X.index <= cutoff + pd.Timedelta(days=7))]
    if len(week):
        pred = point.predict(week)
        actual = y.reindex(week.index)
        mape = float(np.mean(np.abs((actual - pred) / actual))) * 100
        print(f"Sanity check - first unseen week MAPE: {mape:.2f}%")

    imp = pd.Series(point.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("Top 10 feature importances:")
    print(imp.head(10).to_string())
    print(f"Saved models to {MODELS_DIR}")


if __name__ == "__main__":
    main()
