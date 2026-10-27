"""Robustness checks for the LightGBM vs SARIMAX comparison.

The headline Diebold-Mariano test (src/evaluate.py) uses squared-error loss,
which a handful of badly missed holiday and bridge days can dominate. These
checks ask whether the verdict changes with the loss function, the variance
weights, or when errors are averaged by day. They were added after the headline
result was known, so they are sensitivity checks, not a replacement for it.

Reads the cached backtest predictions in ``data/processed/backtest/`` (run
``python -m src.evaluate`` first).

Output: reports/dm_sensitivity.json

Run:  python -m src.sensitivity        (from the repo root)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import holidays as holidays_lib
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import LOCAL_TZ, REPORTS, TARGET, ensure_dirs
from src.evaluate import _load_preds, dm_from_differential, test_window
from src.features import build_features

MODEL_A = "lgbm"
MODEL_B = "sarimax"


def run() -> dict:
    feats = build_features()
    _, test_index = test_window(feats)
    actual = feats[TARGET].reindex(test_index)
    pred_a = _load_preds(MODEL_A)["pred"].reindex(test_index)
    pred_b = _load_preds(MODEL_B)["pred"].reindex(test_index)
    mask = (actual.notna() & pred_a.notna() & pred_b.notna()).to_numpy()
    e_a = (actual - pred_a).to_numpy()[mask]
    e_b = (actual - pred_b).to_numpy()[mask]

    out: dict = {"note": "model 1 = LightGBM, model 2 = SARIMAX(daily)+temp; negative statistic favours LightGBM"}
    for loss, d in (("squared", e_a**2 - e_b**2), ("absolute", np.abs(e_a) - np.abs(e_b))):
        for kernel, bartlett in (("equal_weights", False), ("bartlett", True)):
            res = dm_from_differential(d, bartlett=bartlett)
            out[f"dm_{loss}_{kernel}"] = {"statistic": round(res["statistic"], 3),
                                          "p_value": float(f"{res['p_value']:.3g}")}

    # one number per complete UTC day, the same day definition the backtest uses
    day = pd.Series(test_index[mask].date, index=test_index[mask])
    daily = pd.DataFrame({"lgbm": np.abs(e_a), "sarimax": np.abs(e_b)}).groupby(day.to_numpy()).mean()
    diff = (daily["lgbm"] - daily["sarimax"]).to_numpy()
    out["daily_mae"] = {
        "days": int(len(daily)),
        "lgbm_better_days": int((diff < 0).sum()),
        "paired_t_p_value": float(f"{stats.ttest_1samp(diff, 0.0).pvalue:.3g}"),
        "wilcoxon_p_value": float(f"{stats.wilcoxon(daily['lgbm'], daily['sarimax']).pvalue:.3g}"),
    }

    # ordinary days vs weekends and Spanish public holidays (local date at midday UTC)
    es_holidays = holidays_lib.country_holidays("ES", years=sorted({d.year for d in daily.index}))
    local_date = [pd.Timestamp(d, tz="UTC").replace(hour=12).tz_convert(LOCAL_TZ).date() for d in daily.index]
    special = np.array([(d.weekday() >= 5) or (d in es_holidays) for d in local_date])
    out["daily_mae_by_day_type"] = {
        "weekday_non_holiday": {"days": int((~special).sum()),
                                "lgbm_mw": round(float(daily["lgbm"][~special].mean())),
                                "sarimax_mw": round(float(daily["sarimax"][~special].mean()))},
        "weekend_or_holiday": {"days": int(special.sum()),
                               "lgbm_mw": round(float(daily["lgbm"][special].mean())),
                               "sarimax_mw": round(float(daily["sarimax"][special].mean()))},
    }

    # how concentrated the squared error is
    sq = pd.DataFrame({"lgbm": e_a**2, "sarimax": e_b**2}).groupby(day.to_numpy()).sum()
    out["squared_error_share_of_worst_5_days"] = {
        name: round(float(sq[name].sort_values(ascending=False).head(5).sum() / sq[name].sum()), 3)
        for name in ("lgbm", "sarimax")
    }
    return out


def main() -> None:
    ensure_dirs()
    result = run()
    (REPORTS / "dm_sensitivity.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
