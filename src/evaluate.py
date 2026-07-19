"""Rolling-origin day-ahead backtest over the final 12 weeks.

Protocol
--------
For every day D in the test window the forecast origin is the end of day
D-1: each model predicts the 24 hours of day D using only information
available at the origin. The origin then rolls forward one day.

* naive-24h / seasonal-naive-168h : pure lags, always valid day-ahead.
* SARIMAX (daily + temp exog)     : one-step daily forecast, disaggregated
  to hourly with the train-period hour-of-week profile; state updated with
  each observed day (Kalman filter append), full refit every 4 weeks.
* LightGBM (point + quantiles)    : direct day-ahead model (all features
  shifted >= 24 h); point model refit weekly, quantile models every 4 weeks.

Each model's backtest predictions are cached under
``data/processed/backtest/`` so stages can run independently
(``--only sarimax|lgbm|quantiles|metrics``); the default runs everything.

Outputs
-------
reports/metrics_overall.csv, reports/metrics_per_hour.csv,
reports/results.md (ready to paste into the README), reports/dm_test.json,
reports/figures/error_by_hour.png, reports/figures/backtest_last_week.png

Run:  python -m src.evaluate        (from the repo root)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import (
    DATA_PROCESSED,
    FIGURES,
    HORIZON,
    LGBM_REFIT_DAYS,
    LOCAL_TZ,
    REPORTS,
    SARIMAX_REFIT_DAYS,
    TARGET,
    TEST_WEEKS,
    ensure_dirs,
)
from src.data_ingestion import load_processed
from src.features import FEATURES, build_features
from src.models import sarimax_model as sm
from src.models.baselines import naive_24h, seasonal_naive_168h
from src.models.lgbm_model import train_point, train_quantiles

warnings.filterwarnings("ignore")

MAIN_MODEL = "LightGBM"
BT_DIR = DATA_PROCESSED / "backtest"
QUANTILE_REFIT_DAYS = 28  # interval models are stable; refit monthly in backtest


# --------------------------------------------------------------- metrics ----
def mape(a: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs((a - p) / a))) * 100


def rmse(a: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mae(a: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(a - p)))


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = HORIZON) -> dict:
    """DM test on squared-error loss with Newey-West variance and the
    Harvey-Leybourne-Newbold small-sample correction.

    Negative statistic => model 1 has significantly lower loss.
    """
    d = e1**2 - e2**2
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    gamma = [float(np.mean(dc[: n - k] * dc[k:])) for k in range(h)]
    var_dbar = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var_dbar <= 0:
        var_dbar = gamma[0] / n
    dm = dbar / np.sqrt(var_dbar)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_adj = float(dm * hln)
    p_value = float(2 * (1 - stats.t.cdf(abs(dm_adj), df=n - 1)))
    return {"statistic": dm_adj, "p_value": p_value, "n": n, "horizon": h}


# ---------------------------------------------------------------- window ----
def test_window(feats: pd.DataFrame) -> tuple[list, pd.DatetimeIndex]:
    counts = pd.Series(1, index=feats.index).groupby(feats.index.date).sum()
    complete_days = counts[counts == 24].index.tolist()
    test_days = complete_days[-TEST_WEEKS * 7 :]
    test_index = feats.index[np.isin(feats.index.date, test_days)]
    return test_days, test_index


def _save_preds(name: str, preds: pd.Series | pd.DataFrame) -> None:
    BT_DIR.mkdir(parents=True, exist_ok=True)
    out = preds.to_frame("pred") if isinstance(preds, pd.Series) else preds
    out.index.name = "time"
    out.to_csv(BT_DIR / f"{name}.csv")
    print(f"cached backtest predictions -> {BT_DIR / f'{name}.csv'}")


def _load_preds(name: str) -> pd.DataFrame:
    df = pd.read_csv(BT_DIR / f"{name}.csv", index_col="time", parse_dates=["time"])
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    return df


# -------------------------------------------------------------- backtests ----
def run_baselines_and_sarimax(df: pd.DataFrame, feats: pd.DataFrame) -> None:
    test_days, test_index = test_window(feats)
    y_full = df[TARGET]
    _save_preds("naive24", naive_24h(y_full, test_index))
    _save_preds("snaive168", seasonal_naive_168h(y_full, test_index))

    print("Running SARIMAX backtest ...")
    daily = sm.make_daily(df)
    train_cut = pd.Timestamp(test_days[0], tz="UTC")
    profile = sm.hour_of_week_profile(df[df.index < train_cut])

    history = daily[daily.index < pd.Timestamp(test_days[0])]
    order, seasonal = sm.load_or_select_order(history[TARGET], sm.exog_matrix(history))
    res = sm.fit(history[TARGET], sm.exog_matrix(history), order, seasonal)

    preds = []
    for i, day in enumerate(test_days):
        day_ts = pd.Timestamp(day)
        if day_ts not in daily.index:
            continue
        exog_next = sm.exog_matrix(daily.loc[[day_ts]])
        daily_pred = float(res.forecast(1, exog=exog_next).iloc[0])
        hourly_index = feats.index[feats.index.date == day]
        preds.append(sm.disaggregate(daily_pred, hourly_index, profile))

        # roll the origin forward: absorb the observed day
        y_new = daily.loc[[day_ts], TARGET]
        if (i + 1) % SARIMAX_REFIT_DAYS == 0:
            upto = daily[daily.index <= day_ts]
            res = sm.fit(upto[TARGET], sm.exog_matrix(upto), order, seasonal)
        else:
            res = res.append(y_new, exog=exog_next, refit=False)
    _save_preds("sarimax", pd.concat(preds).reindex(test_index))


def run_lgbm(feats: pd.DataFrame, refit_days: int) -> None:
    test_days, test_index = test_window(feats)
    X, y = feats[FEATURES], feats[TARGET]
    out = []
    for i in range(0, len(test_days), refit_days):
        block = test_days[i : i + refit_days]
        train_mask = feats.index < pd.Timestamp(block[0], tz="UTC")
        model = train_point(X[train_mask], y[train_mask])
        Xb = X[np.isin(feats.index.date, block)]
        out.append(pd.Series(model.predict(Xb), index=Xb.index))
        print(f"  refit at {block[0]}: predicted {len(Xb)} h")
    _save_preds("lgbm", pd.concat(out).reindex(test_index))


def run_quantiles(feats: pd.DataFrame, refit_days: int) -> None:
    test_days, test_index = test_window(feats)
    X, y = feats[FEATURES], feats[TARGET]
    frames = []
    for i in range(0, len(test_days), refit_days):
        block = test_days[i : i + refit_days]
        train_mask = feats.index < pd.Timestamp(block[0], tz="UTC")
        q_models = train_quantiles(X[train_mask], y[train_mask])
        Xb = X[np.isin(feats.index.date, block)]
        frames.append(pd.DataFrame(
            {f"p{int(q*100)}": m.predict(Xb) for q, m in q_models.items()}, index=Xb.index))
        print(f"  quantile refit at {block[0]}")
    q = pd.concat(frames).reindex(test_index)
    q[:] = np.sort(q.to_numpy(), axis=1)
    _save_preds("lgbm_quantiles", q)


# ------------------------------------------------------------------ report ----
def compute_metrics(feats: pd.DataFrame) -> None:
    test_days, test_index = test_window(feats)
    actual = feats[TARGET].reindex(test_index)

    preds = {
        "Naive-24h": _load_preds("naive24")["pred"],
        "SeasonalNaive-168h": _load_preds("snaive168")["pred"],
        "SARIMAX(daily)+temp": _load_preds("sarimax")["pred"],
        MAIN_MODEL: _load_preds("lgbm")["pred"],
    }
    band = _load_preds("lgbm_quantiles")

    overall_rows, per_hour_frames = [], []
    local_hour = pd.Series(test_index.tz_convert(LOCAL_TZ).hour, index=test_index)
    for name, p in preds.items():
        p = p.reindex(test_index)
        mask = p.notna() & actual.notna()
        a_, p_ = actual[mask].to_numpy(), p[mask].to_numpy()
        overall_rows.append({"model": name, "MAPE_%": mape(a_, p_),
                             "RMSE_MW": rmse(a_, p_), "MAE_MW": mae(a_, p_)})
        hour_frame = pd.DataFrame({"a": actual[mask], "p": p[mask], "hour": local_hour[mask]})
        hour_stats = hour_frame.groupby("hour").apply(
            lambda g: pd.Series({
                "MAPE_%": mape(g["a"].to_numpy(), g["p"].to_numpy()),
                "RMSE_MW": rmse(g["a"].to_numpy(), g["p"].to_numpy()),
                "MAE_MW": mae(g["a"].to_numpy(), g["p"].to_numpy()),
            }), include_groups=False).assign(model=name)
        per_hour_frames.append(hour_stats)

    overall = pd.DataFrame(overall_rows).sort_values("MAPE_%").reset_index(drop=True)
    per_hour = pd.concat(per_hour_frames).reset_index()
    overall.to_csv(REPORTS / "metrics_overall.csv", index=False)
    per_hour.to_csv(REPORTS / "metrics_per_hour.csv", index=False)
    print(overall.round(2).to_string(index=False))

    # interval coverage
    in_band = ((actual >= band["p10"]) & (actual <= band["p90"])).mean()
    coverage = float(in_band) * 100
    print(f"P10-P90 empirical coverage: {coverage:.1f}%  (nominal 80%)")

    # Diebold-Mariano between the two best models
    top2 = overall["model"].head(2).tolist()
    p1, p2 = preds[top2[0]].reindex(test_index), preds[top2[1]].reindex(test_index)
    mask = p1.notna() & p2.notna() & actual.notna()
    dm = diebold_mariano((actual[mask] - p1[mask]).to_numpy(),
                         (actual[mask] - p2[mask]).to_numpy())
    dm.update({"model_1": top2[0], "model_2": top2[1], "loss": "squared error"})
    (REPORTS / "dm_test.json").write_text(json.dumps(dm, indent=2))
    print(f"Diebold-Mariano {top2[0]} vs {top2[1]}: "
          f"DM={dm['statistic']:.2f}, p={dm['p_value']:.2g}")

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(12, 6))
    for name in overall["model"]:
        sub = per_hour[per_hour.model == name]
        ax.plot(sub["hour"], sub["MAPE_%"], marker="o", ms=4, label=name)
    ax.set(title=f"Day-ahead MAPE by hour of day (last {TEST_WEEKS} weeks)",
           xlabel="Hour of day (local)", ylabel="MAPE (%)")
    ax.legend()
    ax.grid(alpha=0.4)
    fig.savefig(FIGURES / "error_by_hour.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    last_week = test_index[-7 * 24 :]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.fill_between(last_week, band.loc[last_week, "p10"], band.loc[last_week, "p90"],
                    alpha=0.25, color="tab:blue", label="LightGBM P10-P90")
    ax.plot(last_week, actual.loc[last_week], color="black", lw=1.6, label="Actual")
    ax.plot(last_week, preds[MAIN_MODEL].loc[last_week], color="tab:blue", lw=1.4,
            label="LightGBM point")
    ax.plot(last_week, preds["SeasonalNaive-168h"].loc[last_week], color="tab:orange",
            lw=1.0, alpha=0.8, label="SeasonalNaive-168h")
    ax.set(title="Day-ahead backtest - final week", ylabel="Load (MW)")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.savefig(FIGURES / "backtest_last_week.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- results.md ----
    md = [
        "## Backtest results - rolling-origin day-ahead, final "
        f"{TEST_WEEKS} weeks ({len(test_index)} hourly forecasts)",
        "",
        overall.round(2).to_markdown(index=False),
        "",
        f"**Diebold-Mariano** ({dm['model_1']} vs {dm['model_2']}, squared-error loss, "
        f"h={HORIZON}, HLN-adjusted): statistic = **{dm['statistic']:.2f}**, "
        f"p-value = **{dm['p_value']:.2g}**",
        "",
        f"**LightGBM P10-P90 interval:** empirical coverage {coverage:.1f}% "
        "(nominal 80%)",
        "",
    ]
    (REPORTS / "results.md").write_text("\n".join(md))
    print(f"Wrote {REPORTS / 'results.md'}")


# ------------------------------------------------------------------ main ----
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["all", "sarimax", "lgbm", "quantiles", "metrics"],
                        default="all", help="run one backtest stage (predictions are cached)")
    parser.add_argument("--refit-days", type=int, default=LGBM_REFIT_DAYS,
                        help="LightGBM refit cadence in the backtest")
    args = parser.parse_args()

    ensure_dirs()
    t0 = time.time()
    df = load_processed()
    feats = build_features(df)
    test_days, test_index = test_window(feats)
    print(f"Backtest window: {test_days[0]} .. {test_days[-1]}  ({len(test_days)} days, "
          f"{len(test_index)} hourly predictions)")

    if args.only in ("all", "sarimax"):
        run_baselines_and_sarimax(df, feats)
    if args.only in ("all", "lgbm"):
        print("Running LightGBM backtest ...")
        run_lgbm(feats, args.refit_days)
    if args.only in ("all", "quantiles"):
        print("Running LightGBM quantile backtest ...")
        run_quantiles(feats, QUANTILE_REFIT_DAYS)
    if args.only in ("all", "metrics"):
        compute_metrics(feats)
    print(f"Stage '{args.only}' finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
