"""Streamlit dashboard for the day-ahead electricity demand forecasting project.

Tabs
----
1. Load profile explorer - interactive seasonal profiles of the demand series
2. Forecast             - next-24h forecast with the P10-P90 interval
3. Model comparison     - backtest metrics, per-hour errors, DM test

Run:  streamlit run app.py        (after running the pipeline, see README)
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.config import FIGURES, LOCAL_TZ, PROCESSED_FILE, REPORTS, TARGET

st.set_page_config(page_title="Day-Ahead Load Forecasting", page_icon="⚡", layout="wide")

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@st.cache_data
def load_data() -> pd.DataFrame | None:
    if not PROCESSED_FILE.exists():
        return None
    df = pd.read_csv(PROCESSED_FILE, index_col="time", parse_dates=["time"])
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    local = df.index.tz_convert(LOCAL_TZ)
    return df.assign(hour=local.hour, weekday=local.weekday, month=local.month,
                     year=local.year)


st.title("⚡ Day-Ahead Electricity Demand Forecasting - Spain")
st.caption(
    "Hourly national load, 4 years. Baselines vs SARIMAX vs LightGBM with "
    "P10-P90 prediction intervals. See the README for methodology."
)

df = load_data()
if df is None:
    st.error(
        "Processed data not found. Run the pipeline first:\n\n"
        "```\npython -m src.data_ingestion\npython -m src.features\n"
        "python -m src.evaluate\npython -m src.forecast\n```"
    )
    st.stop()

tab_explore, tab_forecast, tab_models = st.tabs(
    ["📊 Load profile explorer", "🔮 Forecast", "🏁 Model comparison"]
)

# ------------------------------------------------------------- tab 1 ----
with tab_explore:
    left, right = st.columns([1, 3])
    with left:
        years = sorted(df["year"].unique().tolist())
        year_sel = st.multiselect("Years", years, default=years)
        view = st.radio("View", ["Hour × weekday heatmap", "Intraday profile",
                                 "Daily series", "Load vs temperature"])
    sub = df[df["year"].isin(year_sel)] if year_sel else df

    with right:
        if view == "Hour × weekday heatmap":
            pivot = sub.pivot_table(index="hour", columns="weekday", values=TARGET, aggfunc="mean")
            pivot.columns = [WEEKDAYS[c] for c in pivot.columns]
            fig, ax = plt.subplots(figsize=(9, 6))
            im = ax.imshow(pivot, aspect="auto", cmap="inferno")
            ax.set_xticks(range(7), pivot.columns)
            ax.set_yticks(range(0, 24, 2), pivot.index[::2])
            ax.set_ylabel("Hour of day (local)")
            fig.colorbar(im, label="Mean load (MW)")
            st.pyplot(fig, use_container_width=True)

        elif view == "Intraday profile":
            fig, ax = plt.subplots(figsize=(10, 5))
            for label, s in (("Weekday", sub[sub.weekday < 5]), ("Weekend", sub[sub.weekday >= 5])):
                prof = s.groupby("hour")[TARGET].mean()
                ax.plot(prof.index, prof.values, lw=2.5, label=label)
            ax.set_xlabel("Hour of day (local)")
            ax.set_ylabel("Mean load (MW)")
            ax.legend()
            ax.grid(alpha=0.4)
            st.pyplot(fig, use_container_width=True)

        elif view == "Daily series":
            daily = sub[TARGET].resample("D").mean()
            st.line_chart(daily, height=420)

        else:  # Load vs temperature
            daily = sub.resample("D").mean(numeric_only=True).dropna(subset=[TARGET, "temp"])
            fig, ax = plt.subplots(figsize=(9, 6))
            sc = ax.scatter(daily["temp"], daily[TARGET], s=10, alpha=0.4,
                            c=daily.index.month, cmap="twilight")
            ax.set_xlabel("Daily mean temperature (°C)")
            ax.set_ylabel("Daily mean load (MW)")
            fig.colorbar(sc, label="Month")
            st.pyplot(fig, use_container_width=True)

# ------------------------------------------------------------- tab 2 ----
with tab_forecast:
    fc_file = REPORTS / "forecast_next24h.csv"
    if not fc_file.exists():
        st.info("No forecast found - run `python -m src.forecast` first.")
    else:
        fc = pd.read_csv(fc_file, index_col="time", parse_dates=["time"])
        history = df[TARGET].iloc[-7 * 24 :]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(history.index, history.values, color="black", lw=1.3, label="Observed")
        ax.fill_between(fc.index, fc["p10"], fc["p90"], alpha=0.25, color="tab:blue",
                        label="P10-P90")
        ax.plot(fc.index, fc["point"], color="tab:blue", lw=2.2, label="Point forecast")
        ax.axvline(history.index.max(), color="grey", ls="--", lw=1)
        ax.set_ylabel("Load (MW)")
        ax.legend(loc="upper left")
        fig.autofmt_xdate()
        st.pyplot(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Peak forecast", f"{fc['point'].max():,.0f} MW")
        c2.metric("Trough forecast", f"{fc['point'].min():,.0f} MW")
        c3.metric("Mean band width", f"{(fc['p90'] - fc['p10']).mean():,.0f} MW")
        with st.expander("Forecast table"):
            st.dataframe(fc.round(0))

# ------------------------------------------------------------- tab 3 ----
with tab_models:
    metrics_file = REPORTS / "metrics_overall.csv"
    if not metrics_file.exists():
        st.info("No backtest results - run `python -m src.evaluate` first.")
    else:
        overall = pd.read_csv(metrics_file)
        st.subheader("Rolling-origin day-ahead backtest (final 12 weeks)")
        st.dataframe(overall.round(2), hide_index=True, use_container_width=True)

        dm_file = REPORTS / "dm_test.json"
        if dm_file.exists():
            dm = json.loads(dm_file.read_text())
            st.markdown(
                f"**Diebold-Mariano** ({dm['model_1']} vs {dm['model_2']}): "
                f"statistic `{dm['statistic']:.2f}`, p-value `{dm['p_value']:.2g}` - "
                + ("the accuracy difference is **statistically significant**."
                   if dm["p_value"] < 0.05 else "not significant at the 5% level.")
            )

        per_hour_file = REPORTS / "metrics_per_hour.csv"
        if per_hour_file.exists():
            per_hour = pd.read_csv(per_hour_file)
            fig, ax = plt.subplots(figsize=(11, 5))
            for name, sub in per_hour.groupby("model"):
                ax.plot(sub["hour"], sub["MAPE_%"], marker="o", ms=3, label=name)
            ax.set_xlabel("Hour of day (local)")
            ax.set_ylabel("MAPE (%)")
            ax.legend()
            ax.grid(alpha=0.4)
            st.pyplot(fig, use_container_width=True)

        bt_png = FIGURES / "backtest_last_week.png"
        if bt_png.exists():
            st.image(str(bt_png), caption="Backtest - final week, LightGBM band vs actual")
