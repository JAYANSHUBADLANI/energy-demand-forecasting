# Day-Ahead Electricity Demand Forecasting: Spain

[![tests](https://github.com/JAYANSHUBADLANI/energy-demand-forecasting/actions/workflows/tests.yml/badge.svg)](https://github.com/JAYANSHUBADLANI/energy-demand-forecasting/actions/workflows/tests.yml)

Forecasting the next 24 hours of national electricity load from 4 years of
hourly demand and multi-city weather data, with honest baselines, a classical
statistical model, a gradient-boosting main model, prediction intervals, a
rolling-origin backtest, and a Streamlit dashboard.

![Next-24h forecast](reports/figures/forecast_next24h.png)

## Why this problem matters

Day-ahead load forecasts drive the daily power procurement cycle: utilities and
grid operators bid volumes into the day-ahead market based on these numbers.
Buy too little and the shortfall must be covered at volatile intraday/balancing
prices; buy too much and the surplus is sold back at a loss. Because the
volumes are enormous (Spanish demand averages ~28-29 GW), **even a ~1%
reduction in MAPE translates into materially lower imbalance and procurement
costs**, which is why utilities keep dedicated teams on this exact problem.
Point forecasts alone are not enough: procurement is a decision under
uncertainty, so this project also produces **P10-P90 prediction intervals**.

## Data

Kaggle: [Hourly energy demand generation and weather](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather)
(nicholasjhana), Spain, 2015-2018, hourly.

| File | Contents | Used here |
|---|---|---|
| `energy_dataset.csv` | national load, generation mix, prices (ENTSO-E) | target: **`total load actual`** |
| `weather_features.csv` | weather for Valencia, Madrid, Bilbao, Barcelona, Seville (OpenWeather) | temperature (+ humidity, wind, pressure, clouds), averaged across the 5 cities |

Cleaning (in `src/data_ingestion.py`): tz-aware timestamps parsed to UTC,
duplicate rows dropped (the weather file contains 3,076 duplicated
(city, hour) rows), the series re-indexed to a complete hourly range, gaps of
≤ 24 h time-interpolated (36 missing load values here), Kelvin → °C,
whitespace-polluted city names fixed.
Exact counts for the data you run on are written to `reports/data_quality.json`.

## Methodology

```
raw CSVs ─► ingestion/cleaning ─► EDA ─► feature engineering ─► models ─► rolling-origin backtest ─► next-24h forecast + app
            src/data_ingestion    src/eda    src/features       src/models   src/evaluate              src/forecast, app.py
```

**Task definition.** At the end of day *D*, predict all 24 hours of day *D+1*
(the day-ahead market timeline). Every load-derived feature (lags and rolling
statistics) is shifted ≥ 24 h, so it only uses load observed at the forecast
origin. That makes LightGBM a *direct* day-ahead forecaster with no recursive
error accumulation. `tests/test_features.py` checks it: the load from a cut-off
hour onward is altered and no feature in the following 24 hours may change (a
deliberately leaky 1-hour lag is included to confirm the check can fail).
Weather is the one input that is not origin-safe, see **Weather inputs** below.

**Features** (`src/features.py`): calendar in local time: hour, weekday,
month, weekend, **Spanish public holidays** (`holidays` package), cyclical
sin/cos encodings; load lags (24 h, 48 h, 168 h); day-ahead-safe rolling mean
(24 h, 168 h) and std (24 h); city-averaged temperature and **temperature²**
(the EDA shows the classic U-shape: electric heating below ~15 °C,
air-conditioning above ~22 °C).

![Load vs temperature](reports/figures/load_vs_temperature.png)

**Weather inputs.** `temp` and `temp_sq` are the city-averaged temperature
*observed at the target hour*, so the backtest assumes a perfect weather
forecast. SARIMAX likewise receives the observed daily mean temperature of the
day being forecast. The live next-24h forecast (`src/forecast.py`) is stricter
and feeds in 24-hour persistence instead (the temperature at the same hour
yesterday). No numerical weather prediction is used anywhere, so the backtest
errors below are a best case for a system fed with real forecast weather.
`tests/test_features.py` pins this down: changing one hour's temperature
changes only that hour's `temp` and `temp_sq`.

**Models** (`src/models/`):

| Model | Role |
|---|---|
| Naive-24h (yesterday) | mandatory benchmark |
| Seasonal naive-168h (last week) | mandatory benchmark |
| SARIMAX on daily means, weekly seasonality, exogenous `temp` + centred `temp²` | classical statistical reference; order selected by AIC over a (p,1,q)×(P,D,Q)₇ grid; ranked table in `reports/sarimax_order_selection.csv`; daily forecasts disaggregated to hourly via the train-period hour-of-week profile |
| **LightGBM** on hourly data, full feature set | **main model** |
| LightGBM quantile (pinball loss, α = 0.10/0.50/0.90) | P10-P90 prediction intervals |

**Evaluation protocol** (`src/evaluate.py`): rolling-origin day-ahead backtest
over the **final 12 weeks** (~2,000 hourly forecasts). The origin advances one
day at a time; LightGBM is refit weekly, SARIMAX absorbs each observed day via
its Kalman filter (full refit every 4 weeks). Metrics: MAPE, RMSE, MAE,
overall and per hour of day. The top two models are compared with a
**Diebold-Mariano test** (squared-error loss, h = 24, long-run variance from the
first 23 autocovariances with equal weights as in the original DM test,
Harvey-Leybourne-Newbold small-sample correction).

## Results

Rolling-origin day-ahead backtest, final 12 weeks (2018-10-08 to 2018-12-30,
2,016 hourly forecasts), with the target hour's weather taken as known (see
**Weather inputs**):

| Model | MAPE (%) | RMSE (MW) | MAE (MW) |
|:---|---:|---:|---:|
| **LightGBM** | **4.68** | **2219.1** | **1352.8** |
| SARIMAX(daily)+temp | 6.06 | 2483.5 | 1712.3 |
| Naive-24h | 7.90 | 3271.6 | 2249.0 |
| SeasonalNaive-168h | 9.01 | 3772.6 | 2555.4 |

LightGBM has the lowest error on every metric and at every hour of the day, and
roughly halves the MAPE of the same-hour-last-week baseline.

**Diebold-Mariano** (LightGBM vs SARIMAX, squared-error loss, h = 24,
HLN-adjusted): statistic = **-1.46**, p-value = **0.14**. On this test the gain
over SARIMAX is **not statistically significant** in a 12-week window.

**Why, and what the sensitivity checks say** (`python -m src.sensitivity`, output
in `reports/dm_sensitivity.json`). Squared error is dominated by a few days that
both models miss badly: the five worst days carry 43% of each model's squared
error, and they include the early-December holiday block, 1 November and
Christmas Day. Under absolute-error loss the same comparison gives p = 0.002
(p = 0.0003 with Newey-West weights), and averaged by day LightGBM has the lower
MAE on 67 of 84 days (paired t-test p = 0.003, Wilcoxon p = 0.0002). On ordinary
weekdays its MAE is 1,096 MW against 1,538 MW for SARIMAX; on weekends and
public holidays the gap narrows to 1,866 against 2,061 MW. These checks were
added after the squared-error result was known, so I read them as evidence that
the gain is real on typical days, not as a replacement for the headline test.

**Prediction intervals:** LightGBM P10-P90 empirical coverage **68.5%**
(nominal 80%), over-confident, a known trait of quantile GBMs and a limitation
(conformal calibration is a listed next step).

Per-hour errors follow the usual load-forecasting pattern: the morning ramp
(07:00-08:00 local) is hardest at 7.4-7.7% MAPE for LightGBM, and the night
hours (00:00-04:00) are easiest at 2.7-3.2%:

![MAPE by hour](reports/figures/error_by_hour.png)

![Backtest final week](reports/figures/backtest_last_week.png)

## Repository structure

```
energy-demand-forecasting/
├── data/raw/                  # place the two Kaggle CSVs here (gitignored)
├── data/processed/            # pipeline output (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb           # narrative EDA
│   └── 02_modeling.ipynb      # narrative modelling walkthrough
├── src/
│   ├── config.py              # paths + constants
│   ├── data_ingestion.py      # cleaning, merge, optional Kaggle API download
│   ├── synthetic_data.py      # schema-exact sample generator (testing only)
│   ├── eda.py                 # all EDA figures
│   ├── features.py            # calendar/holiday/lag/rolling/weather features
│   ├── models/
│   │   ├── baselines.py       # naive-24h, seasonal naive-168h
│   │   ├── sarimax_model.py   # AIC order selection, fit, disaggregation
│   │   └── lgbm_model.py      # point + quantile models
│   ├── evaluate.py            # rolling-origin backtest, DM test
│   ├── sensitivity.py         # DM robustness: loss, weights, by-day view
│   └── forecast.py            # next-24h forecast with P10-P90 band
├── app.py                     # Streamlit dashboard
├── tests/                     # leakage, ingestion, DM test and model checks
├── .github/workflows/tests.yml  # CI: pytest on Python 3.10 and 3.12
├── reports/                   # metrics, results.md, data_quality.json
│   └── figures/               # all plots
├── pytest.ini
├── requirements.txt
├── requirements-dev.txt       # requirements.txt + pytest
└── README.md
```

## How to run

```bash
# 1. environment (Python 3.10 to 3.12; CI runs 3.10 and 3.12)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
#    on macOS LightGBM also needs OpenMP: brew install libomp

# 2. data: either place the two Kaggle CSVs in data/raw/, or configure
#    Kaggle API credentials (~/.kaggle/kaggle.json) and let step 3 download.
#    (For a quick smoke test without the real data: python -m src.synthetic_data)

# 3. pipeline (each script also runs standalone)
python -m src.data_ingestion
python -m src.eda
python -m src.features
python -m src.models.sarimax_model     # AIC order selection report
python -m src.models.lgbm_model        # train + persist models
python -m src.evaluate                 # 12-week rolling-origin backtest (a few minutes)
python -m src.sensitivity              # Diebold-Mariano robustness checks
python -m src.forecast                 # next-24h forecast + band

# 4. dashboard
streamlit run app.py

# 5. tests (no Kaggle files needed, the suite builds its own seeded sample)
pip install -r requirements-dev.txt
pytest -q
```

## Limitations & next steps

* **Holiday interactions.** Holidays enter as a single flag; bridge days and
  holiday×hour interaction effects (a holiday reshapes the whole intraday
  profile) are not modelled. This is the main weakness in the results above:
  the days both models miss worst sit around holidays, and they decide the
  squared-error Diebold-Mariano verdict. Regional holidays moved to another day
  (as with 8 December in 2018) are invisible to a national flag.
* **Weather is assumed known.** The backtest uses the temperature observed at
  the target hour (a perfect forecast), so its errors are a best case. The live
  forecast in `src/forecast.py` uses 24-hour persistence instead, which is
  cruder. A production system would consume numerical weather predictions,
  adding their own error. Quantifying how much accuracy is lost when the
  backtest is fed persistence or forecast weather is a natural next step.
* **Single country, single series.** National aggregate only, no regional
  decomposition, and conclusions may not transfer to other grids.
* **Pre-COVID data.** 2015-2018 excludes the structural demand shifts of
  2020+; the model would need retraining and drift monitoring in production.
* **Natural extensions:** holiday-profile features, weather-forecast inputs,
  conformalized intervals, hyperparameter search, and a scheduled retraining
  job with monitoring.
