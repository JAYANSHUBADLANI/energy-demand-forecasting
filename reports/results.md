## Backtest results - rolling-origin day-ahead, final 12 weeks (2016 hourly forecasts)

| model               |   MAPE_% |   RMSE_MW |   MAE_MW |
|:--------------------|---------:|----------:|---------:|
| LightGBM            |     4.68 |   2219.11 |  1352.76 |
| SARIMAX(daily)+temp |     6.06 |   2483.55 |  1712.31 |
| Naive-24h           |     7.9  |   3271.59 |  2249.02 |
| SeasonalNaive-168h  |     9.01 |   3772.58 |  2555.4  |

**Diebold-Mariano** (LightGBM vs SARIMAX(daily)+temp, squared-error loss, h=24, HLN-adjusted): statistic = **-1.46**, p-value = **0.14**

**LightGBM P10-P90 interval:** empirical coverage 68.5% (nominal 80%)
