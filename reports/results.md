## Backtest results - rolling-origin day-ahead, final 12 weeks (2016 hourly forecasts)

| model               |   MAPE_% |   RMSE_MW |   MAE_MW |
|:--------------------|---------:|----------:|---------:|
| LightGBM            |     1.66 |    570.6  |   454.41 |
| SARIMAX(daily)+temp |     2.22 |    821.55 |   630.44 |
| SeasonalNaive-168h  |     2.34 |    866.48 |   644.72 |
| Naive-24h           |     5.04 |   1863.69 |  1385.08 |

**Diebold-Mariano** (LightGBM vs SARIMAX(daily)+temp, squared-error loss, h=24, HLN-adjusted): statistic = **-6.09**, p-value = **1.3e-09**

**LightGBM P10-P90 interval:** empirical coverage 71.0% (nominal 80%)
