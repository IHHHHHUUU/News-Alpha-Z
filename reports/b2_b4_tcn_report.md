# B2+B4 TCN Factor Report

## Summary

- Factor: `FullTextNewsAlpha_raw = gate_news_prob * (fusion_pred - factor_only_pred)`
- Configuration: B2 mean chunk pooling, B4 conventional mixture, dual-stream TCN
- Lookback: 30 trading days
- TCN: 4 layers, `kernel_size=3`, `dilations=[1, 2, 4, 8]`, `hidden_dim=128`, `dropout=0.1`
- Label: `future_20d_market_adjusted_return`
- Train period: 2018-01-02 to 2020-12-31
- Validation period: 2021-01-04 to 2021-12-31
- Test period: 2022-01-03 to 2023-12-28

## Training Result

- Best validation loss: `0.005961623951720962`
- Best epoch: `1`
- Stopped epoch: `6`
- Batch size: `512`
- Note: GPU memory usage was low on 4090D; future B2 runs can increase batch size.

## IC / RankIC

| Split | Rows | IC | ICIR | RankIC | RankICIR | Avg coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 66,353 | -0.003486 | -0.514800 | -0.003803 | -0.555899 | 87.77 |
| Valid | 22,428 | 0.018564 | 2.363897 | 0.029668 | 3.708597 | 89.00 |
| Test | 44,500 | 0.002227 | 0.306965 | 0.008662 | 1.198156 | 85.44 |

## Long-Short Diagnostics

| Split | Annualized return | Sharpe | Max drawdown |
| --- | ---: | ---: | ---: |
| Train | -0.523075 | -0.424761 | -0.969811 |
| Valid | 3.578811 | 2.463452 | -0.882968 |
| Test | 0.893998 | 1.300796 | -0.877254 |

## Outputs

- Factor tables:
  - `data/factors/b2_b4_tcn_train.parquet`
  - `data/factors/b2_b4_tcn_valid.parquet`
  - `data/factors/b2_b4_tcn_test.parquet`
- Evaluation summaries:
  - `data/reports/b2_b4_tcn/summary_by_split.csv`
  - `data/reports/b2_b4_tcn/test/ic_by_date.csv`
  - `data/reports/b2_b4_tcn/test/rolling_rankic.csv`
  - `data/reports/b2_b4_tcn/test/long_short_returns.csv`
  - `data/reports/b2_b4_tcn/test/decile_returns.csv`
- Test plots:
  - `data/reports/b2_b4_tcn/test/plots/cumulative_long_short.png`
  - `data/reports/b2_b4_tcn/test/plots/rolling_60d_rankic.png`
  - `data/reports/b2_b4_tcn/test/plots/decile_returns.png`
  - `data/reports/b2_b4_tcn/test/plots/coverage.png`
  - `data/reports/b2_b4_tcn/test/plots/average_gate_news_prob.png`

## Notes

- Mean gate news probability is low: test split average `0.029732`.
- Test RankIC is positive but modest. Validation is much stronger than test, so the result should be treated as promising but not yet stable.
- The run selected epoch 1 by validation loss; later epochs reduced train loss but worsened validation loss.
