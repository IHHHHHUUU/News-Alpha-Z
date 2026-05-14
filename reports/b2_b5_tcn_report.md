# B2+B5 TCN Factor Report

## Summary

- Factor: `FullTextNewsAlpha_raw = gate_news_prob * (fusion_pred - factor_only_pred)`
- Configuration: B2 mean chunk pooling, B5 decoupled mixture, dual-stream TCN
- Lookback: 30 trading days
- TCN: 4 layers, `kernel_size=3`, `dilations=[1, 2, 4, 8]`, `hidden_dim=128`, `dropout=0.1`
- Label: `future_20d_market_adjusted_return`
- Train period: 2018-01-02 to 2020-12-31
- Validation period: 2021-01-04 to 2021-12-31
- Test period: 2022-01-03 to 2023-12-28

## Training Result

- Stage 1 factor best validation loss: `0.0059415892909980405` at epoch `7`
- Stage 2 fusion best validation loss: `0.006003934576848492` at epoch `21`
- Stage 3 gate best validation loss: `0.6931513671955545` at epoch `25`
- Batch size: `2048`
- Gate temperature: `0.5`

## IC / RankIC

| Split | Rows | IC | ICIR | RankIC | RankICIR | Avg coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 66,353 | -0.007451 | -1.070340 | -0.003796 | -0.557607 | 87.77 |
| Valid | 22,428 | 0.071706 | 9.935827 | 0.064644 | 8.018494 | 89.00 |
| Test | 44,500 | 0.018415 | 1.874801 | 0.023394 | 2.341468 | 85.44 |

## Long-Short Diagnostics

| Split | Annualized return | Sharpe | Max drawdown |
| --- | ---: | ---: | ---: |
| Train | -0.784941 | -0.798399 | -0.991079 |
| Valid | 141.770632 | 7.139309 | -0.624118 |
| Test | 1.758771 | 1.479710 | -0.992294 |

## Gate Diagnostics

- Train gate target mean: `0.499825`
- Valid gate target mean: `0.500019`
- Test mean gate news probability: `0.313164`
- The gate target distribution is tightly centered near 0.5, while the fitted gate assigns materially larger probabilities than B2+B4.

## Outputs

- Factor tables:
  - `data/factors/b2_b5_tcn_train.parquet`
  - `data/factors/b2_b5_tcn_valid.parquet`
  - `data/factors/b2_b5_tcn_test.parquet`
- Evaluation summaries:
  - `data/reports/b2_b5_tcn/summary_by_split.csv`
  - `data/reports/b2_b5_tcn/test/ic_by_date.csv`
  - `data/reports/b2_b5_tcn/test/rolling_rankic.csv`
  - `data/reports/b2_b5_tcn/test/long_short_returns.csv`
  - `data/reports/b2_b5_tcn/test/decile_returns.csv`
- Test plots:
  - `data/reports/b2_b5_tcn/test/plots/cumulative_long_short.png`
  - `data/reports/b2_b5_tcn/test/plots/rolling_60d_rankic.png`
  - `data/reports/b2_b5_tcn/test/plots/decile_returns.png`
  - `data/reports/b2_b5_tcn/test/plots/coverage.png`
  - `data/reports/b2_b5_tcn/test/plots/average_gate_news_prob.png`

## Notes

- Test IC and RankIC are both meaningfully higher than the B2+B4 TCN run.
- Validation is much stronger than test, so the result still needs robustness checks across seeds and batch sizes.
- The long-short max drawdown remains very large, so portfolio construction and decay diagnostics should be treated cautiously.
