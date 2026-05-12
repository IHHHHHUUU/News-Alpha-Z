# Data Prep QA Report

## File Inventory
| File | Exists | Size | Shape |
|---|---:|---:|---:|
| `data/processed/universe_clean_top100.csv` | True | 1.3 KB | 89 x 3 |
| `data/processed/prices_clean.parquet` | True | 4.4 MB | 154,954 x 8 |
| `data/features/return_labels_clean.parquet` | True | 8.8 MB | 133,368 x 12 |
| `data/features/price_factors_clean.parquet` | True | 24.5 MB | 133,368 x 22 |
| `data/processed/panel_price_only_clean.parquet` | True | 29.7 MB | 133,368 x 28 |
| `data/interim/news_clean_universe_2018_2023.csv` | True | 1.1 GB | 355,633 x 7 |
| `data/processed/calendar_clean.parquet` | True | 10.4 KB | 1,759 x 1 |
| `data/interim/chunks_clean_2018_2023.parquet` | True | 563.8 MB | 813,770 x 9 |
| `data/processed/panel_with_news_coverage_clean.parquet` | True | 29.8 MB | 133,368 x 28 |

## Universe And Tickers
- Clean universe size from txt: 89.
- Universe CSV rows: 89.
- Ticker counts: prices=89, panel=89, chunks=86.
- Universe tickers missing in prices: none.
- Chunk tickers outside clean universe: none.
- Universe tickers with no chunks: JNJ, JPM, PFE.

## Date Ranges
- prices: 2017-01-03 to 2023-12-28.
- labels: 2018-01-02 to 2023-12-28.
- factors: 2018-01-02 to 2023-12-28.
- panel_price_only: 2018-01-02 to 2023-12-28.
- chunks: 2018-01-02 to 2023-12-28.
- calendar: 2017-01-03 to 2023-12-28.
- Chunk dates outside trading calendar: 0.
- Chunk rows outside panel date range: 0.

## No-Lookahead Alignment
- Pytest result: NOT RUN because `pytest` is not installed in the active Python environment.
- Manual alignment assertions: PASS.
- Real chunk event alignment failures when recomputed: 0.

Pre-close samples around 16:00 America/New_York:
_No samples found._

After-close samples around 16:00 America/New_York:
| ticker | publish_time | ny_time | date | expected_date | alignment_ok | title |
| --- | --- | --- | --- | --- | --- | --- |
| ADBE | 2023-12-16 21:00:00+00:00 | 2023-12-16 16:00:00-05:00 | 2023-12-18 | 2023-12-18 | True | The Zacks Analyst Blog Highlights Microsoft, Amazon, Alphabet, Adobe and Meta Pl |
| AMGN | 2023-12-16 21:00:00+00:00 | 2023-12-16 16:00:00-05:00 | 2023-12-18 | 2023-12-18 | True | DIA, GS, MCD, AMGN: ETF Inflow Alert |
| BA | 2023-12-16 21:00:00+00:00 | 2023-12-16 16:00:00-05:00 | 2023-12-18 | 2023-12-18 | True | Airbus wins order from easyJet for additional 157 aircraft |
| BA | 2023-12-16 21:00:00+00:00 | 2023-12-16 16:00:00-05:00 | 2023-12-18 | 2023-12-18 | True | US STOCKS-Wall St set to open higher as investors pin hopes on Fed rate cuts |
| BKNG | 2023-12-16 21:00:00+00:00 | 2023-12-16 16:00:00-05:00 | 2023-12-18 | 2023-12-18 | True | BKNG Factor-Based Stock Analysis |

Weekend samples:
| ticker | publish_time | ny_time | date | expected_date | alignment_ok | title |
| --- | --- | --- | --- | --- | --- | --- |
| AAPL | 2022-06-05 00:00:00+00:00 | 2022-06-04 20:00:00-04:00 | 2022-06-06 | 2022-06-06 | True | 2 Top Stocks to Buy With $500 |
| AAPL | 2022-06-05 00:00:00+00:00 | 2022-06-04 20:00:00-04:00 | 2022-06-06 | 2022-06-06 | True | If I Could Buy Only 1 Warren Buffett Stock, This Would Be It |
| AAPL | 2022-06-05 00:00:00+00:00 | 2022-06-04 20:00:00-04:00 | 2022-06-06 | 2022-06-06 | True | Stock Market Definition |
| AAPL | 2022-06-06 00:00:00+00:00 | 2022-06-05 20:00:00-04:00 | 2022-06-06 | 2022-06-06 | True | 3 Tech Stocks Investors Should Always Buy During Market Weakness |
| AAPL | 2022-06-06 00:00:00+00:00 | 2022-06-05 20:00:00-04:00 | 2022-06-06 | 2022-06-06 | True | 91% of Warren Buffett's Portfolio Is in These 4 Sectors |

## Chunk Quality
- Rows: 813,770. Unique chunk IDs: 813,770.
- Max token_count: 256; rows above 256: 0.
- Max chunks per ticker-date: 64; ticker-date groups above 64: 0.
- Empty chunk_text rows: 0; token_count < 5 rows: 3,302; duplicate chunk_id rows: 0.

Chunks per stock-day distribution:
| Metric | Value |
|---|---:|
| count | 66284.00 |
| mean | 12.28 |
| std | 13.75 |
| min | 1.00 |
| 50% | 7.00 |
| 75% | 15.00 |
| 90% | 29.00 |
| 95% | 45.00 |
| 99% | 64.00 |
| max | 64.00 |

Chunks per stock-day bins:
| Bin | Stock-days |
|---|---:|
| 1 | 1,849 |
| 2-4 | 19,887 |
| 5-16 | 30,204 |
| 17-32 | 8,722 |
| 33-64 | 5,622 |
| >64 | 0 |

Top ticker coverage by chunk_count:
| ticker | stock_days | news_count | chunk_count |
| --- | --- | --- | --- |
| GOOG | 1213 | 8344 | 37697 |
| DIS | 1132 | 8435 | 33275 |
| WMT | 1256 | 8300 | 30612 |
| AMD | 1437 | 7197 | 25333 |
| CVX | 1437 | 7260 | 23860 |
| MSFT | 415 | 6545 | 22870 |
| AAPL | 388 | 6438 | 22545 |
| TSLA | 411 | 6871 | 21962 |
| XOM | 1052 | 6556 | 21204 |
| GS | 1398 | 6838 | 20721 |
| COST | 1293 | 4338 | 19000 |
| KO | 1329 | 4486 | 18416 |
| WFC | 1352 | 5024 | 18240 |
| CRM | 1261 | 4453 | 17646 |
| BA | 921 | 5819 | 17410 |
| V | 1343 | 4445 | 17330 |
| NKE | 1330 | 4434 | 16654 |
| GE | 1346 | 4542 | 15734 |
| MRK | 1320 | 4701 | 15396 |
| QCOM | 1266 | 4220 | 14663 |

Bottom ticker coverage by chunk_count:
| ticker | stock_days | news_count | chunk_count |
| --- | --- | --- | --- |
| ORLY | 513 | 752 | 3033 |
| ODFL | 444 | 670 | 2872 |
| FAST | 475 | 768 | 2715 |
| EXC | 506 | 779 | 2520 |
| ROP | 426 | 634 | 2510 |
| XEL | 518 | 730 | 2459 |
| CPRT | 339 | 499 | 2173 |
| ANSS | 267 | 330 | 1473 |
| CCEP | 202 | 236 | 841 |
| REGN | 56 | 118 | 372 |
| ADP | 35 | 72 | 214 |
| AVGO | 1 | 7 | 27 |
| LOW | 3 | 7 | 24 |
| LLY | 1 | 7 | 23 |
| LMT | 2 | 7 | 21 |
| IBM | 1 | 7 | 19 |
| KLAC | 4 | 7 | 19 |
| CTAS | 4 | 7 | 19 |
| LRCX | 5 | 7 | 18 |
| LIN | 3 | 7 | 18 |

## Coverage By Year
| year | stock_days | news_count | chunk_count |
| --- | --- | --- | --- |
| 2018 | 9558 | 20903 | 89601 |
| 2019 | 8889 | 18467 | 83529 |
| 2020 | 9623 | 23440 | 118351 |
| 2021 | 10323 | 23905 | 109152 |
| 2022 | 13487 | 39957 | 184978 |
| 2023 | 14404 | 53280 | 228159 |

## Panel With News Coverage
- Output: `data/processed/panel_with_news_coverage_clean.parquet`.
- Shape: 133,368 x 28.
- Stock-days with chunk_count > 0: 66,277.
- Sum news_count: 223,848; sum chunk_count: 813,751.
- Chunk ticker-date groups not present in the price-factor panel: 7 groups / 19 chunk rows (CRWD 2019-06-06, DDOG 2019-09-18, and five ZS dates in early 2018).

## Known Limitations And Next Step
- `pytest` is not installed in the active environment, so repository tests could not be executed with the requested command; equivalent manual assertions and real-data recomputation passed.
- News coverage is limited to the 89 ticker clean universe and the available FNSPID article subset.
- Three universe tickers have no final chunks, and 19 chunk rows do not merge into the stock-day panel because their ticker-date keys are absent from the price-factor panel.
- The next stage is offline FinBERT/DeBERTa chunk embedding after reviewing this QA report; no model download or GPU embedding was started during this QA run.
