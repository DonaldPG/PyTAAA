# Phase 3: Synthetic CAGR Data Generation & Validation

## Overview

Phase 3 validates that PyTAAA's backtest pipeline correctly selects high-return stocks and adapts to portfolio rotations when provided synthetic price data with **known CAGR ground truth**.

### Core Concept

We generate 5 years (1,260 trading days) of synthetic price data for 90 tickers across 9 CAGR tiers:

- Tier 1-10 stocks: +20% CAGR
- Tier 2-10 stocks: +15% CAGR
- Tier 3-10 stocks: +12% CAGR
- ...
- Tier 9-10 stocks: -6% CAGR

Every ~126 trading days (6 months), tickers rotate between tiers to test portfolio adaptation.

## Generated Artifacts

### Synthetic HDF5 File

Path: `studies/synthetic_cagr/data/synthetic_naz100.hdf5`
- Shape: 1,260 rows (trading days) × 90 columns (tickers)
- Date range: 2019-01-01 to 2023-10-30
- Tickers: SYNTH_+20%_01 through SYNTH_-6%_10
- Price data: Geometric trend with OpenSimplex noise

### Ground Truth CSV

Path: `studies/synthetic_cagr/data/ground_truth.csv`
- Columns: date, ticker, assigned_cagr, price
- Records: 113,400 (1,260 days × 90 tickers)
- Maps each date+ticker to assigned CAGR for validation

## Validation Pipeline

### Phase 3a: Data Generation

```bash
uv run python studies/synthetic_cagr/synthetic_hdf5_builder.py
```

- Generates the synthetic HDF5 and ground_truth.csv
- Takes ~30 seconds
- Output: 1,260 rows × 90 columns with realistic price series

### Phase 3b: Backtesting

```bash
uv run python studies/synthetic_cagr/backtest_runner.py
```

Runs `computeDailyBacktest()` for three models:
- `naz100_hma`: HMA-based signal
- `naz100_pine`: Percentile channel signal
- `naz100_pi`: SMA-based signal

Output: `portfolio_values.csv` with date index and per-model portfolio values

### Phase 3c: Evaluation

```bash
uv run python studies/synthetic_cagr/evaluate_synthetic.py
```

Three pass/fail evaluations:

**Evaluation 1: CAGR Pass Band (19%-21%)**
- Portfolio CAGR should be within [19%, 21%]
- Rationale: Tiers average to ~19-21%, so portfolio should achieve this
- Pass: All 3 models within band

**Evaluation 2: Selection Accuracy (≥70%)**
- Daily stock selection should favor high-CAGR tiers (≥+10%)
- Accuracy: fraction of top-N stocks from high-CAGR tiers
- Pass: Overall accuracy ≥ 70%

**Evaluation 3: Rotation Responsiveness (<60 days)**
- Portfolio weights should adapt within 60 days of rotation
- Rotations occur at: days 126, 252, 378, ..., 1134 (±10 day jitter)
- Pass: All models show weight changes near rotation dates

Output: `evaluation_report.txt` with per-evaluation results

### Phase 3d: Plotting

```bash
uv run python studies/synthetic_cagr/plot_results_synthetic.py
```

Generates portfolio value plots (one per model) showing:
- Black curve: Backtested portfolio (log scale, lw=4)
- Red dashed curve: Oracle +20% CAGR reference (lw=3)
- Title: Model name + Sharpe + Annual return + Avg DD
- Metrics table: Period, Sharpe, Annual Return, Avg DD

## Running Full Suite

```bash
uv run python studies/synthetic_cagr/synthetic_hdf5_builder.py
uv run python studies/synthetic_cagr/backtest_runner.py
uv run python studies/synthetic_cagr/evaluate_synthetic.py
uv run python studies/synthetic_cagr/plot_results_synthetic.py
```

## Expected Results

**If systems work correctly:**
- CAGR: 19-21% ✓
- Selection accuracy: 70-85% ✓
- Rotation responsiveness: Changes within 60 days ✓

**If look-ahead bias or selection bugs exist:**
- CAGR: >25% (buys winners too early)
- Selection accuracy: >95% (perfect foresight)
- Rotation responsiveness: Changes <5 days (instant detection)

## Directory Structure

```
studies/synthetic_cagr/
├── data/
│   ├── synthetic_naz100.hdf5
│   └── ground_truth.csv
├── plots/
│   ├── synthetic_naz100_hma.png
│   ├── synthetic_naz100_pine.png
│   └── synthetic_naz100_pi.png
├── experiment_output/
│   └── backtest_results/
│       ├── portfolio_values.csv
│       ├── backtest_metrics.csv
│       └── evaluation_report.txt
├── noise_utils.py
├── synthetic_hdf5_builder.py
├── backtest_runner.py
├── evaluate_synthetic.py
├── plot_results_synthetic.py
└── README.md
```

## Technical Notes

### Noise Generation

OpenSimplex 1D noise creates smooth, autocorrelated price movements:
```
price[t] = P0 * exp(CAGR/252 * t) * (1 + noise[t])
```

Noise amplitude ~1%, frequency scaled to real stock statistics.

### Rotation Mechanism

Ticker assignments permute every ~126 trading days (±10 day jitter),
maintaining 10 stocks per tier. Tests portfolio adaptation to regime change.

### Phase 4 Integration

Lightweight pytest tests in `tests/test_synthetic_backtest.py` use this
infrastructure on smaller (20 ticker × 252 day) synthetic data to validate
backtest correctness without real market data dependency.
