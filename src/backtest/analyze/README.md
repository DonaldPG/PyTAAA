# Backtest Results Analysis

This script analyzes the Monte Carlo backtest results from the Excel spreadsheet `pytaaa_sp500_pine_montecarlo.xlsx` and identifies the top-performing parameter sets based on various risk-adjusted performance metrics.

## Purpose

The goal is to extract and copy the PNG plot files corresponding to the best parameter combinations from a large set of Monte Carlo simulations. This allows for visual inspection of the equity curves and performance charts for the most promising trading strategies.

## Methodology

1. **Data Source**: The Excel file contains 700 rows of backtest results, each representing a unique combination of parameters tested in Monte Carlo simulations.

2. **Key Columns**:
   - `run`: The timestamp identifier for the simulation batch (e.g., `run_20251214_1243`)
   - `trial`: The trial number within that batch (0-99, formatted as 4-digit zero-padded)
   - Various performance metrics: Sharpe ratios, Sortino ratios, CAGR, drawdowns, etc. over different time periods

3. **Ranking Criteria**: The script sorts the results by the primary metric: **sum ranks** (ascending, lower sum ranks is better).

4. **Selection Process**:
   - Sort the entire dataset by sum ranks in ascending order (lowest sum ranks = best)
   - Select the top 25 parameter sets
   - Match each set to its corresponding PNG file based on `run` and `trial` columns

5. **PNG File Naming**: Files are named as `PyTAAA_monteCarloBacktest_run_{timestamp}_{trial:04d}.png`

## Output

- **Destination Folder**: `/Users/donaldpg/pyTAAA_data/sp500_pine/backtest_best_params`
- **Files Copied**: 25 PNG files representing the top-performing parameter combinations
- **Console Output**: Lists each copied file for verification

## Usage

Run the script from the project root:

```bash
python src/backtest/analyze/analyze_backtest_results.py
```

## Dependencies

- pandas
- openpyxl (for Excel reading)
- shutil, os (standard library)

## Rationale for Metric Selection

- **Sum Ranks**: A composite ranking metric that aggregates multiple performance indicators. Lower values indicate better overall ranking across various criteria.
- **Derivation**: "Sum ranks" is calculated by ranking each parameter set on individual performance metrics (e.g., Sharpe ratio, Sortino ratio, CAGR, drawdown, etc.), assigning a rank position to each (1 = best, 2 = second best, etc.), and then summing these ranks across all metrics. A lower sum indicates the parameter set performed well across multiple dimensions simultaneously, providing a holistic assessment rather than optimizing for a single metric.