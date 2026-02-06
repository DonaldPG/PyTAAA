# PyTAAA Backtest Module Documentation

## Overview

The `src/backtest/` package provides a modular, well-tested framework for Monte Carlo backtesting of trading strategies. This refactored code extracts functionality from the monolithic backtest script into reusable, testable components.

## Package Structure

```
src/backtest/
├── __init__.py          # Package exports
├── config.py            # Configuration classes and constants
├── montecarlo.py        # Monte Carlo simulation functions
└── plotting.py          # Visualization utilities
```

## Modules

### 1. Configuration (`src/backtest/config.py`)

Contains all constants, configuration parameters, and file paths used throughout the backtesting system.

#### Classes

**`TradingConstants`**
- Trading day constants for performance calculations
- `TRADING_DAYS_PER_YEAR = 252`
- Period constants: `TRADING_DAYS_1_YEAR`, `TRADING_DAYS_2_YEARS`, etc.
- `INITIAL_PORTFOLIO_VALUE = 10000.0`
- CAGR validation bounds

**`BacktestConfig`**
- Monte Carlo simulation settings (`DEFAULT_RANDOM_TRIALS = 250`)
- Plot configuration (`PLOT_Y_MIN`, `HISTOGRAM_BINS`)
- Parameter defaults for pyTAAA and Linux editions
- Exploration ranges for parameter optimization
- Symbol file configurations

**`FilePathConfig`**
- Data directory paths
- Output file paths
- Class methods for generating filenames:
  - `get_symbols_file()`
  - `get_output_csv_filename(date_str, runnum)`
  - `get_plot_filename(date_str, runnum, iter_num)`
  - `get_backtest_values_filepath()`

#### Usage Example

```python
from src.backtest.config import TradingConstants, BacktestConfig, FilePathConfig

# Use trading day constants
days_per_year = TradingConstants.TRADING_DAYS_PER_YEAR

# Get default parameter values
num_trials = BacktestConfig.DEFAULT_RANDOM_TRIALS

# Generate output filename
output_file = FilePathConfig.get_plot_filename("2025-01-15", "run001", 42)
```

---

### 2. Monte Carlo Simulation (`src/backtest/montecarlo.py`)

Functions for running Monte Carlo simulations to optimize trading strategy parameters.

#### Functions

**`random_triangle(low, mid, high, size=1)`**
- Generates random values using averaged uniform and triangular distributions
- Produces values more concentrated around the middle than pure uniform

**`create_temporary_json(base_json_fn, realization_params, iter_num)`**
- Creates temporary JSON configuration for a single Monte Carlo realization
- Returns path to temporary file

**`cleanup_temporary_json(temp_json_fn)`**
- Removes temporary JSON file after use

**`calculate_sharpe_ratio(daily_gains, trading_days=252)`**
- Calculates annualized Sharpe ratio from daily gain ratios

**`calculate_period_metrics(portfolio_value, trading_days=252)`**
- Returns dict with metrics for each period (15Y, 10Y, 5Y, 3Y, 2Y, 1Y)
- Each period contains: `sharpe`, `return`, `cagr`, `days`

**`calculate_drawdown_metrics(portfolio_value)`**
- Returns average drawdown for each time period

**`beat_buy_hold_test(strategy_metrics, buyhold_metrics)`**
- Weighted Sharpe ratio comparison
- Returns positive score if strategy beats buy & hold

**`beat_buy_hold_test2(strategy_metrics, buyhold_metrics, strategy_dd, buyhold_dd)`**
- Comprehensive test considering returns, positive returns, and drawdowns
- Returns score from 0 to 1

#### Classes

**`MonteCarloBacktest`**
- Manages Monte Carlo simulation execution
- Methods:
  - `generate_random_params(iteration, uptrendSignalMethod)` - Generate random parameters
  - `generate_variation_params(base_params, param_to_vary)` - Vary single parameter
  - `update_best_result(params, sharpe)` - Track best performing parameters

#### Usage Example

```python
from src.backtest.montecarlo import (
    MonteCarloBacktest,
    calculate_sharpe_ratio,
    calculate_period_metrics,
)

# Initialize Monte Carlo manager
mc = MonteCarloBacktest(
    base_json_fn="config.json",
    n_trials=250,
    hold_months=[1, 2, 3, 6, 12]
)

# Generate random parameters
params = mc.generate_random_params(
  iteration=0,
  uptrendSignalMethod='percentileChannels'
)

# Calculate Sharpe ratio
sharpe = calculate_sharpe_ratio(daily_gains)

# Get period metrics
metrics = calculate_period_metrics(portfolio_values)
print(f"5-Year Sharpe: {metrics['5Yr']['sharpe']:.2f}")
```

---

### 3. Plotting (`src/backtest/plotting.py`)

Functions and classes for creating backtest visualization plots.

#### Functions

**`calculate_plot_range(plotmax, ymin=7000.0)`**
- Calculates log-scale range for y-axis positioning

**`get_y_position(plotrange, fraction, ymin=7000.0)`**
- Calculates y-coordinate for text placement on log-scale plot

**`format_performance_metrics(sharpe, return_val, cagr, drawdown, show_cagr=True)`**
- Formats metrics for display
- Returns tuple of (formatted_sharpe, formatted_metric, formatted_drawdown)

**`create_monte_carlo_histogram(portfolio_values, datearray, n_bins, ymin, ymax)`**
- Creates histogram data for Monte Carlo visualization
- Returns 3D RGB array suitable for `imshow`

**`plot_signal_diagnostic(datearray, prices, symbol, nan_count)`**
- Plots diagnostic chart for a single symbol's signals

**`plot_lower_panel(q_minus_sma, month_pct_invested, datearray)`**
- Plots the lower panel with Q-SMA and percent invested

#### Classes

**`BacktestPlotter`**
- Main class for creating backtest visualization plots
- Constructor: `BacktestPlotter(plotmax, ymin, figsize)`
- Methods:
  - `setup_figure()` - Initialize matplotlib figure
  - `plot_performance_table(metrics, x_position, show_cagr, color)`
  - `plot_portfolio_values(buy_hold, trading, var_pct)`
  - `setup_x_axis_dates(datearray, max_labels)`
  - `add_info_text(symbols_file, last_symbols, beat_bh_pct, beat_bh_var_pct)`
  - `save_plot(output_dir, prefix, date_str, runnum, iteration)`

#### Usage Example

```python
from src.backtest.plotting import (
    BacktestPlotter,
    format_performance_metrics,
)

# Create plotter
plotter = BacktestPlotter(plotmax=1e9, ymin=7000.0)

# Format metrics for display
f_sharpe, f_cagr, f_dd = format_performance_metrics(
    sharpe=1.25,
    return_val=1.15,
    cagr=0.12,
    drawdown=-0.08,
    show_cagr=True
)
print(f"Sharpe: {f_sharpe}, CAGR: {f_cagr}, Drawdown: {f_dd}")
```

---

## Testing

Run all backtest module tests:

```bash
uv run pytest tests/test_backtest_config.py tests/test_backtest_montecarlo.py tests/test_backtest_plotting.py -v
```

Test coverage includes:
- **68 tests** covering all public functions and classes
- Configuration constants validation
- Monte Carlo parameter generation
- Sharpe ratio and metrics calculations
- Plotting utility functions

---

## Migration Guide

### Importing from New Modules

Replace old inline code with imports from the new modules:

```python
# Old way (inline code)
TRADING_DAYS = 252
cagr = (end_value / start_value) ** (252 / days) - 1.0

# New way (using config module)
from src.backtest.config import TradingConstants
cagr = (end_value / start_value) ** (TradingConstants.TRADING_DAYS_PER_YEAR / days) - 1.0
```

### Using Monte Carlo Functions

```python
# Old way (copy-pasted code)
def random_triangle(low, mid, high, size=1):
    # ... implementation

# New way (import from module)
from src.backtest.montecarlo import random_triangle
```

---

## Files

| File | Description |
|------|-------------|
| `PyTAAA_backtest_sp500_pine.py` | Original backtest script (DO NOT MODIFY) |
| `PyTAAA_backtest_sp500_pine_refactored.py` | Refactored version with module imports |
| `src/backtest/config.py` | Configuration constants and classes |
| `src/backtest/montecarlo.py` | Monte Carlo simulation functions |
| `src/backtest/plotting.py` | Visualization utilities |
| `tests/test_backtest_*.py` | Unit tests for each module |

---

## Version History

- **v1.0.0** (2025-12-08): Initial refactoring
  - Extracted configuration to `config.py`
  - Extracted Monte Carlo logic to `montecarlo.py`
  - Extracted plotting utilities to `plotting.py`
  - Added comprehensive unit tests (68 tests)
  - Updated documentation
