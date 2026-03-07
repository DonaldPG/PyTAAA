# Session Summary: Active Mask for Ex-Index Stocks

## Date and Context

March 2, 2025 — continuation of the orchestration-refactor branch.
Previous session had identified EBAY appearing in the portfolio despite
being removed from the Nasdaq-100 index around February 26, 2026.

## Problem Statement

Stocks that have been removed from the index (e.g., EBAY) were
receiving positive portfolio weights because:

1. `loadQuotes_fromHDF` loads ALL 215 historical symbols from HDF5,
   not just the ~100 currently in the index.
2. Removed stocks have trailing NaN in HDF5 (yfinance stops updating
   them). After `cleantoend`, these become constant flat prices.
3. `computeSignal2D` has no knowledge of index membership and may
   produce an "up" or neutral signal for flat-priced ex-members.
4. `sharpeWeightedRank_2D` then assigns them positive weight.

This affected both live portfolio recommendations and historical
backtesting.

## Solution Overview

Build a boolean `active_mask` (shape: n_stocks × n_days) from the raw
NaN pattern in HDF5 data BEFORE cleaning. Apply it to `signal2D` and
`signal2D_daily` to force signals to 0 for inactive (ex-index) stocks.

The mask is built by examining each stock's price series:
- **Trailing NaN** = stock was removed from the index (inactive for
  those dates and all subsequent dates).
- **Leading NaN** = stock was not yet listed/added (inactive for those
  dates).
- **Interior NaN** (gaps while in index) = still considered active.

CASH is always marked active regardless of NaN pattern.

## Key Changes

### `functions/data_loaders.py`
- Added `_build_active_mask_from_raw(raw_adjclose)`: builds the
  boolean mask from the raw HDF5 NaN pattern before cleaning.
- Updated `load_quotes_for_analysis()` with new keyword parameter
  `include_active_mask: bool = False`. When `True`, returns a 4-tuple
  `(adjClose, symbols, datearray, active_mask)`. Default `False`
  preserves backward compatibility (3-tuple).
- Prints a summary line on load: how many symbols are active on the
  last date vs. inactive (removed from index).

### `functions/output_generators.py`
- `compute_portfolio_metrics()`: Added `active_mask=None` parameter.
  After the monthly hold forward-fill and after the rebalance mismatch
  check, applies the mask to both `signal2D` and `signal2D_daily`
  before calling `sharpeWeightedRank_2D`.
- `write_rank_list_html()`: Fixed company name lookup — was using
  hardcoded `companyNames.txt` (does not exist); now uses
  `read_company_names_local()` which resolves `Naz100_companyNames.txt`.
  Renamed column header to "Rank (start of month)". Fixed a stale
  variable reference (`sym_fmt` → `sym_stripped`). Accepts optional
  `datearray` parameter.

### `functions/PortfolioPerformanceCalcs.py`
- Calls `load_quotes_for_analysis` with `include_active_mask=True`
  and unpacks 4-tuple.
- Passes `active_mask=active_mask` to `compute_portfolio_metrics`.
- Passes `datearray` to `write_rank_list_html`.

### `functions/backtesting/monte_carlo_runner.py`
- Calls `load_quotes_for_analysis` with `include_active_mask=True`
  and unpacks 4-tuple.
- Passes `active_mask=active_mask` to `run_single_monte_carlo_realization`.

### `functions/backtesting/core_backtest.py`
- `run_single_monte_carlo_realization()`: Added `active_mask=None`
  parameter; passes it through to `execute_single_backtest`.
- `execute_single_backtest()`: Added `active_mask=None` parameter;
  applies mask to `signal2D` and `signal2D_daily` after the rolling
  window filter and monthly hold forward-fill, before weight
  computation.

## Technical Details

- **NaN detection timing**: The mask is built AFTER CASH is appended
  but BEFORE `interpolate`/`cleantobeginning`/`cleantoend` run, so
  the trailing NaN fingerprint is still present for removed stocks.
- **Interior NaN tolerance**: Only leading and trailing NaN are
  treated as inactive; interior NaN (data gaps while in index) do not
  shrink the active window.
- **All callers backward-compatible**: `include_active_mask=False`
  by default; callers that do NOT pass `include_active_mask=True`
  (e.g., `studies/`, `PyTAAA_backtest_sp500_pine_refactored.py`,
  tests) continue to receive a 3-tuple unchanged.

## Testing

- All pre-existing tests pass (83 pass, 1 pre-existing failure in
  `test_backtesting_parameter_exploration.py` unrelated to this work).
- Ran `PYTHONPATH=$(pwd) uv run pytest tests/ -x -q`.

## Commits

- `fe233cb` — shares/buyprice formatting (previous session)
- `f15748b` — active_mask implementation + rank table HTML fixes

## Follow-up Items

- **Buy-and-hold benchmark** (`BuyHoldFinalValue`) still averages over
  all 215 symbols including ex-members. A complete fix would apply
  the `active_mask` to the `value` array before averaging, computing
  B&H only from stocks active at each date. Left for a future session.
- The pre-existing test failure in `generate_random_parameters` should
  be investigated and fixed separately.
