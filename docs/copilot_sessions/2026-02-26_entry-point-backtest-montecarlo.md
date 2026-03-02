# 2026-02-26: JSON-Driven Monte Carlo Backtest CLI Tool (Modular Refactor v2)

## Date and Context

2026-02-26 — Implements GitHub issue "Implement JSON-driven Monte Carlo
backtest CLI tool (modular refactor v2)" on branch
`copilot/refactor-json-cli-backtest-tool-again`.

## Problem Statement

`PyTAAA_backtest_sp500_pine_refactored.py` hardcodes paths and model-
specific configuration.  The goal was to create a reusable, JSON-driven
CLI tool with a modular architecture that:
- Derives all paths from a JSON config file
- Supports all trading models dynamically (sp500_pine, naz100_hma, etc.)
- Uses timestamped output filenames to prevent overwrites
- Uses standardised exit codes (0=success, 1=error, 2=config, 3=data)
- Makes zero modifications to existing code

## Solution Overview

A thin Click CLI entry point (`pytaaa_backtest_montecarlo.py`) delegates
to a new `functions/backtesting/` package that provides modular helpers
for configuration, parameter exploration, CSV output, and Monte Carlo
orchestration.

## Key Changes

### New files

| File | Purpose |
|------|---------|
| `pytaaa_backtest_montecarlo.py` | Click CLI entry point (~160 lines) |
| `functions/backtesting/__init__.py` | Package with re-exports |
| `functions/backtesting/config_helpers.py` | Path extraction and validation |
| `functions/backtesting/parameter_exploration.py` | 3-phase random parameter generation |
| `functions/backtesting/output_writers.py` | CSV / JSON export helpers |
| `functions/backtesting/monte_carlo_runner.py` | Monte Carlo orchestration loop |
| `tests/test_backtesting_config_helpers.py` | 10 unit tests |
| `tests/test_backtesting_parameter_exploration.py` | 9 unit tests |
| `tests/test_backtesting_output_writers.py` | 5 unit tests |
| `tests/test_pytaaa_backtest_montecarlo.py` | 4 CLI tests |

### JSON configs updated

Added `"backtest_monte_carlo_trials": 250` to the `Valuation` section of:
- `pytaaa_generic.json`
- `pytaaa_model_switching_params.json`
- `pytaaa_sp500_pine_montecarlo.json`

## Technical Details

### CLI Usage

```bash
# Default trials (from JSON or 250)
uv run python pytaaa_backtest_montecarlo.py --json pytaaa_sp500_pine_montecarlo.json

# Override trials
uv run python pytaaa_backtest_montecarlo.py --json config.json --trials 3
```

### Output filenames

CSV results: `{model_id}_montecarlo_{YYYY-M-D}_{runnum}.csv`
Optimised params: `{model_id}_optimized_{YYYY-M-D}.json`

Both are written to `{performance_store}/pngs/`.

### Model identifier extraction

The model identifier (e.g. `sp500_pine`) is extracted from the
second-to-last component of the `webpage` path in the JSON config.

### Lazy import pattern

`PyTAAA_backtest_sp500_pine_refactored.py` contains module-level code
that executes on import.  `monte_carlo_runner.py` imports
`run_single_monte_carlo_realization` lazily (inside the function body)
to prevent side-effects at import time.

### 3-phase parameter generation

- Phase 1 (iter < trials/4): Broad triangular sampling for exploration
- Phase 2 (trials/4 ≤ iter < trials-1): Base defaults with single-parameter variation
- Phase 3 (iter == trials-1): Linux-edition defaults

## Testing

28 unit tests added; all passing.  CodeQL reported 0 alerts.

Run with:
```bash
PYTHONPATH=$(pwd) uv run pytest tests/test_backtesting_*.py tests/test_pytaaa_backtest_montecarlo.py -v
```

## Follow-up Items

- 5 local E2E tests (require production HDF5 data) to be run by the
  repository owner after merging
- Consider adding a `--dry-run` flag for CI/CD validation without data
