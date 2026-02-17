## Plan: Rolling Window Data Quality Filtering

Implement a 50-day rolling window filter to zero signals for stocks with insufficient quote history (<50% valid non-constant data), ensuring portfolio defaults to 100% CASH when data quality is inadequate. Applied after technical indicators but before monthly rebalancing. Window size configurable via JSON parameters.

**Steps**
1. Create [functions/rolling_window_filter.py](functions/rolling_window_filter.py) with `apply_rolling_window_filter(adjClose, signal2D, window_size)` function implementing validation logic
2. Add `window_size` (default 50) and `enable_rolling_filter` (default true) to JSON configuration schema and example files ([pytaaa_generic.json](pytaaa_generic.json), [pytaaa_model_switching_params.json](pytaaa_model_switching_params.json))
3. Integrate filter call in [functions/output_generators.py](functions/output_generators.py) in `compute_portfolio_metrics()` after `computeSignal2D()` but before monthly signal holding logic, passing `params['window_size']`
4. Integrate filter call in [functions/dailyBacktest.py](functions/dailyBacktest.py) in `computeDailyBacktest()` after `computeSignal2D()` calls, passing `params['window_size']`
5. Integrate filter call in [functions/dailyBacktest_pctLong.py](functions/dailyBacktest_pctLong.py) after `computeSignal2D()` calls for consistency in historical analysis, passing `params['window_size']`
6. Implement error handling for edge cases (insufficient data, NaN values)
7. Create comprehensive unit tests in [tests/](tests/) for filter function and integration
8. Update documentation in [docs/](docs/) explaining the filter mechanism and configuration

**Verification**
Run full backtest with/without filter enabled; verify <20% performance degradation; ensure filtered stocks allocate to CASH; all unit/integration tests pass; code coverage >90% for new code; test different window_size values.

**Decisions**
- Chose signal-level filtering over Sharpe ratio approach for robustness
- Fixed 50-day window as default based on user clarification, but made configurable via JSON
- 50% threshold for valid non-constant data
- Integration after technical indicators in live system (compute_portfolio_metrics) and backtests for consistency
- Removed deprecated PyTAAA.py references, using current entry point pytaaa_main.py -> run_pytaaa.py -> PortfolioPerformanceCalcs
- Made window_size configurable in JSON instead of hardcoded constant for flexibility

This plan is structured for agentic coding assistance, with clear file paths, function signatures, and step-by-step implementation guidance. Each step includes specific integration points and configuration requirements.