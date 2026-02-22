# PyTAAA Oracle Delay Studies

## Introduction
This study quantifies how much predictive accuracy matters by comparing
portfolio performance under perfect versus delayed knowledge of price extrema.
It is a research-only workflow that lives under the studies/ directory and does
not alter production code.

## Methodology
- Use centered-window extrema detection to identify local highs and lows.
- Generate oracle buy signals and optionally delay them.
- Simulate monthly rebalancing portfolios using top-N stock selection.
- Compare ranking methods, including forward-return oracle ranking and
  extrema-slope ranking.

## Assumptions
- Uses current NASDAQ100 constituents (survivorship bias).
- Assumes ideal execution at close prices.
- Simplified transaction cost model.
- Forward-return ranking is intentionally unknowable and used for
  sensitivity analysis.

## Scenarios
Scenarios are defined by:
- Window half-width for extrema detection.
- Delay days applied to signals or slope ranking.
- Top-N portfolio size.

Scenario grids are configured via JSON in:
- studies/nasdaq100_scenarios/params/

## Results Interpretation
- The delay=0 oracle is an upper bound on performance.
- Performance decay as delay increases indicates sensitivity to timing.
- Slope ranking indicates whether price trend strength adds signal value.

## Caveats
- Oracle information is not attainable in live trading.
- Monthly cadence may understate short-term effects.
- Results depend on the NASDAQ100 data window used.

## Usage Instructions

### Quick Start
```bash
export PYTHONPATH=$(pwd)

uv run python studies/nasdaq100_scenarios/run_full_study.py \
  --config studies/nasdaq100_scenarios/params/default_scenario.json
```

### Outputs
- Plots: studies/nasdaq100_scenarios/plots/
- Summary JSON: studies/nasdaq100_scenarios/results/

### Example Plots
- studies/nasdaq100_scenarios/plots/portfolio_histories.png
- studies/nasdaq100_scenarios/plots/parameter_sensitivity_total_return.png

### Tests
```bash
uv run python -m pytest studies/nasdaq100_scenarios/tests/
```

## References
- Implementation plan: plans/experiment-trading-lows-highs-delays.md
- Study README: studies/nasdaq100_scenarios/README.md
- Oracle signal logic: studies/nasdaq100_scenarios/oracle_signals.py
- Portfolio backtest: studies/nasdaq100_scenarios/portfolio_backtest.py

## Follow-up Recommendations
- Add a minimal integration test script for end-to-end runs.
- Document the JSON schema for scenario configuration.
- Validate sensitivity results on additional data windows.
