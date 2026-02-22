# NASDAQ100 Oracle Lows-Highs Delay Studies

## Purpose

Research studies to quantify the value of predictive accuracy in stock trading by comparing oracle scenarios with perfect vs. delayed information about price extrema.

## Scope

- **Data Source**: Real NASDAQ100 prices from existing HDF5 files
- **Date Range**: User-configurable via JSON (clipped to available data)
- **Signal Generation**: Oracle knowledge of centered-window lows/highs
- **Portfolio Simulation**: Monthly rebalance with top-N stock selection
- **Research Question**: How much performance degrades with information delay?

## Directory Structure

```
nasdaq100_scenarios/
├── params/              # JSON configuration files
├── results/             # Output metrics and summary data (gitignored)
├── plots/               # Generated visualizations (gitignored)
├── notes/               # Research notes and observations
├── data_loader.py       # HDF5 data loading with date clipping
├── oracle_signals.py    # Extrema detection and signal generation
├── portfolio_backtest.py # Monthly portfolio simulation
├── plotting.py          # Multi-curve visualization
└── run_full_study.py    # Main orchestrator script
```

## Dependencies on Production Code

This study code **reuses** (does not duplicate) production PyTAAA functions:

- `functions/UpdateSymbols_inHDF5.py::loadQuotes_fromHDF` — Load NASDAQ100 HDF5 data
- `functions/data_loaders.py::load_quotes_for_analysis` — Data preprocessing pattern
- `functions/dailyBacktest.py` — Month boundary and rebalance logic reference
- `run_normalized_score_history.py` — Multi-curve plotting conventions

**Isolation Principle**: All study-specific logic lives in `studies/`. No modifications to production code.

## Expected Outputs

### Per-Scenario Outputs
- Portfolio history time series (CSV or numpy array)
- Holdings log (which stocks selected at each rebalance)
- Performance metrics (final value, CAGR, max drawdown, Sharpe ratio)

### Aggregate Outputs
- Multi-curve plots: portfolio value over time for all scenarios
- Parameter sensitivity panels: 3×3 grids (delay × window × top_n)
- Summary JSON: all scenarios' metrics in machine-readable format
- Buy-and-hold baseline for reference

### Documentation Outputs
- Wiki page: `docs/pytaaa-oracle-delay-studies.md`
- Research notes in `notes/` folder with observations

## Usage

### Quick Start

```bash
# Set PYTHONPATH to project root
export PYTHONPATH=$(pwd)

# Run with default parameters
uv run python studies/nasdaq100_scenarios/run_full_study.py \
  --config studies/nasdaq100_scenarios/params/default_scenario.json

# Check outputs
ls studies/nasdaq100_scenarios/results/
ls studies/nasdaq100_scenarios/plots/
```

### Custom Configuration

Create a new JSON file in `params/` based on `default_scenario.json`:

```bash
cp studies/nasdaq100_scenarios/params/default_scenario.json \
   studies/nasdaq100_scenarios/params/my_custom_study.json

# Edit date range, parameter lists, etc.
# Then run with custom config
uv run python studies/nasdaq100_scenarios/run_full_study.py \
  --config studies/nasdaq100_scenarios/params/my_custom_study.json
```

### Testing

```bash
# Run unit tests for individual modules
uv run pytest studies/nasdaq100_scenarios/tests/

# Run integration test with minimal parameters
uv run python studies/nasdaq100_scenarios/test_integration.sh
```

## Interpretation Guidelines

### What Scenarios Represent

1. **Perfect Oracle (delay=0)**: Upper bound on performance with perfect foresight
2. **Delayed Oracle (delay>0)**: Realistic scenarios with information lag
3. **Window Sensitivity**: How detection period affects extrema identification
4. **Top-N Sensitivity**: Diversification vs concentration tradeoff

### Performance Gaps Meaning

- **Large gap (oracle vs delay=40)**: High value of real-time information
- **Small gap**: Strategy is robust to information delays
- **Negative excess return**: Oracle signal worse than buy-and-hold (unexpected)

### Caveats and Limitations

- **Survivorship Bias**: Uses current NASDAQ100 list (may not match historical)
- **No Slippage**: Assumes perfect execution at extrema prices
- **Simplified Costs**: Transaction costs are fixed per trade (not percentage-based)
- **Knowable vs Unknowable**: Forward return ranking is intentionally unknowable
- **Monthly Cadence**: More reactive strategies (daily rebalance) not tested

## Contributing

This is a research study branch. Key principles:

1. **No Production Contamination**: Keep all study code under `studies/`
2. **Reuse, Don't Duplicate**: Import from `functions/`, don't copy-paste
3. **Document Assumptions**: Every parameter choice needs justification
4. **Track Temporary Files**: Use TODO list in implementation plan

## References

- Implementation Plan: [plans/experiment-trading-lows-highs-delays.md](../../plans/experiment-trading-lows-highs-delays.md)
- PyTAAA Architecture: [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md)
- Coding Conventions: [.github/copilot-instructions.md](../../.github/copilot-instructions.md)

## Status

**Current Phase**: Phase 1 - Project Scaffolding and Configuration  
**Last Updated**: 2026-02-21  
**Branch**: `experiment/trading-lows-highs-delays`
