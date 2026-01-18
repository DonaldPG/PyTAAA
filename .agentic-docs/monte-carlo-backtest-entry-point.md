# PyTAAA Monte Carlo Backtesting Entry Point - Product Plan

## Mission

**Pitch:** A dedicated CLI entry point for extensive Monte Carlo parameter optimization in PyTAAA trading strategies, enabling users to run hundreds of trials with individual performance plots for each parameter set. For now, the intent is to use the plots for each monte carlo trial plus the csv file to manually evaluate what ideal set of parameters should be used for the production version of this trading model

**Users:** PyTAAA developers and quantitative traders who need to optimize trading strategy parameters through statistical simulation.

**Problem:** 
- Current production system includes limited Monte Carlo trials (16) as part of daily updates
- No dedicated tool for extensive parameter optimization
- Summary plots aggregate all trials, making it hard to analyze individual parameter sets
- Production code changes risk affecting daily operations

**Differentiators:** 
- Isolated entry point that doesn't affect production runs
- Configurable number of trials (default 100+)
- Individual summary plots for each trial
- Fixed parameters for reproducible optimization
- Direct integration with existing PyTAAA backtest logic

**Key Features:** 
- CLI interface with configurable trial count
- Fixed file paths and base parameters
- Random generation of trading parameters (MA periods, thresholds, etc.)
- Individual PNG plots for each trial's performance
- Sharpe ratio and return metrics calculation
- Buy & hold comparison for each trial
- Results logging and best parameter identification

## Technical Architecture

**Framework:** Python CLI application using Click framework
**Storage:** 
- JSON for configuration parameters
- HDF5 for historical quote data
- PNG files for performance plots
- CSV for results summary
**GUI:** Matplotlib for generating performance charts
**Environment/Dependency Management:** UV for Python package management
**Testing:** Integration with existing PyTAAA test suite
**Environments:** Local development with UV virtual environments
**Repository:** Git with feature branches for development

## Roadmap

**Phase 1: Core MVP (2-4 days)**
- [x] Update pytaaa-backtest-montecarlo.py entry point script. It does a good job specifying fixed parameters and basic parameters, but needs to call code that only performs montecarlo trials instead of production code (run_pytaaa) that is mainly focused on daily price updates, html file updates, and start of month stock selections. I want to get most of the current click.options from a single input json file instead of from CLI options using click. The only exceptions are the n_trials param and the json to use for other click.options that are in the file now: 
    '--symbols_file', required=True, help='Path to symbols file'
    '--performance_store', required=True, help='Path to performance store directory'
    '--webpage_store', required=True, help='Path to webpage store directory'
    '--stock_list', default='Naz100', help='Stock list name'
    '--uptrend_signal_method' default='percentileChannels', help='Uptrend signal method'
- [x] Implement parameter generation using MonteCarloBacktest class
- [x] Integrate with existing backtest logic from PyTAAA_backtest_sp500_pine_refactored.py
- [x] Generate individual plots for each trial
- [x] Basic results logging to CSV. Use columns currently used by functions/dailyBacktest_pctLong.py and add comuns for buy and hold compound annual growth rates (CAGR) for the same time intervals as Sharpe ratios. Also add CAGR for these time intervals for the trading system with current monte carlo parameters for each trial 

**Phase 1, Part 2: Code Consolidation (1-2 days)**
- [ ] Use a copy of dailyBacktest_pctLong.py in src/backtest as the foundation for the core backtest logic because:

    - More comprehensive signal method support
    - Better plotting capabilities (can be made optional)
    - More detailed performance metrics
    - Proven stability with multiple uptrend methods

- [ ] Modify src/backtest/dailyBacktest_pctLong.py to:

    - Add a return_results parameter that returns metrics dict instead of plotting
    - Extract the Monte Carlo loop into a separate runner function
    - Make signal method configurable rather than hardcoded

- [ ] For Monte Carlo runner:

    - Use the parameter generation from montecarlo_runner.py
    - Add CAGR calculations to match the CSV output
    - Keep individual trial plotting but make it optional
    - Add progress tracking for large trial counts

- [ ] Create src/backtest/functions/ directory and copy modified backtest functions for better organization
- [ ] Modify dailyBacktest_pctLong.py to add return_results parameter that returns metrics dict instead of plotting
- [ ] Extract Monte Carlo loop into a separate runner function in src/backtest/montecarlo_runner.py
- [ ] Make signal method configurable rather than hardcoded in the backtest logic
- [ ] Update Monte Carlo runner to use parameter generation from existing MonteCarloBacktest class
- [ ] Add CAGR calculations to match CSV output requirements
- [ ] Make individual trial plotting optional in the runner
- [ ] Add progress tracking for large trial counts

**Phase 2: Differentiators (1-2 days)**
- [ ] Add best parameter tracking based on Sharpe ratio
- [ ] Implement buy & hold comparison to compare with each trial, but only needs computing once
- [ ] Add configurable parameter ranges
- [ ] Create summary report of top performing parameter sets
- [ ] Add trial progress indicators

**Phase 3: Scale & Polish (3-5 days)**
- [ ] Optimize performance for large trial counts
- [ ] Add parallel execution for multiple trials
- [ ] Enhanced plotting with comparative metrics
- [ ] Parameter sensitivity analysis
- [ ] Export results in multiple formats (JSON, CSV, PDF)


## Decision Log

| Date | ID | Status | Category | Decision | Context | Consequences |
|------|----|--------|----------|----------|---------|--------------|
| 2025-12-10 | DEC-001 | Pending | Architecture | Use existing MonteCarloBacktest class vs create new | Need to decide whether to extend existing class or create specialized version | Affects code reuse and maintenance |
| 2025-12-10 | DEC-002 | Pending | UI/UX | Individual plots per trial vs aggregated summary | User requested individual plots for detailed analysis | Increases storage requirements and generation time |
| 2025-12-10 | DEC-003 | Pending | Performance | Sequential vs parallel trial execution | Large number of trials may benefit from parallelization | Requires additional dependencies and complexity |
| 2025-12-10 | DEC-004 | Pending | Integration | Direct call to backtest logic vs API wrapper | Need to isolate backtest execution from production updates | Ensures no impact on production system |

## Implementation Notes

The entry point should directly call the Monte Carlo backtest logic from the existing PyTAAA system (similar to how `run_pytaaa` includes it), but modified to:
- code from folder functions should be re-used if possible. if modifications are required, start with a copy of functions that are put in src/backtest/ or src/backtest/functions using the location and organization of folders and files that makes codebase more modern and easy to maintain
- Run configurable number of trials (default 250 instead of 16)
- Generate individual summary plots for each trial instead of aggregated plots
- Use fixed parameters for file paths and base configuration
- Allow optimization of trading parameters through random generation

This ensures the new entry point is a specialized version of the Monte Carlo backtest without affecting the production daily update process.