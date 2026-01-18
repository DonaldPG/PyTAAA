| Computation Aspect | dailyBacktest_pctLong.py | montecarlo_runner.py |
| --- | --- | --- |
| Signal Method Support | Multiple: HMAs, percentileChannels, etc. via uptrendSignalMethod param | Hardcoded to percentileChannels only |
| Signal Computation | TAfunctions with configurable methods, MA calculations, filters | percentileChannel_2D from refactored file with fixed lowPct/hiPct |
| Weight Calculation | sharpeWeightedRank_2D with configurable constraints | Same sharpeWeightedRank_2D but with random weight factors |
| Portfolio Value Calculation | Full time series with monthly rebalancing | Full time series with monthly rebalancing |
| Performance Metrics | Sharpe, returns, drawdowns for Life/3Yr/1Yr/6Mo/3Mo/1Mo periods | Sharpe, returns, CAGRs for 15Yr/10Yr/5Yr/3Yr/2Yr/1Yr periods |
| Buy & Hold Comparison | Computed for same periods, plotted as red line | Computed once for all trials, used in CSV comparison |
| Data Normalization | Normalizes to $10,000 starting value | Same normalization approach |
| Plotting Complexity | Comprehensive: histograms, stock price overlays, Monte Carlo distributions, detailed stats table | Simple: portfolio value line plot with basic title |
| Output Format | PNG plot with embedded statistics | CSV log + individual PNG plots per trial |
| Monte Carlo Execution | Multiple trials (randomtrials) in single execution | Single trial per execution, multiple separate runs |
| Parameter Variation | Fixed params with optional variation logic | Random params generated per trial using MonteCarloBacktest |
| Data Persistence | Saves results to HDF5 | No HDF5 saving, only CSV logging |
| Error Handling | Continues execution with fallbacks | Continues with minimal results dict |
| Code Structure | Monolithic function with embedded plotting | Modular with separate runner calling backtest logic |

## Recommendation for src/backtest

Use a copy of dailyBacktest_pctLong.py in src/backtest as the foundation for the core backtest logic because:

- More comprehensive signal method support
- Better plotting capabilities (can be made optional)
- More detailed performance metrics
- Proven stability with multiple uptrend methods

Modify src/backtest/dailyBacktest_pctLong.py to:

- Add a return_results parameter that returns metrics dict instead of plotting
- Extract the Monte Carlo loop into a separate runner function
- Make signal method configurable rather than hardcoded

For Monte Carlo runner:

- Use the parameter generation from montecarlo_runner.py
- Add CAGR calculations to match the CSV output
- Keep individual trial plotting but make it optional
- Add progress tracking for large trial counts

This gives you the best of both: flexible, comprehensive backtesting with efficient Monte Carlo optimization.