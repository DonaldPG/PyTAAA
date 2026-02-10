# PyTAAA System Summary

**Last Updated:** February 9, 2026

---

## 1. What PyTAAA Does

PyTAAA (**Py**thon **T**actical **A**sset **A**llocation **A**dvisor) is a quantitative stock trading system that:

1. **Ranks stocks** within a universe (Nasdaq 100 or S&P 500) using technical analysis signals — moving averages, channel fits, Sharpe-weighted rankings, and trend detection.
2. **Selects a portfolio** of the top-ranked stocks (typically 7) and computes target allocations.
3. **Backtests** the strategy against buy-and-hold baselines using historical data going back to ~1991.
4. **Generates trade recommendations** on a monthly rebalancing schedule (first weekday of each month).
5. **Publishes results** as static HTML dashboards with PNG charts, and optionally sends email notifications.
6. **Switches between multiple trading models** via the "Abacus" meta-model, which uses Monte Carlo–optimized normalized scoring to select the best-performing model each month.

PyTAAA is designed for **entertainment and research purposes only**. The author explicitly disclaims responsibility for investment decisions made using its output (see [`README.md`](../README.md:15)).

### Trading Models Available

| Model Name | Stock Universe | Signal Method | Description |
|---|---|---|---|
| `naz100_pine` | Nasdaq 100 | Pine Script methodology | Channel-based trend signals |
| `naz100_hma` | Nasdaq 100 | Hull Moving Averages | HMA-based uptrend detection |
| `naz100_pi` | Nasdaq 100 | Pi methodology (3 MAs) | Triple moving average crossover |
| `sp500_hma` | S&P 500 | Hull Moving Averages | HMA applied to broader market |
| `sp500_pine` | S&P 500 | Pine Script methodology | Channel signals on S&P 500 |
| `cash` | — | — | Money market / safe harbor position |

The **Abacus meta-model** dynamically switches between these models monthly based on normalized performance scoring across multiple lookback periods.

---

## 2. How to Use PyTAAA

### Prerequisites

- Python 3.11+ (managed via `uv`)
- Dependencies listed in [`pyproject.toml`](../pyproject.toml:1) and [`requirements.txt`](../requirements.txt)
- Access to stock quote data (Yahoo Finance via `yfinance`)
- A JSON configuration file (see [`pytaaa_generic.json`](../pytaaa_generic.json:1) for template)

### Initial Setup

1. **Create an HDF5 file** holding historical stock quotes:
   ```bash
   python re-generateHDF5.py
   ```
   > **Note:** [`re-generateHDF5.py`](../re-generateHDF5.py:1) uses legacy Python 2 syntax (`print` statements, `nose`, `la` imports) and is effectively non-functional in the current Python 3.11+ environment. Quote data is now managed through [`UpdateSymbols_inHDF5.py`](../functions/UpdateSymbols_inHDF5.py:679) via the `UpdateHDF_yf()` function.

2. **Configure the JSON parameters file** with email credentials, stock list, signal method, and valuation parameters. See [`pytaaa_generic.json`](../pytaaa_generic.json:1) for the full schema.

3. **Run PyTAAA**:
   ```bash
   # Modern JSON-based entry point
   uv run python pytaaa_main.py --json path/to/config.json

   # Legacy scheduler-based entry point
   python PyTAAA.py
   ```

### Daily Operations

```bash
# Daily portfolio tracking and webpage update
uv run python daily_abacus_update.py \
    --json /path/to/pytaaa_config.json
```

### Monthly Rebalancing Workflow

1. Run `recommend_model.py` to get the current model recommendation
2. Review model rankings and normalized scores
3. Execute trades manually on brokerage platform
4. Update `PyTAAA_holdings.params` with new positions

### Monte Carlo Optimization

```bash
# Optimize model-switching parameters
uv run python run_monte_carlo.py \
    --json pytaaa_model_switching_params.json \
    --search explore-exploit

# Transfer best parameters to config
uv run python update_json_from_csv.py \
    --csv abacus_best_performers.csv \
    --row 42 \
    --json pytaaa_model_switching_params.json
```

---

## 3. Entry Points

### Primary Entry Points

| Script | Purpose | How to Run |
|---|---|---|
| [`pytaaa_main.py`](../pytaaa_main.py:1) | Modern CLI entry point using Click; delegates to `run_pytaaa()` | `uv run python pytaaa_main.py --json config.json` |
| [`PyTAAA.py`](../PyTAAA.py:1) | Legacy scheduler-based entry point; runs on a timer loop | `python PyTAAA.py` |
| [`run_pytaaa.py`](../run_pytaaa.py:1) | Core execution logic (JSON-aware); called by `pytaaa_main.py` | Imported, not run directly |
| [`daily_abacus_update.py`](../daily_abacus_update.py:1) | Daily portfolio tracking with smart quote updates | `uv run python daily_abacus_update.py --json config.json` |
| [`recommend_model.py`](../recommend_model.py:1) | Generate model-switching recommendations | `uv run python recommend_model.py --json config.json` |
| [`run_monte_carlo.py`](../run_monte_carlo.py:1) | Monte Carlo parameter optimization | `uv run python run_monte_carlo.py --json config.json` |

### Utility Entry Points

| Script | Purpose | How to Run |
|---|---|---|
| [`update_json_from_csv.py`](../update_json_from_csv.py:1) | Transfer Monte Carlo results to JSON config | `uv run python update_json_from_csv.py --csv file.csv --row N --json config.json` |
| [`modify_saved_state.py`](../modify_saved_state.py:1) | Inspect/modify Monte Carlo saved state | `uv run python modify_saved_state.py --show` |
| [`run_normalized_score_history.py`](../run_normalized_score_history.py:1) | Plot historical normalized scores for all models | `uv run python run_normalized_score_history.py --json config.json` |
| [`compute_allocations.py`](../compute_allocations.py:1) | Compute new share allocations from target percentages | Used as library; see class `AllocationComputer` |
| [`compute_new_allocations.py`](../compute_new_allocations.py:1) | Alternative allocation computation (duplicate functionality) | Used as library |
| [`pytaaa_quotes_update.py`](../pytaaa_quotes_update.py:1) | Clean and fix stored quote data | `uv run python pytaaa_quotes_update.py --json config.json` |
| [`re-generateHDF5.py`](../re-generateHDF5.py:1) | Legacy HDF5 regeneration (Python 2 syntax, non-functional) | Not usable in current environment |

### Entry Point Descriptions

#### [`pytaaa_main.py`](../pytaaa_main.py:1) — Modern CLI Entry Point
Uses Click to accept a `--json` argument pointing to a JSON configuration file. Delegates all work to [`run_pytaaa()`](../run_pytaaa.py:32). This is the recommended way to run the core PyTAAA analysis pipeline.

#### [`PyTAAA.py`](../PyTAAA.py:1) — Legacy Scheduler Entry Point
The original entry point. Uses a custom [`scheduler.py`](../scheduler.py:1) module to run [`IntervalTask()`](../PyTAAA.py:58) on a configurable timer (default: every 24 hours for 15 days). Reads configuration from legacy `.params` files rather than JSON. Still functional but superseded by `pytaaa_main.py`.

#### [`run_pytaaa.py`](../run_pytaaa.py:32) — Core Execution Logic
The main computational pipeline:
1. Loads JSON configuration and symbols file
2. Checks for stock universe changes (added/removed tickers)
3. Updates HDF5 quote data from Yahoo Finance
4. Runs [`PortfolioPerformanceCalcs()`](../functions/PortfolioPerformanceCalcs.py:37) to compute rankings and weightings
5. Retrieves current holdings and prices
6. Calculates portfolio value, lifetime profit, and annualized returns
7. Calls [`calculateTrades()`](../functions/calculateTrades.py:20) for trade recommendations
8. Sends email notifications (if portfolio value changed)
9. Generates HTML web page via [`writeWebPage()`](../functions/WriteWebPage_pi.py:167)

#### [`daily_abacus_update.py`](../daily_abacus_update.py:1) — Daily Portfolio Tracker
Automated daily wrapper that:
- Detects the active trading model from `PyTAAA_holdings.params`
- Routes to the correct symbols file (Naz100/SP500) based on active model
- Updates quotes only when market is open and data is stale
- Generates HTML dashboard pages and PNG performance charts
- Designed for cron job automation (weekdays at 6:30 AM)

#### [`recommend_model.py`](../recommend_model.py:1) — Model Recommendation Engine
Generates trading recommendations using the Abacus model-switching methodology:
- Loads backtest data for all models via [`BacktestDataLoader`](../functions/abacus_backtest.py:20)
- Computes normalized performance scores using [`MonteCarloBacktest`](../functions/MonteCarloBacktest.py:1)
- Ranks models by composite score (Sharpe, Sortino, drawdown metrics)
- Shows recommendations for both target date and first weekday of month
- Generates `recommendation_plot.png` with visual analysis

#### [`run_monte_carlo.py`](../run_monte_carlo.py:1) — Parameter Optimizer
Runs thousands of backtesting simulations to find optimal:
- Lookback periods (3 values, permutation-invariant)
- Normalization parameters (central values, standard deviations)
- Performance metric weights
- Supports explore/exploit/explore-exploit search strategies
- Saves state for resume capability via `monte_carlo_state.pkl`

---

## 4. Functions Library — Stock Trading Methods

### Signal Generation ([`functions/TAfunctions.py`](../functions/TAfunctions.py:1))

This is the largest module (~4,100+ lines) containing the core technical analysis functions:

| Function | Description |
|---|---|
| [`computeSignal2D()`](../functions/TAfunctions.py:1493) | Main signal computation: generates 2D buy/sell signals using configurable uptrend methods (HMAs, percentile channels, min-max channels) |
| [`sharpeWeightedRank_2D()`](../functions/TAfunctions.py:2731) | Ranks stocks by Sharpe-weighted performance across multiple time windows; produces final portfolio weights |
| [`SMA()`](../functions/TAfunctions.py:1244), [`SMA_2D()`](../functions/TAfunctions.py:1331) | Simple Moving Average (1D and 2D) |
| [`hma()`](../functions/TAfunctions.py:1252) | Hull Moving Average — faster-responding MA used in HMA signal methods |
| [`dpgchannel()`](../functions/TAfunctions.py:549), [`dpgchannel_2D()`](../functions/TAfunctions.py:569) | Custom price channel computation (min/max envelope) |
| [`percentileChannel_2D()`](../functions/TAfunctions.py:521) | Percentile-based price channels for trend detection |
| [`recentTrendAndMidTrendChannelFitWithAndWithoutGap()`](../functions/TAfunctions.py:910) | Computes recent trend slopes with and without gap periods for momentum analysis |
| [`recentSharpeWithAndWithoutGap()`](../functions/TAfunctions.py:831) | Recent Sharpe ratio calculation with gap handling |
| [`recentTrendComboGain()`](../functions/TAfunctions.py:1075) | Composite trend/gain metric combining multiple timeframes |
| [`move_sharpe_2D()`](../functions/TAfunctions.py:1450) | Rolling Sharpe ratio across 2D stock arrays |
| [`move_martin_2D()`](../functions/TAfunctions.py:1745) | Rolling Martin ratio (Ulcer Performance Index) |
| [`interpolate()`](../functions/TAfunctions.py:200) | Linear interpolation of missing values in price series |
| [`cleantobeginning()`](../functions/TAfunctions.py:375), [`cleantoend()`](../functions/TAfunctions.py:430) | Fill NaN values at series boundaries |
| [`despike_2D()`](../functions/TAfunctions.py:1403) | Remove outlier spikes from price data |
| [`cleanspikes()`](../functions/TAfunctions.py:455) | Remove gradient outliers using standard deviation thresholds |

### Portfolio Performance ([`functions/PortfolioPerformanceCalcs.py`](../functions/PortfolioPerformanceCalcs.py:37))

| Function | Description |
|---|---|
| [`PortfolioPerformanceCalcs()`](../functions/PortfolioPerformanceCalcs.py:37) | Master function: loads quotes from HDF5, computes signals, runs backtests, generates plots, and returns ranked stock lists with weights and prices |

### Backtesting

| Module | Key Functions | Description |
|---|---|---|
| [`functions/dailyBacktest.py`](../functions/dailyBacktest.py:19) | [`computeDailyBacktest()`](../functions/dailyBacktest.py:19) | Simulates daily trading with configurable parameters (number of stocks, holding period, signal method, channel widths) |
| [`functions/dailyBacktest_pctLong.py`](../functions/dailyBacktest_pctLong.py:24) | [`plotRecentPerfomance3()`](../functions/dailyBacktest_pctLong.py:24), [`dailyBacktest_pctLong()`](../functions/dailyBacktest_pctLong.py:1571) | Generates performance plots with percent-long indicators and Monte Carlo overlays |
| [`functions/MonteCarloBacktest.py`](../functions/MonteCarloBacktest.py:1) | `MonteCarloBacktest` class | Monte Carlo simulation engine with Numba-optimized metrics computation, model ranking, and adaptive search strategies |

### Portfolio Metrics ([`functions/PortfolioMetrics.py`](../functions/PortfolioMetrics.py:1))

| Function | Description |
|---|---|
| [`calculate_sharpe_sortino_ratios()`](../functions/PortfolioMetrics.py:34) | Annualized Sharpe and Sortino ratios from portfolio values |
| [`calculate_cagr()`](../functions/PortfolioMetrics.py:77) | Compound Annual Growth Rate |
| [`calculate_avg_drawdown()`](../functions/PortfolioMetrics.py:111) | Average drawdown from peak |
| [`calculate_period_metrics()`](../functions/PortfolioMetrics.py:150) | Metrics for standard periods (3M, 6M, 1Y, 3Y, 5Y, 10Y, 20Y, 30Y, MAX) |
| [`analyze_model_switching_effectiveness()`](../functions/PortfolioMetrics.py:565) | Compares model-switching portfolio against individual models |

### Statistical Analysis ([`functions/allstats.py`](../functions/allstats.py:1))

| Method | Description |
|---|---|
| [`allstats.sharpe()`](../functions/allstats.py:9) | Sharpe ratio from daily price series (annualized, 252 trading days) |
| [`allstats.monthly_sharpe()`](../functions/allstats.py:24) | Sharpe ratio assuming monthly data |
| [`allstats.sortino()`](../functions/allstats.py:39) | Sortino ratio using lower partial moments |

### Data Management

| Module | Key Functions | Description |
|---|---|---|
| [`functions/UpdateSymbols_inHDF5.py`](../functions/UpdateSymbols_inHDF5.py:1) | [`UpdateHDF_yf()`](../functions/UpdateSymbols_inHDF5.py:679), [`loadQuotes_fromHDF()`](../functions/UpdateSymbols_inHDF5.py:39), [`createHDF()`](../functions/UpdateSymbols_inHDF5.py:1231) | Download quotes via yfinance, store/load from HDF5 |
| [`functions/readSymbols.py`](../functions/readSymbols.py:1) | [`read_symbols_list_web()`](../functions/readSymbols.py:271), [`read_symbols_list_local()`](../functions/readSymbols.py:183), [`get_symbols_changes()`](../functions/readSymbols.py:724) | Read symbol lists from web (Wikipedia) or local files; detect index changes |
| [`functions/quotes_adjClose.py`](../functions/quotes_adjClose.py:7) | [`downloadQuotes()`](../functions/quotes_adjClose.py:7) | Download adjusted close prices via yfinance |
| [`functions/clean_quote_data.py`](../functions/clean_quote_data.py:1) | [`fix_quotes()`](../functions/clean_quote_data.py:475) | Clean and repair stored quote data |
| [`functions/GetParams.py`](../functions/GetParams.py:1) | [`get_json_params()`](../functions/GetParams.py:219), [`get_holdings()`](../functions/GetParams.py:170), [`get_status()`](../functions/GetParams.py:489) | Configuration loading from JSON and legacy `.params` files |

### Trade Calculation ([`functions/calculateTrades.py`](../functions/calculateTrades.py:20))

| Function | Description |
|---|---|
| [`calculateTrades()`](../functions/calculateTrades.py:20) | Computes buy/sell/exchange recommendations with minimum trade thresholds ($400 exchange, $800 purchase) and commission tracking ($4.95/trade) |
| [`trade_today()`](../functions/calculateTrades.py:372) | Simplified trade execution for current-day recommendations |

### Web Page Generation ([`functions/WriteWebPage_pi.py`](../functions/WriteWebPage_pi.py:167))

| Function | Description |
|---|---|
| [`writeWebPage()`](../functions/WriteWebPage_pi.py:167) | Generates complete HTML dashboard with embedded charts, rankings tables, and performance data |
| [`ftpMoveDirectory()`](../functions/WriteWebPage_pi.py:5) | Uploads generated web content to remote server via SFTP (paramiko) |
| [`piMoveDirectory()`](../functions/WriteWebPage_pi.py:97) | Copies web content to local web-accessible directory (Raspberry Pi deployment) |

### Visualization ([`functions/MakeValuePlot.py`](../functions/MakeValuePlot.py:1))

| Function | Description |
|---|---|
| [`makeValuePlot()`](../functions/MakeValuePlot.py:23) | Portfolio value over time with channel analysis |
| [`makeUptrendingPlot()`](../functions/MakeValuePlot.py:205) | Number of stocks in uptrend over time |
| [`makeNewHighsAndLowsPlot()`](../functions/MakeValuePlot.py:314) | Market breadth: new highs vs new lows |
| [`makeTrendDispersionPlot()`](../functions/MakeValuePlot.py:360) | Trend dispersion across stock universe |
| [`makeDailyMonteCarloBacktest()`](../functions/MakeValuePlot.py:551) | Monte Carlo backtest visualization |
| [`makeDailyChannelOffsetSignal()`](../functions/MakeValuePlot.py:704) | Channel offset signal visualization |
| [`makeStockCluster()`](../functions/MakeValuePlot.py:646) | Stock clustering visualization |

### Other Functions

| Module | Description |
|---|---|
| [`functions/CountNewHighsLows.py`](../functions/CountNewHighsLows.py:19) | Counts new 252-day highs and lows across the stock universe; used as a market breadth indicator |
| [`functions/stock_cluster.py`](../functions/stock_cluster.py:29) | K-means clustering of stocks based on return correlations |
| [`functions/CheckMarketOpen.py`](../functions/CheckMarketOpen.py:1) | Checks if US stock market is open/closed; determines last trading day of month |
| [`functions/SendEmail.py`](../functions/SendEmail.py:1) | Sends HTML email notifications via Gmail SMTP |
| [`functions/GetYieldCurve.py`](../functions/GetYieldCurve.py:12) | Fetches US Treasury yield curve data (legacy, uses Python 2 urllib) |
| [`functions/ftp_quotes.py`](../functions/ftp_quotes.py:9) | Copies HDF5 quote files between machines via SFTP |
| [`functions/scheduler.py`](../scheduler.py:1) | Custom thread-based task scheduler with retry logic |
| [`functions/logger_config.py`](../functions/logger_config.py:1) | Centralized logging with rotating file handlers |
| [`functions/abacus_recommend.py`](../functions/abacus_recommend.py:1) | Recommendation engine classes: `ConfigurationHelper`, `DateHelper`, `ModelRecommender`, `RecommendationDisplay`, `PlotGenerator` |
| [`functions/abacus_backtest.py`](../functions/abacus_backtest.py:1) | Backtest data loading and model path management for Abacus system |

---

## 5. Web Pages Generated

PyTAAA generates several static HTML pages and PNG charts:

### Main Dashboard
- **`pyTAAAweb.html`** — Primary dashboard showing current holdings, portfolio value, trade recommendations, and links to all chart pages

### Chart Pages (linked from dashboard)
- **`pyTAAAweb_symbolCharts_MonthStartRank.html`** — Stock charts ordered by ranking at start of month
- **`pyTAAAweb_symbolCharts_TodayRank.html`** — Stock charts ordered by current ranking
- **`pyTAAAweb_symbolCharts_recentGainRank.html`** — Charts ordered by recent gain ranking
- **`pyTAAAweb_symbolCharts_recentComboGainRank.html`** — Charts ordered by recent combo gain ranking
- **`pyTAAAweb_symbolCharts_recentTrendRatioRank.html`** — Charts ordered by trend ratio ranking
- **`pyTAAAweb_symbolCharts_recentSharpeRatioRank.html`** — Charts ordered by recent Sharpe ranking

### PNG Charts
- **`PyTAAA_value.png`** — Portfolio value over time with channel analysis
- **Uptrending plot** — Number of stocks in uptrend
- **New highs/lows plot** — Market breadth indicator
- **Trend dispersion plot** — Trend dispersion across universe
- **Monte Carlo backtest plot** — Backtest with Monte Carlo overlays
- **Channel offset signal plot** — Channel-based trading signals
- **Stock cluster plot** — Clustering visualization
- **`recommendation_plot.png`** — Model recommendation analysis (from `recommend_model.py`)

### Deployment Targets
Web content is deployed via:
1. **SFTP** to a remote web server ([`ftpMoveDirectory()`](../functions/WriteWebPage_pi.py:5))
2. **Local copy** to a Raspberry Pi web directory ([`piMoveDirectory()`](../functions/WriteWebPage_pi.py:97))
3. **pytaaa_web** companion application (see Section 7)

---

## 6. Configuration System

PyTAAA uses two configuration approaches:

### Modern: JSON Configuration
The JSON file (e.g., [`pytaaa_generic.json`](../pytaaa_generic.json:1)) contains sections for:
- **Email** — SMTP credentials and recipients
- **Text_from_email** — SMS notification settings
- **FTP** — Remote server credentials for web deployment
- **stock_server** — Quote download server identification
- **Setup** — Runtime and pause time settings
- **Valuation** — All trading parameters including:
  - `symbols_file` — Path to stock symbol list
  - `stockList` — Universe identifier (`Naz100` or `SP500`)
  - `uptrendSignalMethod` — Signal generation method (`HMAs`, `minmax`, `3MAs`)
  - `numberStocksTraded` — Portfolio size (typically 7)
  - `monthsToHold` — Holding period
  - Moving average parameters (`MA1`, `MA2`, `MA3`, `sma2factor`)
  - Channel parameters (`narrowDays`, `mediumDays`, `wideDays`)
  - Risk parameters (`riskDownside_min`, `riskDownside_max`)

### Legacy: `.params` Files
- **`PyTAAA.params`** — Original configuration file (key-value pairs)
- **`PyTAAA_holdings.params`** — Current portfolio holdings (stocks, shares, buy prices, ranks)
- **`PyTAAA_status.params`** — Daily portfolio value history
- **`PyTAAA_ranks.params`** — Current stock rankings

### Model-Switching Configuration
- **`pytaaa_model_switching_params.json`** — Abacus-specific parameters including model paths, lookback periods, normalization values, and performance metric weights

---

## 7. Complementary Codebase: pytaaa_web

The [`pytaaa_web`](/Users/donaldpg/PyProjects/pytaaa_web) repository is a **FastAPI-based web application** that provides a modern dashboard for visualizing PyTAAA trading model performance. It is a read-only consumer of PyTAAA's output data.

### Key Characteristics
- **Tech Stack:** FastAPI, PostgreSQL, SQLAlchemy 2.0 (async), Alembic, Docker
- **Purpose:** Dashboard for tracking 6 trading strategies (5 base models + Abacus meta-model)
- **Data Source:** Reads from PyTAAA output files (`PyTAAA_status.params`, `PyTAAA_holdings.params`, `PyTAAA_ranks.params`, backtest files)
- **Does NOT** run any trading logic — purely visualization and data presentation

### Relationship to PyTAAA
- PyTAAA generates `.params` files and JSON configs as its primary output
- `pytaaa_web` ingests these files into a PostgreSQL database via CLI commands
- The web app provides REST API endpoints and an HTML dashboard
- Data sync can be automated via `rsync` from the PyTAAA machine to a Raspberry Pi running `pytaaa_web`

### Deployment Architecture
```
PyTAAA (Mac) → generates .params files
    ↓ rsync
Raspberry Pi → pytaaa_web (FastAPI + PostgreSQL in Docker)
    ↓ nginx reverse proxy
Internet → https://yourpi.duckdns.org
```

For full details, see the [`pytaaa_web` README](/Users/donaldpg/PyProjects/pytaaa_web/README.md) and [`spec.md`](/Users/donaldpg/PyProjects/pytaaa_web/spec.md).

---

## 8. References

1. **Hull Moving Average (HMA):** Hull, A. (2005). *Hull Moving Average*. Developed to reduce lag in traditional moving averages while maintaining smoothness. Used in `naz100_hma` and `sp500_hma` models. Implementation: [`hma()`](../functions/TAfunctions.py:1252).

2. **Sharpe Ratio:** Sharpe, W.F. (1966). "Mutual Fund Performance." *Journal of Business*, 39(1), 119-138. Used extensively for stock ranking and portfolio evaluation. Implementation: [`allstats.sharpe()`](../functions/allstats.py:9).

3. **Sortino Ratio:** Sortino, F.A. & van der Meer, R. (1991). "Downside Risk." *Journal of Portfolio Management*, 17(4), 27-31. Used for downside-risk-adjusted performance measurement. Implementation: [`allstats.sortino()`](../functions/allstats.py:39).

4. **Monte Carlo Simulation:** Applied to parameter optimization for model-switching decisions. The explore-exploit search strategy draws from multi-armed bandit literature. Implementation: [`MonteCarloBacktest`](../functions/MonteCarloBacktest.py:1).

5. **Tactical Asset Allocation (TAA):** Faber, M.T. (2007). "A Quantitative Approach to Tactical Asset Allocation." *Journal of Wealth Management*. PyTAAA implements a momentum-based TAA strategy with monthly rebalancing.

6. **Market Breadth Indicators:** New highs/lows counting as implemented in [`newHighsAndLows()`](../functions/CountNewHighsLows.py:19) follows standard market breadth analysis methodology used in technical analysis.

7. **Yahoo Finance Data:** Stock quotes sourced via the `yfinance` Python package, which provides adjusted close prices accounting for splits and dividends. Implementation: [`downloadQuotes()`](../functions/quotes_adjClose.py:7).

8. **HDF5 Storage:** Hierarchical Data Format version 5, used for efficient storage and retrieval of large numerical datasets. PyTAAA stores all historical quote data in HDF5 via the `tables` (PyTables) library. Implementation: [`loadQuotes_fromHDF()`](../functions/UpdateSymbols_inHDF5.py:39).

9. **Ulcer Performance Index / Martin Ratio:** Martin, P. & McCann, B. (1989). *The Investor's Guide to Fidelity Funds*. Used in [`move_martin_2D()`](../functions/TAfunctions.py:1745) for risk-adjusted performance measurement.

10. **Numba JIT Compilation:** Used in [`MonteCarloBacktest.py`](../functions/MonteCarloBacktest.py:58) to accelerate numerical computations for Monte Carlo simulations via just-in-time compilation.
