##### Usage:

1. create an hdf file that holds stock quotes

   "python re-generateHDF5.py"
  
2. edit the file containing information for PyTAA to report results
   - options exist to send an email to your desired email from another email account (they don't have to be the same email account)
   - edit PyTAAA.params with a text editor and replace example values with your information

3. run PyTAAA with the command: "python PyTAAA.py"
   - the code updates quotes and re-runs every few hours. It runs un-interrupted for 2 weeks (duration can be changed in PyTAAA.params).
   - on the 2nd trading day of the month, PyTAAA recommends new stock holdings
   
4. It's up to the user to decide if they want to do anything with the recommendations. This is designed and provided for entertainment only. The author does not accept and responsibility for anything done by others with the recommendations.

5. To let the code know how to track a portfolio for you, manually update the stock holdings in "PyTAAA_holdings.params".

6. A web page is created in the folder listed in the json file with PyTAAA parameters ("webpage"). In Windows, you can double-click pyTAAAweb.html to see the latest status and holdings, as recommended by PyTAAA.


##### Model Switching Abacus Method:

• **Dynamic Portfolio Optimization**: Advanced model switching system that dynamically selects between multiple trading models (cash, naz100_pine, naz100_hma, naz100_pi, sp500_hma) based on performance metrics

• **Monte Carlo Backtesting**: Uses statistical simulation to optimize lookback periods and model selection parameters through thousands of iterations

• **Normalized Performance Scoring**: Evaluates models using normalized metrics (Sharpe ratio, Sortino ratio, drawdown) to make objective switching decisions

• **Monthly Rebalancing**: Automatically evaluates and switches between stock universes on the first weekday of each month

• **Centralized JSON Configuration**: All entry points now use a single JSON configuration file for data locations, model parameters, and output directories

• **Entry Points Available**:
  - `run_monte_carlo.py` - optimize model parameters  
  - `modify_saved_state.py` - State management
  - `recommend_model.py` - generate trading recommendations
  - `run_monte_carlo.sh` - shell wrapper script
  - `run_normalized_score_history.py` - Score history analysis
  - `update_json_from_csv.py` - Update JSON config with normalization parameters from CSV
  - `daily_abacus_update.py` - Daily portfolio tracking and webpage updates


##### Configuration Management:

• **update_json_from_csv.py**: Utility script to update JSON configuration files with normalization parameters from Monte Carlo CSV or Excel results

  **Purpose**: Transfer optimal normalization parameters from successful Monte Carlo runs to the JSON configuration used by recommend_model.py

  **Usage**:
  ```bash
  # From CSV file by row number
  uv run python update_json_from_csv.py \
      --csv abacus_best_performers.csv \
      --row 42 \
      --json pytaaa_model_switching_params.json
  
  # From Excel file by actual Excel row number (e.g., row 419 as shown in Excel)
  uv run python update_json_from_csv.py \
      --xlsx abacus_best_performers_3.xlsx \
      --row 419 \
      --json pytaaa_model_switching_params.json
  
  # Search by date and time (works with both CSV and Excel)
  uv run python update_json_from_csv.py \
      --xlsx abacus_best_performers_3.xlsx \
      --date "2025-09-13" \
      --time "16:43:08" \
      --json pytaaa_model_switching_params.json
  
  # Preview parameters without updating (dry run)
  uv run python update_json_from_csv.py \
      --xlsx abacus_best_performers_3.xlsx \
      --row 419 \
      --json pytaaa_model_switching_params.json \
      --show-only
  
  # Update without creating backup
  uv run python update_json_from_csv.py \
      --csv abacus_best_performers.csv \
      --row 42 \
      --json pytaaa_model_switching_params.json \
      --no-backup
  ```

  **Workflow**:
  1. Run Monte Carlo optimization to generate CSV/Excel results
  2. Review results to identify high-performing configurations
  3. Use this script to copy normalization parameters to JSON config
  4. Run recommend_model.py with updated parameters

  **Key Features**:
  - **CSV and Excel Support**: Works with both .csv and .xlsx files
  - **Intuitive Row Selection**: For Excel files, use the actual row number as shown in Excel (no calculations needed)
  - **Date/Time Search**: Find specific runs by timestamp
  - **Parameter Extraction**: Extracts central values, standard deviations, lookback periods, performance metric weights, and metric blending parameters
  - **Performance Display**: Shows Final Value (formatted as $X,XXX,XXX), Annual Return, Sharpe/Sortino ratios, Max Drawdown, and lookback periods
  - **Safe Updates**: Creates automatic backup of JSON file (unless --no-backup)
  - **Dry Run Mode**: Preview parameters with --show-only before updating

  **Excel File Notes**:
  - Header row is expected at row 13
  - When you specify `--row 419`, it reads Excel row 419 exactly as you see it
  - Requires `openpyxl` library: `uv add openpyxl`


##### Monte Carlo Optimization:

• **run_monte_carlo.py**: Monte Carlo optimization engine for discovering optimal model-switching parameters

  **Purpose**: Run thousands of backtesting simulations to find optimal lookback periods, normalization parameters, and model selection criteria. Results are saved to CSV/Excel for analysis.

  **Usage**:
  ```bash
  # Standard Monte Carlo run with 100 iterations
  uv run python run_monte_carlo.py \
      --json pytaaa_model_switching_params.json \
      --iterations 100

  # Extended run with exploration mode
  uv run python run_monte_carlo.py \
      --json pytaaa_model_switching_params.json \
      --iterations 1000 \
      --search-mode explore

  # Exploitation mode (refine around known good parameters)
  uv run python run_monte_carlo.py \
      --json pytaaa_model_switching_params.json \
      --iterations 500 \
      --search-mode exploit

  # Resume from saved state
  uv run python run_monte_carlo.py \
      --json pytaaa_model_switching_params.json \
      --iterations 500 \
      --resume

  # Custom output file
  uv run python run_monte_carlo.py \
      --json pytaaa_model_switching_params.json \
      --iterations 200 \
      --output custom_results.csv
  ```

  **Workflow**:
  1. Configure JSON with model paths and date ranges
  2. Run Monte Carlo optimization (can take hours for large iterations)
  3. Review CSV/Excel results sorted by performance metrics
  4. Use `update_json_from_csv.py` to transfer best parameters to config
  5. Validate with `recommend_model.py` before live trading

  **Key Features**:
  - **Adaptive Search**: Explore mode for broad search, exploit mode for refinement
  - **State Persistence**: Save/resume capability via `monte_carlo_state.pkl`
  - **Comprehensive CSV Output**: All parameters and metrics logged for analysis
  - **Real-time Progress**: Visual progress bars and ETA estimates
  - **Performance Metrics**: Final value, Sharpe ratio, Sortino ratio, drawdown analysis
  - **Parameter Optimization**: Lookback periods, normalization values, metric weights
  - **Best Performance Tracking**: Automatically tracks and displays top configurations

  **Search Modes**:
  - **explore**: Random sampling across wide parameter ranges (default)
  - **exploit**: Focused search around previously successful parameters
  - **Resume**: Continue from saved state with same search mode

  **Output Files**:
  - `monte_carlo_results.csv` - Full results of all iterations
  - `monte_carlo_state.pkl` - State file for resume capability
  - `monte_carlo_best_performance.png` - Chart of best configuration found


• **run_monte_carlo.sh**: Shell wrapper script for batch Monte Carlo optimization runs

  **Purpose**: Convenient bash script for running multiple Monte Carlo sessions with different configurations or resuming long-running optimizations.

  **Usage**:
  ```bash
  # Basic run with default settings
  ./run_monte_carlo.sh

  # Specify custom JSON config
  ./run_monte_carlo.sh pytaaa_model_switching_params.json

  # Run with custom iteration count
  ./run_monte_carlo.sh pytaaa_model_switching_params.json 1000

  # Background execution with nohup
  nohup ./run_monte_carlo.sh pytaaa_model_switching_params.json 5000 &
  ```

  **Key Features**:
  - **Simplified Execution**: No need to remember Python command syntax
  - **Environment Setup**: Automatically configures PYTHONPATH and virtual environment
  - **Logging**: Redirects output to log files for long-running sessions
  - **Error Recovery**: Catches interrupts and saves state before exit


• **modify_saved_state.py**: Utility for inspecting and modifying Monte Carlo state files

  **Purpose**: View, edit, or reset the saved Monte Carlo state to control resume behavior or adjust search parameters.

  **Usage**:
  ```bash
  # Display current saved state
  uv run python modify_saved_state.py --show

  # Reset state to start fresh
  uv run python modify_saved_state.py --reset

  # Change search mode in saved state
  uv run python modify_saved_state.py --set-mode exploit

  # Update best parameters manually
  uv run python modify_saved_state.py \
      --set-lookbacks 50,150,250

  # View best parameters from state
  uv run python modify_saved_state.py --show-best
  ```

  **Key Features**:
  - **State Inspection**: View all saved state details
  - **Mode Switching**: Change between explore/exploit without losing progress
  - **Parameter Override**: Manually set best parameters for exploitation
  - **State Reset**: Clear state to start fresh optimization
  - **Backup Creation**: Automatically backs up state before modifications


##### Model Recommendation:

• **recommend_model.py**: Generate trading recommendations based on model-switching methodology

  **Purpose**: Produce actionable trading recommendations for the current date and first weekday of the month using optimized model selection parameters.

  **Usage**:
  ```bash
  # Generate recommendation for today
  uv run python recommend_model.py \
      --json pytaaa_model_switching_params.json

  # Recommendation for specific date
  uv run python recommend_model.py \
      --json pytaaa_model_switching_params.json \
      --date 2025-01-15

  # Use saved Monte Carlo lookbacks
  uv run python recommend_model.py \
      --json pytaaa_model_switching_params.json \
      --lookbacks use-saved

  # Custom lookback periods
  uv run python recommend_model.py \
      --json pytaaa_model_switching_params.json \
      --lookbacks 50,100,200

  # Generate without plot
  uv run python recommend_model.py \
      --json pytaaa_model_switching_params.json \
      --no-plot
  ```

  **Workflow**:
  1. Update JSON config with optimal parameters (from Monte Carlo + CSV update)
  2. Run recommend_model.py to get current trading recommendation
  3. Review model rankings and performance metrics
  4. Manually execute trades based on recommendation
  5. Update PyTAAA_holdings.params with new positions

  **Key Features**:
  - **Dual Date Recommendations**: Always shows both target date and first weekday of month
  - **Model Rankings**: Complete performance ranking of all available models
  - **Normalized Scoring**: Uses same metrics as Monte Carlo optimization
  - **Visual Output**: Generates recommendation_plot.png with full details
  - **Parameter Display**: Shows lookback periods, normalization values, performance metrics
  - **Closest Date Matching**: Finds nearest available date if target not in data

  **Output Details**:
  - Best model recommendation with normalized score
  - Model rankings table (all models ranked by performance)
  - Portfolio metrics (Final Value, Annual Return, Sharpe Ratio)
  - Lookback periods used for analysis
  - Parameter summary (central values, std deviations)


##### Performance Analysis:

• **run_normalized_score_history.py**: Analyze historical normalized score performance across all models

  **Purpose**: Generate comprehensive historical analysis charts showing how normalized performance scores evolved over time for each trading model.

  **Usage**:
  ```bash
  # Analyze with actual (live) data
  uv run python run_normalized_score_history.py \
      --json pytaaa_model_switching_params.json \
      --data-format actual

  # Analyze with backtested data
  uv run python run_normalized_score_history.py \
      --json pytaaa_model_switching_params.json \
      --data-format backtested

  # Custom lookback periods
  uv run python run_normalized_score_history.py \
      --json pytaaa_model_switching_params.json \
      --lookbacks 30,90,180

  # Specific date range
  uv run python run_normalized_score_history.py \
      --json pytaaa_model_switching_params.json \
      --start-date 2024-01-01 \
      --end-date 2024-12-31

  # Custom output location
  uv run python run_normalized_score_history.py \
      --json pytaaa_model_switching_params.json \
      --output-dir ./analysis_results
  ```

  **Key Features**:
  - **Multi-Model Comparison**: Side-by-side performance of all trading models
  - **Normalized Metrics**: Uses same scoring methodology as model selection
  - **Time Series Visualization**: Charts showing score evolution over time
  - **Model Switching Periods**: Highlights when model switches occurred
  - **Statistical Summary**: Mean, median, std dev for each model's scores
  - **Export Options**: Saves charts as PNG and data as CSV

  **Output Files**:
  - `combined_normalized_score_history_actual.png` - Chart with actual data
  - `combined_normalized_score_history_backtested.png` - Chart with backtest data
  - `normalized_score_data.csv` - Raw data for further analysis


##### Daily Operations:

• **daily_abacus_update.py**: Daily portfolio tracking and webpage updates with intelligent quote management

  **Purpose**: Automated daily wrapper that detects the active trading model, updates stock quotes only when needed, and generates HTML dashboard content.

  **Usage**:
  ```bash
  # Standard daily update
  uv run python daily_abacus_update.py \
      --json /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json

  # Verbose mode for debugging
  uv run python daily_abacus_update.py \
      --json /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json \
      --verbose
  ```

  **Key Features**:
  - **Active Model Detection**: Automatically reads `trading_model:` from PyTAAA_holdings.params
  - **Dynamic Data Source Routing**: Routes to correct symbols file (Naz100/SP500) based on active model
  - **Smart Quote Updates**: Only updates quotes when market is open and data is stale
  - **Weekend/Holiday Handling**: Skips unnecessary updates during market closures
  - **Web Dashboard Generation**: Creates HTML pages and PNG plots for portfolio tracking
  - **Multi-Model Support**: Handles cash, naz100_pine, naz100_hma, naz100_pi, sp500_hma models

  **Intelligent Update Logic**:
  - Checks market status using existing PyTAAA market detection
  - Verifies HDF5 quote freshness (last modified date)
  - Skips quote downloads if market closed or quotes current
  - Always refreshes web content even when quotes unchanged

  **Web Output**:
  - Generates `pyTAAAweb.html` dashboard
  - Creates PNG charts: stock performance, backtest results
  - Outputs to configured `web_output_dir` in JSON

  **Automation Ready**:
  ```bash
  # Crontab example - run weekdays at 6:30 AM
  30 6 * * 1-5 cd /Users/donaldpg/PyProjects/worktree2/PyTAAA && \
      uv run python daily_abacus_update.py \
      --json /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json
  ```

  **Error Handling**:
  - Validates JSON configuration structure
  - Creates required files if missing (holdings, status, ranks)
  - Provides detailed logging with --verbose flag
  - Cleans up temporary files on completion or error


##### Complete Model Switching Workflow:

**Phase 1: Initial Setup**
1. Create/update JSON configuration with model paths and data locations
2. Ensure all model data stores have current portfolio value files
3. Set up web output directory for dashboard generation

**Phase 2: Parameter Optimization**
1. Run `run_monte_carlo.py` with 1000+ iterations in explore mode
2. Review CSV results, sort by Final Value or Sharpe Ratio
3. Identify top 5-10 performing configurations
4. Run additional Monte Carlo in exploit mode around best parameters
5. Select final optimal configuration from refined results

**Phase 3: Configuration Update**
1. Use `update_json_from_csv.py` to transfer best parameters to JSON config
   - Extracts lookback periods
   - Copies normalization parameters (central values, std deviations)
   - Updates performance metric weights
   - Sets metric blending parameters
2. Validate updated JSON structure
3. Create backup of working configuration

**Phase 4: Generate Recommendations**
1. Run `recommend_model.py` to get current model recommendation
2. Review model rankings and normalized scores
3. Check recommendation_plot.png for visual confirmation
4. Compare with `run_normalized_score_history.py` output for context

**Phase 5: Execute Trades (Manual)**
1. Review recommended model from Phase 4
2. Calculate position sizes for new model
3. Execute trades on brokerage platform
4. Update `PyTAAA_holdings.params` with:
   - New stock positions and shares
   - Current buy prices
   - Trading model name (e.g., `trading_model: naz100_pine`)

**Phase 6: Daily Tracking**
1. Set up cron job for `daily_abacus_update.py`
2. Script automatically:
   - Detects active model from holdings file
   - Updates quotes when market is open
   - Generates HTML dashboard
   - Creates performance charts
3. Check web dashboard daily at configured `web_output_dir`

**Phase 7: Monthly Rebalancing**
1. On first weekday of month, run `recommend_model.py`
2. Compare recommendation with current holdings
3. If model switch recommended:
   - Review performance metrics carefully
   - Execute trades if switching makes sense
   - Update holdings file with new model
4. If no switch, continue with current model

**Phase 8: Periodic Re-optimization**
1. Every 3-6 months, re-run Monte Carlo optimization
2. Check if market conditions have shifted
3. Update JSON configuration if new parameters perform better
4. Use `run_normalized_score_history.py` to verify improvement

**Emergency Procedures**:
- Use `modify_saved_state.py --reset` if Monte Carlo gets stuck
- Use `daily_abacus_update.py --verbose` to diagnose update issues
- Keep backups of JSON config before parameter updates
- Monitor log files: `monte_carlo_backtest.log`, `recommend_model.log`


##### Notes:

Backtest plots that start ca. 1991 contain different stocks for historical testing than those created by 're-generateHDF5.py'. Therefore backtest plots will not match those created by PyTAAA.py and shown on the created web page. This is due to changes in the Nasdaq 100 index over time.

The backtest plots show only an approximation to "Buy & Hold" investing. This is particularly true for the Daily backtest that is created every time the PyTAAA code runs. Buy & Hold is approximated on the plot by the red value curves. The calculations assume that equal dollar investments are made in all the current stocks in the Nasdaq 100 index. For example, note that the current Nasdaq 100 stocks as of February 2014 did not have the same performance during 2000-2003 as the stocks in the index during 2000-2003. Whereas the Nasdaq Index lost more than 50% of its peak value, the stocks that are in the index as of February 2014 AND were also in the index in 2000, maintained nearly constant value over the period. Similar cautions need to be made about the historical backtest performance of PyTAAA trading recommendations. Therefore, hypothetical performance as portrayed by PyTAAA backtests should be viewed as untested and unverified. Actual investment performance under real market conditions will almost certainly be lower.

PyTAAA will reflect changes in the Nasdaq 100 index over time automatically. It checks a web page each time it runs to ensure that current stocks it can choose match the index.
