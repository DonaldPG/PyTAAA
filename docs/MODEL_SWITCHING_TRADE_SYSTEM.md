# Model-Switching Trading System Documentation

## Table of Contents
1. [Background and Context](#background-and-context)
2. [Monte Carlo Parameter Search Methodology](#monte-carlo-parameter-search-methodology)
3. [Model-Switching Operation for Monthly Trading](#model-switching-operation-for-monthly-trading)

---

## Background and Context

### Overview
The PyTAAA Model-Switching Trading System is an advanced algorithmic trading framework that dynamically selects between multiple trading models based on historical performance analysis. The system combines two primary components:

1. **Monte Carlo Parameter Search** (`run_monte_carlo.py`): Optimizes model parameters and identifies the best-performing configurations
2. **Model Recommendation System** (`recommend_model.py`): Provides monthly trading recommendations for manual portfolio management

### Trading Models Available
The system supports five distinct trading models:

- **cash**: Cash/money market position (safe harbor)
- **naz100_pine**: NASDAQ-100 using Pine Script methodology
- **naz100_hma**: NASDAQ-100 using Hull Moving Average methodology  
- **naz100_pi**: NASDAQ-100 using Pi methodology
- **sp500_hma**: S&P 500 using Hull Moving Average methodology

### Key Features
- **Permutation-Invariant Optimization**: Efficiently handles parameter combinations without redundant testing
- **Configurable Search Strategies**: Choose between exploration, exploitation, or dynamic strategies
- **Responsive Interrupt Handling**: Graceful stopping with state preservation
- **Flexible Data Sources**: Supports both actual trading data and backtested portfolio values
- **Monthly Rebalancing**: Designed for monthly trading frequency with first-weekday execution

---

## Monte Carlo Parameter Search Methodology

### Purpose
The Monte Carlo parameter search optimizes lookback periods used in model selection algorithms. It uses sophisticated exploration/exploitation strategies to efficiently discover the best-performing parameter combinations.

### How It Works

#### 1. Parameter Space Exploration
- **Lookback Periods**: Tests combinations of 3 lookback periods (e.g., [50, 150, 250] days)
- **Permutation Invariance**: Treats [50, 150, 250] and [250, 50, 150] as identical combinations
- **Canonical Forms**: Stores all combinations in sorted order to eliminate redundancy
- **Duplicate Avoidance**: Prevents retesting of previously evaluated parameter sets

#### 2. Search Strategies
The system supports three configurable search strategies:

**Explore-Exploit (Default)**
```bash
uv run python run_monte_carlo.py --search explore-exploit
```
- Dynamic strategy that starts with exploration and transitions to exploitation
- Balances parameter space coverage with performance optimization
- Recommended for most use cases

**Pure Exploration**
```bash
uv run python run_monte_carlo.py --search explore
```
- Focuses on comprehensive parameter space coverage
- Uses random lookback generation throughout
- Best for initial parameter discovery

**Pure Exploitation**
```bash
uv run python run_monte_carlo.py --search exploit
```
- Focuses computational resources on best-performing combinations
- Uses UCB1 algorithm for selection
- Best for refining known good parameters

#### 3. Performance Evaluation
For each parameter combination, the system evaluates:
- **Normalized Score**: Composite performance metric averaging normalized individual metrics
- **Annual Return**: Annualized portfolio return
- **Sharpe Ratio**: Risk-adjusted return metric
- **Sortino Ratio**: Downside risk-adjusted return
- **Maximum Drawdown**: Largest portfolio decline
- **Average Drawdown**: Mean portfolio decline periods
- **Daily Returns**: Average daily performance
- **Volatility**: Portfolio return variance

The **normalized score** is calculated by:
1. Normalizing each metric using predefined central values and standard deviations
2. Averaging the normalized values across all performance metrics (excluding final portfolio value)
3. Setting CASH portfolios to exactly 0.0 as the neutral baseline reference

#### 4. Model Selection Algorithm
At each trading date, the system:
1. Calculates performance metrics for each model over specified lookback periods
2. Ranks models across all metrics using the lookback periods
3. Applies configurable weights to different performance metrics
4. Selects the model with the best weighted average rank

### Running Monte Carlo Parameter Search

#### Basic Usage
```bash
# Use default settings (explore-exploit strategy, 250 iterations)
uv run python run_monte_carlo.py

# Use pure exploration strategy
uv run python run_monte_carlo.py --search explore

# Use pure exploitation strategy
uv run python run_monte_carlo.py --search exploit
```

#### Configuration
Edit `pytaaa_model_switching_params.json` to customize:

```json
{
  "monte_carlo": {
    "max_iterations": 250,
    "min_iterations_for_exploit": 50,
    "trading_frequency": "monthly",
    "min_lookback": 10,
    "max_lookback": 400,
    "data_format": "backtested"
  },
  "model_selection": {
    "n_lookbacks": 3,
    "performance_metrics": {
      "gain_loss_weight": 1.0,
      "sharpe_ratio_weight": 1.0,
      "sortino_ratio_weight": 1.0,
      "max_drawdown_weight": 1.0,
      "avg_drawdown_weight": 1.0,
      "daily_return_weight": 1.0,
      "volatility_weight": 1.0
    }
  }
}
```

#### Output and Results
The Monte Carlo search produces:

1. **Console Output**: Real-time progress with iteration counts and time estimates
2. **Performance Plot**: Visual chart saved to `assets/model_switching_portfolio_performance.png`
3. **State File**: `monte_carlo_state.pkl` containing optimization state and best parameters
4. **Log Files**: Detailed execution logs in `monte_carlo_run.log`

#### Key Benefits
- **Efficiency**: 6x memory reduction through permutation-invariant tracking
- **Responsiveness**: Ctrl+C interruption with full state preservation
- **Robustness**: Handles data gaps and missing values gracefully
- **Scalability**: Supports arbitrary numbers of lookback periods
- **Reproducibility**: State persistence enables resuming interrupted runs

---

## Model-Switching Operation for Monthly Trading

### Purpose
The model recommendation system provides actionable trading recommendations for manual portfolio management. It analyzes current market conditions and recommends which trading model to follow for the upcoming month.

### When to Use
- **Monthly**: Run at the beginning of each month (first weekday)
- **Mid-Month Reviews**: Run any time for current market assessment
- **Strategy Changes**: When considering portfolio rebalancing

### How It Works

#### 1. Recommendation Generation
The system analyzes multiple lookback periods to determine which trading model has performed best recently. It generates recommendations for:
- **Current Date**: Today's recommended model based on latest data
- **Month Start**: Recommendation for the first weekday of current month

#### 2. Lookback Analysis
Using the specified lookback periods (e.g., 50, 150, 250 days), the system:
1. Calculates performance metrics for each model over each lookback period
2. Ranks models by their performance across all metrics
3. Applies weighted scoring to determine the best-performing model
4. Provides the recommendation with supporting analysis

### Running Model Recommendations

#### Basic Usage
```bash
# Generate recommendation for today and first weekday of current month
uv run python recommend_model.py

# Generate recommendation for specific date
uv run python recommend_model.py --date 2025-08-15

# Use custom lookback periods
uv run python recommend_model.py --lookbacks "25,50,100"

# Use optimized parameters from Monte Carlo search
uv run python recommend_model.py --lookbacks use-saved

# Combine specific date with optimized parameters
uv run python recommend_model.py --date 2025-08-01 --lookbacks use-saved
```

#### Parameter Sources
The system supports three sources for lookback periods:

**1. Configuration Defaults**
```bash
uv run python recommend_model.py
```
Uses `default_lookbacks` from configuration file (default: [50, 150, 250])

**2. User-Specified Parameters**
```bash
uv run python recommend_model.py --lookbacks "60,120,240"
```
Uses custom lookback periods provided by user

**3. Monte Carlo Optimized Parameters**
```bash
uv run python recommend_model.py --lookbacks use-saved
```
Uses best parameters discovered by Monte Carlo parameter search

#### Configuration
Edit `pytaaa_model_switching_params.json` to customize defaults:

```json
{
  "recommendation_mode": {
    "default_lookbacks": [50, 150, 250],
    "output_format": "both",
    "generate_plot": true,
    "show_model_ranks": true
  }
}
```

### Interpreting Recommendations

#### Sample Output
```
============================================================
MODEL RECOMMENDATION RESULTS
============================================================

Recommendation Parameters:
  Lookback periods: [50, 150, 250] days
  Target date: 2025-08-02 (Saturday)
  First weekday of month: 2025-08-01 (Friday)

----------------------------------------
Model ranks on 2025-08-02:
----------------------------------------
  Recommended model: naz100_hma
  Analysis lookbacks: [50, 150, 250] days
  Available models: cash, naz100_pine, naz100_hma, naz100_pi, sp500_hma

----------------------------------------
Model ranks on 2025-08-01:
----------------------------------------
  Recommended model: sp500_hma
  Analysis lookbacks: [50, 150, 250] days
  Available models: cash, naz100_pine, naz100_hma, naz100_pi, sp500_hma

============================================================
RECOMMENDATION SUMMARY
============================================================
Generated recommendations for manual stock selection update
Based on model-switching trading system methodology
Review recommendations and manually update portfolio holdings
```

#### Key Information
- **Recommended Model**: The trading strategy to follow for the specified period
- **Analysis Lookbacks**: The lookback periods used in the analysis
- **Available Models**: All models considered in the ranking process
- **Date Context**: Shows both current date and month-start recommendations

### Manual Trading Process

#### Monthly Workflow
1. **First Weekday of Month**: Run recommendation system
   ```bash
   uv run python recommend_model.py --lookbacks use-saved
   ```

2. **Review Recommendation**: Analyze the suggested model and supporting data

3. **Update Portfolio**: Manually adjust holdings to follow the recommended model's strategy
   - Review the recommended model's current stock selections
   - Rebalance portfolio to match the model's allocations
   - Consider transaction costs and tax implications

4. **Monitor Performance**: Track how the selected model performs during the month

5. **Month-End Review**: Optionally run recommendations mid-month for confirmation

#### Risk Management
- **Cash Model**: Default to cash position during uncertain market conditions
- **Diversification**: Consider geographic and sector diversification (NASDAQ vs S&P 500)
- **Drawdown Monitoring**: Watch for excessive portfolio declines
- **Model Switching Frequency**: Avoid excessive switching that increases transaction costs

### Best Practices

#### For Parameter Optimization
1. **Run Monte Carlo Search Monthly**: Update optimized parameters regularly
2. **Use Appropriate Iterations**: 250+ iterations for robust parameter discovery
3. **Monitor Search Strategy**: Use explore-exploit for balanced optimization
4. **Preserve State**: Always save state for interrupted runs

#### For Trading Recommendations
1. **Use Optimized Parameters**: Prefer `--lookbacks use-saved` after Monte Carlo optimization
2. **Check Both Dates**: Review both current and month-start recommendations
3. **Validate Data Quality**: Ensure recommendation dates have sufficient historical data
4. **Document Decisions**: Keep records of recommendations and actual trading decisions

#### For System Maintenance
1. **Regular Updates**: Keep portfolio value files current
2. **Data Validation**: Verify data integrity before important trading decisions
3. **Backup State Files**: Preserve `monte_carlo_state.pkl` files
4. **Log Review**: Monitor log files for errors or warnings

### Troubleshooting

#### Common Issues
- **No Data Available**: Ensure portfolio value files exist and contain recent data
- **Insufficient History**: Verify lookback periods don't exceed available data
- **Configuration Errors**: Check JSON configuration file syntax
- **Missing State File**: Run Monte Carlo search before using `--lookbacks use-saved`

#### Data Requirements
- **Minimum History**: At least max(lookback_periods) + 30 days of data
- **Data Format**: Proper format in portfolio value files
- **Date Coverage**: Continuous daily data without large gaps
- **Model Coverage**: All configured models must have corresponding data files

---

## State Management and Debugging

The system provides comprehensive state management capabilities for development and debugging:

#### Monte Carlo State Utility (`modify_saved_state.py`)
A dedicated utility script for inspecting and modifying saved Monte Carlo state:

```bash
# Inspect current saved state and view all canonical combinations
uv run python modify_saved_state.py inspect

# Remove all combinations containing a specific lookback value
uv run python modify_saved_state.py remove-lookback 150

# Remove a specific combination (e.g., [50, 150, 250])
uv run python modify_saved_state.py remove-combination 50 150 250

# Reset entire state (removes all accumulated learning)
uv run python modify_saved_state.py reset

# Use custom state file
uv run python modify_saved_state.py inspect --file custom_state.pkl
```

The utility provides:
- **State Inspection**: View all canonical combinations, performance scores, and visit counts
- **Selective Removal**: Remove problematic combinations or specific lookback values
- **Top Performer Analysis**: Display top 10 performing combinations with scores
- **Backup Creation**: Automatic backup before modifications (use `--no-backup` to skip)
- **Safe Operations**: Confirmation prompts before destructive operations

#### State File Structure
The `monte_carlo_state.pkl` file contains:
- **Canonical Combinations**: Permutation-invariant lookback combinations
- **Performance Scores**: Running average performance for each combination
- **Visit Counts**: Number of times each combination has been tested
- **Configuration**: Min/max lookbacks and search parameters
- **Timestamp**: Last modification time

---

## Summary

The PyTAAA Model-Switching Trading System provides a sophisticated framework for optimizing and executing algorithmic trading strategies. The two-phase approach—Monte Carlo parameter optimization followed by monthly model recommendations—enables systematic and data-driven trading decisions while maintaining the flexibility for manual oversight and risk management.

**Key Advantages:**
- Systematic parameter optimization using advanced algorithms
- Monthly rebalancing frequency reduces transaction costs
- Multiple model selection provides diversification opportunities
- Manual oversight maintains control over trading decisions
- Robust state management ensures continuity across sessions