# Model-Switching Trading System Documentation

## Table of Contents
1. [Background and Context](#background-and-context)
2. [Monte Carlo Parameter Search Methodology](#monte-carlo-parameter-search-methodology)
3. [Model-Switching Operation for Monthly Trading](#model-switching-operation-for-monthly-trading)
4. [Usage Examples and Shell Scripts](#usage-examples-and-shell-scripts)

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

## Usage Examples and Shell Scripts

### Shell Script Runner (`run_monte_carlo.sh`)

The shell script provides an automated way to run multiple Monte Carlo optimization cycles with configurable parameters and state management.

#### Basic Usage Examples

```bash
# Run 5 Monte Carlo cycles with default settings
./run_monte_carlo.sh 5

# Run 10 cycles with exploration strategy
./run_monte_carlo.sh 10 explore

# Run 3 cycles with exploitation strategy and verbose output
./run_monte_carlo.sh 3 exploit --verbose

# Run 7 cycles with dynamic explore-exploit strategy
./run_monte_carlo.sh 7 explore-exploit

# Run 2 cycles with verbose output (default strategy)
./run_monte_carlo.sh 2 --verbose
```

#### State Management Examples

```bash
# Run 5 cycles, resetting state after each run (fresh exploration)
./run_monte_carlo.sh 5 --reset

# Run 3 cycles with exploration strategy, resetting after each
./run_monte_carlo.sh 3 explore --reset

# Combine verbose mode with state reset
./run_monte_carlo.sh 2 exploit --verbose --reset
```

#### Advanced Combinations

```bash
# Long optimization run with state persistence
./run_monte_carlo.sh 20 explore-exploit

# Statistical analysis with independent runs
./run_monte_carlo.sh 10 explore --reset

# Parameter refinement with exploitation focus
./run_monte_carlo.sh 15 exploit --verbose
```

### Individual Script Usage

#### Monte Carlo Parameter Optimization

```bash
# Basic optimization with default settings
uv run python run_monte_carlo.py

# Use specific search strategies
uv run python run_monte_carlo.py --search explore
uv run python run_monte_carlo.py --search exploit
uv run python run_monte_carlo.py --search explore-exploit

# Enable detailed performance breakdown
uv run python run_monte_carlo.py --verbose
uv run python run_monte_carlo.py --search exploit --verbose
```

#### Model Recommendation Generation

```bash
# Generate recommendations for current date
uv run python recommend_model.py

# Use specific date for analysis
uv run python recommend_model.py --date 2025-08-15

# Custom lookback periods
uv run python recommend_model.py --lookbacks "25,50,100"
uv run python recommend_model.py --lookbacks "60,120,240"

# Use Monte Carlo optimized parameters
uv run python recommend_model.py --lookbacks use-saved

# Combine date and optimized parameters
uv run python recommend_model.py --date 2025-08-01 --lookbacks use-saved
```

#### State Management and Debugging

```bash
# Inspect current Monte Carlo state
uv run python modify_saved_state.py inspect

# View state from custom file
uv run python modify_saved_state.py inspect --file custom_state.pkl

# Remove problematic combinations
uv run python modify_saved_state.py remove-lookback 150
uv run python modify_saved_state.py remove-combination 50 150 250

# Reset state (with confirmation)
uv run python modify_saved_state.py reset

# Reset state without backup or confirmation (for automation)
uv run python modify_saved_state.py reset --no-backup --no-confirm

# Reset specific file without confirmation
uv run python modify_saved_state.py reset --file custom_state.pkl --no-confirm
```

### Workflow Examples

#### Weekly Optimization Workflow

```bash
#!/bin/bash
# Weekly parameter optimization with exploration focus

echo "Starting weekly Monte Carlo optimization..."

# Run exploration cycles for parameter discovery
./run_monte_carlo.sh 10 explore --verbose

# Follow with exploitation cycles for refinement
./run_monte_carlo.sh 5 exploit --verbose

# Generate final recommendations
uv run python recommend_model.py --lookbacks use-saved

echo "Weekly optimization completed"
```

#### Monthly Trading Workflow

```bash
#!/bin/bash
# Monthly trading recommendation workflow

MONTH_START=$(date -d "$(date +'%Y-%m-01')" +'%Y-%m-%d')

echo "Generating monthly trading recommendations..."

# Update parameters with recent exploration
./run_monte_carlo.sh 3 explore-exploit --verbose

# Generate month-start recommendations
uv run python recommend_model.py --date "$MONTH_START" --lookbacks use-saved

# Generate current date recommendations for comparison
uv run python recommend_model.py --lookbacks use-saved

echo "Monthly recommendations generated"
```

#### Comparative Analysis Workflow

```bash
#!/bin/bash
# Compare different search strategies

echo "Running comparative analysis of search strategies..."

# Save current state
cp monte_carlo_state.pkl monte_carlo_state.backup.pkl

# Test pure exploration (with reset)
echo "Testing pure exploration..."
./run_monte_carlo.sh 5 explore --reset > exploration_results.log

# Test pure exploitation
echo "Testing pure exploitation..."
./run_monte_carlo.sh 5 exploit --reset > exploitation_results.log

# Test balanced approach
echo "Testing explore-exploit..."
./run_monte_carlo.sh 5 explore-exploit --reset > balanced_results.log

# Restore original state
mv monte_carlo_state.backup.pkl monte_carlo_state.pkl

echo "Comparative analysis completed"
```

### Configuration Examples

#### High-Performance Configuration

```json
{
  "monte_carlo": {
    "max_iterations": 1000,
    "min_iterations_for_exploit": 100,
    "trading_frequency": "monthly",
    "min_lookback": 20,
    "max_lookback": 500,
    "data_format": "backtested"
  },
  "model_selection": {
    "n_lookbacks": 3,
    "performance_metrics": {
      "sharpe_ratio_weight": 2.0,
      "sortino_ratio_weight": 2.0,
      "max_drawdown_weight": 1.5,
      "avg_drawdown_weight": 1.0,
      "annualized_return_weight": 1.5
    }
  }
}
```

#### Conservative Configuration

```json
{
  "monte_carlo": {
    "max_iterations": 250,
    "min_iterations_for_exploit": 50,
    "trading_frequency": "monthly",
    "min_lookback": 30,
    "max_lookback": 200,
    "data_format": "backtested"
  },
  "model_selection": {
    "n_lookbacks": 3,
    "performance_metrics": {
      "max_drawdown_weight": 2.0,
      "avg_drawdown_weight": 1.5,
      "sharpe_ratio_weight": 1.0,
      "sortino_ratio_weight": 1.0,
      "annualized_return_weight": 0.5
    }
  }
}
```

### Output Examples and Interpretation

#### Monte Carlo Output

```
==============================================================
Monte Carlo Loop Runner
==============================================================
Number of runs: 5
Search strategy: explore-exploit
Verbose mode: enabled
Reset state after each run: disabled
Start time: Sat Aug 17 10:30:00 PDT 2025
Working directory: /Users/donaldpg/PyProjects/worktree/PyTAAA.master
==============================================================

--------------------------------------------------------------
Starting Monte Carlo Run 1 of 5
Time: Sat Aug 17 10:30:05 PDT 2025
--------------------------------------------------------------
Executing: uv run python run_monte_carlo.py --search explore-exploit --verbose

Running 250 Monte Carlo iterations...
Using 5 models over 8729 trading days
Press Ctrl+C to stop early with current best result

==================================================
NEW BEST PERFORMANCE FOUND!
==================================================
Final Value: $2,847,891
Annual Return: 45.23%
Sharpe Ratio: 1.8942
Sortino Ratio: 2.156
Max Drawdown: -38.2%
Avg Drawdown: -8.4%
Normalized Score: 0.847
Lookback periods: [67, 143, 289] days
==================================================

✓ Run 1 completed successfully
Progress: 1/5 runs completed (1 successful, 0 failed)
```

#### Recommendation Output

```
============================================================
MODEL RECOMMENDATION RESULTS
============================================================
Recommendation Parameters:
  Lookback periods: [67, 143, 289] days (from saved state)
  Target date: 2025-08-17 (Saturday)
  First weekday of month: 2025-08-01 (Friday)

----------------------------------------
Model ranks on 2025-08-17:
----------------------------------------
1. naz100_hma     1.234
2. sp500_hma      0.987
3. naz100_pi      0.765
4. naz100_pine    0.543
5. cash          0.000

Recommended model: naz100_hma

----------------------------------------
Model ranks on 2025-08-01:
----------------------------------------
1. sp500_hma      1.456
2. naz100_hma     1.234
3. naz100_pi      0.876
4. naz100_pine    0.654
5. cash          0.000

Recommended model: sp500_hma

============================================================
RECOMMENDATION SUMMARY
============================================================
Current date recommendation: naz100_hma (score: 1.234)
Month start recommendation: sp500_hma (score: 1.456)
Analysis based on optimized parameters from Monte Carlo search
Review recommendations and manually update portfolio holdings
============================================================
```

### Performance Monitoring

#### Log File Analysis

```bash
# Monitor real-time progress
tail -f monte_carlo_run.log

# View exploitation statistics
grep "Exploitation rate" monte_carlo_exploitation.log

# Check for errors
grep "ERROR\|WARNING" *.log

# View best performance history
grep "NEW BEST PERFORMANCE" monte_carlo_run.log
```

#### State Analysis

```bash
# View top performing combinations
uv run python modify_saved_state.py inspect | head -20

# Count total combinations explored
uv run python modify_saved_state.py inspect | grep "Total canonical combinations"

# Check efficiency improvement
uv run python modify_saved_state.py inspect | grep "improvement factor"
```

### Troubleshooting Examples

#### Common Issues and Solutions

```bash
# Issue: No previous state file
# Solution: Initialize with exploration
./run_monte_carlo.sh 1 explore

# Issue: Corrupted state file
# Solution: Reset state and restart
uv run python modify_saved_state.py reset --no-backup
./run_monte_carlo.sh 5 explore-exploit

# Issue: Insufficient data for lookbacks
# Solution: Use smaller lookback ranges
uv run python recommend_model.py --lookbacks "20,40,80"

# Issue: Memory issues with large state files
# Solution: Clean up old combinations
uv run python modify_saved_state.py remove-lookback 500
```

#### Data Validation

```bash
# Check data file existence and format
for model in naz100_pine naz100_hma naz100_pi sp500_hma; do
    file="/Users/donaldpg/pyTAAA_data/$model/data_store/pyTAAAweb_backtestPortfolioValue.params"
    if [ -f "$file" ]; then
        echo "$model: $(wc -l < "$file") lines"
        head -3 "$file"
        echo "---"
    else
        echo "$model: FILE MISSING"
    fi
done
```

### Best Practices Summary

#### For Development and Testing

```bash
# Development cycle with frequent resets
./run_monte_carlo.sh 3 explore --reset --verbose

# Testing with different strategies
for strategy in explore exploit explore-exploit; do
    echo "Testing $strategy..."
    ./run_monte_carlo.sh 2 $strategy --reset > "test_$strategy.log"
done
```

#### For Production Use

```bash
# Weekly optimization (persistent state)
./run_monte_carlo.sh 15 explore-exploit --verbose

# Monthly recommendations with optimized parameters
uv run python recommend_model.py --lookbacks use-saved

# Backup state before major changes
cp monte_carlo_state.pkl "monte_carlo_state.backup_$(date +%Y%m%d).pkl"
```

#### For Analysis and Research

```bash
# Generate independent parameter sets for comparison
./run_monte_carlo.sh 10 explore --reset > independent_run_1.log
./run_monte_carlo.sh 10 explore --reset > independent_run_2.log
./run_monte_carlo.sh 10 explore --reset > independent_run_3.log

# Long-term parameter evolution study
./run_monte_carlo.sh 50 explore-exploit --verbose > evolution_study.log
```

This comprehensive set of examples covers all major usage patterns and provides practical guidance for both development and production use of the PyTAAA Model-Switching Trading System.

---