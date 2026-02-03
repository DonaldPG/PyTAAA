# Model Switching Selection Comparison

This document compares how model selection is performed across three entry points in the PyTAAA system.

## Overview

All three entry points use the **same underlying model selection algorithm** (`MonteCarloBacktest._select_best_model()`), but they differ in their **purpose**, **configuration**, and **when/how they apply the selection**.

---

## Entry Point 1: recommend_model.py

### Purpose
- **Manual trading recommendations** for human decision-making
- Shows recommendations for target date AND first weekday of current month
- Generates visual plots and detailed analysis

### Configuration Source
```python
# Defaults to: pytaaa_model_switching_params.json
# Can override with: --json <custom_config_path>
```

### Lookback Configuration
```python
# Default lookbacks from config:
recommendation_lookbacks = ConfigurationHelper.get_recommendation_lookbacks(
    lookbacks, config
)
# Uses: config['recommendation_mode']['default_lookbacks']
# Currently: [55, 157, 174]
```

### Model Selection Process
```python
# 1. Initialize MonteCarloBacktest with JSON config
monte_carlo = MonteCarloBacktest(
    model_paths=model_choices,
    iterations=1,  # Single iteration for recommendation
    trading_frequency='monthly',
    search_mode='exploit',  # Use exploitation for recommendations
    json_config=config
)

# 2. Apply normalization values from JSON if available
if normalization_values:
    monte_carlo.CENTRAL_VALUES = normalization_values['central_values']
    monte_carlo.STD_VALUES = normalization_values['std_values']

# 3. Generate recommendations using ModelRecommender
recommender = ModelRecommender(monte_carlo, recommendation_lookbacks)
plot_text = recommender.display_recommendations(
    dates, target_date, first_weekday
)

# 4. Calculate model-switching portfolio for display
model_switching_portfolio = monte_carlo._calculate_model_switching_portfolio(
    recommendation_lookbacks
)
```

### Key Characteristics
- **Single iteration** (iterations=1)
- **Exploit mode** - uses best known parameters
- **Fixed lookbacks** - uses saved/configured lookbacks
- **Two-date analysis** - target date + first weekday of month
- **Human-readable output** - detailed console display + plots
- **Load previous state** - continues from monte_carlo_state.pkl if exists

### Models Used
```python
# From BacktestDataLoader.build_model_paths():
model_choices = {
    "cash": "",
    "naz100_pine": "...pyTAAAweb_backtestPortfolioValue.params",
    "naz100_hma": "...pyTAAAweb_backtestPortfolioValue.params",
    "naz100_pi": "...pyTAAAweb_backtestPortfolioValue.params",
    "sp500_hma": "...pyTAAAweb_backtestPortfolioValue.params",
    "sp500_pine": "...pyTAAAweb_backtestPortfolioValue.params",
}
```

### Selection Algorithm Call
```python
# Called by ModelRecommender.display_recommendations()
# For each recommendation date:
recommended_model, lookbacks_used = monte_carlo._select_best_model(
    target_idx,
    lookbacks=recommendation_lookbacks
)
```

---

## Entry Point 2: run_monte_carlo.sh → run_monte_carlo.py

### Purpose
- **Parameter optimization** through Monte Carlo exploration
- **Continuous learning** - each run builds on previous state
- **Performance testing** - evaluates different lookback combinations
- Finds optimal normalization values and lookback periods

### Configuration Source
```bash
# Shell script passes JSON config to Python:
./run_monte_carlo.sh 5 explore --json=config.json
```

### Lookback Configuration
```python
# DYNAMIC - randomly generated each iteration
lookbacks = monte_carlo._generate_diverse_lookbacks(
    n_lookbacks, 
    iteration=current_iteration
)

# Parameters from config:
n_lookbacks = config['model_selection']['n_lookbacks']  # Typically 3
min_lookback = config['monte_carlo']['min_lookback']    # Default: 10
max_lookback = config['monte_carlo']['max_lookback']    # Default: 252

# Example generated lookbacks:
# [45, 89, 201]  # Random, changes each iteration
# [12, 67, 189]  # Different next iteration
# [156, 234, 67] # Exploring parameter space
```

### Model Selection Process
```python
# 1. Initialize with multiple iterations
monte_carlo = MonteCarloBacktest(
    model_paths=model_choices,
    iterations=50000,  # Many iterations for exploration
    min_iterations_for_exploit=50,
    trading_frequency='monthly',
    search_mode=search,  # Can be explore/exploit/explore-exploit
    verbose=verbose,
    json_config=config
)

# 2. Apply normalization values (or randomize)
if not randomize and normalization_values:
    monte_carlo.CENTRAL_VALUES = normalization_values['central_values']
    monte_carlo.STD_VALUES = normalization_values['std_values']
else:
    # Randomize for exploration
    monte_carlo.CENTRAL_VALUES = {
        'annual_return': np.random.choice([0.425, 0.435, ...]),
        'sharpe_ratio': np.random.choice([1.35, 1.45]),
        # ... other metrics randomized
    }

# 3. Load previous state for continuous learning
if os.path.exists(state_file):
    monte_carlo.load_state(state_file)

# 4. Run optimization
results = monte_carlo.run()

# 5. Save state for next run
monte_carlo.save_state(state_file)
```

### Key Characteristics
- **Many iterations** (typically 50,000)
- **Dynamic search mode** - explore, exploit, or explore-exploit
- **Random lookbacks** - explores parameter space each iteration
- **Continuous learning** - state persists across runs
- **Automatic optimization** - finds best parameters automatically
- **Can randomize normalization** - explores different targets
- **Focus periods** - can specify custom date ranges for optimization

### Search Strategies
1. **Explore Mode** (`explore`)
   - Random parameter selection
   - Discovers new parameter combinations
   - Useful early in optimization

2. **Exploit Mode** (`exploit`)
   - Uses best known parameters
   - Refines around known good solutions
   - Useful after extensive exploration

3. **Explore-Exploit Mode** (`explore-exploit`, default)
   - Balances exploration and exploitation
   - Initially explores (first 50 iterations)
   - Gradually shifts to exploitation
   - Best for long-running optimization

### Models Used
Same as recommend_model.py but reads from either:
- `PyTAAA_status.params` (actual values)
- `pyTAAAweb_backtestPortfolioValue.params` (backtested values)

Controlled by `config['monte_carlo']['data_format']`

### Selection Algorithm Call
```python
# Called inside run() during each iteration:
# For each iteration with different random lookbacks:
for iteration in range(iterations):
    # Generate new random lookbacks
    lookbacks = self._generate_diverse_lookbacks(n_lookbacks, iteration)
    
    # Calculate portfolio with these lookbacks
    for t in trading_dates:
        current_model, _ = self._select_best_model(
            t, 
            lookbacks=lookbacks  # Random lookbacks this iteration
        )
        # Use selected model to calculate returns
```

---

## Entry Point 3: daily_abacus_update.py (Currently Disabled)

### Purpose
- **Automated daily updates** of tracking portfolio
- Updates stock prices and portfolio valuations
- **WRITES model-switching values** to params file (currently disabled)
- Updates web dashboard content

### Configuration Source
```python
# Always uses JSON config:
config = load_config_file(args.json)
# Typically: abacus_combined_PyTAAA_status.params.json
```

### Lookback Configuration
```python
# Currently DISABLED, but when enabled:
# Uses same logic as recommend_model.py
lookbacks = ConfigurationHelper.get_recommendation_lookbacks(
    "use-saved", config
)
# Uses: config['recommendation_mode']['default_lookbacks']
# Currently: [55, 157, 174]
```

### Model Selection Process (When Enabled)
```python
# CURRENTLY COMMENTED OUT in daily_abacus_update.py lines 862-888
# 
# try:
#     from functions.abacus_backtest import write_abacus_backtest_to_params_file
#     from functions.abacus_recommend import ConfigurationHelper
#     
#     # Get lookback parameters from config
#     ConfigurationHelper.ensure_config_defaults(config)
#     lookbacks = ConfigurationHelper.get_recommendation_lookbacks(
#         "use-saved", config
#     )
#     
#     # Write model-switching values to params file
#     success = write_abacus_backtest_to_params_file(
#         json_config_path=args.json,
#         lookbacks=lookbacks
#     )
# except Exception as e:
#     logger.error(f"Error updating abacus backtest file: {e}")

# Inside write_abacus_backtest_to_params_file():
# 1. Initialize MonteCarloBacktest
monte_carlo = MonteCarloBacktest(
    model_paths=model_choices,
    iterations=1,
    trading_frequency='monthly',
    search_mode='exploit',
    json_config=config
)

# 2. Calculate model-switching portfolio with selections
portfolio_values, model_selections = (
    _calculate_model_switching_portfolio_with_selections(
        monte_carlo, lookbacks
    )
)

# 3. Write to params file
# - Column 3: model-switching portfolio values
# - Column 6: selected model names for each date
```

### Key Characteristics (When Enabled)
- **Automated execution** - runs daily via cron/scheduler
- **Single iteration** (iterations=1)
- **Exploit mode** - uses best known parameters
- **Fixed lookbacks** - uses saved/configured lookbacks
- **File modification** - updates params file in-place
- **Adds model selection column** - column 6 with model names
- **Updates column 3** - overwrites with model-switching values
- **Currently DISABLED** - to test if it affects portfolio values

### Models Used
Same 5 models as recommend_model.py:
```python
model_choices = {
    "cash": "",
    "naz100_pine": "...pyTAAAweb_backtestPortfolioValue.params",
    "naz100_hma": "...pyTAAAweb_backtestPortfolioValue.params",
    "naz100_pi": "...pyTAAAweb_backtestPortfolioValue.params",
    "sp500_hma": "...pyTAAAweb_backtestPortfolioValue.params",
    "sp500_pine": "...pyTAAAweb_backtestPortfolioValue.params",
}
```

### Selection Algorithm Call
```python
# Called by _calculate_model_switching_portfolio_with_selections()
# For each trading date in history:
for t in range(1, n_dates):
    if monte_carlo._should_trade(date_val):
        if t >= max(lookbacks):
            current_model, _ = monte_carlo._select_best_model(
                t, 
                lookbacks=lookbacks  # Fixed lookbacks from config
            )
```

---

## Core Selection Algorithm: `_select_best_model()`

All three entry points use this **same underlying algorithm** in `MonteCarloBacktest._select_best_model()`:

```python
def _select_best_model(self, current_idx: int, 
                      lookbacks: Optional[List[int]] = None,
                      iteration: Optional[int] = None) -> Tuple[str, List[int]]:
    """Select best performing model based on multiple metrics and lookbacks."""
    
    # 1. Get configuration
    config = self.json_config or load_config('pytaaa_model_switching_params.json')
    metric_weights = config['model_selection']['performance_metrics']
    
    # 2. Use passed lookbacks or generate new ones
    if lookbacks is None:
        lookbacks = self._generate_diverse_lookbacks(n_lookbacks, iteration)
    
    # 3. Force consistent alphabetical ordering
    models = sorted(list(self.portfolio_histories.keys()))
    
    # 4. Calculate metrics for each lookback period
    all_ranks = np.zeros((len(models), 5 * len(lookbacks)))
    
    for i, lookback_period in enumerate(lookbacks):
        start_idx = max(0, current_idx - lookback_period)
        start_date = pd.Timestamp(self.dates[start_idx])
        end_date = pd.Timestamp(self.dates[current_idx - 1])
        
        # Calculate metrics for each model
        for model in models:
            if model == "cash":
                portfolio_values = np.ones(lookback_period) * 10000.0
            else:
                portfolio_values = self.portfolio_histories[model][start_date:end_date].values
            
            metrics = compute_daily_metrics(portfolio_values)
            metrics_list.append(metrics)
        
        # Rank models for this lookback period
        period_ranks = rank_models(metrics_list)
        
        # Apply metric weights
        weights = np.array([
            metric_weights['sharpe_ratio_weight'],
            metric_weights['sortino_ratio_weight'],
            metric_weights['max_drawdown_weight'],
            metric_weights['avg_drawdown_weight'],
            metric_weights['annualized_return_weight']
        ])
        
        all_ranks[:, i*5:(i+1)*5] = period_ranks.T * weights[:, np.newaxis].T
    
    # 5. Calculate average rank across all metrics and lookbacks
    avg_ranks = np.mean(all_ranks, axis=1)
    
    # 6. Select model with best (lowest) average rank
    best_model_idx = np.argmin(avg_ranks)
    return models[best_model_idx], lookbacks
```

### Performance Metrics Used (5 metrics)
1. **Sharpe Ratio** - risk-adjusted returns
2. **Sortino Ratio** - downside risk-adjusted returns
3. **Max Drawdown** - worst peak-to-trough decline
4. **Average Drawdown** - average of all drawdowns
5. **Annualized Return** - geometric mean annual return

### Ranking Process
- Each metric calculated for each model
- Models ranked 1st (best) to Nth (worst) for each metric
- Ranks weighted by metric importance (from config)
- Process repeated for each lookback period
- Final selection: model with **lowest average weighted rank**

---

## Key Differences Summary

| Aspect | recommend_model.py | run_monte_carlo.py | daily_abacus_update.py |
|--------|-------------------|-------------------|------------------------|
| **Purpose** | Manual recommendations | Parameter optimization | Automated updates |
| **Iterations** | 1 (single calculation) | 50,000 (exploration) | 1 (single calculation) |
| **Lookbacks** | Fixed [55, 157, 174] | Random each iteration | Fixed [55, 157, 174] |
| **Search Mode** | Exploit | Explore/Exploit/Dynamic | Exploit |
| **State Persistence** | Reads previous state | Reads & writes state | No state usage |
| **Normalization** | From JSON | Can randomize | From JSON |
| **Output** | Console + plots | State file + plots | Params file update |
| **Execution** | Manual on-demand | Manual batches | Automated daily |
| **Currently Active** | ✅ Yes | ✅ Yes | ❌ Disabled (commented out) |

---

## Selection Algorithm Consistency

### What's Identical Across All Entry Points
1. **Core algorithm**: `MonteCarloBacktest._select_best_model()`
2. **5 performance metrics**: Sharpe, Sortino, MaxDD, AvgDD, AnnReturn
3. **Ranking methodology**: Multi-metric weighted ranking
4. **Model ordering**: Alphabetically sorted for consistency
5. **Model set**: Same 5 models (naz100_pine/hma/pi, sp500_hma/pine, cash)
6. **Monthly rebalancing**: All use monthly trading frequency

### What Differs Between Entry Points
1. **Lookback periods**: Fixed vs. random
2. **Number of iterations**: 1 vs. 50,000
3. **Search strategy**: Exploit only vs. explore/exploit modes
4. **Output destination**: Console/plot vs. state file vs. params file
5. **Execution pattern**: Manual vs. automated
6. **Learning**: Static vs. continuous improvement

---

## Current Status

### Active Entry Points
1. **recommend_model.py** ✅
   - Fully operational
   - Used for manual trading decisions
   - Generates plots and recommendations

2. **run_monte_carlo.py** ✅
   - Fully operational
   - Actively optimizing parameters
   - State saved in `monte_carlo_state.pkl`

### Disabled Entry Point
3. **daily_abacus_update.py** ❌
   - Model-switching write function COMMENTED OUT (lines 862-888)
   - Testing if this affects final portfolio values
   - Main PyTAAA update still runs (prices, holdings, web content)
   - Only the params file column 3/6 update is disabled

### Reason for Disabling
User is testing whether `write_abacus_backtest_to_params_file()` causes portfolio value discrepancies. The hypothesis is that this function might be interfering with the values shown in `recommend_model.py` plot vs. the params file.

---

## Example Selection at One Point in Time

Assuming we're at date index 5000, here's how each entry point would select a model:

### recommend_model.py
```python
# Fixed lookbacks from config
lookbacks = [55, 157, 174]

# At date 5000:
# - Look back 55 days: metrics for each model over days 4945-4999
# - Look back 157 days: metrics for each model over days 4843-4999
# - Look back 174 days: metrics for each model over days 4826-4999
# Rank models for each lookback, average ranks across all 3 periods
# Result: "sp500_pine" (for example)
```

### run_monte_carlo.py
```python
# Random lookbacks generated for this iteration
lookbacks = [45, 89, 201]  # Different each time

# At date 5000:
# - Look back 45 days: metrics over days 4955-4999
# - Look back 89 days: metrics over days 4911-4999
# - Look back 201 days: metrics over days 4799-4999
# Rank models for each lookback, average ranks across all 3 periods
# Result: "naz100_hma" (might differ due to different lookbacks)
```

### daily_abacus_update.py (when enabled)
```python
# Same as recommend_model.py
lookbacks = [55, 157, 174]

# At date 5000:
# Identical calculation to recommend_model.py
# Result: "sp500_pine" (same as recommend_model.py)

# But ALSO:
# - Writes this result to column 6 of params file
# - Updates column 3 with portfolio value using this selection
```

---

## Testing Strategy

To verify consistency, you could:

1. **Run recommend_model.py** with specific date and note selected model
2. **Check state file** from run_monte_carlo.py to see if same lookbacks give same selection
3. **Re-enable daily_abacus_update.py** write function and verify column 3 matches

The key insight: **Same lookbacks + same date = same model selection**, because all three use the identical `_select_best_model()` algorithm.

---

## Recommendations

1. **Keep recommend_model.py active** - essential for manual decisions
2. **Keep run_monte_carlo.py active** - continuously improves parameters
3. **Test daily_abacus_update.py** - verify it doesn't interfere with values
4. **Ensure lookback consistency** - all should use [55, 157, 174] from config
5. **Monitor state file** - `monte_carlo_state.pkl` tracks best parameters found
6. **Verify model ordering** - alphabetical sorting prevents order-dependent bugs

---

## File Locations

```
PyTAAA/
├── recommend_model.py              # Entry point 1
├── run_monte_carlo.py              # Entry point 2 (called by shell)
├── run_monte_carlo.sh              # Shell wrapper for entry point 2
├── daily_abacus_update.py          # Entry point 3
├── functions/
│   ├── MonteCarloBacktest.py       # Core algorithm (_select_best_model)
│   ├── abacus_recommend.py         # Recommendation display logic
│   └── abacus_backtest.py          # Params file writing logic
├── pytaaa_model_switching_params.json  # Legacy config
└── abacus_combined_PyTAAA_status.params.json  # Abacus config
```

