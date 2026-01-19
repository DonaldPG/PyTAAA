# Recommendation System Architecture

**Last Updated:** January 19, 2026

## Overview

The recommendation system generates model recommendations for manual trading decisions based on the Abacus model-switching methodology. This document describes the refactored architecture after reducing `recommend_model.py` from 671 lines to 165 lines (75% reduction).

## Architecture

### Module Organization

```
recommend_model.py (165 lines)
├── CLI interface and orchestration
├── Configuration loading
└── High-level workflow coordination

functions/abacus_recommend.py (587 lines)
├── ConfigurationHelper - Configuration management
├── DateHelper - Date calculations
├── ModelRecommender - Recommendation generation
├── RecommendationDisplay - Output formatting
└── PlotGenerator - Plot generation

functions/abacus_backtest.py (116 lines)
└── BacktestDataLoader - Model data loading

functions/MonteCarloBacktest.py (existing)
└── Core Monte Carlo simulation engine
```

## Module Responsibilities

### 1. recommend_model.py
**Purpose:** CLI entry point and workflow orchestration

**Responsibilities:**
- Parse command-line arguments
- Load JSON configuration
- Initialize Monte Carlo backtesting
- Orchestrate recommendation generation
- Handle errors and logging

**Usage:**
```bash
# Generate recommendations with JSON config
uv run python recommend_model.py \
  --json /path/to/config.json

# Use custom lookback periods
uv run python recommend_model.py \
  --json /path/to/config.json \
  --lookbacks 50,150,250

# Use saved lookbacks from Monte Carlo optimization
uv run python recommend_model.py \
  --json /path/to/config.json \
  --lookbacks use-saved

# Generate for specific date
uv run python recommend_model.py \
  --json /path/to/config.json \
  --date 2026-01-15
```

### 2. functions/abacus_recommend.py
**Purpose:** Core recommendation engine

#### ConfigurationHelper
Manages configuration loading and validation.

**Methods:**
- `load_best_lookbacks_from_state(state_file)` - Load lookbacks from saved Monte Carlo state
- `get_recommendation_lookbacks(lookbacks_arg, config)` - Parse lookbacks from CLI or config
- `ensure_config_defaults(config)` - Validate config and populate defaults

**Example:**
```python
from functions.abacus_recommend import ConfigurationHelper

# Load configuration
config = json.load(open('config.json'))
ConfigurationHelper.ensure_config_defaults(config)

# Get lookbacks
lookbacks = ConfigurationHelper.get_recommendation_lookbacks(
    lookbacks_arg="55,157,174",
    config=config
)
# Returns: [55, 157, 174]
```

#### DateHelper
Handles date calculations for recommendations.

**Methods:**
- `get_first_weekday_of_month(target_date)` - Find first weekday of month
- `get_recommendation_dates(date_str)` - Generate recommendation dates (target + first weekday)
- `find_closest_trading_date(target, available_dates)` - Find nearest trading date

**Example:**
```python
from functions.abacus_recommend import DateHelper
from datetime import date

# Get recommendation dates
dates, target, first_weekday = DateHelper.get_recommendation_dates("2026-01-19")
# Returns:
#   dates = [date(2026, 1, 19), date(2026, 1, 1)]
#   target = date(2026, 1, 19)
#   first_weekday = date(2026, 1, 1)

# Find closest trading date
available = [date(2026, 1, 16), date(2026, 1, 17), date(2026, 1, 20)]
closest, diff = DateHelper.find_closest_trading_date(date(2026, 1, 19), available)
# Returns: date(2026, 1, 20), 1
```

#### ModelRecommender
Generates model recommendations based on historical performance.

**Methods:**
- `__init__(monte_carlo, lookbacks)` - Initialize with Monte Carlo instance
- `get_recommendation_for_date(target_date)` - Get best model for specific date
- `rank_models_at_date(date_idx, used_lookbacks)` - Rank all models by normalized score
- `generate_recommendation_text(dates, target_date, first_weekday)` - Generate plot text
- `display_recommendations(dates, target_date, first_weekday)` - Display to console + return plot text

**Example:**
```python
from functions.abacus_recommend import ModelRecommender
from datetime import date

# Initialize
recommender = ModelRecommender(monte_carlo, lookbacks=[55, 157, 174])

# Get recommendation for specific date
best_model, rankings, days_diff = recommender.get_recommendation_for_date(
    date(2026, 1, 19)
)
# Returns:
#   best_model = "naz100_pine"
#   rankings = [("naz100_pine", 0.845), ("sp500_hma", 0.723), ...]
#   days_diff = 0

# Display recommendations to console
plot_text = recommender.display_recommendations(
    dates=[date(2026, 1, 19), date(2026, 1, 1)],
    target_date=date(2026, 1, 19),
    first_weekday=date(2026, 1, 1)
)
```

#### RecommendationDisplay
Formats and displays parameter summaries.

**Methods:**
- `__init__(monte_carlo)` - Initialize with Monte Carlo instance
- `display_parameters_summary(lookbacks, model_switching_portfolio)` - Display parameter info

**Example:**
```python
from functions.abacus_recommend import RecommendationDisplay

display = RecommendationDisplay(monte_carlo)
display.display_parameters_summary(
    lookbacks=[55, 157, 174],
    model_switching_portfolio=portfolio_values
)
```

#### PlotGenerator
Generates recommendation plots with model-switching data.

**Methods:**
- `__init__(monte_carlo, config)` - Initialize with Monte Carlo instance and config
- `generate_recommendation_plot(lookbacks, dates, target_date, first_weekday, output_path)` - Generate plot

**Example:**
```python
from functions.abacus_recommend import PlotGenerator

plot_gen = PlotGenerator(monte_carlo, config)
success = plot_gen.generate_recommendation_plot(
    lookbacks=[55, 157, 174],
    dates=[date(2026, 1, 19), date(2026, 1, 1)],
    target_date=date(2026, 1, 19),
    first_weekday=date(2026, 1, 1),
    output_path="/path/to/recommendation_plot.png"
)
```

### 3. functions/abacus_backtest.py
**Purpose:** Backtest data management

#### BacktestDataLoader
Loads and validates model backtest data.

**Methods:**
- `__init__(config)` - Initialize with optional configuration
- `build_model_paths(data_format, json_config_path)` - Build model path dictionary
- `validate_model_paths(model_paths)` - Validate file existence

**Example:**
```python
from functions.abacus_backtest import BacktestDataLoader

# Initialize
loader = BacktestDataLoader(config)

# Build model paths
model_paths = loader.build_model_paths(
    data_format='backtested',
    json_config_path='/path/to/config.json'
)
# Returns:
# {
#     'cash': '',
#     'naz100_pine': '/Users/.../pyTAAAweb_backtestPortfolioValue.params',
#     'sp500_hma': '/Users/.../pyTAAAweb_backtestPortfolioValue.params',
#     ...
# }

# Validate paths
validated = loader.validate_model_paths(model_paths)
```

## Data Flow

```
1. CLI Input (recommend_model.py)
   ↓
2. Configuration Loading (ConfigurationHelper)
   ↓
3. Date Calculation (DateHelper)
   ↓
4. Model Path Building (BacktestDataLoader)
   ↓
5. Monte Carlo Initialization (MonteCarloBacktest)
   ↓
6. Recommendation Generation (ModelRecommender)
   ↓
7. Display Output (RecommendationDisplay)
   ↓
8. Plot Generation (PlotGenerator)
```

## Testing

### Test Coverage
- **28 total tests** (all passing)
- **test_abacus_recommend.py**: 21 tests
  - DateHelper: 13 tests
  - ConfigurationHelper: 6 tests
  - RecommendationDisplay: 2 tests
- **test_abacus_backtest.py**: 7 tests
  - BacktestDataLoader: 7 tests

### Running Tests
```bash
# Run all tests
uv run pytest tests/test_abacus_*.py -v

# Run specific module tests
uv run pytest tests/test_abacus_recommend.py -v
uv run pytest tests/test_abacus_backtest.py -v
```

## Configuration

### JSON Configuration Structure
```json
{
  "recommendation_mode": {
    "default_lookbacks": [55, 157, 174],
    "output_format": "both",
    "generate_plot": true,
    "show_model_ranks": true
  },
  "models": {
    "base_folder": "/Users/donaldpg/pyTAAA_data",
    "model_choices": {
      "cash": "",
      "naz100_pine": "{base_folder}/naz100_pine/data_store/{data_file}",
      "sp500_hma": "{base_folder}/sp500_hma/data_store/{data_file}"
    }
  },
  "monte_carlo": {
    "data_format": "actual",
    "trading_frequency": "monthly",
    "min_lookback": 10,
    "max_lookback": 400
  },
  "model_selection": {
    "performance_metrics": {
      "sharpe_ratio_weight": 1.0,
      "sortino_ratio_weight": 1.0,
      "max_drawdown_weight": 1.0,
      "avg_drawdown_weight": 1.0,
      "annualized_return_weight": 1.0
    }
  }
}
```

## Output

### Console Output
- Recommendation parameters (lookbacks, dates)
- Model initialization status
- Recommendation results for each date:
  - Best model selection
  - Model rankings with normalized scores
- Parameter summary:
  - Lookback periods
  - Normalization parameters
  - Portfolio performance metrics
  - Model-switching effectiveness

### Plot Output
- Model-switching portfolio value over time
- All individual model portfolios
- Recommendation analysis text overlay:
  - Analysis timestamp
  - Lookback periods used
  - Target date
  - Model-switching portfolio metrics
  - Recommendations with model rankings

## Performance Metrics

### Normalized Score Calculation
The system uses a normalized score combining multiple performance metrics:

```
normalized_score = mean([
    (sharpe_ratio - central_sharpe) / std_sharpe * weight_sharpe,
    (sortino_ratio - central_sortino) / std_sortino * weight_sortino,
    (max_drawdown - central_dd) / std_dd * weight_dd,
    (avg_drawdown - central_avgdd) / std_avgdd * weight_avgdd,
    (annual_return - central_return) / std_return * weight_return
])
```

### Model Selection
At each recommendation date:
1. Calculate lookback window performance for all models
2. Compute normalized scores using centralized normalization values
3. Rank models by normalized score (highest to lowest)
4. Select top-ranked model as recommendation

## Refactoring History

### Phase 3: Incremental Refactoring (Completed)
- **Phase 3.1**: Extracted DateHelper class (57 lines removed)
- **Phase 3.2**: Extracted ModelRecommender class (203 lines removed)
- **Phase 3.3**: Extracted RecommendationDisplay class (63 lines removed)
- **Phase 3.4**: Extracted BacktestDataLoader class (47 lines removed)
- **Phase 3.5**: Extracted ConfigurationHelper and PlotGenerator (136 lines removed)

### Results
- **Original**: 671 lines
- **Refactored**: 165 lines
- **Reduction**: 506 lines (75%)
- **Verification**: Zero differences in output across all phases

## Future Enhancements

### Potential Additions
1. **Backtest Generation Script** - Standalone tool to generate portfolio value history files
2. **Additional Display Formats** - JSON, CSV output options
3. **Historical Tracking** - Store recommendation history for analysis
4. **Performance Comparison** - Compare recommendation accuracy vs actual outcomes
5. **Multi-Timeframe Analysis** - Support different trading frequencies

## Related Documentation
- [Refactoring Plan](.github/RECOMMEND_MODEL_REFACTORING_PLAN.md)
- [Model Switching Trade System](MODEL_SWITCHING_TRADE_SYSTEM.md)
- [Daily Operations Guide](DAILY_OPERATIONS_GUIDE.md)
