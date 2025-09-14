# Monthly Holdings Update System

## Overview

The Monthly Holdings Update System provides automated integration with the PyTAAA recommendation system to execute monthly portfolio updates. It reads recommendations from `recommend_model.py`, simulates trades, and updates holdings files with proper transaction history.

## Features

- **Automated Recommendations**: Integrates with existing `recommend_model.py`
- **Transaction Recording**: Maintains detailed trade history in holdings files
- **Backup Protection**: Creates timestamped backups before modifications
- **Dry Run Mode**: Preview changes without executing
- **Multiple Models**: Supports all PyTAAA trading models
- **Configuration Management**: JSON-based configuration system

## Usage

### Basic Usage

```bash
# Auto-detect trading model and run update
uv run python monthly_update.py

# Specify trading model
uv run python monthly_update.py --model naz100_hma

# Preview changes without executing
uv run python monthly_update.py --dry-run

# Specify trade date
uv run python monthly_update.py --date 2025-08-01

# Verbose output
uv run python monthly_update.py --verbose
```

### Command Line Options

- `--model`: Trading model (naz100_hma, sp500_hma, naz100_pi, naz100_pine, abacus_combined)
- `--date`: Trade date in YYYY-MM-DD format (defaults to today)
- `--dry-run`: Simulation mode - no files modified
- `--config`: Path to configuration file
- `--verbose`: Enable detailed logging

## Configuration

The system uses `monthly_update_config.json` for configuration:

```json
{
  "monthly_update_config": {
    "backup_enabled": true,
    "validation_enabled": true,
    "recommend_model_path": "recommend_model.py",
    "recommendation_timeout": 300,
    "holdings_file_locations": {
      "naz100_hma": "/Users/donaldpg/pyTAAA_data/naz100_hma/data_store/PyTAAA_holdings.params",
      "sp500_hma": "/Users/donaldpg/pyTAAA_data/sp500_hma/data_store/PyTAAA_holdings.params"
    }
  },
  "trading_config": {
    "trade_cost": 7.95,
    "max_position_size": 0.10,
    "minimum_cash_reserve": 1000.0
  }
}
```

## Architecture

### Core Components

1. **MonthlyHoldingsUpdater**: Main orchestration class
2. **RecommendationIntegration**: Interface with recommend_model.py
3. **HoldingsFileManager**: File operations and backup management
4. **PortfolioCalculator**: Trade calculations and validation
5. **MonthlyUpdateConfig**: Configuration management

### Data Models

- **PortfolioState**: Current portfolio holdings
- **TradeRecommendation**: Individual trade details
- **TradeExecution**: Complete monthly update record
- **RecommendationResult**: Recommendation system output

## Output Format

The system appends monthly updates to holdings files in this format:

```
TradeDate: 2025-08-22
trading_model: naz100_hma
info:  Sell 2025-08-22 AAPL       100  160.00
info:  Buy 2025-08-22 MSFT        50  280.00
stocks:      MSFT    CASH
shares:      50      12000.0
buyprice:    280.0   1.0
```

## Integration

### With recommend_model.py

The system automatically:
1. Runs `recommend_model.py --lookbacks use-saved`
2. Parses recommendation output
3. Determines required portfolio changes
4. Simulates trade execution

### With PyTAAA System

- Uses existing `functions.GetParams` for portfolio reading
- Maintains compatibility with existing holdings file format
- Leverages existing logging infrastructure

## Error Handling

- **Backup Protection**: Automatic backups before modifications
- **Validation**: Comprehensive input/output validation
- **Rollback**: Restore from backup on errors
- **Logging**: Detailed error tracking and debugging

## Testing

Run the test suite:

```bash
uv run python tests/monthly_update/test_monthly_update.py
```

Tests cover:
- Portfolio state management
- Trade calculations
- File operations
- Recommendation parsing
- Integration scenarios

## Safety Features

1. **Dry Run Mode**: Always test before executing
2. **Automatic Backups**: Timestamped file backups
3. **Input Validation**: Comprehensive data validation
4. **Error Recovery**: Automatic rollback on failures
5. **Audit Trail**: Complete transaction logging

## Limitations

- Currently simulates trades (no actual broker integration)
- Requires manual verification of recommendations
- Limited to monthly update frequency
- Cash-only fallback for complex scenarios

## Future Enhancements

- Real broker API integration
- Multi-asset portfolio support
- Advanced trade optimization
- Performance analytics
- Email/SMS notifications