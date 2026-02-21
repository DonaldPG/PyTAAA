# PyTAAA System Summary and JSON Configuration Guide

*Generated on August 23, 2025*

## Executive Summary

The PyTAAA (Python Tactical Asset Allocation Algorithm) project has evolved into a sophisticated, multi-model algorithmic trading system that implements tactical asset allocation strategies across different market universes (NASDAQ-100 and S&P 500) using various technical analysis methodologies.

## System Architecture Overview

### Core Components Accomplished

1. **Multi-Model Trading System**
   - NAZ100 Pine (percentile channels method)
   - NAZ100 HMA (Hull Moving Average method) 
   - NAZ100 PI (3 Simple Moving Average method)
   - SP500 Pine (percentile channels method)
   - SP500 HMA (Hull Moving Average method)
   - Abacus Combined Portfolio (dynamic universe switching)

2. **Advanced Portfolio Management**
   - Monte Carlo optimization for parameter selection
   - Dynamic model switching based on performance metrics
   - Automated daily portfolio updates
   - Monthly universe evaluation and switching
   - Comprehensive performance tracking and diagnostics

3. **Data Management System**
   - HDF5 data storage for efficient price data handling
   - Automated quote updates from Yahoo Finance
   - Symbol list management for different universes
   - Performance history tracking with .params files

4. **Web Interface and Reporting**
   - Automated web page generation
   - Portfolio performance charts and rankings
   - FTP deployment capabilities
   - Email and SMS notification system

5. **Backtesting and Analysis Framework**
   - Historical performance backtesting
   - Risk metrics calculation (Sharpe ratio, Sortino ratio, max drawdown)
   - Monte Carlo simulation for strategy optimization
   - Comprehensive diagnostics and validation tools

## JSON Configuration Architecture

The system uses a standardized JSON configuration approach that replaces the older .params file system. This provides better maintainability, version control, and programmatic access to configuration settings.

### 1. Main Trading System Configuration Files

#### Pattern: `pytaaa_model_switching_params.json`
**Purpose**: Master configuration for the model switching system
**Key Sections**:
- `Email`: Email notification settings
- `Text_from_email`: SMS notification configuration  
- `FTP`: Web deployment settings
- `stock_server`: Quote download server configuration
- `Setup`: Runtime and scheduling parameters
- `Valuation`: Core trading parameters and technical analysis settings
- `model_selection`: Performance metric weights for model selection
- `monte_carlo`: Monte Carlo optimization parameters
- `recommendation_mode`: Model recommendation settings
- `models`: Model path mappings and configurations

**Common Parameters**:
```json
{
    "Valuation": {
        "symbols_file": "/path/to/symbols.txt",
        "stockList": "Naz100|SP500",
        "uptrendSignalMethod": "HMAs|PIs|minmax",
        "numberStocksTraded": 7,
        "monthsToHold": 1,
        "LongPeriod": 600,
        "stddevThreshold": 8.714,
        "MA1": 176,
        "MA2": 8, 
        "MA3": 11,
        "sma2factor": 1.536,
        "rankThresholdPct": 0.1330,
        "trade_cost": 7.95,
        "window_size": 50,
        "enable_rolling_filter": true
    }
}
```

### Data Quality Controls

#### Rolling Window Data Quality Filter

**Purpose**: Ensures signal reliability by filtering stocks with insufficient historical data quality within a rolling window.

**Configuration Parameters**:
- `window_size`: Size of rolling window in trading days (default: 50)
- `enable_rolling_filter`: Enable/disable the filter (default: false for performance)

**Logic**: For each stock and date, examines the adjusted close prices in the preceding `window_size` days. If less than 50% of the data points are valid (non-NaN) and non-constant, the signal for that stock/date is set to 0.0, ensuring the portfolio defaults to 100% CASH for unreliable data.

**Integration**: Applied after technical indicator signal generation but before monthly rebalancing in backtesting scenarios only.

#### SP500 Pre-2022 CASH Allocation

**Purpose**: Forces 100% CASH allocation for SP500 data before 2022-01-01 to address early-period data quality issues.

**Logic**: For SP500 universe (`stockList == 'SP500'`), any date before 2022-01-01 automatically sets all stock signals to 0.0, ensuring complete CASH allocation. This overrides the rolling window filter when applicable.

**Rationale**: Early SP500 data may have quality issues that make stock signals unreliable. Forcing CASH allocation during this period reduces portfolio volatility while maintaining the ability to participate in later, higher-quality data periods.

**Integration**: Applied after technical indicators in both live trading and backtesting, takes precedence over rolling window filtering.

### 2. Individual Model Configuration Files

#### Pattern: `{model_name}_PyTAAA_status.params.json`
**Examples**: 
- `naz100_hma_PyTAAA_status.params.json`
- `naz100_pi_PyTAAA_status.params.json`
- `sp500_hma_PyTAAA_status.params.json`

**Purpose**: Individual model-specific configurations
**Key Sections**:
- `model_info`: Model metadata and description
- `data_paths`: File and directory paths specific to the model
- `model_parameters`: Technical analysis parameters for the specific model

**Structure**:
```json
{
    "model_info": {
        "model_name": "naz100_hma",
        "universe": "naz100",
        "methodology": "hma",
        "description": "NASDAQ-100 using Hull Moving Average methodology"
    },
    "data_paths": {
        "base_path": "/Users/donaldpg/pyTAAA_data/naz100_hma",
        "symbols_file": "/path/to/symbols.txt",
        "hdf5_file": "/path/to/data.hdf5",
        "performance_store": "/path/to/data_store",
        "webpage": "/path/to/web_output"
    },
    "model_parameters": {
        "stockList": "Naz100",
        "uptrendSignalMethod": "HMAs",
        "numberStocksTraded": 7,
        // ... additional technical parameters
    }
}
```

### 3. Abacus Combined Portfolio Configuration

#### Pattern: `abacus_config.json`
**Purpose**: Configuration for the dynamic universe-switching Abacus portfolio
**Key Features**:
- Dynamic switching between NAZ100 and SP500 universes
- Monte Carlo-based universe evaluation
- Performance-driven decision making

**Structure**:
```json
{
    "portfolio_name": "naz100_sp500_abacus",
    "current_universe": "naz100",
    "last_evaluation_date": "2025-08-01",
    "switching_enabled": true,
    "lookback_periods": [47, 177, 178],
    "data_paths": {
        "naz100": "/Users/donaldpg/pyTAAA_data/Naz100",
        "sp500": "/Users/donaldpg/pyTAAA_data/SP500"
    },
    "output_path": "/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus",
    "universe_evaluation": {
        "enabled": true,
        "method": "recommend_model",
        "comparison_metrics": ["sharpe_ratio", "sortino_ratio", "max_drawdown"],
        "decision_threshold": 0.05
    }
}
```

### 4. Monthly Update Configuration

#### Pattern: `monthly_update_config.json`
**Purpose**: Configuration for automated monthly operations
**Key Sections**:
- Holdings file management
- Portfolio diagnostics
- Performance tracking
- Automated maintenance tasks

### 5. Generic Configuration Template

#### Pattern: `pytaaa_generic.json`
**Purpose**: Template configuration for new model setups
**Usage**: Base template that can be customized for new trading models

## Key JSON Configuration Patterns and Standards

### 1. File Path Management
All JSON configurations use absolute paths for reliability:
```json
{
    "data_paths": {
        "base_path": "/Users/donaldpg/pyTAAA_data/{model_name}",
        "symbols_file": "{base_path}/symbols/{universe}_symbols.txt",
        "hdf5_file": "{base_path}/symbols/{universe}_Symbols_.hdf5",
        "performance_store": "{base_path}/data_store",
        "webpage": "{base_path}/pyTAAA_web"
    }
}
```

### 2. Technical Analysis Parameters
Standardized parameter naming across all models:
```json
{
    "model_parameters": {
        "MA1": 89,           // Primary moving average period
        "MA2": 21,           // Secondary moving average period  
        "MA3": 8,            // Tertiary moving average period
        "sma2factor": 1.0,   // SMA scaling factor
        "rankThresholdPct": 0.15,  // Ranking threshold percentage
        "stddevThreshold": 2.0,    // Standard deviation threshold
        "LongPeriod": 180,         // Long-term analysis period
        "trade_cost": 7.95         // Transaction cost per trade
    }
}
```

### 3. Performance Metrics Configuration
Consistent performance evaluation across all models:
```json
{
    "model_selection": {
        "n_lookbacks": 3,
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

### 4. Monte Carlo Configuration
Optimization parameters for strategy tuning:
```json
{
    "monte_carlo": {
        "max_iterations": 1500,
        "min_iterations_for_exploit": 375,
        "trading_frequency": "monthly",
        "use_daily_close_only": true,
        "portfolio_rebalance_day": 1,
        "min_lookback": 50,
        "max_lookback": 200
    }
}
```

## Common JSON File Instructions and Usage

### 1. Reading JSON Configurations

The system uses the `get_json_params()` function in `functions/GetParams.py`:

```python
def get_json_params(json_fn, verbose=False):
    """Load and parse JSON configuration file"""
    with open(json_fn, 'r') as json_file:
        config = json.load(json_file)
    
    # Extract sections
    email_section = config.get('Email')
    valuation_section = config.get('Valuation')
    setup_section = config.get('Setup')
    
    # Convert to parameters dictionary
    params = {}
    params['numberStocksTraded'] = int(valuation_section["numberStocksTraded"])
    params['trade_cost'] = float(valuation_section["trade_cost"])
    # ... additional parameter extraction
    
    return params
```

### 2. Path Resolution Functions

Helper functions for resolving paths from JSON configs:

```python
def get_performance_store(json_fn):
    """Get performance store path from JSON config"""
    with open(json_fn, 'r') as json_file:
        config = json.load(json_file)
    valuation_section = config.get('Valuation')
    return valuation_section["performance_store"]

def get_webpage_store(json_fn):
    """Get webpage output path from JSON config"""
    with open(json_fn, 'r') as json_file:
        config = json.load(json_file)
    valuation_section = config.get('Valuation')
    return valuation_section["webpage"]
```

### 3. Model Configuration Mapping

Dynamic model configuration selection:

```python
def get_model_config_path(trading_model: str, base_path: str = None) -> str:
    """Get appropriate JSON config file based on trading model"""
    model_config_map = {
        'naz100_hma': 'naz100_hma_PyTAAA_status.params.json',
        'naz100_pi': 'naz100_pi_PyTAAA_status.params.json', 
        'sp500_hma': 'sp500_hma_PyTAAA_status.params.json',
        'abacus_combined': 'abacus_combined_PyTAAA_status.params.json'
    }
    return os.path.join(base_path or os.getcwd(), model_config_map[trading_model])
```

### 4. Configuration Validation

Best practices for JSON config validation:

1. **Required Fields Check**: Validate all required sections exist
2. **Path Validation**: Ensure all file paths are accessible
3. **Parameter Ranges**: Validate numerical parameters are within expected ranges
4. **Type Checking**: Ensure parameters have correct data types

### 5. Configuration Updates

When updating JSON configurations:

1. **Backup First**: Always backup existing configs before modifications
2. **Validate Syntax**: Use JSON linters to ensure valid JSON syntax
3. **Test Changes**: Run validation scripts after configuration changes
4. **Version Control**: Commit configuration changes with descriptive messages

## Directory Structure and Data Flow

### Data Store Organization
```
/Users/donaldpg/pyTAAA_data/
├── Naz100/                    # NASDAQ-100 data
│   ├── symbols/               # Symbol lists and HDF5 data
│   └── data_store/           # Performance tracking files
├── SP500/                     # S&P 500 data  
│   ├── symbols/
│   └── data_store/
├── naz100_hma/               # NAZ100 HMA model
├── naz100_pi/                # NAZ100 PI model
├── sp500_hma/                # SP500 HMA model
└── naz100_sp500_abacus/      # Combined Abacus portfolio
    ├── data_store/
    ├── pyTAAA_web/
    └── config/
```

### Configuration File Locations
- **Main configs**: Project root directory
- **Model-specific configs**: Project root (named by model)
- **Runtime configs**: Generated in data_store directories
- **Abacus configs**: Project root and abacus data directory

## System Integration Points

### 1. Daily Operations
- **Quote Updates**: `pytaaa_quotes_update.py` using model configs
- **Portfolio Updates**: `daily_abacus_update.py` for Abacus portfolio
- **Performance Tracking**: Automated via main trading scripts

### 2. Monthly Operations  
- **Universe Evaluation**: `monthly_universe_evaluation.py`
- **Holdings Management**: `monthly_holdings_update.py`
- **Performance Analysis**: Comprehensive diagnostics

### 3. Web Interface
- **Automated Generation**: HTML reports and charts
- **FTP Deployment**: Automated web publishing
- **Performance Dashboards**: Real-time portfolio status

## Advanced Features Implemented

### 1. Monte Carlo Optimization
- **Parameter Space Exploration**: Automated parameter optimization
- **Performance Validation**: Backtesting with optimized parameters
- **State Management**: Persistent optimization state

### 2. Model Recommendation System
- **Performance-Based Selection**: Automated model recommendation
- **Lookback Analysis**: Multiple timeframe evaluation
- **Risk-Adjusted Metrics**: Comprehensive performance scoring

### 3. Dynamic Universe Switching
- **Abacus Portfolio**: Automated NAZ100/SP500 switching
- **Performance Monitoring**: Continuous universe evaluation
- **Seamless Transitions**: Automated data source switching

## Maintenance and Operations

### 1. Regular Maintenance Tasks
- **Daily**: Quote updates, portfolio rebalancing
- **Monthly**: Universe evaluation, holdings file updates
- **Quarterly**: Performance analysis, parameter optimization

### 2. Monitoring and Alerts
- **Email Notifications**: Trade signals and system status
- **SMS Alerts**: Critical system notifications
- **Log File Monitoring**: Comprehensive system logging

### 3. Backup and Recovery
- **Configuration Backups**: Automated config file backups  
- **Data Backups**: HDF5 and performance data backups
- **State File Management**: Monte Carlo state preservation

## Future Enhancement Opportunities

1. **Additional Universes**: Russell 2000, international markets
2. **Enhanced Risk Management**: VaR, CVaR implementation
3. **Machine Learning Integration**: ML-based parameter optimization
4. **Real-time Trading**: Interactive Brokers API integration
5. **Mobile Interface**: Mobile-responsive web interface

## Conclusion

The PyTAAA system represents a mature, production-ready algorithmic trading platform with comprehensive configuration management, automated operations, and sophisticated portfolio optimization capabilities. The JSON configuration architecture provides a flexible, maintainable foundation for continued system evolution and enhancement.

---

*This document serves as the definitive guide to the PyTAAA system architecture and JSON configuration patterns. For specific operational procedures, refer to the dedicated operational guides in the docs folder.*