# Daily Tracking Portfolio Implementation Summary

## Overview
This document summarizes the work completed on August 23, 2025 to create a daily tracking portfolio system for PyTAAA that automatically updates stock prices and generates web content for the abacus model switching system.

## Key Achievement: daily_abacus_update.py System

### Core Implementation
Created a comprehensive daily update system (`daily_abacus_update.py`) that:
- Automatically detects the active trading model from holdings files
- Updates stock prices only if not already done today by individual models
- Generates updated web content for HTML dashboard display
- Uses centralized JSON configuration for all model switching scripts

### System Architecture

#### 1. Central JSON Configuration
- **Primary config file**: `/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json`
- **Local enhanced config file**: `abacus_combined_PyTAAA_status.params.json` (contains full configuration with model choices)
- All model switching scripts share the same JSON configuration approach
- JSON files are copied to local codebase for updates, then copied back to original location

#### 2. Trading Models System
The system manages 5 trading models as defined in the JSON configuration:
1. `naz100_pine` - NASDAQ 100 with Pine indicator
2. `naz100_hma` - NASDAQ 100 with Hull Moving Average
3. `naz100_pi` - NASDAQ 100 with Pi indicator
4. `sp500_hma` - S&P 500 with Hull Moving Average
5. `cash` - Cash position (empty path in model_choices)

#### 3. Data Store Locations
Each model maintains its own data store as specified in JSON:
- `/Users/donaldpg/pyTAAA_data/naz100_hma/data_store/PyTAAA_status.params`
- `/Users/donaldpg/pyTAAA_data/naz100_pine/data_store/PyTAAA_status.params`
- `/Users/donaldpg/pyTAAA_data/naz100_pi/data_store/PyTAAA_status.params`
- `/Users/donaldpg/pyTAAA_data/sp500_hma/data_store/PyTAAA_status.params`

#### 4. Abacus Tracking Portfolio Location
- **Data store**: `/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/data_store/`
- **Holdings file**: `PyTAAA_holdings.params` (contains `trading_model:` tag)
- **Status file**: `PyTAAA_status.params`
- **Ranks file**: `PyTAAA_ranks.params`
- **Web output**: `/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pyTAAA_web`

### Updated JSON Structure

The `abacus_combined_PyTAAA_status.params.json` file now contains:

#### Core Configuration Sections:
```json
{
  "models": {
    "base_folder": "/Users/donaldpg/pyTAAA_data",
    "model_choices": {
      "cash": "",
      "naz100_hma": "{base_folder}/naz100_hma/data_store/{data_file}",
      "sp500_hma": "{base_folder}/sp500_hma/data_store/{data_file}",
      "naz100_pi": "{base_folder}/naz100_pi/data_store/{data_file}",
      "naz100_pine": "{base_folder}/naz100_pine/data_store/{data_file}"
    }
  },
  "monte_carlo": {
    "initial_capital": 10000.0,
    "max_iterations": 1500,
    "min_iterations_for_exploit": 375,
    "data_files": {
      "actual": "PyTAAA_status.params",
      "backtested": "pyTAAAweb_backtestPortfolioValue.params"
    }
  },
  "recommendation_mode": {
    "default_lookbacks": [20, 86, 177],
    "output_format": "both",
    "generate_plot": true,
    "show_model_ranks": true
  }
}
```

### Key Features Implemented

#### 1. Active Model Detection
The system automatically detects which trading model is currently active by:
- Reading the `trading_model:` line from the abacus holdings file
- Using the model_choices configuration to dynamically build file paths
- Supporting template-based path construction with {base_folder} and {data_file} placeholders

#### 2. Intelligent Update Logic
The daily update system:
- Only updates stock prices if not already done today by individual models
- Automatically switches data sources based on active model using JSON configuration
- Creates initial files if they don't exist
- Validates all configuration before execution

#### 3. Web Content Generation
Generates HTML web dashboard files showing:
- Current portfolio value and performance
- Active trading model and holdings
- Daily price updates and trends
- Model switching recommendations
- Output stored in configured web_output_dir

### Command Line Usage

#### Basic Daily Update
```bash
uv run python daily_abacus_update.py --json /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json
```

#### With Verbose Logging
```bash
uv run python daily_abacus_update.py --json /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json --verbose
```

### Integration with Model Switching Scripts

All model switching scripts use the same JSON configuration approach:

#### 1. Monte Carlo Optimization (via run_monte_carlo.sh)
```bash
./run_monte_carlo.sh 30 --explore-exploit --reset
```
- Uses model_choices from JSON configuration
- Supports template-based file path construction

#### 2. Model Recommendation
```bash
uv run python recommend_model.py --json /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json --lookbacks "139, 149, 158"
```
- Uses recommendation_mode settings from JSON
- Default lookbacks configurable in JSON

#### 3. Daily Abacus Updates
```bash
uv run python daily_abacus_update.py --json /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json --verbose
```

## Implementation Steps to Reproduce

### Step 1: Set Up JSON Configuration
1. Create the central JSON configuration file at:
   `/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json`

2. Create local enhanced configuration:
   `abacus_combined_PyTAAA_status.params.json`

3. Ensure JSON contains all required sections:
   - `models` with `base_folder` and `model_choices`
   - `monte_carlo` configuration with data file mappings
   - `recommendation_mode` with default lookbacks
   - `Valuation` section with symbols and performance store paths
   - `web_output_dir` for HTML dashboard files

### Step 2: Configure Model Choices
The `model_choices` in the JSON uses template-based paths:
```json
"model_choices": {
  "cash": "",
  "naz100_hma": "{base_folder}/naz100_hma/data_store/{data_file}",
  "sp500_hma": "{base_folder}/sp500_hma/data_store/{data_file}",
  "naz100_pi": "{base_folder}/naz100_pi/data_store/{data_file}",
  "naz100_pine": "{base_folder}/naz100_pine/data_store/{data_file}"
}
```

### Step 3: Create Daily Update Script
1. Implement `daily_abacus_update.py` with these core functions:
   - `detect_active_trading_model()` - Reads trading_model from holdings file
   - `load_config_file()` - Loads and validates JSON configuration
   - `update_config_with_active_model()` - Updates config with active model data
   - `validate_config_structure()` - Ensures all required parameters exist
   - `ensure_required_files()` - Creates initial PyTAAA files if missing

2. Add command-line interface with `--json` parameter for configuration file

3. Integrate with existing `run_pytaaa()` function for actual updates

### Step 4: Configure Active Model Detection
1. Ensure holdings file contains `trading_model:` line at the end
2. Configure model choices in JSON with template-based paths
3. Set up automatic detection and path resolution based on active model

### Step 5: Set Up Data Stores
1. Create data store directories for each model:
   - `/Users/donaldpg/pyTAAA_data/naz100_hma/data_store/`
   - `/Users/donaldpg/pyTAAA_data/naz100_pine/data_store/`
   - `/Users/donaldpg/pyTAAA_data/naz100_pi/data_store/`
   - `/Users/donaldpg/pyTAAA_data/sp500_hma/data_store/`

2. Create abacus tracking data store:
   - `/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/data_store/`

3. Initialize required parameter files in each data store:
   - `PyTAAA_status.params`
   - `PyTAAA_holdings.params` (with trading_model tag for abacus)
   - `PyTAAA_ranks.params`

### Step 6: Configure Web Output
1. Set up web output directory: `/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pyTAAA_web`
2. Configure HTML template generation for dashboard
3. Set up automatic web file updates with current portfolio data

### Step 7: Testing and Validation
1. Test active model detection with different trading models
2. Verify configuration loading and validation
3. Test daily updates with --verbose flag
4. Validate web content generation
5. Test integration with Monte Carlo and recommendation systems

## Key Files Created/Modified

### New Files
- `daily_abacus_update.py` - Main daily update script
- `abacus_combined_PyTAAA_status.params.json` - Enhanced configuration with full model setup
- Various backup and log files for tracking execution

### Modified Files
- Updated JSON configurations to support template-based model switching
- Enhanced holdings files to include trading_model tags
- Updated web templates for dashboard display

## Configuration Requirements

### Primary JSON Structure
The main configuration file should contain:
```json
{
  "models": {
    "base_folder": "/Users/donaldpg/pyTAAA_data",
    "model_choices": {
      "cash": "",
      "naz100_hma": "{base_folder}/naz100_hma/data_store/{data_file}",
      "sp500_hma": "{base_folder}/sp500_hma/data_store/{data_file}",
      "naz100_pi": "{base_folder}/naz100_pi/data_store/{data_file}",
      "naz100_pine": "{base_folder}/naz100_pine/data_store/{data_file}"
    }
  },
  "monte_carlo": {
    "data_files": {
      "actual": "PyTAAA_status.params",
      "backtested": "pyTAAAweb_backtestPortfolioValue.params"
    }
  },
  "Valuation": {
    "performance_store": "/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/data_store",
    "symbols_file": "/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/symbols/abacus_symbols.txt"
  },
  "web_output_dir": "/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pyTAAA_web"
}
```

## Safety Features Implemented

1. **Configuration Validation** - Verifies all required parameters exist
2. **File Existence Checks** - Creates missing files automatically
3. **Active Model Detection** - Prevents wrong data source usage
4. **Template-based Path Resolution** - Flexible model configuration
5. **Comprehensive Logging** - Tracks all operations for debugging
6. **Error Handling** - Graceful failure with detailed error messages

## Production Deployment

### Cron Job Setup
```bash
# Run daily at 6:30 AM weekdays
30 6 * * 1-5 cd /Users/donaldpg/PyProjects/worktree/PyTAAA.master && uv run python daily_abacus_update.py --json /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json
```

### Manual Execution
```bash
# Standard update
uv run python daily_abacus_update.py --json /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json

# Verbose debugging
uv run python daily_abacus_update.py --json /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json --verbose
```

## Script Integration Requirements

Based on the usage instructions, all model switching scripts must use the shared JSON configuration:

1. **run_monte_carlo.py** (called by run_monte_carlo.sh)
   - Must read model_choices from JSON configuration
   - Uses template-based path resolution

2. **recommend_model.py**
   - Uses --json parameter for configuration file
   - Reads recommendation_mode settings from JSON

3. **daily_abacus_update.py** (formerly run_abacus_daily.py)
   - Uses --json parameter for configuration file
   - Performs daily stock updates and web page generation

## Success Metrics

The implementation successfully achieved:
- ✅ Centralized JSON configuration for all model switching scripts
- ✅ Template-based model path configuration with {base_folder} and {data_file} placeholders
- ✅ Automatic active model detection from holdings files
- ✅ Intelligent stock price updates (only if needed)
- ✅ Web content generation for HTML dashboard
- ✅ Integration with existing PyTAAA infrastructure
- ✅ Support for all 5 trading models (including cash)
- ✅ Comprehensive logging and error handling
- ✅ Command-line interface for manual and automated execution

## Next Steps

To extend or maintain this system:
1. Monitor daily execution logs for any issues
2. Add additional trading models by updating model_choices in JSON
3. Enhance web dashboard with more metrics
4. Integrate with alerting systems for failure notifications
5. Add performance monitoring and optimization features
6. Ensure JSON file copy workflow (local → update → copy back) is followed for configuration changes

---

*This summary captures the complete daily tracking portfolio implementation completed on August 23, 2025, including the updated JSON structure with template-based model configuration. Use this as a reference to recreate the system from a prior git commit.*