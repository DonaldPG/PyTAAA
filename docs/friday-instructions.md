# Friday Re-Implementation Instructions: Daily Tracking Portfolio

## Quick Start: What Was Built Today (August 23, 2025)

Created `daily_abacus_update.py` system that:
- Automatically detects active trading model from holdings file `trading_model:` tag
- Updates stock prices only if not already done today by individual models
- Generates HTML web dashboard for abacus model switching system
- Uses centralized JSON configuration shared across all model switching scripts

## Core Files to Create

### 1. Enhanced JSON Configuration File
**Location**: `abacus_combined_PyTAAA_status.params.json` (local working copy)

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
  },
  "Valuation": {
    "performance_store": "/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/data_store"
  },
  "web_output_dir": "/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pyTAAA_web"
}
```

### 2. Main Script: daily_abacus_update.py
**Key Functions to Implement**:

```python
def detect_active_trading_model(holdings_file_path):
    """Read trading_model: tag from holdings file"""
    
def load_config_file(json_file_path):
    """Load and validate JSON configuration with template substitution"""
    
def update_config_with_active_model(config, active_model):
    """Update config paths based on detected active model"""
    
def validate_config_structure(config):
    """Ensure all required JSON sections exist"""
    
def ensure_required_files(config):
    """Create PyTAAA parameter files if missing"""
```

**Command Line Interface**:
```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="JSON configuration file path")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
```

## Implementation Steps

### Step 1: Create JSON Configuration
1. Copy existing config from `/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json`
2. Enhance with template-based model_choices using `{base_folder}` and `{data_file}` placeholders
3. Add all required sections (models, monte_carlo, recommendation_mode, Valuation, web_output_dir)

### Step 2: Implement Active Model Detection
- Read `/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/data_store/PyTAAA_holdings.params`
- Extract `trading_model:` line (last line of file)
- Use model_choices mapping to resolve data source paths

### Step 3: Create Daily Update Script
- Build on existing `run_pytaaa()` function infrastructure
- Add JSON configuration loading with template substitution
- Implement intelligent update logic (check if prices already updated today)
- Generate web dashboard files in configured web_output_dir

### Step 4: Set Up Data Store Structure
**Required directories**:
- `/Users/donaldpg/pyTAAA_data/naz100_hma/data_store/`
- `/Users/donaldpg/pyTAAA_data/naz100_pine/data_store/`
- `/Users/donaldpg/pyTAAA_data/naz100_pi/data_store/`
- `/Users/donaldpg/pyTAAA_data/sp500_hma/data_store/`
- `/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/data_store/`

**Required files in each data_store**:
- `PyTAAA_status.params`
- `PyTAAA_holdings.params` (abacus version needs `trading_model:` tag)
- `PyTAAA_ranks.params`

### Step 5: Configure Web Output
- Create `/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pyTAAA_web` directory
- Implement HTML dashboard generation showing current portfolio, active model, and performance

## Key Integration Points

### Template Path Resolution
```python
# Example: naz100_hma model with PyTAAA_status.params file
template = "{base_folder}/naz100_hma/data_store/{data_file}"
resolved = template.format(
    base_folder="/Users/donaldpg/pyTAAA_data",
    data_file="PyTAAA_status.params"
)
# Result: "/Users/donaldpg/pyTAAA_data/naz100_hma/data_store/PyTAAA_status.params"
```

### Holdings File Format
```
# Standard holdings content...
# Last line must be:
trading_model: naz100_hma
```

### Command Line Usage
```bash
# Basic daily update
uv run python daily_abacus_update.py --json /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json

# With verbose logging  
uv run python daily_abacus_update.py --json /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json --verbose
```

## Testing Checklist

1. **Config Loading**: Verify JSON loads with all required sections
2. **Model Detection**: Test with different trading_model values in holdings file
3. **Path Resolution**: Confirm template substitution works for all models
4. **Update Logic**: Verify stock prices update only when needed
5. **Web Generation**: Check HTML files created in web_output_dir
6. **Error Handling**: Test with missing files/directories

## Integration with Other Scripts

All model switching scripts now use the same JSON approach:
- `recommend_model.py --json <config_file>`
- `run_monte_carlo.sh` (reads model_choices from JSON)
- `daily_abacus_update.py --json <config_file>`

## Production Deployment

**Cron job**:
```bash
30 6 * * 1-5 cd /Users/donaldpg/PyProjects/worktree/PyTAAA.master && uv run python daily_abacus_update.py --json /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json
```

## Success Criteria

- ✅ Single JSON config file controls all model switching
- ✅ Active model auto-detection from holdings file
- ✅ Smart stock price updates (avoid duplicates)
- ✅ Web dashboard shows current portfolio and active model
- ✅ Template-based path configuration for flexibility
- ✅ Comprehensive error handling and logging

---

**Implementation Priority**: Focus on the core daily_abacus_update.py script first, then enhance JSON configuration and web generation. The system builds on existing PyTAAA infrastructure, so integrate rather than rewrite existing functionality.