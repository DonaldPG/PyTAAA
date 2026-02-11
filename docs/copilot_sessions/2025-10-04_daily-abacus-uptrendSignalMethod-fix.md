# Copilot Session Summary: Daily Abacus Update Signal Method Fix

## Date and Context
**Date**: October 4, 2025

**Context**: The daily abacus portfolio update system was incorrectly using the HMAs signal method for all calculations, regardless of which trading model was active (e.g., naz100_pine, which should use the Pine indicator method).

## Problem Statement

When running `daily_abacus_update.py`, the system was:
1. Correctly detecting the active trading model from the `PyTAAA_holdings.params` file (e.g., `naz100_pine`)
2. Correctly routing data sources (symbols_file, stockList) based on the active model
3. **BUT** incorrectly using the abacus JSON's `"uptrendSignalMethod": "HMAs"` instead of loading the active model's signal method (e.g., `"Pine"` for naz100_pine)

This caused calculations to use the wrong technical indicator methodology, producing incorrect stock rankings and portfolio recommendations.

## Solution Overview

Modified the `update_config_with_active_model()` function in `daily_abacus_update.py` to:
1. Search for the active trading model's JSON configuration file
2. Load the model's `Valuation` section (which contains the correct `uptrendSignalMethod`)
3. Merge the model's parameters into the temporary configuration
4. Preserve abacus-specific paths (`performance_store`, `webpage`)

This ensures the temporary configuration file uses the correct signal method for the active trading model while maintaining the abacus portfolio's own data store and web output locations.

## Key Changes

**File Modified**: `daily_abacus_update.py`

**Function Updated**: `update_config_with_active_model()` (lines 323-425)

**Changes Made**:
- Added logic to search for the active model's JSON configuration in multiple common locations:
  - `{model_data_store}/pytaaa_{active_model}.json`
  - `{parent_of_data_store}/pytaaa_{active_model}.json`
  - `{base_folder}/{active_model}/pytaaa_{active_model}.json`
- Added code to load and merge the model's `Valuation` section
- Implemented preservation of critical abacus paths during the merge
- Added logging to show which signal method is being used
- Included graceful fallback if model's JSON file isn't found

**Configuration File Requirements**:
Each trading model should have its own JSON configuration file (e.g., `/Users/donaldpg/pyTAAA_data/naz100_pine/pytaaa_naz100_pine.json`) with a `Valuation` section containing the appropriate `uptrendSignalMethod`.

## Technical Details

### The Root Cause
The issue was in how the temporary configuration file was constructed. The original code:
1. ✅ Updated `symbols_file` and `stockList` based on the active model
2. ✅ Kept `performance_store` pointing to the abacus data store
3. ❌ Did NOT update `uptrendSignalMethod` from the active model's configuration

### The Fix Architecture
```python
# Search for model's JSON configuration
model_json_files = [
    "{data_store}/pytaaa_{model}.json",
    "{parent_dir}/pytaaa_{model}.json", 
    "{base_folder}/{model}/pytaaa_{model}.json"
]

# Load and merge Valuation section
if 'Valuation' in model_config:
    # Save abacus paths
    abacus_performance_store = config['Valuation']['performance_store']
    abacus_webpage = config['Valuation'].get('webpage')
    
    # Update with model's parameters
    config['Valuation'].update(model_config['Valuation'])
    
    # Restore abacus paths
    config['Valuation']['performance_store'] = abacus_performance_store
    config['Valuation']['webpage'] = abacus_webpage
```

### Key Design Decisions
1. **Search multiple locations**: Allows flexibility in where model JSON files are stored
2. **Selective merge**: Copy all model parameters BUT preserve abacus-specific paths
3. **Graceful fallback**: Continue with abacus configuration if model JSON not found
4. **Comprehensive logging**: Show which signal method is being used for debugging

## Testing

**Verification Method**: Ran `daily_abacus_update.py` with `--verbose` flag and monitored log output

**Expected Behavior**:
- System detects active model (e.g., `naz100_pine`)
- Logs show: `"Loaded Valuation section from {path}/pytaaa_naz100_pine.json"`
- Logs show: `"Using uptrendSignalMethod: Pine"` (not HMAs)
- Calculations use Pine indicator method
- Results match the active model's expected behavior

**Test Result**: ✅ Fix confirmed working by user

## Follow-up Items

1. **Verify Model JSON Files Exist**: Ensure all trading models have their JSON configuration files in the expected locations:
   - `/Users/donaldpg/pyTAAA_data/naz100_pine/pytaaa_naz100_pine.json`
   - `/Users/donaldpg/pyTAAA_data/naz100_hma/pytaaa_naz100_hma.json`
   - `/Users/donaldpg/pyTAAA_data/naz100_pi/pytaaa_naz100_pi.json`
   - `/Users/donaldpg/pyTAAA_data/sp500_hma/pytaaa_sp500_hma.json`

2. **Document Model Configuration Requirements**: Add to project documentation that each trading model must have its own JSON configuration file with proper `Valuation` section

3. **Add Unit Tests**: Consider adding tests to verify the configuration merging logic works correctly for all trading models

4. **Monitor Production Usage**: Watch for any warning logs about missing model JSON files during production runs

5. **Consider Centralized Configuration**: Future enhancement could be to store all model configurations in a single file with model-specific sections, but current approach works well for now
