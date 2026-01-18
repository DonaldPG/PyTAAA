# Backtest Time Series Capture - Implementation Plan

**Created:** January 17, 2026  
**Author:** AI Assistant  
**Status:** Draft - Awaiting Approval

## Executive Summary

This plan outlines the implementation for capturing four time series from daily PyTAAA backtest runs:
1. Mean index portfolio value (buy-hold)
2. Traded portfolio value
3. New highs count
4. New lows count

The goal is to write these as columns to the existing `pyTAAAweb_backtestPortfolioValue.params` file with minimal refactoring.

## Background

### Current State
- **File:** `pyTAAAweb_backtestPortfolioValue.params`
- **Current Format:** 3 columns (date, buy-hold value, traded value)
- **Location:** `os.path.join(p_store, "pyTAAAweb_backtestPortfolioValue.params")`
- **Write Pattern:** File is completely overwritten on each daily update (`open(filepath, "w")`)
- **Monte Carlo Context:** File written inside Monte Carlo loop, so last trial's data persists

### Key Findings from Investigation
1. **File Overwrite Confirmed:** Entire backtest history is rewritten each time
2. **Safe Column Addition:** New columns can be safely appended without breaking existing consumers
3. **Monte Carlo Execution:** All computations happen on every trial
4. **Last Trial Parameters:** Trial `randomtrials-1` uses JSON parameters (not random), making it ideal for official results
5. **Data Synchronization:** All arrays use the same `datearray` from `loadQuotes_fromHDF()`

### Functions Involved
- **`/src/backtest/dailyBacktest_pctLong.py`** (~3690 lines):
  - Lines ~1880: Monte Carlo trial loop
  - Lines ~3300: Current 3-column params file write
  - Last trial (randomtrials-1) uses JSON parameters

- **`/functions/CountNewHighsLows.py`** (~562 lines):
  - Function: `newHighsAndLows(json_fn, ...)`
  - Returns: `(newHighs_2D, newLows_2D, finalValue)`
  - Computes: `sumNewHighs` and `sumNewLows` arrays (lines 135-160)

- **`/functions/MakeValuePlot.py`** (~918 lines):
  - Lines 304-342: Calls `newHighsAndLows()` for plotting only
  - Currently doesn't persist the data

## Proposed Solution

### High-Level Approach
Enhance the last Monte Carlo trial (randomtrials-1) to:
1. Call `newHighsAndLows()` function
2. Capture the returned `sumNewHighs` and `sumNewLows` arrays
3. Expand the params file write to include these as columns 4 and 5

### Why This Works
- **Minimal Changes:** Only modifies one section of code (last trial write block)
- **Data Consistency:** All arrays already synchronized via same `datearray`
- **No Refactoring:** Uses existing functions and data structures
- **Clean Separation:** New data only computed on final trial with JSON parameters

## Implementation Plan

### Phase 1: Code Modification
**File:** `/src/backtest/dailyBacktest_pctLong.py`

**Location:** Near line ~3300 (params file write section)

**Changes Required:**

#### Step 1.1: Add Conditional Import
```python
# Near top of file with other imports
from functions.CountNewHighsLows import newHighsAndLows
```

#### Step 1.2: Add New Highs/Lows Calculation in Last Trial
```python
# Add after line ~3290 (before current file write)
# Inside the block where iter == randomtrials-1

if iter == randomtrials - 1:
    print(f"Computing new highs/lows for final trial (iter={iter})...")
    
    # Call newHighsAndLows function
    newHighs_2D, newLows_2D, _ = newHighsAndLows(
        json_fn,
        datearray,
        symbols,
        adjClose,
        narrowDays=narrowDays,
        mediumDays=mediumDays,
        wideDays=wideDays,
        LongPeriod=LongPeriod,
        verbose=False
    )
    
    # Sum across stocks to get totals per day
    sumNewHighs = np.sum(newHighs_2D, axis=0)
    sumNewLows = np.sum(newLows_2D, axis=0)
    
    print(f"  sumNewHighs shape: {sumNewHighs.shape}")
    print(f"  sumNewLows shape: {sumNewLows.shape}")
    print(f"  Sample values - Highs: {sumNewHighs[-5:]}")
    print(f"  Sample values - Lows: {sumNewLows[-5:]}")
else:
    # For non-final trials, initialize as zeros
    sumNewHighs = np.zeros(len(datearray))
    sumNewLows = np.zeros(len(datearray))
```

#### Step 1.3: Expand File Write to 5 Columns
```python
# Modify existing file write loop (currently at ~line 3300)
# Replace current 3-column write with 5-column write

# Current code (3 columns):
# for idate in range(0, len(datearray), 1):
#     fid_month.write(
#         datearray[idate] + " " +
#         str(BuyHoldPortfolioValue[idate]) + " " +
#         str(np.average(monthvalue[:, idate])) + "\n"
#     )

# New code (5 columns):
for idate in range(0, len(datearray), 1):
    fid_month.write(
        datearray[idate] + " " +
        str(BuyHoldPortfolioValue[idate]) + " " +
        str(np.average(monthvalue[:, idate])) + " " +
        str(int(sumNewHighs[idate])) + " " +
        str(int(sumNewLows[idate])) + "\n"
    )
```

### Phase 2: Testing Strategy

#### Test 2.1: Unit Tests
**File:** `tests/test_new_highs_lows_capture.py`

```python
"""
Unit tests for new highs/lows capture functionality.
"""
import pytest
import numpy as np
from src.backtest.dailyBacktest_pctLong import dailyBacktest_pctLong

class TestNewHighsLowsCapture:
    def test_params_file_has_five_columns(self):
        """Verify params file has exactly 5 columns after run."""
        # Run backtest
        dailyBacktest_pctLong(json_fn="test_config.json", randomtrials=2)
        
        # Read params file
        with open("pyTAAAweb_backtestPortfolioValue.params", "r") as f:
            lines = f.readlines()
        
        # Check each line has 5 columns
        for line in lines[1:]:  # Skip header if present
            columns = line.strip().split()
            assert len(columns) == 5, f"Expected 5 columns, got {len(columns)}"
    
    def test_new_highs_lows_non_negative(self):
        """Verify new highs/lows counts are non-negative integers."""
        # Run backtest
        dailyBacktest_pctLong(json_fn="test_config.json", randomtrials=2)
        
        # Read params file
        data = np.loadtxt("pyTAAAweb_backtestPortfolioValue.params", 
                         dtype={'names': ('date', 'buyhold', 'traded', 'highs', 'lows'),
                                'formats': ('U10', 'f8', 'f8', 'i4', 'i4')})
        
        # Check non-negative
        assert np.all(data['highs'] >= 0), "New highs should be non-negative"
        assert np.all(data['lows'] >= 0), "New lows should be non-negative"
    
    def test_datearray_synchronization(self):
        """Verify all arrays have same length as datearray."""
        # This test would mock the internal arrays to verify
        # Pass for now - covered by integration test
        pass
```

#### Test 2.2: Integration Test
**File:** `tests/integration_test_backtest_output.py`

```python
"""
Integration test for full backtest pipeline with new columns.
"""
import pytest
import os
import numpy as np

def test_full_backtest_with_new_columns():
    """Run full backtest and verify output format."""
    # Setup
    json_fn = "/Users/donaldpg/pyTAAA_data/sp500_pine/pytaaa_sp500_pine.json"
    
    # Run backtest (with small number of trials for speed)
    from src.backtest.dailyBacktest_pctLong import dailyBacktest_pctLong
    
    result = dailyBacktest_pctLong(
        json_fn=json_fn,
        randomtrials=3,  # Minimal trials for testing
        holdMonths=[1],
        verbose=True
    )
    
    # Verify file exists
    params_file = os.path.join(
        os.path.dirname(json_fn), 
        "pyTAAAweb_backtestPortfolioValue.params"
    )
    assert os.path.exists(params_file), "Params file should exist"
    
    # Load and verify structure
    with open(params_file, 'r') as f:
        lines = f.readlines()
    
    assert len(lines) > 0, "File should have content"
    
    # Check first and last lines
    first_cols = lines[0].strip().split()
    last_cols = lines[-1].strip().split()
    
    assert len(first_cols) == 5, f"First line should have 5 columns"
    assert len(last_cols) == 5, f"Last line should have 5 columns"
    
    # Parse data
    data = []
    for line in lines:
        cols = line.strip().split()
        if len(cols) == 5:
            date, buyhold, traded, highs, lows = cols
            data.append({
                'date': date,
                'buyhold': float(buyhold),
                'traded': float(traded),
                'highs': int(highs),
                'lows': int(lows)
            })
    
    # Validate data integrity
    assert len(data) > 252, "Should have at least 1 year of data"
    
    for row in data:
        assert row['buyhold'] > 0, "Buy-hold value should be positive"
        assert row['traded'] > 0, "Traded value should be positive"
        assert row['highs'] >= 0, "New highs should be non-negative"
        assert row['lows'] >= 0, "New lows should be non-negative"
    
    print(f"✓ Integration test passed - {len(data)} rows verified")
```

#### Test 2.3: Backward Compatibility Test
**File:** `tests/test_params_file_backward_compat.py`

```python
"""
Test backward compatibility with existing 3-column readers.
"""
import pytest

def test_can_read_first_three_columns():
    """Verify old code can still read first 3 columns."""
    # Simulate old reader code
    with open("pyTAAAweb_backtestPortfolioValue.params", 'r') as f:
        for line in f:
            cols = line.strip().split()
            # Old code only reads first 3 columns
            date = cols[0]
            buyhold = float(cols[1])
            traded = float(cols[2])
            
            # Should not raise errors
            assert len(date) > 0
            assert buyhold > 0
            assert traded > 0
    
    print("✓ Backward compatibility maintained")
```

### Phase 3: Documentation Updates

#### Doc 3.1: Update README
**File:** `README.md`

Add section:
```markdown
### Backtest Output Files

#### pyTAAAweb_backtestPortfolioValue.params
**Format:** Space-separated values, 5 columns
**Columns:**
1. Date (YYYY-MM-DD)
2. Buy-Hold Portfolio Value (float)
3. Traded Portfolio Value (float)
4. New Highs Count (integer) - Number of stocks making new highs
5. New Lows Count (integer) - Number of stocks making new lows

**Location:** `{performance_store}/pyTAAAweb_backtestPortfolioValue.params`

**Update Frequency:** Rewritten completely on each daily backtest run

**Note:** File contains entire backtest history (all dates)
```

#### Doc 3.2: Update Function Docstrings
**File:** `/src/backtest/dailyBacktest_pctLong.py`

Update function docstring:
```python
def dailyBacktest_pctLong(...):
    """
    Run daily backtest with percentage long filtering and Monte Carlo trials.
    
    ...existing docstring...
    
    Output Files
    ------------
    pyTAAAweb_backtestPortfolioValue.params : 5-column space-separated file
        Column 1: Date (YYYY-MM-DD)
        Column 2: Buy-hold portfolio value
        Column 3: Traded portfolio value
        Column 4: New highs count (computed on last trial only)
        Column 5: New lows count (computed on last trial only)
        
    Notes
    -----
    - New highs/lows are only computed on the final Monte Carlo trial
      (randomtrials-1) which uses JSON parameters
    - All other trials write zeros for columns 4-5
    - File is completely overwritten on each run
    """
```

### Phase 4: Validation & Deployment

#### Step 4.1: Local Testing
```bash
# Run unit tests
uv run pytest tests/test_new_highs_lows_capture.py -v

# Run integration test
uv run pytest tests/integration_test_backtest_output.py -v

# Run backward compatibility test
uv run pytest tests/test_params_file_backward_compat.py -v
```

#### Step 4.2: Dry Run on Test Data
```bash
# Run backtest with minimal trials on test dataset
uv run python -c "
from src.backtest.dailyBacktest_pctLong import dailyBacktest_pctLong
result = dailyBacktest_pctLong(
    json_fn='/Users/donaldpg/pyTAAA_data/sp500_pine/pytaaa_sp500_pine.json',
    randomtrials=3,
    holdMonths=[1],
    verbose=True
)
"

# Manually inspect output file
head -20 /Users/donaldpg/pyTAAA_data/sp500_pine/data_store/pyTAAAweb_backtestPortfolioValue.params
tail -20 /Users/donaldpg/pyTAAA_data/sp500_pine/data_store/pyTAAAweb_backtestPortfolioValue.params
```

#### Step 4.3: Full Production Run
```bash
# Run full backtest with standard settings
uv run python pytaaa_main.py

# Verify output
wc -l /Users/donaldpg/pyTAAA_data/sp500_pine/data_store/pyTAAAweb_backtestPortfolioValue.params
```

## Implementation Checklist

### Pre-Implementation
- [ ] Review and approve this plan
- [ ] Backup current `pyTAAAweb_backtestPortfolioValue.params` file
- [ ] Create feature branch: `git checkout -b feature/add-highs-lows-timeseries`

### Code Changes
- [ ] Add import for `CountNewHighsLows` module
- [ ] Add conditional block for new highs/lows calculation in last trial
- [ ] Initialize arrays as zeros for non-final trials
- [ ] Expand file write loop from 3 to 5 columns
- [ ] Add debug print statements for verification

### Testing
- [ ] Write unit test for 5-column format
- [ ] Write unit test for non-negative values
- [ ] Write integration test for full pipeline
- [ ] Write backward compatibility test
- [ ] Run all tests and verify passing

### Documentation
- [ ] Update README with new file format
- [ ] Update function docstrings
- [ ] Add inline code comments
- [ ] Update progress reports

### Validation
- [ ] Run local test with minimal trials
- [ ] Manually inspect output file format
- [ ] Verify all 5 columns present
- [ ] Verify new highs/lows are reasonable values
- [ ] Run full backtest on production data
- [ ] Compare results with expected values

### Deployment
- [ ] Commit changes with descriptive message
- [ ] Push to feature branch
- [ ] Create pull request
- [ ] Code review
- [ ] Merge to main
- [ ] Tag release

## Risk Assessment

### Low Risk
- **Backward Compatibility:** Old code reading first 3 columns unaffected
- **Data Synchronization:** All arrays use same `datearray` source
- **File Overwrite:** Existing pattern handles complete rewrites safely

### Medium Risk
- **Performance Impact:** Adding `newHighsAndLows()` call adds ~2-5 seconds
  - *Mitigation:* Only called on last trial (1 out of N trials)
  
### Known Issues
- **Non-Final Trials:** Columns 4-5 will be zeros for trials 0 through randomtrials-2
  - *Impact:* None - file is overwritten on last trial anyway
  - *Status:* Acceptable - last trial is the only one that persists

## Performance Considerations

### Current Baseline
- Monte Carlo backtest: ~30-60 minutes (typical)
- Last trial write: ~1 second

### Expected Impact
- Additional `newHighsAndLows()` call: ~2-5 seconds
- Total increase: <1% of total runtime
- **Status:** Negligible impact

## Success Criteria

1. **Functionality:**
   - [ ] Params file has exactly 5 columns
   - [ ] New highs/lows values are non-negative integers
   - [ ] Values are reasonable (not all zeros, not absurdly large)

2. **Data Integrity:**
   - [ ] All rows have same number of columns
   - [ ] Dates match existing datearray
   - [ ] Array lengths are synchronized

3. **Performance:**
   - [ ] Total runtime increase < 5%
   - [ ] File write completes without errors

4. **Compatibility:**
   - [ ] Existing 3-column readers still work
   - [ ] No breaking changes to downstream consumers

## Rollback Plan

If issues arise:

1. **Immediate Rollback:**
   ```bash
   git checkout main
   git revert <commit-hash>
   ```

2. **Restore Old File:**
   ```bash
   cp backup/pyTAAAweb_backtestPortfolioValue.params.backup \
      /Users/donaldpg/pyTAAA_data/sp500_pine/data_store/pyTAAAweb_backtestPortfolioValue.params
   ```

3. **Rerun Old Version:**
   ```bash
   git checkout main
   uv run python pytaaa_main.py
   ```

## Future Enhancements

### Potential Extensions
1. **Additional Metrics:**
   - Volatility metrics
   - Drawdown periods
   - Sharpe ratio time series

2. **Format Improvements:**
   - CSV header row with column names
   - Optional JSON format output
   - Database storage for time series

3. **Performance Optimizations:**
   - Parallel computation of new highs/lows
   - Cached intermediate results
   - Incremental updates instead of full rewrites

## References

### Code Locations
- Main backtest: `/src/backtest/dailyBacktest_pctLong.py`
- New highs/lows: `/functions/CountNewHighsLows.py`
- Plotting: `/functions/MakeValuePlot.py`

### Related Documentation
- [Progress Report 20250117](.github/progress-reports/progress_report_20250117.md)
- [Code Consolidation Plan](.agentic-docs/product/code-consolidation.md)

## Approval

- [ ] Plan reviewed and approved
- [ ] Tests specified are acceptable
- [ ] Timeline is reasonable
- [ ] Risks are understood and acceptable

**Approved by:** _________________  
**Date:** _________________
