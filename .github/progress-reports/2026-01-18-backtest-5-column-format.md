# Progress Report: Backtest 5-Column Format Implementation

**Date:** January 18, 2026  
**Status:** ✓ Completed

## Summary

Successfully implemented 5-column format for `pyTAAAweb_backtestPortfolioValue.params` file to capture new highs/lows counts alongside backtest portfolio values. Updated all file writers and plot readers to support the new format.

## Objectives

- Capture 4 time series in backtest output: mean portfolio value, traded portfolio value, new highs count, new lows count
- Minimal refactoring to existing codebase
- Maintain backward compatibility for plot generation

## Changes Implemented

### 1. Backtest Output Files Modified

**File Format Change:**
- **Old:** 3 columns - `date buyhold_value traded_value`
- **New:** 5 columns - `date buyhold_value traded_value new_highs new_lows`
- **Separator:** Single space (important for `split(" ")` parsing)
- **Decimal Format:** 1 decimal place for counts (e.g., `123.4`)

#### Modified Files:

**A. functions/dailyBacktest.py** (Lines 343-376)
- Added newHighsAndLows computation with tuple parameters
- Parameters: `num_days_highlow=(73,293)`, `num_days_cumu=(50,159)`, `HighLowRatio=(1.654,2.019)`, `HighPctile=(8.499,8.952)`, `HGamma=(1.,1.)`, `LGamma=(1.176,1.223)`
- Returns 3D arrays: `(num_stocks, num_dates, num_parameter_sets)`
- Summed across stocks (axis=0) then parameter sets (axis=-1) to get 1D arrays
- Write format: `f"{datearray[idate]} {BuyHoldPortfolioValue[idate]} {np.average(monthvalue[:,idate])} {sumNewHighs[idate]:.1f} {sumNewLows[idate]:.1f}"`

**B. functions/dailyBacktest_pctLong.py** (Lines 1630-1661, 3095-3107)
- Monte Carlo backtest with 17 trials on macOS
- Conditional parameters based on stockList (Naz100 vs SP500)
- Same array handling and write format as dailyBacktest.py
- Only runs if plot file >20 hours old

### 2. Plot Reading Functions Fixed

All plot functions reading `pyTAAAweb_backtestPortfolioValue.params` were updated from reading columns [0,2,4] to [0,1,2]:

**A. functions/MakeValuePlot.py** (4 locations fixed)
- Line 456-457: PyTAAA_backtestWithTrend.png
- Line 878-879: PyTAAA_backtestWithOffsetChannelSignal.png
- Changed `statusline_list[2]` → `statusline_list[1]` (buyhold)
- Changed `statusline_list[4]` → `statusline_list[2]` (traded)

**B. new_functions/MakeValuePlot.py** (2 locations fixed)
- Line 381-382: PyTAAA_backtestWithTrend.png
- Line 759-760: PyTAAA_backtestWithOffsetChannelSignal.png

### 3. Error Handling Improvements

**functions/MakeValuePlot.py** - Added error reporting to 7 silent exception handlers:
- Line 55: PyTAAA_status.params parsing
- Line 233: numberUptrendingStocks_status.params (1st)
- Line 264: MeanTrendDispersion_status.params
- Line 385: DailyChannelOffsetSignal_status.params
- Line 435: numberUptrendingStocks_status.params (2nd)
- Line 473: backtestPortfolioValue.params (1st plot)
- Line 899: backtestPortfolioValue.params (2nd plot)

Changed from:
```python
except:
    break
```

To:
```python
except Exception as e:
    if statusline.strip():
        print(f" Warning: Error parsing line {i} in [filename]: {e}")
        print(f"   Line content: {statusline[:100]}")
        print(f"   Split result length: {len(statusline_list)}, content: {statusline_list}")
    break
```

## Technical Details

### newHighsAndLows Function

**Location:** functions/CountNewHighsLows.py

**Key Requirements:**
- ALL six parameters must be tuples of same length when using tuple mode:
  - `num_days_highlow`
  - `num_days_cumu`
  - `HighLowRatio`
  - `HighPctile`
  - `HGamma`
  - `LGamma`

**Returns:**
- 3D arrays with shape: `(num_stocks, num_dates, num_parameter_sets)`
- Must sum across axis=0 (stocks) then axis=-1 (parameter sets) to get 1D

### Array Dimension Handling

```python
# Call newHighsAndLows
newHighs_2D, newLows_2D, _ = newHighsAndLows(
    json_fn, 
    num_days_highlow=(73,293),
    num_days_cumu=(50,159),
    HighLowRatio=(1.654,2.019),
    HighPctile=(8.499,8.952),
    HGamma=(1.,1.),
    LGamma=(1.176,1.223),
    makeQCPlots=False,
    outputStats=False
)

# Sum across stocks (axis=0)
sumNewHighs = np.sum(newHighs_2D, axis=0)
sumNewLows = np.sum(newLows_2D, axis=0)

# Flatten if multi-dimensional (sum across parameter sets)
if sumNewHighs.ndim > 1:
    sumNewHighs = np.sum(sumNewHighs, axis=-1)
if sumNewLows.ndim > 1:
    sumNewLows = np.sum(sumNewLows, axis=-1)
```

### File Format Specification

**Output File:** `{performance_store}/pyTAAAweb_backtestPortfolioValue.params`

**Format:**
```
YYYY-MM-DD buyhold_value traded_value new_highs_count new_lows_count
```

**Example:**
```
1991-01-02 10000.0 10000.0 0.0 0.0
1991-01-03 10015.2 10015.2 5.3 2.1
1991-01-04 10023.7 10020.5 8.7 1.4
```

**Parsing:**
```python
statusline_list = statusline.split(" ")
if len(statusline_list) == 5:
    date = datetime.datetime.strptime(statusline_list[0], '%Y-%m-%d')
    buyhold_value = float(statusline_list[1])
    traded_value = float(statusline_list[2])
    new_highs = float(statusline_list[3])
    new_lows = float(statusline_list[4])
```

## Testing & Validation

### Verification Steps Completed:
✓ Both backtest files write 5-column format with single spaces  
✓ Format: "date buyhold_value traded_value new_highs.1 new_lows.1"  
✓ Arrays properly dimensioned (1D) to match datearray length  
✓ All plot reading functions use correct column indices [1] and [2]  
✓ Error reporting added to catch parsing issues  
✓ File written with 8826 lines (confirmed correct length)  

### Files Written:
- `/Users/donaldpg/pyTAAA_data/naz100_pine/data_store/pyTAAAweb_backtestPortfolioValue.params`

## Known Issues & Solutions

### Issue 1: Function Signature Mismatch
**Problem:** newHighsAndLows expects `json_fn` as first param  
**Solution:** Changed to call `newHighsAndLows(json_fn, ...)` letting function load its own data

### Issue 2: Tuple Parameter Requirement
**Problem:** All 6 parameters must be tuples when using tuple mode  
**Solution:** Ensured all parameters use matching-length tuples

### Issue 3: Array Dimension Mismatch
**Problem:** 3D arrays returned, needed 1D to match datearray  
**Solution:** Added conditional array flattening with axis=-1 sum

### Issue 4: File Format Parsing
**Problem:** Double spaces caused `split(" ")` to create extra empty strings  
**Solution:** Changed output format from double spaces to single spaces

### Issue 5: Silent Error Handling
**Problem:** Inner except blocks swallowing errors without reporting  
**Solution:** Added detailed error messages with line numbers and content

## Parameters Used

### Naz100:
```python
num_days_highlow=(73,293)
num_days_cumu=(50,159)
HighLowRatio=(1.654,2.019)
HighPctile=(8.499,8.952)
HGamma=(1.,1.)
LGamma=(1.176,1.223)
```

### SP500:
```python
num_days_highlow=(73,146)
num_days_cumu=(76,108)
HighLowRatio=(2.293,1.573)
HighPctile=(12.197,11.534)
HGamma=(1.157,.568)
LGamma=(.667,1.697)
```

## Impact

- **Backtest Functions:** 2 files modified (regular + Monte Carlo)
- **Plot Functions:** 2 files modified (4 plot readers total)
- **Error Handling:** 7 silent exception handlers improved
- **Backward Compatibility:** Plot functions still read columns 1-2 correctly

## Next Steps

- Run pytaaa_main.py to verify complete integration
- Monitor for parsing errors via new error reporting
- Consider extracting new highs/lows data for separate analysis
- Document parameters for optimization studies

## References

- Implementation plan: `.agentic-docs/product/backtest-timeseries-capture-plan.md`
- Test files: `tests/test_new_highs_lows_capture.py`
- Integration test: `tests/integration_test_backtest_output.py`
