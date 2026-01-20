# Abacus Backtest Date Padding Implementation

## Date and Context
**Date:** January 19, 2025  
**Branch:** feature/abacus_model_switching  
**Context:** Continuation of Phase 6 abacus model-switching implementation. The backtest portfolio value file was starting in January 2000 instead of the desired January 1991 start date (matching Nasdaq data range).

## Problem Statement
The `pyTAAAweb_backtestPortfolioValue.params` file was being generated with a date range starting in 2000 rather than 1991. The issue occurred because:
1. The file used the shortest model date range (SP500 models starting in 2000) instead of the longest (Nasdaq models starting in 1991)
2. The existing implementation calculated abacus values first, then prepended dates - but this meant placeholder values had abacus calculations already applied
3. Sometimes the file would start in 1991, sometimes in 2000 - needed to ensure it always starts in 1991

## Solution Overview
Restructured `write_abacus_backtest_portfolio_values()` function in [functions/abacus_backtest.py](functions/abacus_backtest.py) to implement a 3-step process:

### STEP 1: Pad to 1991 if needed
- Check if existing file starts before 1991-01-02 (Nasdaq start date)
- If not, generate padding dates from Nasdaq trading calendar
- Create 5-column padded rows with placeholder values:
  - Column 1 (buy-hold): 10000.0
  - Column 2 (system): 10000.0
  - Column 3: -1000
  - Column 4: -100000.0
  - (Column 5 added later)
- Combine padded rows with existing data

### STEP 2: Calculate abacus values
- Run MonteCarloBacktest on full date range (1991-2026)
- Generate abacus portfolio values and model names for all dates
- Uses model-switching logic with monthly rebalancing

### STEP 3: Update columns
- Replace column 3 with calculated abacus portfolio values
- Add column 6 with model names
- Model name is "CASH" (uppercase) when:
  - Portfolio value equals 10000.0, OR
  - Model returned is "cash" (case-insensitive)
- Otherwise use actual model name (e.g., "naz100_pine", "sp500_hma")

## Key Changes

### Modified Files
- [functions/abacus_backtest.py](functions/abacus_backtest.py) (~437 lines)
  - Lines 195-227: Force MonteCarloBacktest to use longest model date range
  - Lines 248-280: Read existing file into flexible list-of-lists structure
  - Lines 282-302: STEP 1 - Padding logic with Nasdaq date range
  - Lines 304-320: STEP 2 - Calculate abacus values
  - Lines 322-360: STEP 3 - Update column 3 and add column 6
  - Lines 362-380: Write final file and report statistics

### Data Structure Changes
- Changed from separate lists (existing_dates, existing_col2, etc.) to unified `existing_cols` list-of-lists
- Allows flexible handling of variable column counts during processing
- Padding creates 5 columns initially, final output has 6 columns

## Technical Details

### Date Range Sources
- **Nasdaq models** (naz100_pine, naz100_hma, naz100_pi): 8,826 dates from 1991-01-02
- **SP500 models** (sp500_hma, sp500_pine): 6,560 dates from 2000-01-03
- **Total range after processing**: 8,835 dates from 1991-01-02 to 2026-01-16

### Placeholder Values Rationale
- Columns 1 & 2 (10000.0): Represents initial $10,000 investment with no change
- Column 3 (-1000): Negative sentinel value indicating no real data available
- Column 4 (-100000.0): Negative sentinel value for missing data
- These values are intentionally chosen to be obvious placeholders that get replaced during STEP 3

### Architecture Benefits
1. **Separation of concerns**: Padding is independent of calculations
2. **Flexibility**: Handles files starting in either 1991 or 2000
3. **Clarity**: Three distinct steps make logic easier to understand and maintain
4. **Correctness**: Ensures all calculations run on complete date range

## Testing

### Test Scenario 1: File already starts in 1991
```
Result:
- Loaded 8,826 Nasdaq dates, 6,560 SP500 dates
- Using date range from naz100_pine: 8,835 dates from 1991-01-02 to 2026-01-16
- Padded 0 dates, updated 8,835 dates
- Total: 8,835 dates from 1991-01-02 to 2026-01-16
- Final Value: $9,300,662,896, Sharpe: 1.49
```

Verified output format:
```
1991-01-02 10000.0 10000.00 -1000 -100000.0 CASH
1991-01-03 10000.0 10000.00 -1000 -100000.0 CASH
...
```

### Test Scenario 2: File starts in 2000 (padding needed)
Not executed during session, but code path is implemented and ready to test.

## Integration Points

### Daily Workflow
The function is called from [functions/MakeValuePlot.py](functions/MakeValuePlot.py#L620-L623):
```python
if "abacus" in json_fn.lower():
    from functions.abacus_backtest import write_abacus_backtest_portfolio_values
    write_abacus_backtest_portfolio_values(json_fn)
```

### File Locations
- Input/Output: `{p_store}/pyTAAAweb_backtestPortfolioValue.params`
- For abacus: `/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/data_store/`

## Follow-up Items
1. ✅ Implemented 3-step padding architecture
2. ✅ Tested with file already starting in 1991 (no padding scenario)
3. ⏭️ Test with file starting in 2000 (padding scenario) - code ready but not tested
4. ⏭️ Monitor performance on next daily run to ensure integration works smoothly
5. ⏭️ Consider adding validation to verify padding was applied correctly (date continuity check)

## Commits
- Previous commits:
  - dcb317b: Initial extension of date range to 1991
  - 95f5187: Updated placeholder values
- This session:
  - Restructured to 3-step pad-first architecture

## References
- Model switching documentation: [docs/MODEL_SWITCHING_TRADE_SYSTEM.md](../MODEL_SWITCHING_TRADE_SYSTEM.md)
- Daily operations guide: [docs/DAILY_OPERATIONS_GUIDE.md](../DAILY_OPERATIONS_GUIDE.md)
- Previous session: [docs/copilot_sessions/](./README.md) (if applicable)
