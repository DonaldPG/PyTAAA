# Implementation Guide: 5-Column Backtest Output Format

## Overview

This guide provides step-by-step instructions to implement a 5-column format for backtest output files that captures portfolio values and new highs/lows counts. Use this to replicate the changes in another codebase.

## Requirements

- Python with numpy
- Backtest functions that write portfolio values
- Plot functions that read backtest output
- `newHighsAndLows` function available (from CountNewHighsLows.py)

## Part 1: File Format Specification

### Target Format

**File:** `pyTAAAweb_backtestPortfolioValue.params`

**Columns:** 5 space-separated values per line
1. Date (YYYY-MM-DD format)
2. Buy-and-hold portfolio value (float)
3. Traded portfolio value (float)
4. New highs count (float with 1 decimal)
5. New lows count (float with 1 decimal)

**Example:**
```
1991-01-02 10000.0 10000.0 0.0 0.0
1991-01-03 10015.2 10015.2 5.3 2.1
1991-01-04 10023.7 10020.5 8.7 1.4
```

**Important:** Use **single space** as separator (not double space or tabs)

## Part 2: Modify Backtest Write Functions

### Step 1: Import Required Function

Add to imports:
```python
from functions.CountNewHighsLows import newHighsAndLows
```

### Step 2: Call newHighsAndLows Function

Before the write loop, add computation:

```python
# Compute new highs and lows counts
# Note: ALL six parameters MUST be tuples of same length
if stockList == 'Naz100':
    newHighs_2D, newLows_2D, _ = newHighsAndLows(
        json_fn,
        num_days_highlow=(73, 293),
        num_days_cumu=(50, 159),
        HighLowRatio=(1.654, 2.019),
        HighPctile=(8.499, 8.952),
        HGamma=(1., 1.),
        LGamma=(1.176, 1.223),
        makeQCPlots=False,
        outputStats=False
    )
elif stockList == 'SP500':
    newHighs_2D, newLows_2D, _ = newHighsAndLows(
        json_fn,
        num_days_highlow=(73, 146),
        num_days_cumu=(76, 108),
        HighLowRatio=(2.293, 1.573),
        HighPctile=(12.197, 11.534),
        HGamma=(1.157, .568),
        LGamma=(.667, 1.697),
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

### Step 3: Update Write Loop

Change from 3-column to 5-column format:

**Before:**
```python
for idate in range(len(datearray)):
    textmessage += f"{datearray[idate]} {BuyHoldPortfolioValue[idate]} {np.average(monthvalue[:,idate])}\n"
```

**After:**
```python
for idate in range(len(datearray)):
    textmessage += f"{datearray[idate]} {BuyHoldPortfolioValue[idate]} {np.average(monthvalue[:,idate])} {sumNewHighs[idate]:.1f} {sumNewLows[idate]:.1f}\n"
```

**Key Points:**
- Use `{value:.1f}` format for 1 decimal place on counts
- Use **single space** between columns (not `"  "` double space)
- Ensure arrays are all same length (should match `len(datearray)`)

## Part 3: Update Plot Reader Functions

### Step 4: Find All File Readers

Search for code that reads `pyTAAAweb_backtestPortfolioValue.params`:

```bash
grep -n "pyTAAAweb_backtestPortfolioValue.params" functions/*.py
```

### Step 5: Update Reading Logic

For each location that reads the file, update column indices:

**Before:**
```python
statusline_list = statusline.split(" ")
if len(statusline_list) == 5:
    backtestDate.append(datetime.datetime.strptime(statusline_list[0], '%Y-%m-%d'))
    backtestBHvalue.append(float(statusline_list[2]))      # WRONG INDEX
    backtestSystemvalue.append(float(statusline_list[4]))  # WRONG INDEX
```

**After:**
```python
statusline_list = statusline.split(" ")
if len(statusline_list) == 5:
    backtestDate.append(datetime.datetime.strptime(statusline_list[0], '%Y-%m-%d'))
    backtestBHvalue.append(float(statusline_list[1]))      # ✓ CORRECT
    backtestSystemvalue.append(float(statusline_list[2]))  # ✓ CORRECT
```

**Column Index Mapping:**
- Index 0: date string
- Index 1: buy-and-hold value ← **use this**
- Index 2: traded value ← **use this**
- Index 3: new highs count (available but not used in plots yet)
- Index 4: new lows count (available but not used in plots yet)

## Part 4: Improve Error Handling (Optional but Recommended)

### Step 6: Add Error Reporting to Silent Exception Handlers

**Before:**
```python
for i in range(numlines):
    try:
        statusline = lines[i]
        statusline_list = statusline.split(" ")
        if len(statusline_list) == 5:
            # ... parse columns ...
    except:
        break  # SILENT ERROR - NO INFORMATION
```

**After:**
```python
for i in range(numlines):
    try:
        statusline = lines[i]
        statusline_list = statusline.split(" ")
        if len(statusline_list) == 5:
            # ... parse columns ...
    except Exception as e:
        if statusline.strip():  # Only print if line is not empty
            print(f" Warning: Error parsing line {i} in backtestPortfolioValue: {e}")
            print(f"   Line content: {statusline[:100]}")
            print(f"   Split result length: {len(statusline_list)}, content: {statusline_list}")
        break
```

## Part 5: Testing & Validation

### Step 7: Verify File Format

After running backtest, check the output file:

```bash
# Check file exists and has correct number of lines
wc -l /path/to/pyTAAAweb_backtestPortfolioValue.params

# Check first few lines have 5 columns
head -5 /path/to/pyTAAAweb_backtestPortfolioValue.params

# Verify all lines have exactly 5 columns
awk '{print NF}' /path/to/pyTAAAweb_backtestPortfolioValue.params | sort -u
# Should print: 5
```

### Step 8: Test Plot Generation

Run the plot generation functions and verify:
- No parsing errors appear
- Plots are generated successfully
- Values look reasonable (not zeros or NaN)

## Common Issues & Solutions

### Issue: Array Dimension Mismatch

**Symptoms:** ValueError about array shapes not matching

**Solution:** Ensure flattening is done correctly:
```python
# After summing across stocks
if sumNewHighs.ndim > 1:
    sumNewHighs = np.sum(sumNewHighs, axis=-1)
```

### Issue: All Parameters Must Be Tuples

**Symptoms:** TypeError or incorrect array shapes

**Solution:** ALL six parameters must be tuples of same length:
```python
# ✓ CORRECT - all tuples, same length
num_days_highlow=(73, 293)
num_days_cumu=(50, 159)
HighLowRatio=(1.654, 2.019)
HighPctile=(8.499, 8.952)
HGamma=(1., 1.)
LGamma=(1.176, 1.223)

# ✗ WRONG - mixing scalars and tuples
num_days_highlow=(73, 293)  # tuple
num_days_cumu=50            # scalar - ERROR!
```

### Issue: Double Spaces in Output

**Symptoms:** `split(" ")` creates empty strings, len() checks fail

**Solution:** Use single space separator:
```python
# ✓ CORRECT
textmessage += f"{date} {value1} {value2}\n"

# ✗ WRONG
textmessage += f"{date}  {value1}  {value2}\n"  # double spaces
```

### Issue: Column Index Off-by-One

**Symptoms:** Plot values look wrong or are zero

**Solution:** Remember Python 0-indexing:
- `statusline_list[0]` = date
- `statusline_list[1]` = buyhold value
- `statusline_list[2]` = traded value
- `statusline_list[3]` = new highs
- `statusline_list[4]` = new lows

## Files to Modify (Typical)

1. **Backtest Writers:**
   - `functions/dailyBacktest.py`
   - `functions/dailyBacktest_pctLong.py`
   - Any other functions that write portfolio values

2. **Plot Readers:**
   - `functions/MakeValuePlot.py` (multiple locations)
   - Any custom plotting functions

## Validation Checklist

- [ ] File has exactly 5 space-separated columns
- [ ] First column is date in YYYY-MM-DD format
- [ ] Columns 2-5 are numeric (floats)
- [ ] New highs/lows have 1 decimal place
- [ ] All lines have same number of columns
- [ ] No double spaces between columns
- [ ] Array dimensions match (all 1D, same length as datearray)
- [ ] Plot functions read columns [1] and [2] for values
- [ ] Error handling reports parsing issues
- [ ] Plots generate without errors

## Parameter Tuning

The parameters for `newHighsAndLows` can be tuned based on your stock list:

**Naz100 (Technology-heavy):**
- Shorter lookback periods: (73, 293) days
- Moderate smoothing: (50, 159) days
- Lower ratio: (1.654, 2.019)

**SP500 (Broader market):**
- Similar short lookback: (73, 146) days
- Different smoothing: (76, 108) days
- Higher ratio: (2.293, 1.573)

Adjust based on:
- Market volatility characteristics
- Average stock count in universe
- Desired signal sensitivity

## Summary

This implementation:
1. Adds 2 new columns to backtest output (new highs/lows)
2. Maintains single-space separation for parsing
3. Uses 1 decimal place for count values
4. Updates all plot readers to use correct column indices
5. Improves error reporting for debugging

The format is extensible - additional columns can be added later by following the same pattern.
