# Phase 4b: Side Effects Map for PortfolioPerformanceCalcs.py

**Created:** February 13, 2026  
**File Analyzed:** `functions/PortfolioPerformanceCalcs.py` (639 lines)  
**Purpose:** Detailed mapping of all side effects to guide extraction

---

## Summary Statistics

- **Total Print Statements:** ~20
- **File Write Operations:** 2 (direct file writes)
- **Plot Generation Calls:** 2 major plotting loops (lines 350-420 and 421-540)
- **External Function Calls with Side Effects:** 
  - `computeDailyBacktest()` (line 229) - writes files
  - `sharpeWeightedRank_2D()` (line 575) - generates QC plots
  - `textmessageOutsideTrendChannel()` (line 636) - sends text messages

---

## Side Effects by Category

### 1. Print Statements (User Feedback)

| Line | Type | Content |
|------|------|---------|
| 34 | Progress | "... inside PortfolioPerformanceCalcs..." |
| 36-37 | Debug | Symbol directory and file |
| 40 | Debug | JSON directory |
| 44 | Debug | Filename for load |
| 224-226 | Diagnostic | signal2D stats (min, mean, max) |
| 257-258 | Success | Daily backtest update confirmation |
| 260 | Debug | adjClose value check |
| 274-275 | Success | Uptrending stocks file update |
| 277-278 | Error | File update failure |
| 418 | Debug | Plot filepath |
| 530 | Debug | Plot filepath |
| 539 | Error | Plot generation failure |
| 545-551 | Results | Currently uptrending symbols (in loop) |
| 602-603 | Results | B&H and portfolio final values |
| 605-607 | Results | Top ranking choices (in loop) |
| 609-613 | Debug | Final results summary |

### 2. File Write Operations

#### A. Direct File Writes (Lines 267-278)

```python
# Write uptrending stocks status
filepath = os.path.join(web_dir, "pyTAAAweb_numberUptrendingStocks_status.params")
with open(filepath, "w") as f:
    f.write(textmessage)
```

**Files Written:**
- `pyTAAAweb_numberUptrendingStocks_status.params`

**Data Format:**
```
<date> <num_uptrending> <active_count>
```

#### B. Indirect File Writes (via function calls)

**Line 229: `computeDailyBacktest()`**
- Writes: `pyTAAAweb_backtestPortfolioValue.params`
- Writes: `pyTAAAweb_backtestBuyHoldValue.params`
- Side effect: Cannot be easily separated

### 3. Plot Generation

#### A. First Plotting Loop (Lines 350-420)

**Purpose:** Generate full history plots for all symbols  
**File Pattern:** `0_<SYMBOL>.png`  
**Location:** `web_dir` from `get_webpage_store(json_fn)`

**Conditional:** Only runs if `hourOfDay >= 1 or 11 < hourOfDay < 13`

**Key Steps:**
1. Check plot file age (skip if < 20 hours old)
2. Despike quotes using `despike_2D()`
3. Create plot with:
   - Adjusted close price (black line)
   - Signal overlay (thin line)
   - Despiked prices
   - Percentile channels (if applicable)
   - Date stamp and value text
4. Save to `0_<SYMBOL>.png`

**matplotlib calls:**
- `plt.clf()`
- `plt.grid(True)`
- `plt.plot()` (4-5 calls)
- `plt.text()` (2 calls)
- `plt.title()`
- `plt.yscale('log')`
- `plt.savefig()`

#### B. Second Plotting Loop (Lines 421-540)

**Purpose:** Generate recent (2-year) trend plots with channels  
**File Pattern:** `0_recent_<SYMBOL>.png`  
**Location:** `web_dir` from `get_webpage_store(json_fn)`

**Conditional:** Only runs if `hourOfDay >= 1 or 11 < hourOfDay < 13`

**Key Steps:**
1. Check plot file age (skip if < 20 hours old)
2. Compute trend channels using `recentTrendAndMidTrendChannelFitWithAndWithoutGap()`
3. Create detailed plot with:
   - Signal overlays (monthly and daily)
   - Despiked prices
   - Percentile channels (if applicable)
   - Upper/lower trend lines (yellow)
   - No-gap trend lines (blue)
   - Date stamp and value text
4. Save to `0_recent_<SYMBOL>.png`

**matplotlib calls:** (many more)
- `plt.figure(10)`
- `plt.clf()`
- `plt.grid(True)`
- `plt.plot()` (10+ calls)
- `plt.ylim()`, `plt.xlim()`
- `plt.text()`
- `plt.title()`
- `plt.tick_params()` (2 calls)
- `plt.savefig()`

#### C. Indirect Plot Generation (Line 575)

**Function:** `sharpeWeightedRank_2D(..., makeQCPlots=True)`

**Side Effect:** Generates quality control plots  
**Cannot be easily separated** without modifying function signature

### 4. External Communications

#### Line 636: `textmessageOutsideTrendChannel(symbols, adjClose)`

**Purpose:** Send text message alerts for stocks breaking trend channels  
**Conditional:** Only if market is currently open  
**Side Effect:** External SMS/notification

---

## Data Flow Analysis

### Inputs (from parameters)
- `symbol_directory` (str)
- `symbol_file` (str)
- `params` (dict) - 20+ parameters
- `json_fn` (str)

### Computation Flow

```
1. Load data (load_quotes_for_analysis)
   ↓
2. Compute gainloss, value, activeCount (lines 47-73)
   ↓
3. Extract parameters (lines 75-95)
   ↓
4. Compute signal2D (lines 207-211)
   ↓
5. Hold signal monthly (lines 214-221)
   ↓
6. [SIDE EFFECT] Write daily backtest (line 229)
   ↓
7. [SIDE EFFECT] Print stats (lines 224-226)
   ↓
8. [SIDE EFFECT] Write uptrending stocks file (lines 267-278)
   ↓
9. [SIDE EFFECT] Generate all plots (lines 350-540)
   ↓
10. Compute monthgainlossweight via sharpeWeightedRank_2D (line 575)
    ↓
11. Compute monthvalue (lines 591-599)
    ↓
12. [SIDE EFFECT] Print results (lines 602-613)
    ↓
13. [SIDE EFFECT] Send text alerts (line 636)
    ↓
14. Return results (line 638)
```

### Outputs (return values)
- `datearray[-1]` (last date)
- `last_symbols_text` (list of symbol strings)
- `last_symbols_weight` (list of floats)
- `last_symbols_price` (list of floats)

---

## Pure Computation Blocks (Can Be Extracted)

### Block 1: Basic Calculations (Lines 47-73)
- `gainloss` computation
- `value` computation
- `BuyHoldFinalValue` computation
- `lastEmptyPriceIndex` computation
- `activeCount` computation

### Block 2: Signal Computation (Lines 207-221)
- `signal2D` computation via `computeSignal2D()`
- Monthly signal holding
- `numberStocks` and `dailyNumberUptrendingStocks`

### Block 3: Weight Computation (Lines 575-580)
- `monthgainlossweight` via `sharpeWeightedRank_2D()`
- (Note: This function has side effects - makeQCPlots=True)

### Block 4: Portfolio Value Computation (Lines 591-599)
- `monthvalue` computation
- `numberSharesCalc` computation

### Block 5: Results Extraction (Lines 606-613)
- Extract symbols, weights, prices for holdings > 0

---

## Side Effect Dependencies

### Plot Generation Dependencies
- Requires: `adjClose`, `symbols`, `datearray`, `signal2D`, `signal2D_daily`, `quotes_despike`
- Optionally: `lowChannel`, `hiChannel` (if percentileChannels method)
- Optionally: `upperTrend`, `lowerTrend`, `NoGapUpperTrend`, `NoGapLowerTrend`

### File Write Dependencies
- Requires: `dailyNumberUptrendingStocks`, `activeCount`, `datearray`
- Requires: `web_dir` from `get_webpage_store(json_fn)`

### Text Message Dependencies
- Requires: `symbols`, `adjClose`
- Requires: market status check

---

## Extraction Strategy Recommendations

### Phase 4b1: Extract Plot Generation (Lowest Risk)

**Create:** `functions/output_generators.py`

**New Function:** `generate_portfolio_plots(plot_data, params, output_dir)`

**Input Structure:**
```python
plot_data = {
    'adjClose': adjClose,
    'symbols': symbols,
    'datearray': datearray,
    'signal2D': signal2D,
    'signal2D_daily': signal2D_daily,
    'lowChannel': lowChannel,  # optional
    'hiChannel': hiChannel,    # optional
    'despike_params': {
        'LongPeriod': LongPeriod,
        'stddevThreshold': stddevThreshold
    },
    'trend_params': {
        'minperiod': params['minperiod'],
        'maxperiod': params['maxperiod'],
        # ... etc
    }
}
```

**Extraction Lines:** 350-420, 421-540

### Phase 4b2: Extract File Writing (Medium Risk)

**New Function:** `write_portfolio_status_files(file_data, output_dir)`

**Input Structure:**
```python
file_data = {
    'dailyNumberUptrendingStocks': dailyNumberUptrendingStocks,
    'activeCount': activeCount,
    'datearray': datearray
}
```

**Extraction Lines:** 267-278

**Note:** `computeDailyBacktest()` file writes are harder to extract - defer or create separate issue

### Phase 4b3: Create Pure Computation Function (Highest Risk)

**New Function:** `compute_portfolio_metrics(adjClose, symbols, datearray, params)`

**Returns:**
```python
{
    'gainloss': gainloss,
    'value': value,
    'BuyHoldFinalValue': BuyHoldFinalValue,
    'signal2D': signal2D,
    'signal2D_daily': signal2D_daily,
    'lowChannel': lowChannel,  # if applicable
    'hiChannel': hiChannel,    # if applicable
    'monthgainlossweight': monthgainlossweight,
    'monthvalue': monthvalue,
    'numberSharesCalc': numberSharesCalc,
    'dailyNumberUptrendingStocks': dailyNumberUptrendingStocks,
    'activeCount': activeCount,
    'last_symbols_text': last_symbols_text,
    'last_symbols_weight': last_symbols_weight,
    'last_symbols_price': last_symbols_price
}
```

**Challenges:**
1. `sharpeWeightedRank_2D()` has side effect (makeQCPlots=True) - need to handle
2. Print statements scattered throughout - need to remove or log
3. Many intermediate variables - need to track dependencies carefully

### Phase 4b4: Orchestration Refactor

**New Structure:**
```python
def PortfolioPerformanceCalcs(symbol_directory, symbol_file, params, json_fn):
    # 1. Load data
    adjClose, symbols, datearray = load_quotes_for_analysis(...)
    
    # 2. Compute (pure)
    results = compute_portfolio_metrics(adjClose, symbols, datearray, params)
    
    # 3. Write daily backtest (still has side effects)
    computeDailyBacktest(...)
    
    # 4. Write status files
    write_portfolio_status_files(...)
    
    # 5. Generate plots
    if should_generate_plots():  # time check
        generate_portfolio_plots(results, params, output_dir)
    
    # 6. Print results
    print_portfolio_results(results)
    
    # 7. Send alerts
    if market_open():
        send_alerts(...)
    
    # 8. Return
    return results['last_date'], results['symbols'], results['weights'], results['prices']
```

---

## Risk Assessment

### High Risk Areas
1. **Line 575:** `sharpeWeightedRank_2D()` call with `makeQCPlots=True` - side effect inside computation
2. **Lines 350-540:** Plot generation loops are complex with many edge cases
3. **Floating point precision:** Many numpy operations - must preserve exact semantics

### Medium Risk Areas
1. **File writes:** Simple but critical for downstream processes
2. **Print statements:** Mixed user feedback and debugging - need to categorize
3. **Return value structure:** Must maintain backward compatibility

### Low Risk Areas
1. **Parameter extraction:** (lines 75-95) - straightforward
2. **Basic computations:** (lines 47-73) - deterministic numpy operations

---

## Testing Strategy

### Shadow Tests Required

1. **Test 1: Computation Results**
   - Compare all intermediate arrays (gainloss, value, signal2D, monthvalue, etc.)
   - Use `np.allclose()` with appropriate tolerance

2. **Test 2: File Output**
   - Compare file checksums for `pyTAAAweb_numberUptrendingStocks_status.params`
   - Line-by-line comparison

3. **Test 3: Return Values**
   - Exact match for symbols, weights, prices
   - Exact match for last date

4. **Test 4: Plot Files**
   - Check file existence and modification times
   - Visual inspection or image comparison (optional)

### Property Tests

1. **Invariant 1:** Sum of weights equals 1.0 (or 0 if no positions)
2. **Invariant 2:** All weights >= 0
3. **Invariant 3:** Number of positions <= `numberStocksTraded`
4. **Invariant 4:** Portfolio value is monotonically constructed (no jumps)

---

## Open Questions

1. **Q1:** Should `sharpeWeightedRank_2D(makeQCPlots=True)` be changed to False in pure function?
   - **Answer:** Yes - generate plots separately in output phase

2. **Q2:** How to handle print statements in pure function?
   - **Answer:** Remove from pure function, move to orchestrator

3. **Q3:** Should `computeDailyBacktest()` be refactored in Phase 4b?
   - **Answer:** No - defer to separate phase (too risky)

4. **Q4:** Return dict or named tuple?
   - **Answer:** Dict is more flexible for incremental development

5. **Q5:** How to test plot generation without visual inspection?
   - **Answer:** Check file existence, size, modification time - sufficient for refactoring validation

---

## Next Steps

1. **Create shadow test infrastructure** (before any extraction)
2. **Start with Phase 4b1** (plot extraction - lowest risk)
3. **Validate after each sub-phase** (don't batch)
4. **Document surprises** as they arise

**Status:** Analysis complete, ready to begin Phase 4b1
