# Recommend Model Refactoring Plan

**Date:** January 19, 2026  
**Objective:** Refactor `recommend_model.py` by extracting functionality into reusable modules while ensuring identical output

---

## Overview

### Current State
- `recommend_model.py` is a 621-line monolithic script
- Contains model recommendation logic, parameter display, plotting, and CLI interface
- Reads backtested portfolio data from multiple model `.params` files
- Generates recommendation plot and console output

### Target State
- Extract core functionality into two new modules:
  - `functions/abacus_backtest.py` - Backtest data generation and management
  - `functions/abacus_recommend.py` - Recommendation logic and analysis
- Keep `recommend_model.py` as thin CLI entry point
- Maintain 100% output compatibility (plots, files, console output)

### Success Criteria
- ✅ Refactored code produces byte-identical recommendation_plot.png
- ✅ Console output matches baseline exactly
- ✅ Can generate pyTAAAweb_backtestPortfolioValue.params (3+ columns)
- ✅ All tests pass
- ✅ No regressions in functionality

---

## Phase 1: Baseline Establishment

### 1.1 Run Current Version and Capture Baseline
**Goal:** Establish reference outputs for comparison

**Steps:**
- [ ] Create baseline directory: `.github/refactoring_baseline/`
- [ ] Run current recommend_model.py with test JSON:
  ```bash
  cd /Users/donaldpg/PyProjects/worktree2/PyTAAA
  uv run python recommend_model.py \
    --json /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json \
    > .github/refactoring_baseline/console_output.txt 2>&1
  ```
- [ ] Copy outputs to baseline directory:
  - `recommendation_plot.png` → `.github/refactoring_baseline/recommendation_plot_baseline.png`
  - Console output (already captured above)
  - Log file if generated
  - Model-Switching Portfolio dates and values (heavy black line in upper subplot)
- [ ] Document runtime parameters in baseline:
  - Lookback periods used
  - Date tested
  - Models included
  - Data format (backtested)

**Validation:**
- [ ] Baseline files exist and are non-empty
- [ ] recommendation_plot.png opens and displays correctly
- [ ] Console output shows expected recommendations

---

## Phase 2: Code Analysis and Architecture Design

### 2.1 Analyze Current Code Structure
**Map existing functionality:**

**Current `recommend_model.py` contains:**
1. **Utility Functions** (Lines 26-135)
   - `get_first_weekday_of_month()` - Date helper
   - `get_recommendation_dates()` - Date processing
   - `load_best_params_from_saved_state()` - State management
   - `get_recommendation_lookbacks()` - Parameter resolution

2. **Recommendation Engine** (Lines 138-285)
   - `generate_recommendation_output()` - Core recommendation logic
   - Model selection and ranking
   - Normalized score calculation

3. **Display Functions** (Lines 288-350)
   - `display_parameters_info()` - Parameter summary output

4. **Main CLI Handler** (Lines 353-621)
   - Configuration loading
   - Model path resolution
   - MonteCarloBacktest initialization
   - Plot generation
   - Error handling

### 2.2 Design New Module Structure

**Target Architecture:**

```
functions/
├── abacus_backtest.py       # Backtest data management
│   ├── BacktestDataLoader
│   │   ├── load_model_data()
│   │   ├── validate_data_files()
│   │   └── get_common_date_range()
│   ├── BacktestPortfolioGenerator
│   │   ├── generate_portfolio_values()
│   │   ├── calculate_model_switching_portfolio()
│   │   └── write_params_file()
│   └── BacktestConfig
│       └── from_json()
│
├── abacus_recommend.py       # Recommendation engine
│   ├── ModelRecommender
│   │   ├── __init__(monte_carlo, lookbacks)
│   │   ├── get_recommendation_for_date()
│   │   ├── rank_models_at_date()
│   │   └── generate_recommendation_text()
│   ├── RecommendationDisplay
│   │   ├── display_parameters_summary()
│   │   ├── display_model_rankings()
│   │   └── create_recommendation_plot()  # Wrapper for MonteCarloBacktest.create_monte_carlo_plot()
│   └── DateHelper
│       ├── get_first_weekday_of_month()
│       ├── get_recommendation_dates()
│       └── find_closest_trading_date()
│
└── (existing modules remain)
    ├── MonteCarloBacktest.py  # Contains create_monte_carlo_plot() - used by RecommendationDisplay
    ├── GetParams.py
    ├── PortfolioMetrics.py
    └── ... (other existing modules unchanged)
```

**recommend_model.py becomes:**
- CLI argument parsing (click)
- High-level orchestration
- Calls functions from abacus_recommend.py
- Minimal business logic (< 150 lines)

---

## Phase 3: Incremental Refactoring

### 3.1 Extract Date Utilities
**Target Module:** `functions/abacus_recommend.py`

**Steps:**
- [ ] Create `functions/abacus_recommend.py`
- [ ] Add module docstring and imports
- [ ] Create `DateHelper` class:
  - [ ] Move `get_first_weekday_of_month()` → `DateHelper.get_first_weekday()`
  - [ ] Move `get_recommendation_dates()` → `DateHelper.get_recommendation_dates()`
  - [ ] Add `find_closest_trading_date()` helper
- [ ] Add unit tests in `tests/test_abacus_recommend.py`
- [ ] Update `recommend_model.py` imports
- [ ] Run and verify: output matches baseline

**Testing:**
```python
# tests/test_abacus_recommend.py
from functions.abacus_recommend import DateHelper
from datetime import date

def test_get_first_weekday():
    # Jan 1, 2026 is Wednesday
    result = DateHelper.get_first_weekday(date(2026, 1, 15))
    assert result == date(2026, 1, 1)  # First weekday of Jan 2026
```

### 3.2 Extract Recommendation Engine
**Target Module:** `functions/abacus_recommend.py`

**Steps:**
- [ ] Create `ModelRecommender` class in `abacus_recommend.py`
- [ ] Move recommendation logic from `generate_recommendation_output()`:
  - [ ] `__init__(monte_carlo, lookbacks, config)`
  - [ ] `get_recommendation_for_date(target_date)` → returns (best_model, rankings)
  - [ ] `rank_models_at_date(date_idx)` → returns sorted [(model, score), ...]
  - [ ] `generate_recommendation_text(dates, target_date, first_weekday)` → returns plot_text
- [ ] Keep normalized score calculation in ModelRecommender
- [ ] Update `recommend_model.py` to use ModelRecommender
- [ ] Run and verify: output matches baseline

**API Design:**
```python
recommender = ModelRecommender(monte_carlo, lookbacks=[55, 157, 174])
best_model, rankings = recommender.get_recommendation_for_date(date(2026, 1, 19))
plot_text = recommender.generate_recommendation_text([date(2026, 1, 19)], ...)
```

### 3.3 Extract Display Functions
**Target Module:** `functions/abacus_recommend.py`

**Steps:**
- [ ] Create `RecommendationDisplay` class
- [ ] Move `display_parameters_info()` → `RecommendationDisplay.display_parameters_summary()`
- [ ] Extract plot generation logic → `RecommendationDisplay.create_recommendation_plot()`
- [ ] Update `recommend_model.py` to use RecommendationDisplay
- [ ] Run and verify: plot matches baseline pixel-for-pixel

**Validation:**
```bash
# Compare plots byte-for-byte
diff .github/refactoring_baseline/recommendation_plot_baseline.png \
     /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pyTAAA_web/recommendation_plot.png
# Should show no differences
```

### 3.4 Extract Backtest Data Management
**Target Module:** `functions/abacus_backtest.py`

**Steps:**
- [ ] Create `functions/abacus_backtest.py`
- [ ] Create `BacktestDataLoader` class:
  - [ ] `load_model_data(model_paths)` → dict of DataFrames
  - [ ] `validate_data_files(model_paths)` → checks existence
  - [ ] `get_common_date_range(data_dict)` → intersection of dates
- [ ] Create `BacktestPortfolioGenerator` class:
  - [ ] `__init__(config, model_data)`
  - [ ] `calculate_model_switching_portfolio(lookbacks)` → portfolio values array
  - [ ] `write_params_file(filepath, data, columns=['date', 'buyhold', 'traded'])`
- [ ] Add comprehensive docstrings following PEP 257
- [ ] Run and verify: can generate pyTAAAweb_backtestPortfolioValue.params

**API Design:**
```python
from functions.abacus_backtest import BacktestDataLoader, BacktestPortfolioGenerator

loader = BacktestDataLoader()
model_data = loader.load_model_data(model_paths)

generator = BacktestPortfolioGenerator(config, model_data)
portfolio = generator.calculate_model_switching_portfolio([55, 157, 174])
generator.write_params_file(output_path, portfolio, columns=['date', 'buyhold', 'traded'])
```

### 3.5 Simplify Main Entry Point
**Target:** `recommend_model.py`

**Steps:**
- [ ] Remove all extracted functions (now in modules)
- [ ] Keep only:
  - Click decorators and CLI parsing
  - Configuration loading logic
  - High-level orchestration calls
  - Error handling and logging
- [ ] Import from new modules:
  ```python
  from functions.abacus_recommend import ModelRecommender, RecommendationDisplay, DateHelper
  from functions.abacus_backtest import BacktestDataLoader, BacktestPortfolioGenerator
  ```
- [ ] Target: < 150 lines in `main()`
- [ ] Run and verify: complete end-to-end test

---

## Phase 4: Testing and Validation

### 4.1 Unit Tests
**Create test files:**

- [ ] `tests/test_abacus_recommend.py`
  - [ ] Test DateHelper.get_first_weekday() for various months
  - [ ] Test DateHelper.get_recommendation_dates() with edge cases
  - [ ] Test ModelRecommender with mock MonteCarloBacktest
  - [ ] Test ranking algorithm with known portfolios
  
- [ ] `tests/test_abacus_backtest.py`
  - [ ] Test BacktestDataLoader with sample .params files
  - [ ] Test date range intersection logic
  - [ ] Test portfolio generation with mock data
  - [ ] Test params file writing format (3-column vs 5-column)

**Run tests:**
```bash
cd /Users/donaldpg/PyProjects/worktree2/PyTAAA
uv run pytest tests/test_abacus_recommend.py -v
uv run pytest tests/test_abacus_backtest.py -v
```

### 4.2 Integration Tests
- [ ] Run refactored recommend_model.py with same parameters as baseline
- [ ] Capture new output:
  ```bash
  uv run python recommend_model.py \
    --json /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json \
    > .github/refactoring_baseline/console_output_refactored.txt 2>&1
  ```
- [ ] Compare console outputs:
  ```bash
  diff .github/refactoring_baseline/console_output.txt \
       .github/refactoring_baseline/console_output_refactored.txt
  ```
- [ ] Compare plots (should be byte-identical or visually identical):
  ```bash
  # Visual comparison
  open .github/refactoring_baseline/recommendation_plot_baseline.png
  open /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pyTAAA_web/recommendation_plot.png
  ```

### 4.3 Regression Testing
**Test different scenarios:**

- [ ] Run with `--lookbacks use-saved` option
- [ ] Run with custom lookbacks: `--lookbacks 50,100,200`
- [ ] Run with specific date: `--date 2025-12-15`
- [ ] Run without JSON config (legacy mode)
- [ ] Test error handling: invalid JSON path, missing data files

**Validation checklist for each scenario:**
- [ ] No Python exceptions raised
- [ ] Recommendation output generated
- [ ] Plot created successfully
- [ ] Console output is reasonable

---

## Phase 5: Documentation and Cleanup

### 5.1 Code Documentation
- [ ] Add comprehensive module docstrings to:
  - `functions/abacus_recommend.py`
  - `functions/abacus_backtest.py`
- [ ] Ensure all functions have PEP 257-compliant docstrings
- [ ] Add type hints to all function signatures
- [ ] Add inline comments for complex logic (complete sentences, 72 chars)

### 5.2 Update Project Documentation
- [ ] Create `docs/RECOMMENDATION_SYSTEM.md`:
  - Architecture overview
  - Module responsibilities
  - Usage examples
  - API reference
- [ ] Update `README.md` if needed
- [ ] Add docstring examples showing typical usage

### 5.3 Commit Strategy
**Commit incrementally after each validated phase:**

```bash
# After Phase 3.1
git add functions/abacus_recommend.py tests/test_abacus_recommend.py
git commit -m "refactor: Extract date utilities to abacus_recommend module"

# After Phase 3.2
git add functions/abacus_recommend.py recommend_model.py
git commit -m "refactor: Extract ModelRecommender class for recommendation logic"

# After Phase 3.3
git add functions/abacus_recommend.py recommend_model.py
git commit -m "refactor: Extract RecommendationDisplay for output formatting"

# After Phase 3.4
git add functions/abacus_backtest.py
git commit -m "feat: Add BacktestDataLoader and BacktestPortfolioGenerator modules"

# After Phase 3.5
git add recommend_model.py
git commit -m "refactor: Simplify recommend_model.py entry point"

# After Phase 4 (all tests passing)
git add tests/ .github/refactoring_baseline/
git commit -m "test: Add comprehensive test suite for refactored modules"

# After Phase 5
git add docs/
git commit -m "docs: Add RECOMMENDATION_SYSTEM architecture documentation"
```

---

## Phase 6: Backtest File Generation Feature

### 6.1 Add Backtest Generation Script
**Goal:** Standalone script to generate pyTAAAweb_backtestPortfolioValue.params

**Steps:**
- [ ] Create `generate_abacus_backtest.py` entry point
- [ ] Use BacktestPortfolioGenerator from abacus_backtest.py
- [ ] Accept CLI arguments:
  - `--json` - path to configuration JSON
  - `--output` - output file path (default: from JSON config)
  - `--columns` - number of columns (3 or 5)
  - `--format` - data format ('actual' or 'backtested')
- [ ] Generate file with proper format:
  - Column 1: Date (YYYY-MM-DD)
  - Column 2: Buy-and-hold value
  - Column 3: Traded system value
  - Optional Column 4: New highs count
  - Optional Column 5: New lows count

**Example usage:**
```bash
uv run python generate_abacus_backtest.py \
  --json /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json \
  --output /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/data_store/pyTAAAweb_backtestPortfolioValue.params \
  --columns 3
```

### 6.2 Validate Backtest Output
- [ ] Generate 3-column params file
- [ ] Verify format matches existing files:
  ```bash
  head -10 /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/data_store/pyTAAAweb_backtestPortfolioValue.params
  ```
- [ ] Check date range starts correctly (after 500 index skip)
- [ ] Verify values are reasonable (positive, increasing trend)
- [ ] Test that recommend_model.py can read generated file

---

## Rollback Plan

**If critical issues arise:**

1. **Immediate Rollback**
   ```bash
   git checkout HEAD~1 recommend_model.py
   git checkout HEAD~1 functions/abacus_*.py
   ```

2. **Preserve Baseline**
   - Keep `.github/refactoring_baseline/` directory intact
   - Use for comparison debugging

3. **Incremental Recovery**
   - Identify last working commit from Phase checkpoints
   - Cherry-pick successful changes
   - Skip problematic refactoring step

---

## Risk Mitigation

### Potential Risks

1. **Floating-point differences** in calculations
   - **Mitigation:** Use `np.allclose()` instead of exact equality
   - Accept differences < 1e-10

2. **Plot rendering variations** (timestamps, font rendering)
   - **Mitigation:** Compare structural elements, not pixels
   - Verify plot data series are identical

3. **Date/time dependencies** in output
   - **Mitigation:** Use fixed date for testing (not "today")
   - Pass explicit `--date` parameter

4. **Import cycles** between new modules
   - **Mitigation:** Keep dependency direction one-way
   - abacus_backtest.py → no dependencies on abacus_recommend.py
   - abacus_recommend.py → can import from abacus_backtest.py

5. **Breaking existing code** that imports recommend_model functions
   - **Mitigation:** Check for external imports with grep:
     ```bash
     grep -r "from recommend_model import" .
     ```

---

## Success Metrics

### Quantitative
- [ ] 0 linting errors in new modules
- [ ] 100% test coverage for new modules
- [ ] Console output diff < 5 lines (timestamps only)
- [ ] Plot visual comparison: identical
- [ ] Performance: refactored version within 10% of baseline runtime

### Qualitative
- [ ] Code is more maintainable (shorter functions, clear responsibilities)
- [ ] Modules are reusable in other contexts
- [ ] Error messages are clearer
- [ ] Easier to add new recommendation strategies

---

## Timeline Estimate

- **Phase 1:** 30 minutes (baseline establishment)
- **Phase 2:** 1 hour (analysis and design)
- **Phase 3:** 4-6 hours (incremental refactoring)
  - 3.1: 1 hour
  - 3.2: 1.5 hours
  - 3.3: 1 hour
  - 3.4: 1.5 hours
  - 3.5: 1 hour
- **Phase 4:** 2-3 hours (testing)
- **Phase 5:** 1-2 hours (documentation)
- **Phase 6:** 2 hours (backtest generation)

**Total estimated time:** 11-15 hours

---

## Checklist Summary

### Pre-Refactoring
- [ ] Baseline outputs captured
- [ ] Architecture designed and reviewed
- [ ] Test strategy defined

### During Refactoring
- [ ] Each phase validated before proceeding
- [ ] Commits made after each successful phase
- [ ] Tests pass continuously

### Post-Refactoring
- [ ] All integration tests pass
- [ ] Documentation complete
- [ ] Baseline comparison shows identical output
- [ ] Code review completed
- [ ] Merged to main branch

---

## Notes

- This refactoring prioritizes **safety over speed**
- Each step includes validation against baseline
- Rollback plan available at every phase
- Focus on incremental, testable changes
- Maintain backward compatibility throughout

---

**Document Status:** Draft  
**Last Updated:** January 19, 2026  
**Author:** Copilot Session  
**Approval Required:** Yes (review before starting Phase 3)
