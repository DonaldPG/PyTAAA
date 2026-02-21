# PyTAAA Codebase Analysis and Refactoring Proposal

## Executive Summary

This document provides a comprehensive analysis of the refactoring performed in the PyTAAA.master codebase and proposes a refactoring plan for the current codebase (worktree2). The analysis covers:

1. The refactoring journey in PyTAAA.master
2. Key differences in stock selection and weighting strategies between the two codebases
3. A detailed proposal for refactoring the current codebase

---

## Part 1: PyTAAA.master Refactoring Journey

### 1.1 Overview of Changes

The refactoring in PyTAAA.master was performed in December 2005 (note: likely 2025 based on context) on branch `feature/add-highs-lows-timeseries`. The key changes included:

#### Entry Point Changes
- **Old Entry Point:** `PyTAAA_backtest_sp500_pine_refactored.py` (~107KB, 3100+ lines)
- **New Entry Point:** `pytaaa_backtest_montecarlo.py` (~800 bytes, 21 lines)

The new entry point is dramatically simpler, delegating all logic to modular components.

#### New Modular Structure
```
src/backtest/
├── __init__.py           # Package exports
├── config.py             # Configuration classes and constants
├── montecarlo.py         # Monte Carlo simulation functions
├── montecarlo_runner.py  # Monte Carlo trial orchestration
├── plotting.py           # Visualization utilities
├── dailyBacktest_pctLong.py  # Core backtest logic (~190KB)
├── functions/
│   ├── GetParams.py      # Parameter handling
│   ├── TAfunctions.py    # Technical analysis functions
│   └── UpdateSymbols_inHDF5.py  # Data loading
└── analyze/
    └── analyze_backtest_results.py
```

### 1.2 Key Refactoring Details

#### Configuration Module (`src/backtest/config.py`)
Created centralized configuration with three main classes:

1. **`TradingConstants`**
   - Trading day constants (252 days/year)
   - Period constants (1Y, 2Y, 3Y, 5Y, 10Y, 15Y)
   - Initial portfolio value (10,000)
   - CAGR validation bounds

2. **`BacktestConfig`**
   - Monte Carlo settings (DEFAULT_RANDOM_TRIALS = 250)
   - Default parameters for pyTAAA and Linux editions
   - Parameter exploration ranges for optimization
   - Symbol file configurations

3. **`FilePathConfig`**
   - Centralized file path management
   - Methods for generating output filenames

#### Monte Carlo Module (`src/backtest/montecarlo.py`)
- `random_triangle()` - Random parameter generation using triangular distribution
- `create_temporary_json()` - Creates temp config for each trial
- `calculate_sharpe_ratio()` - Sharpe ratio calculation
- `calculate_period_metrics()` - Metrics for multiple time periods
- `calculate_drawdown_metrics()` - Drawdown analysis
- `beat_buy_hold_test()` - Strategy vs buy-hold comparison

#### Monte Carlo Runner (`src/backtest/montecarlo_runner.py`)
- Orchestrates multiple Monte Carlo trials
- Generates random parameters using `MonteCarloBacktest` class
- Outputs results to CSV with extensive metrics
- Includes progress tracking with tqdm

#### Plotting Module (`src/backtest/plotting.py`)
- `BacktestPlotter` class for creating visualizations
- Helper functions for log-scale calculations
- Signal diagnostic plotting
- Lower panel plotting

### 1.3 Agentic Documentation

The `.agentic-docs` folder contains two key documents:

1. **`monte-carlo-backtest-entry-point.md`** - Product plan for the CLI entry point
2. **`product/backtest-timeseries-capture-plan.md`** - Implementation plan for capturing new highs/lows time series

The time series capture plan outlines adding two new columns to the backtest output:
- Column 4: New Highs Count
- Column 5: New Lows Count

---

## Part 2: Comparison of Codebases

### 2.1 Structural Differences

| Aspect | PyTAAA.master (Refactored) | Current (worktree2) |
|--------|---------------------------|---------------------|
| Entry Point | `pytaaa_backtest_montecarlo.py` (21 lines) | `PyTAAA_backtest_sp500_pine_refactored.py` (~140KB) |
| Configuration | Centralized in `src/backtest/config.py` | Local `TradingConstants` class (lines 87+) |
| Monte Carlo | Separate runner with CSV output | Integrated in main file |
| Package Structure | `src/backtest/` with submodules | Flat `functions/` directory |
| Data Loading | `src/backtest/functions/UpdateSymbols_inHDF5.py` | `functions/UpdateSymbols_inHDF5.py` |

### 2.2 Stock Selection Strategy Comparison

Both codebases use similar core strategies but with important differences:

#### Common Elements
1. **Signal Generation:** Both use `computeSignal2D` for identifying uptrending stocks
2. **Ranking:** Both use `sharpeWeightedRank_2D` for stock ranking
3. **Weight Calculation:** Both compute portfolio weights based on Sharpe ratios

#### Key Differences

**In PyTAAA.master:**
- Uses `percentileChannels` as the primary uptrend signal method
- Parameter ranges defined in `BacktestConfig`:
  - `LONG_PERIOD_RANGE`: (55, 280)
  - `LOW_PCT_RANGE`: (10.0, 30.0)
  - `HI_PCT_RANGE`: (70.0, 90.0)
- Exploration parameters for Monte Carlo:
  - `EXPLORATION_LONG_PERIOD`: (190, 370, 550)
  - `EXPLORATION_STDDEV_THRESHOLD`: (5.0, 7.50, 10.0)

**In Current Codebase (worktree2):**
- More complex `sharpeWeightedRank_2D` implementation (~2250 lines)
- Additional weight constraint parameters:
  - `max_weight_factor`: 2.0-5.0 (triangular distribution)
  - `min_weight_factor`: 0.1-0.5
  - `absolute_max_weight`: 0.7-1.0
- Enhanced ranking with deltaRank methodology
- Includes `UnWeightedRank_2D` alternative

### 2.3 Weighting Strategy Differences

#### Current Codebase (worktree2) - More Advanced
```python
# Weight constraint parameters (lines 627-632)
max_weight_factor = random.triangular(2.0, 3.0, 5.0)
min_weight_factor = random.triangular(0.1, 0.3, 0.5)
absolute_max_weight = random.triangular(0.7, 0.9, 1.0)
apply_constraints = True
```

**Key Features:**
1. **Dynamic Weight Constraints:**
   - `max_weight_factor`: Maximum weight as multiple of equal weight (default 3.0)
   - `min_weight_factor`: Minimum weight as fraction of equal weight (default 0.3)
   - `absolute_max_weight`: Hard cap on any single position (default 0.9)

2. **Risk-Adjusted Weighting:**
   - Uses `1./riskDownside` for risk measure
   - Modifies weights with inverse risk
   - Scales to sum to 1.0

3. **Early Period Handling:**
   - Allocates 100% to CASH for early period (2000-2002)
   - Fallback to equal weights when no eligible stocks

#### PyTAAA.master - Simpler Approach
- Uses fixed parameter ranges from `BacktestConfig`
- Triangular distribution for exploration
- Fewer weight constraint options

### 2.4 Feature Differences

| Feature | PyTAAA.master | Current (worktree2) |
|---------|---------------|---------------------|
| CLI Entry Point | ✅ Click-based | ❌ Script-based |
| CSV Output | ✅ Comprehensive metrics | ⚠️ Limited |
| Weight Constraints | ⚠️ Basic | ✅ Advanced |
| Model Switching | ❌ Not in backtest | ✅ In MonteCarloBacktest |
| Numba Optimization | ❌ Not used | ✅ For performance |
| Focus Period Blending | ❌ Not available | ✅ Configurable |

---

## Part 3: Refactoring Proposal for Current Codebase

### 3.1 Goals

1. **Simplify Entry Point:** Create a CLI entry point similar to `pytaaa_backtest_montecarlo.py`
2. **Modularize Code:** Organize backtest-specific code into `functions/backtest/`
3. **Preserve Functionality:** Keep existing data loading and analysis functions
4. **Blend Best Features:** Incorporate improvements from both codebases
5. **Avoid Duplication:** Don't create multiple versions of the same functionality

### 3.2 Proposed Structure

```
functions/
├── ... (existing files unchanged)
├── backtest/
│   ├── __init__.py           # Package exports
│   ├── config.py             # Configuration (adapted from master)
│   ├── montecarlo.py         # Monte Carlo functions
│   ├── montecarlo_runner.py  # Trial orchestration
│   ├── plotting.py           # Visualization utilities
│   └── entrypoint.py         # CLI entry point
```

### 3.3 Phase-by-Phase Plan

**Assumptions:**
- Single human developer using AI code assistant for implementation
- Human reviews and approves all code changes
- AI writes ~90% of code, human writes ~10% (review, approvals, minor edits)
- Access to advanced AI models (Claude Sonnet 4.5, etc.) for faster implementation

#### Phase 1: Create Backtest Module Infrastructure
**Duration:** 1-2 days
**AI Effort:** ~4-6 hours
**Human Effort:** ~1-2 hours (review)

**Context & Resources:**

**Reference Files (for patterns to copy):**
1. `/Users/donaldpg/PyProjects/PyTAAA.master/src/backtest/config.py` - Full implementation of config classes
2. `/Users/donaldpg/PyProjects/PyTAAA.master/src/backtest/montecarlo.py` - Monte Carlo functions
3. `/Users/donaldpg/PyProjects/worktree2/PyTAAA/PyTAAA_backtest_sp500_pine_refactored.py` lines 87-200 - Local TradingConstants

**Code Patterns to Use:**
- Use dataclasses for configuration classes
- Follow the exact class structure from PyTAAA.master/src/backtest/config.py
- Import existing constants from current codebase where possible

**Data Flow:**
```
JSON Config → BacktestConfig → Monte Carlo Functions → CSV Output
     ↓
FilePathConfig → Output file paths
     ↓
TradingConstants → Performance calculations
```

**Algorithms:**
1. **random_triangle(low, mid, high, size)**: 
   - Generates random values using averaged uniform and triangular distributions
   - Formula: (uniform + triangular) / 2
   
2. **calculate_sharpe_ratio(daily_gains, trading_days=252)**:
   - Input: Array of daily gain ratios
   - Output: Annualized Sharpe ratio
   - Formula: (mean(gains) - 1) / std(gains) * sqrt(trading_days)

3. **calculate_period_metrics(portfolio_value, trading_days=252)**:
   - Computes metrics for each period: 20Y, 15Y, 10Y, 5Y, 3Y, 2Y, 1Y
   - Returns: sharpe, return, cagr, days for each period

4. **calculate_drawdown_metrics(portfolio_value)**:
   - Computes average drawdown for each time period

**Goals:**
1. Create `functions/backtest/__init__.py` that exports all public API
2. Create `functions/backtest/config.py` with three classes matching PyTAAA.master
3. Create `functions/backtest/montecarlo.py` with utility functions
4. All imports should work without errors
5. Basic unit tests should pass

**Success Criteria (Measurable):**
- [ ] `from functions.backtest import TradingConstants` works (imported from worktree2)
- [ ] `from functions.backtest import BacktestConfig` works (copied from PyTAAA.master)
- [ ] `from functions.backtest import FilePathConfig` works (copied from PyTAAA.master)
- [ ] `TradingConstants.TRADING_DAYS_PER_YEAR == 252`
- [ ] `BacktestConfig.DEFAULT_RANDOM_TRIALS == 250`
- [ ] `random_triangle(1, 2, 3, 100).shape == (100,)`
- [ ] All new files pass `python -m py_compile functions/backtest/*.py`
- [ ] All tests pass: `pytest tests/test_phase1_*.py -v`

**Checklist:**
- [ ] Create `functions/backtest/` directory
- [ ] Create `functions/backtest/__init__.py` with package exports
- [ ] Import TradingConstants from worktree2 (PyTAAA_backtest_sp500_pine_refactored)
- [ ] Create `functions/backtest/config.py`:
  - [ ] Import TradingConstants from existing location
  - [ ] Copy BacktestConfig from PyTAAA.master
  - [ ] Copy FilePathConfig from PyTAAA.master
- [ ] Create `functions/backtest/montecarlo.py`:
  - [ ] Copy random_triangle() from PyTAAA.master
  - [ ] Add create_temporary_json() function
  - [ ] Add cleanup_temporary_json() function
  - [ ] Copy calculate_sharpe_ratio() from PyTAAA.master
  - [ ] Copy calculate_period_metrics() from PyTAAA.master
  - [ ] Copy calculate_drawdown_metrics() from PyTAAA.master
- [ ] Create tests/test_phase1_config.py
- [ ] Create tests/test_phase1_montecarlo.py (including file I/O test)
- [ ] Run tests and verify all pass

#### Phase 2: Create CLI Entry Point
**Duration:** 1 day
**AI Effort:** ~2-3 hours
**Human Effort:** ~30 minutes (review)

**Context & Resources:**

**Reference Files:**
1. `/Users/donaldpg/PyProjects/PyTAAA.master/pytaaa_backtest_montecarlo.py` - Reference CLI implementation
2. `/Users/donaldpg/PyProjects/PyTAAA.master/src/backtest/montecarlo_runner.py` - Runner logic

**Code Patterns:**
- Use Click library for CLI (already available in project)
- Follow exact structure from PyTAAA.master entry point
- Delegate to existing backtest logic

**Data Flow:**
```
CLI Arguments → run_montecarlo() → dailyBacktest_pctLong() → CSV/PNG Output
```

**Goals:**
1. Create working CLI entry point
2. Integrate with existing backtest functions
3. Generate valid CSV output

**Success Criteria:**
- [ ] `python -m functions.backtest.entrance --help` shows help text
- [ ] `python -m functions.backtest.entrance --config test.json --n_trials 2` runs without error
- [ ] Output CSV file is created with expected columns
- [ ] Click is properly installed as dependency

**Tests to Write:**

```python
# tests/test_phase2_entrypoint.py

import pytest
from click.testing import CliRunner
from functions.backtest.entrance import main

class TestCLIEntryPoint:
    """Test CLI entry point."""
    
    def test_help_text(self):
        """Verify --help shows help text."""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert '--n-trials' in result.output
        assert '--config' in result.output
        assert '--output-csv' in result.output
    
    def test_missing_config(self):
        """Verify error when config is missing."""
        runner = CliRunner()
        result = runner.invoke(main, ['--n_trials', '2'])
        assert result.exit_code != 0
        assert 'Error' in result.output or 'Required' in result.output
```

---

### PHASE 2: Create CLI Entry Point

#### Stage 1: Planning (Software & Data Architecture)

**Architecture Considerations:**
- Use Click library (industry standard for Python CLIs)
- Follow Unix philosophy: do one thing well
- Support both interactive and scripted usage
- Maintainability through separation of CLI from logic

**Data Flow Design:**
```
CLI Arguments → Validation → Runner Function → Existing Backtest → Output
```

**Key Design Decisions:**
1. Use Click decorators for argument parsing
2. Separate runner function from CLI entry point
3. Allow configuration via JSON file (not just CLI args)
4. Support both human-readable progress and machine-readable output

**Persona Notes:**
> "The CLI should be thin - it just parses arguments and delegates to the runner. This keeps the business logic testable and reusable from other code."

---

#### Stage 2: Plan Revision (Constructive Critic)

**Potential Issues to Address:**
- What if JSON config file doesn't exist?
- How to handle partial CLI arguments vs full config?
- Progress output vs quiet mode for automation
- Exit codes for scripting integration

**Edge Cases:**
- Invalid JSON in config file
- Missing required arguments
- File permission errors
- Very large n_trials values

**Improvements Suggested:**
1. Add `--quiet` flag for automation
2. Add `--dry-run` to validate without executing
3. Use exit codes: 0=success, 1=error, 2=validation failure
4. Add version info

**Critique Summary:**
> "Good separation of concerns but missing error handling strategy. Add specific exit codes and validation before expensive operations."

---

#### Stage 3: Implementation (Mid-Level Developer)

**Implementation Steps:**
1. Install Click if not present (check requirements.txt)
2. Create entrance.py with @click.command()
3. Add options: --n_trials, --config, --output_csv, --plot_individual
4. Create run_montecarlo() function that orchestrates
5. Integrate with existing dailyBacktest_pctLong
6. Add error handling with meaningful messages
7. Write CLI tests

**Code Style:**
- Use Click's type validation
- Add help text to all options
- Keep main() under 20 lines
- Extract business logic to separate function

**Developer Notes:**
> "I'll follow the pattern from PyTAAA.master but add better error handling. The runner function will be reusable from other entry points."

---

#### Stage 4: Code Cleanup (Agentic AI Critic)

**Review Checklist:**
- [ ] Help text is clear and complete
- [ ] Error messages guide user to solution
- [ ] Exit codes are appropriate
- [ ] No hardcoded paths in CLI
- [ ] Tests cover both happy path and errors

**Suggested Refactors:**
1. Extract argument validation to separate function
2. Add color to error messages (using click.secho)
3. Consider adding shell completion

**AI Critic Summary:**
> "Solid implementation. Consider adding a context object for sharing state between CLI and runner."

#### Phase 3: Integrate Advanced Weight Constraints
**Duration:** 1-2 days
**AI Effort:** ~4-6 hours
**Human Effort:** ~1 hour (review)

**Context & Resources:**

**Reference Files:**
1. `/Users/donaldpg/PyProjects/worktree2/PyTAAA/functions/TAfunctions.py` lines 627-880 - Weight constraint implementation
2. `/Users/donaldpg/PyProjects/worktree2/PyTAAA/PyTAAA_backtest_sp500_pine_refactored.py` lines 620-700 - Weight constraint parameters
3. `/Users/donaldpg/PyProjects/PyTAAA.master/src/backtest/config.py` - Parameter defaults

**Code Patterns:**
- Current codebase has advanced weight constraints in `sharpeWeightedRank_2D()`
- Key parameters: max_weight_factor, min_weight_factor, absolute_max_weight
- Uses triangular distribution for random exploration

**Algorithm - Weight Constraint Logic:**
```python
def apply_weight_constraints(raw_weights, max_factor, min_factor, abs_max):
    """
    1. Calculate equal_weight = 1.0 / n_stocks
    2. Calculate max_weight = min(equal_weight * max_factor, abs_max)
    3. Calculate min_weight = equal_weight * min_factor
    4. Clip weights to [min_weight, max_weight]
    5. Renormalize to sum to 1.0
    """
    equal_weight = 1.0 / len(raw_weights)
    max_weight = min(equal_weight * max_factor, abs_max)
    min_weight = equal_weight * min_factor
    constrained = np.clip(raw_weights, min_weight, max_weight)
    constrained /= np.sum(constrained)  # Renormalize
    return constrained
```

**Goals:**
1. Document weight constraint implementation differences
2. Create unified parameter interface
3. Ensure backward compatibility with existing code

**Success Criteria:**
- [ ] Weight constraints produce same output as current implementation
- [ ] All three constraint parameters are configurable
- [ ] Default values match current codebase: max_factor=3.0, min_factor=0.3, abs_max=0.9
- [ ] Integration test passes with existing backtest

**Tests to Write:**

```python
# tests/test_phase3_weight_constraints.py

import pytest
import numpy as np
from functions.backtest.weight_constraints import apply_weight_constraints

class TestWeightConstraints:
    """Test weight constraint functions."""
    
    def test_weights_sum_to_one(self):
        """Verify constrained weights sum to 1.0."""
        raw = np.array([0.1, 0.2, 0.3, 0.4])
        constrained = apply_weight_constraints(raw, 3.0, 0.3, 0.9)
        assert abs(np.sum(constrained) - 1.0) < 1e-6
    
    def test_max_weight_respected(self):
        """Verify no weight exceeds max."""
        raw = np.array([0.5, 0.5])  # 50% each - exceeds 90% cap
        constrained = apply_weight_constraints(raw, 3.0, 0.3, 0.9)
        assert np.max(constrained) <= 0.9 + 1e-6
    
    def test_min_weight_applied(self):
        """Verify minimum weight is applied."""
        raw = np.array([0.25, 0.25, 0.25, 0.25])  # Equal weights
        constrained = apply_weight_constraints(raw, 3.0, 0.3, 0.9)
        min_weight = (1.0/4) * 0.3  # Equal weight * min_factor
        assert np.min(constrained[constrained > 0]) >= min_weight - 1e-6
    
    def test_preserves_relative_order(self):
        """Verify relative ordering of weights is preserved."""
        raw = np.array([0.1, 0.3, 0.6])  # Ascending
        constrained = apply_weight_constraints(raw, 3.0, 0.3, 0.9)
        assert np.all(np.diff(constrained) <= 1e-6)  # Still ascending or equal
```

---

### PHASE 3: Integrate Advanced Weight Constraints

#### Stage 1: Planning (Software & Data Architecture)

**Architecture Considerations:**
- Weight constraints are critical for risk management
- Must preserve existing behavior to avoid breaking changes
- Should be configurable but have safe defaults
- Consider performance impact on large portfolios

**Data Flow Design:**
```
Raw Weights → Apply Constraints → Normalized Weights → Portfolio
     ↓
max_factor, min_factor, abs_max (configurable)
```

**Key Design Decisions:**
1. Extract constraint logic to separate pure function
2. Use config from BacktestConfig for defaults
3. Validate constraints before applying
4. Handle edge case of all-zero weights

**Persona Notes:**
> "Weight constraints protect against concentration risk. The function must be deterministic and fast - it's called for every date in the backtest."

---

#### Stage 2: Plan Revision (Constructive Critic)

**Potential Issues to Address:**
- What if constraints result in all zeros?
- Numerical stability of normalization
- Interaction with other weight modifications
- Performance at scale (1000+ stocks)

**Edge Cases:**
- Single stock portfolio
- All weights equal
- Negative weights (should error)
- NaN/Inf in inputs

**Improvements Suggested:**
1. Add validation that input weights are non-negative
2. Add fallback to equal weights if all constrained to zero
3. Add logging for constraint violations
4. Consider vectorized implementation

**Critique Summary:**
> "The algorithm is sound but lacks defensive programming. Add input validation and edge case handling."

---

#### Stage 3: Implementation (Mid-Level Developer)

**Implementation Steps:**
1. Create weight_constraints.py module
2. Implement apply_weight_constraints() function
3. Add input validation
4. Add fallback logic for edge cases
5. Add unit tests
6. Integrate with sharpeWeightedRank_2D
7. Run integration tests

**Code Style:**
- Pure function (no side effects)
- Comprehensive docstring with examples
- Type hints for all parameters
- Handle edge cases explicitly

**Developer Notes:**
> "I'll extract the constraint logic from TAfuctions.py and make it a standalone function. This makes it testable and reusable."

---

#### Stage 4: Code Cleanup (Agentic AI Critic)

**Review Checklist:**
- [ ] All edge cases handled
- [ ] Numerical stability verified
- [ ] Performance acceptable
- [ ] Tests cover boundary conditions
- [ ] Documentation is clear

**Suggested Refactors:**
1. Add @njit decorator for Numba speedup
2. Add weight constraint visualization for debugging
3. Consider adding constraint preset profiles

**AI Critic Summary:**
> "Good implementation. Consider adding a decorator for timing to track performance impact."

---

#### Phase 4: Data Flow Integration
**Duration:** 1 day
**AI Effort:** ~2-3 hours
**Human Effort:** ~30 minutes (review)

**Context & Resources:**

**Reference Files:**
1. `/Users/donaldpg/PyProjects/worktree2/PyTAAA/functions/data_loaders.py` - Data loading functions
2. `/Users/donaldpg/PyProjects/worktree2/PyTAAA/functions/UpdateSymbols_inHDF5.py` - HDF5 data access
3. `/Users/donaldpg/PyProjects/worktree2/PyTAAA/functions/quotes_for_list_adjClose.py` - Quote retrieval
4. `/Users/donaldpg/PyProjects/worktree2/PyTAAA/functions/ta/signal_generation.py` - Signal computation
5. `/Users/donaldpg/PyProjects/worktree2/PyTAAA/functions/ta/channels.py` - Channel calculations

**Data Flow Diagram:**
```
JSON Config
    ↓
loadQuotes_fromHDF() → datearray, symbols, adjClose, volume
    ↓
percentileChannel_2D() / computeSignal2D() → signal2D
    ↓
sharpeWeightedRank_2D() → monthgainlossweight
    ↓
dailyBacktest_pctLong() → portfolio values, metrics
    ↓
CSV / PNG Output
```

**Key Functions to Import:**
- `loadQuotes_fromHDF()` from `functions.UpdateSymbols_inHDF5`
- `computeSignal2D()` from `functions.ta.signal_generation`
- `percentileChannel_2D()` from `functions.ta.channels`
- `sharpeWeightedRank_2D()` from `functions.TAfunctions`

**Goals:**
1. Verify all imports work correctly
2. Ensure data flows through pipeline
3. Integration test with real data

**Success Criteria:**
- [ ] All imports resolve without errors
- [ ] Can load sample data from existing JSON config
- [ ] Signal generation produces valid output
- [ ] End-to-end test with small dataset passes

**Tests to Write:**

```python
# tests/test_phase4_data_flow.py

import pytest
import os
import json

class TestDataFlowIntegration:
    """Test data flow through backtest pipeline."""
    
    @pytest.fixture
    def sample_config(self):
        """Load sample JSON config."""
        config_path = "pytaaa_sp500_pine.json"
        if os.path.exists(config_path):
            with open(config_path) as f:
                return json.load(f)
        pytest.skip("Config file not found")
    
    def test_imports_work(self):
        """Verify all required imports work."""
        from functions.UpdateSymbols_inHDF5 import loadQuotes_fromHDF
        from functions.ta.signal_generation import computeSignal2D
        from functions.ta.channels import percentileChannel_2D
        from functions.TAfunctions import sharpeWeightedRank_2D
        # If we get here, imports work
        assert True
    
    def test_config_loading(self, sample_config):
        """Verify config can be loaded."""
        assert 'Valuation' in sample_config or 'numberStocksTraded' in sample_config
```

---

### PHASE 4: Data Flow Integration

#### Stage 1: Planning (Software & Data Architecture)

**Architecture Considerations:**
- Integration must preserve existing data flow
- Minimize coupling between modules
- Handle missing data gracefully
- Consider caching strategies for performance

**Data Flow Design:**
```
JSON → Config → Data Loader → TA Functions → Weight Calc → Backtest → Output
```

**Key Design Decisions:**
1. Use dependency injection for data loaders
2. Add abstraction layer for data sources
3. Implement data validation at boundaries

**Persona Notes:**
> "Data flow integration is about connecting existing components. We want loose coupling so we can swap implementations later."

---

#### Stage 2: Plan Revision (Constructive Critic)

**Potential Issues to Address:**
- Circular import dependencies
- Data format mismatches between modules
- Missing data handling
- Performance bottlenecks

**Edge Cases:**
- HDF5 file corruption
- Missing symbols
- Date range gaps
- Memory usage with large datasets

**Improvements Suggested:**
1. Add retry logic for transient failures
2. Implement data validation at each boundary
3. Add logging for debugging
4. Consider lazy loading for large datasets

**Critique Summary:**
> "Need explicit error handling for data loading failures. Add retry logic for transient failures."

---

#### Stage 3: Implementation (Mid-Level Developer)

**Implementation Steps:**
1. Verify all imports work correctly
2. Test data loading with sample config
3. Verify signal generation produces valid output
4. Run small end-to-end test
5. Add error handling for failures

**Code Style:**
- Handle exceptions gracefully
- Add meaningful error messages
- Log important events
- Keep functions focused

**Developer Notes:**
> "I'll verify each import path works and add fallback handling for missing data."

---

#### Stage 4: Code Cleanup (Agentic AI Critic)

**Review Checklist:**
- [ ] Import paths are correct and minimal
- [ ] Error messages guide user to solution
- [ ] No hardcoded paths
- [ ] Data validation at boundaries
- [ ] Logging is appropriate

**Suggested Refactors:**
1. Add a data validation module
2. Consider caching layer for frequently accessed data
3. Add type hints to data structures

**AI Critic Summary:**
> "Solid implementation. Consider adding a data validation module to catch issues early."

---

#### Phase 5: Testing and Validation
**Duration:** 1-2 days
**AI Effort:** ~4-6 hours
**Human Effort:** ~2 hours (review, testing)

**Context & Resources:**

**Reference Files:**
1. `/Users/donaldpg/PyProjects/PyTAAA.master/tests/` - Test structure in master
2. `/Users/donaldpg/PyProjects/worktree2/PyTAAA/tests/` - Existing tests
3. `/Users/donaldpg/PyProjects/PyTAAA.master/src/backtest/montecarlo_runner.py` - Full integration

**Testing Strategy:**
- Unit tests for each module (Phase 1-3 tests)
- Integration tests for data flow (Phase 4)
- End-to-end tests with real data
- Compare outputs with original implementation

**Goals:**
1. All unit tests pass
2. Integration tests pass
3. Backward compatibility verified
4. Performance acceptable

**Success Criteria:**
- [ ] `pytest tests/test_phase1_*.py` - All pass
- [ ] `pytest tests/test_phase2_*.py` - All pass
- [ ] `pytest tests/test_phase3_*.py` - All pass
- [ ] `pytest tests/test_phase4_*.py` - All pass
- [ ] CLI entry point works with real config
- [ ] Output matches original implementation (within tolerance)
- [ ] No regression in existing tests

**Integration Tests to Write:**

```python
# tests/test_phase5_integration.py

import pytest
import os
import json

class TestEndToEndBacktest:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def config_path(self):
        """Get path to test config."""
        return "pytaaa_sp500_pine.json"
    
    def test_cli_runs_successfully(self, config_path):
        """Test CLI runs without errors."""
        if not os.path.exists(config_path):
            pytest.skip("Config not found")
        
        from click.testing import CliRunner
        from functions.backtest.entrance import main
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            # Create a minimal test
            result = runner.invoke(main, [
                '--config', config_path,
                '--n_trials', '2',
                '--output_csv', 'test_results.csv'
            ])
            # Allow for expected errors (data issues, etc.)
            # but not import errors or structural failures
            assert 'Traceback' not in result.output
    
    def test_csv_output_format(self):
        """Test CSV has expected columns."""
        import csv
        if not os.path.exists('test_results.csv'):
            pytest.skip("Results file not found")
            
        with open('test_results.csv') as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            
            # Check for expected columns
            expected = ['run', 'trial', 'Portfolio Final Value', 'Portfolio Sharpe']
            for col in expected:
                assert col in fields, f"Missing column: {col}"
```

---

### PHASE 5: Testing and Validation

#### Stage 1: Planning (Software & Data Architecture)

**Architecture Considerations:**
- Test coverage should cover all new modules
- Integration tests verify end-to-end functionality
- Regression tests ensure no breaking changes
- Performance tests establish baselines

**Testing Strategy:**
```
Unit Tests → Integration Tests → E2E Tests → Regression Tests
     ↓            ↓              ↓            ↓
  Phase 1-3    Phase 4      Full Run    Compare
```

**Key Design Decisions:**
1. Use pytest framework (already in project)
2. Create tests/ directory for new tests
3. Use fixtures for shared test data
4. Parametrize tests where applicable

**Persona Notes:**
> "Testing validates our assumptions. We'll use the existing pytest setup and add tests specific to the new backtest module."

---

#### Stage 2: Plan Revision (Constructive Critic)

**Potential Issues to Address:**
- What tests catch regressions?
- How do we compare outputs between implementations?
- What are acceptable performance baselines?
- How often should tests run?

**Edge Cases:**
- Empty data sets
- Extreme parameter values
- Network failures
- File permission errors

**Improvements Suggested:**
1. Add performance benchmarks
2. Add snapshot tests for output format
3. Add CI/CD integration
4. Add test coverage reporting

**Critique Summary:**
> "Need clear success criteria for each test type. Add baseline comparisons for regression testing."

---

#### Stage 3: Implementation (Mid-Level Developer)

**Implementation Steps:**
1. Run all Phase 1-4 tests
2. Add integration tests for full pipeline
3. Add comparison tests with original implementation
4. Add performance benchmarks
5. Document test results

**Code Style:**
- Use descriptive test names
- Add docstrings to test classes
- Use fixtures for setup/teardown
- Keep tests focused and independent

**Developer Notes:**
> "I'll run all tests and verify no regressions. Then add any missing coverage."

---

#### Stage 4: Code Cleanup (Agentic AI Critic)

**Review Checklist:**
- [ ] All tests pass
- [ ] Test coverage is adequate (>80%)
- [ ] No test duplication
- [ ] Tests are maintainable
- [ ] Documentation is complete

**Suggested Refactors:**
1. Add test coverage reporting
2. Add CI/CD pipeline
3. Add performance regression tests

**AI Critic Summary:**
> "Good test coverage. Consider adding automated test reporting."

---

### 3.4 Hybrid Feature Selection

Based on analysis, here are the recommended features to blend:

#### From PyTAAA.master (Adopt)
1. **CLI Entry Point Structure** - Simple, focused on Monte Carlo
2. **Configuration Classes** - Centralized parameter management
3. **CSV Output Format** - Comprehensive metrics tracking
4. **Modular Package Structure** - Clean separation of concerns

#### From Current Codebase (Keep)
1. **Advanced Weight Constraints** - `max_weight_factor`, `min_weight_factor`, `absolute_max_weight`
2. **Numba Optimization** - Performance improvements
3. **Model Switching** - Multi-model portfolio selection
4. **Focus Period Blending** - Configurable time period weighting
5. **Early Period Handling** - CASH allocation logic

#### New (Create)
1. **Unified Configuration** - Blend both approaches
2. **Hybrid Weight Calculation** - Best of both implementations
3. **Enhanced CSV Output** - Add missing metrics from master

### 3.5 Entry Point Design

The new entry point should be similar to:

```python
# functions/backtest/entrance.py
import click
from functions.backtest.montecarlo_runner import run_montecarlo

@click.command()
@click.option('--n_trials', default=250, help='Number of Monte Carlo trials')
@click.option('--config', 'config_file', required=True, 
              type=click.Path(exists=True), help='Path to JSON config')
@click.option('--output_csv', default='montecarlo_results.csv',
              help='Path to output CSV file')
@click.option('--plot_individual', is_flag=True,
              help='Generate individual plots for each trial')
def main(n_trials, config_file, output_csv, plot_individual):
    run_montecarlo(
        n_trials=n_trials,
        base_json_fn=config_file,
        output_csv=output_csv,
        plot_individual=plot_individual
    )

if __name__ == '__main__':
    main()
```

---

## Part 4: Risk Assessment

### 4.1 Low Risk Items
- Configuration class extraction
- CLI entry point creation
- CSV output formatting

### 4.2 Medium Risk Items
- Weight constraint blending (may affect portfolio performance)
- Module import changes

### 4.3 Mitigation Strategies
1. **Maintain Original Functions:** Keep existing implementations as fallbacks
2. **Comprehensive Testing:** Compare outputs before/after changes
3. **Incremental Changes:** Implement one phase at a time
4. **Feature Flags:** Use configuration to toggle between implementations

---

## Summary

The PyTAAA.master codebase underwent significant refactoring in December 2025, creating a modular structure with a simple CLI entry point. The current codebase (worktree2) has more advanced features in some areas (weight constraints, model switching) but lacks the clean organization of the refactored version.

The proposed refactoring plan aims to:
1. Adopt the modular structure from PyTAAA.master
2. Preserve advanced features from the current codebase
3. Create a simple, maintainable entry point
4. Avoid duplicating existing functionality

### Total Time Estimate

| Phase | Duration | AI Effort | Human Effort |
|-------|----------|-----------|--------------|
| Phase 1: Infrastructure | 1-2 days | ~4-6 hours | ~1-2 hours |
| Phase 2: CLI Entry Point | 1 day | ~2-3 hours | ~30 min |
| Phase 3: Weight Integration | 1-2 days | ~4-6 hours | ~1 hour |
| Phase 4: Data Integration | 1 day | ~2-3 hours | ~30 min |
| Phase 5: Testing | 1-2 days | ~4-6 hours | ~2 hours |
| **Total** | **5-8 days** | **~16-24 hours** | **~5-6 hours** |

**Note:** With access to advanced AI models (Claude Sonnet 4.5), implementation speed may be faster. The AI effort estimates assume the AI does ~90% of coding while human does ~10% (reviews, approvals, minor edits).

---

## Part 5: AI Implementation Prompts

## Part 5: AI Implementation Prompts

### Phase 1 Implementation Prompt

Copy and paste this prompt to an AI coding assistant to implement Phase 1:

```
## Task: Implement Phase 1 - Create Backtest Module Infrastructure

### Objective
Create the initial module infrastructure for a new `functions/backtest/` package that will contain refactored backtesting code.

### ⚠️ IMPORTANT: Absolute File Paths
All file paths below are RELATIVE to the worktree2 codebase at:
`/Users/donaldpg/PyProjects/worktree2/PyTAAA/`

The PyTAAA.master codebase is at:
`/Users/donaldpg/PyProjects/PyTAAA.master/`

### Directory Structure to Create
```
functions/backtest/
├── __init__.py
├── config.py
└── montecarlo.py
```

### Directory Structure for Tests
```
tests/
├── test_phase1_config.py
└── test_phase1_montecarlo.py
```

### Reference Files - EXACT LOCATIONS

**From PyTAAA.master (READ ONLY - use for patterns):**
1. `/Users/donaldpg/PyProjects/PyTAAA.master/src/backtest/config.py` - Source of TradingConstants, BacktestConfig, FilePathConfig classes
2. `/Users/donaldpg/PyProjects/PyTAAA.master/src/backtest/montecarlo.py` - Source of random_triangle, calculate_sharpe_ratio, etc.

**From worktree2 (READ and COPY):**
3. `/Users/donaldpg/PyProjects/worktree2/PyTAAA/PyTAAA_backtest_sp500_pine_refactored.py` lines 87-200 - Local TradingConstants (ALREADY EXISTS - don't copy, use existing)
4. `/Users/donaldpg/PyProjects/worktree2/PyTAAA/functions/MonteCarloBacktest.py` - EXISTS but is DIFFERENT from what we need

### ⚠️ CRITICAL: Copy vs Import Decision

**DO NOT COPY from PyTAAA.master for TradingConstants.** The worktree2 version at lines 87-200 in PyTAAA_backtest_sp500_pine_refactored.py is the canonical version to use.

**DO COPY from PyTAAA.master:**
- BacktestConfig class structure
- FilePathConfig class structure
- All montecarlo.py functions (random_triangle, calculate_sharpe_ratio, etc.)

### Requirements

#### 1. functions/backtest/__init__.py
Create a package init file that exports:
- TradingConstants (from worktree2 - see below)
- BacktestConfig (copy from PyTAAA.master)
- FilePathConfig (copy from PyTAAA.master)
- random_triangle (copy from PyTAAA.master)
- calculate_sharpe_ratio (copy from PyTAAA.master)
- calculate_period_metrics (copy from PyTAAA.master)
- calculate_drawdown_metrics (copy from PyTAAA.master)

**HOW TO GET TradingConstants:**
Do NOT copy. Add this import at the top of __init__.py:
```python
# Import from existing location in worktree2
from PyTAAA_backtest_sp500_pine_refactored import TradingConstants
```

#### 2. functions/backtest/config.py
Create these classes:

**TradingConstants:** DO NOT CREATE - import from worktree2
```python
# At top of config.py:
from PyTAAA_backtest_sp500_pine_refactored import TradingConstants

# Then add BacktestConfig and FilePathConfig copied from PyTAAA.master
```

**BacktestConfig** (COPY from PyTAAA.master `/Users/donaldpg/PyProjects/PyTAAA.master/src/backtest/config.py`):
- DEFAULT_RANDOM_TRIALS = 250
- PYTAAA_NUMBER_STOCKS_TRADED = 6
- PYTAAA_MONTHS_TO_HOLD = 1
- PYTAAA_LONG_PERIOD = 412
- And ALL other parameters from PyTAAA.master config.py

**FilePathConfig** (COPY from PyTAAA.master):
- SP500_DATA_PATH
- SP500_PINE_DATA_PATH  
- JSON_CONFIG_FILE
- OUTPUT_DIR
- All helper methods

#### 3. functions/backtest/montecarlo.py
COPY these functions EXACTLY from `/Users/donaldpg/PyProjects/PyTAAA.master/src/backtest/montecarlo.py`:

**random_triangle(low, mid, high, size=1)**
**create_temporary_json(base_json_fn, realization_params, iter_num)**
**cleanup_temporary_json(temp_json_fn)**
**calculate_sharpe_ratio(daily_gains, trading_days=252)**
**calculate_period_metrics(portfolio_value, trading_days=252)**
**calculate_drawdown_metrics(portfolio_value)**

### ⚠️ MonteCarloBacktest Class Clarification

The success criteria mentions `MonteCarloBacktest` but this is NOT in PyTAAA.master's config.py.

**DO NOT create it in Phase 1.** This is handled in later phases.

Remove this from Success Criteria:
- ❌ `from functions.backtest import MonteCarloBacktest works`

### Tests to Write

Create `tests/test_phase1_config.py`:

```python
import pytest
from functions.backtest import (
    TradingConstants, 
    BacktestConfig, 
    FilePathConfig
)

class TestTradingConstants:
    """Test TradingConstants class - imported from worktree2."""
    
    def test_trading_days_per_year(self):
        """Verify TRADING_DAYS_PER_YEAR is 252."""
        assert TradingConstants.TRADING_DAYS_PER_YEAR == 252
    
    def test_period_constants(self):
        """Verify period constants are correct."""
        assert TradingConstants.TRADING_DAYS_1_YEAR == 252
        assert TradingConstants.TRADING_DAYS_2_YEARS == 504
        assert TradingConstants.TRADING_DAYS_5_YEARS == 1260
    
    def test_initial_portfolio_value(self):
        """Verify initial portfolio value."""
        assert TradingConstants.INITIAL_PORTFOLIO_VALUE == 10000.0

class TestBacktestConfig:
    """Test BacktestConfig class - copied from PyTAAA.master."""
    
    def test_default_random_trials(self):
        """Verify default random trials is 250."""
        assert BacktestConfig.DEFAULT_RANDOM_TRIALS == 250
    
    def test_pytaaa_defaults(self):
        """Verify pyTAAA default parameters."""
        assert BacktestConfig.PYTAAA_NUMBER_STOCKS_TRADED == 6
        assert BacktestConfig.PYTAAA_MONTHS_TO_HOLD == 1
    
    def test_exploration_ranges(self):
        """Verify exploration ranges are tuples."""
        assert isinstance(BacktestConfig.LONG_PERIOD_RANGE, tuple)
        assert isinstance(BacktestConfig.EXPLORATION_LONG_PERIOD, tuple)

class TestFilePathConfig:
    """Test FilePathConfig class - copied from PyTAAA.master."""
    
    def test_sp500_data_path_exists(self):
        """Verify SP500_DATA_PATH is set."""
        assert hasattr(FilePathConfig, 'SP500_DATA_PATH')
        assert isinstance(FilePathConfig.SP500_DATA_PATH, str)
```

Create `tests/test_phase1_montecarlo.py`:

```python
import pytest
import numpy as np
import os
import tempfile
from functions.backtest.montecarlo import (
    random_triangle,
    calculate_sharpe_ratio,
    calculate_period_metrics,
    calculate_drawdown_metrics,
    create_temporary_json,
    cleanup_temporary_json
)

class TestRandomTriangle:
    """Test random_triangle function."""
    
    def test_output_shape(self):
        """Verify output has correct shape."""
        result = random_triangle(1, 2, 3, 100)
        assert result.shape == (100,)
    
    def test_output_range(self):
        """Verify output values are within bounds."""
        result = random_triangle(1, 2, 3, 1000)
        assert np.all(result >= 1.0)
        assert np.all(result <= 3.0)
    
    def test_mean_near_mid(self):
        """Verify mean is near the mid point."""
        result = random_triangle(1, 2, 3, 10000)
        mean_val = np.mean(result)
        assert 1.8 < mean_val < 2.2  # Should be near 2.0

class TestCalculateSharpeRatio:
    """Test calculate_sharpe_ratio function."""
    
    def test_positive_returns(self):
        """Test with positive returns."""
        daily_gains = np.array([1.001, 1.002, 1.001, 1.003, 1.002])
        sharpe = calculate_sharpe_ratio(daily_gains)
        assert isinstance(sharpe, (float, np.floating))
    
    def test_zero_std(self):
        """Test with zero standard deviation."""
        daily_gains = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        sharpe = calculate_sharpe_ratio(daily_gains)
        assert sharpe == 0.0  # No risk

class TestCalculatePeriodMetrics:
    """Test calculate_period_metrics function."""
    
    def test_output_keys(self):
        """Verify output has expected keys."""
        portfolio = np.cumprod(np.random.rand(1000) * 0.1 + 0.9)
        metrics = calculate_period_metrics(portfolio)
        expected_keys = ['5Yr', '3Yr', '2Yr', '1Yr']
        for key in expected_keys:
            assert key in metrics
    
    def test_cagr_bounds(self):
        """Verify CAGR is within reasonable bounds."""
        portfolio = np.array([10000] * 252)  # Flat line
        metrics = calculate_period_metrics(portfolio)
        # Should have valid CAGR even for flat portfolio
        assert '1Yr' in metrics

class TestCalculateDrawdownMetrics:
    """Test calculate_drawdown_metrics function."""
    
    def test_output_structure(self):
        """Verify output structure."""
        portfolio = np.cumprod(np.random.rand(500) * 0.1 + 0.9)
        dd_metrics = calculate_drawdown_metrics(portfolio)
        assert isinstance(dd_metrics, dict)

class TestCreateTemporaryJson:
    """Test create_temporary_json function."""
    
    def test_creates_file(self):
        """Verify temp JSON file is created."""
        # Create a minimal test config
        test_params = {'test': 'value', 'numberStocksTraded': 5}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_file = create_temporary_json(
                os.path.join(tmpdir, 'base.json'),
                test_params,
                1
            )
            
            # Verify file was created
            assert os.path.exists(temp_file)
            
            # Verify it can be read back
            import json
            with open(temp_file) as f:
                data = json.load(f)
            assert data['test'] == 'value'
            
            # Cleanup
            cleanup_temporary_json(temp_file)
            assert not os.path.exists(temp_file)
```

### ✅ CORRECTED Success Criteria

- [ ] `from functions.backtest import TradingConstants` works
- [ ] `from functions.backtest import BacktestConfig` works  
- [ ] `from functions.backtest import FilePathConfig` works
- [ ] `TradingConstants.TRADING_DAYS_PER_YEAR == 252`
- [ ] `BacktestConfig.DEFAULT_RANDOM_TRIALS == 250`
- [ ] `random_triangle(1, 2, 3, 100).shape == (100,)`
- [ ] All new files pass `python -m py_compile functions/backtest/*.py`
- [ ] All tests pass: `pytest tests/test_phase1_*.py`

### Dependency Check
Ensure numpy is available (should already be in project):
```bash
python -c "import numpy; print(numpy.__version__)"
```

### Notes
- Use numpy for array operations
- Keep functions pure (no side effects) except for create_temporary_json
- Use dataclasses where appropriate
```

---

## Part 6: Phase-by-Phase Detailed Workflow

Each phase follows a 4-stage process:

### Stage 1: Planning (Software & Data Architecture Persona)
*Persona: Senior architect with deep understanding of data flow, system design, and financial software patterns*

### Stage 2: Plan Revision (Constructive Critic Persona)  
*Persona: Experienced software designer who challenges assumptions, finds edge cases, and improves design*

### Stage 3: Implementation (Mid-Level Developer Persona)
*Persona: Competent developer who writes clean, testable code following specifications*

### Stage 4: Code Cleanup (Agentic AI Critic Persona)
*Persona: AI agent that reviews code for quality, suggests improvements, and ensures best practices*

---

### PHASE 1: Create Backtest Module Infrastructure

#### Stage 1: Planning (Software & Data Architecture)

**Architecture Considerations:**
- Module should follow Python package conventions
- Configuration classes should use immutable patterns where possible
- Monte Carlo functions should be stateless for testability
- Consider future extensibility for additional backtest methods

**Data Flow Design:**
```
JSON Config → BacktestConfig → Monte Carlo Functions → Metrics Dict
     ↓
FilePathConfig → Output file paths (via class methods)
     ↓
TradingConstants → Performance calculations (static values)
```

**Key Design Decisions:**
1. Use class-level constants (not instances) for TradingConstants
2. Use @classmethod for FilePathConfig path generators
3. MonteCarloBacktest class for stateful parameter generation
4. Separate pure functions from I/O-heavy operations

**Persona Notes:**
> "This phase establishes the foundation. The config module will be imported throughout the system, so stability and clarity are paramount. I'm structuring FilePathConfig with class methods to allow runtime path customization without code changes."

---

#### Stage 2: Plan Revision (Constructive Critic)

**Potential Issues to Address:**
- What happens if PyTAAA.master changes their config? Should we mirror or diverge?
- How do we handle path resolution on different systems?
- Should config values be validated at import time or runtime?
- Thread safety concerns for Monte Carlo parallel execution

**Edge Cases:**
- Empty portfolio values for drawdown calculation
- Division by zero in Sharpe ratio
- Invalid date ranges in period metrics
- File path resolution failures

**Improvements Suggested:**
1. Add runtime path validation in FilePathConfig
2. Add validation for TradingConstants bounds
3. Add type hints to all function signatures
4. Consider using frozen dataclasses for immutable configs

**Critique Summary:**
> "The plan is solid but needs defensive programming. Add input validation and handle edge cases explicitly rather than relying on numpy errors."

---

#### Stage 3: Implementation (Mid-Level Developer)

**Implementation Steps:**
1. Create `functions/backtest/` directory
2. Create `__init__.py` with package exports
3. Implement TradingConstants class
4. Implement BacktestConfig class
5. Implement FilePathConfig class
6. Implement random_triangle function
7. Implement calculate_sharpe_ratio function
8. Implement calculate_period_metrics function
9. Implement calculate_drawdown_metrics function
10. Write unit tests

**Code Style:**
- Use numpy for all array operations
- Add docstrings to all public functions
- Use type hints for function parameters
- Follow PEP 8 naming conventions
- Keep functions under 50 lines

**Developer Notes:**
> "I'll structure each class to match PyTAAA.master exactly first, then add improvements. The Monte Carlo functions should be pure - no file I/O inside calculate_ functions."

---

#### Stage 4: Code Cleanup (Agentic AI Critic)

**Review Checklist:**
- [ ] All imports are necessary and organized (stdlib, then third-party, then local)
- [ ] No hardcoded values that should be constants
- [ ] Docstrings explain "why" not just "what"
- [ ] Error messages are actionable
- [ ] Type hints are complete
- [ ] No mutable default arguments
- [ ] Tests cover edge cases

**Suggested Refactors:**
1. Extract magic numbers (e.g., 252) to TradingConstants
2. Add @staticmethod decorators where appropriate
3. Consider using enum for period keys instead of strings
4. Add deprecation warnings if copying from PyTAAA.master directly

**AI Critic Summary:**
> "Code is functional but could benefit from stronger typing. Consider adding a ConfigValidator class to catch issues early."

This hybrid approach should result in a codebase that is easier to maintain while retaining the sophisticated trading logic that makes the system effective.
