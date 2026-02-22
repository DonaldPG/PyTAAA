# PyTAAA Refactoring Plan: Agentic AI Implementation Guide

**Version:** 1.0  
**Last Updated:** February 9, 2026  
**Branch:** `refactor/modernize`  
**Status:** Draft - Pending Constructive Critic Review

---

## Executive Summary

This plan provides a phased, test-driven approach to refactoring the PyTAAA codebase based on the analysis in [`docs/RECOMMENDATIONS.md`](../docs/RECOMMENDATIONS.md). The goal is to modernize the codebase while maintaining **exact functional equivalence** — all outputs must remain identical.

### Key Principles

1. **Test-First**: Every phase includes comprehensive tests to validate no behavioral changes
2. **Incremental**: Small, reviewable changes with clear success criteria
3. **Reversible**: Each phase is a git commit; any phase can be reverted independently
4. **Validated**: End-to-end testing against known-good outputs before and after each phase

---

## Pre-Flight Checklist (Before Any Phase Begins)

### Environment Setup

```bash
# 1. Create and checkout feature branch
git checkout -b refactor/modernize

# 2. Establish baseline - run all validation commands and capture outputs
cd /Users/donaldpg/PyProjects/worktree2/PyTAAA

# Create baseline directory
mkdir -p .refactor_baseline/{before,after}

# Run baseline tests - these commands must produce identical outputs after each phase
uv run python pytaaa_main.py --json /Users/donaldpg/pyTAAA_data_static/naz100_pine/pytaaa_naz100_pine.json 2>&1 | tee .refactor_baseline/before/pytaaa_naz100_pine.log
uv run python pytaaa_main.py --json /Users/donaldpg/pyTAAA_data_static/naz100_hma/pytaaa_naz100_hma.json 2>&1 | tee .refactor_baseline/before/pytaaa_naz100_hma.log
uv run python pytaaa_main.py --json /Users/donaldpg/pyTAAA_data_static/naz100_pi/pytaaa_naz100_pi.json 2>&1 | tee .refactor_baseline/before/pytaaa_naz100_pi.log
uv run python pytaaa_main.py --json /Users/donaldpg/pyTAAA_data_static/sp500_hma/pytaaa_sp500_hma.json 2>&1 | tee .refactor_baseline/before/pytaaa_sp500_hma.log
uv run python pytaaa_main.py --json /Users/donaldpg/pyTAAA_data_static/sp500_pine/pytaaa_sp500_pine.json 2>&1 | tee .refactor_baseline/before/pytaaa_sp500_pine.log
uv run python recommend_model.py --json /Users/donaldpg/pyTAAA_data_static/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json 2>&1 | tee .refactor_baseline/before/pytaaa_abacus_recommendation.log
uv run python daily_abacus_update.py --json /Users/donaldpg/pyTAAA_data_static/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json --verbose 2>&1 | tee .refactor_baseline/before/pytaaa_abacus_daily.log

# Capture .params file checksums for comparison
find /Users/donaldpg/pyTAAA_data_static -name "*.params" -exec md5sum {} \; > .refactor_baseline/before/params_checksums.txt

# 3. Verify tests pass before starting
uv run pytest tests/ -v 2>&1 | tee .refactor_baseline/before/pytest_baseline.log
```

### Baseline Requirements Met?

- [ ] All 7 end-to-end commands executed successfully
- [ ] All `.params` files captured with checksums
- [ ] All pytest tests pass (or known failures documented)
- [ ] Git branch `refactor/modernize` created and checked out
- [ ] `.refactor_baseline/` directory in `.gitignore`

---

## Phase 1: Foundation — Dead Code Removal & Documentation

**Complexity:** Low  
**Risk:** Low  
**Estimated Time:** 2-3 AI sessions  
**AI Model Recommendation:** Kimi K2.5 (high accuracy on pattern matching, lower cost)

### 1.1 Goals

1. Remove dead code (commented-out functions, unused imports, superseded definitions)
2. Add comprehensive docstrings to all public functions
3. Remove Python 2 compatibility code
4. Establish clean foundation for subsequent phases

### 1.2 Files to Modify

| File | Changes |
|------|---------|
| `functions/CheckMarketOpen.py` | Remove 3 superseded `get_MarketOpenOrClosed()` definitions (lines 29-75) |
| `functions/TAfunctions.py` | Remove commented-out `interpolate()`, `cleantobeginning()` (lines 35-94, 96+) |
| `functions/quotes_for_list_adjClose.py` | Remove commented-out class and functions (lines 17-115) |
| `functions/GetParams.py` | Add docstrings to all public functions |
| `re-generateHDF5.py` | Delete or archive (Python 2 syntax, non-functional) |

### 1.3 Detailed Checklist

#### Task 1.1: Audit Dead Code

- [ ] Search for all multi-line commented-out code blocks (`'''` or `"""` containing function definitions)
- [ ] Identify functions with multiple definitions (same name, different implementations)
- [ ] List all `try/except ImportError` blocks for Python 2/3 compatibility
- [ ] Document findings in `.refactor_baseline/dead_code_audit.md`

#### Task 1.2: Remove Dead Code

- [ ] `functions/CheckMarketOpen.py`: Keep only the active `get_MarketOpenOrClosed()` (lines 77-104)
- [ ] `functions/TAfunctions.py`: Remove lines 35-94 (commented interpolate/cleantobeginning)
- [ ] `functions/quotes_for_list_adjClose.py`: Remove lines 17-115 (commented webpage_companies_extractor class)
- [ ] `functions/quotes_for_list_adjClose.py`: Remove lines 959-961 (deprecated retry logic with `print "."`)
- [ ] `functions/quotes_for_list_adjClose.py`: Remove lines 980-987 (deprecated import fallback)
- [ ] Delete `re-generateHDF5.py` or move to `archive/` directory

#### Task 1.3: Add Docstrings

For each public function in modified files, add Google-style docstrings:

```python
def function_name(param1: type1, param2: type2) -> return_type:
    """Short description.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: When this exception occurs
    """
```

Priority functions to document:
- [ ] `functions/GetParams.py`: All `get_*` functions
- [ ] `functions/CheckMarketOpen.py`: `get_MarketOpenOrClosed()`, `CheckMarketOpen()`
- [ ] `functions/TAfunctions.py`: `strip_accents()`, `normcorrcoef()`

#### Task 1.4: Write Tests

Create `tests/test_phase1_cleanup.py`:

```python
"""Tests for Phase 1 cleanup - verify dead code removal doesn't break functionality."""

import pytest
import importlib

class TestDeadCodeRemoval:
    """Verify that removed dead code wasn't actually used."""
    
    def test_check_market_open_imports(self):
        """CheckMarketOpen module imports successfully."""
        from functions import CheckMarketOpen
        assert hasattr(CheckMarketOpen, 'get_MarketOpenOrClosed')
        assert hasattr(CheckMarketOpen, 'CheckMarketOpen')
    
    def test_tafunctions_imports(self):
        """TAfunctions module imports successfully."""
        from functions import TAfunctions
        # Verify key functions still exist
        assert hasattr(TAfunctions, 'interpolate')
        assert hasattr(TAfunctions, 'cleantobeginning')
        assert hasattr(TAfunctions, 'cleantoend')
    
    def test_no_duplicate_definitions(self):
        """Verify no function is defined multiple times in same file."""
        import ast
        import inspect
        from functions import CheckMarketOpen
        
        source = inspect.getsource(CheckMarketOpen)
        tree = ast.parse(source)
        
        function_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_names.append(node.name)
        
        # Check for duplicates
        duplicates = [name for name in set(function_names) if function_names.count(name) > 1]
        assert len(duplicates) == 0, f"Duplicate function definitions found: {duplicates}"
```

#### Task 1.5: Validation

- [ ] Run pytest: `uv run pytest tests/test_phase1_cleanup.py -v`
- [ ] Run all 7 end-to-end commands, compare outputs to baseline
- [ ] Verify `.params` file checksums match baseline
- [ ] Document any differences in `.refactor_baseline/phase1_differences.md`

#### Task 1.6: Commit

```bash
git add -A
git commit -m "Phase 1: Remove dead code and add docstrings

- Remove 3 superseded get_MarketOpenOrClosed() definitions
- Remove commented-out interpolate/cleantobeginning in TAfunctions.py
- Remove commented webpage_companies_extractor class
- Delete non-functional re-generateHDF5.py (Python 2)
- Add Google-style docstrings to public functions
- Add tests/test_phase1_cleanup.py

All end-to-end tests pass with identical outputs to baseline."
```

---

## Phase 2: Exception Handling — Replace Bare `except:` Clauses

**Complexity:** Medium  
**Risk:** Medium (may expose currently-silent failures)  
**Estimated Time:** 3-4 AI sessions  
**AI Model Recommendation:** Claude 4.5 Sonnet (better at reasoning about exception hierarchies)

### 2.1 Goals

1. Replace all bare `except:` with specific exception types
2. Add logging for caught exceptions
3. Maintain identical behavior for expected failure modes
4. Document any exposed failures for future fixes

### 2.2 Safety-First Approach

Instead of directly changing `except:` to specific exceptions, we use a **two-step process**:

**Step 2.1 (Logging Mode):** Add logging to understand what exceptions are actually caught
**Step 2.2 (Fix Mode):** Replace with specific exceptions based on observed behavior

### 2.3 Files to Modify (Priority Order)

| Priority | File | Bare `except:` Count |
|----------|------|---------------------|
| P0 | `functions/CheckMarketOpen.py` | 3 |
| P0 | `PyTAAA.py` | 4 |
| P0 | `run_pytaaa.py` | 4 |
| P1 | `functions/TAfunctions.py` | 6 |
| P1 | `functions/MakeValuePlot.py` | 9 |
| P1 | `functions/WriteWebPage_pi.py` | 9 |
| P2 | `functions/quotes_for_list_adjClose.py` | 11 |
| P2 | `functions/dailyBacktest_pctLong.py` | 2 |
| P2 | `functions/PortfolioPerformanceCalcs.py` | 2 |
| P3 | Other files (lower risk) | ~90 |

### 2.4 Detailed Checklist

#### Task 2.1: Create Exception Logging Decorator

Create `functions/exception_logger.py`:

```python
"""Temporary utility for logging exceptions during refactoring.

This module provides a decorator to log exceptions without changing
control flow. Used during Phase 2 to understand exception patterns.
"""

import logging
import functools
import traceback

logger = logging.getLogger(__name__)

def log_exceptions(func):
    """Decorator that logs all exceptions before re-raising.
    
    Usage:
        @log_exceptions
        def my_function():
            try:
                risky_operation()
            except:  # bare except
                fallback()
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(
                f"Exception in {func.__name__}: {type(e).__name__}: {e}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            raise
    return wrapper

class ExceptionLogger:
    """Context manager to log exceptions in a block."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.warning(
                f"Caught exception: {exc_type.__name__}: {exc_val}\n"
                f"Traceback:\n{''.join(traceback.format_exception(exc_type, exc_val, exc_tb))}"
            )
        return False  # Don't suppress the exception
```

#### Task 2.2: Instrument P0 Files (Logging Mode)

For each bare `except:` in P0 files, wrap to log the exception type:

```python
# BEFORE:
try:
    risky_operation()
except:
    fallback()

# AFTER (Step 2.1 - Logging Mode):
try:
    risky_operation()
except Exception as _e:
    import logging
    logging.getLogger(__name__).debug(
        f"Bare except caught {type(_e).__name__}: {_e} in {__file__}:{inspect.currentframe().f_lineno}"
    )
    fallback()
```

- [ ] Instrument `functions/CheckMarketOpen.py` (3 locations)
- [ ] Instrument `PyTAAA.py` (4 locations)
- [ ] Instrument `run_pytaaa.py` (4 locations)

#### Task 2.3: Run and Collect Exception Logs

- [ ] Run all 7 end-to-end commands with logging enabled
- [ ] Collect and analyze exception types caught
- [ ] Document findings in `.refactor_baseline/exception_types_observed.md`

Example expected findings:
```markdown
## functions/CheckMarketOpen.py:24
- Observed: `AttributeError` (regex didn't match), `urllib.error.URLError` (network)
- Recommendation: `except (AttributeError, urllib.error.URLError)`

## run_pytaaa.py:27
- Observed: `FileNotFoundError` (directory doesn't exist)
- Recommendation: `except FileNotFoundError`
```

#### Task 2.4: Replace with Specific Exceptions (Fix Mode)

Based on observed exception types, replace bare `except:`:

```python
# AFTER (Step 2.2 - Fix Mode):
import urllib.error

try:
    risky_operation()
except (AttributeError, urllib.error.URLError) as e:
    logger.debug(f"Market status check failed: {e}")
    status = 'no Market Open/Closed status available'
```

- [ ] Update `functions/CheckMarketOpen.py` with specific exceptions
- [ ] Update `PyTAAA.py` with specific exceptions
- [ ] Update `run_pytaaa.py` with specific exceptions

#### Task 2.5: Write Tests

Create `tests/test_phase2_exceptions.py`:

```python
"""Tests for Phase 2 exception handling changes."""

import pytest
import urllib.error
from unittest.mock import patch, MagicMock

class TestCheckMarketOpenExceptions:
    """Test that CheckMarketOpen handles exceptions properly."""
    
    def test_get_market_open_or_closed_handles_url_error(self):
        """Verify URLError is handled gracefully."""
        from functions.CheckMarketOpen import get_MarketOpenOrClosed
        
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError("Network error")
            
            # Should not raise, should return default status
            result = get_MarketOpenOrClosed()
            assert result == 'no Market Open/Closed status available'
    
    def test_get_market_open_or_closed_handles_attribute_error(self):
        """Verify AttributeError (regex fail) is handled gracefully."""
        from functions.CheckMarketOpen import get_MarketOpenOrClosed
        
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b'invalid html without expected pattern'
            mock_urlopen.return_value = mock_response
            
            result = get_MarketOpenOrClosed()
            assert result == 'no Market Open/Closed status available'
```

#### Task 2.6: Validation

- [ ] Run pytest: `uv run pytest tests/test_phase2_exceptions.py -v`
- [ ] Run all 7 end-to-end commands, compare outputs to baseline
- [ ] Verify `.params` file checksums match baseline
- [ ] Check that no new exceptions propagate (previously caught by bare `except:`)

#### Task 2.7: Commit

```bash
git add -A
git commit -m "Phase 2: Replace bare except clauses with specific exceptions

- Add exception logging decorator for debugging
- Replace bare except: in CheckMarketOpen.py with (AttributeError, URLError)
- Replace bare except: in PyTAAA.py with specific exceptions
- Replace bare except: in run_pytaaa.py with specific exceptions
- Add tests/test_phase2_exceptions.py

Observed exception types documented in exception_types_observed.md
All end-to-end tests pass with identical outputs."
```

---

## Phase 3: JSON Migration — Complete Legacy-to-JSON Transition

**Complexity:** Medium  
**Risk:** Medium (configuration system changes)  
**Estimated Time:** 3-4 AI sessions  
**AI Model Recommendation:** Claude 4.5 Sonnet (complex reasoning about config systems)

### 3.1 Goals

1. Migrate remaining legacy `.params` usage to JSON
2. Remove legacy `GetParams()` function and related functions
3. Update all entry points to use JSON exclusively
4. Deprecate `PyTAAA.py` in favor of `pytaaa_main.py`

### 3.2 Files to Modify

| File | Changes |
|------|---------|
| `functions/GetParams.py` | Remove legacy functions, keep only JSON-based |
| `PyTAAA.py` | Add deprecation warning, redirect to `pytaaa_main.py` |
| `daily_abacus_update.py` | Verify JSON-only operation |
| `scheduler.py` | Update to work with JSON-based entry points |

### 3.3 Detailed Checklist

#### Task 3.1: Audit Legacy Usage

- [ ] Search for all calls to `GetParams()`, `GetHoldings()`, `GetStatus()`, `PutStatus()`, `GetFTPParams()`
- [ ] Identify which entry points still use legacy functions
- [ ] Document in `.refactor_baseline/legacy_usage_audit.md`

#### Task 3.2: Migrate Remaining Callers

If any callers still use legacy functions:

- [ ] Update caller to use `get_json_params()`, `get_holdings()`, etc.
- [ ] Add JSON configuration if missing
- [ ] Test the migrated entry point

#### Task 3.3: Remove Legacy Functions

From `functions/GetParams.py`, remove:

- [ ] `GetParams()` (legacy, ~lines 551-650)
- [ ] `GetHoldings()` (legacy, ~lines 650-750)
- [ ] `GetStatus()` (legacy, ~lines 750-850)
- [ ] `PutStatus()` (legacy, ~lines 850-950)
- [ ] `GetFTPParams()` (legacy)

Keep:
- `get_json_params()`
- `get_holdings()`
- `get_status()`
- `put_status()`
- `get_json_ftp_params()`
- `get_symbols_file()`
- `get_performance_store()`
- `get_webpage_store()`

#### Task 3.4: Deprecate PyTAAA.py

Update `PyTAAA.py`:

```python
#!/usr/bin/env python3
"""Legacy entry point for PyTAAA.

DEPRECATED: Use pytaaa_main.py instead.

This file is kept for backward compatibility but will be removed
in a future release. Please migrate to the JSON-based entry point.
"""

import warnings
import sys

def main():
    warnings.warn(
        "PyTAAA.py is deprecated. Use 'uv run python pytaaa_main.py --json config.json' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Redirect to modern entry point
    from pytaaa_main import main as modern_main
    sys.argv = ['pytaaa_main.py', '--json', 'pytaaa_generic.json']
    modern_main()

if __name__ == "__main__":
    main()
```

#### Task 3.5: Write Tests

Create `tests/test_phase3_json_migration.py`:

```python
"""Tests for Phase 3 JSON migration."""

import pytest
import json
import tempfile
import os

class TestJsonParamsOnly:
    """Verify only JSON-based config functions exist."""
    
    def test_no_legacy_getparams_function(self):
        """Legacy GetParams() function should not exist."""
        from functions import GetParams
        
        # Should NOT have the legacy function
        assert not hasattr(GetParams, 'GetParams')
        
        # Should have the modern function
        assert hasattr(GetParams, 'get_json_params')
    
    def test_get_json_params_works(self):
        """Modern JSON params function works correctly."""
        from functions.GetParams import get_json_params
        
        # Create a test config
        test_config = {
            "stockList": "Naz100",
            "Valuation": {
                "performance_store": "/tmp/test",
                "webpage": "/tmp/web"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            temp_path = f.name
        
        try:
            params = get_json_params(temp_path)
            assert params['stockList'] == 'Naz100'
        finally:
            os.unlink(temp_path)
```

#### Task 3.6: Validation

- [ ] Run pytest: `uv run pytest tests/test_phase3_json_migration.py -v`
- [ ] Run all 7 end-to-end commands, compare outputs to baseline
- [ ] Verify `.params` file checksums match baseline
- [ ] Test that `PyTAAA.py` shows deprecation warning but still works

#### Task 3.7: Commit

```bash
git add -A
git commit -m "Phase 3: Complete JSON migration, deprecate legacy config

- Remove legacy GetParams(), GetHoldings(), GetStatus(), PutStatus()
- Remove legacy GetFTPParams() function
- Add deprecation warning to PyTAAA.py
- Redirect PyTAAA.py to pytaaa_main.py
- Add tests/test_phase3_json_migration.py

All end-to-end tests pass with identical outputs.
PyTAAA.py shows deprecation warning but remains functional."
```

---

## Phase 4: Testability — Separate Computation from I/O

**Complexity:** High  
**Risk:** High (architectural changes)  
**Estimated Time:** 5-6 AI sessions  
**AI Model Recommendation:** Claude 4.5 Sonnet or o1 (complex refactoring)

### 4.1 Goals

1. Extract data loading from computation functions
2. Extract plot generation from computation functions
3. Extract file writing from computation functions
4. Enable unit testing of core logic without file system access

### 4.2 Primary Target

`functions/PortfolioPerformanceCalcs.py` - the main orchestration function

### 4.3 Detailed Checklist

#### Task 4.1: Analyze Current Structure

Read and document `PortfolioPerformanceCalcs()`:

- [ ] Identify all file I/O operations
- [ ] Identify all computation steps
- [ ] Identify all plot generation calls
- [ ] Document data flow in `.refactor_baseline/portfolio_perf_analysis.md`

#### Task 4.2: Extract Data Loading

Create `functions/data_loaders.py`:

```python
"""Data loading functions separated from computation."""

from typing import Tuple, List
import numpy as np
from functions.UpdateSymbols_inHDF5 import loadQuotes_fromHDF

def load_quotes_for_analysis(
    symbols_file: str,
    params: dict
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Load and prepare quote data for analysis.
    
    Args:
        symbols_file: Path to symbols file
        params: Configuration parameters
        
    Returns:
        Tuple of (adjClose_array, symbols_list, date_array)
    """
    # Extracted from PortfolioPerformanceCalcs
    ...
```

#### Task 4.3: Extract Computation Core

Refactor `PortfolioPerformanceCalcs()` into:

```python
def compute_portfolio_metrics(
    adjClose: np.ndarray,
    symbols: List[str],
    datearray: np.ndarray,
    params: dict
) -> dict:
    """Compute portfolio metrics from loaded data.
    
    Pure computation function - no file I/O.
    
    Args:
        adjClose: 2D array of adjusted close prices [symbols x dates]
        symbols: List of ticker symbols
        datearray: Array of dates
        params: Configuration parameters
        
    Returns:
        Dictionary containing:
        - rankings: Stock rankings
        - weights: Portfolio weights
        - signals: Trading signals
        - backtest_results: Backtest data
    """
    ...

def PortfolioPerformanceCalcs(json_fn: str) -> dict:
    """Main entry point - orchestrates loading, computation, and output.
    
    This is the original function signature for backward compatibility.
    Internally delegates to pure functions.
    """
    # Load data
    adjClose, symbols, datearray = load_quotes_for_analysis(...)
    
    # Compute
    results = compute_portfolio_metrics(adjClose, symbols, datearray, params)
    
    # Generate outputs
    generate_plots(results, params)
    write_output_files(results, params)
    
    return results
```

#### Task 4.4: Write Comprehensive Tests

Create `tests/test_phase4_computation.py`:

```python
"""Tests for Phase 4 computation/I/O separation."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

class TestComputePortfolioMetrics:
    """Test pure computation function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_symbols = 10
        n_days = 252
        
        # Generate synthetic price data with trend
        dates = np.arange(n_days)
        adjClose = np.zeros((n_symbols, n_days))
        
        for i in range(n_symbols):
            trend = 1 + 0.0001 * dates + 0.001 * np.random.randn(n_days)
            adjClose[i] = 100 * np.cumprod(trend)
        
        symbols = [f'STOCK{i}' for i in range(n_symbols)]
        
        return adjClose, symbols, dates
    
    def test_compute_portfolio_metrics_returns_expected_keys(self, sample_data):
        """Verify computation returns expected structure."""
        from functions.PortfolioPerformanceCalcs import compute_portfolio_metrics
        
        adjClose, symbols, dates = sample_data
        params = {
            'numberStocksTraded': 5,
            'monthsToHold': 1,
            # ... other required params
        }
        
        results = compute_portfolio_metrics(adjClose, symbols, dates, params)
        
        assert 'rankings' in results
        assert 'weights' in results
        assert 'signals' in results
    
    def test_compute_portfolio_metrics_deterministic(self, sample_data):
        """Verify computation is deterministic (same input = same output)."""
        from functions.PortfolioPerformanceCalcs import compute_portfolio_metrics
        
        adjClose, symbols, dates = sample_data
        params = {'numberStocksTraded': 5, 'monthsToHold': 1}
        
        results1 = compute_portfolio_metrics(adjClose, symbols, dates, params)
        results2 = compute_portfolio_metrics(adjClose, symbols, dates, params)
        
        np.testing.assert_array_equal(results1['rankings'], results2['rankings'])
```

#### Task 4.5: Validation

- [ ] Run pytest: `uv run pytest tests/test_phase4_computation.py -v`
- [ ] Run all 7 end-to-end commands, compare outputs to baseline
- [ ] Verify `.params` file checksums match baseline
- [ ] Verify new unit tests pass without file system access

#### Task 4.6: Commit

```bash
git add -A
git commit -m "Phase 4: Separate computation from I/O

- Extract data loading to functions/data_loaders.py
- Create pure compute_portfolio_metrics() function
- Refactor PortfolioPerformanceCalcs() as orchestrator
- Add comprehensive unit tests for computation logic
- Enable testing without file system access

All end-to-end tests pass with identical outputs.
New unit tests: 15 tests, all passing."
```

---

## Phase 5: Modularity — Break Up TAfunctions.py

**Complexity:** Very High  
**Risk:** High (major structural changes)  
**Estimated Time:** 6-8 AI sessions  
**AI Model Recommendation:** o1 or Claude 4.5 Sonnet (most capable for complex refactoring)

### 5.1 Goals

1. Split 4,100+ line `TAfunctions.py` into focused modules
2. Maintain backward compatibility via re-exports
3. Preserve exact function behavior
4. Enable independent testing of submodules

### 5.2 Proposed Module Structure

```
functions/
├── TAfunctions.py              # Re-export module (backward compat)
├── moving_averages.py          # SMA, HMA implementations
├── channels.py                 # dpgchannel, percentileChannel
├── trend_analysis.py           # Trend detection and fitting
├── signal_generation.py        # computeSignal2D
├── ranking.py                  # sharpeWeightedRank_2D, etc.
├── data_cleaning.py            # interpolate, cleantobeginning, etc.
└── rolling_metrics.py          # move_sharpe_2D, move_martin_2D, etc.
```

### 5.3 Detailed Checklist

#### Task 5.1: Create Module Structure

- [ ] Create `functions/moving_averages.py`
- [ ] Create `functions/channels.py`
- [ ] Create `functions/trend_analysis.py`
- [ ] Create `functions/signal_generation.py`
- [ ] Create `functions/ranking.py`
- [ ] Create `functions/data_cleaning.py`
- [ ] Create `functions/rolling_metrics.py`

#### Task 5.2: Extract Functions (One Module at a Time)

For each module:

1. Copy relevant functions from `TAfunctions.py`
2. Update imports within the module
3. Add comprehensive docstrings
4. Create tests for the module
5. Update `TAfunctions.py` to import from new module

Example for `functions/moving_averages.py`:

```python
"""Moving average implementations."""

import numpy as np
from typing import Union

def SMA(input_values: np.ndarray, periods: int) -> np.ndarray:
    """Calculate Simple Moving Average.
    
    Args:
        input_values: Input price array
        periods: Number of periods for averaging
        
    Returns:
        Array of SMA values
    """
    ...

def hma(input_values: np.ndarray, periods: int) -> np.ndarray:
    """Calculate Hull Moving Average.
    
    The Hull Moving Average reduces lag while maintaining smoothness
    by using weighted moving averages with square root of period.
    
    Args:
        input_values: Input price array
        periods: Number of periods
        
    Returns:
        Array of HMA values
    """
    ...
```

Update `functions/TAfunctions.py`:

```python
"""Technical analysis functions (backward compatibility module).

All functions are re-exported from focused submodules.
New code should import directly from submodules.
"""

# Re-exports for backward compatibility
from functions.moving_averages import SMA, SMA_2D, hma, hma_pd, SMS
from functions.channels import dpgchannel, dpgchannel_2D, percentileChannel_2D
from functions.trend_analysis import recentTrendAndStdDevs, recentSharpeWithAndWithoutGap
from functions.signal_generation import computeSignal2D
from functions.ranking import sharpeWeightedRank_2D, MAA_WeightedRank_2D, UnWeightedRank_2D
from functions.data_cleaning import interpolate, cleantobeginning, cleantoend, cleanspikes
from functions.rolling_metrics import move_sharpe_2D, move_martin_2D

__all__ = [
    'SMA', 'SMA_2D', 'hma', 'hma_pd', 'SMS',
    'dpgchannel', 'dpgchannel_2D', 'percentileChannel_2D',
    'recentTrendAndStdDevs', 'recentSharpeWithAndWithoutGap',
    'computeSignal2D',
    'sharpeWeightedRank_2D', 'MAA_WeightedRank_2D', 'UnWeightedRank_2D',
    'interpolate', 'cleantobeginning', 'cleantoend', 'cleanspikes',
    'move_sharpe_2D', 'move_martin_2D',
]
```

#### Task 5.3: Write Tests for Each Module

Create `tests/test_moving_averages.py`:

```python
"""Tests for moving_averages module."""

import pytest
import numpy as np
from functions.moving_averages import SMA, hma

class TestSMA:
    """Test Simple Moving Average calculation."""
    
    def test_sma_basic(self):
        """Test SMA with simple input."""
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = SMA(prices, 3)
        
        # Third value should be average of first 3
        assert result[2] == pytest.approx(2.0)
        # Last value should be average of last 3
        assert result[4] == pytest.approx(4.0)
    
    def test_sma_2d(self):
        """Test 2D SMA calculation."""
        prices = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0]
        ])
        result = SMA_2D(prices, 3)
        
        assert result.shape == prices.shape
        assert result[0, 2] == pytest.approx(2.0)
        assert result[1, 2] == pytest.approx(4.0)
```

Repeat for each module.

#### Task 5.4: Validation

- [ ] Run all module-specific tests
- [ ] Run all 7 end-to-end commands, compare outputs to baseline
- [ ] Verify `.params` file checksums match baseline
- [ ] Verify backward compatibility (imports from TAfunctions still work)

#### Task 5.5: Commit

```bash
git add -A
git commit -m "Phase 5: Break up TAfunctions.py into focused modules

- Create functions/moving_averages.py (SMA, HMA)
- Create functions/channels.py (dpgchannel, percentileChannel)
- Create functions/trend_analysis.py (trend detection)
- Create functions/signal_generation.py (computeSignal2D)
- Create functions/ranking.py (sharpeWeightedRank_2D)
- Create functions/data_cleaning.py (interpolate, clean functions)
- Create functions/rolling_metrics.py (Sharpe, Martin ratios)
- TAfunctions.py now re-exports from submodules for backward compat
- Add comprehensive tests for each module

All end-to-end tests pass with identical outputs.
Backward compatibility verified."
```

---

## Phase 6: Polish — Type Annotations, Logging, CLI Standardization

**Complexity:** Medium  
**Risk:** Low (additive changes)  
**Estimated Time:** 4-5 AI sessions  
**AI Model Recommendation:** Kimi K2.5 (good at systematic additions)

### 6.1 Goals

1. Add type annotations to all public functions
2. Migrate `print()` statements to logging
3. Standardize CLI entry points on Click
4. Improve documentation consistency

### 6.2 Detailed Checklist

#### Task 6.1: Add Type Annotations

- [ ] Add types to `functions/moving_averages.py`
- [ ] Add types to `functions/channels.py`
- [ ] Add types to `functions/signal_generation.py`
- [ ] Add types to `functions/ranking.py`
- [ ] Add types to entry points

Example:

```python
from typing import List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray

def computeSignal2D(
    adjClose: NDArray[np.float64],
    symbols: List[str],
    datearray: NDArray[np.datetime64],
    params: dict
) -> Tuple[NDArray[np.int8], NDArray[np.float64]]:
    ...
```

#### Task 6.2: Migrate to Logging

- [ ] Replace `print()` with `logger.debug()` in core modules
- [ ] Keep `print()` only for CLI output in entry points
- [ ] Update `logger_config.py` if needed

#### Task 6.3: Standardize CLI

- [ ] Migrate `daily_abacus_update.py` from argparse to Click
- [ ] Add consistent `--verbose` flag to all entry points
- [ ] Consider Click group for unified CLI

#### Task 6.4: Validation

- [ ] Run mypy if available: `uv run mypy functions/`
- [ ] Run all 7 end-to-end commands
- [ ] Verify log files are created properly

#### Task 6.5: Commit

```bash
git add -A
git commit -m "Phase 6: Add type annotations, logging, CLI standardization

- Add type annotations to all public functions in refactored modules
- Migrate print() to logger.debug() in core computation
- Standardize CLI entry points on Click
- Add --verbose flag consistently

All end-to-end tests pass with identical outputs."
```

---

## Review Process

### Constructive Critic Review (Draft)

Before human review, the plan undergoes AI constructive critic review:

**Reviewer Persona:** Experienced software architect, skeptical of AI-generated plans, focused on:
- Missing edge cases
- Unrealistic timelines
- Unnecessary complexity
- Risk mitigation gaps

**Questions for Critic:**

1. Are the phase boundaries appropriate? Should any phases be combined or split?
2. Is the test coverage adequate for each phase?
3. Are there any "AI slop" patterns (unnecessary abstractions, over-engineering)?
4. What risks are not adequately addressed?
5. Are the AI model recommendations appropriate for each phase?

### Human-in-the-Loop Review

After AI critic review, human review focuses on:

1. **Business Logic Preservation:** Will the refactoring change any trading behavior?
2. **Operational Impact:** Will daily/monthly workflows be disrupted?
3. **Rollback Plan:** Can we quickly revert if issues arise?
4. **Priority Adjustment:** Should any phases be reordered?

### Final Approval Checklist

- [ ] Constructive critic review completed
- [ ] Human review completed
- [ ] All concerns addressed
- [ ] Go/no-go decision for Phase 1

---

## AI Model Recommendations by Phase

| Phase | Recommended Model | Rationale | Cost Estimate |
|-------|-------------------|-----------|---------------|
| 1 | Kimi K2.5 | Pattern matching for dead code, docstring generation | $ |
| 2 | Claude 4.5 Sonnet | Exception hierarchy reasoning, safety analysis | $$ |
| 3 | Claude 4.5 Sonnet | Complex config system migration | $$ |
| 4 | o1 or Claude 4.5 Sonnet | Architectural refactoring, separation of concerns | $$$ |
| 5 | o1 | Most complex refactoring, module decomposition | $$$ |
| 6 | Kimi K2.5 | Systematic type annotation addition | $ |

**Cost Legend:**
- $ = Low cost (~$5-15 per session)
- $$ = Medium cost (~$15-30 per session)
- $$$ = Higher cost (~$30-60 per session)

**Success Probability:**
- All models estimated at >95% success for their assigned phases
- o1 recommended for Phases 4-5 due to complex reasoning requirements
- Kimi K2.5 sufficient for pattern-based tasks (Phases 1, 6)

---

## Rollback Procedures

### Per-Phase Rollback

```bash
# To rollback a specific phase
git log --oneline  # Find the commit before the phase
git revert <phase-commit-hash> --no-edit

# Verify rollback
uv run pytest tests/
# Run end-to-end validation
```

### Full Rollback

```bash
# To completely abandon refactoring
git checkout main
git branch -D refactor/modernize

# Or keep branch but switch back
git checkout main
```

---

## Success Criteria

### Overall Success

The refactoring is successful when:

1. All phases completed and committed
2. All new tests pass
3. All 7 end-to-end commands produce identical outputs to baseline
4. `.params` file checksums match baseline
5. Code coverage increased from baseline
6. No new warnings or errors in logs

### Phase Success Criteria

Each phase must:

1. Have all checklist items completed
2. Pass its dedicated test file
3. Pass all 7 end-to-end validation commands
4. Have a clean git commit
5. Be approved in human review before next phase begins

---

## Appendix A: End-to-End Validation Commands

```bash
# Command 1: naz100_pine
uv run python pytaaa_main.py --json /Users/donaldpg/pyTAAA_data_static/naz100_pine/pytaaa_naz100_pine.json 2>&1 | tee .refactor_baseline/after/pytaaa_naz100_pine.log

# Command 2: naz100_hma
uv run python pytaaa_main.py --json /Users/donaldpg/pyTAAA_data_static/naz100_hma/pytaaa_naz100_hma.json 2>&1 | tee .refactor_baseline/after/pytaaa_naz100_hma.log

# Command 3: naz100_pi
uv run python pytaaa_main.py --json /Users/donaldpg/pyTAAA_data_static/naz100_pi/pytaaa_naz100_pi.json 2>&1 | tee .refactor_baseline/after/pytaaa_naz100_pi.log

# Command 4: sp500_hma
uv run python pytaaa_main.py --json /Users/donaldpg/pyTAAA_data_static/sp500_hma/pytaaa_sp500_hma.json 2>&1 | tee .refactor_baseline/after/pytaaa_sp500_hma.log

# Command 5: sp500_pine
uv run python pytaaa_main.py --json /Users/donaldpg/pyTAAA_data_static/sp500_pine/pytaaa_sp500_pine.json 2>&1 | tee .refactor_baseline/after/pytaaa_sp500_pine.log

# Command 6: recommend_model
uv run python recommend_model.py --json /Users/donaldpg/pyTAAA_data_static/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json 2>&1 | tee .refactor_baseline/after/pytaaa_abacus_recommendation.log

# Command 7: daily_abacus_update
uv run python daily_abacus_update.py --json /Users/donaldpg/pyTAAA_data_static/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json --verbose 2>&1 | tee .refactor_baseline/after/pytaaa_abacus_daily.log

# Compare outputs
diff -r .refactor_baseline/before .refactor_baseline/after
```

---

## Appendix B: File Inventory

### Files to be Modified by Phase

**Phase 1:**
- `functions/CheckMarketOpen.py`
- `functions/TAfunctions.py`
- `functions/quotes_for_list_adjClose.py`
- `functions/GetParams.py`
- `re-generateHDF5.py` (delete)

**Phase 2:**
- `functions/CheckMarketOpen.py`
- `PyTAAA.py`
- `run_pytaaa.py`
- `functions/TAfunctions.py`
- `functions/MakeValuePlot.py`
- `functions/WriteWebPage_pi.py`
- Plus ~10 more files

**Phase 3:**
- `functions/GetParams.py`
- `PyTAAA.py`
- `daily_abacus_update.py`
- `scheduler.py`

**Phase 4:**
- `functions/PortfolioPerformanceCalcs.py`
- New: `functions/data_loaders.py`

**Phase 5:**
- `functions/TAfunctions.py` (major refactor)
- New: `functions/moving_averages.py`
- New: `functions/channels.py`
- New: `functions/trend_analysis.py`
- New: `functions/signal_generation.py`
- New: `functions/ranking.py`
- New: `functions/data_cleaning.py`
- New: `functions/rolling_metrics.py`

**Phase 6:**
- All refactored modules (type annotations)
- Entry points (CLI standardization)

---

## References

1. [PEP 8 — Style Guide for Python Code](https://peps.python.org/pep-0008/)
2. [PEP 257 — Docstring Conventions](https://peps.python.org/pep-0257/)
3. [PEP 484 — Type Hints](https://peps.python.org/pep-0484/)
4. [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
5. [Martin Fowler - Refactoring](https://refactoring.com/)
6. [`docs/RECOMMENDATIONS.md`](../docs/RECOMMENDATIONS.md) - Original analysis

---

**Next Steps:**

1. Review this plan (constructive critic + human)
2. Address feedback
3. Create `refactor/modernize` branch
4. Execute Pre-Flight Checklist
5. Begin Phase 1
