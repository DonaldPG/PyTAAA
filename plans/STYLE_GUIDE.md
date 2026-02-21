# PyTAAA Refactoring Style Guide

**Purpose:** Coding standards for AI assistants working on PyTAAA refactoring  
**Version:** 1.0  
**Applies to:** All phases of the refactoring plan

---

## 1. General Principles

### 1.1 Functional Equivalence is Paramount

- **Never change algorithmic behavior** during refactoring
- **Never change function signatures** unless explicitly required by the phase
- **Never optimize** unless the phase specifically calls for it
- When in doubt, preserve the existing code exactly

### 1.2 Explicit is Better than Implicit

```python
# GOOD: Explicit exception types
except (FileNotFoundError, PermissionError) as e:
    logger.warning(f"File access failed: {e}")

# BAD: Bare except or generic Exception
except:  # Never do this
    pass

except Exception as e:  # Only when truly generic catch is needed
    logger.error(f"Unexpected error: {e}")
```

### 1.3 Preserve Existing Patterns

When modifying a file, observe and follow the existing patterns:
- If the file uses single quotes, use single quotes
- If the file uses 4-space indentation, use 4 spaces
- If the file has specific import ordering, follow it

---

## 2. Docstrings

### 2.1 Google Style Format

```python
def function_name(param1: type1, param2: type2) -> return_type:
    """Short one-line description.
    
    Longer description if needed. Can span multiple lines and
    include additional context about the function's purpose.
    
    Args:
        param1: Description of param1. Can span multiple
            lines with proper indentation.
        param2: Description of param2.
        
    Returns:
        Description of return value and its format.
        
    Raises:
        FileNotFoundError: When the input file doesn't exist
        ValueError: When param1 is invalid
        
    Example:
        >>> result = function_name("input", 42)
        >>> print(result)
        'expected output'
    """
```

### 2.2 Required Docstring Sections

| Section | Required For | Optional For |
|---------|--------------|--------------|
| Short description | All public functions | Private functions (_prefix) |
| Args | Functions with parameters | Parameterless functions |
| Returns | Functions with return values | None-returning functions |
| Raises | Functions that raise exceptions | Exception-free functions |
| Example | Complex functions | Simple getters/setters |

### 2.3 Module Docstrings

Every module should have a module-level docstring:

```python
"""Module short description.

Longer description of the module's purpose and contents.

Functions:
    function_name: Brief description
    another_function: Brief description

Classes:
    ClassName: Brief description

References:
    [1] Author, "Title", Publication, Year
"""
```

---

## 3. Exception Handling

### 3.1 Exception Hierarchy (Most to Least Specific)

```python
# GOOD: Specific to general
try:
    process_file(filename)
except FileNotFoundError:
    logger.error(f"File not found: {filename}")
    raise
except PermissionError:
    logger.error(f"Permission denied: {filename}")
    return default_value
except OSError as e:
    logger.error(f"OS error processing {filename}: {e}")
    raise
```

### 3.2 Logging Exceptions

```python
# GOOD: Log with context
except SpecificError as e:
    logger.warning(f"Operation failed for {symbol} on {date}: {e}")
    fallback_value = compute_fallback()

# BAD: Silent except
except SpecificError:
    pass  # Never do this

# BAD: Generic log message
except SpecificError as e:
    print(e)  # Use logger, not print
```

### 3.3 Phase 2 Exception Handling Pattern

During Phase 2, use this specific pattern:

```python
# STEP 1: Logging mode (temporary)
try:
    risky_operation()
except Exception as _e:
    import logging, inspect
    logging.getLogger(__name__).debug(
        f"PHASE2_DEBUG: Caught {type(_e).__name__}: {_e} "
        f"at {__file__}:{inspect.currentframe().f_lineno}"
    )
    fallback_code()

# STEP 2: Fix mode (final)
try:
    risky_operation()
except (ExpectedError1, ExpectedError2) as e:
    logger.debug(f"Expected error: {e}")
    fallback_code()
except Exception as e:
    # Safety fallback for unobserved exceptions
    logger.warning(f"Unexpected {type(e).__name__}: {e}")
    fallback_code()
```

---

## 4. Type Annotations

### 4.1 Basic Type Annotations

```python
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from numpy.typing import NDArray

def process_data(
    prices: NDArray[np.float64],
    symbols: List[str],
    config: Dict[str, Union[str, int, float]]
) -> Tuple[NDArray[np.float64], Dict[str, float]]:
    """Process price data and return results."""
    ...
```

### 4.2 Complex Types

```python
from typing import TypedDict, Callable

class Config(TypedDict):
    stockList: str
    numberStocksTraded: int
    monthsToHold: int

SignalMethod = Callable[[NDArray[np.float64]], NDArray[np.int8]]

def apply_signal(
    data: NDArray[np.float64],
    config: Config,
    method: SignalMethod
) -> NDArray[np.int8]:
    ...
```

### 4.3 When to Use Type Annotations

| Location | Required? | Notes |
|----------|-----------|-------|
| Public functions in new modules | Yes | All parameters and return values |
| Public functions in modified modules | Yes | Add when modifying |
| Private functions (_prefix) | Optional | Recommended for clarity |
| Existing unmodified code | No | Don't add just for types |
| Tests | Optional | Can help clarify test data |

---

## 5. Imports

### 5.1 Import Order

```python
"""Module docstring."""

# 1. Standard library imports
import os
import sys
from typing import List, Dict

# 2. Third-party imports
import numpy as np
import pandas as pd

# 3. Local application imports
from functions.TAfunctions import computeSignal2D
from functions.GetParams import get_json_params
```

### 5.2 Import Style

```python
# GOOD: Explicit imports
from typing import List, Dict, Optional
from functions.TAfunctions import computeSignal2D, sharpeWeightedRank_2D

# AVOID: Wildcard imports (unless specifically needed for backward compat)
from functions.TAfunctions import *  # Only in re-export modules

# GOOD: Conditional imports with clear purpose
try:
    from numba import jit
except ImportError:
    jit = lambda f: f  # Fallback decorator
```

---

## 6. Logging

### 6.1 Logger Setup

```python
import logging

logger = logging.getLogger(__name__)

# Use logger, not print
def my_function():
    logger.debug("Detailed diagnostic info")
    logger.info("General information")
    logger.warning("Warning message")
    logger.error("Error message")
```

### 6.2 When to Use Each Level

| Level | Use For | Example |
|-------|---------|---------|
| DEBUG | Detailed diagnostic | "Processing symbol AAPL, day 42" |
| INFO | General progress | "Loaded 500 symbols from HDF5" |
| WARNING | Recoverable issues | "Missing data for symbol XYZ, using interpolation" |
| ERROR | Failures | "Failed to download quotes for symbol ABC" |
| CRITICAL | System failures | "Cannot access data store" |

### 6.3 Migration from print()

When migrating existing code:

```python
# BEFORE:
print("Processing symbol:", symbol)

# AFTER:
logger.debug("Processing symbol: %s", symbol)

# BEFORE:
print("Error:", str(e))

# AFTER:
logger.error("Operation failed: %s", e)
```

---

## 7. Testing

### 7.1 Test Structure

```python
"""Tests for module_name."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from functions.module_name import function_name


class TestFunctionName:
    """Test cases for function_name."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for tests."""
        return {
            'prices': np.array([1.0, 2.0, 3.0]),
            'symbols': ['AAPL', 'GOOGL']
        }
    
    def test_basic_functionality(self, sample_data):
        """Test basic operation with valid input."""
        result = function_name(sample_data['prices'])
        assert result is not None
        
    def test_edge_case_empty_input(self):
        """Test behavior with empty input."""
        result = function_name(np.array([]))
        assert len(result) == 0
        
    def test_invalid_input_raises(self):
        """Test that invalid input raises appropriate exception."""
        with pytest.raises(ValueError, match="expected error message"):
            function_name(None)
```

### 7.2 Test Naming

- Test class: `Test<FunctionName>` or `Test<Feature>`
- Test method: `test_<scenario>_<expected_behavior>`
- Test fixture: Descriptive noun phrase

### 7.3 Assertions

```python
# GOOD: Specific assertions
assert result == expected
assert len(items) == 5
assert isinstance(obj, ExpectedType)
np.testing.assert_array_equal(arr1, arr2)
np.testing.assert_allclose(arr1, arr2, rtol=1e-5)

# GOOD: pytest approx for floats
assert value == pytest.approx(3.14159, rel=1e-5)

# GOOD: Exception testing
with pytest.raises(ValueError, match="specific message"):
    function(invalid_input)
```

---

## 8. NumPy and Array Operations

### 8.1 Array Creation

```python
# GOOD: Explicit dtypes
prices = np.zeros((n_symbols, n_days), dtype=np.float64)
signals = np.ones((n_symbols,), dtype=np.int8)

# GOOD: Use numpy random with seed for reproducibility
np.random.seed(42)
returns = np.random.randn(n_days)
```

### 8.2 Array Operations

```python
# GOOD: Vectorized operations
returns = (prices[:, 1:] - prices[:, :-1]) / prices[:, :-1]

# AVOID: Python loops when vectorized works
for i in range(n_symbols):  # Slow
    for j in range(n_days):
        returns[i, j] = ...
```

### 8.3 Handling NaN Values

```python
# GOOD: Explicit NaN handling
if np.isnan(value):
    fallback_value = interpolate_missing(...)

# GOOD: Array-wise NaN handling
valid_mask = ~np.isnan(prices)
mean_price = np.nanmean(prices, axis=1)
```

---

## 9. Git Commits

### 9.1 Commit Message Format

```
Phase N: Brief description

- Detailed change 1
- Detailed change 2
- Detailed change 3

All end-to-end tests pass with identical outputs.
[Optional: New tests: X tests, all passing]
```

### 9.2 Commit Scope

- One phase per commit (or sub-phase if split)
- Include all related test changes in the same commit
- Don't mix phases in a single commit

---

## 10. Performance Considerations

### 10.1 Maintain Performance

```python
# GOOD: Preserve existing optimizations
@jit(nopython=True)  # Keep numba decorators
def compute_metric(data):
    ...

# GOOD: Pre-allocate arrays
result = np.empty_like(input_array)
for i in range(n):
    result[i] = compute(i)  # Reuse allocated array
```

### 10.2 Profiling

If performance changes are suspected:

```python
import time

start = time.time()
result = function_to_test()
elapsed = time.time() - start
logger.info(f"Function completed in {elapsed:.3f}s")
```

---

## 11. Backward Compatibility

### 11.1 Function Re-exports

When splitting modules, maintain backward compatibility:

```python
# functions/TAfunctions.py
"""Backward compatibility re-exports."""

from functions.moving_averages import SMA, hma
from functions.ranking import sharpeWeightedRank_2D

__all__ = ['SMA', 'hma', 'sharpeWeightedRank_2D', ...]
```

### 11.2 Deprecation Warnings

When changing behavior:

```python
import warnings

def old_function():
    warnings.warn(
        "old_function is deprecated. Use new_function instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return new_function()
```

---

## 12. Common Patterns in PyTAAA

### 12.1 Configuration Access

```python
# GOOD: Use get_json_params
from functions.GetParams import get_json_params

params = get_json_params(json_fn)
stock_list = params['stockList']
valuation = params['Valuation']
```

### 12.2 Quote Data Loading

```python
# GOOD: Use loadQuotes_fromHDF
from functions.UpdateSymbols_inHDF5 import loadQuotes_fromHDF

adjClose, symbols, datearray = loadQuotes_fromHDF(symbols_file)
```

### 12.3 Signal Computation

```python
# GOOD: Use computeSignal2D with params
from functions.TAfunctions import computeSignal2D

signal = computeSignal2D(
    adjClose, symbols, datearray,
    params, verbose=True
)
```

---

## 13. Prohibited Patterns

The following are **not allowed** during refactoring:

### 13.1 Never Do

```python
# 1. Never use bare except
except:  # FORBIDDEN
    pass

# 2. Never silently ignore exceptions
except Exception as e:
    pass  # FORBIDDEN

# 3. Never change algorithmic logic
# (unless the phase specifically requires it)

# 4. Never remove existing tests
# (only add new ones)

# 5. Never commit without running tests
```

### 13.2 Avoid

```python
# 1. Avoid deeply nested try/except
try:
    try:
        try:
            ...
        except:
            ...
    except:
        ...
except:
    ...

# 2. Avoid mutable default arguments
def bad(items=[]):  # AVOID
    ...

def good(items=None):  # GOOD
    if items is None:
        items = []

# 3. Avoid global state modifications
global_variable = value  # AVOID unless necessary
```

---

## 14. Review Checklist for AI Assistants

Before submitting work:

- [ ] All tests pass (`uv run pytest tests/ -v`)
- [ ] End-to-end validation shows identical outputs
- [ ] No bare `except:` clauses introduced
- [ ] All new functions have docstrings
- [ ] Type annotations added where required
- [ ] No `print()` statements in library code (use `logger`)
- [ ] Git commit message follows format
- [ ] No sensitive data in logs or error messages
- [ ] Backward compatibility maintained
- [ ] Performance is equivalent or better

---

**End of Style Guide**
