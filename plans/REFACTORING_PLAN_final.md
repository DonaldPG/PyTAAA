# PyTAAA Refactoring Plan: Agentic AI Implementation Guide

**Version:** 2.0  
**Last Updated:** February 9, 2026  
**Branch:** `chore/copilot-codebase-refresh`  
**Status:** Ready for Human Review

---

## Executive Summary

This plan provides a phased, test-driven approach to refactoring the PyTAAA codebase based on the analysis in [`docs/RECOMMENDATIONS.md`](../docs/RECOMMENDATIONS.md). The goal is to modernize the codebase while maintaining **exact functional equivalence** — all outputs must remain identical.

### Key Principles

1. **Test-First**: Every phase includes comprehensive tests to validate no behavioral changes
2. **Incremental**: Small, reviewable changes with clear success criteria
3. **Reversible**: Each phase is a git commit; any phase can be reverted independently
4. **Validated**: End-to-end testing against known-good outputs before and after each phase
5. **Static Data**: All testing uses frozen static data to eliminate data drift issues

---

## Critical Update: Static Data for Testing

To ensure reproducible and reliable testing, **all end-to-end (e2e) testing must use the static data copy** located at `/Users/donaldpg/pyTAAA_data_static/` instead of the live `/Users/donaldpg/pyTAAA_data/` directory.

### Static Data Setup

The `pyTAAA_data_static` directory contains:
- Frozen copies of all HDF5 stock history files
- Modified JSON parameter files configured to:
  - Use the static data directory
  - Disable internet data updates
  - Produce deterministic outputs

### Why Static Data?

1. **Eliminates Data Drift**: Live market data changes daily, causing false test failures
2. **Deterministic Results**: Same input always produces same output
3. **Isolated Testing**: Refactoring validation is independent of market conditions
4. **Reproducible Baseline**: Baseline can be re-captured at any time

---

## Phase 0: Static Data Setup (Prerequisite)

**Complexity:** Low  
**Risk:** Low  
**Estimated Time:** 1 AI session  
**Status:** Must be completed before Phase 1

### 0.1 Goals

1. Create frozen copy of live data for deterministic testing
2. Modify JSON configurations to use static data paths
3. Disable internet updates in static configurations
4. Verify static data produces expected outputs
5. Document static data structure for future reference

### 0.2 Verification: Check if Static Data Already Exists

```bash
# Check if static data directory exists
ls -la /Users/donaldpg/pyTAAA_data_static/

# If it exists and is properly configured, skip to Phase 1
# If it doesn't exist or is incomplete, proceed with setup below
```

### 0.3 Create Static Data Copy

```bash
# Create static data directory structure
mkdir -p /Users/donaldpg/pyTAAA_data_static

# Copy entire live data directory (preserving structure)
rsync -av --exclude='*.log' --exclude='*.tmp' \
  /Users/donaldpg/pyTAAA_data/ \
  /Users/donaldpg/pyTAAA_data_static/

# Verify copy completed
du -sh /Users/donaldpg/pyTAAA_data/
du -sh /Users/donaldpg/pyTAAA_data_static/
```

### 0.4 Modify JSON Configurations for Static Testing

For each model configuration, create a static version:

```bash
cd /Users/donaldpg/pyTAAA_data_static

# Models to configure:
# - naz100_pine/pytaaa_naz100_pine.json
# - naz100_hma/pytaaa_naz100_hma.json
# - naz100_pi/pytaaa_naz100_pi.json
# - sp500_hma/pytaaa_sp500_hma.json
# - sp500_pine/pytaaa_sp500_pine.json
# - naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json
```

Create a Python script to modify JSON configs:

```python
#!/usr/bin/env python3
"""Modify JSON configs to use static data paths and disable updates."""

import json
import os
from pathlib import Path

STATIC_ROOT = Path("/Users/donaldpg/pyTAAA_data_static")
MODELS = [
    "naz100_pine/pytaaa_naz100_pine.json",
    "naz100_hma/pytaaa_naz100_hma.json", 
    "naz100_pi/pytaaa_naz100_pi.json",
    "sp500_hma/pytaaa_sp500_hma.json",
    "sp500_pine/pytaaa_sp500_pine.json",
    "naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json"
]

def modify_json_for_static_testing(json_path):
    """Modify a JSON config to use static paths and disable updates."""
    with open(json_path, 'r') as f:
        config = json.load(f)
    
    # Update all path references to use static directory
    if 'data_dir' in config:
        config['data_dir'] = str(STATIC_ROOT / json_path.parent.name)
    
    if 'Valuation' in config:
        if 'performance_store' in config['Valuation']:
            old_path = config['Valuation']['performance_store']
            config['Valuation']['performance_store'] = old_path.replace(
                '/Users/donaldpg/pyTAAA_data/',
                str(STATIC_ROOT) + '/'
            )
        if 'webpage' in config['Valuation']:
            old_path = config['Valuation']['webpage']
            config['Valuation']['webpage'] = old_path.replace(
                '/Users/donaldpg/pyTAAA_data/',
                str(STATIC_ROOT) + '/'
            )
    
    # Disable internet updates
    if 'updateQuotes' in config:
        config['updateQuotes'] = False
    
    if 'downloadNewData' in config:
        config['downloadNewData'] = False
    
    # Write back to file
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Modified {json_path}")

def main():
    """Modify all JSON configs for static testing."""
    for model in MODELS:
        json_path = STATIC_ROOT / model
        if json_path.exists():
            modify_json_for_static_testing(json_path)
        else:
            print(f"⚠ Warning: {json_path} not found")

if __name__ == "__main__":
    main()
```

Run the script:

```bash
cd /Users/donaldpg/PyProjects/worktree2/PyTAAA
python scripts/setup_static_data.py
```

### 0.5 Document Static Data Structure

Create `docs/STATIC_DATA_STRUCTURE.md`:

```markdown
# Static Data Structure

**Purpose:** Frozen copy of production data for deterministic refactoring tests
**Location:** `/Users/donaldpg/pyTAAA_data_static/`
**Created:** [DATE]
**Source:** `/Users/donaldpg/pyTAAA_data/` (snapshot date: [DATE])

## Directory Structure

```
pyTAAA_data_static/
├── naz100_pine/
│   ├── symbols/
│   │   ├── Naz100_Symbols.txt
│   │   └── Naz100_Symbols_.hdf5
│   ├── data_store/
│   │   └── *.params (output files)
│   └── pytaaa_naz100_pine.json (configured for static testing)
├── naz100_hma/ [same structure]
├── naz100_pi/ [same structure]
├── sp500_hma/ [same structure]
├── sp500_pine/ [same structure]
└── naz100_sp500_abacus/ [same structure]
```

## Key Modifications from Live Data

1. **JSON configs updated:** All paths point to `/Users/donaldpg/pyTAAA_data_static/`
2. **Internet updates disabled:** `updateQuotes: false`, `downloadNewData: false`
3. **HDF5 files frozen:** No modifications to stock data files
4. **Output isolation:** `.params` files written to static directory (not live)

## Refresh Procedure

To update static data with newer market data:

```bash
# Backup current static data
mv /Users/donaldpg/pyTAAA_data_static /Users/donaldpg/pyTAAA_data_static.backup

# Re-run Phase 0 setup procedure
# This will capture current market state as new baseline
```

## Critical Notes

- **Never** modify HDF5 files in static directory manually
- **Never** run with internet updates enabled on static configs
- **Always** use static data for refactoring validation
- **Only** use live data for production runs
```

### 0.6 Completion Checklist

- [ ] Static data directory exists at `/Users/donaldpg/pyTAAA_data_static/` ✅ **DONE**
- [ ] All 6 model subdirectories present (naz100_pine, naz100_hma, naz100_pi, sp500_pine, sp500_hma, naz100_sp500_abacus) ✅ **DONE**
- [ ] All JSON configs exist ✅ **DONE**
- [ ] JSON configs modified to use static paths ✅ **DONE** (verified in sp500_pine)
- [ ] `docs/STATIC_DATA_STRUCTURE.md` created
- [ ] Verified at least 2 models run successfully (covered in Pre-Flight Checklist item 2)

**Once Phase 0 checklist is complete, proceed to Pre-Flight Checklist below.**

**NOTE:** Most of Phase 0 is already complete. Only documentation needs to be created.

---

## Pre-Flight Checklist (Before Any Phase Begins)

### Environment Setup

```bash
# 1. Create and checkout feature branch
git checkout -b chore/copilot-codebase-refresh

# 2. Establish baseline - run all validation commands and capture outputs
cd /Users/donaldpg/PyProjects/worktree2/PyTAAA

# Create baseline directory
mkdir -p .refactor_baseline/{before,after}

# Run baseline tests using STATIC data - these commands must produce identical outputs after each phase
# Command 1: naz100_pine
uv run python pytaaa_main.py --json /Users/donaldpg/pyTAAA_data_static/naz100_pine/pytaaa_naz100_pine.json 2>&1 | tee .refactor_baseline/before/pytaaa_naz100_pine.log

# Command 2: naz100_hma
uv run python pytaaa_main.py --json /Users/donaldpg/pyTAAA_data_static/naz100_hma/pytaaa_naz100_hma.json 2>&1 | tee .refactor_baseline/before/pytaaa_naz100_hma.log

# Command 3: naz100_pi
uv run python pytaaa_main.py --json /Users/donaldpg/pyTAAA_data_static/naz100_pi/pytaaa_naz100_pi.json 2>&1 | tee .refactor_baseline/before/pytaaa_naz100_pi.log

# Command 4: sp500_hma
uv run python pytaaa_main.py --json /Users/donaldpg/pyTAAA_data_static/sp500_hma/pytaaa_sp500_hma.json 2>&1 | tee .refactor_baseline/before/pytaaa_sp500_hma.log

# Command 5: sp500_pine
uv run python pytaaa_main.py --json /Users/donaldpg/pyTAAA_data_static/sp500_pine/pytaaa_sp500_pine.json 2>&1 | tee .refactor_baseline/before/pytaaa_sp500_pine.log

# Command 6: recommend_model (Abacus)
uv run python recommend_model.py --json /Users/donaldpg/pyTAAA_data_static/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json 2>&1 | tee .refactor_baseline/before/pytaaa_abacus_recommendation.log

# Command 7: daily_abacus_update
uv run python daily_abacus_update.py --json /Users/donaldpg/pyTAAA_data_static/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json --verbose 2>&1 | tee .refactor_baseline/before/pytaaa_abacus_daily.log

# Capture .params file checksums for comparison (from static data)
find /Users/donaldpg/pyTAAA_data_static -name "*.params" -exec md5sum {} \; > .refactor_baseline/before/params_checksums.txt

# Also capture the actual .params files for diff comparison
mkdir -p .refactor_baseline/before/params_files
find /Users/donaldpg/pyTAAA_data_static -name "*.params" -exec cp {} .refactor_baseline/before/params_files/ \;

# 3. Verify tests pass before starting
uv run pytest tests/ -v 2>&1 | tee .refactor_baseline/before/pytest_baseline.log

# 4. Capture performance baseline for key functions
uv run python -c "
import time
import numpy as np
from functions.TAfunctions import computeSignal2D, sharpeWeightedRank_2D
# Load test data
# ... timing code ...
" 2>&1 | tee .refactor_baseline/before/performance_baseline.log
```

### Baseline Requirements Met?

- [ ] All 7 end-to-end commands executed successfully using `pyTAAA_data_static`
- [ ] All `.params` files captured with checksums from static data
- [ ] All pytest tests pass (or known failures documented)
- [ ] Git branch `chore/copilot-codebase-refresh` created and checked out
- [ ] `.refactor_baseline/` directory in `.gitignore`
- [ ] Performance baseline captured

---

## Validation Protocol for Each Phase

After completing each phase, follow this exact validation procedure:

```bash
# 1. Run all 7 e2e commands using STATIC data
cd /Users/donaldpg/PyProjects/worktree2/PyTAAA

uv run python pytaaa_main.py --json /Users/donaldpg/pyTAAA_data_static/naz100_pine/pytaaa_naz100_pine.json 2>&1 | tee .refactor_baseline/after/pytaaa_naz100_pine.log
uv run python pytaaa_main.py --json /Users/donaldpg/pyTAAA_data_static/naz100_hma/pytaaa_naz100_hma.json 2>&1 | tee .refactor_baseline/after/pytaaa_naz100_hma.log
uv run python pytaaa_main.py --json /Users/donaldpg/pyTAAA_data_static/naz100_pi/pytaaa_naz100_pi.json 2>&1 | tee .refactor_baseline/after/pytaaa_naz100_pi.log
uv run python pytaaa_main.py --json /Users/donaldpg/pyTAAA_data_static/sp500_hma/pytaaa_sp500_hma.json 2>&1 | tee .refactor_baseline/after/pytaaa_sp500_hma.log
uv run python pytaaa_main.py --json /Users/donaldpg/pyTAAA_data_static/sp500_pine/pytaaa_sp500_pine.json 2>&1 | tee .refactor_baseline/after/pytaaa_sp500_pine.log
uv run python recommend_model.py --json /Users/donaldpg/pyTAAA_data_static/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json 2>&1 | tee .refactor_baseline/after/pytaaa_abacus_recommendation.log
uv run python daily_abacus_update.py --json /Users/donaldpg/pyTAAA_data_static/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json --verbose 2>&1 | tee .refactor_baseline/after/pytaaa_abacus_daily.log

# 2. Capture checksums
find /Users/donaldpg/pyTAAA_data_static -name "*.params" -exec md5sum {} \; > .refactor_baseline/after/params_checksums.txt

# 3. Copy params files for comparison
mkdir -p .refactor_baseline/after/params_files
find /Users/donaldpg/pyTAAA_data_static -name "*.params" -exec cp {} .refactor_baseline/after/params_files/ \;

# 4. Compare checksums
diff .refactor_baseline/before/params_checksums.txt .refactor_baseline/after/params_checksums.txt

# 5. Compare key params files (ignore timestamps in logs)
diff .refactor_baseline/before/params_files/PyTAAA_holdings.params .refactor_baseline/after/params_files/PyTAAA_holdings.params
diff .refactor_baseline/before/params_files/PyTAAA_ranks.params .refactor_baseline/after/params_files/PyTAAA_ranks.params

# 6. Run unit tests
uv run pytest tests/ -v 2>&1 | tee .refactor_baseline/after/pytest_results.log

# 7. Performance regression test (should be within 10% of baseline)
# Compare .refactor_baseline/before/performance_baseline.log with current
```

### Validation Success Criteria

- [ ] All 7 e2e commands complete without errors
- [ ] `.params` file checksums match baseline exactly (see "What 'Identical' Means" below)
- [ ] All unit tests pass
- [ ] Performance within acceptable tolerance (see "Performance Validation" below)
- [ ] No new warnings or errors in logs

### What "Identical Outputs" Means

**Exact Match Required:**
- Portfolio holdings (symbols, shares, allocations)
- Backtest returns and statistics
- Stock rankings and weights
- Trading signals (buy/sell/hold)

**Fields to IGNORE in Comparison:**
- Timestamps (e.g., "Generated on: 2026-02-11 14:35:22")
- System paths (e.g., "/Users/donaldpg/...")
- Log file names with dates
- Process IDs or run IDs

**Comparison Method:**

```bash
# For .params files - use custom comparison script
python refactor_tools/compare_params_files.py \
  .refactor_baseline/before/params_files/ \
  .refactor_baseline/after/params_files/
```

**Note:** Comparison scripts are in `refactor_tools/` (created separately, can be deleted after refactoring).

`refactor_tools/compare_params_files.py`:

```python
"""Compare .params files ignoring timestamps and system paths."""

import sys
import re
from pathlib import Path

def normalize_line(line):
    """Remove timestamps and system-specific paths."""
    # Remove timestamps like "2026-02-11 14:35:22"
    line = re.sub(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', 'TIMESTAMP', line)
    # Remove full paths, keep only relative paths
    line = re.sub(r'/Users/[^/]+/[^\s]+', 'PATH', line)
    return line

def compare_files(file1, file2):
    """Compare two files ignoring timestamps and paths."""
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = [normalize_line(line.strip()) for line in f1]
        lines2 = [normalize_line(line.strip()) for line in f2]
    
    if lines1 != lines2:
        print(f"✗ MISMATCH: {file1.name}")
        # Show first difference
        for i, (l1, l2) in enumerate(zip(lines1, lines2)):
            if l1 != l2:
                print(f"  Line {i+1} differs:")
                print(f"    Before: {l1[:100]}")
                print(f"    After:  {l2[:100]}")
                break
        return False
    else:
        print(f"✓ MATCH: {file1.name}")
        return True

def main(before_dir, after_dir):
    """Compare all .params files in two directories."""
    before_path = Path(before_dir)
    after_path = Path(after_dir)
    
    all_match = True
    for before_file in before_path.glob('*.params'):
        after_file = after_path / before_file.name
        if not after_file.exists():
            print(f"✗ MISSING: {before_file.name} not found in after/")
            all_match = False
            continue
        
        if not compare_files(before_file, after_file):
            all_match = False
    
    print()
    if all_match:
        print("✓✓✓ ALL FILES MATCH ✓✓✓")
        sys.exit(0)
    else:
        print("✗✗✗ VALIDATION FAILED ✗✗✗")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_params_files.py <before_dir> <after_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
```

### Performance Validation

**"Within 10%" means:**
- Execution time for any benchmark must be ≤ 110% of baseline
- Example: If baseline is 5.0 seconds, after must be ≤ 5.5 seconds
- Speedups are always acceptable (no upper limit)

**How to measure:**

```bash
# Compare performance baseline
python refactor_tools/compare_performance.py \
  .refactor_baseline/before/performance_baseline.json \
  .refactor_baseline/after/performance_baseline.json
```

`refactor_tools/compare_performance.py`:

```python
"""Compare performance baselines and check for regressions."""

import sys
import json

def compare_performance(before_file, after_file, tolerance=0.10):
    """Compare performance, allowing up to 10% slowdown."""
    with open(before_file, 'r') as f:
        before = json.load(f)
    with open(after_file, 'r') as f:
        after = json.load(f)
    
    all_ok = True
    for func_name in before.keys():
        if func_name == 'timestamp':
            continue
        
        before_mean = before[func_name]['mean']
        after_mean = after[func_name]['mean']
        
        delta_pct = ((after_mean - before_mean) / before_mean) * 100
        
        if after_mean > before_mean * (1 + tolerance):
            print(f"✗ REGRESSION: {func_name}")
            print(f"  Before: {before_mean:.4f}s")
            print(f"  After:  {after_mean:.4f}s")
            print(f"  Delta:  {delta_pct:+.1f}% (exceeds {tolerance*100}% threshold)")
            all_ok = False
        elif after_mean < before_mean:
            print(f"✓ SPEEDUP: {func_name}")
            print(f"  Before: {before_mean:.4f}s")
            print(f"  After:  {after_mean:.4f}s")
            print(f"  Delta:  {delta_pct:+.1f}%")
        else:
            print(f"✓ OK: {func_name} ({delta_pct:+.1f}%)")
    
    print()
    if all_ok:
        print("✓✓✓ PERFORMANCE OK ✓✓✓")
        return 0
    else:
        print("✗✗✗ PERFORMANCE REGRESSION DETECTED ✗✗✗")
        return 1

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_performance.py <before.json> <after.json>")
        sys.exit(1)
    sys.exit(compare_performance(sys.argv[1], sys.argv[2]))
```

### Troubleshooting Failed Validation

**If .params files don't match:**

1. **Check for actual differences:**
   ```bash
   diff -u .refactor_baseline/before/params_files/PyTAAA_holdings.params \
           .refactor_baseline/after/params_files/PyTAAA_holdings.params | head -50
   ```

2. **Common false positives:**
   - Timestamps in headers → Update comparison script to ignore
   - Log messages with different ordering (but same content) → Acceptable
   - Floating point rounding (e.g., 1.234567 vs 1.234568) → Use tolerance

3. **Real problems that require rollback:**
   - Different stock symbols in holdings
   - Different portfolio allocations
   - Different backtest returns
   - Missing or extra output files

**If performance regresses:**

1. **Profile the slow function:**
   ```bash
   uv run python -m cProfile -s cumtime pytaaa_main.py --json [config]
   ```

2. **Common causes:**
   - Accidental O(n²) loop introduced
   - Redundant data loading
   - Missing caching

3. **Decision:**
   - <5% regression → Document and proceed
   - 5-10% regression → Human review required
   - >10% regression → Rollback and revise

**If tests fail unexpectedly:**

1. **Check for import errors:**
   ```bash
   uv run python -c "from functions import TAfunctions; print('OK')"
   ```

2. **Check for circular imports:**
   ```bash
   uv run python tests/test_phase[X]_imports.py
   ```

3. **Verify static data integrity:**
   ```bash
   md5sum /Users/donaldpg/pyTAAA_data_static/sp500_pine/symbols/*.hdf5
   ```

---

## Phase 1: Foundation — Dead Code Removal & Documentation

**Complexity:** Low  
**Risk:** Low  
**Estimated Time:** 2-3 AI sessions  
**AI Model Recommendation:** Kimi K2.5 (high accuracy on pattern matching, lower cost)  
**Why:** Pattern-based dead code identification and docstring addition

### 1.1 Goals

1. Remove dead code (commented-out functions, unused imports, superseded definitions)
2. Add comprehensive docstrings to all public functions
3. Remove Python 2 compatibility code
4. Capture performance baseline for key functions
5. Establish clean foundation for subsequent phases

### 1.2 Files to Modify

| File | Changes |
|------|---------|
| `functions/CheckMarketOpen.py` | Remove 3 superseded `get_MarketOpenOrClosed()` definitions (lines 29-75) |
| `functions/TAfunctions.py` | Remove commented-out `interpolate()`, `cleantobeginning()` (lines 35-94, 96+) |
| `functions/quotes_for_list_adjClose.py` | Remove commented-out class and functions (lines 17-115) |
| `functions/GetParams.py` | Add docstrings to all public functions |
| `re-generateHDF5.py` | Move to `archive/` (historical reference, do not delete) |

### 1.3 Detailed Checklist

#### Task 1.1: Create STYLE_GUIDE.md

- [ ] Create `plans/STYLE_GUIDE.md` with coding standards
- [ ] Include Google-style docstring format
- [ ] Include Phase 2 exception handling pattern
- [ ] Include prohibited patterns (no bare `except:`)

#### Task 1.2: Capture Performance Baseline

`refactor_tools/benchmark_performance.py` (already created):

```python
"""Performance baseline capture for key functions."""

import time
import numpy as np
import json
from functions.TAfunctions import computeSignal2D, sharpeWeightedRank_2D
from functions.UpdateSymbols_inHDF5 import loadQuotes_fromHDF

def benchmark_function(func, *args, n_runs=5, **kwargs):
    """Benchmark a function with multiple runs."""
    times = []
    for _ in range(n_runs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        times.append(elapsed)
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }

def main():
    """Capture performance baseline."""
    # Load test data from STATIC directory
    symbols_file = "/Users/donaldpg/pyTAAA_data_static/naz100_pine/symbols/Naz100_Symbols.txt"
    adjClose, symbols, datearray = loadQuotes_fromHDF(symbols_file)
    
    # Use subset for faster testing
    adjClose = adjClose[:, -500:]  # Last 500 days
    datearray = datearray[-500:]
    
    params = {
        'numberStocksTraded': 7,
        'monthsToHold': 1,
        'MA1': 8,
        'MA2': 19,
        'MA3': 176,
        'sma2factor': 1.536,
        'uptrendSignalMethod': 'HMAs'
    }
    
    # Benchmark computeSignal2D
    signal_stats = benchmark_function(
        computeSignal2D, adjClose, symbols, datearray, params
    )
    
    # Benchmark sharpeWeightedRank_2D
    rank_stats = benchmark_function(
        sharpeWeightedRank_2D, adjClose, params
    )
    
    results = {
        'computeSignal2D': signal_stats,
        'sharpeWeightedRank_2D': rank_stats,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('.refactor_baseline/performance_baseline.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Performance baseline captured:")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
```

- [ ] Run benchmark: `uv run python refactor_tools/benchmark_performance.py`

#### Task 1.3: Audit Dead Code

- [ ] Search for all multi-line commented-out code blocks (`'''` or `"""` containing function definitions)
- [ ] Identify functions with multiple definitions (same name, different implementations)
- [ ] List all `try/except ImportError` blocks for Python 2/3 compatibility
- [ ] Document findings in `.refactor_baseline/dead_code_audit.md`

#### Task 1.4: Remove Dead Code

- [ ] `functions/CheckMarketOpen.py`: Keep only the active `get_MarketOpenOrClosed()` (lines 77-104)
- [ ] `functions/TAfunctions.py`: Remove lines 35-94 (commented interpolate/cleantobeginning)
- [ ] `functions/quotes_for_list_adjClose.py`: Remove lines 17-115 (commented webpage_companies_extractor class)
- [ ] `functions/quotes_for_list_adjClose.py`: Remove lines 959-961 (deprecated retry logic with `print "."`)
- [ ] `functions/quotes_for_list_adjClose.py`: Remove lines 980-987 (deprecated import fallback)
- [ ] Move `re-generateHDF5.py` to `archive/` directory (keep for historical reference)

#### Task 1.5: Add Docstrings

For each public function in modified files, add Google-style docstrings (see STYLE_GUIDE.md).

Priority functions to document:
- [ ] `functions/GetParams.py`: All `get_*` functions
- [ ] `functions/CheckMarketOpen.py`: `get_MarketOpenOrClosed()`, `CheckMarketOpen()`
- [ ] `functions/TAfunctions.py`: `strip_accents()`, `normcorrcoef()`

#### Task 1.6: Write Tests

Create `tests/test_phase1_cleanup.py`:

```python
"""Tests for Phase 1 cleanup - verify dead code removal doesn't break functionality."""

import pytest
import ast
import inspect

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
        assert hasattr(TAfunctions, 'interpolate')
        assert hasattr(TAfunctions, 'cleantobeginning')
        assert hasattr(TAfunctions, 'cleantoend')
    
    def test_no_duplicate_definitions_checkmarketopen(self):
        """Verify no function is defined multiple times in CheckMarketOpen.py."""
        from functions import CheckMarketOpen
        
        source = inspect.getsource(CheckMarketOpen)
        tree = ast.parse(source)
        
        function_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_names.append(node.name)
        
        duplicates = [name for name in set(function_names) if function_names.count(name) > 1]
        assert len(duplicates) == 0, f"Duplicate function definitions found: {duplicates}"
    
    def test_regenerate_hdf5_archived(self):
        """Verify re-generateHDF5.py is archived, not deleted."""
        import os
        assert os.path.exists('archive/re-generateHDF5.py'), "File should be archived"
        assert not os.path.exists('re-generateHDF5.py'), "File should not be in root"
```

#### Task 1.7: Validation

- [ ] Run pytest: `uv run pytest tests/test_phase1_cleanup.py -v`
- [ ] Run all 7 end-to-end commands using STATIC data, compare outputs to baseline
- [ ] Verify `.params` file checksums match baseline
- [ ] Verify performance baseline is within 10% (no regressions)
- [ ] Document any differences in `.refactor_baseline/phase1_differences.md`

#### Task 1.8: Commit

```bash
git add -A
git commit -m "Phase 1: Remove dead code and add docstrings

- Create STYLE_GUIDE.md with coding standards
- Capture performance baseline for key functions
- Remove 3 superseded get_MarketOpenOrClosed() definitions
- Remove commented-out interpolate/cleantobeginning in TAfunctions.py
- Remove commented webpage_companies_extractor class
- Move re-generateHDF5.py to archive/ (historical reference)
- Add Google-style docstrings to public functions
- Add tests/test_phase1_cleanup.py and tests/benchmark_performance.py

All end-to-end tests pass with identical outputs to baseline.
Performance within 10% of baseline."
```

---

## Phase 2: Exception Handling — Replace Bare `except:` Clauses

**Complexity:** Medium  
**Risk:** Medium (may expose currently-silent failures)  
**Estimated Time:** 3-4 AI sessions  
**AI Model Recommendation:** Claude 4.5 Sonnet (better at reasoning about exception hierarchies)  
**Why:** Exception type reasoning requires careful analysis

### 2.1 Goals

1. Replace all bare `except:` with specific exception types
2. Add logging for caught exceptions
3. Maintain identical behavior for expected failure modes
4. Include safety fallback for unobserved exceptions
5. Document any exposed failures for future fixes

### 2.2 Safety-First Approach with Fallback

Per critic review, use this improved pattern:

**Step 2.1 (Logging Mode):** Add inline logging to understand what exceptions are actually caught
**Step 2.2 (Fix Mode):** Replace with specific exceptions PLUS safety fallback

```python
# FINAL PATTERN (after Step 2.2):
try:
    risky_operation()
except (ExpectedError1, ExpectedError2) as e:
    logger.debug(f"Expected error: {e}")
    fallback_code()
except Exception as e:
    # Safety fallback for unobserved exceptions (production edge cases)
    logger.warning(f"Unexpected {type(e).__name__}: {e}")
    fallback_code()  # Maintain existing behavior
```

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

#### Task 2.1: Inline Exception Logging (Logging Mode)

For each bare `except:` in P0 files, add inline logging (NO decorator/module):

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
    import logging, inspect
    logging.getLogger(__name__).debug(
        f"PHASE2_DEBUG: Caught {type(_e).__name__}: {_e} "
        f"at {__file__}:{inspect.currentframe().f_lineno}"
    )
    fallback()
```

- [ ] Instrument `functions/CheckMarketOpen.py` (3 locations)
- [ ] Instrument `PyTAAA.py` (4 locations)
- [ ] Instrument `run_pytaaa.py` (4 locations)

#### Task 2.2: Run and Collect Exception Logs

- [ ] Run all 7 end-to-end commands with logging enabled using STATIC data
- [ ] Collect and analyze exception types caught
- [ ] Document findings in `.refactor_baseline/exception_types_observed.md`

#### Task 2.3: Replace with Specific Exceptions + Safety Fallback (Fix Mode)

Based on observed exception types, replace bare `except:` with specific exceptions AND safety fallback:

```python
# AFTER (Step 2.2 - Fix Mode):
import urllib.error
import logging

logger = logging.getLogger(__name__)

try:
    risky_operation()
except (AttributeError, urllib.error.URLError) as e:
    logger.debug(f"Market status check failed: {e}")
    status = 'no Market Open/Closed status available'
except Exception as e:
    # Safety fallback for unobserved exceptions
    logger.warning(f"Unexpected exception {type(e).__name__} in get_MarketOpenOrClosed: {e}")
    status = 'no Market Open/Closed status available'
```

- [ ] Update `functions/CheckMarketOpen.py` with specific exceptions + safety fallback
- [ ] Update `PyTAAA.py` with specific exceptions + safety fallback
- [ ] Update `run_pytaaa.py` with specific exceptions + safety fallback
- [ ] Scrub any logged sensitive data (paths, credentials) before committing

#### Task 2.4: Write Tests

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
    
    def test_unexpected_exception_handled(self):
        """Verify unexpected exceptions are caught by safety fallback."""
        from functions.CheckMarketOpen import get_MarketOpenOrClosed
        
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_urlopen.side_effect = MemoryError("Unexpected error")
            # Should not raise, safety fallback should handle it
            result = get_MarketOpenOrClosed()
            assert result == 'no Market Open/Closed status available'
```

#### Task 2.5: Validation

- [ ] Run pytest: `uv run pytest tests/test_phase2_exceptions.py -v`
- [ ] Run all 7 end-to-end commands using STATIC data, compare outputs to baseline
- [ ] Verify `.params` file checksums match baseline
- [ ] Check that no new exceptions propagate (previously caught by bare `except:`)
- [ ] Verify no sensitive data in logs

#### Task 2.6: Commit

```bash
git add -A
git commit -m "Phase 2: Replace bare except clauses with specific exceptions

- Add inline PHASE2_DEBUG logging (removed in final)
- Replace bare except: in CheckMarketOpen.py with (AttributeError, URLError) + safety fallback
- Replace bare except: in PyTAAA.py with specific exceptions + safety fallback
- Replace bare except: in run_pytaaa.py with specific exceptions + safety fallback
- Scrub sensitive data from logs
- Add tests/test_phase2_exceptions.py with safety fallback test

Observed exception types documented in exception_types_observed.md
All end-to-end tests pass with identical outputs using static data."
```

---

## Phase 3: JSON Migration — Complete Legacy-to-JSON Transition

**Complexity:** Medium  
**Risk:** Medium (configuration system changes)  
**Estimated Time:** 3-4 AI sessions  
**AI Model Recommendation:** Claude 4.5 Sonnet (complex reasoning about config systems)  
**Why:** Config migration requires careful analysis of dependencies

### 3.1 Goals

1. Migrate remaining legacy `.params` usage to JSON
2. Remove legacy `GetParams()` function and related functions
3. Update all entry points to use JSON exclusively
4. Deprecate `PyTAAA.py` in favor of `pytaaa_main.py`
5. Add import compatibility tests
6. Update documentation references to legacy config

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
- [ ] Test the migrated entry point using STATIC data

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
        
        assert not hasattr(GetParams, 'GetParams')
        assert hasattr(GetParams, 'get_json_params')
    
    def test_get_json_params_works(self):
        """Modern JSON params function works correctly."""
        from functions.GetParams import get_json_params
        
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

class TestImportCompatibility:
    """Verify all imports still work after migration."""
    
    def test_all_getparams_imports(self):
        """All GetParams functions can be imported."""
        from functions.GetParams import (
            get_json_params, get_holdings, get_status, put_status,
            get_json_ftp_params, get_symbols_file, get_performance_store,
            get_webpage_store
        )
        assert callable(get_json_params)
        assert callable(get_holdings)
```

#### Task 3.6: Update Documentation

- [ ] Update any docs referencing legacy config functions
- [ ] Update `docs/PYTAAA_SUMMARY.md` if it mentions legacy functions
- [ ] Add migration note to `docs/RECOMMENDATIONS.md`

#### Task 3.7: Validation

- [ ] Run pytest: `uv run pytest tests/test_phase3_json_migration.py -v`
- [ ] Run all 7 end-to-end commands using STATIC data, compare outputs to baseline
- [ ] Verify `.params` file checksums match baseline
- [ ] Test that `PyTAAA.py` shows deprecation warning but still works

#### Task 3.8: Commit

```bash
git add -A
git commit -m "Phase 3: Complete JSON migration, deprecate legacy config

- Remove legacy GetParams(), GetHoldings(), GetStatus(), PutStatus()
- Remove legacy GetFTPParams() function
- Add deprecation warning to PyTAAA.py
- Redirect PyTAAA.py to pytaaa_main.py
- Add import compatibility tests
- Update documentation references

All end-to-end tests pass with identical outputs using static data.
PyTAAA.py shows deprecation warning but remains functional."
```

---

## Phase 4a: Testability — Extract Data Loading (Part 1)

**Complexity:** Medium  
**Risk:** Medium (architectural changes)  
**Estimated Time:** 3-4 AI sessions  
**AI Model Recommendation:** Claude 4.5 Sonnet (architectural refactoring)  
**Why:** Split from Phase 4 per critic review to reduce scope

### 4a.1 Goals

1. Extract data loading from `PortfolioPerformanceCalcs()` into separate module
2. Enable unit testing of data loading logic
3. Maintain backward compatibility
4. Add shadow mode comparison (old vs new implementation)

### 4a.2 Files to Modify

| File | Changes |
|------|---------|
| `functions/PortfolioPerformanceCalcs.py` | Refactor to use extracted data loader |
| New: `functions/data_loaders.py` | Create data loading module |

### 4a.3 Detailed Checklist

#### Task 4a.1: Create Data Loaders Module

Create `functions/data_loaders.py`:

```python
"""Data loading functions separated from computation.

This module provides pure data loading functionality,
separated from computation to enable unit testing.
"""

from typing import Tuple, List, Optional
import numpy as np
from functions.UpdateSymbols_inHDF5 import loadQuotes_fromHDF
from functions.TAfunctions import interpolate, cleantobeginning, cleantoend

def load_quotes_for_analysis(
    symbols_file: str,
    params: dict,
    verbose: bool = False
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Load and prepare quote data for analysis.
    
    Args:
        symbols_file: Path to symbols file
        params: Configuration parameters
        verbose: Whether to print progress
        
    Returns:
        Tuple of (adjClose_array, symbols_list, date_array)
        
    Raises:
        FileNotFoundError: If symbols file doesn't exist
        ValueError: If data loading fails
    """
    # Load from HDF5
    adjClose, symbols, datearray = loadQuotes_fromHDF(symbols_file)
    
    # Clean data
    for ii in range(adjClose.shape[0]):
        adjClose[ii, :] = interpolate(adjClose[ii, :])
        adjClose[ii, :] = cleantobeginning(adjClose[ii, :])
        adjClose[ii, :] = cleantoend(adjClose[ii, :])
    
    return adjClose, symbols, datearray
```

#### Task 4a.2: Refactor PortfolioPerformanceCalcs

Update `functions/PortfolioPerformanceCalcs.py`:

```python
"""Portfolio performance calculations."""

from functions.data_loaders import load_quotes_for_analysis

def PortfolioPerformanceCalcs(json_fn: str) -> dict:
    """Calculate portfolio performance.
    
    Args:
        json_fn: Path to JSON configuration file
        
    Returns:
        Dictionary containing rankings, weights, and other results
    """
    # Load params
    params = get_json_params(json_fn)
    symbols_file = get_symbols_file(json_fn)
    
    # Load data using extracted function
    adjClose, symbols, datearray = load_quotes_for_analysis(symbols_file, params)
    
    # ... rest of computation (unchanged)
```

#### Task 4a.3: Add Shadow Mode Comparison

Create `tests/test_phase4a_shadow.py`:

```python
"""Shadow mode tests for Phase 4a - compare old vs new data loading."""

import pytest
import numpy as np
from functions.data_loaders import load_quotes_for_analysis
from functions.UpdateSymbols_inHDF5 import loadQuotes_fromHDF
from functions.TAfunctions import interpolate, cleantobeginning, cleantoend

class TestDataLoaderShadow:
    """Compare new data loader with inline implementation."""
    
    def test_data_loader_matches_inline(self):
        """New data loader produces identical results to inline code."""
        symbols_file = "/Users/donaldpg/pyTAAA_data_static/naz100_pine/symbols/Naz100_Symbols.txt"
        params = {}
        
        # New implementation
        adjClose_new, symbols_new, datearray_new = load_quotes_for_analysis(
            symbols_file, params
        )
        
        # Inline implementation (copy of original code)
        adjClose_old, symbols_old, datearray_old = loadQuotes_fromHDF(symbols_file)
        for ii in range(adjClose_old.shape[0]):
            adjClose_old[ii, :] = interpolate(adjClose_old[ii, :])
            adjClose_old[ii, :] = cleantobeginning(adjClose_old[ii, :])
            adjClose_old[ii, :] = cleantoend(adjClose_old[ii, :])
        
        # Compare
        np.testing.assert_array_equal(adjClose_new, adjClose_old)
        assert symbols_new == symbols_old
        np.testing.assert_array_equal(datearray_new, datearray_old)
```

#### Task 4a.4: Validation

- [ ] Run pytest: `uv run pytest tests/test_phase4a_shadow.py -v`
- [ ] Run all 7 end-to-end commands using STATIC data
- [ ] Verify `.params` file checksums match baseline
- [ ] Verify shadow mode comparison passes

#### Task 4a.5: Commit

```bash
git add -A
git commit -m "Phase 4a: Extract data loading from PortfolioPerformanceCalcs

- Create functions/data_loaders.py with load_quotes_for_analysis()
- Refactor PortfolioPerformanceCalcs to use extracted loader
- Add shadow mode comparison tests
- Maintain backward compatibility

All end-to-end tests pass with identical outputs using static data.
Shadow mode comparison verified."
```

---

## Phase 4b: Testability — Extract Plot/File I/O (Part 2)

**Complexity:** High  
**Risk:** High (architectural changes)  
**Estimated Time:** 3-4 AI sessions  
**AI Model Recommendation:** o1 (highest reasoning capability for complex refactoring)  
**Why:** Highest risk phase, requires careful architectural reasoning

### 4b.1 Goals

1. Extract plot generation from `PortfolioPerformanceCalcs()`
2. Extract file writing from `PortfolioPerformanceCalcs()`
3. Create pure computation function with no side effects
4. Enable unit testing of core logic without file system access

### 4b.2 Files to Modify

| File | Changes |
|------|---------|
| `functions/PortfolioPerformanceCalcs.py` | Refactor into orchestrator + pure functions |
| New: `functions/output_generators.py` | Plot and file output functions |

### 4b.3 Detailed Checklist

#### Task 4b.1: Create Output Generators Module

Create `functions/output_generators.py`:

```python
"""Output generation functions (plots, files) separated from computation."""

from typing import Dict, Any
import os
import numpy as np
from functions.MakeValuePlot import makeValuePlot, makeUptrendingPlot

def generate_plots(
    results: Dict[str, Any],
    params: Dict[str, Any],
    output_dir: str
) -> None:
    """Generate all plots for the analysis.
    
    Args:
        results: Computation results from compute_portfolio_metrics
        params: Configuration parameters
        output_dir: Directory for output files
    """
    # Generate plots using existing functions
    pass

def write_output_files(
    results: Dict[str, Any],
    params: Dict[str, Any],
    output_dir: str
) -> None:
    """Write output files (.params, etc.).
    
    Args:
        results: Computation results
        params: Configuration parameters
        output_dir: Directory for output files
    """
    pass
```

#### Task 4b.2: Create Pure Computation Function

Refactor `PortfolioPerformanceCalcs()`:

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
    # Pure computation only - no file I/O, no plotting
    ...

def PortfolioPerformanceCalcs(json_fn: str) -> dict:
    """Main entry point - orchestrates loading, computation, and output.
    
    This is the original function signature for backward compatibility.
    Internally delegates to pure functions.
    """
    from functions.data_loaders import load_quotes_for_analysis
    from functions.output_generators import generate_plots, write_output_files
    
    # Load data
    params = get_json_params(json_fn)
    symbols_file = get_symbols_file(json_fn)
    adjClose, symbols, datearray = load_quotes_for_analysis(symbols_file, params)
    
    # Compute (pure function)
    results = compute_portfolio_metrics(adjClose, symbols, datearray, params)
    
    # Generate outputs
    output_dir = params.get('output_dir', '.')
    generate_plots(results, params, output_dir)
    write_output_files(results, params, output_dir)
    
    return results
```

#### Task 4b.3: Write Tests

Create `tests/test_phase4b_computation.py`:

```python
"""Tests for Phase 4b pure computation function."""

import pytest
import numpy as np
from functions.PortfolioPerformanceCalcs import compute_portfolio_metrics

class TestComputePortfolioMetrics:
    """Test pure computation function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_symbols = 10
        n_days = 252
        
        dates = np.arange(n_days)
        adjClose = np.zeros((n_symbols, n_days))
        
        for i in range(n_symbols):
            trend = 1 + 0.0001 * dates + 0.001 * np.random.randn(n_days)
            adjClose[i] = 100 * np.cumprod(trend)
        
        symbols = [f'STOCK{i}' for i in range(n_symbols)]
        
        return adjClose, symbols, dates
    
    def test_compute_portfolio_metrics_returns_expected_keys(self, sample_data):
        """Verify computation returns expected structure."""
        adjClose, symbols, dates = sample_data
        params = {
            'numberStocksTraded': 5,
            'monthsToHold': 1,
            'MA1': 8,
            'MA2': 19,
            'MA3': 176,
            'sma2factor': 1.536,
            'uptrendSignalMethod': 'HMAs'
        }
        
        results = compute_portfolio_metrics(adjClose, symbols, dates, params)
        
        assert 'rankings' in results
        assert 'weights' in results
        assert 'signals' in results
    
    def test_compute_portfolio_metrics_deterministic(self, sample_data):
        """Verify computation is deterministic."""
        adjClose, symbols, dates = sample_data
        params = {'numberStocksTraded': 5, 'monthsToHold': 1}
        
        results1 = compute_portfolio_metrics(adjClose, symbols, dates, params)
        results2 = compute_portfolio_metrics(adjClose, symbols, dates, params)
        
        np.testing.assert_array_equal(results1['rankings'], results2['rankings'])
```

#### Task 4b.4: Validation

- [ ] Run pytest: `uv run pytest tests/test_phase4b_computation.py -v`
- [ ] Run all 7 end-to-end commands using STATIC data
- [ ] Verify `.params` file checksums match baseline
- [ ] Verify performance within 10% of baseline

#### Task 4b.5: Commit

```bash
git add -A
git commit -m "Phase 4b: Separate computation from I/O

- Extract plot generation to functions/output_generators.py
- Extract file writing to functions/output_generators.py
- Create pure compute_portfolio_metrics() function
- Refactor PortfolioPerformanceCalcs() as orchestrator
- Add comprehensive unit tests for computation logic
- Enable testing without file system access

All end-to-end tests pass with identical outputs using static data.
New unit tests: 15 tests, all passing."
```

---

## Phase 5: Modularity — Break Up TAfunctions.py

**Complexity:** Very High  
**Risk:** High (major structural changes)  
**Estimated Time:** 6-8 AI sessions  
**AI Model Recommendation:** o1 (most capable for complex refactoring)  
**Why:** Most complex phase, requires careful module decomposition

### 5.1 Goals

1. Split 4,100+ line `TAfunctions.py` into focused modules
2. Maintain backward compatibility via re-exports
3. Preserve exact function behavior
4. Enable independent testing of submodules
5. Add circular import detection

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
3. Add comprehensive docstrings per STYLE_GUIDE.md
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
        
        assert result[2] == pytest.approx(2.0)
        assert result[4] == pytest.approx(4.0)
    
    def test_sma_2d(self):
        """Test 2D SMA calculation."""
        prices = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0]
        ])
        from functions.moving_averages import SMA_2D
        result = SMA_2D(prices, 3)
        
        assert result.shape == prices.shape
        assert result[0, 2] == pytest.approx(2.0)
        assert result[1, 2] == pytest.approx(4.0)
```

Repeat for each module.

#### Task 5.4: Add Circular Import Detection

Create `tests/test_phase5_imports.py`:

```python
"""Tests for Phase 5 import compatibility."""

import pytest

class TestCircularImports:
    """Verify no circular imports exist."""
    
    def test_no_circular_imports(self):
        """All modules import without circular dependency errors."""
        # This test will fail during import if circular imports exist
        from functions import moving_averages
        from functions import channels
        from functions import trend_analysis
        from functions import signal_generation
        from functions import ranking
        from functions import data_cleaning
        from functions import rolling_metrics
        from functions import TAfunctions
        
        assert True  # If we get here, no circular imports

class TestBackwardCompatibility:
    """Verify backward compatibility of re-exports."""
    
    def test_tafunctions_reexports(self):
        """All expected functions available from TAfunctions."""
        from functions import TAfunctions
        
        assert hasattr(TAfunctions, 'SMA')
        assert hasattr(TAfunctions, 'hma')
        assert hasattr(TAfunctions, 'computeSignal2D')
        assert hasattr(TAfunctions, 'sharpeWeightedRank_2D')
        assert hasattr(TAfunctions, 'interpolate')
    
    def test_direct_imports(self):
        """Functions can be imported directly from submodules."""
        from functions.moving_averages import SMA, hma
        from functions.ranking import sharpeWeightedRank_2D
        from functions.signal_generation import computeSignal2D
        
        assert callable(SMA)
        assert callable(hma)
        assert callable(sharpeWeightedRank_2D)
        assert callable(computeSignal2D)
```

#### Task 5.5: Update Architecture Documentation

- [ ] Update `docs/ARCHITECTURE.md` with new module structure
- [ ] Update module diagrams
- [ ] Add note about backward compatibility

#### Task 5.6: Validation

- [ ] Run all module-specific tests
- [ ] Run all 7 end-to-end commands using STATIC data
- [ ] Verify `.params` file checksums match baseline
- [ ] Verify backward compatibility (imports from TAfunctions still work)
- [ ] Verify no circular imports

#### Task 5.7: Commit

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
- Add circular import detection tests
- Add backward compatibility tests
- Update architecture documentation

All end-to-end tests pass with identical outputs using static data.
Backward compatibility verified. No circular imports."
```

---

## Phase 6: Polish — Type Annotations, Logging, CLI Standardization

**Complexity:** Medium  
**Risk:** Low (additive changes)  
**Estimated Time:** 4-5 AI sessions  
**AI Model Recommendation:** Kimi K2.5 (good at systematic additions)  
**Why:** Pattern-based systematic additions

### 6.1 Goals

1. Add type annotations to all public functions
2. Migrate `print()` statements to logging
3. Standardize CLI entry points on Click
4. Improve documentation consistency
5. Security review

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

#### Task 6.4: Security Review

- [ ] Verify no sensitive data (credentials, API keys) in logs
- [ ] Check that file paths in logs don't expose system information
- [ ] Review exception messages for sensitive data
- [ ] Document findings in `.refactor_baseline/security_review.md`

#### Task 6.5: Final Documentation Sync

- [ ] Update all docs to reflect final code structure
- [ ] Update function references in markdown files
- [ ] Create `REFACTORING_STATUS.md` tracking completion

#### Task 6.6: Validation

- [ ] Run mypy if available: `uv run mypy functions/`
- [ ] Run all 7 end-to-end commands using STATIC data
- [ ] Verify log files are created properly
- [ ] Verify no sensitive data in logs

#### Task 6.7: Commit

```bash
git add -A
git commit -m "Phase 6: Add type annotations, logging, CLI standardization

- Add type annotations to all public functions in refactored modules
- Migrate print() to logger.debug() in core computation
- Standardize CLI entry points on Click
- Add --verbose flag consistently
- Security review: no sensitive data in logs
- Final documentation synchronization

All end-to-end tests pass with identical outputs using static data.
Refactoring complete."
```

---

## Human Review Checkpoints

Per critic review, explicit human review is required at these points:

### Checkpoint 1: After Phase 2

**Required before starting Phase 3:**
- [ ] Review exception handling changes
- [ ] Verify safety fallback pattern is in place
- [ ] Check that no unexpected exceptions are being raised
- [ ] Approve continuation to Phase 3

### Checkpoint 2: After Phase 4b

**Required before starting Phase 5:**
- [ ] Review I/O separation changes
- [ ] Verify `compute_portfolio_metrics()` is pure function
- [ ] Check that all tests pass with static data
- [ ] Approve continuation to Phase 5

### Final Review: After Phase 6

**Required before merging to main:**
- [ ] Complete review of all changes
- [ ] Verify all documentation is updated
- [ ] Run full test suite
- [ ] Approve merge to main

---

## AI Model Recommendations by Phase

| Phase | Recommended Model | Rationale | Cost Estimate |
|-------|-------------------|-----------|---------------|
| 1 | Kimi K2.5 | Pattern matching for dead code, docstring generation | $ |
| 2 | Claude 4.5 Sonnet | Exception hierarchy reasoning, safety analysis | $$ |
| 3 | Claude 4.5 Sonnet | Complex config system migration | $$ |
| 4a | Claude 4.5 Sonnet | Architectural refactoring | $$ |
| 4b | o1 | Highest complexity I/O separation | $$$ |
| 5 | o1 | Most complex refactoring, module decomposition | $$$ |
| 6 | Kimi K2.5 | Systematic type annotation addition | $ |

**Cost Legend:**
- $ = Low cost (~$5-15 per session)
- $$ = Medium cost (~$15-30 per session)
- $$$ = Higher cost (~$30-60 per session)

**Success Probability:**
- All models estimated at >95% success for their assigned phases
- o1 recommended for Phases 4b-5 due to complex reasoning requirements
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
# Run end-to-end validation using STATIC data
```

### Full Rollback

```bash
# To completely abandon refactoring
git checkout main
git branch -D chore/copilot-codebase-refresh

# Or keep branch but switch back
git checkout main
```

---

## Success Criteria

### Overall Success

The refactoring is successful when:

1. All phases completed and committed
2. All new tests pass
3. All 7 end-to-end commands produce identical outputs to baseline using STATIC data
4. `.params` file checksums match baseline exactly
5. Code coverage increased from baseline
6. No new warnings or errors in logs
7. Performance within 10% of baseline (no regressions)
8. All documentation updated
9. Security review passed

### Phase Success Criteria

Each phase must:

1. Have all checklist items completed
2. Pass its dedicated test file
3. Pass all 7 end-to-end validation commands using STATIC data
4. Have a clean git commit
5. Be approved in human review (if checkpoint phase)

---

## Appendix A: End-to-End Validation Commands (Using Static Data)

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

# Command 6: recommend_model (Abacus)
uv run python recommend_model.py --json /Users/donaldpg/pyTAAA_data_static/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json 2>&1 | tee .refactor_baseline/after/pytaaa_abacus_recommendation.log

# Command 7: daily_abacus_update
uv run python daily_abacus_update.py --json /Users/donaldpg/pyTAAA_data_static/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json --verbose 2>&1 | tee .refactor_baseline/after/pytaaa_abacus_daily.log

# Compare checksums
diff .refactor_baseline/before/params_checksums.txt .refactor_baseline/after/params_checksums.txt
```

---

## Appendix B: Static Data Directory Structure

```
pyTAAA_data_static/
├── naz100_pine/
│   ├── symbols/
│   │   ├── Naz100_Symbols.txt
│   │   └── Naz100_Symbols_.hdf5
│   ├── data_store/
│   │   └── *.params (output files)
│   └── pytaaa_naz100_pine.json (modified for static testing)
├── naz100_hma/
│   └── ... same structure ...
├── naz100_pi/
│   └── ... same structure ...
├── sp500_hma/
│   └── ... same structure ...
├── sp500_pine/
│   └── ... same structure ...
└── naz100_sp500_abacus/
    └── ... same structure ...
```

---

## Appendix C: References

1. [PEP 8 — Style Guide for Python Code](https://peps.python.org/pep-0008/)
2. [PEP 257 — Docstring Conventions](https://peps.python.org/pep-0257/)
3. [PEP 484 — Type Hints](https://peps.python.org/pep-0484/)
4. [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
5. [Martin Fowler - Refactoring](https://refactoring.com/)
6. [`docs/RECOMMENDATIONS.md`](../docs/RECOMMENDATIONS.md) - Original analysis
7. [`plans/CRITIC_REVIEW.md`](plans/CRITIC_REVIEW.md) - Critic review feedback
8. [`plans/STYLE_GUIDE.md`](plans/STYLE_GUIDE.md) - Coding standards

---

## Next Steps

1. **Human Review:** Review this updated plan
2. **Address Feedback:** Incorporate any additional feedback
3. **Create Branch:** `git checkout -b refactor/modernize`
4. **Execute Pre-Flight:** Run baseline capture using static data
5. **Begin Phase 1:** Start with dead code removal

---

**Plan Status:** Ready for execution pending human approval
