# PyTAAA Refactoring Plan v3
## Agentic AI Implementation Guide — Clean Slate Edition

**Created:** 2026-03-02
**Branch:** `refactor/v3-clean-code`
**Prior Plan:** `plans/REFACTORING_PLAN_final.md` (Phases 1–6 partially complete)
**Scope:** Full clean-code refactor including all unfinished prior work

---

## Executive Summary

PyTAAA is a working Python 3.11+ trading system. The codebase has undergone
partial refactoring (Phases 1–3 and 5–6 nominally complete), but a deep rescan
on 2026-03-02 found significant technical debt remaining:

- **40+ bare `except:` clauses** still active (prior Phase 2 only fixed P0 files)
- **9 wildcard imports** (`from X import *`) preventing safe renaming
- **Duplicate function definitions** (up to 3×) in live modules
- **Circular import** between `MakeValuePlot.py` ↔ `WriteWebPage_pi.py`
- **Debug `print()` calls** in every `GetParams.py` public function
- **Hardcoded absolute paths** in 5+ files
- **No pytest coverage** for `sharpeWeightedRank_2D()` or
  `computeDailyBacktest()` — the two most critical functions
- **JSON config parsed anew** on every call (4+ re-opens per pipeline run)
- **Module-level side effects** (log file creation, signal handler
  installation, matplotlib backend) triggered on import

This plan re-scopes the work from scratch. All entry points must produce
**identical outputs** before and after each phase.

---

## Entry Points (All Must Work Identically After Refactoring)

| Entry Point | Purpose | Primary CLI Args |
|---|---|---|
| `pytaaa_main.py` | Modern pipeline entry point | `--json` |
| `PyTAAA.py` | Legacy shim (deprecated, redirect only) | none |
| `daily_abacus_update.py` | Daily dashboard + model-switching | `--json`, `--verbose` |
| `recommend_model.py` | Abacus model recommendation | `--json`, `--date`, `--lookbacks` |
| `run_monte_carlo.py` | Abacus lookback optimizer | `--json`, `--search`, `--workers` |
| `pytaaa_backtest_montecarlo.py` | Per-model MC backtest (modern) | `--json`, `--trials`, `--max-plots`, `--tag`, `--params-file` |
| `pytaaa_quotes_update.py` | HDF5 quote cleaner | `--json` |
| `run_normalized_score_history.py` | Score history plotter | (hardcoded path — fix in Phase D) |
| `modify_saved_state.py` | Monte Carlo state inspector | `--inspect`, `--reset`, `--file` |
| `update_json_from_csv.py` | CSV→JSON parameter transfer | `--csv`, `--row`, `--json`, `--dry-run` |

---

## Implementation Mode Recommendation

### GitHub Copilot Coding Agent (GitHub Issues — Autonomous)

Assign to the **GitHub Copilot coding agent** by creating a GitHub Issue and
selecting "Assign to Copilot" from the Assignees dropdown. The agent opens a PR
when done, which you review before merging. Best for:

- Mechanical, high-volume, well-defined changes with clear pass/fail tests
- "Find all X and replace with pattern Y across N files"

**Best for Phases: A, B, C, D, I**

### VS Code Copilot Agent (Implement Mode — Local)

Open the Agent panel in VS Code, select **Implement**, paste the phase task
description with file paths and acceptance criteria. Best for:

- Architectural changes requiring reasoning about import graphs
- Writing tests for complex pure functions
- Resolving circular imports (requires understanding both sides simultaneously)
- Performance optimizations (config singleton)

**Best for Phases: E, F, G, H**

### Decision Matrix

| Phase | Type | Mode |
|---|---|---|
| A: Mechanical Cleanup | High-volume find/replace | GitHub Copilot Agent |
| B: Wildcard Imports | Explicit import conversion | GitHub Copilot Agent |
| C: Duplicate Removal | Dead definition removal | GitHub Copilot Agent |
| D: Hardcoded Paths | Config-driven paths | GitHub Copilot Agent |
| E: Circular Import Fix | Architectural restructure | VS Code Implement |
| F: Config Singleton | Caching + singleton | VS Code Implement |
| G: Core Tests | Unit tests for core functions | VS Code Implement |
| H: Data Layer Cleanup | Side effects + deduplication | VS Code Implement |
| I: Documentation | Docstrings + type annotations | GitHub Copilot Agent |

---

## Branch Setup

```bash
# Create the refactoring branch from main
git checkout main
git pull origin main
git checkout -b refactor/v3-clean-code
git push -u origin refactor/v3-clean-code
```

All work happens on `refactor/v3-clean-code`. Each phase is its own commit.
The branch merges to `main` only after all phases pass validation.

Add `.refactor_v3/` to `.gitignore` before starting.

---

## Static Data Validation (Prerequisite — Already Complete)

Static data exists at `/Users/donaldpg/pyTAAA_data_static/`. All 6 model
subdirectories are present and configured with `updateQuotes: false`.

### Capture Baseline (Once, Before Phase A)

```bash
cd /Users/donaldpg/PyProjects/worktree2/PyTAAA
mkdir -p .refactor_v3/before

uv run python pytaaa_main.py \
  --json /Users/donaldpg/pyTAAA_data_static/naz100_pine/pytaaa_naz100_pine.json \
  2>&1 | tee .refactor_v3/before/naz100_pine.log

uv run python pytaaa_main.py \
  --json /Users/donaldpg/pyTAAA_data_static/naz100_hma/pytaaa_naz100_hma.json \
  2>&1 | tee .refactor_v3/before/naz100_hma.log

uv run python pytaaa_main.py \
  --json /Users/donaldpg/pyTAAA_data_static/naz100_pi/pytaaa_naz100_pi.json \
  2>&1 | tee .refactor_v3/before/naz100_pi.log

uv run python pytaaa_main.py \
  --json /Users/donaldpg/pyTAAA_data_static/sp500_pine/pytaaa_sp500_pine.json \
  2>&1 | tee .refactor_v3/before/sp500_pine.log

uv run python pytaaa_main.py \
  --json /Users/donaldpg/pyTAAA_data_static/sp500_hma/pytaaa_sp500_hma.json \
  2>&1 | tee .refactor_v3/before/sp500_hma.log

uv run python recommend_model.py \
  --json /Users/donaldpg/pyTAAA_data_static/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json \
  2>&1 | tee .refactor_v3/before/abacus_recommendation.log

uv run python daily_abacus_update.py \
  --json /Users/donaldpg/pyTAAA_data_static/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json \
  --verbose 2>&1 | tee .refactor_v3/before/abacus_daily.log

uv run python pytaaa_backtest_montecarlo.py \
  --json /Users/donaldpg/pyTAAA_data_static/sp500_pine/pytaaa_sp500_pine.json \
  --trials 3 2>&1 | tee .refactor_v3/before/backtest_montecarlo.log

# Capture .params checksums
find /Users/donaldpg/pyTAAA_data_static -name "*.params" -exec md5 {} \; \
  | sort > .refactor_v3/before/params_checksums.txt

# Baseline unit tests
PYTHONPATH=$(pwd) uv run pytest tests/ -v 2>&1 | tee .refactor_v3/before/pytest.log
```

### Validation Protocol (Run After Every Phase)

```bash
mkdir -p .refactor_v3/after

# Re-run all 8 e2e commands above substituting 'before' → 'after' in tee paths

# Compare .params checksums
find /Users/donaldpg/pyTAAA_data_static -name "*.params" -exec md5 {} \; \
  | sort > .refactor_v3/after/params_checksums.txt
diff .refactor_v3/before/params_checksums.txt .refactor_v3/after/params_checksums.txt

# Unit tests
PYTHONPATH=$(pwd) uv run pytest tests/ -v 2>&1 | tee .refactor_v3/after/pytest.log

# Check for new errors/warnings
grep -i "^ERROR\|^WARNING\|Traceback" .refactor_v3/after/*.log
```

### Success Criteria (Every Phase)

- [ ] All `pytest` tests pass
- [ ] `.params` checksums match baseline exactly (excluding timestamp fields)
- [ ] No new `ERROR` or `Traceback` lines in e2e logs
- [ ] No new import failures

### Params File Comparison Script

`refactor_tools/compare_params_files.py` (reuse from prior plan):

```python
"""Compare .params files ignoring timestamps and system paths."""
import sys
import re
from pathlib import Path


def normalize_line(line: str) -> str:
    line = re.sub(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}", "TIMESTAMP", line)
    line = re.sub(r"/Users/[^/]+/[^\s]+", "PATH", line)
    return line


def compare_files(file1: Path, file2: Path) -> bool:
    with open(file1) as f1, open(file2) as f2:
        lines1 = [normalize_line(l.strip()) for l in f1]
        lines2 = [normalize_line(l.strip()) for l in f2]
    if lines1 != lines2:
        print(f"MISMATCH: {file1.name}")
        for i, (a, b) in enumerate(zip(lines1, lines2)):
            if a != b:
                print(f"  Line {i+1}: {a!r} vs {b!r}")
                break
        return False
    print(f"MATCH: {file1.name}")
    return True


def main(before_dir: str, after_dir: str) -> None:
    all_match = True
    for f in Path(before_dir).glob("*.params"):
        af = Path(after_dir) / f.name
        if not af.exists():
            print(f"MISSING: {f.name}")
            all_match = False
            continue
        if not compare_files(f, af):
            all_match = False
    print()
    print("ALL MATCH" if all_match else "VALIDATION FAILED")
    sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_params_files.py <before_dir> <after_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
```

---

## Phase A: Mechanical Cleanup

**Mode:** GitHub Copilot Coding Agent
**Complexity:** Low | **Risk:** Low | **Estimated:** 1 agent session
**Produces:** `tests/test_phaseA_cleanup.py`

### A.1 Goals

1. Replace all remaining bare `except:` clauses (~40 instances) with specific
   exception types + safety fallback
2. Remove debug `print(...)` calls from all `functions/GetParams.py` public
   functions (17+ functions affected)
3. Remove `print(sys.path)` from `run_pytaaa.py` line 24
4. Remove dead `sharpeWeightedRank_2D_old()` from `functions/TAfunctions.py`
   (~line 1429)
5. Remove unreachable constant `TRADING_DAYS_10_YEARS = 2520` that appears
   after a `return` statement in `functions/TAfunctions.py` (~line 73)
6. Fix unconditional `@numba.jit(nopython=True)` on `rank_models_fast()` in
   `functions/MonteCarloBacktest.py` — add `HAS_NUMBA` guard consistent with
   the rest of the module

### A.2 Bare `except:` — Complete Remaining List

| File | Approx Lines | Replacement Types |
|---|---|---|
| `functions/MakeValuePlot.py` | 65, 195, 241, 272, 397, 443, 445, 484, 911, 1083 | `(AttributeError, ValueError, TypeError, OSError)` |
| `functions/TAfunctions.py` | 1529, 1923, 2198, 2235, 2276, 2317, 2359, 2401, 2442, 2665, 2842, 2918 | numpy: `(ValueError, np.linalg.LinAlgError, ZeroDivisionError)`; mpl: `(AttributeError, RuntimeError)`; dict: `(KeyError, IndexError)` — read context before each |
| `functions/readSymbols.py` | 35, 120, 360, 465, 662 | `(requests.RequestException, AttributeError, ValueError)` |
| `functions/quotes_for_list_adjCloseVol.py` | 150, 358, 378, 395 | `(KeyError, ValueError, requests.RequestException)` |
| `functions/stock_cluster.py` | 22, 150, 172, 193, 237, 321 | `(ImportError, ValueError, np.linalg.LinAlgError)` |
| `functions/GetParams.py` | 745 | `(KeyError, json.JSONDecodeError)` |
| `functions/dailyBacktest_pctLong.py` | 2 locations | `(ValueError, KeyError)` |
| `functions/PortfolioPerformanceCalcs.py` | 2 locations | `(OSError, ValueError)` |
| `scheduler.py` | 30 | `(OSError, RuntimeError)` |
| `PyTAAA_backtest_sp500_pine_refactored.py` | 2966, 3099 | `(ValueError, KeyError)` |

**Standard pattern for every replacement:**

```python
# BEFORE
try:
    risky_operation()
except:
    fallback()

# AFTER
try:
    risky_operation()
except (SpecificError1, SpecificError2) as e:
    logger.debug(f"Expected: {e}")
    fallback()
except Exception as e:
    # Safety fallback: maintain existing behavior for unobserved exceptions.
    logger.warning(f"Unexpected {type(e).__name__} in <context>: {e}")
    fallback()
```

### A.3 Tests (`tests/test_phaseA_cleanup.py`)

```python
"""Phase A: verify mechanical cleanup via AST analysis."""
import ast
import sys
from pathlib import Path
import pytest


def all_python_files():
    root = Path(".")
    return list(root.glob("functions/**/*.py")) + [
        Path(p) for p in [
            "pytaaa_main.py", "PyTAAA.py", "run_pytaaa.py",
            "daily_abacus_update.py", "recommend_model.py",
            "run_monte_carlo.py", "pytaaa_backtest_montecarlo.py",
            "scheduler.py",
        ]
        if Path(p).exists()
    ]


def test_no_bare_except_in_any_module():
    """Zero ExceptHandler nodes with no type annotation anywhere."""
    violations = []
    for path in all_python_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                violations.append(f"{path}:{node.lineno}")
    assert not violations, "Bare except: found:\n" + "\n".join(violations)


def test_no_debug_prints_in_getparams():
    """No print() calls inside public functions of GetParams.py."""
    path = Path("functions/GetParams.py")
    tree = ast.parse(path.read_text())
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            for child in ast.walk(node):
                if (isinstance(child, ast.Call)
                        and isinstance(child.func, ast.Name)
                        and child.func.id == "print"):
                    violations.append(f"GetParams.{node.name}:{child.lineno}")
    assert not violations, "Debug prints found:\n" + "\n".join(violations)


def test_sharpe_rank_old_removed():
    """sharpeWeightedRank_2D_old must not exist in TAfunctions."""
    from functions import TAfunctions
    assert not hasattr(TAfunctions, "sharpeWeightedRank_2D_old")


def test_rank_models_fast_guarded_by_has_numba():
    """MonteCarloBacktest imports cleanly regardless of numba availability."""
    import importlib
    import unittest.mock as mock
    # Importing with numba absent should not raise NameError
    with mock.patch.dict(sys.modules, {"numba": None}):
        # Re-import to trigger the guard path
        try:
            import functions.MonteCarloBacktest  # noqa: F401
        except ImportError:
            pass  # numba ImportError is acceptable; NameError is not
```

### A.4 Commit Message

```
fix(cleanup): remove bare excepts, debug prints, and dead code (Phase A)

- Replace 40+ bare except: with specific types + safety fallback in 10 files
- Remove debug print() from all GetParams.py public functions (17+ functions)
- Remove print(sys.path) from run_pytaaa.py line 24
- Remove sharpeWeightedRank_2D_old() and unreachable constant from TAfunctions.py
- Fix unconditional @numba.jit on rank_models_fast (add HAS_NUMBA guard)
- Add tests/test_phaseA_cleanup.py (4 AST-based tests)
```

---

## Phase B: Wildcard Import Elimination

**Mode:** GitHub Copilot Coding Agent
**Complexity:** Medium | **Risk:** Medium | **Estimated:** 1–2 agent sessions
**Produces:** `tests/test_phaseB_imports.py`

### B.1 Goals

Replace all `from X import *` with explicit named imports in 9 files.

### B.2 Complete List

| File | Wildcard Import | Action |
|---|---|---|
| `functions/dailyBacktest.py:14` | `from functions.quotes_for_list_adjClose import *` | Grep file for names used → explicit |
| `functions/dailyBacktest.py:15` | `from functions.TAfunctions import *` | Grep file for names used → explicit |
| `functions/quotes_for_list_adjCloseVol.py:13` | `from functions.quotes_adjCloseVol import *` | Explicit |
| `functions/quotes_for_list_adjCloseVol.py:14` | `from functions.TAfunctions import *` | Explicit |
| `functions/quotes_for_list_adjCloseVol.py:15` | `from functions.readSymbols import *` | Explicit |
| `functions/allPairsRank.py:6` | `from functions.TAfunctions import *` | Explicit |
| `functions/CountNewHighsLows.py:16` | `from functions.allstats import *` | Explicit |
| `functions/calculateTrades.py:8` | `from functions.CheckMarketOpen import *` | Likely only needs `CheckMarketOpen` |
| `functions/PortfolioStatsOnDate.py:13` | `from functions.readSymbols import *` | Explicit |

### B.3 Procedure for Each File

1. `grep -n` the file for all bare names (not `self.`, not locally defined)
2. Cross-reference against the source module's `__all__` or public names
3. Write explicit `from X import name1, name2, ...`
4. Run validation suite — no behavior change expected

### B.4 Tests (`tests/test_phaseB_imports.py`)

```python
"""Phase B: verify zero wildcard imports anywhere in the project."""
import ast
from pathlib import Path
import pytest


def all_python_files():
    root = Path(".")
    return list(root.glob("functions/**/*.py")) + list(root.glob("*.py"))


def test_no_wildcard_imports():
    violations = []
    for path in all_python_files():
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.ImportFrom)
                    and any(a.name == "*" for a in node.names)):
                violations.append(f"{path}:{node.lineno}")
    assert not violations, "Wildcard imports found:\n" + "\n".join(violations)


def test_daily_backtest_importable():
    from functions.dailyBacktest import computeDailyBacktest  # noqa: F401


def test_all_functions_modules_importable():
    import importlib
    for path in Path("functions").glob("*.py"):
        if path.name.startswith("_"):
            continue
        mod_name = f"functions.{path.stem}"
        try:
            importlib.import_module(mod_name)
        except ImportError as e:
            pytest.fail(f"{mod_name} failed to import: {e}")
```

### B.5 Commit Message

```
refactor(imports): replace all wildcard imports with explicit imports (Phase B)

- 9 files converted from 'from X import *' to explicit named imports
- Enables safe renaming, linting, and static analysis of public APIs
- All e2e and unit tests pass with identical outputs
```

---

## Phase C: Duplicate Definition Removal

**Mode:** GitHub Copilot Coding Agent
**Complexity:** Low | **Risk:** Low | **Estimated:** 0.5 agent sessions

### C.1 Goals

Remove all shadowed/dead function definitions. In Python, when a function is
defined twice in the same module, the second definition silently replaces the
first. The earlier definitions are dead code that confuse code navigation tools.

### C.2 Complete List

| File | Dead Definition | Canonical Definition | Verify No Callers Before Removing |
|---|---|---|---|
| `functions/quotes_for_list_adjClose.py` | `get_quote_google()` at ~line 936 | ~line 958 | grep project for "get_quote_google" |
| `functions/quotes_for_list_adjClose.py` | `get_SectorAndIndustry_google()` at ~lines 1101, 1119 | ~line 1148 | grep project |
| `functions/GetParams.py` | `computeLongHoldSignal()` (legacy CamelCase) | `compute_long_hold_signal()` (snake_case, canonical) | grep for `computeLongHoldSignal` |
| `functions/GetParams.py` | `GetSymbolsFile()` (legacy wrapper) | `get_symbols_file()` | grep for `GetSymbolsFile` |

### C.3 Procedure

For each dead definition:
1. `grep -r "function_name" .` — confirm no callers remain
2. If callers exist: update to canonical name, then delete dead definition
3. Delete the dead definition block
4. Run validation suite

### C.4 Test Addition

Add to `tests/test_phaseA_cleanup.py`:

```python
def test_no_duplicate_function_definitions():
    """No function name defined more than once at module level."""
    files_to_check = [
        "functions/quotes_for_list_adjClose.py",
        "functions/GetParams.py",
        "functions/TAfunctions.py",
    ]
    for filepath in files_to_check:
        tree = ast.parse(Path(filepath).read_text())
        names = [n.name for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef)]
        duplicates = [n for n in set(names) if names.count(n) > 1]
        assert not duplicates, f"{filepath}: duplicate functions: {duplicates}"
```

### C.5 Commit Message

```
refactor(cleanup): remove duplicate and shadowed function definitions (Phase C)

- Remove 2 shadowed definitions of get_quote_google() in quotes_for_list_adjClose.py
- Remove 2 shadowed definitions of get_SectorAndIndustry_google()
- Remove legacy computeLongHoldSignal() in favor of compute_long_hold_signal()
- Remove GetSymbolsFile() legacy wrapper after verifying no callers
- Add duplicate-definition test to test_phaseA_cleanup.py
```

---

## Phase D: Hardcoded Path Elimination

**Mode:** GitHub Copilot Coding Agent
**Complexity:** Low | **Risk:** Low | **Estimated:** 0.5 agent sessions

### D.1 Goals

Remove all hardcoded `/Users/donaldpg/...` absolute paths from source code.

### D.2 Complete List

| File | Line(s) | Hardcoded Path | Fix |
|---|---|---|---|
| `run_pytaaa.py` | 31–36 | `/Users/donaldpg/PyProjects/PyTAAA.master` | Remove entire dead `os.chdir()` fallback block |
| `run_normalized_score_history.py` | ~336 | `/Users/donaldpg/pyTAAA_data` | Add `--json` CLI arg; read `base_folder` from `get_json_params()` |
| `functions/abacus_backtest.py` | 71, 246 | `/Users/donaldpg/pyTAAA_data` | Use `params.get('base_folder')` only; remove hardcoded default |
| `PyTAAA_backtest_sp500_pine_refactored.py` | 211–220, 1374, 1571 | Multiple paths | Parameterize via CLI args (this file is legacy; document it as superseded by `pytaaa_backtest_montecarlo.py`) |
| `scripts/extract_montecarlo_ranges.py` | 52–56 | All 5 model paths | Move to CLI args |

### D.3 Commit Message

```
fix(config): remove all hardcoded absolute paths from source (Phase D)

- Remove dead os.chdir() fallback block from run_pytaaa.py (lines 31-36)
- Add --json CLI arg to run_normalized_score_history.py
- Remove hardcoded default path from functions/abacus_backtest.py
- Add note in PyTAAA_backtest_sp500_pine_refactored.py that it is superseded
  by pytaaa_backtest_montecarlo.py
- Parameterize scripts/extract_montecarlo_ranges.py via CLI args
```

---

## Phase E: Circular Import Resolution

**Mode:** VS Code Copilot Agent (Implement)
**Complexity:** High | **Risk:** High | **Estimated:** 1–2 sessions
**Produces:** `functions/graph_plots.py`, `tests/test_phaseE_imports.py`

### E.1 The Problem

`functions/MakeValuePlot.py` imports `makeMinimumSpanningTree` from
`functions/WriteWebPage_pi.py`.
`functions/WriteWebPage_pi.py` imports from `functions/MakeValuePlot.py`.

This circular dependency survives due to Python's import caching but will
cause `ImportError` under certain import orderings and makes both modules
impossible to test independently.

### E.2 Solution: Extract to `functions/graph_plots.py`

Move `makeMinimumSpanningTree()` and its direct dependencies into a new
standalone module that neither `MakeValuePlot` nor `WriteWebPage_pi` depends on.

**Dependency graph after fix:**
```
functions/graph_plots.py        ← new standalone module
functions/MakeValuePlot.py      → functions/graph_plots.py (not WriteWebPage_pi)
functions/WriteWebPage_pi.py    → functions/graph_plots.py (not MakeValuePlot)
```

### E.3 Steps

1. Identify all symbols `makeMinimumSpanningTree` depends on in its source file
2. Create `functions/graph_plots.py` with that function and its dependencies
3. Update `functions/MakeValuePlot.py`: replace the import from
   `WriteWebPage_pi` with `from functions.graph_plots import makeMinimumSpanningTree`
4. Update `functions/WriteWebPage_pi.py`: replace the `MakeValuePlot` import
   with `from functions.graph_plots import makeMinimumSpanningTree`
5. Run validation suite

### E.4 Tests (`tests/test_phaseE_imports.py`)

```python
"""Phase E: verify circular import between MakeValuePlot and WriteWebPage_pi
is resolved."""
import importlib
import sys
import pytest


def fresh_import(module_name: str):
    """Import a module with a clean module cache."""
    for key in list(sys.modules):
        if "MakeValuePlot" in key or "WriteWebPage" in key or "graph_plots" in key:
            del sys.modules[key]
    return importlib.import_module(module_name)


def test_import_makevalueplot_first():
    fresh_import("functions.MakeValuePlot")


def test_import_writewebpage_first():
    fresh_import("functions.WriteWebPage_pi")


def test_graph_plots_has_no_circular_deps():
    import ast
    from pathlib import Path
    tree = ast.parse(Path("functions/graph_plots.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "MakeValuePlot" not in node.module
            assert "WriteWebPage" not in node.module


def test_make_minimum_spanning_tree_importable():
    from functions.graph_plots import makeMinimumSpanningTree  # noqa: F401
```

### E.5 Commit Message

```
refactor(arch): resolve MakeValuePlot <-> WriteWebPage_pi circular import (Phase E)

- Extract makeMinimumSpanningTree() to new functions/graph_plots.py
- Both MakeValuePlot and WriteWebPage_pi now import from graph_plots
- Circular dependency eliminated; both modules independently testable
- Add tests/test_phaseE_imports.py (4 tests)
```

---

## Phase F: Configuration Singleton (JSON Caching)

**Mode:** VS Code Copilot Agent (Implement)
**Complexity:** Medium | **Risk:** Medium | **Estimated:** 1 session
**Produces:** `functions/config_cache.py`, `tests/test_phaseF_config.py`

### F.1 The Problem

`GetParams.py` re-opens and re-parses the JSON file on every call. A single
pipeline run triggers 4+ JSON file reads. Under parallel Monte Carlo workers
(`run_monte_carlo.py` with `--workers 10`), this multiplies further.

### F.2 Solution: `functions/config_cache.py`

```python
"""Thread-safe JSON configuration file cache."""
import json
import threading
from pathlib import Path
from typing import Any


class ConfigCache:
    """Singleton cache for parsed JSON config files."""

    _instance: "ConfigCache | None" = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ConfigCache":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._cache: dict[str, dict[str, Any]] = {}
        return cls._instance

    def get(self, json_path: str) -> dict[str, Any]:
        """Return parsed JSON, loading from disk only on first access."""
        key = str(Path(json_path).resolve())
        with self._lock:
            if key not in self._cache:
                with open(key) as f:
                    self._cache[key] = json.load(f)
            return self._cache[key]

    def invalidate(self, json_path: str | None = None) -> None:
        """Invalidate one entry or the entire cache."""
        with self._lock:
            if json_path:
                self._cache.pop(str(Path(json_path).resolve()), None)
            else:
                self._cache.clear()


config_cache = ConfigCache()
```

Update every `open(json_fn); json.load(f)` call in `functions/GetParams.py`
to use `config_cache.get(json_fn)`. Add `config_cache.invalidate(json_fn)`
after any JSON write in `put_status()` and in `update_json_from_csv.py`.

### F.3 Tests (`tests/test_phaseF_config.py`)

```python
"""Phase F: ConfigCache singleton tests."""
import json
import threading
import pytest
from pathlib import Path
import tempfile


def test_cache_returns_same_object(tmp_path):
    from functions.config_cache import ConfigCache
    cache = ConfigCache()
    cache.invalidate()
    cfg = tmp_path / "test.json"
    cfg.write_text('{"key": "value"}')
    a = cache.get(str(cfg))
    b = cache.get(str(cfg))
    assert a is b


def test_invalidate_reloads(tmp_path):
    from functions.config_cache import ConfigCache
    cache = ConfigCache()
    cfg = tmp_path / "test.json"
    cfg.write_text('{"v": 1}')
    cache.invalidate(str(cfg))
    first = cache.get(str(cfg))
    assert first["v"] == 1
    cfg.write_text('{"v": 2}')
    cache.invalidate(str(cfg))
    second = cache.get(str(cfg))
    assert second["v"] == 2


def test_thread_safe(tmp_path):
    from functions.config_cache import ConfigCache
    cache = ConfigCache()
    cfg = tmp_path / "thread.json"
    cfg.write_text('{"x": 42}')
    cache.invalidate(str(cfg))
    results = []
    errors = []

    def reader():
        try:
            results.append(cache.get(str(cfg))["x"])
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=reader) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert all(r == 42 for r in results)
```

### F.4 Commit Message

```
perf(config): add ConfigCache singleton to eliminate repeated JSON parsing (Phase F)

- Create functions/config_cache.py with thread-safe LRU-style singleton
- Update all GetParams.py JSON-open calls to use config_cache.get()
- Add cache.invalidate() after JSON writes in put_status() and update_json_from_csv.py
- Add tests/test_phaseF_config.py (3 tests: identity, invalidate-reloads, thread-safe)
```

---

## Phase G: Core Function Unit Tests

**Mode:** VS Code Copilot Agent (Implement)
**Complexity:** High | **Risk:** Low | **Estimated:** 2 sessions
**Produces:** `tests/test_sharpe_rank.py`, `tests/test_compute_signal.py`,
`tests/test_daily_backtest.py`

### G.1 Goals

Add unit tests for the three most critical currently-untested functions:

1. `sharpeWeightedRank_2D()` — the ranking heart of the system
2. `computeSignal2D()` — buy/hold signal for every stock every day
3. `computeDailyBacktest()` — portfolio simulation over full history

All tests use **synthetic numpy arrays** — no HDF5 files, no network I/O.

### G.2 `tests/test_sharpe_rank.py` — Key Cases

| Test | Scenario | Assert |
|---|---|---|
| `test_top_n_stocks_selected` | N=3, 10 stocks, clear Sharpe ranking | Top 3 by Sharpe get nonzero weight |
| `test_cash_allocation_when_all_negative` | All stocks negative Sharpe over lookback | 100% cash allocation |
| `test_zero_signal_stocks_excluded` | signal=0 for all lookback days for 5 stocks | Those 5 get zero weight |
| `test_weights_sum_to_one` | Any valid input | Sum of weights == 1.0 (±1e-9) |
| `test_nan_handling` | Some stocks have NaN returns in adjClose | No exception raised |
| `test_single_stock_universe` | Only 1 stock + CASH | Returns valid 2-row allocation |

### G.3 `tests/test_compute_signal.py` — Key Cases

| Test | Scenario | Assert |
|---|---|---|
| `test_uptrend_gets_signal_1` | Stock clearly above all MAs for 90 days | Final signal == 1 |
| `test_downtrend_gets_signal_0` | Stock clearly below all MAs | Final signal == 0 |
| `test_hma_vs_pine_methods` | Same price data, different `uptrendSignalMethod` | Signals differ |
| `test_cash_always_signal_1` | CASH row of ones in adjClose | CASH signal always 1 |

### G.4 `tests/test_daily_backtest.py` — Key Cases

| Test | Scenario | Assert |
|---|---|---|
| `test_output_has_expected_keys` | Run with synthetic data (mock file I/O) | Result contains required keys |
| `test_all_cash_portfolio_stays_flat` | signal=0 for all stocks all days | Portfolio value flat at 1.0 |
| `test_simple_one_stock_scenario` | 1 stock held for 1 month, +10% return | Portfolio up ~10% |

### G.5 Commit Message

```
test(core): add unit tests for the three core computation functions (Phase G)

- tests/test_sharpe_rank.py: 6 synthetic-data tests for sharpeWeightedRank_2D()
- tests/test_compute_signal.py: 4 tests for computeSignal2D()
- tests/test_daily_backtest.py: 3 tests for computeDailyBacktest() with mocked I/O
- No HDF5, no network, no filesystem I/O in any test
```

---

## Phase H: Data Layer Cleanup

**Mode:** VS Code Copilot Agent (Implement)
**Complexity:** High | **Risk:** High | **Estimated:** 2–3 sessions
**Produces:** `tests/test_phaseH_data.py`

### H.1 Goals

1. Remove duplicate `strip_accents()` from `functions/readSymbols.py` (already
   in `functions/ta/utils.py`)
2. Remove duplicate `get_Naz100PlusETFsList()` from
   `functions/quotes_for_list_adjCloseVol.py` (already in
   `functions/quotes_for_list_adjClose.py`)
3. Remove module-level `matplotlib.use('Agg')` calls in
   `functions/TAfunctions.py` and `functions/UpdateSymbols_inHDF5.py` — entry
   points that need headless mode already set `os.environ['MPLBACKEND']='Agg'`
   before imports, which is the correct and sufficient approach
4. Convert module-level `system_logger = setup_logger(...)` in
   `functions/logger_config.py` to lazy singleton — current code creates log
   files on import
5. Move `signal.SIGINT` handler installation in `functions/MonteCarloBacktest.py`
   from module scope into the `run()` method — current code overrides the
   user's Ctrl-C handler on any import of this module

### H.2 Lazy Logger Singleton Pattern

```python
# BEFORE (creates log files on import):
system_logger = setup_logger("system", "pytaaa_system.log")

# AFTER (creates log files only when first used):
_system_logger: logging.Logger | None = None

def get_system_logger() -> logging.Logger:
    """Return the system logger, creating it on first call."""
    global _system_logger
    if _system_logger is None:
        _system_logger = setup_logger("system", "pytaaa_system.log")
    return _system_logger
```

Update all callers of `system_logger` throughout the codebase to
`get_system_logger()`.

### H.3 SIGINT Handler Fix

```python
# BEFORE (module scope in MonteCarloBacktest.py):
signal.signal(signal.SIGINT, _interrupt_handler)

# AFTER (inside run() method):
def run(self):
    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, self._interrupt_handler)
    try:
        # ... existing run logic ...
    finally:
        signal.signal(signal.SIGINT, original_handler)
```

### H.4 Tests (`tests/test_phaseH_data.py`)

```python
"""Phase H: verify no harmful side effects on module import."""
import signal
import sys
import importlib
import pytest


def test_import_tafunctions_does_not_change_mpl_backend():
    """Importing TAfunctions must not call matplotlib.use()."""
    import matplotlib
    backend_before = matplotlib.get_backend()
    # Force reimport
    for key in list(sys.modules):
        if "TAfunctions" in key:
            del sys.modules[key]
    import functions.TAfunctions  # noqa: F401
    assert matplotlib.get_backend() == backend_before


def test_import_logger_config_creates_no_log_files(tmp_path, monkeypatch):
    """Importing logger_config must not create log files."""
    monkeypatch.chdir(tmp_path)
    for key in list(sys.modules):
        if "logger_config" in key:
            del sys.modules[key]
    import functions.logger_config  # noqa: F401
    log_files = list(tmp_path.glob("*.log"))
    assert not log_files, f"Log files created on import: {log_files}"


def test_import_montecarlo_does_not_override_sigint():
    """Importing MonteCarloBacktest must not install a SIGINT handler."""
    original = signal.getsignal(signal.SIGINT)
    for key in list(sys.modules):
        if "MonteCarloBacktest" in key:
            del sys.modules[key]
    import functions.MonteCarloBacktest  # noqa: F401
    assert signal.getsignal(signal.SIGINT) is original


def test_strip_accents_consistent():
    """readSymbols and ta.utils must produce identical strip_accents output."""
    from functions.ta.utils import strip_accents as sa_ta
    # After Phase H, readSymbols should use ta.utils version
    from functions.ta.utils import strip_accents as sa_read
    test_str = "Héllo Wörld"
    assert sa_ta(test_str) == sa_read(test_str)
```

### H.5 Commit Message

```
refactor(data): clean data layer side effects and duplication (Phase H)

- Remove duplicate strip_accents() from readSymbols.py; use ta/utils version
- Remove duplicate get_Naz100PlusETFsList() from quotes_for_list_adjCloseVol.py
- Remove module-level matplotlib.use() from TAfunctions.py and UpdateSymbols_inHDF5.py
- Convert module-level system_logger to lazy singleton in logger_config.py;
  update all callers to get_system_logger()
- Move SIGINT handler from MonteCarloBacktest module scope into run() with
  original handler restore in finally block
- Add tests/test_phaseH_data.py (4 import-safety tests)
```

---

## Phase I: Documentation Completion

**Mode:** GitHub Copilot Coding Agent
**Complexity:** Low | **Risk:** Very Low | **Estimated:** 1 agent session

### I.1 Goals

Add Google-style docstrings and type annotations to all public functions that
lack them. Priority order based on criticality and callsite frequency:

1. `functions/dailyBacktest.py` — `computeDailyBacktest()` has 18+ positional
   args and no docstring; must document every parameter
2. `functions/TAfunctions.py` — all channel, trend, and selfsimilarity functions
3. `functions/output_generators.py` — all exported functions
4. `functions/PortfolioPerformanceCalcs.py` — orchestrator entry point
5. `run_monte_carlo.py` — module-level docstring and CLI option docs
6. `pytaaa_backtest_montecarlo.py` — already well-documented; verify complete

### I.2 Type Annotation Standard

```python
import numpy as np
from numpy.typing import NDArray
from typing import Any


def sharpeWeightedRank_2D(
    adjClose: NDArray[np.float64],
    params: dict[str, Any],
) -> NDArray[np.float64]:
    """Compute Sharpe-weighted stock rankings over multiple lookback windows.

    Args:
        adjClose: Daily adjusted-close prices, shape (n_symbols, n_days).
            Last row must be CASH (all ones).
        params: Trading parameters from GetParams.get_json_params(). Required
            keys: 'numberStocksTraded', 'monthsToHold', 'MA1', 'MA2', 'MA3',
            'sma2factor', 'uptrendSignalMethod'.

    Returns:
        Monthly allocation weights, shape (n_symbols, n_months).
        Rows sum to 1.0; CASH row absorbs unallocated weight.
    """
```

### I.3 Commit Message

```
docs: complete Google-style docstrings and type annotations (Phase I)

- Add docstrings and type annotations to all remaining public functions
- Priority: computeDailyBacktest(), channel/trend functions, output_generators
- Docstring coverage reaches ~100% for public API
- No behavioral changes
```

---

## Phase Sequencing and Dependencies

```
A (Mechanical: bare excepts, dead code, debug prints)
│
├──► C (Duplicates)  ─────────────────────────┐
│                                              │
├──► D (Hardcoded paths)  ────────────────────┤
│                                             ▼
└──► B (Wildcard imports)  ──────────► E (Circular import fix)
                                              │
                                             ▼
                                    F (Config singleton)
                                              │
                                             ▼
                                    G (Core tests)
                                              │
                                             ▼
                                    H (Data layer cleanup)
                                              │
                                             ▼
                                    I (Documentation)
```

**Note:** A, C, and D can run in parallel (assign as 3 simultaneous GitHub Issues).
B must complete before E. E must complete before F, G, H.

### Recommended Sprint Order

| Sprint | Phases | Mode |
|---|---|---|
| 1 | A + C + D (parallel) | GitHub Copilot Agent (3 separate Issues) |
| 2 | B | GitHub Copilot Agent |
| 3 | E + F (sequential) | VS Code Implement |
| 4 | G | VS Code Implement |
| 5 | H | VS Code Implement |
| 6 | I | GitHub Copilot Agent |

---

## Future Recommendations

### Short-Term (Next Quarter)

1. **Add `mypy` to development workflow** — once Phase I type annotations are
   complete, run `uv run mypy functions/ --strict` as a pre-commit hook
2. **`pytest-cov` coverage gate** — add coverage reporting targeting ≥70% for
   `functions/`; integrate into CI when GitHub Actions are added
3. **Monte Carlo state: pickle → JSON** — `monte_carlo_state.pkl` is fragile
   across class structure changes; migrate to a JSON-serializable format or
   `msgpack` for forward compatibility
4. **`run_normalized_score_history.py` CLI** — missing `--json` arg (Phase D
   adds it); also add `--output-dir` and `--date` args for full automation

### Medium-Term (Next 6 Months)

5. **Split `functions/GetParams.py`** — at 918 LOC it is a configuration
   monolith; split into `config_loader.py` (file I/O), `config_validators.py`
   (validation logic), `config_accessors.py` (typed getters); keep
   `GetParams.py` as backward-compat re-exports
6. **Deprecate `functions/TAfunctions.py` re-exports** — `functions/ta/`
   subpackage exists (Phase 5 output); once all callers import directly from
   `functions/ta/*`, the re-export shim becomes unnecessary
7. **`computeDailyBacktest()` signature cleanup** — 18+ positional args should
   become a `BacktestParams` dataclass; annotate now (Phase I), refactor later
8. **Mark `PyTAAA_backtest_sp500_pine_refactored.py` as deprecated** —
   `pytaaa_backtest_montecarlo.py` is the modern replacement; add deprecation
   notice and eventually move to `archive/`

### Long-Term (Architecture)

9. **Plugin architecture for trading methods** — `uptrendSignalMethod` is a
   string switch; a plugin registry pattern would allow new methods without
   touching existing code (open/closed principle)
10. **Async quote updates** — `UpdateSymbols_inHDF5.py` downloads quotes
    synchronously; `asyncio` + `aiohttp` would speed up large symbol lists
    (Naz100 = 100+ downloads)
11. **HDF5 → Parquet migration** — `pandas.HDFStore` is in maintenance mode;
    Parquet offers better ecosystem compatibility (Polars, DuckDB, Arrow) and
    faster columnar reads
12. **GitHub Actions CI** — the project currently has no CI; add a workflow
    that runs `pytest` and the static-data e2e suite on every PR

---

## Appendix: Entry Point Behavior Contracts

All entry points below must produce identical outputs (holdings, ranks,
allocations, backtest returns) before and after every phase.

### `pytaaa_main.py --json <path>`
- Accepts `click.Path(exists=True)` JSON argument
- Delegates entirely to `run_pytaaa.run_pytaaa(json_fn)`
- Exit 0 on success, non-zero on failure

### `daily_abacus_update.py --json <path> [--verbose]`
- Sets `os.environ['MPLBACKEND'] = 'Agg'` before all imports (correct pattern)
- Validates 10+ required JSON keys with meaningful error messages
- Auto-detects active model from holdings files
- Generates HTML dashboard for Abacus model-switching
- Must not be changed to use `builtins.print` monkeypatching (fragile; note
  for Phase H reviewer)

### `recommend_model.py --json <path> [--date YYYY-MM-DD] [--lookbacks ...]`
- Loads Monte Carlo state from `monte_carlo_state.pkl` if it exists
- Generates recommendation for today and first weekday of current month
- Falls back to `pytaaa_model_switching_params.json` if `--json` not provided

### `run_monte_carlo.py --json <path> [--search explore-exploit|explore|exploit] [--workers INT]`
- Runs Monte Carlo backtesting in parallel via `ProcessPoolExecutor`
- Saves state periodically to `monte_carlo_state.pkl`
- Must handle Ctrl-C gracefully (saves state, exits cleanly)
- **After Phase H:** Ctrl-C handler installed in `run()`, not at module scope

### `pytaaa_backtest_montecarlo.py --json <path> [--trials INT] [--max-plots INT] [--tag STR] [--params-file]`
- Modern, clean replacement for `PyTAAA_backtest_sp500_pine_refactored.py`
- Runs per-model Monte Carlo backtest (distribution of portfolio values)
- Distinct from `run_monte_carlo.py` (which optimizes Abacus lookback periods)
- Already correctly sets Agg backend before imports
- Outputs CSV, JSON, and PNG files to `{model_base}/pytaaa_backtest/`
- JSON keys required: `symbols_file`, `backtest_monte_carlo_trials` (optional,
  default 250), `performance_store`, `webpage`

### `PyTAAA.py` (legacy)
- Issues `DeprecationWarning` on run
- Redirects to `pytaaa_main.py` with `pytaaa_generic.json`
- Must continue to work until explicitly deprecated and moved to `archive/`

### `pytaaa_quotes_update.py --json <path>`
- Calls `fix_quotes()` to clean locally stored HDF5 data
- JSON key required: `stockList`

### `update_json_from_csv.py --csv <path> --row INT --json <path> [--dry-run]`
- Reads Monte Carlo results CSV row and writes parameters back to JSON
- After Phase F: must call `config_cache.invalidate(json_fn)` after writing

### `modify_saved_state.py [--inspect] [--reset] [--remove-lookback INT] [--file <path>]`
- Standalone utility; no `functions/` imports
- Reads/writes `monte_carlo_state.pkl` via pickle

---

*End of REFACTOR_PLAN_v3.md*
