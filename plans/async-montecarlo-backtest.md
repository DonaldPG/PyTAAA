# Async Monte Carlo Backtest Implementation Plan

**Feature**: Fire-and-Forget Background Monte Carlo Backtest Generation  
**Branch**: `feature/async-montecarlo-backtest`  
**Base Branch**: `main`  
**Created**: February 24, 2026  
**Status**: Planning

---

## Table of Contents

1. [GitHub Agent Implementation Guide](#github-agent-implementation-guide) ⭐
2. [Background & Context](#background--context)
3. [Problem Statement](#problem-statement)
4. [Proposed Solution](#proposed-solution)
5. [Success Criteria](#success-criteria)
6. [Technical Architecture](#technical-architecture)
7. [Phased Implementation](#phased-implementation)
8. [Testing Strategy](#testing-strategy)
9. [Code Review Checkpoints](#code-review-checkpoints)
10. [Risk Mitigation](#risk-mitigation)
11. [Rollback Plan](#rollback-plan)

---

## GitHub Agent Implementation Guide

### 🤖 What the Agent Can Do (75% Complete Automation)

**Full Implementation:** GitHub Copilot agent can implement **all code** for Phases 0-3:
- ✅ Create and push feature branch
- ✅ Write `functions/background_montecarlo_runner.py` (~200 lines)
- ✅ Refactor `functions/MakeValuePlot.py` (add async mode toggle)
- ✅ Integrate with `functions/WriteWebPage_pi.py`
- ✅ Update JSON configuration templates
- ✅ Write comprehensive documentation
- ✅ Create test automation scripts

**Testing:** Agent can implement and run **29 out of 39 tests (74%)**:
- ✅ All unit tests for new modules
- ✅ All error handling tests (mock/simulate errors)
- ✅ Functional tests with test/mock data
- ✅ Performance tests with small datasets
- ✅ Integration tests with mock configs
- ✅ Regression tests (existing test suite)

**Deliverables:** Agent creates production-ready PR including:
- ✅ Complete feature implementation
- ✅ Comprehensive test suite (29 tests passing)
- ✅ Full documentation with examples
- ✅ Helper scripts for testing/monitoring
- ✅ Commit messages following conventions
- ✅ Code review checkpoints completed

### 🏠 What Requires Local Execution (10 Tests, 26%)

**Local-Only Tests** (requires production environment):
1. **E2E-01, E2E-02**: Full Monte Carlo backtest (12+ trials) — needs production HDF5 data
2. **PERF-01**: Baseline performance benchmark — needs real market data with multiple trials
3. **INT-01 to INT-04**: Production config validation:
   - `pytaaa_model_switching_params.json` (naz100_hma)
   - `pytaaa_naz100_pine.json`
   - `pytaaa_naz100_pi.json`
   - `pytaaa_sp500_hma.json`
4. **REG-03**: Manual smoke test of entire system
5. **Final benchmarks**: Performance validation with production workload (5-10 min completion time)

**Why Local-Only?**
- These tests require access to your HDF5 quote files (`pyTAAA_data/*.h5`)
- Production configs reference specific symbol universes and time periods
- Real market data needed to validate correctness
- Monte Carlo simulations require actual computational load (4-51 trials depending on platform)
- Performance benchmarks require realistic execution time measurements

**Important:** These 10 tests are **validation-only, not implementation blockers**. Agent can create test stubs/fixtures that you run locally after PR is ready.

### 🎯 Recommended Workflow

1. **Create GitHub Issue** with this plan (link to `plans/async-montecarlo-backtest.md`)
2. **Assign to GitHub Copilot Agent** 
3. **Agent Implements**:
   - Phases 0-3 (all code)
   - All automated tests (29 tests)
   - Documentation and scripts
   - Opens PR with all changes
4. **You Review PR**:
   - Code review checkpoints
   - Test coverage completeness
   - Documentation accuracy
5. **You Run Local Tests**:
   - Pull PR branch: `git checkout feature/async-montecarlo-backtest`
   - Run E2E tests: `PYTHONPATH=$(pwd) uv run pytest tests/test_async_montecarlo_e2e.py -v`
   - Validate performance with production configs
   - Verify PNG files appear after 5-10 minutes
6. **Merge if Passing**: All tests green, ready for production

**Estimated Time:**
- Agent implementation: 8 hours (autonomous work)
- Your review: 1 hour
- Your local testing: 0.5 hour (mostly waiting for backtest completion)
- **Total human time: 1.5 hours**

### 📋 Test Distribution Summary

| Category | 🤖 Agent | 🏠 Local | Total |
|----------|----------|----------|-------|
| Functional Tests | 7 | 2 | 9 |
| Error Handling | 8 | 0 | 8 |
| Performance | 5 | 1 | 6 |
| Integration | 5 | 4 | 9 |
| Regression | 4 | 3 | 7 |
| **TOTAL** | **29** | **10** | **39** |

---

## Background & Context

### Current State

PyTAAA generates **Monte Carlo backtest plots** during each execution cycle:
- `PyTAAA_monteCarloBacktest.png` — Full history backtest with trend indicators
- `PyTAAA_monteCarloBacktestRecent.png` — Recent 2-year backtest portion

**Current behavior:**
- Backtest is generated **synchronously** in [functions/MakeValuePlot.py](../functions/MakeValuePlot.py#L551) → `makeDailyMonteCarloBacktest()`
- Called from [functions/WriteWebPage_pi.py](../functions/WriteWebPage_pi.py#L494) during web page generation
- Executes `dailyBacktest_pctLong()` which performs 4-51 Monte Carlo trials (platform-dependent)
- Main program **blocks** for ~5-10 minutes waiting for backtest completion
- Backtest is skipped if plots are less than 20 hours old (smart caching)

**Impact:**
- User waits unnecessarily for program completion
- No incremental value during wait time (web page HTML already created, just waiting for PNG plots)
- Backtest is the last major computation bottleneck after async plot generation was implemented

### Business Need

Enable **fire-and-forget** Monte Carlo backtest generation so:
1. Main program completes immediately (better user experience)
2. Backtest runs in background without blocking
3. Web page updates with tables/holdings immediately
4. Backtest PNG plots appear 5-10 minutes later
5. Logs provide visibility into background process

### Prior Art

The codebase already uses:
- Async plot generation (recently implemented) via `background_plot_generator.py`
- `pickle` for serialization (MonteCarloBacktest, abacus_recommend)
- Fire-and-forget subprocess pattern with `start_new_session=True`
- Log file redirection for background process monitoring

**This implementation reuses the same patterns** proven successful in async plot generation.

---

## Problem Statement

**Primary Goal**: Decouple Monte Carlo backtest generation from main program execution to reduce user wait time from ~8-10 minutes to ~0 seconds.

**Secondary Goals**:
1. Maintain backward compatibility (async mode opt-in/opt-out via JSON config)
2. Provide visibility into background process status via log file
3. Handle errors gracefully without affecting main program
4. Respect 20-hour freshness check (don't regenerate recent plots)

**Constraints**:
- Must work with existing matplotlib-based plotting code
- Must run in headless environment (no display)
- Must clean up temporary files
- No parallel execution needed (unlike stock plots) - single background process
- Must preserve platform-specific trial counts (pi: 12, MacOS: 13, Windows64: 15, etc.)

**Key Difference from Plot Generation:**
- Plot generation benefits from parallel workers (200 plots → ProcessPoolExecutor)
- Monte Carlo backtest is **already parallelized internally** (trials run in parallel)
- Only need **single async subprocess**, not worker pool

---

## Proposed Solution

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│ Main Program (run_pytaaa.py → WriteWebPage)                     │
├─────────────────────────────────────────────────────────────────┤
│ 1. Compute portfolio metrics (Phase 2)                          │
│ 2. Write status files (Phase 3.1-3.2)                           │
│ 3. Generate stock plots async (Phase 3.3)                       │
│ 4. Build web page HTML with reports (Phase 3.4)                 │
│ 5. Generate Monte Carlo backtest (Phase 3.5) — NEW ASYNC POINT │
│    ├─ Check if plots > 20 hours old                            │
│    ├─ Serialize JSON config path to env var                    │
│    ├─ Launch detached process with stdout → montecarlo_log.log │
│    └─ Return immediately (don't wait!)                          │
│ 6. Send email, finalize web page (Phase 3.6)                    │
│ 7. Exit (background process continues independently)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              | Fire-and-forget subprocess
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Background Worker (functions/background_montecarlo_runner.py)   │
├─────────────────────────────────────────────────────────────────┤
│ 1. Parse CLI arguments (--json-file)                            │
│ 2. Call dailyBacktest_pctLong(json_fn)                          │
│    ├─ Load quotes from HDF5                                     │
│    ├─ Run Monte Carlo trials (4-51 depending on platform)       │
│    ├─ Generate 2 PNG plots                                      │
│    └─ Write CSV backtest results                                │
│ 3. Exit (success or error logged)                               │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Background Monte Carlo Runner** (`functions/background_montecarlo_runner.py`)
   - Standalone CLI script invoked via `python -m`
   - Minimal wrapper around existing `dailyBacktest_pctLong()`
   - No parallel workers needed (unlike plot generation)
   - Forces matplotlib Agg backend

2. **Spawn Function** (`functions/MakeValuePlot._spawn_background_montecarlo()`)
   - Checks if plots are > 20 hours old (respect freshness check)
   - Spawns detached subprocess with JSON file path
   - Redirects stdout/stderr to `montecarlo_backtest.log`
   - Returns immediately

3. **Modified Wrapper Function** (`functions/MakeValuePlot.makeDailyMonteCarloBacktest()`)
   - Adds `async_mode` parameter (default: from params dict)
   - Branches to async or sync path based on `async_mode`
   - Sync path: calls `dailyBacktest_pctLong()` directly (existing behavior)

4. **Configuration Parameters** (JSON)
   - `async_montecarlo_backtest` (boolean, default: `true`)
   - **No worker count needed** (single process, internally parallelized)

---

## Success Criteria

### Functional Requirements

✅ **FR1**: Main program completes without waiting for backtest when async mode enabled  
✅ **FR2**: Backtest is generated in background within 5-10 minutes of program completion  
✅ **FR3**: Both PNG files are created successfully (or skipped if fresh)  
✅ **FR4**: Log file captures all progress and errors  
✅ **FR5**: Synchronous mode continues to work unchanged (backward compatibility)  
✅ **FR6**: Configuration is controlled via JSON parameter  
✅ **FR7**: Platform-specific trial counts are preserved (pi: 12, MacOS: 13, Windows64: 15)  
✅ **FR8**: 20-hour freshness check is respected (skip if plots are recent)  

### Non-Functional Requirements

✅ **NFR1**: No zombie processes (background process terminates cleanly)  
✅ **NFR2**: Matplotlib operates in Agg (headless) backend  
✅ **NFR3**: Log file is human-readable with timestamps  
✅ **NFR4**: Errors in backtest generation don't crash main program  
✅ **NFR5**: Background process has access to project root via PYTHONPATH  
✅ **NFR6**: Web page HTML includes plot references (plots may not exist yet)  
✅ **NFR7**: CSV backtest results are written correctly  

### Performance Targets

| Metric | Current | Target (Async) | Measurement |
|--------|---------|----------------|-------------|
| Main program blocking time | ~8-10 min | < 10 sec | Wall clock time |
| Total backtest time (12 trials) | ~8-10 min | ~8-10 min | Log timestamps |
| User perceived wait time | ~8-10 min | ~0 sec | User experience |
| CPU usage | Max available | Same | System monitor |

**Note**: Total computation time is unchanged (backtest still takes 5-10 min), but main program is unblocked.

---

## Technical Architecture

### Data Flow

```python
# Phase 1: Main Program (writeWebPage → makeDailyMonteCarloBacktest)
params, json_fn → spawn subprocess with --json-file flag

# Phase 2: Background Process (background_montecarlo_runner.py)
json_fn → dailyBacktest_pctLong() → load HDF5 → Monte Carlo trials → PNG files + CSV

# Phase 3: Cleanup
Log file closed, background process exits
```

### File System Layout

```
<output_dir>/
├── montecarlo_backtest.log                # Background process log (standard name)
├── PyTAAA_monteCarloBacktest.png          # Generated full-history plot
├── PyTAAA_monteCarloBacktestRecent.png    # Generated recent plot
├── pyTAAAweb_backtestPctLong.params       # CSV backtest results (overwritten)
└── pyTAAAweb_backtestTodayMontecarloPortfolioValue.csv  # Daily values (overwritten)
```

### Process Lifecycle

```bash
# Main program
python run_pytaaa.py --json config.json
  ├─> writeWebPage() → makeDailyMonteCarloBacktest() 
  │   ├─> check if plots > 20 hours old
  │   └─> spawn: python -m functions.background_montecarlo_runner \\
  │                  --json-file config.json > montecarlo_backtest.log 2>&1 &
  └─> continue immediately

# Background process (detached)
python -m functions.background_montecarlo_runner --json-file config.json
  ├─> load params from config.json
  ├─> call dailyBacktest_pctLong(config.json)
  │   ├─> run 12 Monte Carlo trials (platform-dependent)
  │   ├─> generate 2 PNG plots
  │   └─> write CSV results
  └─> exit (log remains)
```

---

## Phased Implementation

### Phase 0: Branch Setup (10 minutes)

**Goal**: Create feature branch from `main`

- [ ] Checkout `main` branch
- [ ] Pull latest changes
- [ ] Verify async plot generation is merged (`ad39390` or later)
- [ ] Create new branch `feature/async-montecarlo-backtest`
- [ ] Verify branch with `git branch --show-current`
- [ ] Push empty branch to remote

**Verification**:
```bash
git checkout main
git pull origin main
git log --oneline -n 1  # Should show ad39390 or later
git checkout -b feature/async-montecarlo-backtest
git push -u origin feature/async-montecarlo-backtest
```

**Code Review Checkpoint**: N/A (no code changes)

---

### Phase 1: Background Monte Carlo Runner Script (2 hours)

**Goal**: Create standalone background worker script

**Files to Create**:
- `functions/background_montecarlo_runner.py` (new, ~200 lines)

**Implementation Checklist**:

- [ ] Create module skeleton with CLI argument parsing
  - [ ] `--json-file` (required)
  - [ ] Parse using `argparse`
- [ ] Implement `main()` function
  - [ ] Parse CLI arguments
  - [ ] Validate JSON file exists
  - [ ] Call `dailyBacktest_pctLong(json_fn, verbose=True)`
  - [ ] Handle exceptions gracefully
  - [ ] Print summary with timestamps
  - [ ] Exit with appropriate codes (0/1/2)
- [ ] Add error handling
  - [ ] Catch FileNotFoundError (JSON or HDF5 missing)
  - [ ] Catch KeyError (missing parameters)
  - [ ] Catch general exceptions with traceback
- [ ] Force matplotlib Agg backend at module level
  - [ ] `matplotlib.use("Agg")` before pyplot import
- [ ] Add logging with timestamps `[HH:MM:SS]`
- [ ] Document CLI interface in docstring
- [ ] Test imports (no actual execution yet)

**Example Module Structure**:
```python
"""Background Monte Carlo backtest runner.

Standalone CLI script for running Monte Carlo backtest in fire-and-forget mode.
"""
import argparse
import logging
import sys
import os
import datetime

# Force Agg backend before importing pyplot
import matplotlib
matplotlib.use("Agg")

from functions.dailyBacktest_pctLong import dailyBacktest_pctLong
from functions.logger_config import get_logger

logger = get_logger(__name__, log_file=None)  # Log to stdout


def main():
    """Entry point for background Monte Carlo backtest runner."""
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo backtest in background"
    )
    parser.add_argument(
        "--json-file",
        required=True,
        help="Path to JSON configuration file"
    )
    args = parser.parse_args()
    
    # ... implementation ...


if __name__ == "__main__":
    main()
```

**Testing Checklist**:

- [ ] **Unit Test**: `test_cli_parsing()` with valid arguments
- [ ] **Unit Test**: `test_cli_parsing()` with missing `--json-file` (exits with error)
- [ ] **Unit Test**: `test_main()` with test config file (mock `dailyBacktest_pctLong`)
- [ ] **Integration Test**: Create test JSON manually, run script
- [ ] **Integration Test**: Verify PNG files created (with mock data)
- [ ] **Integration Test**: Verify log output format
- [ ] **Integration Test**: Test with invalid JSON file (error handling)
- [ ] **Integration Test**: Test with missing HDF5 data (error handling)

**Verification Commands**:
```bash
# Syntax check
python -m py_compile functions/background_montecarlo_runner.py

# Import test
python -c "from functions.background_montecarlo_runner import main"

# Help text
python -m functions.background_montecarlo_runner --help

# Unit tests
PYTHONPATH=$(pwd) uv run pytest tests/test_background_montecarlo_runner.py -v
```

**Commit**:
```bash
git add functions/background_montecarlo_runner.py
git add tests/test_background_montecarlo_runner.py
git commit -m "feat: add background Monte Carlo backtest runner

- Standalone CLI script for async backtest generation
- Minimal wrapper around dailyBacktest_pctLong()
- Graceful error handling with exit codes
- Timestamp logging to stdout
- Forces matplotlib Agg backend
- Platform-specific trial counts preserved
"
```

**Code Review Checkpoint #1**:
- [ ] Review by: _______________ Date: ___________
- [ ] Code style follows PEP 8
- [ ] Docstrings present and accurate
- [ ] Error handling comprehensive
- [ ] No hardcoded paths
- [ ] Matplotlib backend forced to Agg
- [ ] CLI arguments documented in `--help`
- [ ] No resource leaks (files, processes)
- [ ] Tests cover happy path and error cases
- [ ] PYTHONPATH handling mirrors plot generation pattern

---

### Phase 2: Refactor makeDailyMonteCarloBacktest for Async Mode (2 hours)

**Goal**: Add async mode toggle to existing Monte Carlo backtest generation

**Files to Modify**:
- `functions/MakeValuePlot.py` (refactor existing code)

**Implementation Checklist**:

- [ ] Implement `_spawn_background_montecarlo()` helper
  - [ ] Accept `json_fn` and `web_dir` parameters
  - [ ] Set PYTHONPATH environment variable (copy from plot generation pattern)
  - [ ] Build subprocess command with `--json-file` argument
  - [ ] Open log file in write mode: `montecarlo_backtest.log`
  - [ ] Spawn detached subprocess with `start_new_session=True`
  - [ ] Print confirmation messages
  - [ ] Return immediately (no waiting)
  
- [ ] Update `makeDailyMonteCarloBacktest()` signature
  - [ ] Add `async_mode: bool = None` parameter (None = read from params)
  - [ ] Update docstring with new parameters
  
- [ ] Update `makeDailyMonteCarloBacktest()` implementation
  - [ ] Read `async_montecarlo_backtest` from params if `async_mode is None`
  - [ ] Keep existing 20-hour freshness check logic (lines 609-619)
  - [ ] Branch on `async_mode` **after** freshness check:
    - [ ] If `async_mode=True`: call `_spawn_background_montecarlo(json_fn, webpage_dir)`
    - [ ] If `async_mode=False`: call `dailyBacktest_pctLong(json_fn)` (existing behavior)
  - [ ] Return HTML text in both cases (plots may not exist yet in async mode)
  
- [ ] Update imports
  - [ ] Add `import subprocess`
  - [ ] Add `import sys`
  - [ ] Add `import os` (if not already present)

**Testing Checklist**:

- [ ] **Unit Test**: `test_spawn_process_invoked()` (mock subprocess.Popen)
- [ ] **Unit Test**: `test_spawn_sets_pythonpath()` (verify env var)
- [ ] **Unit Test**: `test_spawn_creates_log_file()`
- [ ] **Unit Test**: `test_spawn_returns_immediately()` (no blocking)
- [ ] **Regression Test**: Verify sync mode unchanged (`async_mode=False`)
- [ ] **Regression Test**: Run existing tests with no changes
- [ ] **Integration Test**: Call `makeDailyMonteCarloBacktest()` with `async_mode=True`
- [ ] **Integration Test**: Verify subprocess spawned (ps aux | grep background_montecarlo)
- [ ] **Integration Test**: Verify log file created
- [ ] **Integration Test**: Wait for completion, verify PNG files
- [ ] **Integration Test**: Test freshness check (recent plots → skip)

**Verification Commands**:
```bash
# Import test
python -c "from functions.MakeValuePlot import makeDailyMonteCarloBacktest"

# Syntax check
python -m py_compile functions/MakeValuePlot.py

# Run existing tests (should still pass)
PYTHONPATH=$(pwd) uv run pytest tests/ -v -k MakeValuePlot

# Test async mode (integration)
PYTHONPATH=$(pwd) uv run pytest tests/test_async_montecarlo_integration.py -v
```

**Commit**:
```bash
git add functions/MakeValuePlot.py
git add tests/test_makeDailyMonteCarloBacktest_async.py
git commit -m "feat: add async mode to makeDailyMonteCarloBacktest

- Add _spawn_background_montecarlo() for async mode
- Add async_mode parameter (default from params dict)
- Maintain backward compatibility (async_mode=False)
- Subprocess detachment via start_new_session=True
- Log redirection to montecarlo_backtest.log
- Respect 20-hour freshness check in both modes
- Set PYTHONPATH for subprocess module access
"
```

**Code Review Checkpoint #2**:
- [ ] Review by: _______________ Date: ___________
- [ ] Async/sync branching logic is clear
- [ ] No regression in synchronous mode
- [ ] Subprocess detachment works correctly
- [ ] Log file path is correct (web_dir)
- [ ] PYTHONPATH environment variable set correctly
- [ ] Freshness check respected in both modes
- [ ] Parameters documented in docstring
- [ ] Backward compatibility preserved

---

### Phase 3: Integration with Web Page Generation (1.5 hours)

**Goal**: Wire async Monte Carlo backtest into WriteWebPage pipeline

**Files to Modify**:
- `functions/WriteWebPage_pi.py`
- `pytaaa_generic.json`
- `pytaaa_model_switching_params.json`

**Implementation Checklist**:

- [ ] Update `writeWebPage()` function call to `makeDailyMonteCarloBacktest()`
  - [ ] Currently line 494: `figure6_htmlText = makeDailyMonteCarloBacktest(json_fn)`
  - [ ] No changes needed - async mode read from params dict inside function
  - [ ] Add comment explaining async behavior
  
- [ ] Update `pytaaa_generic.json` template
  - [ ] Add `"async_montecarlo_backtest": true` (default enabled per user request)
  - [ ] Add comment explaining the parameter
  - [ ] Place near `async_plot_generation` for consistency
  
- [ ] Update `pytaaa_model_switching_params.json` (production config)
  - [ ] Add `"async_montecarlo_backtest": true`
  - [ ] Ensures production config has the parameter
  
- [ ] Test parameter loading
  - [ ] Verify GetParams reads new parameter
  - [ ] Verify default value handled correctly
  - [ ] Verify override works

**Testing Checklist**:

- [ ] **Unit Test**: `test_params_loading_async_false()`
- [ ] **Unit Test**: `test_params_loading_async_true()`
- [ ] **Unit Test**: `test_params_loading_default_when_missing()` (should default to True)
- [ ] **Integration Test**: Run full pipeline with `async_mode=False`
- [ ] **Integration Test**: Run full pipeline with `async_mode=True`
- [ ] **Integration Test**: Verify main program completes quickly
- [ ] **Integration Test**: Verify PNG plots appear after 5-10 minutes
- [ ] **Integration Test**: Check log file for errors
- [ ] **Integration Test**: Verify CSV results written correctly

**Verification Commands**:
```bash
# Test sync mode (baseline)
cat > test_config.json << 'EOF'
{
  "async_montecarlo_backtest": false,
  "stockList": "Naz100",
  ...
}
EOF
time PYTHONPATH=$(pwd) uv run python run_pytaaa.py --json test_config.json
# Should wait ~8-10 minutes

# Test async mode
cat > test_config.json << 'EOF'
{
  "async_montecarlo_backtest": true,
  "stockList": "Naz100",
  ...
}
EOF
time PYTHONPATH=$(pwd) uv run python run_pytaaa.py --json test_config.json
# Should complete in < 3 minutes (no Monte Carlo wait)

# Check background process
ps aux | grep background_montecarlo

# Monitor log
tail -f <web_dir>/montecarlo_backtest.log

# Wait and verify plots created
ls -l <web_dir>/PyTAAA_montecarlo*.png
```

**Commit**:
```bash
git add functions/WriteWebPage_pi.py
git add pytaaa_generic.json
git add pytaaa_model_switching_params.json
git add tests/test_async_montecarlo_integration.py
git commit -m "feat: integrate async Monte Carlo backtest with web page generation

- Add async_montecarlo_backtest parameter to JSON configs (default: true)
- Update writeWebPage() to support async mode via params
- Add integration tests for full pipeline
- Document usage in pytaaa_generic.json
- Enable by default per user request
"
```

**Code Review Checkpoint #3**:
- [ ] Review by: _______________ Date: ___________
- [ ] JSON parameter added to all relevant configs
- [ ] Default value is True (opt-out, per user request)
- [ ] Integration with WriteWebPage is clean
- [ ] Documentation explains async behavior
- [ ] Tests cover both sync and async modes
- [ ] Backward compatibility maintained (missing param → default True)

---

### Phase 4: Documentation and Final Testing (1.5 hours)

**Goal**: Create comprehensive documentation and validate production readiness

**Files to Create/Modify**:
- `docs/ASYNC_MONTECARLO_BACKTEST.md` (new)
- `docs/DAILY_OPERATIONS_GUIDE.md` (update)
- `README.md` (update)

**Implementation Checklist**:

- [ ] Create `docs/ASYNC_MONTECARLO_BACKTEST.md`
  - [ ] Feature overview
  - [ ] Configuration guide
  - [ ] Troubleshooting section
  - [ ] Log file monitoring
  - [ ] Performance characteristics
  - [ ] Comparison with sync mode
  
- [ ] Update `docs/DAILY_OPERATIONS_GUIDE.md`
  - [ ] Add section on async Monte Carlo backtest
  - [ ] Explain when plots appear vs. other outputs
  - [ ] Add troubleshooting tips
  
- [ ] Update `README.md`
  - [ ] Mention async Monte Carlo feature
  - [ ] Link to detailed documentation
  
- [ ] Create monitoring helper script (optional)
  - [ ] `scripts/monitor_montecarlo_backtest.sh`
  - [ ] Check if background process is running
  - [ ] Tail log file
  - [ ] Show plot file modification times

**Testing Checklist**:

- [ ] **E2E-01**: Run full PyTAAA pipeline with `pytaaa_model_switching_params.json` (async=true)
- [ ] **E2E-02**: Verify main program completes in < 3 minutes
- [ ] **E2E-03**: Verify PNG plots appear within 10 minutes
- [ ] **E2E-04**: Verify CSV results correct
- [ ] **E2E-05**: Run again with async=false, verify behavior unchanged
- [ ] **REG-01**: Run all existing tests (should pass)
- [ ] **REG-02**: Run pytaaa_main.py with 4 production configs (int tests)
- [ ] **REG-03**: Manual smoke test - verify web page, email, all outputs
- [ ] **PERF-01**: Measure main program completion time (should be < 3 min)
- [ ] **PERF-02**: Measure total backtest time (should be 5-10 min, unchanged)

**Verification Commands**:
```bash
# Run all tests
PYTHONPATH=$(pwd) uv run pytest tests/ -v

# E2E test with production config
time PYTHONPATH=$(pwd) uv run python pytaaa_main.py \\
    --json pytaaa_model_switching_params.json

# Monitor background process
ps aux | grep background_montecarlo
tail -f <web_dir>/montecarlo_backtest.log

# Wait and verify outputs
sleep 600  # 10 minutes
ls -lh <web_dir>/PyTAAA_montecarlo*.png
cat <web_dir>/pyTAAAweb_backtestPctLong.params | head
```

**Commit**:
```bash
git add docs/ASYNC_MONTECARLO_BACKTEST.md
git add docs/DAILY_OPERATIONS_GUIDE.md
git add README.md
git add scripts/monitor_montecarlo_backtest.sh  # if created
git commit -m "docs: add comprehensive async Monte Carlo backtest documentation

- Document feature overview and configuration
- Update daily operations guide
- Add troubleshooting section
- Create monitoring helper script
- Update README with feature mention
"
```

**Code Review Checkpoint #4**:
- [ ] Review by: _______________ Date: ___________
- [ ] Documentation is clear and comprehensive
- [ ] Troubleshooting section covers common issues
- [ ] Daily operations guide updated
- [ ] Monitoring tools are helpful
- [ ] All E2E tests passing
- [ ] Performance targets met

---

### Phase 5: Create GitHub Issue and Open PR (30 minutes)

**Goal**: Open PR for review and create GitHub issue for agent assignment

**Implementation Checklist**:

- [ ] Create GitHub issue with template:
  - [ ] Title: "Implement async Monte Carlo backtest generation"
  - [ ] Link to this plan document
  - [ ] List deliverables
  - [ ] Assign to GitHub Copilot agent (if using)
  
- [ ] Open PR from `feature/async-montecarlo-backtest` to `main`
  - [ ] Use PR template
  - [ ] Link to GitHub issue
  - [ ] List all changes
  - [ ] Include test results summary
  - [ ] Request code review
  
- [ ] Update PR with local test results
  - [ ] Run production config tests locally
  - [ ] Post results in PR comment
  - [ ] Verify PNG plots appear

**PR Template**:
```markdown
## Summary
Implements fire-and-forget async Monte Carlo backtest generation to reduce main program runtime from ~8-10 minutes to ~2 minutes.

## Changes
- New module: `functions/background_montecarlo_runner.py`
- Modified: `functions/MakeValuePlot.py` (async mode toggle)
- Modified: `functions/WriteWebPage_pi.py` (integration)
- Modified: `pytaaa_generic.json`, `pytaaa_model_switching_params.json` (config)
- Tests: 29 automated tests passing, 10 local validation tests

## Testing
- ✅ All automated tests passing (29/29)
- ✅ Local E2E tests with production configs (4/4)
- ✅ Performance targets met (main program < 3 min)
- ✅ Backward compatibility verified (sync mode unchanged)

## Documentation
- Comprehensive feature documentation in `docs/ASYNC_MONTECARLO_BACKTEST.md`
- Updated daily operations guide
- Monitoring helper script

## Related
- Closes #<issue_number>
- Follows patterns from #24 (async plot generation)
```

**Verification Commands**:
```bash
# Push branch
git push origin feature/async-montecarlo-backtest

# Create PR (using GitHub CLI)
gh pr create --base main --head feature/async-montecarlo-backtest \\
    --title "feat: Async Monte Carlo backtest generation" \\
    --body-file pr_template.md

# Run final validation
PYTHONPATH=$(pwd) uv run pytest tests/ -v
```

**Code Review Checkpoint #5**:
- [ ] Review by: _______________ Date: ___________
- [ ] PR description is clear and complete
- [ ] All automated tests passing
- [ ] Local validation tests passing
- [ ] Documentation reviewed
- [ ] No breaking changes
- [ ] Ready to merge

---

## Testing Strategy

### Unit Tests (15 tests)

**background_montecarlo_runner.py** (8 tests):
- `test_cli_parsing_valid()`: Valid arguments parsed correctly
- `test_cli_parsing_missing_json()`: Missing `--json-file` exits with error
- `test_main_calls_dailyBacktest()`: Verify `dailyBacktest_pctLong()` called (mock)
- `test_main_handles_file_not_found()`: JSON file missing handled gracefully
- `test_main_handles_hdf5_missing()`: HDF5 file missing handled gracefully
- `test_main_handles_key_error()`: Missing JSON params handled gracefully
- `test_main_logs_timestamps()`: Log output includes timestamps
- `test_main_exit_codes()`: Correct exit codes (0=success, 1=error, 2=invalid args)

**MakeValuePlot.py** (7 tests):
- `test_spawn_subprocess_invoked()`: `subprocess.Popen()` called with correct args (mock)
- `test_spawn_sets_pythonpath()`: PYTHONPATH env var set correctly
- `test_spawn_creates_log_file()`: Log file opened in write mode
- `test_spawn_returns_immediately()`: Function returns without blocking
- `test_async_mode_true_spawns()`: `async_mode=True` triggers spawn
- `test_async_mode_false_calls_direct()`: `async_mode=False` calls `dailyBacktest_pctLong()` directly
- `test_async_mode_from_params()`: `async_mode=None` reads from params dict

### Integration Tests (9 tests)

**INT-01 to INT-04**: Production config validation (local only):
- `pytaaa_model_switching_params.json` (naz100_hma)
- `pytaaa_naz100_pine.json`
- `pytaaa_naz100_pi.json`
- `pytaaa_sp500_hma.json`

**INT-05**: Async mode end-to-end (agent can automate with test data):
- Run full pipeline with `async_montecarlo_backtest=true`
- Verify subprocess spawned
- Verify log file created
- Verify PNG files created (after wait)

**INT-06**: Sync mode regression:
- Run full pipeline with `async_montecarlo_backtest=false`
- Verify behavior unchanged from baseline

**INT-07**: Freshness check respected:
- Create fresh PNG plots (< 20 hours old)
- Run pipeline, verify backtest skipped

**INT-08**: Missing parameter defaults to True:
- Remove `async_montecarlo_backtest` from config
- Verify defaults to True (opt-out)

**INT-09**: Error handling in background process:
- Simulate HDF5 file missing
- Verify error logged, process exits cleanly

### End-to-End Tests (7 tests)

**E2E-01**: Full PyTAAA pipeline with async=true (local only):
- Run `pytaaa_main.py` with production config
- Measure main program completion time (< 3 min)
- Verify PNG plots appear within 10 minutes

**E2E-02**: Full PyTAAA pipeline with async=false (local only):
- Run `pytaaa_main.py` with production config
- Measure total completion time (~10 min)
- Verify PNG plots exist immediately

**E2E-03**: CSV results correctness (agent can automate with test data):
- Compare async vs sync mode CSV outputs
- Verify identical results

**E2E-04**: Web page HTML correctness (agent can automate):
- Verify HTML includes plot references
- Verify HTML generated even if plots don't exist yet

**E2E-05**: Log file monitoring (agent can automate):
- Tail log file during backtest
- Verify progress messages appear
- Verify no errors logged

**E2E-06**: Platform-specific trial counts (local only):
- Verify pi/MacOS/Windows use correct trial counts
- Check log for trial count messages

**E2E-07**: Abacus backtest portfolio update (agent can automate with mock):
- Verify `write_abacus_backtest_portfolio_values()` called for abacus configs
- Verify column 3 updated correctly

### Regression Tests (7 tests)

**REG-01**: Existing test suite passes (agent can automate):
- Run `pytest tests/ -v`
- All tests pass

**REG-02**: Sync mode unchanged (agent can automate):
- Run with `async_montecarlo_backtest=false`
- Compare outputs to baseline

**REG-03**: Manual smoke test (local only):
- Run full PyTAAA with production config
- Verify web page, email, all outputs

**REG-04**: Plot generation still works (agent can automate):
- Verify async plot generation not affected
- Both features work together

**REG-05**: Email sent correctly (agent can automate with mock):
- Verify email sent before backtest completes
- Email contains correct holdings/trades

**REG-06**: Status file updated (agent can automate):
- Verify `put_status()` called before wait
- Cumulative value saved correctly

**REG-07**: No zombie processes (agent can automate):
- Run multiple times in succession
- Verify no orphaned background processes

### Performance Tests (6 tests)

**PERF-01**: Baseline time measurement (local only):
- Measure sync mode completion time (baseline)
- Measure async mode main program time (target < 3 min)

**PERF-02**: Background process time (agent can automate with small dataset):
- Measure backtest generation time (should be ~same as sync)

**PERF-03**: CPU usage (agent can monitor):
- Verify main program doesn't wait for CPU
- Background process uses available CPU

**PERF-04**: Memory usage (agent can monitor):
- Verify no memory leaks in subprocess spawn
- Background process memory reasonable

**PERF-05**: Log file size (agent can automate):
- Verify log doesn't grow unbounded
- Overwrite mode prevents accumulation

**PERF-06**: Subprocess spawn overhead (agent can automate):
- Measure time to spawn subprocess (< 1 sec)
- Verify negligible overhead

### Error Handling Tests (8 tests)

**ERR-01**: JSON file not found:
- Test with invalid path
- Verify error message logged

**ERR-02**: HDF5 file not found:
- Remove HDF5 file temporarily
- Verify error handled gracefully

**ERR-03**: Missing JSON parameter:
- Remove required param
- Verify error or default used

**ERR-04**: Invalid JSON syntax:
- Corrupt JSON file
- Verify parsing error handled

**ERR-05**: Matplotlib import error:
- Mock matplotlib import failure
- Verify error logged

**ERR-06**: Subprocess spawn failure:
- Mock `subprocess.Popen()` failure
- Verify main program continues

**ERR-07**: Background process crash:
- Simulate crash in `dailyBacktest_pctLong()`
- Verify logged, doesn't affect main program

**ERR-08**: Permission denied on log file:
- Mock log file write failure
- Verify error handled

---

## Code Review Checkpoints

### Checkpoint #1: Background Runner Module
- [ ] Code style follows PEP 8
- [ ] Docstrings present and accurate
- [ ] Error handling comprehensive
- [ ] No hardcoded paths
- [ ] Matplotlib backend forced to Agg
- [ ] CLI arguments documented
- [ ] Tests cover error cases

### Checkpoint #2: MakeValuePlot Refactoring
- [ ] Async/sync branching logic clear
- [ ] No regression in sync mode
- [ ] Subprocess detachment correct
- [ ] PYTHONPATH set correctly
- [ ] Freshness check respected
- [ ] Backward compatible

### Checkpoint #3: WriteWebPage Integration
- [ ] JSON params added correctly
- [ ] Default value is True (opt-out)
- [ ] Integration clean
- [ ] Tests cover both modes

### Checkpoint #4: Documentation
- [ ] Documentation clear
- [ ] Troubleshooting comprehensive
- [ ] Daily ops guide updated
- [ ] E2E tests passing

### Checkpoint #5: PR Review
- [ ] All tests passing
- [ ] Documentation complete
- [ ] No breaking changes
- [ ] Ready to merge

---

## Risk Mitigation

### Risk 1: Background process fails silently
**Mitigation**: 
- Comprehensive error handling in background runner
- All errors logged to `montecarlo_backtest.log`
- Exit codes indicate success/failure
- Tests verify error logging

### Risk 2: PNG plots never appear
**Mitigation**:
- 20-hour freshness check ensures regeneration
- Log file shows progress and errors
- Manual monitoring script available
- Tests verify plot creation

### Risk 3: Race condition with web page generation
**Mitigation**:
- HTML references plots that may not exist yet (browser shows broken image)
- User can refresh page after 5-10 minutes
- Freshness check prevents stale plots
- No data corruption possible (files are independent)

### Risk 4: PYTHONPATH not set correctly
**Mitigation**:
- Copy proven pattern from async plot generation
- Tests verify PYTHONPATH in subprocess env
- Local validation catches issues early

### Risk 5: Platform-specific trial counts lost
**Mitigation**:
- No changes to `dailyBacktest_pctLong()` internals
- Platform detection logic unchanged
- Tests verify trial counts correct

### Risk 6: CSV results corrupted
**Mitigation**:
- No changes to CSV writing logic
- Tests compare async vs sync outputs
- File writes are atomic (OS-level)

### Risk 7: Zombie processes
**Mitigation**:
- Use `start_new_session=True` for clean detachment
- Tests verify no orphaned processes
- Background process exits cleanly on error

### Risk 8: Log file grows unbounded
**Mitigation**:
- Use write mode ('w') not append mode ('a')
- Each run overwrites previous log
- No accumulation over time

---

## Rollback Plan

### If Issues Found After Merge

1. **Quick disable**: Set `async_montecarlo_backtest: false` in JSON configs
   - No code changes needed
   - Reverts to proven sync behavior
   - Can be done in < 1 minute

2. **Git revert**: Revert the merge commit
   ```bash
   git revert <merge_commit_hash>
   git push origin main
   ```

3. **Emergency patch**: If specific bug found, create hotfix branch
   - Fix the issue
   - Fast-track PR review
   - Merge to main

### Monitoring After Deployment

1. **Check log files**: `tail -f <web_dir>/montecarlo_backtest.log`
2. **Verify plots**: `ls -l <web_dir>/PyTAAA_montecarlo*.png`
3. **Check processes**: `ps aux | grep background_montecarlo`
4. **Compare results**: CSV outputs should match sync mode

### Success Metrics (First Week)

- Main program completes in < 3 minutes (vs ~10 min baseline)
- PNG plots appear within 10 minutes of program completion
- No errors in `montecarlo_backtest.log`
- CSV results identical to sync mode
- No zombie processes
- No user complaints about missing plots

---

## Appendix A: Example Log Output

### Sync Mode (Current)
```
[09:45:00] Starting daily backtest with 12 Monte Carlo trials
[09:45:02] Trial 1/12 complete
[09:45:04] Trial 2/12 complete
...
[09:53:15] Trial 12/12 complete
[09:53:17] Generating PNG plots
[09:53:35] PyTAAA_monteCarloBacktest.png created
[09:53:38] PyTAAA_monteCarloBacktestRecent.png created
[09:53:40] CSV results written
```

### Async Mode (New)
**Main program log**:
```
[09:45:00] Spawning background Monte Carlo backtest
[09:45:01] Background process started, continuing...
[09:45:02] --> Log: /web/montecarlo_backtest.log
[09:45:03] Sending email with portfolio updates
[09:45:10] Program complete (no wait for backtest)
```

**Background log** (`montecarlo_backtest.log`):
```
[09:45:01] Background Monte Carlo backtest starting
[09:45:02] Config: /path/to/pytaaa_model_switching_params.json
[09:45:03] Platform: MacOS, trials: 13
[09:45:05] Trial 1/13 complete
[09:45:07] Trial 2/13 complete
...
[09:53:42] Trial 13/13 complete
[09:53:45] Generating PNG plots
[09:54:03] PyTAAA_monteCarloBacktest.png created
[09:54:06] PyTAAA_monteCarloBacktestRecent.png created
[09:54:08] CSV results written
[09:54:09] Background backtest complete (success)
```

---

## Appendix B: Configuration Examples

### Enable Async Mode (Default)
```json
{
  "async_montecarlo_backtest": true,
  "async_plot_generation": true,
  "stockList": "Naz100",
  ...
}
```

### Disable Async Mode (Legacy)
```json
{
  "async_montecarlo_backtest": false,
  "async_plot_generation": false,
  "stockList": "Naz100",
  ...
}
```

### Async Backtest, Sync Plots
```json
{
  "async_montecarlo_backtest": true,
  "async_plot_generation": false,
  "stockList": "Naz100",
  ...
}
```

---

## Appendix C: Troubleshooting

### PNG plots never appear
1. Check if background process is running: `ps aux | grep background_montecarlo`
2. Check log file: `cat <web_dir>/montecarlo_backtest.log`
3. Verify HDF5 files exist: `ls pyTAAA_data/*.h5`
4. Check disk space: `df -h`

### Main program still waits
1. Verify `async_montecarlo_backtest: true` in JSON config
2. Check for typos in parameter name
3. Verify GetParams reads the parameter correctly
4. Check if freshness check skipped backtest (plots < 20 hours old)

### CSV results differ from sync mode
1. This should never happen - file a bug report
2. Compare trial counts in logs (should match platform)
3. Verify HDF5 data unchanged
4. Check for race conditions (unlikely)

### Zombie processes
1. List all background processes: `ps aux | grep background_montecarlo`
2. Kill manually if needed: `kill <pid>`
3. File a bug report (should auto-cleanup)
4. Verify `start_new_session=True` in spawn code

### Log file too large
1. Verify write mode ('w') not append mode ('a')
2. Check if multiple instances running
3. Truncate manually if needed: `> montecarlo_backtest.log`

---

**End of Plan**
