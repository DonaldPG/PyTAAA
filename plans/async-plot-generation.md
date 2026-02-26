# Async Plot Generation Implementation Plan

**Feature**: Fire-and-Forget Background Plot Generation with Parallel Workers  
**Branch**: `feature/async-plot-generation`  
**Base Branch**: `copilot/review-update-docstrings-comments`  
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

### 🤖 What the Agent Can Do (79% Complete Automation)

**Full Implementation:** GitHub Copilot agent can implement **all code** for Phases 0-4:
- ✅ Create and push feature branch
- ✅ Write `functions/background_plot_generator.py` (~400 lines)
- ✅ Refactor `functions/output_generators.py` (extract helpers, add async mode)
- ✅ Integrate with `functions/PortfolioPerformanceCalcs.py`
- ✅ Update JSON configuration templates
- ✅ Write comprehensive documentation
- ✅ Create test automation scripts

**Testing:** Agent can implement and run **38 out of 48 tests (79%)**:
- ✅ All unit tests for new modules
- ✅ All error handling tests (mock/simulate errors)
- ✅ Functional tests with test/mock data
- ✅ Performance tests with small datasets
- ✅ Integration tests with mock configs
- ✅ Regression tests (existing test suite)

**Deliverables:** Agent creates production-ready PR including:
- ✅ Complete feature implementation
- ✅ Comprehensive test suite (38 tests passing)
- ✅ Full documentation with examples
- ✅ Helper scripts for testing/monitoring
- ✅ Commit messages following conventions
- ✅ Code review checkpoints completed

### 🏠 What Requires Local Execution (10 Tests, 21%)

**Local-Only Tests** (requires production environment):
1. **E2E-02, E2E-04**: Full dataset tests (100 symbols) — needs production HDF5 data
2. **PERF-01**: Baseline performance benchmark — needs 100 symbols with real market data
3. **INT-01 to INT-04**: Production config validation:
   - `pytaaa_model_switching_params.json` (naz100_hma)
   - `pytaaa_naz100_pine.json` 
   - `pytaaa_naz100_pi.json`
   - `pytaaa_sp500_hma.json`
4. **REG-03**: Manual smoke test of entire system
5. **Final benchmarks**: Performance validation with production workload

**Why Local-Only?**
- These tests require access to your HDF5 quote files (`pyTAAA_data/*.h5`)
- Production configs reference specific symbol universes and time periods
- Real market data needed to validate correctness
- Performance benchmarks require actual computational load

**Important:** These 10 tests are **validation-only, not implementation blockers**. Agent can create test stubs/fixtures that you run locally after PR is ready.

### 🎯 Recommended Workflow

1. **Create GitHub Issue** with this plan (link to `plans/async-plot-generation.md`)
2. **Assign to GitHub Copilot Agent** 
3. **Agent Implements**:
   - Phases 0-4 (all code)
   - Phase 5 agent-runnable tests (38 tests)
   - Creates test stubs for local-only tests
   - Opens PR with all changes
4. **You Review PR**:
   - Code review checkpoints
   - Review test coverage (should be ~80%)
   - Check documentation completeness
5. **You Run Local Tests**:
   - Pull PR branch: `git checkout feature/async-plot-generation`
   - Run INT-01 to INT-04 with production configs
   - Run full performance benchmarks
   - Validate E2E-02, E2E-04 with 100 symbols
6. **Merge if Passing**: All tests green, ready for production

**Estimated Time:**
- Agent implementation: 14 hours (autonomous work)
- Your review: 1-2 hours
- Your local testing: 0.5-1 hour
- **Total human time: 1.5-3 hours**

### 📋 Test Distribution Summary

| Category | 🤖 Agent | 🏠 Local | Total |
|----------|----------|----------|-------|
| Functional Tests | 11 | 2 | 13 |
| Error Handling | 10 | 0 | 10 |
| Performance | 8 | 1 | 9 |
| Integration | 5 | 4 | 9 |
| Regression | 4 | 3 | 7 |
| **TOTAL** | **38** | **10** | **48** |

---

## Background & Context

### Current State

PyTAAA generates approximately **200 PNG plot files** (2 per symbol) during each execution cycle:
- `0_<symbol>.png` — Full history plots
- `0_recent_<symbol>.png` — Recent 2-year plots with trend channels

**Current behavior:**
- Plots are generated **synchronously** in [functions/output_generators.py](../functions/output_generators.py#L79)
- Main program **blocks** for ~5-10 minutes waiting for plot generation
- Plots are only generated outside market hours (1am-1pm, excluding 8am-11am)
- Plots are skipped if less than 20 hours old (smart caching)

**Impact:**
- User waits unnecessarily for program completion
- No value delivered during wait time (HTML pages already created)
- Sequential generation is slower than necessary

### Business Need

Enable **fire-and-forget** plot generation so:
1. Main program completes immediately (better user experience)
2. Plots are generated in parallel (faster overall completion)
3. System remains responsive during plot generation
4. Logs provide visibility into background process

### Prior Art

The codebase already uses:
- `pickle` for serialization (MonteCarloBacktest, abacus_recommend)
- Background processes (not currently for plots)
- Worker pools (not currently applied)

---

## Problem Statement

**Primary Goal**: Decouple plot generation from main program execution to reduce user wait time from ~10 minutes to ~0 seconds.

**Secondary Goals**:
1. Parallelize plot generation to reduce overall completion time (10 min → 2-5 min with 4 workers)
2. Maintain backward compatibility (async mode opt-in via JSON config)
3. Provide visibility into background process status via log file
4. Handle errors gracefully without affecting main program

**Constraints**:
- Must work with existing matplotlib-based plotting code
- Must respect 20-hour freshness check (don't regenerate recent plots)
- Must run in headless environment (no display)
- Must clean up temporary files
- Default to 2 parallel workers (conservative CPU usage)

---

## Proposed Solution

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│ Main Program (run_pytaaa.py → PortfolioPerformanceCalcs)       │
├─────────────────────────────────────────────────────────────────┤
│ 1. Compute portfolio metrics (Phase 2)                          │
│ 2. Write status files (Phase 3.1-3.2)                           │
│ 3. Spawn background plot generation (Phase 3.3)                 │
│    ├─ Serialize data to /tmp/pytaaa_plots_XXXXX.pkl            │
│    ├─ Launch detached process with stdout → plot_generation.log│
│    └─ Return immediately (don't wait!)                          │
│ 4. Continue with reports, emails, webpage (Phase 3.4-3.5)      │
│ 5. Exit (background process continues independently)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Detached process
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Background Worker (functions/background_plot_generator.py)      │
├─────────────────────────────────────────────────────────────────┤
│ 1. Load pickle file from /tmp                                   │
│ 2. Create ProcessPoolExecutor (max_workers=2)                   │
│ 3. Submit plot generation tasks                                 │
│ 4. Wait for all tasks to complete                               │
│ 5. Clean up pickle file                                         │
│ 6. Exit                                                          │
└─────────────────────────────────────────────────────────────────┘
         │ Worker 1        │ Worker 2
         ▼                 ▼
    Generate plots     Generate plots
    (50 symbols)       (50 symbols)
```

### Key Components

1. **Background Plot Generator** (`functions/background_plot_generator.py`)
   - Standalone CLI script invoked via `python -m`
   - Accepts `--data-file` and `--max-workers` arguments
   - Logs to stdout (redirected by caller)
   - Uses `ProcessPoolExecutor` for parallel execution

2. **Spawn Function** (`functions/output_generators._spawn_background_plot_generation()`)
   - Serializes plot data to temporary pickle file
   - Opens log file: `<output_dir>/plot_generation.log`
   - Launches subprocess with `start_new_session=True` (detachment)
   - Returns immediately

3. **Modified Main Function** (`functions/output_generators.generate_portfolio_plots()`)
   - Adds `async_mode` parameter (default: `False`)
   - Adds `max_workers` parameter (default: `2`)
   - Branches to async or sync path based on `async_mode`

4. **Configuration Parameters** (JSON)
   - `async_plot_generation` (boolean, default: `false`)
   - `plot_generation_workers` (integer, default: `2`)

---

## Success Criteria

### Functional Requirements

✅ **FR1**: Main program completes without waiting for plots when async mode enabled  
✅ **FR2**: Plots are generated in background within 5-10 minutes of program completion  
✅ **FR3**: All 200 PNG files are created successfully (or skipped if fresh)  
✅ **FR4**: Log file captures all progress and errors  
✅ **FR5**: Parallel workers reduce overall plot generation time  
✅ **FR6**: Synchronous mode continues to work unchanged (backward compatibility)  
✅ **FR7**: Configuration is controlled via JSON parameters  

### Non-Functional Requirements

✅ **NFR1**: No zombie processes (background process terminates cleanly)  
✅ **NFR2**: Temporary pickle files are cleaned up after load  
✅ **NFR3**: Matplotlib operates in Agg (headless) backend  
✅ **NFR4**: Process pool workers don't exceed configured limit  
✅ **NFR5**: Log file is human-readable with timestamps  
✅ **NFR6**: Errors in plot generation don't crash main program  
✅ **NFR7**: CPU usage is reasonable (default 2 workers)  

### Performance Targets

| Metric | Current | Target (Async) | Measurement |
|--------|---------|----------------|-------------|
| Main program blocking time | ~10 min | < 10 sec | Wall clock time |
| Total plot generation time (100 plots) | ~10 min | ~5 min (2 workers) | Log timestamps |
| User perceived wait time | ~10 min | ~0 sec | User experience |
| CPU cores used | 1 | 2-4 | System monitor |

---

## Technical Architecture

### Data Flow

```python
# Phase 1: Main Program
adjClose, symbols, datearray, signal2D, signal2D_daily, params → pickle file

# Phase 2: Background Process
pickle file → load → ProcessPoolExecutor → [Worker1, Worker2, ...] → PNG files

# Phase 3: Cleanup
pickle file deleted, log file closed
```

### File System Layout

```
/tmp/
├── pytaaa_plots_ABC123.pkl    # Temporary data (deleted after load)

<output_dir>/
├── plot_generation.log         # Background process log (standard name)
├── 0_AAPL.png                  # Generated plots
├── 0_recent_AAPL.png
├── 0_MSFT.png
├── 0_recent_MSFT.png
└── ...
```

### Process Lifecycle

```bash
# Main program
python run_pytaaa.py --json config.json
  ├─> spawn: python -m functions.background_plot_generator \\
  │            --data-file /tmp/pytaaa_plots_ABC.pkl \\
  │            --max-workers 2 \\
  │            > /path/to/web_dir/plot_generation.log 2>&1 &
  └─> continue immediately

# Background process (detached)
python -m functions.background_plot_generator
  ├─> load pickle
  ├─> start ProcessPoolExecutor(max_workers=2)
  ├─> submit 200 tasks (100 full history + 100 recent)
  ├─> wait for completion
  ├─> clean up pickle
  └─> exit (log remains)
```

---

## Phased Implementation

### Phase 0: Branch Setup (15 minutes)

**Goal**: Create feature branch from correct base

- [ ] Checkout `copilot/review-update-docstrings-comments`
- [ ] Pull latest changes
- [ ] Create new branch `feature/async-plot-generation`
- [ ] Verify branch with `git branch --show-current`
- [ ] Push empty branch to remote

**Verification**:
```bash
git checkout copilot/review-update-docstrings-comments
git pull origin copilot/review-update-docstrings-comments
git checkout -b feature/async-plot-generation
git push -u origin feature/async-plot-generation
```

**Code Review Checkpoint**: N/A (no code changes)

---

### Phase 1: Background Plot Generator Script (4 hours)

**Goal**: Create standalone background worker script with parallel execution

**Files to Create**:
- `functions/background_plot_generator.py` (new, ~400 lines)

**Implementation Checklist**:

- [ ] Create module skeleton with CLI argument parsing
  - [ ] `--data-file` (required)
  - [ ] `--max-workers` (default: 2)
- [ ] Implement `load_plot_data()` function
  - [ ] Load pickle file
  - [ ] Validate required keys
  - [ ] Return data dict
- [ ] Implement `_should_regenerate_plot()` function
  - [ ] Check file existence
  - [ ] Check modification time (20-hour threshold)
  - [ ] Return boolean
- [ ] Implement `generate_single_full_history_plot()` function
  - [ ] Accept (symbol_index, data_dict) tuple
  - [ ] Import matplotlib with Agg backend
  - [ ] Generate one full-history plot
  - [ ] Save to PNG file
  - [ ] Return (idx, symbol, success, error_msg)
  - [ ] Handle exceptions gracefully
- [ ] Implement `generate_single_recent_plot()` function
  - [ ] Accept (symbol_index, data_dict) tuple
  - [ ] Import matplotlib with Agg backend
  - [ ] Fit trend channels
  - [ ] Generate one recent plot
  - [ ] Save to PNG file
  - [ ] Return (idx, symbol, success, error_msg)
  - [ ] Handle exceptions gracefully
- [ ] Implement `main()` function
  - [ ] Parse CLI arguments
  - [ ] Load plot data
  - [ ] Create ProcessPoolExecutor(max_workers)
  - [ ] Submit full-history plot tasks
  - [ ] Wait for completion, log results
  - [ ] Submit recent plot tasks
  - [ ] Wait for completion, log results
  - [ ] Clean up pickle file
  - [ ] Print summary with timestamps
- [ ] Add error handling
  - [ ] Catch FileNotFoundError
  - [ ] Catch pickle errors
  - [ ] Catch matplotlib errors
  - [ ] Exit with appropriate codes (0/1/2)
- [ ] Force matplotlib Agg backend at module level
- [ ] Add logging with timestamps `[HH:MM:SS]`
- [ ] Test imports (no actual execution yet)

**Testing Checklist**:

- [ ] **Unit Test**: `test_load_plot_data()` with valid pickle
- [ ] **Unit Test**: `test_load_plot_data()` with invalid pickle (KeyError)
- [ ] **Unit Test**: `test_should_regenerate_plot()` with fresh file
- [ ] **Unit Test**: `test_should_regenerate_plot()` with old file
- [ ] **Unit Test**: `test_should_regenerate_plot()` with missing file
- [ ] **Integration Test**: Create test pickle manually, run script
- [ ] **Integration Test**: Verify PNG files created
- [ ] **Integration Test**: Verify log output format
- [ ] **Integration Test**: Test with `--max-workers 1`
- [ ] **Integration Test**: Test with `--max-workers 4`
- [ ] **Integration Test**: Test with invalid data file (error handling)

**Verification Commands**:
```bash
# Syntax check
python -m py_compile functions/background_plot_generator.py

# Import test
python -c "from functions.background_plot_generator import main"

# Help text
python -m functions.background_plot_generator --help

# Unit tests
PYTHONPATH=$(pwd) uv run pytest tests/test_background_plot_generator.py -v
```

**Commit**:
```bash
git add functions/background_plot_generator.py
git add tests/test_background_plot_generator.py
git commit -m "feat: add background plot generator with parallel workers

- Standalone CLI script for async plot generation
- ProcessPoolExecutor for parallel execution
- Configurable max_workers (default: 2)
- Graceful error handling with exit codes
- Timestamp logging to stdout
- 20-hour freshness check respected
"
```

**Code Review Checkpoint #1**:
- [ ] Review by: _______________ Date: ___________
- [ ] Code style follows PEP 8
- [ ] Docstrings present and accurate
- [ ] Error handling comprehensive
- [ ] No hardcoded paths
- [ ] Matplotlib backend forced to Agg
- [ ] ProcessPoolExecutor used correctly
- [ ] No resource leaks (files, processes)
- [ ] Tests cover happy path and error cases

---

### Phase 2: Extract and Refactor Plot Generation (3 hours)

**Goal**: Refactor existing plot generation code to be callable from background worker

**Files to Modify**:
- `functions/output_generators.py` (refactor existing code)

**Implementation Checklist**:

- [ ] Extract `_generate_full_history_plots()` from existing code
  - [ ] Move lines 144-198 into new function
  - [ ] Add proper function signature with all parameters
  - [ ] Add docstring (Google style)
  - [ ] No changes to logic, just extraction
- [ ] Extract `_generate_recent_plots()` from existing code
  - [ ] Move lines 199-323 into new function
  - [ ] Add proper function signature with all parameters
  - [ ] Add docstring (Google style)
  - [ ] No changes to logic, just extraction
- [ ] Implement `_spawn_background_plot_generation()` helper
  - [ ] Accept all plot data as parameters
  - [ ] Add `max_workers` parameter (default: 2)
  - [ ] Create temporary pickle file with `tempfile.mkstemp()`
  - [ ] Serialize data dict to pickle
  - [ ] Print serialization stats (size, symbols)
  - [ ] Prepare log file path: `<output_dir>/plot_generation.log`
  - [ ] Launch subprocess with `Popen()`
  - [ ] Redirect stdout/stderr to log file
  - [ ] Use `start_new_session=True` for detachment
  - [ ] Return immediately (don't wait)
  - [ ] Print confirmation messages
- [ ] Update `generate_portfolio_plots()` signature
  - [ ] Add `async_mode: bool = False` parameter
  - [ ] Add `max_workers: int = 2` parameter
  - [ ] Update docstring with new parameters
- [ ] Update `generate_portfolio_plots()` implementation
  - [ ] Add async mode branch at top (after time-of-day check)
  - [ ] If `async_mode=True`: call `_spawn_background_plot_generation()` and return
  - [ ] If `async_mode=False`: call extracted functions (existing behavior)
- [ ] Update imports
  - [ ] Add `import subprocess`
  - [ ] Add `import pickle`
  - [ ] Add `import tempfile`
  - [ ] Add `import sys`

**Testing Checklist**:

- [ ] **Unit Test**: `test_spawn_creates_pickle_file()`
- [ ] **Unit Test**: `test_spawn_creates_log_file()`
- [ ] **Unit Test**: `test_spawn_returns_immediately()`
- [ ] **Regression Test**: Verify sync mode unchanged (async_mode=False)
- [ ] **Regression Test**: Run existing tests with no changes
- [ ] **Integration Test**: Call `generate_portfolio_plots(async_mode=True)`
- [ ] **Integration Test**: Verify pickle file created
- [ ] **Integration Test**: Verify subprocess spawned (ps aux | grep background_plot)
- [ ] **Integration Test**: Verify log file created
- [ ] **Integration Test**: Wait for completion, verify PNG files

**Verification Commands**:
```bash
# Import test
python -c "from functions.output_generators import generate_portfolio_plots"

# Syntax check
python -m py_compile functions/output_generators.py

# Run existing tests (should still pass)
PYTHONPATH=$(pwd) uv run pytest tests/ -v -k output_generators

# Test async spawn (integration)
PYTHONPATH=$(pwd) uv run pytest tests/test_async_integration.py -v
```

**Commit**:
```bash
git add functions/output_generators.py
git add tests/test_output_generators_async.py
git commit -m "feat: add async mode to generate_portfolio_plots

- Extract _generate_full_history_plots() helper
- Extract _generate_recent_plots() helper
- Add _spawn_background_plot_generation() for async mode
- Add async_mode and max_workers parameters
- Maintain backward compatibility (async_mode=False default)
- Subprocess detachment via start_new_session=True
- Log redirection to plot_generation.log
"
```

**Code Review Checkpoint #2**:
- [ ] Review by: _______________ Date: ___________
- [ ] Extracted functions maintain identical behavior
- [ ] No regression in synchronous mode
- [ ] Pickle serialization includes all required data
- [ ] Subprocess detachment works correctly
- [ ] Log file path is correct (output_dir)
- [ ] Temporary file cleanup handled by background process
- [ ] Parameters documented in docstring
- [ ] Backward compatibility preserved

---

### Phase 3: Integration with Main Pipeline (2 hours)

**Goal**: Wire async plot generation into PortfolioPerformanceCalcs

**Files to Modify**:
- `functions/PortfolioPerformanceCalcs.py`
- `pytaaa_generic.json`

**Implementation Checklist**:

- [ ] Update `PortfolioPerformanceCalcs()` function
  - [ ] Read `async_plot_generation` from params (default: False)
  - [ ] Read `plot_generation_workers` from params (default: 2)
  - [ ] Pass to `generate_portfolio_plots()` calls (both branches)
  - [ ] Add comment explaining async behavior
- [ ] Update `pytaaa_generic.json` template
  - [ ] Add `"async_plot_generation": false`
  - [ ] Add `"plot_generation_workers": 2`
  - [ ] Add explanatory comments for both parameters
  - [ ] Place near other performance-related settings
- [ ] Test parameter loading
  - [ ] Verify GetParams reads new parameters
  - [ ] Verify defaults applied when missing
  - [ ] Verify override works

**Testing Checklist**:

- [ ] **Unit Test**: `test_params_loading_async_false()`
- [ ] **Unit Test**: `test_params_loading_async_true()`
- [ ] **Unit Test**: `test_params_loading_workers_custom()`
- [ ] **Unit Test**: `test_params_loading_defaults_when_missing()`
- [ ] **Integration Test**: Run full pipeline with async_mode=False
- [ ] **Integration Test**: Run full pipeline with async_mode=True
- [ ] **Integration Test**: Verify main program completes quickly
- [ ] **Integration Test**: Verify plots appear after 5-10 minutes
- [ ] **Integration Test**: Check log file for errors
- [ ] **Integration Test**: Verify correct number of workers used

**Verification Commands**:
```bash
# Test sync mode (baseline)
cat > test_config.json << 'EOF'
{
  "async_plot_generation": false,
  "plot_generation_workers": 2
}
EOF
time PYTHONPATH=$(pwd) uv run python run_pytaaa.py --json test_config.json
# Should wait ~10 minutes

# Test async mode
cat > test_config.json << 'EOF'
{
  "async_plot_generation": true,
  "plot_generation_workers": 2
}
EOF
time PYTHONPATH=$(pwd) uv run python run_pytaaa.py --json test_config.json
# Should complete in < 1 minute

# Check background process
ps aux | grep background_plot_generator

# Monitor log
tail -f <web_dir>/plot_generation.log
```

**Commit**:
```bash
git add functions/PortfolioPerformanceCalcs.py
git add pytaaa_generic.json
git commit -m "feat: integrate async plot generation into main pipeline

- Read async_plot_generation from JSON config
- Read plot_generation_workers from JSON config
- Pass parameters to generate_portfolio_plots()
- Add config parameters to pytaaa_generic.json template
- Default behavior unchanged (async_mode=false)
"
```

**Code Review Checkpoint #3**:
- [ ] Review by: _______________ Date: ___________
- [ ] JSON parameters properly documented
- [ ] Parameter defaults sensible
- [ ] Integration doesn't break existing workflows
- [ ] Both sync and async modes tested
- [ ] Error handling appropriate

---

### Phase 4: Documentation and Testing Scripts (2 hours)

**Goal**: Document feature and create helper scripts for testing/monitoring

**Files to Create/Modify**:
- `docs/PYTAAA_SYSTEM_SUMMARY_AND_JSON_GUIDE.md` (update)
- `scripts/test_async_plots.sh` (new)
- `scripts/monitor_plot_generation.sh` (new)

**Implementation Checklist**:

- [ ] Update documentation in `docs/PYTAAA_SYSTEM_SUMMARY_AND_JSON_GUIDE.md`
  - [ ] Add "Asynchronous Plot Generation" section
  - [ ] Document `async_plot_generation` parameter
  - [ ] Document `plot_generation_workers` parameter
  - [ ] Explain behavior differences (sync vs async)
  - [ ] Show log file location and monitoring
  - [ ] Provide example configurations
  - [ ] Add performance benchmarks table
  - [ ] Add troubleshooting section
- [ ] Create `scripts/test_async_plots.sh`
  - [ ] Enable async mode in JSON
  - [ ] Run main program
  - [ ] Verify quick completion
  - [ ] Check for background process
  - [ ] Monitor log file
  - [ ] Wait for completion
  - [ ] Verify PNG files created
  - [ ] Restore original config
- [ ] Create `scripts/monitor_plot_generation.sh`
  - [ ] Find latest log file
  - [ ] Show summary (total, completed, errors)
  - [ ] Tail last 20 lines
  - [ ] Show estimated completion time
  - [ ] Check for background process

**Testing Checklist**:

- [ ] **Documentation Test**: Run through examples manually
- [ ] **Script Test**: Run `test_async_plots.sh` successfully
- [ ] **Script Test**: Run `monitor_plot_generation.sh` during generation
- [ ] **Usability Test**: Ask colleague to follow documentation
- [ ] **Readability Test**: Review documentation clarity

**Verification Commands**:
```bash
# Test documentation examples
# (follow examples from PYTAAA_SYSTEM_SUMMARY_AND_JSON_GUIDE.md)

# Test scripts
chmod +x scripts/test_async_plots.sh
chmod +x scripts/monitor_plot_generation.sh
./scripts/test_async_plots.sh
./scripts/monitor_plot_generation.sh
```

**Commit**:
```bash
git add docs/PYTAAA_SYSTEM_SUMMARY_AND_JSON_GUIDE.md
git add scripts/test_async_plots.sh
git add scripts/monitor_plot_generation.sh
git commit -m "docs: document async plot generation feature

- Add comprehensive documentation to system guide
- Create test_async_plots.sh for automated testing
- Create monitor_plot_generation.sh for monitoring
- Add troubleshooting section
- Include performance benchmarks
"
```

**Code Review Checkpoint #4**:
- [ ] Review by: _______________ Date: ___________
- [ ] Documentation clear and comprehensive
- [ ] Examples accurate and tested
- [ ] Scripts executable and functional
- [ ] Troubleshooting section helpful

---

### Phase 5: End-to-End Testing and Validation (3 hours)

**Goal**: Comprehensive testing across all scenarios

**Test Environment Markers**:
- 🤖 **Agent-Runnable**: Can be implemented and executed by GitHub Copilot agent in CI/test environment with mock/test data
- 🏠 **Local-Only**: Requires local execution with production environment (HDF5 data, production configs, real market data)

**Testing Checklist**:

#### Functional Tests
- [ ] 🤖 **E2E-01**: Sync mode with small dataset (10 symbols) — *Use test data*
- [ ] 🏠 **E2E-02**: Sync mode with full dataset (100 symbols) — *Requires production HDF5*
- [ ] 🤖 **E2E-03**: Async mode with small dataset (10 symbols) — *Use test data*
- [ ] 🏠 **E2E-04**: Async mode with full dataset (100 symbols) — *Requires production HDF5*
- [ ] 🤖 **E2E-05**: Async mode with 1 worker — *Use test data*
- [ ] 🤖 **E2E-06**: Async mode with 2 workers (default) — *Use test data*
- [ ] 🤖 **E2E-07**: Async mode with 4 workers — *Use test data*
- [ ] 🤖 **E2E-08**: Async mode with 8 workers — *Use test data*
- [ ] 🤖 **E2E-09**: Fresh plots (all regenerated) — *Mock timestamp checks*
- [ ] 🤖 **E2E-10**: Cached plots (all skipped, < 20 hours old) — *Mock fresh files*
- [ ] 🤖 **E2E-11**: Mixed plots (some fresh, some cached) — *Mock mixed timestamps*
- [ ] 🤖 **E2E-12**: percentileChannels method (with lowChannel/hiChannel) — *Use test data*
- [ ] 🤖 **E2E-13**: Other signal methods (without channels) — *Use test data*

#### Error Handling Tests
- [ ] 🤖 **ERR-01**: Missing pickle file — *Delete test pickle*
- [ ] 🤖 **ERR-02**: Corrupted pickle file — *Write invalid pickle*
- [ ] 🤖 **ERR-03**: Invalid data in pickle — *Pickle with wrong structure*
- [ ] 🤖 **ERR-04**: Missing output directory — *Use non-existent path*
- [ ] 🤖 **ERR-05**: No write permissions to output dir — *chmod 000 test dir*
- [ ] 🤖 **ERR-06**: Out of disk space (simulate) — *Mock disk full error*
- [ ] 🤖 **ERR-07**: matplotlib import error (simulate) — *Mock import failure*
- [ ] 🤖 **ERR-08**: Worker process crash (simulate) — *Raise exception in worker*
- [ ] 🤖 **ERR-09**: Main program killed mid-spawn — *Send SIGKILL to parent*
- [ ] 🤖 **ERR-10**: Background process killed mid-generation — *Send SIGKILL to worker*

#### Performance Tests
- [ ] 🏠 **PERF-01**: Measure sync mode time (100 symbols) — *Requires production data*
- [ ] 🤖 **PERF-02**: Measure async spawn time — *Small test dataset*
- [ ] 🤖 **PERF-03**: Measure async total time (1 worker) — *Small test dataset*
- [ ] 🤖 **PERF-04**: Measure async total time (2 workers) — *Small test dataset*
- [ ] 🤖 **PERF-05**: Measure async total time (4 workers) — *Small test dataset*
- [ ] 🤖 **PERF-06**: Verify speedup with more workers — *Small test dataset*
- [ ] 🤖 **PERF-07**: Monitor CPU usage — *psutil monitoring*
- [ ] 🤖 **PERF-08**: Monitor memory usage — *psutil monitoring*
- [ ] 🤖 **PERF-09**: Check for memory leaks (long run) — *Extended test run*

#### Integration Tests
- [ ] 🏠 **INT-01**: Run with naz100_hma config — *Requires production config + HDF5 data*
- [ ] 🏠 **INT-02**: Run with naz100_pine config — *Requires production config + HDF5 data*
- [ ] 🏠 **INT-03**: Run with naz100_pi config — *Requires production config + HDF5 data*
- [ ] 🏠 **INT-04**: Run with sp500_hma config — *Requires production config + HDF5 data*
- [ ] 🤖 **INT-05**: Verify HTML pages reference correct PNG files — *Mock HTML + files*
- [ ] 🤖 **INT-06**: Verify webpage loads correctly after async generation — *Mock data*
- [ ] 🤖 **INT-07**: Multiple concurrent runs (different configs) — *Test configs*
- [ ] 🤖 **INT-08**: Run during market hours (plots skipped) — *Mock time check*
- [ ] 🤖 **INT-09**: Run outside market hours (plots generated) — *Mock time check*

#### Regression Tests
- [ ] 🤖 **REG-01**: Existing unit tests still pass — *pytest existing tests*
- [ ] 🤖 **REG-02**: Existing integration tests still pass — *pytest if exist in repo*
- [ ] 🏠 **REG-03**: Manual smoke test of entire system — *Full production workflow*
- [ ] 🤖 **REG-04**: Backtest functionality unaffected — *Run test backtest*
- [ ] 🤖 **REG-05**: Monte Carlo functionality unaffected — *Run test Monte Carlo*
- [ ] 🤖 **REG-06**: Email sending unaffected — *Mock email check*
- [ ] 🤖 **REG-07**: Webpage generation unaffected — *Verify HTML created*

**GitHub Agent Implementation Note**:
- Agent can implement and run all 🤖 tests (38 out of 48 tests)
- Agent should create test stubs/fixtures for 🏠 tests (10 tests)
- Local validation of 🏠 tests required before production deployment
- 🏠 tests are **not blockers** for PR creation—they validate with production data post-merge

**Pytest Marker Strategy**:

All tests should be marked with pytest markers for filtering:

```python
import pytest

@pytest.mark.agent_runnable
def test_e2e_01_sync_mode_small_dataset():
    """Sync mode with small dataset (10 symbols) - Use test data."""
    # Test implementation using mock/test data
    pass

@pytest.mark.local_only
def test_int_01_naz100_hma_config():
    """Run with naz100_hma config - Requires production config + HDF5 data."""
    # Test stub - requires production environment
    pytest.skip("Requires production HDF5 data - run locally")
```

**Running Tests Selectively**:

```bash
# Agent: Run only agent-runnable tests (CI/automated)
pytest tests/ -v -m agent_runnable

# Local: Run only local tests (production validation)
pytest tests/ -v -m local_only

# Local: Run all tests (full validation)
pytest tests/ -v

# Agent: Run with coverage (automated CI)
pytest tests/ -v -m agent_runnable --cov=functions --cov-report=term-missing
```

**Configure pytest.ini**:

```ini
[tool:pytest]
markers =
    agent_runnable: Tests that can be run by GitHub agent with mock/test data
    local_only: Tests that require production environment (HDF5, configs)
```

**Test Execution Log**:
```
Test ID  | Status | Time (s) | Notes
---------|--------|----------|----------------------------------
E2E-01   | PASS   | 45       | All 20 plots generated
E2E-02   | PASS   | 612      | All 200 plots generated
E2E-03   | PASS   | 5        | Main program, 48s background
E2E-04   | PASS   | 8        | Main program, 324s background
...
```

**Performance Benchmarks**:
```
Configuration            | Main (s) | Background (s) | Total (s) | Speedup
-------------------------|----------|----------------|-----------|--------
Sync, 100 symbols        | 612      | N/A            | 612       | 1.0x
Async, 1 worker          | 8        | 598            | 606       | 1.0x
Async, 2 workers         | 8        | 324            | 332       | 1.8x
Async, 4 workers         | 8        | 168            | 176       | 3.5x
Async, 8 workers         | 8        | 92             | 100       | 6.1x
```

**Commit**:
```bash
git add tests/test_e2e_async_plots.py
git commit -m "test: add comprehensive E2E tests for async plot generation

- 13 functional tests covering all scenarios
- 10 error handling tests
- 9 performance benchmarks
- 9 integration tests with production configs
- 7 regression tests
- Performance benchmarks documented
"
```

**Code Review Checkpoint #5 (Final)**:
- [ ] Review by: _______________ Date: ___________
- [ ] All tests passing
- [ ] Performance targets met
- [ ] No regressions detected
- [ ] Error handling comprehensive
- [ ] Ready for production deployment

---

## Testing Strategy

### Test Pyramid

```
        ┌─────────────────┐
        │  E2E Tests (10) │  ← Full system, real data
        └─────────────────┘
            ┌───────────────────┐
            │ Integration (15)  │  ← Multiple components
            └───────────────────┘
                ┌─────────────────────┐
                │  Unit Tests (30)    │  ← Individual functions
                └─────────────────────┘
```

### Agent-Runnable vs Local-Only Test Distribution

**Total Tests: 48**

| Category | 🤖 Agent-Runnable | 🏠 Local-Only | Subtotal |
|----------|-------------------|----------------|----------|
| Functional Tests (E2E) | 11 | 2 | 13 |
| Error Handling Tests (ERR) | 10 | 0 | 10 |
| Performance Tests (PERF) | 8 | 1 | 9 |
| Integration Tests (INT) | 5 | 4 | 9 |
| Regression Tests (REG) | 4 | 3 | 7 |
| **TOTAL** | **38 (79%)** | **10 (21%)** | **48** |

**Key Insights:**
- GitHub Copilot agent can implement and run **79% of all tests**
- Only **10 tests** require production environment (HDF5 data, production configs)
- **None of the 10 local-only tests are blockers** for code implementation
- Local tests validate behavior with production data, not implementation correctness

**Agent Can Complete:**
- ✅ All implementation (Phases 0-4)
- ✅ All unit tests (background_plot_generator, output_generators, integration)
- ✅ All error handling tests (mock/simulate errors)
- ✅ Functional tests with mock data (11 out of 13)
- ✅ Performance tests with small datasets (8 out of 9)
- ✅ Most integration tests (5 out of 9)
- ✅ Most regression tests (4 out of 7)

**Local Validation Required:**
- 🏠 E2E-02, E2E-04: Full dataset tests (100 symbols)
- 🏠 PERF-01: Baseline performance benchmark (100 symbols)
- 🏠 INT-01 to INT-04: Production config validation
- 🏠 REG-03: Manual smoke test
- 🏠 Final performance benchmarks with production data

### Test Coverage Goals

| Component | Target | Actual |
|-----------|--------|--------|
| background_plot_generator.py | 85% | ___ |
| output_generators.py | 80% | ___ |
| PortfolioPerformanceCalcs.py | 75% | ___ |
| Overall | 80% | ___ |

### Continuous Testing

**GitHub Agent (Automated):**
1. Run unit tests: `pytest tests/ -v`
2. Run integration tests: `pytest tests/integration/ -v`
3. Run agent-runnable E2E tests: `pytest tests/e2e/ -v -m agent_runnable`
4. Check code coverage: `pytest --cov=functions tests/`

**After each phase:**
1. Agent runs all 🤖 tests in CI
2. Agent commits passing code
3. Code review checkpoint
4. Proceed to next phase

**After all phases (Local):**
1. Pull PR branch locally
2. Run 🏠 local-only tests with production data
3. Run full E2E test suite: `pytest tests/e2e/ -v`
4. Run performance benchmarks: `./scripts/benchmark_async_plots.sh`
5. Manual testing with production configs (INT-01 to INT-04)
6. Load testing (multiple concurrent runs)
7. Validate results, merge PR

---

## Code Review Checkpoints

### Review Process

Each phase ends with a code review checkpoint. Reviewer should check:

**Code Quality**:
- [ ] Follows PEP 8 style guide
- [ ] Type hints present where appropriate
- [ ] Docstrings follow Google style
- [ ] Comments explain *why*, not *what*
- [ ] No commented-out code
- [ ] No debug print statements
- [ ] No hardcoded values

**Functionality**:
- [ ] Implementation matches specification
- [ ] Edge cases handled
- [ ] Error handling comprehensive
- [ ] No obvious bugs
- [ ] Logic is clear and maintainable

**Testing**:
- [ ] Unit tests present and passing
- [ ] Integration tests present and passing
- [ ] Test coverage adequate (> 80%)
- [ ] Tests are meaningful (not just for coverage)
- [ ] Error cases tested

**Documentation**:
- [ ] Code is self-documenting
- [ ] Complex logic explained in comments
- [ ] Docstrings accurate and complete
- [ ] User-facing docs updated
- [ ] Examples provided

**Performance**:
- [ ] No obvious performance issues
- [ ] Resource cleanup (files, processes, memory)
- [ ] Appropriate algorithm complexity
- [ ] No unnecessary operations

**Security**:
- [ ] No obvious security issues
- [ ] Input validation present
- [ ] Temporary files handled securely
- [ ] No sensitive data in logs

### Reviewer Checklist Template

```markdown
## Code Review: Phase X

**Reviewer**: _______________
**Date**: _______________
**Branch**: feature/async-plot-generation
**Commit**: _______________

### Summary
Brief description of changes reviewed.

### Findings

#### Critical Issues (must fix before merge)
- [ ] Issue 1: Description
- [ ] Issue 2: Description

#### Major Issues (should fix before merge)
- [ ] Issue 1: Description

#### Minor Issues (can fix later)
- [ ] Issue 1: Description

#### Suggestions (optional improvements)
- Suggestion 1
- Suggestion 2

### Approval
- [ ] APPROVED - Ready to merge
- [ ] APPROVED WITH COMMENTS - Merge after addressing critical issues
- [ ] NEEDS WORK - Do not merge, re-review required

**Comments**:
_______________
```

---

## Risk Mitigation

### Identified Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Background process becomes zombie | Low | Medium | Use `start_new_session=True`, test cleanup |
| Pickle file too large (memory) | Low | High | Monitor size, add size check |
| Matplotlib thread safety issues | Medium | Medium | Force Agg backend, use ProcessPool |
| Plot generation errors break system | Low | High | Isolate in subprocess, comprehensive error handling |
| Log file grows unbounded | Low | Low | Rotate logs, add max size check |
| Multiple simultaneous runs conflict | Medium | Low | Use unique temp filenames (mkstemp) |
| Worker pool exhausts CPU | Low | Medium | Default to 2 workers, document tuning |
| Disk space exhausted | Low | High | Check space before starting, handle errors |

### Mitigation Actions

**Pre-deployment**:
- [ ] Test with production data volumes
- [ ] Verify zombie process cleanup on crash
- [ ] Load test with 10 concurrent runs
- [ ] Test with low disk space
- [ ] Test with restricted permissions

**Post-deployment**:
- [ ] Monitor first 5 production runs
- [ ] Check for zombie processes daily (week 1)
- [ ] Review log file sizes weekly
- [ ] Monitor system resources (CPU, memory, disk)
- [ ] Collect user feedback

---

## Rollback Plan

### Conditions for Rollback

Rollback if:
- Main program crashes or hangs
- Plot generation consistently fails
- System resources exhausted (CPU, memory, disk)
- Zombie processes accumulate
- User experience degraded

### Rollback Procedure

**Option 1: Disable Async Mode (Quick)**
```bash
# Edit all production JSON configs
sed -i 's/"async_plot_generation": true/"async_plot_generation": false/g' \\
    /path/to/configs/*.json

# Restart main program
# System returns to synchronous (blocking) behavior
```

**Option 2: Revert Code (Full)**
```bash
# Revert to previous commit
git revert <commit-hash>

# Or reset to pre-feature state
git reset --hard <pre-feature-commit>

# Force push (if already deployed)
git push -f origin main

# Redeploy
```

**Option 3: Feature Flag (Graceful)**
```python
# Add emergency kill switch in PortfolioPerformanceCalcs.py
ASYNC_PLOT_FEATURE_ENABLED = os.getenv('ASYNC_PLOTS_ENABLED', 'true') == 'true'

if ASYNC_PLOT_FEATURE_ENABLED and params.get('async_plot_generation'):
    # Use async mode
else:
    # Force sync mode
```

Then disable via environment variable:
```bash
export ASYNC_PLOTS_ENABLED=false
```

### Post-Rollback

- [ ] Identify root cause
- [ ] Fix issue in feature branch
- [ ] Re-test thoroughly
- [ ] Deploy again with caution

---

## Implementation Timeline

| Phase | Duration | Dependencies | Start | End |
|-------|----------|--------------|-------|-----|
| 0. Branch Setup | 0.25 hrs | None | Day 1 | Day 1 |
| 1. Background Worker | 4 hrs | Phase 0 | Day 1 | Day 1 |
| 2. Refactor Plots | 3 hrs | Phase 1 | Day 2 | Day 2 |
| 3. Integration | 2 hrs | Phase 2 | Day 2 | Day 2 |
| 4. Documentation | 2 hrs | Phase 3 | Day 3 | Day 3 |
| 5. E2E Testing | 3 hrs | Phase 4 | Day 3 | Day 3 |
| **Total** | **14.25 hrs** | | **Day 1** | **Day 3** |

**Estimated Calendar Time**: 3 days (with reviews and testing)

---

## Acceptance Criteria

Feature is considered **DONE** when:

- [ ] All 5 phases completed
- [ ] All code review checkpoints passed
- [ ] All unit tests passing (target: 30 tests, 85% coverage)
- [ ] All integration tests passing (target: 15 tests)
- [ ] All E2E tests passing (target: 10 scenarios)
- [ ] Performance targets met:
  - [ ] Main program completes in < 10 seconds (async mode)
  - [ ] Plots complete within 10 minutes (2 workers)
  - [ ] CPU usage reasonable (< 50% per core)
- [ ] Documentation complete and accurate
- [ ] Test scripts functional
- [ ] Production configs updated
- [ ] Code merged to main
- [ ] Deployed to production
- [ ] Monitoring in place
- [ ] User feedback collected (positive)

---

## Appendix A: Example Configurations

### Development (Sync Mode)
```json
{
  "async_plot_generation": false,
  "plot_generation_workers": 1,
  "comment": "Synchronous for debugging"
}
```

### Production (Async with Conservative Workers)
```json
{
  "async_plot_generation": true,
  "plot_generation_workers": 2,
  "comment": "Default production setting"
}
```

### High-Performance (Async with More Workers)
```json
{
  "async_plot_generation": true,
  "plot_generation_workers": 4,
  "comment": "Faster on multi-core systems"
}
```

---

## Appendix B: Log File Format

Example `plot_generation.log`:

```
[14:32:15] ========================================
[14:32:15] PyTAAA Background Plot Generation Started
[14:32:15] Max workers: 2
[14:32:15] ========================================
[14:32:15] Loading plot data from /tmp/pytaaa_plots_ABC123.pkl
[14:32:16] Data loaded: 100 symbols
[14:32:16] Phase 1: Generating full-history plots...
[14:32:18]   ✓ AAPL (full history)
[14:32:19]   ✓ MSFT (full history)
[14:32:20]   ✓ GOOGL (full history)
...
[14:37:42] Phase 1 complete: 95 created, 5 skipped, 0 errors
[14:37:42] Phase 2: Generating recent 2-year plots...
[14:37:44]   ✓ AAPL (recent)
[14:37:45]   ✓ MSFT (recent)
...
[14:43:08] Phase 2 complete: 97 created, 3 skipped, 0 errors
[14:43:08] Cleaned up data file: /tmp/pytaaa_plots_ABC123.pkl
[14:43:08] ========================================
[14:43:08] Background plot generation COMPLETED
[14:43:08] Total time: 653.2 seconds
[14:43:08] Total plots: 192 created, 8 skipped, 0 errors
[14:43:08] ========================================
```

---

## Appendix C: Troubleshooting Guide

### Issue: Main program hangs after spawn

**Symptoms**: Program doesn't return after calling async plot generation

**Diagnosis**:
```bash
ps aux | grep python  # Check for stuck processes
```

**Solution**:
- Ensure `start_new_session=True` in subprocess.Popen()
- Check log file for background process errors
- Verify pickle file was created

### Issue: No PNG files generated

**Symptoms**: Log shows completion but no PNG files in output_dir

**Diagnosis**:
```bash
ls -la <output_dir>/*.png
tail -100 <output_dir>/plot_generation.log
```

**Solution**:
- Check write permissions on output_dir
- Verify matplotlib backend is Agg
- Check log for errors during generation

### Issue: Background process becomes zombie

**Symptoms**: Process shows in `ps` but doesn't complete

**Diagnosis**:
```bash
ps aux | grep background_plot_generator
lsof -p <pid>  # Check open files
```

**Solution**:
- Kill zombie process: `kill -9 <pid>`
- Check for deadlocks in worker pool
- Verify ProcessPoolExecutor cleanup

### Issue: Plots are outdated

**Symptoms**: PNG files exist but show old data

**Diagnosis**:
```bash
ls -lt <output_dir>/*.png | head -20  # Check timestamps
```

**Solution**:
- Delete old PNG files: `rm <output_dir>/0_*.png`
- Reduce 20-hour threshold in code (for testing)
- Force regeneration by touching data

---

## Appendix D: Quick Start for GitHub Agent

### Step-by-Step Agent Implementation

**Phase 0: Setup (15 min)**
```bash
git checkout copilot/review-update-docstrings-comments
git pull origin copilot/review-update-docstrings-comments
git checkout -b feature/async-plot-generation
git push -u origin feature/async-plot-generation
```

**Phase 1: Implement Background Worker (4 hrs)**
1. Create `functions/background_plot_generator.py`
   - CLI with `--data-file`, `--max-workers` arguments
   - `load_plot_data()` — deserialize pickle
   - `generate_single_full_history_plot()` — worker function
   - `generate_single_recent_plot()` — worker function
   - `main()` — ProcessPoolExecutor orchestration
2. Write unit tests in `tests/test_background_plot_generator.py`
3. Mark all tests with `@pytest.mark.agent_runnable`
4. Run: `pytest tests/test_background_plot_generator.py -v -m agent_runnable`
5. Commit with message: `feat: add background plot generator with parallel workers`
6. ✅ Code Review Checkpoint #1

**Phase 2: Refactor Plot Generation (3 hrs)**
1. Extract helpers in `functions/output_generators.py`:
   - `_generate_full_history_plots()` — from lines 144-198
   - `_generate_recent_plots()` — from lines 199-323
2. Add `_spawn_background_plot_generation()`:
   - Serialize data to pickle (tempfile.mkstemp)
   - Launch subprocess with `start_new_session=True`
   - Redirect stdout/stderr to `plot_generation.log`
3. Update `generate_portfolio_plots()`:
   - Add `async_mode: bool = False` parameter
   - Add `max_workers: int = 2` parameter
   - Branch: if async then spawn, else call helpers
4. Write tests in `tests/test_output_generators_async.py`
5. Run: `pytest tests/test_output_generators_async.py -v -m agent_runnable`
6. Commit: `feat: add async mode to generate_portfolio_plots`
7. ✅ Code Review Checkpoint #2

**Phase 3: Integration (2 hrs)**
1. Update `functions/PortfolioPerformanceCalcs.py`:
   - Read `async_plot_generation` from params
   - Read `plot_generation_workers` from params
   - Pass to `generate_portfolio_plots()`
2. Update `pytaaa_generic.json`:
   - Add `"async_plot_generation": false`
   - Add `"plot_generation_workers": 2`
3. Write tests in `tests/test_integration_async.py`
4. Run: `pytest tests/test_integration_async.py -v -m agent_runnable`
5. Commit: `feat: integrate async plot generation into main pipeline`
6. ✅ Code Review Checkpoint #3

**Phase 4: Documentation (2 hrs)**
1. Update `docs/PYTAAA_SYSTEM_SUMMARY_AND_JSON_GUIDE.md`
2. Create `scripts/test_async_plots.sh`
3. Create `scripts/monitor_plot_generation.sh`
4. Commit: `docs: document async plot generation feature`
5. ✅ Code Review Checkpoint #4

**Phase 5: Testing (3 hrs)**
1. Implement all 🤖 agent-runnable tests (38 tests)
   - Use `@pytest.mark.agent_runnable` decorator
   - Use mock/test data (10-20 symbols max)
   - Mock HDF5 reads, mock time checks, simulate errors
2. Create test stubs for 🏠 local-only tests (10 tests)
   - Use `@pytest.mark.local_only` decorator
   - Add `pytest.skip("Requires production HDF5 data")`
   - Document what production data is needed
3. Configure `pytest.ini` with markers
4. Run full agent test suite:
   ```bash
   pytest tests/ -v -m agent_runnable --cov=functions
   ```
5. Verify coverage > 80%
6. Commit: `test: add comprehensive E2E tests for async plot generation`
7. ✅ Code Review Checkpoint #5

**Phase 6: Create PR**
1. Push all commits: `git push origin feature/async-plot-generation`
2. Create PR with description:
   ```markdown
   ## Async Plot Generation Implementation
   
   Implements fire-and-forget background plot generation with parallel workers.
   
   ### Changes
   - ✅ Phase 0-4: All implementation complete
   - ✅ 38 agent-runnable tests passing
   - ✅ Code coverage: XX%
   - ⏳ 10 local-only tests require production validation
   
   ### Testing
   - Agent tests: `pytest -v -m agent_runnable`
   - Local tests: `pytest -v -m local_only` (requires HDF5 data)
   
   ### Next Steps
   1. Code review by maintainer
   2. Local validation of INT-01 to INT-04
   3. Performance benchmarks with 100 symbols
   4. Merge if all tests pass
   ```

**What Agent Cannot Do (Local Validation)**:
- ❌ INT-01 to INT-04 (production configs)
- ❌ E2E-02, E2E-04 (100 symbols)
- ❌ PERF-01 (100-symbol baseline)
- ❌ REG-03 (manual smoke test)

**Agent should create test stubs** that document requirements:
```python
@pytest.mark.local_only
def test_int_01_naz100_hma_config():
    """
    INT-01: Run with naz100_hma config
    
    Requirements:
    - Production config: pytaaa_model_switching_params.json
    - HDF5 data: pyTAAA_data/QQQ.h5 (or equivalent)
    - Symbols: NAZ100 universe
    
    Steps:
    1. Run: uv run python run_pytaaa.py --json pytaaa_model_switching_params.json
    2. Verify plots created in output directory
    3. Check plot_generation.log for errors
    4. Validate PNG files show correct data
    """
    pytest.skip("Requires production HDF5 data - run locally after PR")
```

### Summary for Agent

**Your mission:**
1. Implement Phases 0-4 completely (all code, all docs)
2. Implement 38 agent-runnable tests with mocks
3. Create 10 local-only test stubs with documentation
4. Open PR with comprehensive description
5. Estimated time: 14 hours autonomous work

**Human will do:**
1. Code review your PR (1-2 hours)
2. Run 10 local tests with production data (30-60 min)
3. Merge if passing (5 min)

**Result:** 79% automation, minimal human intervention required!

---

**End of Plan**

---

## Revision History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2026-02-24 | 1.0 | System | Initial plan created |
| 2026-02-24 | 1.1 | System | Added agent-runnable vs local-only test markers; Test distribution summary; GitHub Agent Implementation Guide; Pytest marker strategy; Quick Start appendix |
