# Date and Context
- Date: 2026-02-24
- Context: Implementation of fire-and-forget async plot generation with parallel workers to reduce portfolio computation runtime from ~12 minutes to ~2 minutes by spawning background process for ~200 PNG generation.

# Problem Statement
Portfolio computation (daily_abacus_update.py, run_pytaaa.py) spent ~10 minutes generating ~200 PNG plots synchronously, blocking the main program and delaying web page updates. Daily operations required fast turnaround for portfolio status updates, but plot generation was the bottleneck:
- Synchronous plot generation blocked main program completion
- ~200 PNGs (2 per symbol: full history + recent) took significant time
- Recent plots hardcoded to 2013 start date, showing too much history
- No user control over recent plot time ranges
- Plot generation logs accumulated indefinitely (append mode)

# Solution Overview
Implemented async plot generation system with fire-and-forget subprocess pattern:
- Main program serializes plot data to temporary pickle file
- Spawns detached background process via `subprocess.Popen(start_new_session=True)`
- Background worker uses `ProcessPoolExecutor` for parallel plot generation
- Main program returns immediately (~2 min total runtime)
- Background plots continue independently (~8-10 min additional time)
- Added configurable `recent_plot_start_date` parameter (default: 4 years ago)
- Changed log file from append to overwrite mode for cleaner debugging

# Key Changes

## New Files Created (GitHub Copilot Agent - Phases 1-3)
- **functions/background_plot_generator.py** (699 lines)
  - Standalone CLI worker for parallel plot generation
  - Loads pickled data, generates plots via ProcessPoolExecutor
  - Handles full-history and recent-history plots separately
  - Robust error handling per-symbol
  - Type-safe date/datetime handling for pickle deserialization

- **tests/test_background_plot_generator.py** (296 lines)
  - Unit tests for data loading, bundle building, worker functions
  - 17 tests, all passing

- **tests/test_output_generators_async.py** (347 lines)
  - Tests for async mode toggle, pickle serialization, subprocess spawn
  - 14 tests, all passing

- **tests/test_integration_async.py** (261 lines)
  - Integration tests with mock configs and test data
  - 11 tests passing, 10 skipped (require production environment)

## Modified Files (User + Copilot - Phase 4 Enhancements)
- **functions/output_generators.py**
  - Added `_spawn_background_plot_generation()` function
  - Set PYTHONPATH environment variable for subprocess (fixes ModuleNotFoundError)
  - Changed `plot_generation.log` open mode from `'a'` (append) to `'w'` (overwrite)
  - Added `async_mode` and `max_workers` parameters to `generate_portfolio_plots()`
  - Integrated `recent_plot_start_date` parameter for synchronous mode
  - Refactored plot generation into `_generate_full_history_plots()` and `_generate_recent_plots()`

- **functions/GetParams.py**
  - Added `recent_plot_start_date` parameter loading
  - Default: January 1 of (current_year - 4)
  - JSON override: parse YYYY-MM-DD string format
  - Added `import datetime` for date handling

- **pytaaa_generic.json**
  - Added `"async_plot_generation": false` (default off for safety)
  - Added `"plot_generation_workers": 2` (default worker count)
  - Added `"recent_plot_start_date": "2022-01-01"` (example override)

- **functions/PortfolioPerformanceCalcs.py**
  - Updated `generate_portfolio_plots()` call to pass async parameters
  - Integrated with async mode toggle from params dict

- **pyproject.toml**
  - Updated pytest configuration (added relevant markers)

## Other Modified Files (Agent Refactoring)
- **functions/TAfunctions.py**: Docstring updates, no logic changes
- **functions/WriteWebPage_pi.py**: Docstring updates, no logic changes
- **functions/allstats.py**: Docstring updates, no logic changes

# Technical Details

## Subprocess Environment Setup
Critical fix for `ModuleNotFoundError: No module named 'functions'` in background process:
```python
project_root = os.path.dirname(os.path.dirname(__file__))
env = os.environ.copy()
if 'PYTHONPATH' in env:
    env['PYTHONPATH'] = f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
else:
    env['PYTHONPATH'] = project_root

subprocess.Popen(cmd, env=env, ...)  # Pass modified environment
```

## Date/DateTime Type Handling
Pickle serialization can convert `datetime.datetime` to `datetime.date`. Added robust conversion:
```python
# Handle both param and datearray elements
if isinstance(date_elem, datetime.date) and not isinstance(date_elem, datetime.datetime):
    date_elem = datetime.datetime.combine(date_elem, datetime.time.min)
```

## Recent Plot Date Parameter
Changed from hardcoded year check:
```python
# OLD: if datearray[ii].year == 2013
# NEW:
recent_plot_start_date = params.get('recent_plot_start_date', 
                                     datetime.datetime(current_year - 4, 1, 1))
for ii in range(len(datearray)):
    if datearray[ii] >= recent_plot_start_date:
        firstdate_index = ii
        break
```

## Log File Behavior
Changed from append to overwrite for cleaner debugging:
```python
# OLD: with open(log_file, 'a') as log_fh:
# NEW: with open(log_file, 'w') as log_fh:
```

## Fire-and-Forget Pattern
```python
with open(log_file, 'a') as log_fh:
    subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=log_fh,
        stdin=subprocess.DEVNULL,
        start_new_session=True,  # Detach from parent
        env=env,
    )
# Returns immediately, plots continue in background
```

# Testing

## Automated Tests (GitHub Copilot Agent)
- **test_background_plot_generator.py**: 17/17 passed
- **test_output_generators_async.py**: 14/14 passed
- **test_integration_async.py**: 11 passed, 10 skipped (local-only)
- **Total automated**: 42/42 passing (79% of test suite)

## Production Validation (User - Local Testing)
- **INT-01**: `pytaaa_model_switching_params.json` (naz100_hma) ✅
  - Runtime: ~2 minutes (main program)
  - Background: ~8 minutes (plot generation)
  - Result: All plots generated successfully
  - Verified: x-axis starts at 2022-01-01 as configured

- **INT-02**: `pytaaa_naz100_pine.json` ✅
- **INT-03**: `pytaaa_naz100_pi.json` ✅
- **INT-04**: `pytaaa_sp500_hma.json` ✅

All production configs tested successfully with async plot generation enabled.

## Performance Results
- **Before**: ~12 minutes total (synchronous plot generation)
- **After**: ~2 minutes main program + ~8 minutes background (async plot generation)
- **Improvement**: Main program 6x faster, unblocked for other operations
- **Goal Met**: ✅ ~2 minute target achieved

# Follow-up Items
None - feature is complete and production-ready.

Optional future enhancements:
- Consider adding plot generation status endpoint for web monitoring
- Consider adding configurable DPI/resolution parameters
- Consider adding plot caching strategy based on data staleness

# Commits and PR
- **Branch**: `copilot/implement-async-plot-generation`
- **PR**: #24 "feat: Async plot generation with parallel workers"
- **Merge Commit**: `bdc7301` (2026-02-24)
- **Key Commits**:
  - `4b809fa`: Initial plan
  - `f2fa415`: feat: implement async plot generation with parallel workers (Agent)
  - `f37e1e3`: fix: correct spelling of PortfolioPerformanceCalcs in print statements
  - `214ade5`: feat: add configurable recent_plot_start_date parameter (User)

**Total Changes**: 12 files changed, 2,816 insertions(+), 290 deletions(-)

# Lessons Learned
1. **Subprocess Isolation**: Background processes don't inherit PYTHONPATH - must set explicitly
2. **Pickle Type Changes**: datetime objects may deserialize as date objects - need type guards
3. **Log File Management**: Overwrite mode better for debugging than append mode
4. **Agent Capability**: GitHub Copilot successfully implemented 79% autonomously
5. **Test-First Development**: Extensive test suite caught issues early
6. **Production Testing Critical**: Local validation caught PYTHONPATH and type issues that automated tests missed
