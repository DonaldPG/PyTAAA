# Date and Context
- Date: 2026-02-24
- Context: Implementation of fire-and-forget async Monte Carlo backtest generation to reduce portfolio computation runtime from ~10 minutes to ~2 minutes by spawning background process for Monte Carlo backtest (2 PNG plots).

# Problem Statement
Portfolio computation (daily_abacus_update.py, run_pytaaa.py) spent ~5-10 minutes generating Monte Carlo backtest plots synchronously, blocking the main program completion. Daily operations required fast turnaround for portfolio status updates, but Monte Carlo backtest was the last major bottleneck after async plot generation was implemented:
- Synchronous backtest blocked main program completion
- 2 PNGs (full history + recent) generated via ~12 Monte Carlo trials
- No user control over when backtest runs
- No way to prevent duplicate background jobs

# Solution Overview
Implemented async Monte Carlo backtest system with fire-and-forget subprocess pattern (same architecture as async plot generation):
- Main program checks for stale plots (>20 hours old)
- Spawns detached background process for Monte Carlo backtest
- Kills any existing background jobs for same config before starting new one
- Background worker uses existing `dailyBacktest_pctLong()` function
- Main program returns immediately (~2 min total runtime)
- Background backtest continues independently (~5-10 min additional time)
- Log file recreated (not appended) for each run

# Key Changes

## New Files Created (GitHub Copilot Agent - Auto-generated)
- **functions/background_montecarlo_runner.py** (112 lines)
  - Standalone CLI worker for Monte Carlo backtest
  - Minimal wrapper around `dailyBacktest_pctLong()`
  - Forces Agg backend for headless execution
  - Robust error handling and exit codes

- **tests/test_background_montecarlo_runner.py** (148 lines)
  - Unit tests for CLI parsing, main function, module attributes
  - 11 tests, all passing

- **tests/test_makeDailyMonteCarloBacktest_async.py** (260 lines)
  - Tests for async mode toggle, spawn function, sync/async dispatch
  - 13 tests, all passing

## Modified Files (User + Copilot - Iterative Fixes)
- **functions/MakeValuePlot.py**
  - Added `_kill_existing_montecarlo_processes()` function (66 lines)
    - Searches for running `background_montecarlo_runner` processes
    - Kills processes using same JSON config file
    - Prevents duplicate Monte Carlo computations
  - Added `_spawn_background_montecarlo()` function (53 lines)
    - Sets PYTHONPATH environment variable for subprocess
    - Spawns detached process with `start_new_session=True`
    - Redirects stdout/stderr to `montecarlo_backtest.log`
    - Ensures web directory exists with `os.makedirs()`
  - Updated `makeDailyMonteCarloBacktest()` signature
    - Added `async_mode: bool = False` parameter
    - Reads from params dict if not explicitly set
    - Branches to async or sync path based on flag
    - Respects 20-hour freshness check in both modes

- **functions/WriteWebPage_pi.py**
  - Reads `async_montecarlo_backtest` parameter from config
  - Passes to `makeDailyMonteCarloBacktest(async_mode=...)`
  - Default: `True` (opt-out behavior per user request)

- **pytaaa_generic.json**
  - Added `"async_montecarlo_backtest": true` (default enabled)

- **pytaaa_model_switching_params.json**
  - Added `"async_montecarlo_backtest": true` (production config)

## Config Structure Fix
- Initially tried `get_web_output_dir()` but configs use `Valuation.webpage`
- Corrected to use `get_webpage_store()` to match all other functions
- Added `os.makedirs(web_dir, exist_ok=True)` to ensure directory exists

# Technical Details

## Process Detection and Termination
Critical feature to prevent duplicate jobs:
```python
def _kill_existing_montecarlo_processes(json_fn: str) -> None:
    """Kill any existing background Monte Carlo processes for same config."""
    ps_output = subprocess.check_output(["ps", "aux"], text=True)
    json_fn_normalized = os.path.abspath(json_fn)
    
    for line in ps_output.splitlines():
        if "background_montecarlo_runner" in line and "--json-file" in line:
            if json_fn in line or json_fn_normalized in line:
                pid = int(parts[1])
                if pid != os.getpid():
                    os.kill(pid, 15)  # SIGTERM
```

## Log File Recreation
Log file is always recreated (not appended):
```python
# First write: recreate/truncate
with open(log_file, "w") as log_fh:
    log_fh.write(f"[{datetime.datetime.now().isoformat()}] Spawning...\n")

# Second write: append subprocess output
with open(log_file, "a") as log_fh:
    subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh, ...)
```

## Fire-and-Forget Pattern
```python
# Set PYTHONPATH for subprocess module access
env = os.environ.copy()
env["PYTHONPATH"] = project_root

# Spawn detached process
subprocess.Popen(
    cmd, env=env,
    stdout=log_fh, stderr=log_fh,
    stdin=subprocess.DEVNULL,
    start_new_session=True,  # Detach from parent
)
# Returns immediately, Monte Carlo continues in background
```

## Config Path Resolution
Corrected from initial attempt:
```python
# WRONG: get_web_output_dir() returns root-level web_output_dir
# webpage_dir = get_web_output_dir(json_fn)

# CORRECT: get_webpage_store() returns Valuation.webpage
webpage_dir = get_webpage_store(json_fn)
```

# Testing

## Automated Tests (GitHub Copilot Agent)
- **test_background_montecarlo_runner.py**: 11/11 passed
- **test_makeDailyMonteCarloBacktest_async.py**: 13/13 passed
- **Total automated**: 24/24 passing

## Production Validation (User - Local Testing)
- ✅ Tested with `pytaaa_naz100_pine.json`
- ✅ Main program completes in ~2 minutes
- ✅ Background process runs independently
- ✅ Log file created at correct location
- ✅ Duplicate process detection works (kills existing jobs)
- ✅ Log file recreated (not appended) on each run

## Issues Found and Fixed
1. **Config path mismatch**: Initial use of `get_web_output_dir()` caused KeyError
   - Fixed: Use `get_webpage_store()` like all other functions
2. **Directory not found**: FileNotFoundError when creating log
   - Fixed: Added `os.makedirs(web_dir, exist_ok=True)`
3. **No duplicate prevention**: Multiple jobs could run simultaneously
   - Fixed: Added `_kill_existing_montecarlo_processes()` before spawn

## Performance Results
- **Before**: ~10 minutes total (synchronous Monte Carlo)
- **After**: ~2 minutes main program + ~5-10 minutes background (async)
- **Improvement**: Main program 5x faster, unblocked for other operations
- **Goal Met**: ✅ ~2 minute target achieved

# Follow-up Items
None - feature is production-ready.

Optional future enhancements:
- Consider adding health check endpoint for background process monitoring
- Consider adding configurable timeout for Monte Carlo trials
- Consider adding email notification when backtest completes

# Commits and PR
- **Branch**: `copilot/implement-async-monte-carlo`
- **PR**: #26 "Implement fire-and-forget async Monte Carlo backtest generation"
- **Key Commits**:
  - `f280556`: Initial plan
  - `dbf5e68`: feat: implement fire-and-forget async Monte Carlo backtest generation (Agent)
  - `5d8eb04`: fix: use get_web_output_dir instead of get_webpage_store (incorrect)
  - `403326c`: fix: revert to get_webpage_store and ensure directory exists (corrected)
  - `eb68db7`: feat: kill existing Monte Carlo processes before spawning new one (final enhancement)

**Total Changes**: 7 files changed, 700+ insertions(+)

# Lessons Learned
1. **Config Path Consistency**: Always use the same config accessor function as other code in the same file
2. **Directory Creation**: Ensure output directories exist before creating files
3. **Process Management**: Important to prevent duplicate background jobs
4. **Log File Management**: Recreation vs. append - recreation better for debugging
5. **Agent Capability**: GitHub Copilot successfully implemented core functionality autonomously
6. **Test-First Validation**: Automated tests (24) caught structural issues, local testing caught config issues
7. **Fire-and-Forget Pattern**: Same pattern as async plot generation works perfectly for Monte Carlo

# Comparison with Async Plot Generation
Both features use the same architecture but differ in scope:

| Aspect | Async Plot Generation | Async Monte Carlo Backtest |
|--------|----------------------|---------------------------|
| **Files Generated** | ~200 PNGs (2 per symbol) | 2 PNGs + CSV |
| **Parallelization** | ProcessPoolExecutor (2-4 workers) | Single process (internally parallel) |
| **Computation Time** | ~8-10 min → ~8-10 min (unchanged) | ~5-10 min → ~5-10 min (unchanged) |
| **Main Program Wait** | ~10 min → <1 min (10x faster) | ~10 min → ~2 min (5x faster) |
| **Log File** | `plot_generation.log` | `montecarlo_backtest.log` |
| **Process Detection** | Not needed (short-lived) | Required (prevents duplicates) |
| **Config Parameter** | `async_plot_generation` | `async_montecarlo_backtest` |
| **Default** | `false` (opt-in) | `true` (opt-out) |

Both features eliminate the final bottlenecks in the PyTAAA pipeline, enabling near-instant web page updates with background computation completion.
