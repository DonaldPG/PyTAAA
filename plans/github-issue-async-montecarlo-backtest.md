# GitHub Issue: Implement Async Monte Carlo Backtest Generation

**Title**: Implement fire-and-forget async Monte Carlo backtest generation  
**Labels**: `enhancement`, `performance`, `copilot-agent`  
**Assignee**: GitHub Copilot Agent  
**Project**: PyTAAA Async Optimizations  
**Milestone**: v2.1 - Background Processing  

---

## Problem Statement

PyTAAA currently blocks for **5-10 minutes** during Monte Carlo backtest generation near the end of each execution cycle. This backtest generates 2 PNG plots that are included in the web page but don't need to complete before the main program finishes.

**Current Flow:**
```
run_pytaaa.py → writeWebPage() → makeDailyMonteCarloBacktest() → dailyBacktest_pctLong()
                                                                    ↓ (blocks 5-10 min)
                                                                  2 PNG plots
```

**Impact:**
- User waits unnecessarily for program completion
- Web page tables/holdings could update immediately
- Last remaining computation bottleneck after async plot generation (#24)

---

## Proposed Solution

Implement fire-and-forget background process for Monte Carlo backtest generation, following the proven pattern from async plot generation (#24):

1. **Background Worker**: Create `functions/background_montecarlo_runner.py` - minimal wrapper around existing `dailyBacktest_pctLong()`
2. **Async Mode Toggle**: Add `async_mode` parameter to `makeDailyMonteCarloBacktest()` in `functions/MakeValuePlot.py`
3. **Subprocess Spawn**: Detached background process with log redirection to `montecarlo_backtest.log`
4. **JSON Configuration**: Add `async_montecarlo_backtest` parameter (default: `true` for opt-out behavior)

**New Flow:**
```
run_pytaaa.py → writeWebPage() → makeDailyMonteCarloBacktest()
                                  ↓ (if async_mode=true)
                                  spawn background process → returns immediately
                                  ↓ (main program continues)
                                  Email sent, web page finalized, program exits
                                  
(background) → dailyBacktest_pctLong() → 2 PNG plots (5-10 min later)
```

**Key Difference from Plot Generation:**
- Plot generation uses `ProcessPoolExecutor` with multiple workers (200 plots in parallel)
- Monte Carlo backtest is **single process** (already internally parallelized with multiple trials)
- No worker pool needed - simple fire-and-forget subprocess

---

## Success Criteria

### Functional Requirements
- ✅ Main program completes without waiting when `async_montecarlo_backtest=true`
- ✅ Backtest generates in background within 5-10 minutes
- ✅ Both PNG files created: `PyTAAA_monteCarloBacktest.png`, `PyTAAA_monteCarloBacktestRecent.png`
- ✅ Log file captures all progress: `montecarlo_backtest.log`
- ✅ Synchronous mode works unchanged (`async_montecarlo_backtest=false`)
- ✅ 20-hour freshness check respected (skip if plots recent)
- ✅ Platform-specific trial counts preserved (pi: 12, MacOS: 13, Windows64: 15)

### Performance Requirements
- Main program blocking time: **< 10 seconds** (down from 5-10 minutes)
- Total backtest time: 5-10 min (unchanged, but asynchronous)
- User perceived wait time: **~0 seconds**

### Non-Functional Requirements
- No zombie processes (clean termination)
- Matplotlib Agg backend (headless)
- Human-readable log with timestamps
- Errors don't crash main program
- PYTHONPATH set for subprocess module access

---

## Deliverables

### Code Changes

1. **New Files:**
   - `functions/background_montecarlo_runner.py` (~200 lines)
   - `tests/test_background_montecarlo_runner.py` (8 unit tests)
   - `tests/test_makeDailyMonteCarloBacktest_async.py` (7 unit tests)
   - `tests/test_async_montecarlo_integration.py` (9 integration tests)
   - `docs/ASYNC_MONTECARLO_BACKTEST.md` (comprehensive documentation)

2. **Modified Files:**
   - `functions/MakeValuePlot.py` (add async mode, ~50 lines changed)
   - `functions/WriteWebPage_pi.py` (minimal changes, comments)
   - `pytaaa_generic.json` (add `async_montecarlo_backtest: true`)
   - `pytaaa_model_switching_params.json` (add parameter)
   - `docs/DAILY_OPERATIONS_GUIDE.md` (update with async behavior)
   - `README.md` (mention feature)

3. **Optional Helpers:**
   - `scripts/monitor_montecarlo_backtest.sh` (monitoring helper)

### Testing

**Automated Tests (29 tests - agent can implement):**
- 15 unit tests (CLI parsing, spawn function, error handling)
- 9 integration tests (5 agent-automated, 4 local-only)
- 7 end-to-end tests (4 agent-automated, 3 local-only)
- 7 regression tests (4 agent-automated, 3 local-only)
- 6 performance tests (5 agent-automated, 1 local-only)
- 8 error handling tests (agent-automated)

**Local Validation (10 tests - user runs):**
- Production config tests (4 configs: naz100_hma, naz100_pine, naz100_pi, sp500_hma)
- Full E2E with real Monte Carlo trials (12+ trials)
- Performance benchmarking with production data
- Manual smoke test

### Documentation

- Feature overview and architecture
- Configuration guide with examples
- Troubleshooting section
- Log monitoring instructions
- Performance characteristics
- Comparison table: async vs sync mode

---

## Implementation Plan

**See detailed plan**: [`plans/async-montecarlo-backtest.md`](../plans/async-montecarlo-backtest.md)

**Phased Approach:**
- **Phase 0**: Branch setup (10 min)
- **Phase 1**: Background runner script (2 hours)
- **Phase 2**: MakeValuePlot refactoring (2 hours)
- **Phase 3**: WriteWebPage integration (1.5 hours)
- **Phase 4**: Documentation and testing (1.5 hours)
- **Phase 5**: PR and review (30 min)

**Total Agent Time**: ~8 hours (autonomous)  
**Total User Time**: ~1.5 hours (review + local testing)

---

## Technical Details

### Background Runner Module

**File**: `functions/background_montecarlo_runner.py`

```python
"""Background Monte Carlo backtest runner.

Minimal wrapper for fire-and-forget Monte Carlo backtest generation.
"""
import argparse
import matplotlib
matplotlib.use("Agg")

from functions.dailyBacktest_pctLong import dailyBacktest_pctLong

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-file", required=True)
    args = parser.parse_args()
    
    print(f"[{datetime.datetime.now():%H:%M:%S}] Starting Monte Carlo backtest")
    dailyBacktest_pctLong(args.json_file, verbose=True)
    print(f"[{datetime.datetime.now():%H:%M:%S}] Backtest complete")

if __name__ == "__main__":
    main()
```

### Spawn Function

**File**: `functions/MakeValuePlot.py`

```python
def _spawn_background_montecarlo(json_fn: str, web_dir: str) -> None:
    """Spawn detached background process for Monte Carlo backtest."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    env = os.environ.copy()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = project_root
    
    cmd = [
        sys.executable, "-m", "functions.background_montecarlo_runner",
        "--json-file", json_fn
    ]
    
    log_file = os.path.join(web_dir, "montecarlo_backtest.log")
    with open(log_file, 'w') as log_fh:
        subprocess.Popen(
            cmd, env=env,
            stdout=log_fh, stderr=log_fh,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    
    print(f"Spawned background Monte Carlo backtest")
    print(f"  Log: {log_file}")
```

### Configuration

**File**: `pytaaa_generic.json`

```json
{
  "async_plot_generation": false,
  "plot_generation_workers": 2,
  "async_montecarlo_backtest": true,
  "stockList": "Naz100",
  ...
}
```

---

## Dependencies

- Follows patterns from #24 (async plot generation)
- Reuses `dailyBacktest_pctLong()` (no changes needed)
- Requires `matplotlib`, `subprocess`, `argparse` (already in dependencies)
- No new external dependencies

---

## Acceptance Criteria

**Agent Implementation (75% of work):**
- [ ] All code written and pushed to `feature/async-montecarlo-backtest` branch
- [ ] 29 automated tests passing
- [ ] Documentation complete
- [ ] PR opened with detailed description
- [ ] Code follows PEP 8 and project conventions
- [ ] All code review checkpoints addressed

**User Validation (25% of work):**
- [ ] Pull PR branch locally
- [ ] Run 10 local validation tests with production configs
- [ ] Verify main program completes in < 3 minutes
- [ ] Verify PNG plots appear within 10 minutes
- [ ] Verify CSV results match sync mode
- [ ] No errors in `montecarlo_backtest.log`
- [ ] No zombie processes after multiple runs

**Merge Criteria:**
- [ ] All tests passing (automated + local)
- [ ] Performance targets met
- [ ] Documentation reviewed and approved
- [ ] No regressions in existing functionality
- [ ] Code review approved by user

---

## Testing Instructions for Agent

### Unit Tests
```bash
# Test background runner module
PYTHONPATH=$(pwd) uv run pytest tests/test_background_montecarlo_runner.py -v

# Test MakeValuePlot async mode
PYTHONPATH=$(pwd) uv run pytest tests/test_makeDailyMonteCarloBacktest_async.py -v
```

### Integration Tests (Agent Can Automate)
```bash
# Test with mock config and test data
PYTHONPATH=$(pwd) uv run pytest tests/test_async_montecarlo_integration.py -v -k "not local"
```

### Regression Tests
```bash
# Verify no breakage in existing tests
PYTHONPATH=$(pwd) uv run pytest tests/ -v
```

---

## Testing Instructions for User (Local Validation)

### Production Config Tests
```bash
# Test all 4 production configs with async=true
for config in pytaaa_model_switching_params.json pytaaa_naz100_pine.json pytaaa_naz100_pi.json pytaaa_sp500_hma.json; do
    echo "Testing $config"
    time PYTHONPATH=$(pwd) uv run python pytaaa_main.py --json $config
    
    # Check background process
    sleep 2
    ps aux | grep background_montecarlo
    
    # Monitor log
    tail -n 20 <web_dir>/montecarlo_backtest.log
    
    # Wait and verify plots
    sleep 600  # 10 minutes
    ls -lh <web_dir>/PyTAAA_montecarlo*.png
done
```

### Performance Benchmark
```bash
# Measure sync mode (baseline)
time PYTHONPATH=$(pwd) uv run python pytaaa_main.py --json test_sync.json
# Expected: 8-10 minutes

# Measure async mode (target)
time PYTHONPATH=$(pwd) uv run python pytaaa_main.py --json test_async.json
# Expected: < 3 minutes (main program)
```

### E2E Validation
```bash
# Full pipeline test
PYTHONPATH=$(pwd) uv run python pytaaa_main.py --json pytaaa_model_switching_params.json

# Verify outputs
ls -lh <web_dir>/PyTAAA_montecarlo*.png
cat <web_dir>/pyTAAAweb_backtestPctLong.params | head
tail -n 50 <web_dir>/montecarlo_backtest.log
```

---

## Rollback Plan

If issues arise after merge:

1. **Quick disable** (1 minute):
   ```json
   {"async_montecarlo_backtest": false}
   ```

2. **Git revert** (5 minutes):
   ```bash
   git revert <merge_commit>
   git push origin main
   ```

3. **Hotfix** (if specific bug found):
   - Create `hotfix/async-montecarlo-fix` branch
   - Fix issue
   - Fast-track PR review

---

## Related Work

- **PR #24**: Async plot generation (same pattern)
  - Implemented fire-and-forget subprocess pattern
  - Proven stable in production
  - This issue reuses the same architecture

- **Session Summary**: `docs/copilot_sessions/2026-02-24_async-plot-generation-implementation.md`
  - Documents lessons learned
  - PYTHONPATH environment variable setup
  - Log file management best practices

---

## Questions?

- **Q**: Why default `async_montecarlo_backtest: true` (opt-out)?  
  **A**: User requested opt-out behavior for faster default experience. Conservative users can set to `false`.

- **Q**: Why no parallel workers like plot generation?  
  **A**: Monte Carlo backtest is already internally parallelized (12-51 trials). Single background process is sufficient.

- **Q**: What happens if plots don't exist yet?  
  **A**: Web page HTML references plots, browser shows broken image until plots appear (5-10 min). User can refresh page.

- **Q**: Are CSV results identical to sync mode?  
  **A**: Yes, same computation, just asynchronous. Tests verify correctness.

---

## Agent Instructions

**Implementation Workflow:**

1. **Read the detailed plan**: [`plans/async-montecarlo-backtest.md`](../plans/async-montecarlo-backtest.md)

2. **Create branch**: `feature/async-montecarlo-backtest` from `main`

3. **Implement Phases 0-3** (all code):
   - Phase 1: Background runner module + tests
   - Phase 2: MakeValuePlot refactoring + tests
   - Phase 3: WriteWebPage integration + tests

4. **Write documentation** (Phase 4):
   - `docs/ASYNC_MONTECARLO_BACKTEST.md`
   - Update `docs/DAILY_OPERATIONS_GUIDE.md`
   - Update `README.md`

5. **Run automated tests**:
   ```bash
   PYTHONPATH=$(pwd) uv run pytest tests/ -v
   ```

6. **Open PR** (Phase 5):
   - Use PR template from plan
   - Link to this issue
   - List all changes and test results

7. **Post PR comment**:
   - "Ready for local validation"
   - List 10 local-only tests user should run
   - Provide testing commands

**Important Notes:**
- Follow patterns from async plot generation (#24)
- Copy PYTHONPATH setup exactly (proven working)
- Use write mode ('w') for log file (not append)
- Respect 20-hour freshness check logic
- No changes to `dailyBacktest_pctLong()` internals

**When in doubt**: Refer to `docs/copilot_sessions/2026-02-24_async-plot-generation-implementation.md` for proven patterns.

---

**Estimated Agent Time**: 8 hours autonomous work  
**Estimated User Time**: 1.5 hours (review + local tests)  
**Expected Outcome**: Main program runtime reduced from ~10 min to ~2 min

---

*This issue can be assigned to GitHub Copilot Agent for autonomous implementation following the detailed plan.*
