## Plan: Rolling Window Filter Diagnostics and Fix

TL;DR — Diagnose why the rolling-window data-quality filter does not prevent artificially-infilling symbols (notably `JEF`) from being selected across 2015–2018, add deterministic tests, and implement a minimal, well-reviewed fix that preserves intended monthly-hold semantics. Key changes: make `apply_rolling_window_filter` robust to optional args, decide on `inplace` behavior (copy-by-default with optional `inplace=True`), ensure monthly rebalance uses post-filter daily signals, and tighten `sharpeWeightedRank_2D` usage so zeros persist.

**Context & Evidence**
- Relevant files: [functions/rolling_window_filter.py](functions/rolling_window_filter.py#L1-L171), [functions/dailyBacktest.py](functions/dailyBacktest.py), [functions/TAfunctions.py](functions/TAfunctions.py)
- Scan log file: `pytaaa_sp500_pine.log` contains rolling filter debug lines that show the filter detecting low volatility and zeroing signals for `JEF`, e.g.: 

    DEBUG: Checking JEF on 2015-09-01: gainloss_std=0.000002
    RollingFilter: Zeroing JEF on 2015-09-01 due to low gainloss_std=0.000002

- Despite the filter zeroing `JEF` on those dates, later diagnostic output shows `JEF` selected with large portfolio weight during 2015–2018. Example logs from selection printing (note the column labeled `Signal2D` is the input; `Seignal2D` is the returned value in earlier prints):

    ... inside print_JEF_selections ...
    Date       | Signal2D | JEF Weight
    -----------------------------------------
    2015-05-01 | 1.000000 | 1.0000
    2015-05-04 | 1.000000 | 1.0000
    2015-05-06 | 1.000000 | 1.0000

- Monthly holdings at year starts show `JEF` dominating the portfolio in 2015–2018, e.g.:

    STOCK SELECTIONS AT BEGINNING OF SELECTED YEARS
    2015-01-02: [JEF:0.6942, LUV:0.0910, ...] Sum=1.0000
    2016-01-04: [JEF:0.7143, ANDV:0.0714, ...] Sum=1.0000

These contradict the rolling filter debug evidence that `JEF` quotes are infilled and should be excluded.

**Goal**
Create a reproducible diagnostic and fix workflow so an autonomous assistant can: run focused tests, analyze mismatches between old/new data loaders and the rolling filter, add unit tests capturing the bug, and apply the minimal code changes with code-review and cleanup steps.

**Steps**
1. Reproduce focused failures locally
   - Run failing tests to collect concise traces:

```bash
uv run python -m pytest tests/test_rolling_window_filter.py -q
uv run python -m pytest tests/test_phase4a_shadow.py::TestDataLoaderShadow::test_data_loader_matches_inline_naz100 -q -vv
uv run python -m pytest tests/test_phase4b_shadow.py::TestPhase4b1PlotExtraction::test_plot_files_generated_before_refactor -q -vv
```

   - Capture: failing assertion diffs (for data loader) and the `TypeError` traces in the rolling filter tests.

2. Rolling filter (phase 1) — Make function robust and explicit
   - Goal: fix the `TypeError: 'NoneType' object is not subscriptable` when `datearray` or `symbols` are omitted in tests, and establish `inplace` contract.
   - Code changes (minimal):
     - Update `apply_rolling_window_filter` in [functions/rolling_window_filter.py](functions/rolling_window_filter.py#L1-L171):
       - Accept a new optional arg `inplace: bool = False`.
       - If `datearray` is None, avoid indexing it; build `date_str = "<no-date>"` for debug prints.
       - If `symbols` is None, use `symbol_str = "<no-symbol>"` for debug prints.
       - Keep `signal_out = signal2D.copy()` when `inplace=False`. If `inplace=True`, modify `signal2D` in-place and return it (for backwards compatibility).
       - Ensure every debug print uses these safe fallbacks.
   - Tests to add/update:
     - Add `test_datearray_none` and `test_symbols_none` to `tests/test_rolling_window_filter.py` verifying no exception when optional args omitted.
     - Keep `test_in_place_modification` but assert behavior for both `inplace=True` and `inplace=False`.
   - Verification: re-run `tests/test_rolling_window_filter.py` and confirm no TypeErrors.

3. Monthly-hold selection semantics (phase 2)
   - Goal: ensure monthly holdings are decided from the filtered daily signals and forward-filled without being overwritten by later processing.
   - Confirm implementation in [functions/dailyBacktest.py](functions/dailyBacktest.py): the rebalance logic should read from `signal2D_daily` (post-filter) at rebalance dates and forward-fill into `signal2D`. If not, update accordingly.
   - Add tests to assert that when `signal2D_daily` contains zeros for a symbol on rebalance date, that symbol's monthly holdings remain zero for the month.
   - Verification: small synthetic test creating `adjClose` and `signal2D_daily` with a zeroed symbol at rebalance and asserting monthly forward-fill preserves the zero.

4. Prevent destructive normalization in ranking function (phase 3)
   - Goal: ensure `sharpeWeightedRank_2D` (in [functions/TAfunctions.py](functions/TAfunctions.py)) does not normalize or mutate `signal2D` in a way that re-enables previously-zeroed signals.
   - Action:
     - Use a binary mask `signal_mask = (signal2D > 0).astype(float)` early and use it for multiplications (gainloss, monthgainloss) and selection checks.
     - Avoid subtracting/normalizing `signal2D` in-place.
   - Tests: extend `tests/test_sharpe_weighted_rank.py` to include a simple case where `signal2D_daily` zeros are present and ensure weights computed for the month do not include zeroed symbols.

5. Data-loader parity (phase 4)
   - Problem: `tests/test_phase4a_shadow.py` shows ~0.1–0.5% mismatches between `adjClose_new` and `adjClose_old`.
   - Investigation steps (read-only first):
     - Run a small script that computes `np.where(adjClose_new != adjClose_old)` and prints the first 20 mismatches with row, column, new, old, and the sequence of transforms applied.
     - Confirm types (float32 vs float64), `np.nan` handling, order of `interpolate`, `cleantobeginning`, `cleantoend`, and whether any in-place mutations occur.
   - Fix approach:
     - Align the new loader to apply identical transforms with identical dtype and `np.nan` behavior, or document deliberate differences and update tests to allow exact tolerance if appropriate.
   - Verification: run the two failing data-loader shadow tests until arrays are identical.

6. Full regression and backtest validation (phase 5)
   - Run the full pytest suite; if green, run a controlled backtest on sp500 dataset and inspect `pytaaa_sp500_pine.log` for the expected RollingFilter output and selection prints for `JEF`.
   - Commands:

```bash
uv run python -m pytest -q
uv run python pytaaa_main.py --json /Users/donaldpg/pytaaa_data/sp500_pine/pytaaa_sp500_pine.json 2>&1 | tee pytaaa_sp500_pine.log
```

   - Confirm log contains lines like the earlier RollingFilter zeroing statements and that `print_JEF_selections` reflects zeros (no large JEF weights) for dates that were filtered.

7. Code review and cleanup (after each phase)
   - For each phase, do a focused code review commit with these checks:
     - No debug prints left except those gated by `verbose` or a logger at `DEBUG` level.
     - New argument `inplace` documented in docstring and `functions/README` if present.
     - Tests added/updated with clear assertions and comments.
     - Lint formatting (PEP8) and type annotations where appropriate.
   - Run fast tests covering only changed modules before committing.

**Checklist (phase-gated)**
- Phase 1 — Rolling filter robustness
  - [ ] Add safe fallbacks for `datearray`/`symbols` and `inplace` param
  - [ ] Add tests for `datearray=None` and `symbols=None`
  - [ ] Fix or update `test_in_place_modification` to test both behaviors
  - [ ] Run `tests/test_rolling_window_filter.py` and pass
  - [ ] Code review & clean

- Phase 2 — Monthly-hold semantics
  - [ ] Verify `dailyBacktest` uses `signal2D_daily` at rebalance
  - [ ] Add synthetic test demonstrating forward-fill preserves zeros
  - [ ] Run targeted tests and pass
  - [ ] Code review & clean

- Phase 3 — Ranking function safety
  - [ ] Use `signal_mask` inside `sharpeWeightedRank_2D` and remove destructive normalization
  - [ ] Add tests to ensure zeroed signals stay zero in weight output
  - [ ] Run `tests/test_sharpe_weighted_rank.py` (or new tests) and pass
  - [ ] Code review & clean

- Phase 4 — Data-loader parity
  - [ ] Produce mismatch report for `adjClose_new` vs `adjClose_old`
  - [ ] Align transform pipeline and dtype/NaN handling
  - [ ] Update or relax tests only if change is deliberate and documented
  - [ ] Run failing `test_phase4a_shadow.py` tests and pass
  - [ ] Code review & clean

- Phase 5 — Integration & backtest
  - [ ] Run full pytest suite
  - [ ] Run signaled backtest and inspect `pytaaa_sp500_pine.log` for expected behavior
  - [ ] Final code review, update docs, and commit

**Agent handoff instructions (for a new copilot agent)**
- Prioritize reproducing failing tests locally. Use the exact pytest commands above to get failing traces.
- Start with Phase 1 changes (robustness) — they are low-risk and unblock many tests.
- For each change, run only the focused test set that covers the module first, then expand to full tests.
- Keep changes minimal and add descriptive commit messages like `fix(rolling-filter): handle missing datearray and add inplace flag`.
- After implementing, run `uv run python -m pytest tests/test_rolling_window_filter.py -q` and attach the stdout to the PR.

**Diagnostics examples (log vs code expected behavior)**
- Log shows:

    DEBUG: Checking JEF on 2015-09-01: gainloss_std=0.000002
    RollingFilter: Zeroing JEF on 2015-09-01 due to low gainloss_std=0.000002

  Code expectation: when a window has near-zero gain-loss std, `signal_out[stock_idx, date_idx]` should be set to 0.0 and these zeros must be used by `dailyBacktest` at rebalance to prevent selection for the month.

- But selection prints show (contradicting expectation):

    2015-01-02: [JEF:0.6942, ...]

  This implies a later step re-enabled JEF (e.g., normalization in `sharpeWeightedRank_2D`, in-place mutation issues, or monthly-selection logic reading pre-filter signals). The plan's phases target these possible causes.

**Review & critique checklist (for plan improvements)**
- Does the plan list all code paths that read/write `signal2D`? If not, add a search step to locate all usages and document them.
- Are there performance/memory constraints from copying large arrays? Add a performance note and optional benchmark test in Phase 1.
- Confirm expected behavior for `inplace` default (copy-by-default) — if the codebase tests or callers assume in-place, consider toggling default or updating callers/tests.
- Add logging-to-logger migration step if project prefers structured logging over prints.

**Appendix — Quick verification commands**

```bash
# Run focused rolling filter tests
uv run python -m pytest tests/test_rolling_window_filter.py -q

# Run the two data-loader shadow tests
uv run python -m pytest tests/test_phase4a_shadow.py::TestDataLoaderShadow::test_data_loader_matches_inline_naz100 -q -vv
uv run python -m pytest tests/test_phase4a_shadow.py::TestDataLoaderShadow::test_data_loader_matches_inline_sp500 -q -vv

# Run full suite
uv run python -m pytest -q

# Re-run a short backtest and capture logs
uv run python pytaaa_main.py --json /Users/donaldpg/pytaaa_data/sp500_pine/pytaaa_sp500_pine.json 2>&1 | tee pytaaa_sp500_pine.log
# Then inspect for example lines
grep -n "RollingFilter: Zeroing JEF" pytaaa_sp500_pine.log | head
grep -n "STOCK SELECTIONS AT BEGINNING" pytaaa_sp500_pine.log | sed -n '1,40p'
```

---

End of plan.
