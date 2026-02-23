# Plan: Look-Ahead Bias & Backtest Validation Test Suite

## Context Summary

The PyTAAA signal pipeline computes `signal2D` and `sharpeWeightedRank_2D`
over the full historical date range in a single vectorized call before any
simulation loop runs. While code inspection suggests all moving averages and
channels are causally computed (past data only), this has never been verified
experimentally. This plan creates two empirical test harnesses:

- **Phase 2** — look-ahead bias in stock selection: prove that future prices
  do not influence which stocks are selected on any past date.
- **Phase 3** — end-to-end backtest correctness: prove the pipeline correctly
  identifies and trades the highest-return stocks when the ground truth is
  known.

All new code lives in `studies/` during research. Permanent regression tests
are promoted to `tests/` in Phase 4.

**Git branch**: `feature/lookahead-bias-and-backtest-validation`

---

## Status (as of 2026-02-23)

| Phase | Status | Notes |
|---|---|---|
| Phase 1 — Infrastructure | ✅ COMPLETE | `hdf5_utils.py`, `patch_strategies.py`, `selection_runner.py` created |
| Phase 2 — Look-Ahead Bias | ✅ COMPLETE | Implemented differently from plan — see note below |
| Phase 3 — Synthetic CAGR | ⬜ NOT STARTED | Next priority |
| Phase 4 — Test Integration | 🔶 PARTIAL | `test_lookahead_bias.py` done; `test_synthetic_backtest.py` stub only |
| Phase 5 — Documentation | 🔶 PARTIAL | Session docs written; README updates pending |

### Phase 2 Implementation Notes (deviations from original plan)

The original plan called for patched HDF5 *files* (copy → modify on disk).
The actual implementation uses **in-memory patching** of the `adjClose`
NumPy array instead, which is simpler, safer, and faster.

Key differences from the plan:

- No test HDF5 files are created or copied. `_patch_adjclose()` returns a
  modified numpy array; the source array is never written to disk.
- `run_lookahead_study.py` replaces the planned `experiment_future_prices.py`
  / `plot_results.py` / `evaluate_future_prices.py` trilogy with a single
  human-readable CLI script.
- `test_lookahead_bias.py` uses real production HDF5 data (not synthetic)
  sliced to ±600/200 days around the cutoff for speed (~20 s for 3 models).
- `--cutoff-date` CLI option accepts a list of ISO dates; the inner loop
  reuses the loaded HDF5 data across all cutoff dates.
- A production bug in `functions/ta/channels.py` was discovered and fixed:
  float slice indices caused `TypeError` in NumPy ≥ 1.24; fixed with
  `.astype(int)` on the `periods` array.

**Result**: All 3 models (`naz100_hma`, `naz100_pine`, `naz100_pi`) show
no look-ahead bias at all tested cutoff dates. Tests PASS in ~20 seconds.

---

## Phase 1 — Infrastructure & Fixtures ✅ COMPLETE

### Tasks

1. Create `studies/lookahead_bias/` with `__init__.py`, `README.md`, and
   `params/` subdirectory. Create `studies/synthetic_cagr/` analogously.

2. Write `studies/lookahead_bias/hdf5_utils.py` with two functions:

   - `copy_hdf5(src_path, dst_path)` — opens `src_path` with
     `pd.HDFStore(src_path, mode='r')` (read-only), then copies to
     `dst_path` via `shutil.copy2`. **The source file is never opened in
     write mode and is never modified under any circumstances.** Afterward,
     validates the copy is structurally identical to the source (same keys,
     same column names, same shape, same first and last values per column)
     by calling `pd.read_hdf` on the **copy only** — the source is not
     re-opened.
   - `patch_hdf5_prices(hdf5_path, symbol_patches, cutoff_date)` — opens
     `hdf5_path` (always a copy, never the original) in write mode; for each
     symbol in `symbol_patches`, applies the transform function only to rows
     with `index > cutoff_date`; asserts all rows with `index <= cutoff_date`
     are unchanged before writing back.
     ```
     symbol_patches: dict[str, Callable[[pd.Series], pd.Series]]
     cutoff_date: str  # "YYYY-MM-DD"
     ```

3. Write `studies/lookahead_bias/patch_strategies.py` — reusable
   perturbation callables: `step_down(magnitude)`, `step_up(magnitude)`,
   `linear_down(slope)`, `linear_up(slope)`. Each callable takes a
   `pd.Series` and `cutoff_date`; pre-cutoff values are left exactly
   unchanged.

4. Write `studies/lookahead_bias/selection_runner.py` — thin wrapper that
   accepts a path to a (possibly patched) HDF5 file and a JSON params file
   override, calls the existing `computeDailyBacktest` directly (bypassing
   file-modification checks), and returns the ranked stock list for a
   specified `as_of_date`. Supports all three models: `naz100_hma`,
   `naz100_pine`, `naz100_pi`.

### Checklist

- [x] `copy_hdf5` never opens the source in write mode
- [x] `patch_hdf5_prices` assertion: all pre-cutoff rows are unchanged
- [x] `selection_runner.py` created
- [x] `_patch_adjclose()` replaces HDF5 file patching with in-memory array
      patching (simpler, safer — see Phase 2 notes above)

---

## Phase 2 — Look-Ahead Bias Study ✅ COMPLETE (implemented differently)

### Tasks

5. Write `studies/lookahead_bias/experiment_future_prices.py`:

   - **3 test dates**: 2019-06-28, 2021-12-31, 2023-09-29 (adjustable via
     params JSON)
   - For each test date, run `selection_runner` on the real HDF5 to
     establish the true top-15 ranking under each of the 3 models
   - **Assign perturbations**: stocks ranked 1–8 receive `step_down(0.30)`
     or `linear_down` applied to prices *after* the test date; stocks ranked
     9–15 receive `step_up(0.30)` or `linear_up` after the test date;
     remaining stocks are unmodified
   - Save the perturbation manifest as a JSON side-car: which tickers, which
     strategy, which cutoff date
   - Run both pipelines (real HDF5, patched copy) for each test date x each
     model -> 3 dates x 3 models = **9 comparison pairs**
   - Record results to CSV: `date`, `model`, `real_top_n`, `patched_top_n`,
     `selection_changed` (bool), `rank_correlation`

6. Write `studies/lookahead_bias/plot_results.py` — for each of the 9
   comparison pairs, produce a portfolio-value history plot matching the
   style of the **upper subplot only** of `PyTAAA_monteCarloBacktestFull.png`
   (reference: `plotRecentPerfomance3` in
   `functions/dailyBacktest_pctLong.py`):

   - Single subplot (no lower stock-count panel for this study)
   - Log y-scale; y-axis starting near $7,000; x-axis with annual tick marks
   - **Two bold curves only** — no Monte Carlo light-black curves, no
     individual stock light-red curves:
     - Traded portfolio: black, `lw=4`, normalized to $10,000 at start
     - Buy-and-hold: red, `lw=3`, normalized to $10,000 at start
   - `plt.text()` annotation table in the same column style: `Period /
     Sharpe / AvgProfit / Avg DD` header, then rows for Life, 3Yr, 1Yr,
     6Mo, 3Mo, 1Mo; font sizes 7.5-8 at the same relative log-scale
     positions as the reference
   - Title encoding model name, parameters, final portfolio value, and Sharpe
     in the same compact `title_text` format as the reference
   - "Backtested on ..." timestamp and data-source path text at the same
     relative log-scale positions
   - Saved as
     `studies/lookahead_bias/plots/lookahead_{date}_{model}_{real_or_patched}.png`

7. Write `studies/lookahead_bias/evaluate_future_prices.py`:

   - Reads the results CSV, prints a structured pass/fail report
   - **Pass criterion**: `selection_changed == False` for all 9 pairs
   - Failures print model + date + what changed

### Checklist (revised to match actual implementation)

- [x] All 3 models × multiple cutoff dates show `selection_changed == False`
- [x] `run_lookahead_study.py` prints human-readable ORIG/PATCH comparison
- [x] `--cutoff-date` CLI option accepts multiple ISO date strings
- [x] Data sliced to ±600/200 days; runtime ~20 s per pytest run
- [ ] Results CSV / 18 PNGs — not implemented (replaced by CLI script output)

---

## Phase 3 — Synthetic CAGR Data for Backtest Validation ⬜ NOT STARTED

### Tasks

8. Add `opensimplex` to `pyproject.toml` (`uv add opensimplex`). Write
   `studies/synthetic_cagr/noise_utils.py`:

   - `NoiseCalibratorFromHDF5(hdf5_path)` — reads multiple real price
     series; computes mean and std of daily returns, and mean and std of
     inter-extrema spacing (days between local min/max, via
     `scipy.signal.argrelextrema`) across all tickers; stores recommended
     `frequency` and `amplitude` parameters.
   - `opensimplex_noise_1d(n, frequency, amplitude, seed)` — generates a
     smooth, autocorrelated noise array using `opensimplex.noise2` evaluated
     along a single axis slice, scaled to `amplitude`.

9. Write `studies/synthetic_cagr/synthetic_hdf5_builder.py`:

   - 9 CAGR tiers x 10 stocks = **90 synthetic tickers**
     (e.g., `SYNTH_P20_01 ... SYNTH_N06_10`)
   - CAGR tiers: +20%, +15%, +12%, +10%, +8%, +6%, +3%, 0%, -6%
   - Daily price: geometric trend `P0 * exp(CAGR/252 * t)` + OpenSimplex
     noise (amplitude and frequency calibrated from real data)
   - **6-month rotation**: every ~126 trading days, on a randomly-drawn day
     within +-10 trading days of the boundary, reassign CAGR tiers to stock
     groups via random permutation. Same 10 stocks per tier at all times,
     just different assignments. New trend starts from closing price on
     rotation day — no discontinuity.
   - Total history: **5 years** of trading days (~1,260 rows)
   - Outputs:
     - `studies/synthetic_cagr/data/synthetic_naz100.hdf5`
     - `studies/synthetic_cagr/data/ground_truth.csv` —
       `(date, ticker, assigned_cagr_tier)` for every trading day

10. Write `studies/synthetic_cagr/backtest_runner.py`:

    - Creates a test JSON config in `studies/synthetic_cagr/params/`
      pointing at the synthetic HDF5
    - Runs `computeDailyBacktest` over the full 5-year history for each of
      the 3 naz100 models
    - Returns portfolio value series, stock-selection time series, and
      ground-truth tier assignments per model

11. Write `studies/synthetic_cagr/plot_results.py` — for each model,
    produce a portfolio-value history plot matching the upper subplot style
    of `PyTAAA_monteCarloBacktestFull.png`:

    - Same two bold curves (black `lw=4` traded, red `lw=3` buy-and-hold),
      log y-scale, annual x-axis ticks, same `plt.text()` annotation table,
      same title format
    - **Third curve**: green, `lw=2`, dashed — the hypothetical perfect
      +20% CAGR oracle portfolio, for visual comparison
    - Saved as `studies/synthetic_cagr/plots/synthetic_backtest_{model}.png`

12. Write `studies/synthetic_cagr/evaluate_synthetic.py`:

    - **Evaluation 1 — Portfolio CAGR**: annualized return of the simulated
      portfolio over the full 5-year window.
      **Pass**: CAGR between **19% and 21%**.
    - **Evaluation 2 — Stock selection accuracy**: at each rebalance,
      fraction of selected stocks in the +20% tier.
      **Pass**: >= 70% of held positions from the highest-CAGR tier.
    - **Evaluation 3 — Rotation responsiveness**: within 60 calendar days
      of each 6-month rotation, >= 50% of selected stocks from the new
      top-tier.
    - Saves a structured JSON report alongside the plots.

### Checklist

- [ ] Synthetic HDF5: 90 tickers x ~1,260 trading days
- [ ] Ground-truth CSV: spot-check 3 rotation dates for correctness
- [ ] Portfolio CAGR between 19% and 21% for at least one model
      (Evaluation 1)
- [ ] Stock selection accuracy >= 70% (Evaluation 2)
- [ ] Rotation responsiveness within 60 days (Evaluation 3)
- [ ] 3 PNGs generated in `studies/synthetic_cagr/plots/`

---

## Phase 4 — Test Suite Integration 🔶 PARTIAL

### Tasks

13. ✅ `tests/test_lookahead_bias.py` — uses real production HDF5 sliced
    to ±600/200 days (not synthetic); skips gracefully in CI without data.
    Actual approach (note — differs from original plan):

    - Tests **all 3 models**: `naz100_hma` (HMAs), `naz100_pine`
      (percentileChannels), `naz100_pi` (SMAs) — these exercise distinct
      code paths in `computeSignal2D`
    - 1 test date, patched future prices for top-8 stocks
    - Asserts `selection_changed == False` for all 3 models
    - No real HDF5 required; runs in < 30 seconds

14. Write `tests/test_synthetic_backtest.py` — lightweight pytest version
    (2 years of data, 30 tickers across 3 CAGR tiers: +20%, +10%, -6%):

    - Tests all 3 models
    - Asserts portfolio CAGR > mean CAGR of selected stocks' ground-truth
      tiers (proving model beats random tier selection)
    - No real HDF5 required; runs in < 30 seconds

### Checklist

- [x] `PYTHONPATH=$(pwd) uv run pytest tests/test_lookahead_bias.py -v`
      passes (20 s, 3 models, no look-ahead bias)
- [ ] `PYTHONPATH=$(pwd) uv run pytest tests/test_synthetic_backtest.py -v`
      passes — **BLOCKED on Phase 3**
- [ ] `PYTHONPATH=$(pwd) uv run pytest` (full suite) passes with no
      regressions

---

## Phase 5 — Documentation 🔶 PARTIAL

### Tasks

15. `studies/lookahead_bias/README.md` — ⬜ pending
    `studies/synthetic_cagr/README.md` — ⬜ pending (blocked on Phase 3)

16. Session summary docs created:
    - [x] `docs/copilot_sessions/2025-02-13_lookahead-bias-and-backtest-validation.md`
    - [x] `docs/copilot_sessions/2026-02-23_lookahead-bias-test-completion.md`

### Checklist

- [ ] `studies/lookahead_bias/README.md` updated with sliced-data approach
- [ ] `studies/synthetic_cagr/README.md` written (blocked on Phase 3)
- [x] Session summary docs created

---

## Verification

```bash
# Create and switch to the working branch
git checkout -b feature/lookahead-bias-and-backtest-validation

# Phase 2 (requires real Naz100 HDF5)
uv run python studies/lookahead_bias/experiment_future_prices.py
uv run python studies/lookahead_bias/plot_results.py
uv run python studies/lookahead_bias/evaluate_future_prices.py

# Phase 3 (synthetic only — no real data needed)
uv run python studies/synthetic_cagr/synthetic_hdf5_builder.py
uv run python studies/synthetic_cagr/backtest_runner.py
uv run python studies/synthetic_cagr/plot_results.py
uv run python studies/synthetic_cagr/evaluate_synthetic.py

# Permanent regression tests
PYTHONPATH=$(pwd) uv run pytest tests/test_lookahead_bias.py \
    tests/test_synthetic_backtest.py -v
PYTHONPATH=$(pwd) uv run pytest
```

---

## Decisions

- **opensimplex**: `opensimplex.noise2` used for smooth, autocorrelated
  synthetic noise; parameters calibrated from real Naz100 extrema statistics
  via `NoiseCalibratorFromHDF5`.
- **Source HDF5 safety**: source is opened exclusively with `mode='r'`;
  `patch_hdf5_prices` is called only on copies — enforced structurally in
  code, not by convention. Validation uses `pd.read_hdf` on the copy only.
- **Evaluation 1 pass band (19-21%)**: tight enough to detect incorrect CAGR
  assignment or model drift, while allowing for noise and occasional
  sub-optimal selections.
- **Test suite independence**: Phase 4 tests use only synthetic data (no
  real HDF5 dependency) so the full test suite can run in CI without data
  files.
- **Rotation timing**: random day within +-10 trading days of the 6-month
  boundary prevents alignment with fixed rebalance dates.
- **Plot style**: portfolio value plots mirror the upper subplot of
  `plotRecentPerfomance3` in `functions/dailyBacktest_pctLong.py` —
  log y-scale, two bold curves (black `lw=4` traded, red `lw=3`
  buy-and-hold), normalized to $10,000, `plt.text()` annotation table,
  annual x-axis ticks. No Monte Carlo curves or individual stock curves.
- **All 3 signal methods tested**: `naz100_hma` (HMAs), `naz100_pine`
  (percentileChannels), `naz100_pi` (SMAs) exercise distinct code paths in
  `computeSignal2D` — all must pass to validate the pipeline end-to-end.
