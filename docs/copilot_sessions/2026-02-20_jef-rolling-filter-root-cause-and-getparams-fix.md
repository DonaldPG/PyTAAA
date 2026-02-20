# Session Summary: JEF Rolling Filter Root Cause and GetParams Fix

**Date:** 2026-02-20
**Context:** Investigating why JEF (a stock with artificially linear-infilled prices
in 2015–2018) was incorrectly selected as a portfolio holding during that period
in backtests run via `pytaaa_main.py`, despite the rolling window filter being
designed to exclude it.

---

## Problem Statement

JEF prices were known to be infilled with a synthetic linear trend from 2015 to
2018 (zero real price variance). The rolling window filter in
`functions/rolling_window_filter.py` was designed to detect and zero out signals
for stocks with near-zero gain/loss standard deviation in any rolling window.
Despite this, JEF appeared in portfolio selections at year starts during that
period, e.g.:

```
2015-01-02: [JEF:0.6942, LUV:0.0910, ...] Sum=1.0000
2016-01-04: [JEF:0.7143, ANDV:0.0714, ...] Sum=1.0000
```

---

## Investigation

### Entry Points Examined

| Entry point | Command | Behavior |
|---|---|---|
| **Buggy** | `uv run python pytaaa_main.py --json /Users/donaldpg/pytaaa_data/sp500_pine/pytaaa_sp500_pine.json` | JEF selected in 2015–2018; ~5 min runtime; sends email + writes to live website |
| **Working** | `uv run python PyTAAA_backtest_sp500_pine_refactored.py --trials 1` | JEF correctly excluded; uses same filter/backtest functions |

### Call Chain (buggy path)

```
pytaaa_main.py
  → run_pytaaa.py::run_pytaaa()
    → PortfolioPerformanceCalcs.py::PortfolioPerformanceCalcs()
      → _write_daily_backtest()
        → functions/dailyBacktest.py::computeDailyBacktest()
            _params = get_json_params(json_fn)
            if _params.get('enable_rolling_filter', False):   ← always False
                apply_rolling_window_filter(...)
```

### What Was Checked

1. **`functions/rolling_window_filter.py`** — Confirmed already correct. Has safe
   fallbacks for `datearray=None` and `symbols=None`. Zeros signals correctly when
   `gainloss_std < 0.001`. Returns a copy (no in-place mutation). No changes needed.

2. **`functions/dailyBacktest.py`** — Confirmed the filter gate is correctly written:
   `if _params.get('enable_rolling_filter', False):`. The default `False` is the
   bug trigger — not a logic error in the gate itself.

3. **Production JSON at `/Users/donaldpg/pytaaa_data/sp500_pine/pytaaa_sp500_pine.json`**
   — Confirmed it contains `"enable_rolling_filter": true` and `"window_size": 50`
   inside the `Valuation` section. **The JSON has the correct values.**

4. **`functions/GetParams.py::get_json_params()`** — Searched for all key reads from
   the `Valuation` section. Found reads for `LongPeriod`, `MA1`, `MA2`, `stockList`,
   `symbols_file`, `minperiod`, `maxperiod`, `incperiod`, etc. — but **no read for
   `enable_rolling_filter` or `window_size`**. These keys are silently dropped and
   never included in the returned `params` dict.

5. **`PyTAAA_backtest_sp500_pine_refactored.py`** — Confirmed it uses
   `validate_backtest_parameters()` which defaults `enable_rolling_filter=True`
   and reads directly from the params dict, bypassing `get_json_params`. This
   explains why the working path correctly excludes JEF.

### Secondary Bug Identified (not yet fixed)

In `functions/TAfunctions.py::sharpeWeightedRank_2D()` (~line 918): when
`eligible_stocks` is all-zero outside the 2000–2002 early period (e.g., because
the rolling filter has zeroed JEF signals), a fallback assigns equal weight
`1/n_stocks` to **every** stock, including filtered-out ones. This is a secondary
path by which JEF can re-enter the portfolio even if the filter runs.

---

## Findings and Conclusions

**Root cause (primary):** `get_json_params()` in `functions/GetParams.py` never
reads `enable_rolling_filter` or `window_size` from the JSON `Valuation` section.
The fix is exactly 2 lines added before `return params`.

**Why the working script works:** It uses `validate_backtest_parameters()` which
supplies the correct defaults and reads directly from the params dict.

**Production JSON is fine:** No changes to the JSON are needed or were made.

---

## Changes Made

### 1. Created dev JSON copy (safe to edit)

```bash
cp /Users/donaldpg/pytaaa_data/sp500_pine/pytaaa_sp500_pine.json \
   /Users/donaldpg/PyProjects/worktree2/PyTAAA/pytaaa_sp500_pine_dev.json
```

Verified: `enable_rolling_filter: true` and `window_size: 50` present in
`Valuation` section. Production JSON left untouched.

### 2. Written: `tests/test_jef_not_held_in_portfolio.py`

Three tests:
- `test_dev_json_exists` — sanity check that dev JSON is present
- `test_get_json_params_exposes_enable_rolling_filter` — asserts key present and `True`
- `test_get_json_params_exposes_window_size` — asserts key present and `== 50`

**Observed before fix:** 2 tests FAILED with `AssertionError: 'enable_rolling_filter'
not in params dict`.

### 3. Fixed: `functions/GetParams.py`

Added 2 lines before `return params` (~line 451):

```python
params['enable_rolling_filter'] = bool(
    valuation_section.get('enable_rolling_filter', False)
)
params['window_size'] = int(valuation_section.get('window_size', 50))
```

**Observed after fix:** All 3 tests pass.

```
tests/test_jef_not_held_in_portfolio.py::test_dev_json_exists PASSED
tests/test_jef_not_held_in_portfolio.py::test_get_json_params_exposes_enable_rolling_filter PASSED
tests/test_jef_not_held_in_portfolio.py::test_get_json_params_exposes_window_size PASSED
3 passed in 0.08s
```

### 4. Updated: `plans/2026-02-19_rolling_filter_diagnostics_plan.md`

- Rewrote plan with confirmed root cause, correct entry points, dev JSON paths,
  and a precise 7-phase implementation checklist (replacing the original outdated
  plan which still described unresolved hypotheses).
- All 7 phase checkboxes marked complete by end of session.

### 5. Written: `tests/test_jef_signal_zeroed.py` (Phase 3)

Six tests using synthetic data with a linear-trend price series:
- `test_filter_zeros_jef_after_window_warmup` — confirms signal zeroed after warmup
- `test_filter_preserves_jef_before_window_warmup` — signal untouched before warmup
- `test_filter_preserves_noisy_stock_signals` — realistic prices not filtered
- `test_filter_does_not_mutate_input_signal` — confirms returns a copy
- `test_linear_prices_gainloss_std_below_threshold` — validates dataset assumption
- `test_noisy_prices_gainloss_std_above_threshold` — validates dataset assumption

All 6 passed immediately (filter was already correct; this tests it in isolation).

### 6. Fixed: `functions/TAfunctions.py::sharpeWeightedRank_2D()` (Phase 4)

**Secondary bug:** Two `else` fallback blocks (one for `all_signals_zero` ~line 884,
one for `no eligible stocks` ~line 930) both ran:

```python
# BEFORE — re-enables filtered-out stocks including JEF:
equal_weight = 1.0 / n_stocks
monthgainlossweight[:, j] = equal_weight
```

This spread weight equally across all stocks — including ones the rolling filter
had zeroed out — whenever no stock had a positive signal.

```python
# AFTER — 100% to CASH if present, else leave weights at 0:
cash_idx = symbols.index('CASH') if 'CASH' in symbols else None
if cash_idx is not None:
    monthgainlossweight[:, j] = 0.0
    monthgainlossweight[cash_idx, j] = 1.0
else:
    pass  # already 0.0 from initialization; forward-fill handles continuity
```

Both fallback locations were fixed simultaneously.

### 7. Written: `tests/test_sharpe_rank_respects_zeros.py` (Phase 5)

Five tests in two classes:
- `TestAllSignalsZero::test_jef_weight_is_zero_when_all_signals_zero`
- `TestAllSignalsZero::test_cash_weight_is_one_when_all_signals_zero`
- `TestFilteredStockZeroSignal::test_jef_weight_is_zero_when_signal_is_zero`
- `TestFilteredStockZeroSignal::test_aapl_has_nonzero_weight_post_warmup`
- `TestFilteredStockZeroSignal::test_weight_sum_is_valid`

All 5 passed after the TAfunctions fix.

### 8. Full test suite (Phase 6)

```
uv run python -m pytest -q
```

Result: **164 passed, 4 failed, 2 skipped** in 54:55.

The 4 failures are all pre-existing shadow/refactoring comparison tests
(`test_phase4a_shadow.py`, `test_phase4b_shadow.py`) unrelated to this fix.
All 14 new tests passed.

### 9. Integration check (Phase 7)

**Entry point 1:** `PyTAAA_backtest_sp500_pine_refactored.py --trials 1`

```bash
grep -E "2015|2016|2017|2018" backtest_after_fix.log | grep "JEF"
# → no portfolio selection lines; only RollingFilter: Zeroing JEF entries
```

JEF not selected. Filter active. ✅

**Entry point 2:** `pytaaa_main.py --json /Users/donaldpg/pytaaa_data/sp500_pine/pytaaa_sp500_pine.json`

Completed at `2026-02-20 02:09:24`. Log confirms:

```
DIAG: Date 2015-01-02 JEF daily=0.000000 month=0.000000 mask=0.0
DIAG: Date 2016-01-04 JEF daily=0.000000 month=0.000000 mask=0.0
DIAG: Date 2017-01-02 JEF daily=0.000000 month=0.000000 mask=0.0
DIAG: Date 2018-01-01 JEF daily=0.000000 month=0.000000 mask=0.0
```

JEF zero weight throughout 2015–2018. Webpage written successfully. ✅

---

## Testing

| Phase | Test file | Tests | Result |
|---|---|---|---|
| 1 | `test_jef_not_held_in_portfolio.py` | 3 | 3 pass (2 failed before fix) |
| 3 | `test_jef_signal_zeroed.py` | 6 | 6 pass |
| 5 | `test_sharpe_rank_respects_zeros.py` | 5 | 5 pass |
| 6 | Full suite | 170 | 164 pass, 4 pre-existing failures, 2 skipped |

---

## Follow-up Items

| Item | Notes |
|---|---|
| Pre-existing `test_phase4a/4b_shadow.py` failures | 4 shadow/refactoring comparison tests unrelated to this fix; should be investigated separately |
| `IndexError` in `PyTAAA_backtest_sp500_pine_refactored.py` line ~1958 | `numberStocksUpTrending[iter,:]` out of bounds on `--trials 1` run; pre-existing, not introduced by this fix |
