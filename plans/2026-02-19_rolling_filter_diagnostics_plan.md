## Plan: Rolling Window Filter Diagnostics and Fix (Revised 2026-02-20)

**Status:** Root cause confirmed. Ready for implementation.

### TL;DR

`JEF` (a stock with artificially linear-infilled prices in 2015–2018) appears in
portfolio selections because `get_json_params()` in
[functions/GetParams.py](functions/GetParams.py) never reads `enable_rolling_filter`
or `window_size` from the JSON `Valuation` section. As a result,
`_params.get('enable_rolling_filter', False)` in
[functions/dailyBacktest.py](functions/dailyBacktest.py) always returns `False` and
the rolling window filter never runs for the `pytaaa_main.py` entry point.

A secondary bug in `sharpeWeightedRank_2D`
([functions/TAfunctions.py](functions/TAfunctions.py)) assigns `equal_weight` to
every stock — including filtered-out ones — when `eligible_stocks` is all-zero
outside the 2000–2002 early period, further undermining the filter.

---

### Confirmed Root Cause

| Component | File | Issue |
|---|---|---|
| **Primary** | `functions/GetParams.py` ~line 455 | `get_json_params` returns `params` dict without `enable_rolling_filter` or `window_size` — keys exist in JSON `Valuation` section but are never read |
| **Secondary** | `functions/TAfunctions.py` ~line 918 | Equal-weight fallback assigns `1/n_stocks` to every stock (incl. filtered-out JEF) when `eligible_stocks` all-zero outside 2000–2002 |

**Why the working script works:**
`PyTAAA_backtest_sp500_pine_refactored.py` uses `validate_backtest_parameters()`
which defaults `enable_rolling_filter=True` and reads directly from the params dict,
bypassing `get_json_params`.

---

### Entry Points (do not confuse these)

| Entry point | Command | Notes |
|---|---|---|
| Buggy path | `uv run python pytaaa_main.py --json /Users/donaldpg/pytaaa_data/sp500_pine/pytaaa_sp500_pine.json` | ~5 min; sends email + writes to live website — **do not run against production JSON during testing** |
| Working path | `uv run python PyTAAA_backtest_sp500_pine_refactored.py --trials 1` | Uses same filter/backtest functions but reads config via `validate_backtest_parameters` |

**Dev JSON (safe to edit):**
`/Users/donaldpg/PyProjects/worktree2/PyTAAA/pytaaa_sp500_pine_dev.json`
— copied from production; contains `"enable_rolling_filter": true` and
`"window_size": 50` in the `Valuation` section.

**Production JSON (do not modify):**
`/Users/donaldpg/pytaaa_data/sp500_pine/pytaaa_sp500_pine.json`

---

### Already Done — No Changes Needed

- `functions/rolling_window_filter.py` — safe fallbacks for `datearray=None` and
  `symbols=None` already present; correctly zeros JEF during 2015–2018 infill period
  when called; returns a copy (no in-place mutation). **No changes needed.**
- `functions/dailyBacktest.py` monthly forward-fill logic — correctly reads
  `signal2D_daily` at rebalance dates. **No changes needed.**

---

### Implementation Plan

#### Phase 1 — Write failing test (must fail before fix)

Create `tests/test_jef_not_held_in_portfolio.py`:

```python
from functions.GetParams import get_json_params

DEV_JSON = "pytaaa_sp500_pine_dev.json"

def test_get_json_params_exposes_rolling_filter():
    """Fails before GetParams fix; passes after."""
    params = get_json_params(DEV_JSON)
    assert "enable_rolling_filter" in params, (
        "get_json_params must return enable_rolling_filter"
    )
    assert params["enable_rolling_filter"] is True
    assert "window_size" in params
    assert params["window_size"] == 50
```

Run it: `uv run python -m pytest tests/test_jef_not_held_in_portfolio.py -q`
→ must **FAIL** before the fix.

#### Phase 2 — Fix `functions/GetParams.py` (2 lines)

In `get_json_params()`, before the final `return params` (~line 455), add:

```python
params["enable_rolling_filter"] = bool(
    valuation_section.get("enable_rolling_filter", False)
)
params["window_size"] = int(valuation_section.get("window_size", 50))
```

Re-run the Phase 1 test — must now **PASS**.

#### Phase 3 — Write rolling filter isolation test

Create `tests/test_jef_signal_zeroed.py` using synthetic data with a
linear-trend price series and assert `apply_rolling_window_filter` zeros
`signal2D` for the infilled rows. This test should pass before and after the fix
(it tests the filter in isolation, not the integration path).

#### Phase 4 — Fix equal-weight fallback in `functions/TAfunctions.py` (~line 918)

In the non-early-period branch, replace the equal-weight fallback:

```python
# BEFORE (re-enables all stocks including filtered-out JEF):
equal_weight = 1.0 / n_stocks
monthgainlossweight[:, j] = equal_weight

# AFTER (100% to CASH if present, else leave weights at zero):
cash_indices = [i for i, s in enumerate(symbols) if s == "CASH"]
if cash_indices:
    monthgainlossweight[cash_indices[0], j] = 1.0
# else: leave all weights at 0.0; forward-fill handles continuity
```

#### Phase 5 — Write weight-respects-zeros test

Create `tests/test_sharpe_rank_respects_zeros.py`:
- Synthetic data where `signal2D` for JEF is all-zero.
- Assert `sharpeWeightedRank_2D` returns zero weight for JEF at those dates.

#### Phase 6 — Run full test suite

```bash
uv run python -m pytest -q
```

All three new tests must pass.

#### Phase 7 — Integration check (fast path, no email/web side effects)

Use the working entry point to confirm JEF absent in 2015–2018 selections:

```bash
uv run python PyTAAA_backtest_sp500_pine_refactored.py --trials 1 \
  > backtest_after_fix.log 2>&1
grep -E "2015|2016|2017|2018" backtest_after_fix.log | grep "JEF"
```

No JEF lines in 2015–2018 year-start holdings = success.

**Do NOT run `pytaaa_main.py` against production JSON** — it sends email and writes
to the live website. Use only `pytaaa_sp500_pine_dev.json` if testing that path.

---

### Checklist

**Phase 1 — Failing test**
- [x] Write `tests/test_jef_not_held_in_portfolio.py`
- [x] Confirm test FAILS before fix

**Phase 2 — GetParams fix**
- [x] Add 2 lines to `functions/GetParams.py` before `return params`
- [x] Confirm Phase 1 test now PASSES

**Phase 3 — Filter isolation test**
- [x] Write `tests/test_jef_signal_zeroed.py` (should pass immediately)

**Phase 4 — Sharpe fallback fix**
- [x] Replace equal-weight fallback in `functions/TAfunctions.py` ~line 918
- [x] Confirm no regressions in existing tests

**Phase 5 — Weight test**
- [x] Write `tests/test_sharpe_rank_respects_zeros.py`
- [x] Confirm PASSES

**Phase 6 — Full test suite**
- [x] `uv run python -m pytest -q` → 164 passed, 4 pre-existing failures (shadow tests), 2 skipped

**Phase 7 — Integration**
- [x] `PyTAAA_backtest_sp500_pine_refactored.py --trials 1` → no JEF in 2015–2018 portfolio selections; `RollingFilter: Zeroing JEF` debug lines confirm filter is active. Pre-existing IndexError in post-computation stats (unrelated to this fix).

---

### Key File Map

| File | Role | Change? |
|---|---|---|
| `functions/GetParams.py` | Reads JSON → `params` dict | **YES** — add 2 lines |
| `functions/TAfunctions.py` | `sharpeWeightedRank_2D` weight assignment | **YES** — fix fallback |
| `functions/dailyBacktest.py` | Filter gate + forward-fill | No |
| `functions/rolling_window_filter.py` | Zeros infilled signals | No |
| `pytaaa_sp500_pine_dev.json` | Dev copy of production JSON | No (edit-safe copy) |
| `/Users/donaldpg/pytaaa_data/sp500_pine/pytaaa_sp500_pine.json` | Production JSON | **Never modify** |

---

End of plan.
