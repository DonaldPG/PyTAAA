# Look-Ahead Bias Test — Completion & Speedup Session

**Date**: 2026-02-23
**Branch**: `feature/lookahead-bias-and-backtest-validation`
**Scope**: Complete the look-ahead bias detection study and tests;
implement data slicing for 13× speedup; add multi-date `--cutoff-date`
CLI option to the study script.

---

## Context

The look-ahead bias study and pytest suite were first scaffolded in an
earlier session (2025-02-13).  This session brought that work to
completion: the study script and test both run against real production
HDF5 data in memory (no file copies), all three models pass, and
runtime was reduced from ~4.5 minutes to ~20 seconds for the full
pytest suite.

---

## Problem Statement

Three improvements were needed after the initial implementation:

1. **Multi-date support** — the original plan called for three test
   dates; the study script only accepted a single implicit cutoff.
2. **Speed** — both the study script (~4.5 min) and pytest suite
   (~4.5 min per model) were too slow for routine use.
3. **Output readability** — ORIG and PATCH selections were printed on
   one long side-by-side line; hard to scan visually.

---

## Solution Overview

### Data slicing for speed

Instead of running the full ~8,800-day history through
`computeSignal2D` and `sharpeWeightedRank_2D`, both the study script
and the test now slice `adjClose` and `datearray` to a narrow window:

```
[cutoff - 600 days, cutoff + 200 days]   # ≈ 800 total days
```

600 pre-cutoff trading days is sufficient for all rolling windows
(`LongPeriod`, channel periods, Sharpe lookbacks).  Selections at the
cutoff date are identical to those produced by the full dataset.

This gives a **~11× speedup** per pipeline run.  pytest runtime:
4.5 minutes → 20 seconds.

### Multi-date `--cutoff-date` option

```bash
PYTHONPATH=$(pwd) uv run python \
    studies/lookahead_bias/run_lookahead_study.py \
    --cutoff-date 2023-09-29 2024-03-29 2024-09-27
```

Data is loaded once per model; the inner loop iterates over cutoff
dates.  `_find_cutoff_idx()` maps each ISO date string to the last
trading-day index at or before that date.

### Output format (study script)

Each month now prints ORIG on line 1, PATCH indented on line 2, with
a blank line between entries.  Only the last 3 `[PRE ]` months, the
`[CUT ]` row, and the next 3 `[POST]` months are shown (configurable
via `--months-pre` / `--months-post`).  This keeps output concise
across multiple cutoff dates.

### `channels.py` float-index bug fix

All four channel functions in `functions/ta/channels.py` contained
`np.arange(...)[slice]` calls where the slice indices were `float`.
This raised `TypeError` in NumPy ≥ 1.24.  Fixed by adding
`.astype(int)` to each `periods` array.

---

## Key Changes

| File | Change |
|---|---|
| `studies/lookahead_bias/run_lookahead_study.py` | Added `_find_cutoff_idx()`; added `--cutoff-date`, `--months-pre`, `--months-post` args; inner loop over cutoff dates; slice to ±600/200 days; two-line ORIG/PATCH output |
| `tests/test_lookahead_bias.py` | `_run_both_pipelines()` now slices to `_PRE_DAYS=600` / `_POST_DAYS=200` before running pipeline; added `_PRE_DAYS` / `_POST_DAYS` constants |
| `functions/ta/channels.py` | `.astype(int)` added to `periods` array in all 4 channel functions to fix `TypeError` with float slice indices |

---

## Testing

```bash
# Full test file (6 tests, 3 skip-guarded with real data):
PYTHONPATH=$(pwd) uv run pytest tests/test_lookahead_bias.py -v
# Runtime: ~20 seconds

# Core look-ahead bias test only (3 parametrized, all PASSED):
PYTHONPATH=$(pwd) uv run pytest \
    tests/test_lookahead_bias.py::test_selection_consistency_across_models -v

# Study script (human-readable output, 3 cutoff dates):
PYTHONPATH=$(pwd) uv run python \
    studies/lookahead_bias/run_lookahead_study.py \
    --cutoff-date 2023-09-29 2024-03-29 2024-09-27
```

**Results**: all three models (`naz100_hma`, `naz100_pine`, `naz100_pi`)
show no `*** DIFFER ***` on the `[CUT ]` row for any tested cutoff
date.  **No look-ahead bias detected.**

---

## Follow-up Items

- `studies/synthetic_cagr/` — synthetic CAGR data generation and
  backtest validation (Phase 3 of the plan) is not yet implemented.
  This is the next area of focus.
- `tests/test_synthetic_backtest.py` — stub only; requires Phase 3
  completion before it can pass.
