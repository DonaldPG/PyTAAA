# Copilot Session: NaN-Marking of HDF5 Infilled Prices

**Date:** 2026-03-05
**Branch:** orchestration-refactor

---

## Date and Context

Follow-on session to the algorithm-comparison investigation.  The prior
session established that the production HDF5 file stores prices for all
215 symbols covering 1991–2026, with constant-price or linearly-
interpolated infill for dates when a stock was not in the Nasdaq 100
index.  The user created a copy of the production file and asked for a
script that replaces those infilled values with `NaN`.

---

## Problem Statement

The file `/Users/donaldpg/pyTAAA_data/Naz100/symbols/Naz100_Symbols_nans.hdf5`
(8855 dates × 215 symbols) needs NaN where prices were infilled rather
than reflecting real trading.  Two infill patterns appear:

1. **Leading constant run** — first real price repeated back to
   1991-01-02 for stocks added after the dataset start.
2. **Trailing constant run** — last real price repeated forward to the
   present for delisted/removed stocks.
3. **Mid-history constant block** — short constant runs within the
   real-data period, indicating temporary exclusion from the index.
4. **Mid-history linear interpolation** — constant step size between
   surrounding real prices, indicating gap-filling by interpolation.

---

## Solution Overview

Created `mark_infill_as_nan.py` in the project root.  The script:

1. Reads `Naz100_Symbols_nans.hdf5` (key `'Naz100_Symbols'`).
2. For each symbol computes leading/trailing constant-price run lengths
   by scanning first-differences from each end.
3. Detects mid-history constant runs (abs(d1) < 1e-7 for ≥ 5 days) and
   mid-history linear runs (all abs(d2) < 1e-5 within a consecutive d2
   run ≥ window-2, corresponding to ≥ 5 prices).
4. Skips any symbol in `SKIP_SYMBOLS` (currently just `CASH`).
5. Writes the NaN-modified DataFrame back with the same HDF5 parameters
   as production (complevel=5, complib='blosc').

---

## Key Changes

| File | What changed |
|------|-------------|
| `mark_infill_as_nan.py` | **Created** — NaN-marking script |
| `/Users/donaldpg/pyTAAA_data/Naz100/symbols/Naz100_Symbols_nans.hdf5` | Written with 724,684 NaN cells |

The production file (`Naz100_Symbols_.hdf5`) was **not modified**.

---

## Technical Details

### Detection algorithm

**Leading / trailing runs** — purely iterative, no rolling window.
```python
# Leading:
diffs = np.abs(np.diff(prices))
nonzero = np.where(diffs > CONST_TOL)[0]
lead_length = nonzero[0] + 1  # prices up to first real change
```

**Mid-history constant runs** — run-length scan on abs(d1):
```python
# All consecutive d1 < CONST_TOL of length >= window flagged.
```

**Mid-history linear runs** — run-length scan on abs(d2):
```python
# A d2 run of length k covers k+2 prices.
# ALL d2 values in the run must be < LINEAR_TOL (no bridging).
# Minimum k = max(1, window-2) to ensure >= window prices flagged.
```

The "no bridging" requirement was added after discovering that the
original implementation (which marked both d1[i] and d1[i+1] for each
d2[i] < threshold) could create spurious contiguous runs across real
price reversals.  E.g. AAPL 1993 had a 5-day flag from two isolated
near-zero d2 positions connected through a large reversal; the corrected
algorithm reduced AAPL mid_nan from 49 to 5.

### Constants
| Name | Value | Purpose |
|------|-------|---------|
| `CONST_TOL` | 1e-7 | Max absolute price step to classify as "constant" |
| `LINEAR_TOL` | 1e-5 | Max absolute second-difference to classify as "linear" |
| `DEFAULT_WINDOW` | 5 | Min run length (days) for mid-history detection |

### SKIP_SYMBOLS
`CASH` is excluded from NaN-marking.  Its profiling showed 0 detected
real-data days (the entire series is one constant-price block), suggesting
it was never a genuine Nasdaq 100 constituent.  Leaving it with constant
prices avoids creating a column of all NaN that could break downstream
ranking code.

---

## Testing

### Dry run
```bash
PYTHONPATH=$(pwd) uv run python mark_infill_as_nan.py --dry-run
```
- Confirmed 724,684 NaN cells before writing.
- Spot-checked per-symbol counts against earlier profiling results.
- Verified algorithm fix: AAPL went from 49 mid_nan (old) → 5 (fixed).

### Read-back verification
Read the written file and confirmed:
- Shape still (8855, 215).
- `df.isna().sum().sum()` == 724,684.
- ASML `first_valid_index()` == `1995-03-16` (ASML IPO ≈ 1995).
- Row before ASML first real date is `NaN`.
- `CASH` has 0 NaN rows.

---

## Follow-up Items (after 2026-03-05 session)

- The downstream code (`sharpeWeightedRank_2D`, `equalWeightedRank_2D`)
  currently uses the production file and has no NaN-handling path.
  Future work: wire `Naz100_Symbols_nans.hdf5` into the backtest and
  verify that `computeSignal2D` / ranking functions handle NaN columns
  correctly (see `docs/STOCK_SELECTION_ALGORITHM_COMPARISON.md` §5).
- `CASH` warrants research to confirm it is not a legitimate constituent
  at any point in the history.
- Consider increasing `WINDOW` (e.g. to 10) and re-running dry run to
  assess sensitivity; current 5-day window may still flag legitimate short
  halts.
- `ALGN` shows 299 mid-history NaN days (~5% of its real-data period),
  suggesting multiple brief absences from the index.  Spot-check these
  dates against known index membership data if accuracy is critical.

---

# Copilot Session: Infill Detection Wired into Data Pipeline

**Date:** 2026-03-06
**Branch:** main

---

## Date and Context

Follow-on session to the NaN-marking work above.  The infill-detection
algorithm was extracted into a reusable read-only module, wired into the
production data-loading path to replace the cruder `_build_active_mask_from_raw`
function, and downstream B&H masking was corrected to use explicit boolean
indexing rather than `np.where`.

---

## Problem Statement

1. `_build_active_mask_from_raw` in `data_loaders.py` was a simpler,
   less accurate reimplementation of the same infill-detection logic —
   it only caught trailing runs ≥ 20 days and missed mid-history gaps
   entirely.  It also re-derived what `loadQuotes_fromHDF` had already
   loaded, wasting a processing step.
2. CASH handling was scattered: `detect_infilled_from_df` silently skipped
   CASH via `SKIP_SYMBOLS`, and `data_loaders.py` conditionally appended
   it — creating an implicit dependency between the two.
3. All Buy-and-Hold masking sites used `np.where(active_mask, value, np.nan)`,
   which is functional but misleads readers into thinking it is a
   ternary select rather than a boolean mask operation.
4. A `FileNotFoundError` was triggered because `pytaaa_naz100_pine_nans.json`
   pointed `hdf_store` at a non-existent path.
5. The HDF5 key name (`Naz100_Symbols` / `SP500_Symbols`) was hardcoded
   in `mark_infill_as_nan.py`, making the script fragile when pointed at
   an SP500 file.

---

## Solution Overview

1. Created `functions/detect_infilled.py` — a read-only module that
   returns a boolean DataFrame (True = infilled) without modifying HDF5.
2. Added `detect_infilled_from_df(df)` variant so the already-loaded
   `quote` DataFrame from `loadQuotes_fromHDF` could be reused directly.
3. Replaced `_build_active_mask_from_raw` in `data_loaders.py` with a
   call to `detect_infilled_from_df(quote)`, eliminating the duplicate
   logic and the extra disk read.
4. Unified CASH handling into two mutually exclusive, explicit branches in
   `data_loaders.py`; added a shape assertion to catch mismatches early.
5. Replaced all four `np.where(active_mask, value, np.nan)` calls with
   explicit `value.copy()` + `[~active_mask] = np.nan` bool masking.
6. Added `_derive_listname(path)` to both `detect_infilled.py` and
   `mark_infill_as_nan.py` so the HDF5 key is derived from the filename.
7. Corrected the `hdf_store` path in `pytaaa_naz100_pine_nans.json`.

---

## Key Changes

| File | What changed |
|---|---|
| `functions/detect_infilled.py` | **Created** — read-only infill detection; `detect_infilled()` and `detect_infilled_from_df()`; `_derive_listname()` |
| `functions/data_loaders.py` | Removed `_MIN_CONSTANT_DAYS`, `_build_active_mask_from_raw`; added `detect_infilled_from_df` import; captured `quote` from `loadQuotes_fromHDF`; unified CASH handling; added shape assert; removed dead code block |
| `mark_infill_as_nan.py` | Added `_derive_listname()`; removed hardcoded `LISTNAME = "Naz100_Symbols"` |
| `functions/backtesting/core_backtest.py` | Replaced `np.where` B&H masking with explicit bool copy+set |
| `functions/output_generators.py` | Same np.where fix for `BuyHoldFinalValue` |
| `functions/dailyBacktest.py` | Same np.where fix at two sites (`BuyHoldFinalValue` and `BuyHoldPortfolioValue`) |
| `.../pytaaa_naz100_pine_nans.json` | Fixed `hdf_store` path to actual file location (outside repo) |

---

## Technical Details

### `functions/detect_infilled.py`

New module with the same four-phase detection algorithm as
`mark_infill_as_nan.py` but never modifies any file.

Key functions:

- `_derive_listname(path)` — maps `naz100` → `"Naz100_Symbols"`,
  `sp500` → `"SP500_Symbols"`, else `ValueError`.
- `detect_infilled_from_df(df, window=5)` — accepts an already-loaded
  DataFrame (dates × symbols), returns a same-shape bool DataFrame
  (`True = infilled`).  Prints a summary line.
- `detect_infilled(hdf5_path, window=5)` — reads HDF5, derives key via
  `_derive_listname`, delegates to `detect_infilled_from_df`.

### `SKIP_SYMBOLS` clarification

`SKIP_SYMBOLS = frozenset({"CASH"})` leaves CASH with `False` (not
infilled) throughout the output DataFrame.  When called from
`data_loaders.py` this is an optimisation only — the caller explicitly
forces `active_mask[cash_idx, :] = True` regardless.  For the standalone
`detect_infilled()` / `mark_infill_as_nan.py` path, it is a correctness
requirement: CASH's all-constant series would otherwise be classified as
100% infilled.

### CASH handling consolidated in `data_loaders.py`

Two branches, selected at runtime, never both executing:

```python
if 'CASH' in symbols:
    # Case (a): CASH was in the HDF5 — force its active_mask row True.
    active_mask[symbols.index('CASH'), :] = True
else:
    # Case (b): CASH absent — append to symbols, adjClose, active_mask.
    symbols.append('CASH')
    adjClose = np.vstack([adjClose, np.ones((1, adjClose.shape[1]))])
    active_mask = np.vstack(
        [active_mask, np.ones((1, active_mask.shape[1]), dtype=bool)]
    )
```

A shape assertion follows to guarantee `active_mask.shape == adjClose.shape`
before any downstream consumer can silently mis-broadcast.

### B&H masking fix

All Buy-and-Hold benchmark calculations now read:

```python
_value_masked = value.copy()
_value_masked[~active_mask] = np.nan
BuyHoldPortfolioValue = np.nanmean(_value_masked, axis=0)
```

`np.where(active_mask, value, np.nan)` was removed from all four sites.
The old form is functional but implies a three-way choice; the new form
makes the intent explicit — "null out infilled cells, then average what
remains".

### Array convention (unchanged)

| Array | Shape | `True` means |
|---|---|---|
| `infill_df` (from `detect_infilled_from_df`) | (n_days, n_stocks) | infilled |
| `active_mask` (in pipeline) | (n_stocks, n_days) | real price / in-index |

The inversion and transpose happen once in `data_loaders.py`:
`active_mask = ~infill_df.values.T`

---

## Testing

- Smoke test of `detect_infilled_from_df` on three synthetic symbols
  (leading, trailing, all-constant infill): all assertions passed.
- `detect_infilled_from_df` on the full `Naz100_Symbols_nans.hdf5`:
  717,774 infilled cells detected across 215 symbols (shape 8855 × 215).
- Regression suite: **14 passed**, 1 pre-existing scipy `_highspy`
  failure (unrelated to these changes).

---

## Follow-up Items

- Commit the three modified source files
  (`functions/detect_infilled.py`, `functions/data_loaders.py`,
  `mark_infill_as_nan.py`) — not yet pushed.
- Integration test: run `pytaaa_main.py --json pytaaa_naz100_pine_nans.json`
  to verify the full pipeline works end-to-end with the NaN-masked HDF5.
- `sharpeWeightedRank_2D` and `computeSignal2D` have not yet been audited
  for NaN safety when `active_mask` zeros out signals but cleaned prices
  still produce finite ratios for ex-index stocks.
- `PortfolioStatsOnDate.py` uses the legacy `la.IO(hdf5filename)` path
  and has no `json_fn` / `hdf_store` override — still untouched.
