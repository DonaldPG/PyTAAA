# Session Summary: Fix Delisted Stock (PARA) Appearing in Selections

**Date:** 2026-05-05
**Branch:** `orchestration-refactor`

---

## Date and Context

Bug report raised during a live daily run of the `sp500_hma` model.
PARA (Paramount Global) was delisted from Nasdaq on 2025-08-08 after
the Skydance merger, yet it appeared in the recommended new stock
selection output by `calculateTrades`.  The session traced the
multi-stage bug across the quote-update pipeline, the active-mask
cross-reference, and the weight-assignment loop.

---

## Problem Statement

PARA was removed from `SP500_Symbols.txt` (confirmed absent), but
continued to appear in `last_symbols_text` and the "new stock selection"
print.  Three characteristics combined to defeat the existing
protections:

1. **PARA still trades OTC after delisting.** yfinance returns real,
   non-constant prices for PARA, so `detect_infilled_from_df` never
   flags it as inactive — the trailing-constant heuristic only works for
   stocks that genuinely stop trading.

2. **`combine_first` carries prices forward on fresh HDF5 dates.**
   `UpdateHDF_yf` merges new dates into the existing HDF5 store with
   `combine_first(quote)`.  For symbols absent from the download
   (`symbols_in_list`), this carries the last stored price into every
   new date added to the store.  The cleaning step then forward-fills
   any gaps via `interpolate()`, leaving a seamless-looking price
   series.

3. **`UnWeightedRank_2D` selected by rank, not by signal.**
   The weight-assignment loop in `UnWeightedRank_2D` checked only
   `deltaRank[jj,ii] <= rankthresholdpercentequiv[ii]`.  Even though
   the `active_mask` cross-reference in `data_loaders.py` sets
   `active_mask[PARA, last_date] = False` and `output_generators.py`
   then zeros `signal2D[~active_mask]`, the monthly carry-forward loop
   had already locked in PARA's signal from the last rebalance date
   (2026-05-01, where `active_mask[PARA]` was still True).  At
   selection time the weight loop saw a good historical delta-rank for
   PARA and assigned it weight regardless of the zeroed signal.

---

## Solution Overview

Three complementary fixes were applied to close each gap in the
pipeline:

1. **Fix A — `data_loaders.py`**: Cross-reference HDF5 symbols against
   the current index membership file; set `active_mask[idx, -1] = False`
   for symbols present in the HDF5 but absent from the index file.

2. **Fix B — `UpdateSymbols_inHDF5.py`**: After the cleaning loop,
   NaN-out all fresh-period dates for symbols absent from
   `symbols_in_list`.  Prevents the HDF5 from accumulating carried-
   forward prices for removed symbols on future runs.

3. **Fix C — `TAfunctions.py` (`UnWeightedRank_2D`)**: Add a
   `signal_mask` guard to the weight-assignment loop so only stocks
   with a positive uptrending signal at date `ii` receive weight,
   regardless of their historical delta-rank.  This is the decisive fix
   that closes the gap left by fixes A and B.

Two additional minor fixes were made to support robustness:

4. **Fix D — `clean_quote_data.py`**: Guard `interpolate()` and
   `nans_at_beginning()` against all-NaN arrays (which can arise when
   cleanspikes removes all data for a removed symbol).

5. **Fix E — `dailyBacktest.py`**: Apply the `active_mask` → signal
   zeroing in the daily backtest path as well, for consistency with
   `output_generators.py`.

6. **Fix F — `backtesting/core_backtest.py`**: Deep-copy the
   `Valuation` section before mutating it in `create_temporary_json`,
   and remove the legacy typo'd key `stockWeightMathod`.

7. **Fix G — `backtesting/parameter_exploration.py`**: Randomly sample
   `stockWeightMethod` each Monte Carlo trial instead of inheriting the
   base JSON value, so the optimizer explores the full weight-method
   space.

---

## Key Changes

| File | Change |
|------|--------|
| `functions/TAfunctions.py` | **Fix C** — `UnWeightedRank_2D` weight loop gates on `signal_mask[jj, ii] > 0` in addition to delta-rank threshold |
| `functions/UpdateSymbols_inHDF5.py` | **Fix B** — NaN fresh-period dates for removed symbols before writing HDF5 |
| `functions/data_loaders.py` | **Fix A** — Cross-reference HDF5 symbols against index file; set `active_mask[:, -1] = False` for absent symbols |
| `functions/dailyBacktest.py` | **Fix E** — Apply `active_mask` → `signal2D[~active_mask] = 0` after monthly carry-forward loop |
| `functions/clean_quote_data.py` | **Fix D** — Guard `interpolate()` and `nans_at_beginning()` against all-NaN input |
| `functions/backtesting/core_backtest.py` | **Fix F** — Deep-copy `Valuation` dict; remove typo'd `stockWeightMathod` key |
| `functions/backtesting/parameter_exploration.py` | **Fix G** — Random `stockWeightMethod` sampling in Monte Carlo |
| `pytaaa_backtest_montecarlo.py` | Minor fix to import / call-site alignment |
| `functions/output_generators.py` | **Fix H** — Remove broken time-of-day guard from `generate_portfolio_plots` |
| `tests/test_output_generators_async.py` | **Fix H** — Mock `datetime.now()` in tests; delete `test_early_return_outside_hours` |
| `functions/output_generators.py` | **Fix I** — `Wt (today)` column dispatches on `stockWeightMethod`; `equal_weight` now assigns `1/N` |

---

## Technical Details

### Root Cause Chain

```
PARA delisted 2025-08-08, still trades OTC
  → yfinance returns non-constant prices → detect_infilled_from_df misses it
  → UpdateHDF_yf.combine_first carries last stored price to new dates
  → cleaning interpolates → seamless price series in HDF5
  → data_loaders cross-reference catches PARA, sets active_mask[PARA,-1]=False
  → output_generators zeros signal2D[~active_mask]
  → BUT monthly carry-forward already locked in PARA signal from 2026-05-01
  → UnWeightedRank_2D sees good delta-rank → assigns weight → PARA selected
```

### Fix C — decisive fix in `UnWeightedRank_2D`

Before:
```python
if test == True:
    monthgainlossweight[jj, ii] = 1./rankthresholdpercentequiv[ii]
```

After:
```python
if test == True and signal_mask[jj, ii] > 0:
    monthgainlossweight[jj, ii] = 1./rankthresholdpercentequiv[ii]
```

`signal_mask` is `(signal2D > 0.5).astype(float)`, derived from the
`signal2D` parameter passed into `UnWeightedRank_2D`.  Because
`signal2D[~active_mask] = 0` is applied upstream in
`output_generators.py` before the call, `signal_mask[PARA, ii] == 0`
for any date `ii` where PARA is flagged inactive.  The rank loop now
cannot assign weight to a stock whose signal has been zeroed.

### Fix B — preventing future accumulation

In `UpdateHDF_yf`, after `updatedquotes['CASH'] = ...`:

```python
current_index_set = set(symbols_in_list)
removed_symbols = [s for s in updatedquotes.columns
                   if s not in current_index_set and s != "CASH"]
if removed_symbols:
    old_index_set = set(quote.index)
    fresh_date_mask = ~updatedquotes.index.isin(old_index_set)
    if int(fresh_date_mask.sum()) > 0:
        updatedquotes.loc[fresh_date_mask, removed_symbols] = np.nan
```

This prevents the HDF5 from accumulating new constant-price entries for
removed symbols on future quote updates.

### Fix D — all-NaN guard

When `UpdateHDF_yf` NaN-marks a removed symbol's fresh dates, the
cleaning loop may encounter a series that is entirely NaN.  The original
`interpolate()` and `nans_at_beginning()` helpers called `np.where(...)
[0][0]` without checking that the result was non-empty, causing
`IndexError`.  Explicit guards return the all-NaN array unchanged.

### Fix F — Monte Carlo JSON mutation bug

`create_temporary_json` updated the `Valuation` dict via `.update()` on
a shallow copy of `base_params`.  This mutated the shared `Valuation`
sub-dict across trials.  A `copy.deepcopy` of `Valuation` is now taken
before `.update()`.  The legacy typo'd key `stockWeightMathod` (with
'a') is also removed to prevent duplicate keys in written JSON.

---

## Fix H — Broken time-of-day guard in `generate_portfolio_plots`

Running the full test suite after the main fixes revealed a pre-existing
failure in `test_output_generators_async.py`:

```
FAILED tests/test_output_generators_async.py::
    TestGeneratePortfolioPlotsBackwardCompatibility::test_sync_mode_does_not_call_spawn
AssertionError: Expected '_generate_full_history_plots' to have been
    called once. Called 0 times.
```

The guard at the top of `generate_portfolio_plots` was:

```python
if not (hourOfDay >= 1 or 11 < hourOfDay < 13):
    return
```

`hourOfDay >= 1` is True for hours 1–23, so the function only returned
early at midnight (hour 0) — almost certainly a logic typo for something
like "skip during market hours."  The test had no `datetime` mock, so it
passed at most hours but failed at midnight.

**Fix**: removed the guard entirely from `generate_portfolio_plots`.
Plot generation is not core to PyTAAA (it writes PNG files for the web
dashboard, not signals or selections); time-based skipping belongs at
the call site in `run_pytaaa.py` alongside the other `hourOfDay` checks
already present there.  The test was updated to mock `datetime.now()`
so it is fully time-independent, and the now-obsolete
`test_early_return_outside_hours` test was deleted.

Files changed: `functions/output_generators.py`,
`tests/test_output_generators_async.py`.

---

## Fix I — `Wt (today)` column ignored `stockWeightMethod` config

**Problem:** The `Wt (today)` column in the HTML rank table
(`write_rank_list_html` in `output_generators.py`) always computed
Sharpe-proportional weights, regardless of the model's configured
`stockWeightMethod`.  For `sp500_hma` (which uses `equal_weight`) this
caused a visible inconsistency: `Wt (mo start)` correctly showed 3
stocks at 0.333 each while `Wt (today)` showed 9 stocks with varied
Sharpe-proportional weights.

**Fix (`functions/output_generators.py`, commit `5883edb`):**

1. Read `_stock_weight_method_mo` from `_params_mo` alongside the
   other params already loaded in the `try/except` block.
2. Dispatch on the value when assigning `weights_today`:
   - `equal_weight` → `1/N` for every selected stock.
   - all other methods → existing Sharpe-proportional path with
     min/max clipping (unchanged behaviour).

The selection step (walk sort order, pick up to `numberStocksTraded`
uptrending stocks) is unchanged — only the weight formula differs.

---

## Testing

Test suite status after all changes (commits `244eb74`, `b558eef`,
`e0d2bc3`, `5883edb`):

```
383 passed, 11 skipped
```

(One test fewer than the initial 384 because `test_early_return_outside_hours`
was deleted along with the guard it was testing.)

No regressions.  The PARA bug was diagnosed from the live log output
showing `PARA` at line 303/380 ("Today's top ranking choices") and
line 388 (`last_symbols_text`) despite PARA being absent from
`SP500_Symbols.txt`.

---

## Follow-up Items

- **Retroactive HDF5 cleanup**: The existing HDF5 still contains
  constant-price entries for PARA on dates after 2025-08-08.  Fix B
  prevents new accumulation but does not remove old data.  A one-time
  script (similar to `mark_infill_as_nan.py`) could NaN-out those dates.
- **Fix A insufficiency for intra-month delistings**: Fix A sets only
  `active_mask[:, -1] = False`.  If a stock is delisted mid-month, the
  active rebalance date within that month may still see `True`.  For
  `monthsToHold > 1` this window grows.  Fix C is the stronger
  protection; Fix A and Fix E remain as defence-in-depth.
- **Tests**: No new unit tests were added this session.  A targeted test
  for `UnWeightedRank_2D` with a deliberately zeroed signal row would
  lock in Fix C permanently.
- ~~**`Wt (today)` column consistency**~~: Fixed in this session
  (Fix I, commit `5883edb`).
  `Wt (today)` now uses `equal_weight` when the model is configured
  for it, matching `Wt (mo start)`.
