# HMA Ranking Table Ordering Fix Plan

## Date and Context

- Date: 2026-08-26
- Area reviewed: HMA ranking logic used for webpage tables and the rank ordering shown in the HTML output
- Relevant config examples:
  - /Users/donaldpg/pytaaa_data/sp500_hma/pytaaa_sp500_hma.json
  - /Users/donaldpg/pyTAAA_data/naz100_hma/pytaaa_naz100_hma.json
- Relevant implementation: [functions/output_generators.py](../functions/output_generators.py)

## Problem Statement

The HMA-driven ranking table appears to be sorted by rank for the top portion of the list, but rows beyond roughly rank 10 also appear alphabetically ordered. The likely issue is isolated to the `HMAs` branch in [functions/output_generators.py](../functions/output_generators.py): the current helper sorts by a weight-based tuple that includes the ticker symbol as a final fallback.

This is a focused HMA-only fix. The goal is to correct ordering in the HMA path without changing the behavior of other `uptrendSignalMethod` modes.

## Review Summary

The HMA-specific logic in [functions/output_generators.py](../functions/output_generators.py) contains a special branch:

- `_is_hma_table_mode` is set when:
  - `uptrendSignalMethod == "HMAs"`
  - `stockList` is `SP500` or `Naz100`

When that mode is enabled, the code does this:

- takes `weights_today = last_weights.copy()`
- takes `weights_month_start = monthgainlossweight[:, _month_start_idx]`
- then calls `_rank_from_weights(weights_today)` and `_rank_from_weights(weights_month_start)`
- then renders the table using the resulting order

The likely root cause is the sort key in `_rank_from_weights()`:

- `(float(weight_vec[ji]) <= 0.0, -float(weight_vec[ji]), symbols[ji].strip())`

This means that if multiple stocks are tied or have zero weight, the ordering becomes alphabetical by ticker instead of preserving a more meaningful HMA ranking order.

This issue is confined to the HMA path; the non-HMA logic remains untouched.

## Focused Fix Strategy

### 1. Scope guard

Do not change any logic outside the HMA branch.

The following must remain exactly as-is:

- `SMAs`
- `minmaxChannels`
- `percentileChannels`
- any other `uptrendSignalMethod` value

### 2. Replace the HMA ordering fallback

Within the HMA branch only, change the ordering to avoid using ticker name as the default fallback when there is no meaningful weight distinction.

Required behavior for ties:

- primary: descending weight
- secondary: Sharpe ratio computed over the MA1-day analysis window
- tertiary: a stable prior-order / score ordering derived from the HMA ranking signal
- ticker name must not be used as the standard tie-breaker for ranking/sorting decisions

This is the deterministic non-alphabetic fallback requested for ties: when multiple stocks have equal ranking value, the sort should resolve by the MA1-window Sharpe ratio. If the Sharpe tie persists, use the stable HMA ordering or original index order as the final deterministic fallback.

This keeps the sort deterministic while preserving the intent of the HMA approach.

### 3. Keep the weighting logic intact

Do not alter how the HMA weights are computed. The fix should be limited to the ordering of the displayed table and exported rank list.

The intent is:

- preserve the exact HMA weight vector as currently computed
- fix the sort/order that is applied to those weights in the table output

### 4. Add a regression test for the HMA branch only

Add a compact test focused on `uptrendSignalMethod == "HMAs"` that asserts:

- a zero-weight or tied-weight case does not sort alphabetically by default
- the HMA ordering remains stable and deterministic
- the non-HMA path is unaffected

This test should not broaden into generic ranking tests for other methods.

### 5. Add a stand-alone validation backtest

Before and after the HMA-only ordering fix, run a stand-alone backtest for both HMA universes so the change can be validated against the actual trading logic and not only the generated table order.

Required validation runs:

- Naz100 HMA config: `/Users/donaldpg/pyTAAA_data/naz100_hma/pytaaa_naz100_hma.json`
- SP500 HMA config: `/Users/donaldpg/pytaaa_data/sp500_hma/pytaaa_sp500_hma.json`

Validation goals:

- confirm the HMA branch remains profitable or unchanged in aggregate performance
- confirm the ordering fix does not reduce the quality of the HMA portfolio selection
- confirm the fix preserves or improves backtest outcomes for both universes

This validation is intentionally stand-alone and does not change the non-HMA code path.

### 6. Document the branch-local fix

Add a short code comment near the HMA branch in [functions/output_generators.py](../functions/output_generators.py) explaining:

- HMA tables are ordered by the HMA ranking logic
- ticker-name sorting is not a primary ranking signal
- non-HMA methods are intentionally not modified

## Patch Sketch

This is the intended branch-local change in [functions/output_generators.py](../functions/output_generators.py):

```python
_is_hma_table_mode = (
    str(_params_mo.get("uptrendSignalMethod", "")) == "HMAs"
    and str(_params_mo.get("stockList", "")) in ("SP500", "Naz100")
)

if _is_hma_table_mode:
    # HMA tables must reflect the exact backtest-held weights so
    # month-start and today columns are internally consistent.
    weights_today = last_weights.copy()
    rank_today, _today_sort_order = _rank_from_weights(weights_today)
    rank_month_start, sort_order = _rank_from_weights(weights_month_start)

    # On the first trading day, month start is today by definition.
    if _month_start_idx == (_n_days_all - 1):
        weights_month_start = weights_today.copy()
        rank_month_start = rank_today.copy()
        sort_order = _today_sort_order
```

Replace the current helper logic with a branch-local ordering rule such as:

```python
def _rank_from_weights(weight_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic HMA ranks from a weight vector.

    Ties in the HMA ranking value are resolved by MA1-window Sharpe, not by
    ticker name. This preserves a deterministic, non-alphabetic ordering.
    """
    _eligible = [
        ji for ji in range(_n_stocks_mo)
        if _in_index_mask[ji] and (not _cash_mask[ji])
    ]

    _ma1_days = int(_params_mo.get("MA1", max(20, _long_period_mo)))
    _ma1_sharpe = _compute_sharpe_for_days(weight_vec, _ma1_days)

    _sorted = sorted(
        _eligible,
        key=lambda ji: (
            float(weight_vec[ji]) <= 0.0,
            -float(weight_vec[ji]),
            -_ma1_sharpe[ji],
            _stable_order.get(ji, 0),
        ),
    )
```

Where `_stable_order` is a branch-local mapping derived from the HMA signal or prior ordering, not a global alphabetical fallback. In the HMA branch, the tie-breaker should be built from the MA1-day Sharpe value for each stock, so equal weights sort by the stronger MA1 Sharpe rather than ticker symbol.

## Scope Guard

This patch is intentionally limited to the HMA-only branch.

- `SMAs` remains unchanged
- `minmaxChannels` remains unchanged
- `percentileChannels` remains unchanged
- `HMAs` is the only case modified

## Implementation Sequence

1. Add or adjust the HMA-only ordering helper in [functions/output_generators.py](../functions/output_generators.py)
2. Keep non-HMA code untouched
3. Add a targeted regression test for the HMA branch
4. Run a stand-alone HMA backtest for Naz100 and SP500 using their respective JSON configs
5. Compare baseline vs. patched metrics for the HMA universes to confirm no material performance regression
6. Re-run the relevant rank-table generation or unit test to ensure the HMA order no longer falls back to alphabetical sorting for zero or tied weights

## Validation Backtest Requirements

The backtest validation should be a separate, explicit step from the HTML table check.

Recommended check:

- run the existing backtest entry point against each HMA config before patching
- apply the HMA-only ordering fix
- rerun the same backtests against the same configs
- compare key metrics such as portfolio value growth, drawdown, Sharpe, and total return for both universes

Success criteria:

- the HMA tables sort consistently without alphabetic fallback
- equal-weight or equal-rank ties are resolved by MA1-window Sharpe, not ticker name
- the HMA trading logic retains or improves performance for both the Naz100 and SP500 universes
- no effect on other `uptrendSignalMethod` modes

## Final Decision

This fix will be implemented only in the HMA branch and only for the table/export ordering used by the HMA webpage output. No non-HMA methods will be changed, and the patch will be validated with stand-alone HMA backtests for both Naz100 and SP500 universes.
