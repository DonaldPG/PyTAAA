# HMA Rank-Table Mismatch Plan

## Scope
- Target HMA configs:
  - /Users/donaldpg/pyTAAA_data/sp500_hma/pytaaa_sp500_hma.json
  - /Users/donaldpg/pyTAAA_data/naz100_hma/pytaaa_naz100_hma.json
- Do not change rank/weight computation engines.
- Keep behavior unchanged for:
  - /Users/donaldpg/pyTAAA_data/naz100_pine/pytaaa_naz100_pine.json
  - /Users/donaldpg/pyTAAA_data/naz100_pi/pytaaa_naz100_pi.json
  - /Users/donaldpg/pyTAAA_data/sp500_pine/pytaaa_sp500_pine.json

## Diagnosis Summary
- The HTML table section "Current stocks, with ranks, weights, and prices are" is generated in functions/output_generators.py (write_rank_list_html).
- Column source mismatch is the root cause.
- Wt (mo start) is read from monthgainlossweight[:, month_start_idx].
- Rank (start of month) is derived from a month-start scoring path aligned to month-start weights.
- Wt (today) and Rank (today) are recomputed in write_rank_list_html using a Sharpe + daily-signal ranking path.
- For SP500 HMA, stockWeightMethod is equal_weight and upstream allocation comes from UnWeightedRank_2D (delta-rank/equal-weight monthly allocator), not the Sharpe path used by the HTML recompute.
- For Naz100 HMA, upstream weighting is abs_sharpe_weight, but the table still recomputes "today" with a different signal/ranking path than the monthly-held weight matrix used for month-start.
- Result: HMA pages can show mismatched month-start vs today columns because they are produced by different selectors and signal windows.

## Evidence
- SP500 HMA shows month-start weights of 0.333 for three symbols while today weights are 0.111 across a larger set, proving two different selection pipelines are feeding the same table.
- Naz100 HMA also exhibits rank/weight mismatch on its HMA webpage table, indicating this is not only an equal_weight issue.
- Filesystem path case is not the issue (SP500_hma and sp500_hma resolve to the same file on this host).

## Fix Strategy (Scoped)
- Implement a display-only path in write_rank_list_html that is activated for HMA pages:
  - uptrendSignalMethod == HMAs
  - stockList in {SP500, Naz100}
- In this scoped path:
  - Use monthgainlossweight[:, -1] directly for Wt (today).
  - Keep Wt (mo start) from monthgainlossweight[:, month_start_idx].
  - Derive Rank (today) and Rank (start of month) from the same corresponding weight vectors using one shared deterministic ordering rule.
  - If latest date is the first trading day of the month (month_start_idx == last index), force equality of display columns by construction:
    - Rank (start of month) == Rank (today)
    - Wt (mo start) == Wt (today)
- Leave existing display logic unchanged for non-HMA JSONs.

## Deterministic Display Ordering Rule
- Primary key: positive weight first.
- Secondary key: weight descending.
- Tertiary key: symbol ascending (stable tie-break).
- Exclude CASH from ranked stock positions (or always place CASH at the bottom with zero rank), matching current table conventions.

## Validation Plan
- HMA target checks (first trading day):
  - SP500 HMA
  - Naz100 HMA
  - Confirm month_start_idx == last index.
  - Confirm exact equality of Rank (start of month) vs Rank (today).
  - Confirm exact equality of Wt (mo start) vs Wt (today).
- Regression checks for unaffected JSONs:
  - naz100_pine, naz100_pi, sp500_pine.
  - Verify table generation path and outputs remain byte-for-byte unchanged where possible, or semantically unchanged if timestamp text differs.
- Safety check:
  - Run get_errors on modified files.
  - Regenerate rank list HTML once per JSON and spot-check top rows.

## Files To Change (Planned)
- functions/output_generators.py
  - write_rank_list_html: add a small HMA-only display branch to keep table columns internally consistent.

## Non-Goals
- No changes to UnWeightedRank_2D, sharpeWeightedRank_2D, delta_rank_sharpe_weight_2D, or portfolio backtest calculations.
- No changes to trade recommendation generation.
