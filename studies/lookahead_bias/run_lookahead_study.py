"""
Look-Ahead Bias Study — Human Verification Script
==================================================

PURPOSE
-------
Verify that the PyTAAA stock selection pipeline (signal generation +
Sharpe-weighted ranking) does NOT exhibit look-ahead bias.

WHAT IT DOES
------------
For each of the three trading models (naz100_hma / naz100_pine /
naz100_pi) this script:

  1. Loads parameters from the PRODUCTION JSON files (same files used
     in live trading) via get_json_params — no hardcoded params.

  2. Reads the PRODUCTION HDF5 price data for each model (same file
     used in live trading) via loadQuotes_fromHDF / get_symbols_file.
     All cleaning (interpolation, leading/trailing NaN) is applied via
     load_quotes_for_analysis — identical to the production path.

  3. Runs the EXACT SAME signal+rank code path used in production
     (computeSignal2D and sharpeWeightedRank_2D from the codebase).

  4. Creates a **patched** copy of the adjClose array where all prices
     after a fixed CUTOFF DATE (≈200 trading days before the last date)
     are altered dramatically by reversing the prior performance ranking:
       - Top-half performers up to cutoff → stepped *down* 40 %
       - Bottom-half performers up to cutoff → stepped *up*  40 %

  5. Re-runs the identical pipeline on the patched data.

  6. Prints monthly stock selections from BOTH runs side-by-side:
       - Months marked [PRE ] should show IDENTICAL selections.
       - The month containing the cutoff (marked [CUT]) MUST show
         IDENTICAL selections — any difference proves look-ahead bias.
       - Months marked [POST] are expected to diverge as the patched
         prices propagate through the rolling window calculations.

HOW TO VERIFY (human)
---------------------
Scan the printed output for the [CUT] row.  The "ORIG" and "PATCH"
selections on that row MUST be the same set of stocks.  If they
differ, look-ahead bias is present in the highlighted pipeline code.

USAGE
-----
    cd /path/to/PyTAAA
    PYTHONPATH=$(pwd) uv run python \\
        studies/lookahead_bias/run_lookahead_study.py \\
        --json-hma /path/to/naz100_hma/pytaaa_naz100_hma.json \\
        --json-pine /path/to/naz100_pine/pytaaa_naz100_pine.json \\
        --json-pi /path/to/naz100_pi/pytaaa_naz100_pi.json

    # Production defaults (reads /Users/donaldpg/pyTAAA_data/...):
    PYTHONPATH=$(pwd) uv run python \\
        studies/lookahead_bias/run_lookahead_study.py
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from numpy import isnan
from pathlib import Path

# ---------------------------------------------------------------------------
# Insert project root so codebase imports work without installation
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from functions.ta.signal_generation import computeSignal2D
from functions.TAfunctions import sharpeWeightedRank_2D
from functions.GetParams import get_json_params, get_symbols_file
from functions.data_loaders import load_quotes_for_analysis

# ---------------------------------------------------------------------------
# Default production JSON paths
# ---------------------------------------------------------------------------
_DEFAULT_JSON = {
    "naz100_hma": (
        "/Users/donaldpg/pyTAAA_data/naz100_hma/pytaaa_naz100_hma.json"
    ),
    "naz100_pine": (
        "/Users/donaldpg/pyTAAA_data/naz100_pine/pytaaa_naz100_pine.json"
    ),
    "naz100_pi": (
        "/Users/donaldpg/pyTAAA_data/naz100_pi/pytaaa_naz100_pi.json"
    ),
}


# ===========================================================================
# 1.  Load model params from JSON (uses same get_json_params as production)
# ===========================================================================

def _params_from_json(json_fn: str) -> dict:
    """
    Load the Valuation parameters from a production JSON file using the
    same get_json_params() call that PortfolioPerformanceCalcs uses.

    Returns the params dict expected by run_selection_pipeline.
    """
    p = get_json_params(json_fn)

    # get_json_params stores MA2offset but computeSignal2D also needs it
    # under the key 'MA2offset'; it is already set, but make sma2factor
    # consistent with the field name used in computeSignal2D.
    p.setdefault("sma2factor", p.get("MA2factor", 1.0))

    return p


# ===========================================================================
# 2.  Load production price data from the model's HDF5 file
# ===========================================================================

def _load_production_adjclose(
    json_fn: str,
) -> tuple[np.ndarray, list[str], list]:
    """
    Load adjClose, symbols, and datearray from the production HDF5 file.

    Uses the same get_symbols_file() + load_quotes_for_analysis() path
    that PortfolioPerformanceCalcs uses, including data cleaning
    (interpolation, leading/trailing NaN removal).

    Args:
        json_fn: Path to the model's production JSON configuration file.

    Returns (adjClose, symbols, datearray).
    """
    symbols_file = get_symbols_file(json_fn)
    adjClose, symbols, datearray = load_quotes_for_analysis(
        symbols_file, json_fn
    )
    return adjClose, symbols, datearray


# ===========================================================================
# 3.  Patch function — alter post-cutoff prices
# ===========================================================================

def _patch_adjclose(
    adjClose_orig: np.ndarray,
    symbols: list[str],
    cutoff_idx: int,
    step_down_factor: float = 0.60,
    step_up_factor: float = 1.40,
) -> np.ndarray:
    """
    Return a copy of adjClose where prices AFTER cutoff_idx are altered.
    Prices at and before cutoff_idx are byte-identical to the original.

    Stocks are ranked by their cumulative return up to cutoff_idx:
      - Top half (best performers up to cutoff) → × step_down_factor
      - Bottom half (worst performers up to cutoff) → × step_up_factor

    This tests whether a dramatic post-cutoff trend reversal leaks back
    into selections at the cutoff date (which would indicate look-ahead
    bias).  The jump is applied as an instantaneous step at
    (cutoff_idx + 1) so the price AT cutoff_idx itself is unchanged.
    """
    patched = adjClose_orig.copy()

    # Compute cumulative return up to cutoff_idx for each ticker.
    # Use the first valid (nonzero) price as the starting value.
    n_stocks = adjClose_orig.shape[0]
    returns = np.ones(n_stocks, dtype=float)
    for i in range(n_stocks):
        prices = adjClose_orig[i, : cutoff_idx + 1]
        valid = prices[prices > 0]
        if len(valid) >= 2:
            returns[i] = prices[cutoff_idx] / valid[0]

    # Split into top/bottom half and apply opposing scale factors.
    sorted_idx = np.argsort(returns)
    n = len(sorted_idx)
    bottom_half = sorted_idx[: n // 2]   # Worst performers — stepped up
    top_half    = sorted_idx[n // 2 :]   # Best performers  — stepped down

    if cutoff_idx + 1 < adjClose_orig.shape[1]:
        patched[top_half,    cutoff_idx + 1:] *= step_down_factor
        patched[bottom_half, cutoff_idx + 1:] *= step_up_factor

    return patched


# ===========================================================================
# 4.  Core pipeline — replicates highlighted code in dailyBacktest.py
# ===========================================================================

def run_selection_pipeline(
    adjClose: np.ndarray,
    symbols: list[str],
    datearray: list,
    params: dict,
    json_fn: str = "dummy.json",
) -> np.ndarray:
    """
    Run the stock selection pipeline, replicating the highlighted block from
    dailyBacktest.py:

      - Compute gainloss
      - computeSignal2D  → signal2D
      - Forward-fill monthly selections
      - sharpeWeightedRank_2D → monthgainlossweight

    Returns monthgainlossweight: 2D array (n_stocks, n_days).
    """
    n_stocks, n_days = adjClose.shape

    ##########################################################################
    # Gain/loss computation — identical to dailyBacktest.py
    ##########################################################################
    gainloss = np.ones((n_stocks, n_days), dtype=float)
    gainloss[:, 1:] = adjClose[:, 1:] / adjClose[:, :-1]
    gainloss[isnan(gainloss)] = 1.0

    ##########################################################################
    # Signal generation — same call as in dailyBacktest.py
    ##########################################################################
    if params["uptrendSignalMethod"] == "percentileChannels":
        signal2D, _, _ = computeSignal2D(adjClose, gainloss, params)
    else:
        signal2D = computeSignal2D(adjClose, gainloss, params)

    ##########################################################################
    # Monthly hold logic — identical to dailyBacktest.py
    ##########################################################################
    signal2D_daily = signal2D.copy()
    monthsToHold = params["monthsToHold"]

    last_month_signals = signal2D_daily[:, 0].copy()
    for jj in range(1, n_days):
        is_rebalance = (
            (datearray[jj].month != datearray[jj - 1].month) and
            ((datearray[jj].month - 1) % monthsToHold == 0)
        )
        if is_rebalance:
            last_month_signals = signal2D_daily[:, jj].copy()
        signal2D[:, jj] = last_month_signals

    ##########################################################################
    # Sharpe-weighted rank — identical call signature to dailyBacktest.py
    ##########################################################################
    monthgainlossweight = sharpeWeightedRank_2D(
        json_fn,
        datearray,
        symbols,
        adjClose,
        signal2D,
        signal2D_daily,
        params["LongPeriod"],
        params["numberStocksTraded"],
        params["riskDownside_min"],
        params["riskDownside_max"],
        params["rankThresholdPct"],
        stddevThreshold=params["stddevThreshold"],
        stockList=params.get("stockList", "Naz100"),
    )

    return monthgainlossweight


# ===========================================================================
# 5.  Selection extraction — which stocks have nonzero weight at a date?
# ===========================================================================

def _selected_stocks(
    symbols: list[str],
    weights: np.ndarray,
) -> list[tuple[str, float]]:
    """Return list of (symbol, weight) for all stocks with weight > 0."""
    return [
        (symbols[i], float(weights[i]))
        for i in range(len(symbols))
        if weights[i] > 1e-6
    ]


# ===========================================================================
# 6.  Human-readable monthly comparison printer
# ===========================================================================

def print_monthly_comparison(
    model_name: str,
    datearray: list,
    symbols: list[str],
    weights_orig: np.ndarray,
    weights_patch: np.ndarray,
    cutoff_date,
    months_post_cutoff: int = 3,
    months_pre_cutoff: int = 3,
) -> None:
    """
    Print a monthly comparison of original vs patched selections.

    Marks each row as [PRE ], [CUT], or [POST] relative to the cutoff
    month.  Only the last months_pre_cutoff [PRE] months and the first
    months_post_cutoff [POST] months are shown.

    Human verification rule
    -----------------------
    [PRE ] rows   : ORIG == PATCH  (expected — prices unchanged)
    [CUT ] row    : ORIG == PATCH  (MUST match — proves no look-ahead bias)
    [POST] rows   : ORIG may differ from PATCH (patched prices diverge)
    """
    print()
    print("=" * 90)
    print(f"  MODEL: {model_name}")
    print("=" * 90)
    print(
        f"  Cutoff date: {cutoff_date}  "
        f"(prices ALTERED starting the NEXT trading day)"
    )
    print(
        "  [CUT] row MUST show identical ORIG/PATCH selections "
        "→ proves no look-ahead bias"
    )
    print("-" * 90)
    print(f"  {'TAG':<6}  {'DATE':<12}  ORIG  (PATCH on next line)")
    print("-" * 90)

    # Normalise cutoff_date to a plain datetime.date for consistent access.
    import datetime as _dt
    if hasattr(cutoff_date, 'date') and callable(cutoff_date.date):
        cutoff_date = cutoff_date.date()
    cutoff_month = (cutoff_date.year, cutoff_date.month)

    # --- Collect all displayable rows, then print the windowed subset ---
    # Each entry: (tag, date_str, orig_str, patch_str, diff_marker)
    rows: list[tuple[str, str, str, str, str]] = []
    post_cutoff_count = 0

    for jj in range(1, len(datearray)):
        prev_date = datearray[jj - 1]
        curr_date = datearray[jj]

        # First trading day of a new month only.
        if curr_date.month == prev_date.month:
            continue

        curr_month = (curr_date.year, curr_date.month)

        if curr_month < cutoff_month:
            tag = "[PRE ]"
        elif curr_month == cutoff_month:
            tag = "[CUT ]"
        else:
            post_cutoff_count += 1
            if post_cutoff_count > months_post_cutoff:
                break
            tag = "[POST]"

        orig_sel  = _selected_stocks(symbols, weights_orig[:, jj])
        patch_sel = _selected_stocks(symbols, weights_patch[:, jj])

        orig_str  = ", ".join(
            f"{s}({w:.0%})" for s, w in orig_sel
        ) if orig_sel else "(none)"
        patch_str = ", ".join(
            f"{s}({w:.0%})" for s, w in patch_sel
        ) if patch_sel else "(none)"

        orig_names  = {s for s, _ in orig_sel}
        patch_names = {s for s, _ in patch_sel}
        diff_marker = "  *** DIFFER ***" if orig_names != patch_names else ""

        date_str = (
            str(curr_date.date())
            if hasattr(curr_date, 'date') and callable(curr_date.date)
            else str(curr_date)
        )
        rows.append((tag, date_str, orig_str, patch_str, diff_marker))

    # Print only the last months_pre_cutoff [PRE] rows, the [CUT] row,
    # and all collected [POST] rows.
    pre_rows  = [r for r in rows if r[0] == "[PRE ]"][-months_pre_cutoff:]
    cut_rows  = [r for r in rows if r[0] == "[CUT ]"]
    post_rows = [r for r in rows if r[0] == "[POST]"]

    indent = " " * 26   # Aligns PATCH line under the stock list
    for tag, date_str, orig_str, patch_str, diff_marker in (
        pre_rows + cut_rows + post_rows
    ):
        if tag == "[CUT ]":
            print("-" * 90)
        print(f"  {tag:<6}  {date_str:<12}  {orig_str}")
        print(f"{indent}  {patch_str}{diff_marker}")
        print()
        if tag == "[CUT ]":
            print("-" * 90)

    print("=" * 90)


# ===========================================================================
# 7.  Cutoff index resolver — map a calendar date to a datearray index
# ===========================================================================

def _find_cutoff_idx(datearray: list, target_date_str: str) -> int:
    """
    Return the last index in datearray whose date is ≤ target_date_str.

    target_date_str must be an ISO-format date string (YYYY-MM-DD).
    The datearray must be sorted in ascending chronological order.
    """
    import datetime as _dt
    target = _dt.date.fromisoformat(target_date_str)
    best_idx = 0
    for i, d in enumerate(datearray):
        d_norm = d.date() if (hasattr(d, 'date') and callable(d.date)) else d
        if d_norm <= target:
            best_idx = i
        else:
            break
    return best_idx


# ===========================================================================
# 8.  Main driver
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Look-ahead bias study using production JSON parameters."
    )
    p.add_argument(
        "--json-hma",
        default=_DEFAULT_JSON["naz100_hma"],
        metavar="PATH",
        help="Path to naz100_hma production JSON (default: %(default)s)",
    )
    p.add_argument(
        "--json-pine",
        default=_DEFAULT_JSON["naz100_pine"],
        metavar="PATH",
        help="Path to naz100_pine production JSON (default: %(default)s)",
    )
    p.add_argument(
        "--json-pi",
        default=_DEFAULT_JSON["naz100_pi"],
        metavar="PATH",
        help="Path to naz100_pi production JSON (default: %(default)s)",
    )
    p.add_argument(
        "--cutoff-date",
        nargs="+",
        metavar="YYYY-MM-DD",
        default=None,
        help=(
            "One or more ISO date strings (e.g. 2025-06-01 2025-09-01). "
            "For each date the study slices ±600/200 days around the cutoff "
            "and runs the pipeline independently. "
            "Default: 200 trading days from end of dataset."
        ),
    )
    p.add_argument(
        "--months-pre",
        type=int,
        default=3,
        help="Number of PRE-cutoff months to print (default: 3)",
    )
    p.add_argument(
        "--months-post",
        type=int,
        default=3,
        help="Number of POST-cutoff months to print (default: 3)",
    )
    return p.parse_args()


def main() -> None:
    """Run the look-ahead bias study for all three models."""

    args = _parse_args()

    json_map = {
        "naz100_hma":  args.json_hma,
        "naz100_pine": args.json_pine,
        "naz100_pi":   args.json_pi,
    }

    print()
    print("=" * 90)
    print("  LOOK-AHEAD BIAS STUDY — Human Verification")
    print("=" * 90)
    print()
    print("  Parameters and price data loaded from PRODUCTION files:")
    for model, path in json_map.items():
        print(f"    {model}: {path}")
    print()
    print("  Pipeline: computeSignal2D + monthly hold + sharpeWeightedRank_2D")
    print("  (exact same functions as the highlighted section in dailyBacktest.py)")
    print()

    ##########################################################################
    # Load production parameters and HDF5 price data for each model.
    # Each model is run independently so their different symbol universes
    # and HDF5 files are handled correctly.
    ##########################################################################
    model_entries = {}
    for model_name, json_fn in json_map.items():
        if not os.path.exists(json_fn):
            print(f"  WARNING: JSON not found for {model_name}: {json_fn}")
            print("           Skipping this model.")
            continue
        params = _params_from_json(json_fn)
        print(
            f"  {model_name}: uptrendSignalMethod="
            f"{params['uptrendSignalMethod']}, "
            f"LongPeriod={params['LongPeriod']}, MA1={params['MA1']}, "
            f"MA2={params['MA2']}, MA3={params['MA3']}"
        )
        print(f"    Loading production HDF5 data for {model_name} ...")
        try:
            adjClose_orig, symbols, datearray = _load_production_adjclose(
                json_fn
            )
        except Exception as exc:
            print(f"  WARNING: Could not load HDF5 for {model_name}: {exc}")
            print("           Skipping this model.")
            continue
        model_entries[model_name] = (params, json_fn, adjClose_orig,
                                     symbols, datearray)
        print(
            f"    Loaded {len(symbols)} symbols, "
            f"{adjClose_orig.shape[1]} trading days."
        )

    if not model_entries:
        print("  ERROR: No valid models could be loaded. Exiting.")
        return

    print()

    ##########################################################################
    # Determine cutoff dates to test.  Each cutoff is run independently
    # for every model using a narrow slice of the full dataset — much
    # faster than running all 8 000+ days.  600 pre-cutoff trading days
    # provide enough history for all rolling windows; 200 post-cutoff
    # days give ≈ 8 calendar months of divergence to inspect.
    ##########################################################################
    raw_cutoff_dates = args.cutoff_date   # list[str] | None
    PRE_DAYS  = 600   # Trading days of context before the cutoff
    POST_DAYS = 200   # Trading days of context after the cutoff

    for model_name, (params, json_fn,
                     adjClose_orig, symbols, datearray) in model_entries.items():

        n_days = adjClose_orig.shape[1]

        # Resolve each requested cutoff to an index in this model's
        # datearray.  Default to 200 trading days from end if the user
        # did not supply --cutoff-date.
        if raw_cutoff_dates:
            cutoff_indices = [
                _find_cutoff_idx(datearray, ds) for ds in raw_cutoff_dates
            ]
        else:
            cutoff_indices = [n_days - 200]

        for cutoff_idx in cutoff_indices:
            cutoff_date = datearray[cutoff_idx]

            # Slice dataset to [cutoff-PRE_DAYS, cutoff+POST_DAYS].
            start_idx    = max(0, cutoff_idx - PRE_DAYS)
            end_idx      = min(n_days, cutoff_idx + POST_DAYS + 1)
            adjClose_sl  = adjClose_orig[:, start_idx:end_idx]
            datearray_sl = datearray[start_idx:end_idx]
            cut_sl       = cutoff_idx - start_idx  # Cutoff index in slice

            print(
                f"  Running pipeline for {model_name} "
                f"(uptrendSignalMethod={params['uptrendSignalMethod']}) ..."
            )
            print(f"  Full date range : {datearray[0]} → {datearray[-1]}")
            print(
                f"  Slice range     : {datearray_sl[0]} → "
                f"{datearray_sl[-1]}  ({len(datearray_sl)} days)"
            )
            print(
                f"  Cutoff date     : {cutoff_date}  "
                f"(slice index {cut_sl} of {len(datearray_sl)})"
            )
            print()

            ##################################################################
            # Build patched adjClose (prices after cut_sl are altered)
            ##################################################################
            adjClose_patch_sl = _patch_adjclose(
                adjClose_sl,
                symbols,
                cutoff_idx=cut_sl,
                step_down_factor=0.60,
                step_up_factor=1.40,
            )

            # Sanity check: pre-cutoff prices must be byte-identical.
            assert np.array_equal(
                adjClose_sl[:, : cut_sl + 1],
                adjClose_patch_sl[:, : cut_sl + 1],
            ), "BUG: pre-cutoff prices changed by patch function"

            # Original slice
            weights_orig = run_selection_pipeline(
                adjClose_sl, symbols, datearray_sl, params, json_fn
            )

            # Patched slice (same params, same json_fn, different adjClose)
            weights_patch = run_selection_pipeline(
                adjClose_patch_sl, symbols, datearray_sl, params, json_fn
            )

            # Human-readable comparison
            print_monthly_comparison(
                model_name=model_name,
                datearray=datearray_sl,
                symbols=symbols,
                weights_orig=weights_orig,
                weights_patch=weights_patch,
                cutoff_date=cutoff_date,
                months_post_cutoff=args.months_post,
                months_pre_cutoff=args.months_pre,
            )

    print()
    print("  Study complete.")
    print()
    print("  HOW TO READ THE OUTPUT:")
    print("  [PRE ] — prices identical in both runs → selections MUST match")
    print(
        "  [CUT ] — cutoff month; prices at cutoff are identical → "
        "selections MUST match"
    )
    print(
        "  [POST] — post-cutoff prices differ; selections MAY diverge "
        "(expected behaviour)"
    )
    print(
        "  *** DIFFER *** marker on [CUT] row → look-ahead bias detected!"
    )
    print()


if __name__ == "__main__":
    main()
