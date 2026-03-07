"""
mark_infill_as_nan.py — Mark infilled price data as NaN in HDF5.

Reads the Naz100 HDF5 copy and replaces price data that was infilled
(constant-price or linearly-interpolated) with NaN, marking dates when
a stock was not part of the Nasdaq 100 index.

Infill patterns detected:
  - Leading constant run  : prices from 1991-01-02 that never change
    (stock not yet added to index → first real price repeated back)
  - Trailing constant run : prices at the end that never change
    (stock removed from index → last real price repeated forward)
  - Mid-history constant  : runs of WINDOW_DAYS+ consecutive identical
    prices within the real-data section (temporary exclusion gaps)
  - Mid-history linear    : runs of WINDOW_DAYS+ consecutive days with
    constant price step (linear interpolation over a gap)

Detection uses a minimum window of WINDOW_DAYS (default 5) to avoid
false positives for stocks that legitimately held flat for 1–4 days.

Usage:
    uv run python mark_infill_as_nan.py [--dry-run] [--window 5]

Output:
    Overwrites Naz100_Symbols_nans.hdf5 with NaN-marked prices and
    prints a summary of changes per symbol.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from functions.logger_config import get_logger

###########################################################################
# Configuration
###########################################################################

HDF5_FILE = Path(
    "/Users/donaldpg/pyTAAA_data/Naz100/symbols/Naz100_Symbols_nans.hdf5"
)
# Minimum consecutive days to classify a mid-history run as infill.
DEFAULT_WINDOW = 5

# Symbols to leave untouched (no NaN marking applied).
# CASH has 0 detected real-data days and is likely not a genuine Nasdaq
# 100 constituent — leave its constant-price data in place rather than
# converting the entire series to NaN.
SKIP_SYMBOLS: frozenset[str] = frozenset({"CASH"})

# Tolerance for detecting zero price movement.
CONST_TOL = 1e-7

# Tolerance for detecting zero change in price step (linear infill).
LINEAR_TOL = 1e-5

logger = get_logger(__name__, log_file="mark_infill_as_nan.log")


###########################################################################
# Detection helpers
###########################################################################


def _derive_listname(path: Path) -> str:
    """Derive the HDF5 key name from the filename.

    Args:
        path: Path to the HDF5 file.

    Returns:
        ``'Naz100_Symbols'`` when ``naz100`` appears in the filename,
        ``'SP500_Symbols'`` when ``sp500`` appears, or raises
        ``ValueError`` if neither pattern matches.
    """
    name_lower = path.name.lower()
    if "naz100" in name_lower:
        return "Naz100_Symbols"
    if "sp500" in name_lower:
        return "SP500_Symbols"
    raise ValueError(
        f"Cannot derive HDF5 key from filename '{path.name}'. "
        f"Filename must contain 'naz100' or 'sp500'."
    )


def _leading_infill_length(prices: np.ndarray) -> int:
    """Return number of leading days where price never changes.

    A leading run is identified by finding the first index where the
    absolute price difference exceeds CONST_TOL.  All earlier dates
    are considered infill.  Returns 0 if the first difference is
    already non-zero.

    Args:
        prices: 1-D float array of adjusted close prices, oldest first.

    Returns:
        Number of leading dates (including day 0) to mark as NaN.
    """
    diffs = np.abs(np.diff(prices))
    nonzero = np.where(diffs > CONST_TOL)[0]
    if len(nonzero) == 0:
        # All prices are identical — entire series is infill.
        return len(prices)
    # nonzero[0] is the first index in diffs where price changed, which
    # corresponds to prices[nonzero[0]+1] being the first real price.
    # Everything up to and including prices[nonzero[0]] is infill.
    return int(nonzero[0]) + 1


def _trailing_infill_length(prices: np.ndarray) -> int:
    """Return number of trailing days where price never changes.

    Mirrors _leading_infill_length but works from the end.

    Args:
        prices: 1-D float array, oldest first.

    Returns:
        Number of trailing dates (including the last day) to mark NaN.
    """
    diffs = np.abs(np.diff(prices))
    nonzero = np.where(diffs > CONST_TOL)[0]
    if len(nonzero) == 0:
        return len(prices)
    # Last position in diffs where price changed is nonzero[-1].
    # prices[nonzero[-1]+1] onwards are infill.
    tail_start = int(nonzero[-1]) + 2  # index into prices
    return len(prices) - tail_start


def _mid_infill_mask(
    prices: np.ndarray,
    lead: int,
    trail: int,
    window: int,
) -> np.ndarray:
    """Return a boolean mask for mid-history infill runs.

    Searches the region between the leading and trailing infill blocks
    for runs of WINDOW+ consecutive days that are either constant-price
    or linearly interpolated.

    Constant-price detection: runs of WINDOW+ consecutive d1 values
    where abs(d1) < CONST_TOL.

    Linear-price detection: runs of (WINDOW-2)+ consecutive d2 values
    where all are < LINEAR_TOL, indicating a constant step size.  All
    d2 values in the run must satisfy the criterion — no bridging via
    isolated near-zero d2 values on either side of a larger gap.  A
    d2 run of length k corresponds to (k+2) prices, so we require
    k >= max(1, window - 2).

    Args:
        prices: 1-D float array.
        lead:   Number of leading infill days (already handled).
        trail:  Number of trailing infill days (already handled).
        window: Minimum run length in prices to flag as infill.

    Returns:
        Boolean array of length len(prices); True where mid infill.
    """
    n = len(prices)
    mask = np.zeros(n, dtype=bool)

    # Slice covering only the real-data region.
    lo = lead
    hi = n - trail if trail > 0 else n
    if hi - lo < window + 2:
        return mask

    seg = prices[lo:hi]
    seg_len = len(seg)

    d1 = np.diff(seg)           # shape (seg_len - 1,)
    abs_d1 = np.abs(d1)

    ##################################################################
    # Constant-price runs: consecutive d1 ≈ 0.
    # A run of k d1 positions covers k+1 prices.
    ##################################################################
    i = 0
    d1_len = seg_len - 1
    while i < d1_len:
        if abs_d1[i] < CONST_TOL:
            j = i
            while j < d1_len and abs_d1[j] < CONST_TOL:
                j += 1
            run_price_len = j - i + 1  # number of prices covered
            if run_price_len >= window:
                mask[lo + i: lo + j + 1] = True
            i = j
        else:
            i += 1

    ##################################################################
    # Linear-price runs: ALL consecutive d2 values < LINEAR_TOL.
    # Only mark if the stock has enough data for a d2 computation.
    # A run of k consecutive d2 positions covers k+2 prices.
    ##################################################################
    min_d2_run = max(1, window - 2)

    if seg_len >= 3:
        d2 = np.abs(np.diff(d1))   # shape (seg_len - 2,)
        i = 0
        d2_len = seg_len - 2
        while i < d2_len:
            if d2[i] < LINEAR_TOL:
                j = i
                while j < d2_len and d2[j] < LINEAR_TOL:
                    j += 1
                # d2 run covers d1[i:j+1] → prices[i:j+2] in seg-space.
                d2_run_len = j - i        # number of d2 positions
                run_price_len = d2_run_len + 2
                if d2_run_len >= min_d2_run and run_price_len >= window:
                    mask[lo + i: lo + j + 2] = True
                i = j
            else:
                i += 1

    return mask


###########################################################################
# Main processing
###########################################################################

def mark_infill(
    window: int = DEFAULT_WINDOW,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Read HDF5, mark infill as NaN, optionally write back.

    Args:
        window:  Minimum run length (days) to classify as mid infill.
        dry_run: If True, compute changes but do not write to disk.

    Returns:
        The modified DataFrame (NaN where infill was detected).
    """
    listname = _derive_listname(HDF5_FILE)
    logger.info("Reading %s key=%s", HDF5_FILE, listname)
    df: pd.DataFrame = pd.read_hdf(HDF5_FILE, listname).astype(float)

    n_rows, n_cols = df.shape
    logger.info("Loaded shape: (%d, %d)", n_rows, n_cols)
    print(f"Loaded: {n_rows} dates × {n_cols} symbols")
    print(f"Window: {window} days | dry_run={dry_run}\n")

    total_nan_added = 0
    summary_rows = []

    for symbol in df.columns:
        if symbol in SKIP_SYMBOLS:
            summary_rows.append(
                {
                    "symbol": symbol,
                    "lead": 0,
                    "trail": 0,
                    "mid_nan": 0,
                    "real_days": n_rows,
                    "total_nan": 0,
                }
            )
            continue

        prices = df[symbol].to_numpy(dtype=float)

        lead = _leading_infill_length(prices)
        trail = _trailing_infill_length(prices)
        mid_mask = _mid_infill_mask(prices, lead, trail, window)

        # Build a combined NaN mask for this symbol.
        nan_mask = np.zeros(n_rows, dtype=bool)
        if lead > 0:
            nan_mask[:lead] = True
        if trail > 0:
            nan_mask[n_rows - trail :] = True
        nan_mask |= mid_mask

        n_nan = int(nan_mask.sum())
        n_real = n_rows - n_nan
        n_mid = int(mid_mask.sum())

        summary_rows.append(
            {
                "symbol": symbol,
                "lead": lead,
                "trail": trail,
                "mid_nan": n_mid,
                "real_days": n_real,
                "total_nan": n_nan,
            }
        )
        total_nan_added += n_nan

        if n_nan > 0:
            df.loc[df.index[nan_mask], symbol] = np.nan

    # Print summary table.
    print(
        f"{'Symbol':8s} | {'lead':>6s} | {'trail':>6s} | "
        f"{'mid_nan':>7s} | {'real_days':>9s} | {'total_nan':>9s}"
    )
    print("-" * 62)
    for row in summary_rows:
        print(
            f"{row['symbol']:8s} | {row['lead']:6d} | {row['trail']:6d} | "
            f"{row['mid_nan']:7d} | {row['real_days']:9d} | "
            f"{row['total_nan']:9d}"
        )

    print("-" * 62)
    print(f"{'TOTAL':8s}   {'':6s}   {'':6s}   {'':7s}   {'':9s}   "
          f"{total_nan_added:9d}")
    print(f"\nTotal NaN cells added: {total_nan_added:,}")

    if dry_run:
        print("\n[dry-run] No changes written to disk.")
        logger.info("Dry run complete; %d NaN cells computed.", total_nan_added)
        return df

    ########################################################################
    # Write back to HDF5.
    ########################################################################
    logger.info(
        "Writing %d NaN cells back to %s", total_nan_added, HDF5_FILE
    )
    df.to_hdf(
        str(HDF5_FILE),
        key=listname,
        mode="a",
        format="table",
        append=False,
        complevel=5,
        complib="blosc",
    )
    print(f"\nWrote updated data to:\n  {HDF5_FILE}")
    logger.info("Write complete.")
    return df


###########################################################################
# CLI entry point
###########################################################################

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mark infilled prices as NaN in Naz100 HDF5 copy."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Compute NaN positions but do not write to disk.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        metavar="N",
        help=(
            f"Minimum run length (days) to flag mid-history infill "
            f"(default: {DEFAULT_WINDOW})."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        mark_infill(window=args.window, dry_run=args.dry_run)
    except Exception:
        logger.exception("mark_infill failed")
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
