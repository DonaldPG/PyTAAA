"""
detect_infilled.py — Detect infilled price data in an HDF5 quotes file.

Reads a Naz100 HDF5 file and returns a boolean DataFrame indicating
which price cells were infilled (True) versus real prices (False).
The HDF5 file is never modified.

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

Primary usage:
    from pathlib import Path
    from functions.detect_infilled import detect_infilled

    mask = detect_infilled(
        Path("/Users/donaldpg/pyTAAA_data/Naz100/symbols/Naz100_Symbols_nans.hdf5")
    )
    # mask is a bool DataFrame: True = infilled, False = real price
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from functions.logger_config import get_logger

###########################################################################
# Configuration
###########################################################################

# Minimum consecutive days to classify a mid-history run as infill.
DEFAULT_WINDOW = 5

# Symbols to leave untouched (no NaN marking applied).
# CASH has a constant price of 1.0 across all dates, which would cause every
# cell to be classified as leading/constant infill.  Skip it here so the
# output DataFrame leaves CASH rows as False (not infilled).
#
# Note: when detect_infilled_from_df() is called from data_loaders.py the
# caller explicitly enforces active_mask[CASH] = True regardless, so this
# skip is a processing optimisation in that code path, not a correctness
# requirement.  It IS required for the standalone detect_infilled() path
# (e.g. mark_infill_as_nan) where no caller override exists.
SKIP_SYMBOLS: frozenset[str] = frozenset({"CASH"})

# Tolerance for detecting zero price movement.
CONST_TOL = 1e-7

# Tolerance for detecting zero change in price step (linear infill).
LINEAR_TOL = 1e-5

logger = get_logger(__name__, log_file="detect_infilled.log")


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
        Number of leading dates (including day 0) to mark as infill.
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
        Number of trailing dates (including the last day) to mark infill.
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
# Main functions
###########################################################################

def detect_infilled_from_df(
    df: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """Detect infilled price cells from an already-loaded quotes DataFrame.

    Applies the same leading/trailing/mid-history infill detection as
    :func:`detect_infilled` but accepts a DataFrame that is already in
    memory rather than reading from disk.  Use this when the quotes
    DataFrame has already been loaded (e.g. the ``quote`` return value
    from ``loadQuotes_fromHDF``) to avoid a second HDF5 read.

    The input DataFrame is never modified.

    Args:
        df:     Quotes DataFrame — rows are dates, columns are symbol
                tickers, values are adjusted close prices (float).
                Matches the format returned directly by ``pd.read_hdf``.
        window: Minimum run length (days) to classify a mid-history
                constant or linear segment as infill.  Default 5.

    Returns:
        Boolean ``pd.DataFrame`` — True where a price cell is infilled,
        False where the price is real.  Shape, index, and column labels
        match *df*.
    """
    df = df.astype(float)
    n_rows, n_cols = df.shape
    logger.info(
        "detect_infilled_from_df: shape=(%d, %d) window=%d",
        n_rows, n_cols, window
    )

    # Initialise output boolean DataFrame (all False = real price).
    infill_mask = pd.DataFrame(
        False, index=df.index, columns=df.columns
    )

    total_infilled = 0

    for symbol in df.columns:
        if symbol in SKIP_SYMBOLS:
            continue

        prices = df[symbol].to_numpy(dtype=float)

        lead = _leading_infill_length(prices)
        trail = _trailing_infill_length(prices)
        mid_mask = _mid_infill_mask(prices, lead, trail, window)

        # Build a combined boolean mask for this symbol.
        combined = np.zeros(n_rows, dtype=bool)
        if lead > 0:
            combined[:lead] = True
        if trail > 0:
            combined[n_rows - trail:] = True
        combined |= mid_mask

        n_infilled = int(combined.sum())
        if n_infilled > 0:
            infill_mask[symbol] = combined
            total_infilled += n_infilled

    print(f"detect_infilled_from_df: {total_infilled:,} infilled cells "
          f"detected across {n_cols} symbols.")
    logger.info(
        "Detection complete: %d infilled cells across %d symbols.",
        total_infilled, n_cols
    )

    return infill_mask


def detect_infilled(
    hdf5_path: Path,
    window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """Detect infilled price cells in an HDF5 quotes file.

    Reads the HDF5 file at *hdf5_path* and returns a boolean DataFrame
    with the same shape, row index, and column headers as the stored
    quotes DataFrame.  True indicates an infilled (non-real) price;
    False indicates a genuine market price.

    The source HDF5 file is never modified.

    Args:
        hdf5_path: Full path to the HDF5 quotes file.  Must exist and
                   be readable; raises ``FileNotFoundError`` or
                   ``OSError`` with a descriptive message otherwise.
        window:    Minimum run length (days) to classify a mid-history
                   constant or linear segment as infill.  Default 5.

    Returns:
        Boolean ``pd.DataFrame`` — True where a price cell is infilled,
        False where the price is real.  Shape, index, and column labels
        match the quotes DataFrame stored in the HDF5 file.

    Raises:
        FileNotFoundError: If *hdf5_path* does not exist on disk.
        OSError: If the file exists but cannot be read as an HDF5 store
                 or the expected key is absent.
    """
    hdf5_path = Path(hdf5_path)

    if not hdf5_path.exists():
        raise FileNotFoundError(
            f"HDF5 quotes file not found: {hdf5_path}\n"
            f"Check that the path is correct and the file has been created."
        )

    listname = _derive_listname(hdf5_path)
    logger.info("Reading %s key=%s", hdf5_path, listname)
    try:
        df_quotes: pd.DataFrame = pd.read_hdf(
            str(hdf5_path), listname
        ).astype(float)
    except KeyError:
        raise OSError(
            f"HDF5 file exists but does not contain the expected key "
            f"'{listname}': {hdf5_path}"
        ) from None
    except Exception as exc:
        raise OSError(
            f"Could not read HDF5 file {hdf5_path}: {exc}"
        ) from exc

    n_rows, n_cols = df_quotes.shape
    logger.info("Loaded shape: (%d, %d)", n_rows, n_cols)
    print(f"detect_infilled: {n_rows} dates × {n_cols} symbols | "
          f"window={window}")

    return detect_infilled_from_df(df_quotes, window=window)
