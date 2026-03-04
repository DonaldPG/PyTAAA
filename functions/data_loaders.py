"""Data loading functions separated from computation.

This module provides pure data loading functionality,
separated from computation to enable unit testing.

Phase 4a: Extract data loading from PortfolioPerformanceCalcs
"""

from typing import Optional, Tuple, Union, List
import numpy as np
from functions.UpdateSymbols_inHDF5 import loadQuotes_fromHDF
from functions.TAfunctions import interpolate, cleantobeginning, cleantoend


def _build_active_mask_from_raw(raw_adjclose: np.ndarray) -> np.ndarray:
    """Build a boolean index-membership mask from raw (pre-cleaning) prices.

    A symbol is considered NOT in the index at trailing dates when
    yfinance stopped downloading new quotes for it.  This is detected
    by finding the last date with a real (non-NaN) price; every date
    after that is marked inactive.

    Leading NaN (pre-IPO / pre-listing) is similarly detected by
    finding the first non-NaN date; every date before that is marked
    inactive.

    Interior NaN values (data gaps for stocks still in the index) are
    NOT marked inactive — the stock is still tradeable on those dates.

    Args:
        raw_adjclose: 2D array of adjusted close prices (n_stocks, n_days)
            as loaded from HDF5, before any cleaning (may contain NaN).

    Returns:
        active: boolean array (n_stocks, n_days).  True = stock is
            considered to be in the index on that date.
    """
    n_stocks, n_days = raw_adjclose.shape
    active = np.ones((n_stocks, n_days), dtype=bool)

    for i in range(n_stocks):
        prices = raw_adjclose[i]
        non_nan_idx = np.where(~np.isnan(prices))[0]
        if len(non_nan_idx) == 0:
            active[i, :] = False
            continue
        first_real = non_nan_idx[0]
        last_real = non_nan_idx[-1]
        # Mark pre-listing period inactive (pre-IPO or pre-addition to index).
        if first_real > 0:
            active[i, :first_real] = False
        # Mark post-removal period inactive (trailing NaN = no new quotes,
        # meaning the stock was removed from the watched universe).
        if last_real < n_days - 1:
            active[i, last_real + 1:] = False

    return active


def load_quotes_for_analysis(
    symbols_file: str,
    json_fn: str,
    verbose: bool = False,
    include_active_mask: bool = False,
) -> Union[
    Tuple[np.ndarray, List[str], np.ndarray],
    Tuple[np.ndarray, List[str], np.ndarray, np.ndarray],
]:
    """Load and prepare quote data for analysis.

    Loads quote data from HDF5 file and applies cleaning operations:

    - Interpolation to fill interior NaN gaps
    - Clean data from beginning (copy first real price to leading NaN)
    - Clean data to end (copy last real price to trailing NaN)

    Optionally builds an index-membership mask (``active_mask``) from
    the raw data BEFORE cleaning; trailing NaN indicate the stock was
    removed from the index and no longer downloaded by the updater.

    Args:
        symbols_file: Path to symbols file (e.g., "symbols/Naz100_Symbols.txt")
        json_fn: Path to JSON configuration file.
        verbose: Whether to print progress messages.
        include_active_mask: When True, return a 4th element with the
            boolean membership mask (n_stocks, n_days).  CASH is always
            True.  Defaults to False for backward compatibility.

    Returns:
        If ``include_active_mask=False`` (default)::

            (adjClose, symbols, datearray)

        If ``include_active_mask=True``::

            (adjClose, symbols, datearray, active_mask)

        - *adjClose*: 2D numpy array of adjusted close prices (stocks × days)
        - *symbols*: List of stock ticker symbols
        - *datearray*: Array of dates corresponding to columns
        - *active_mask*: boolean array (stocks × days), True = in index

    Raises:
        FileNotFoundError: If symbols file doesn't exist.
        ValueError: If data loading fails.

    Example:
        >>> adjClose, symbols, dates = load_quotes_for_analysis(
        ...     "symbols/Naz100_Symbols.txt",
        ...     "config/pytaaa_naz100_pine.json"
        ... )
        >>> print(f"Loaded {len(symbols)} symbols, {adjClose.shape[1]} days")
    """
    if verbose:
        print(f"   . Loading quotes from: {symbols_file}")

    # Load from HDF5 (returns 5-tuple, we use first 3)
    adjClose, symbols, datearray, _, _ = loadQuotes_fromHDF(symbols_file, json_fn)

    # Add CASH to symbols list if not present
    if 'CASH' not in symbols:
        symbols.append('CASH')
        # Add a row of 1.0s for CASH prices — all real data, no NaN.
        cash_prices = np.ones((1, adjClose.shape[1]), dtype=float)
        adjClose = np.vstack([adjClose, cash_prices])
        if verbose:
            print(f"   . Added CASH symbol to symbols list")

    if verbose:
        print(f"   . Loaded {adjClose.shape[0]} symbols, {adjClose.shape[1]} days")

    # Build membership mask from raw NaN pattern BEFORE cleaning.
    # Trailing NaN → stock removed from index (yfinance stopped updating).
    # Leading NaN  → stock not yet listed / added to index.
    active_mask = _build_active_mask_from_raw(adjClose)

    if verbose:
        print("   . Cleaning data (interpolate, cleantobeginning, cleantoend)")

    # Clean data for each symbol (in-place modification).
    for i in range(adjClose.shape[0]):
        adjClose[i, :] = interpolate(adjClose[i, :])
        adjClose[i, :] = cleantobeginning(adjClose[i, :])
        adjClose[i, :] = cleantoend(adjClose[i, :])

    # CASH is always active and always priced at 1.0.
    if 'CASH' in symbols:
        cash_idx = symbols.index('CASH')
        adjClose[cash_idx, :] = 1.0
        active_mask[cash_idx, :] = True
        if verbose:
            print("   . Set CASH prices to constant 1.0 for all dates")

    n_active_now = int(active_mask[:, -1].sum())
    n_inactive_now = int((~active_mask[:, -1]).sum())
    print(
        f"   . active_mask: {n_active_now} symbols active on last date, "
        f"{n_inactive_now} inactive (removed from index)"
    )

    if include_active_mask:
        return adjClose, symbols, datearray, active_mask
    return adjClose, symbols, datearray
