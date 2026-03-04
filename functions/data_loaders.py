"""Data loading functions separated from computation.

This module provides pure data loading functionality,
separated from computation to enable unit testing.

Phase 4a: Extract data loading from PortfolioPerformanceCalcs
"""

from typing import Optional, Tuple, Union, List
import numpy as np
from functions.UpdateSymbols_inHDF5 import loadQuotes_fromHDF
from functions.TAfunctions import interpolate, cleantobeginning, cleantoend


##########################################################################
# Minimum number of consecutive trading days with an EXACTLY identical
# price to conclude that the HDF5 data is a "held constant" fill rather
# than real market data.  NDX stocks virtually never trade at exactly the
# same adjusted-close price for 20+ consecutive days; using 20 avoids
# false-positives while reliably catching stocks removed months or years
# ago whose last real price was forward-filled by the updater.
_MIN_CONSTANT_DAYS = 20


def _build_active_mask_from_raw(raw_adjclose: np.ndarray) -> np.ndarray:
    """Build a boolean index-membership mask from raw (pre-cleaning) prices.

    Two types of inactivity are detected in the HDF5 data:

    1. **Trailing NaN** — recent dates for which the updater no longer
       downloaded quotes (stock was removed from the index after the
       data was last fetched).

    2. **Trailing constant price** — dates for which the HDF5 holds the
       stock's last real adjusted-close price exactly unchanged for
       ``_MIN_CONSTANT_DAYS`` or more consecutive trading days.  The
       HDF5 updater does not download removed stocks, so their most
       recent rows carry a ``cleantoend``-held constant value.  This
       pattern catches stocks removed months or years ago, whose
       constant-price run precedes the shorter trailing-NaN region.

    The "truly last active" date is the last date where the price
    actually changed — i.e., the end of the preceding real-market
    segment.  Interior NaN (data gaps while still in the index) are
    NOT marked inactive.

    Args:
        raw_adjclose: 2D array of adjusted close prices (n_stocks, n_days)
            as read from HDF5, before any cleaning (may contain NaN).

    Returns:
        active: boolean array (n_stocks, n_days).  True = stock is in
            the index on that date and eligible for portfolio weight.
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

        # Mark pre-listing period inactive (pre-IPO or index addition).
        if first_real > 0:
            active[i, :first_real] = False

        # Mark trailing NaN inactive (updater stopped downloading).
        if last_real < n_days - 1:
            active[i, last_real + 1:] = False

        # Detect trailing constant-price run starting at last_real and
        # scanning backward.  The HDF5 updater holds the last real price
        # constant for removed stocks; identify the earliest date where
        # the price was still at that held value by finding the last day
        # where it first changed.
        last_price = prices[last_real]
        constant_start = last_real
        for k in range(last_real - 1, first_real - 1, -1):
            if np.isnan(prices[k]) or prices[k] != last_price:
                break
            constant_start = k

        constant_run = last_real - constant_start
        if constant_run >= _MIN_CONSTANT_DAYS:
            # The stock's price has been flat for at least _MIN_CONSTANT_DAYS
            # consecutive days — treat the entire flat region (and the
            # trailing NaN already marked) as inactive.
            active[i, constant_start:] = False

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

    if verbose:
        print(f"   . Loaded {adjClose.shape[0]} symbols, {adjClose.shape[1]} days")

    # Build membership mask from HDF5 data BEFORE adding CASH and BEFORE
    # cleaning.  CASH must not be included here: its price is a constant
    # 1.0 on every date, which would be (correctly) detected as a held-
    # constant flat price and incorrectly marked as inactive.
    active_mask = _build_active_mask_from_raw(adjClose)

    # Add CASH to symbols list if not present, AFTER building the mask.
    # Append a row of True so CASH is always eligible for allocation.
    if 'CASH' not in symbols:
        symbols.append('CASH')
        cash_prices = np.ones((1, adjClose.shape[1]), dtype=float)
        adjClose = np.vstack([adjClose, cash_prices])
        active_mask = np.vstack(
            [active_mask, np.ones((1, adjClose.shape[1]), dtype=bool)]
        )
        if verbose:
            print("   . Added CASH symbol (always active)")

    if verbose:
        print("   . Cleaning data (interpolate, cleantobeginning, cleantoend)")

    # Clean data for each symbol (in-place modification).
    for i in range(adjClose.shape[0]):
        adjClose[i, :] = interpolate(adjClose[i, :])
        adjClose[i, :] = cleantobeginning(adjClose[i, :])
        adjClose[i, :] = cleantoend(adjClose[i, :])

    # CASH is always priced at constant 1.0; enforce after cleaning.
    if 'CASH' in symbols:
        cash_idx = symbols.index('CASH')
        adjClose[cash_idx, :] = 1.0

    n_active_now = int(active_mask[:, -1].sum())
    n_inactive_now = int((~active_mask[:, -1]).sum())
    print(
        f"   . active_mask: {n_active_now} symbols active on last date, "
        f"{n_inactive_now} inactive (removed from index)"
    )

    if include_active_mask:
        return adjClose, symbols, datearray, active_mask
    return adjClose, symbols, datearray

    n_active_now = int(active_mask[:, -1].sum())
    n_inactive_now = int((~active_mask[:, -1]).sum())
    print(
        f"   . active_mask: {n_active_now} symbols active on last date, "
        f"{n_inactive_now} inactive (removed from index)"
    )

    if include_active_mask:
        return adjClose, symbols, datearray, active_mask
    return adjClose, symbols, datearray
