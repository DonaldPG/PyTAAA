"""Data loading functions separated from computation.

This module provides pure data loading functionality,
separated from computation to enable unit testing.

Phase 4a: Extract data loading from PortfolioPerformanceCalcs
"""

from typing import Optional, Tuple, Union, List
import numpy as np
from functions.UpdateSymbols_inHDF5 import loadQuotes_fromHDF
from functions.TAfunctions import interpolate, cleantobeginning, cleantoend
from functions.detect_infilled import detect_infilled_from_df



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

    # Load from HDF5: quote is the raw DataFrame (dates × symbols) used to
    # build the infill mask before any cleaning is applied.
    adjClose, symbols, datearray, quote, _ = loadQuotes_fromHDF(
        symbols_file, json_fn
    )

    if verbose:
        print(f"   . Loaded {adjClose.shape[0]} symbols, {adjClose.shape[1]} days")

    # Build membership mask from the raw quote DataFrame BEFORE any cleaning.
    # detect_infilled_from_df returns True = infilled; invert to get active
    # (True = real price).  Transpose from DataFrame layout (n_days × n_stocks)
    # to the array layout (n_stocks × n_days) used throughout the pipeline.
    #
    # CASH handling is done exactly once, here, in two exclusive branches:
    #   a) CASH is already in the HDF5 (in both quote and symbols): enforce its
    #      active_mask row to True unconditionally, since its constant price of
    #      1.0 would otherwise be flagged as 100% infilled.
    #   b) CASH is not in the HDF5: append it to symbols, adjClose, and
    #      active_mask with all-True (always eligible for allocation).
    infill_df = detect_infilled_from_df(quote)
    active_mask = ~infill_df.values.T  # Shape: (n_stocks, n_days)

    if 'CASH' in symbols:
        # Case (a): CASH came from the HDF5; force its row active.
        cash_idx = symbols.index('CASH')
        active_mask[cash_idx, :] = True
        if verbose:
            print("   . CASH already present in data (forced active)")
    else:
        # Case (b): CASH absent from HDF5; append to all three structures.
        symbols.append('CASH')
        cash_prices = np.ones((1, adjClose.shape[1]), dtype=float)
        adjClose = np.vstack([adjClose, cash_prices])
        active_mask = np.vstack(
            [active_mask, np.ones((1, active_mask.shape[1]), dtype=bool)]
        )
        if verbose:
            print("   . Added CASH symbol (always active)")

    # Guarantee that active_mask, adjClose, and (later) signal2D all share the
    # same first dimension so downstream boolean indexing cannot silently
    # broadcast to a wrong shape.
    assert active_mask.shape == adjClose.shape, (
        f"active_mask shape {active_mask.shape} != "
        f"adjClose shape {adjClose.shape}"
    )

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
