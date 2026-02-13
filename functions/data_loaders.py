"""Data loading functions separated from computation.

This module provides pure data loading functionality,
separated from computation to enable unit testing.

Phase 4a: Extract data loading from PortfolioPerformanceCalcs
"""

from typing import Tuple, List
import numpy as np
from functions.UpdateSymbols_inHDF5 import loadQuotes_fromHDF
from functions.TAfunctions import interpolate, cleantobeginning, cleantoend


def load_quotes_for_analysis(
    symbols_file: str,
    json_fn: str,
    verbose: bool = False
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Load and prepare quote data for analysis.
    
    Loads quote data from HDF5 file and applies cleaning operations:
    - Interpolation to fill gaps
    - Clean data from beginning (remove leading invalid data)
    - Clean data to end (remove trailing invalid data)
    
    Args:
        symbols_file: Path to symbols file (e.g., "symbols/Naz100_Symbols.txt")
        json_fn: Path to JSON configuration file
        verbose: Whether to print progress messages
        
    Returns:
        Tuple of (adjClose_array, symbols_list, date_array):
        - adjClose_array: 2D numpy array of adjusted close prices (stocks x days)
        - symbols_list: List of stock ticker symbols
        - date_array: Array of dates corresponding to columns
        
    Raises:
        FileNotFoundError: If symbols file doesn't exist
        ValueError: If data loading fails
        
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
        print(f"   . Cleaning data (interpolate, cleantobeginning, cleantoend)")
    
    # Clean data for each symbol (in-place modification)
    for i in range(adjClose.shape[0]):
        adjClose[i, :] = interpolate(adjClose[i, :])
        adjClose[i, :] = cleantobeginning(adjClose[i, :])
        adjClose[i, :] = cleantoend(adjClose[i, :])
    
    return adjClose, symbols, datearray
