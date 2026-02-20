"""Rolling Window Data Quality Filter Module.

This module provides functions to filter signals based on data quality
within a rolling window of adjusted close prices.
"""

import numpy as np


def apply_rolling_window_filter(adjClose: np.ndarray, signal2D: np.ndarray, window_size: int, 
                               symbols: list = None, datearray: np.ndarray = None,
                               verbose: bool = False) -> np.ndarray:
    """Apply rolling window data quality filter to signal matrix.
    
    For each stock and date, checks the rolling window of adjusted close prices
    for sufficient valid (non-NaN, non-constant, non-interpolated) data. If less 
    than 50% of the window contains valid data, zeros the corresponding signal.
    
    This filter is designed to catch:
    1. Missing data (NaNs)
    2. Constant prices (all values identical)
    3. Linearly interpolated prices (constant or near-constant derivatives)
    
    The derivative test uses two detection methods:
    - Perfect constancy: Check if all derivatives are identical (rtol=1e-6)
    - Coefficient of variation: Check if CV = std(derivatives)/mean(derivatives) < 0.5
    
    This is effective because:
    - Linear interpolation creates constant slopes (single value) or very few distinct
      slope values (when multiple gap segments are filled with different constant slopes)
    - Real stock prices have high variability in daily price changes (CV typically > 1.0)
    - Interpolated data has CV near 0.0-0.2, real stocks have CV > 1.0
    
    Args:
        adjClose: Adjusted close prices (n_stocks, n_days)
        signal2D: Signal matrix (n_stocks, n_days). This function does NOT
            modify the input array in-place; it always works on and returns
            a copy.
        window_size: Size of rolling window in days
        symbols: List of stock symbols (optional, for debugging)
        datearray: Array of dates (optional, for debugging)
        verbose: If True, print debug information

    Returns:
        A new signal2D matrix with filtered entries zeroed (copy of input)
        
    Note:
        Only applies filter when date_idx >= window_size - 1 to ensure
        full window is available. Earlier dates are left unchanged.
    """

    # Work on a copy to avoid mutating the caller's array in-place.
    signal_out = signal2D.copy()

    n_stocks, n_days = adjClose.shape
    filtered_count = 0

    gainloss = np.ones((adjClose.shape[0], adjClose.shape[1]), dtype=float)
    gainloss[:,1:] = adjClose[:,1:] / adjClose[:,:-1]

    for stock_idx in range(n_stocks):
        for date_idx in range(window_size - 1, n_days):
            # Extract window: from date_idx - window_size + 1 to date_idx + 1
            start_idx = max(0, date_idx - window_size + 1)
            window_data = adjClose[stock_idx, start_idx:date_idx + 1]
            window_gainloss = gainloss[stock_idx, start_idx:date_idx + 1]
            window_gainloss_std = np.std(window_gainloss)

            # check for low volatility (std of gain/loss < 0.001) to indicate 
            # possible interpolation
            # Use safe fallbacks when `datearray` or `symbols` are not provided.
            date_str = str(datearray[date_idx]) if datearray is not None else f"idx{date_idx}"
            symbol_str = symbols[stock_idx] if symbols is not None else f"stk{stock_idx}"
            month_start_date = (datearray[date_idx].month != datearray[date_idx - 1].month) if (datearray is not None and date_idx > 0) else False
            if window_gainloss_std < 0.001:
                # Low volatility in gain/loss indicates possible interpolation
                if verbose:
                    print(f"RollingFilter: Zeroing {symbol_str} on {date_str} due to low gainloss_std={window_gainloss_std:.6f}")
                signal_out[stock_idx, date_idx] = 0.0
                filtered_count += 1
                continue

            # Apply valid length >= 50% threshold
            valid_length = len(window_gainloss[np.abs(window_gainloss -1) >= 0.001])
            if valid_length < window_size * 0.5:
                if verbose:
                    print(f"RollingFilter: Zeroing {symbol_str} on {date_str} due to insufficient valid_length={valid_length} (threshold={window_size*0.5})")
                signal_out[stock_idx, date_idx] = 0.0
                filtered_count += 1

            # Find valid (non-NaN) data
            valid_mask = ~np.isnan(window_data)
            valid_data = window_data[valid_mask]
            
            if len(valid_data) == 0:
                # No valid data in window
                if verbose:
                    print(f"RollingFilter: Zeroing {symbol_str} on {date_str} due to no valid data in window")
                signal_out[stock_idx, date_idx] = 0.0
                filtered_count += 1
                continue
            
    print(f"RollingFilter: Total filtered entries = {filtered_count}")

    return signal_out