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

    print("\n\n ... inside apply_rolling_window_filter")
    print(f"DEBUG: input signal2D shape = {signal2D.shape}")
    try:
        print(f"DEBUG: input signal2D ones count = {np.sum(signal2D == 1)}")
    except Exception:
        pass
    # Identity and symbol-order diagnostics to help confirm the caller
    # is using the returned array and that symbol indexing matches.
    try:
        print(f"DEBUG ids: id(input_signal2D)={id(signal2D)}")
        if symbols is not None and len(symbols) > 0:
            print(f"DEBUG symbols[0:10]={symbols[:10]}")
            if 'JEF' in symbols:
                print(f"DEBUG: JEF index in symbols = {symbols.index('JEF')}")
    except Exception:
        pass

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
            if (
                symbol_str == 'JEF' and \
                month_start_date and \
                (
                    '2015' in date_str or \
                    '2016' in date_str or \
                    '2017' in date_str or \
                    '2018' in date_str
                )
            ):
                print("DEBUG: Checking JEF on {}: gainloss_std={:.6f}".format(date_str, window_gainloss_std))
            if window_gainloss_std < 0.001:
                # Low volatility in gain/loss indicates possible interpolation
                if verbose or (symbols and symbol_str == 'JEF'):
                    print(f"RollingFilter: Zeroing {symbol_str} on {date_str} due to low gainloss_std={window_gainloss_std:.6f}")
                signal_out[stock_idx, date_idx] = 0.0
                filtered_count += 1
                continue

            # Apply valid length >= 50% threshold
            valid_length = len(window_gainloss[np.abs(window_gainloss -1) >= 0.001])
            if valid_length < window_size * 0.5:
                if verbose or (symbols and symbol_str == 'JEF'):
                    print(f"RollingFilter: Zeroing {symbol_str} on {date_str} due to insufficient valid_length={valid_length} (threshold={window_size*0.5})")
                signal_out[stock_idx, date_idx] = 0.0
                filtered_count += 1

            # Find valid (non-NaN) data
            valid_mask = ~np.isnan(window_data)
            valid_data = window_data[valid_mask]
            
            if len(valid_data) == 0:
                # No valid data in window
                if verbose or (symbols and symbol_str == 'JEF'):
                    print(f"RollingFilter: Zeroing {symbol_str} on {date_str} due to no valid data in window")
                signal_out[stock_idx, date_idx] = 0.0
                filtered_count += 1
                continue
            
            # # Check if data is constant (all values equal)
            # if np.allclose(valid_data, valid_data[0]):
            #     # All valid values are the same - consider as no valid non-constant data
            #     valid_non_constant_count = 0
            # else:
            #     # Check for linear interpolation by detecting overly constant derivatives
            #     # Linear interpolation creates constant or near-constant price differences
            #     # Real stock prices have varying derivatives with high coefficient of variation
            #     if len(valid_data) >= 3:
            #         derivatives = np.diff(valid_data)  # Price differences: price[i+1] - price[i]
                    
            #         # Method 1: Check if derivatives are perfectly constant (catches pure linear fill)
            #         is_perfectly_constant = len(derivatives) > 0 and np.allclose(derivatives, derivatives[0], rtol=1e-6)
                    
            #         # Method 2: Check coefficient of variation (std/mean ratio)
            #         # Interpolated data has very low CV (<0.2), real stocks have high CV (>1.0)
            #         mean_deriv = np.mean(derivatives)
            #         if mean_deriv != 0:
            #             coef_variation = np.std(derivatives) / np.abs(mean_deriv)
            #             is_too_constant = coef_variation < 0.5  # Threshold: CV < 0.5 indicates interpolation
            #         else:
            #             is_too_constant = True  # Zero mean derivative = no real price movement
                    
            #         if is_perfectly_constant or is_too_constant:
            #             # Constant or near-constant slope detected = linear interpolation
            #             valid_non_constant_count = 0
                        
            #             # Log JEF filtering for debugging
            #             if symbols and stock_idx < len(symbols) and symbols[stock_idx] == 'JEF':
            #                 if datearray is not None and date_idx < len(datearray):
            #                     date_str = str(datearray[date_idx])
            #                     if verbose or '2015' in date_str or '2016' in date_str or '2017' in date_str or '2018' in date_str:
            #                         mean_val = np.mean(derivatives)
            #                         std_val = np.std(derivatives)
            #                         cv = coef_variation if mean_deriv != 0 else np.inf
            #                         print(f"  ⚠️  FILTERING JEF on {date_str}: CV={cv:.4f}, mean_deriv={mean_val:.6f}, std_deriv={std_val:.6f}, perfectly_constant={is_perfectly_constant}")
            #                         filtered_count += 1
            #         else:
            #             valid_non_constant_count = len(valid_data)
            #     else:
            #         # Not enough data to check derivatives
            #         valid_non_constant_count = len(valid_data)
            
            # # Apply 50% threshold
            # if valid_non_constant_count < window_size * 0.5:
            #     signal2D[stock_idx, date_idx] = 0.0
    
    try:
        print(f"DEBUG after: output signal2D ones count = {np.sum(signal_out == 1)}")
        print(f"DEBUG ids: id(output_signal_out)={id(signal_out)}")
    except Exception:
        pass
    print(f"RollingFilter: Total filtered entries = {filtered_count}")

    return signal_out