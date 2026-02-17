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
        signal2D: Signal matrix (n_stocks, n_days), modified in-place
        window_size: Size of rolling window in days
        symbols: List of stock symbols (optional, for debugging)
        datearray: Array of dates (optional, for debugging)
        verbose: If True, print debug information
        
    Returns:
        Modified signal2D matrix
        
    Note:
        Only applies filter when date_idx >= window_size - 1 to ensure
        full window is available. Earlier dates are left unchanged.
    """
    n_stocks, n_days = adjClose.shape
    filtered_count = 0
    
    for stock_idx in range(n_stocks):
        for date_idx in range(window_size - 1, n_days):
            # Extract window: from date_idx - window_size + 1 to date_idx + 1
            start_idx = max(0, date_idx - window_size + 1)
            window_data = adjClose[stock_idx, start_idx:date_idx + 1]
            
            # Find valid (non-NaN) data
            valid_mask = ~np.isnan(window_data)
            valid_data = window_data[valid_mask]
            
            if len(valid_data) == 0:
                # No valid data in window
                signal2D[stock_idx, date_idx] = 0.0
                continue
            
            # Check if data is constant (all values equal)
            if np.allclose(valid_data, valid_data[0]):
                # All valid values are the same - consider as no valid non-constant data
                valid_non_constant_count = 0
            else:
                # Check for linear interpolation by detecting overly constant derivatives
                # Linear interpolation creates constant or near-constant price differences
                # Real stock prices have varying derivatives with high coefficient of variation
                if len(valid_data) >= 3:
                    derivatives = np.diff(valid_data)  # Price differences: price[i+1] - price[i]
                    
                    # Method 1: Check if derivatives are perfectly constant (catches pure linear fill)
                    is_perfectly_constant = len(derivatives) > 0 and np.allclose(derivatives, derivatives[0], rtol=1e-6)
                    
                    # Method 2: Check coefficient of variation (std/mean ratio)
                    # Interpolated data has very low CV (<0.2), real stocks have high CV (>1.0)
                    mean_deriv = np.mean(derivatives)
                    if mean_deriv != 0:
                        coef_variation = np.std(derivatives) / np.abs(mean_deriv)
                        is_too_constant = coef_variation < 0.5  # Threshold: CV < 0.5 indicates interpolation
                    else:
                        is_too_constant = True  # Zero mean derivative = no real price movement
                    
                    if is_perfectly_constant or is_too_constant:
                        # Constant or near-constant slope detected = linear interpolation
                        valid_non_constant_count = 0
                        
                        # Log JEF filtering for debugging
                        if symbols and stock_idx < len(symbols) and symbols[stock_idx] == 'JEF':
                            if datearray is not None and date_idx < len(datearray):
                                date_str = str(datearray[date_idx])
                                if verbose or '2015' in date_str or '2016' in date_str or '2017' in date_str or '2018' in date_str:
                                    mean_val = np.mean(derivatives)
                                    std_val = np.std(derivatives)
                                    cv = coef_variation if mean_deriv != 0 else np.inf
                                    print(f"  ⚠️  FILTERING JEF on {date_str}: CV={cv:.4f}, mean_deriv={mean_val:.6f}, std_deriv={std_val:.6f}, perfectly_constant={is_perfectly_constant}")
                                    filtered_count += 1
                    else:
                        valid_non_constant_count = len(valid_data)
                else:
                    # Not enough data to check derivatives
                    valid_non_constant_count = len(valid_data)
            
            # Apply 50% threshold
            if valid_non_constant_count < window_size * 0.5:
                signal2D[stock_idx, date_idx] = 0.0
    
    return signal2D