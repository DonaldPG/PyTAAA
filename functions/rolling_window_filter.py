"""Rolling Window Data Quality Filter Module.

This module provides functions to filter signals based on data quality
within a rolling window of adjusted close prices.
"""

import numpy as np


def apply_rolling_window_filter(adjClose: np.ndarray, signal2D: np.ndarray, window_size: int) -> np.ndarray:
    """Apply rolling window data quality filter to signal matrix.
    
    For each stock and date, checks the rolling window of adjusted close prices
    for sufficient valid (non-NaN, non-constant) data. If less than 50% of the
    window contains valid data, zeros the corresponding signal.
    
    Args:
        adjClose: Adjusted close prices (n_stocks, n_days)
        signal2D: Signal matrix (n_stocks, n_days), modified in-place
        window_size: Size of rolling window in days
        
    Returns:
        Modified signal2D matrix
        
    Note:
        Only applies filter when date_idx >= window_size - 1 to ensure
        full window is available. Earlier dates are left unchanged.
    """
    n_stocks, n_days = adjClose.shape
    
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
                # Count valid non-constant data
                valid_non_constant_count = len(valid_data)
            
            # Apply 50% threshold
            if valid_non_constant_count < window_size * 0.5:
                signal2D[stock_idx, date_idx] = 0.0
    
    return signal2D