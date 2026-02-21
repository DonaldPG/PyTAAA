"""Data cleaning and interpolation functions for price data.

This module provides functions for cleaning time series data, including
interpolation of missing values, boundary handling, and spike removal.
"""

import sys
import logging
import numpy as np
from numpy import isnan
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def interpolate(self: NDArray[np.floating], method: str = 'linear', verbose: bool = False) -> NDArray[np.floating]:
    """Interpolate missing values using linear interpolation after the first valid value.
    
    This function fills NaN values in a time series array using linear interpolation,
    but only for values that occur after the first valid (non-NaN) value. Values
    before the first valid value are left as NaN.
    
    The implementation is adapted from pandas Series.interpolate() but simplified
    to support only linear interpolation method.
    
    Args:
        self: 1D numpy array containing time series data with potential NaN values
        method: Interpolation method. Only 'linear' is currently supported
        verbose: If True, print detailed debug information during processing
        
    Returns:
        NDArray[np.floating]: Array with NaN values interpolated (after first valid value)
        
    Example:
        >>> data = np.array([np.nan, np.nan, 1.0, np.nan, 3.0, np.nan])
        >>> result = interpolate(data)
        >>> # Result: [nan, nan, 1.0, 2.0, 3.0, 3.0]
        
    Note:
        - Python version < 2.7.12 and >= 2.7.12 use slightly different logic
        - NaN values at the beginning of the series are preserved
        - Uses numpy.interp for linear interpolation
        
    References:
        Adapted from https://github.com/wesm/pandas/blob/master/pandas/core/series.py
    """
    if sys.version_info < (2, 7, 12):
        if verbose:
            logger.debug(" ... inside interpolate (old).... len(self) = %d", len(self))

        inds = np.arange(len(self))
        values = np.array(self.copy())
        if verbose:
            logger.debug(" ... values = %s", values)
            logger.debug(" ... values.dtype = %s", values.dtype)
            logger.debug(" ... type(values) = %s", type(values))
        invalid = np.isnan(values)
        valid = -1 * invalid
        firstIndex = valid.argmax()

        if verbose:
            logger.debug(" ... inside interpolate .... firstIndex = %d", firstIndex)

        valid = valid[firstIndex:]
        invalid = invalid[firstIndex:]

        if verbose:
            logger.debug(" ... inside interpolate .... len(valid) = %d", len(valid))
            logger.debug(" ... inside interpolate .... len(invalid) = %d", len(invalid))

        inds = inds[firstIndex:]
        result = values.copy()
        result[firstIndex:][invalid] = np.interp(inds[invalid], inds[valid == 0], values[firstIndex:][valid == 0])

        if verbose:
            logger.debug(" ... interpolate (old) finished")

    else:
        if verbose:
            logger.debug(" ... inside interpolate (new) .... len(self) = %d", len(self))
        inds = np.arange(len(self))
        values = np.array(self.copy())
        if verbose:
            logger.debug(" ... values = %s", values)
            logger.debug(" ... values.dtype = %s", values.dtype)
            logger.debug(" ... type(values) = %s", type(values))

        invalid_bool = np.isnan(values)
        valid_mask = ~invalid_bool
        
        # Check if there are any valid values
        if not np.any(valid_mask):
            # All values are NaN, return unchanged
            return values.copy()
        
        firstIndex = np.where(valid_mask)[0][0]
        lastIndex = np.where(valid_mask)[0][-1]

        if verbose:
            logger.debug(" ... inside interpolate .... firstIndex,lastIndex = %d, %d", firstIndex, lastIndex)

        # Only interpolate between firstIndex and lastIndex
        # Create masks for the range we want to interpolate
        range_mask = np.arange(len(values)) >= firstIndex
        range_mask &= np.arange(len(values)) <= lastIndex
        
        # Within this range, interpolate NaN values
        valid_in_range = valid_mask & range_mask
        invalid_in_range = invalid_bool & range_mask
        
        if np.any(invalid_in_range):
            # Get indices and values for interpolation
            all_inds = np.arange(len(values))
            interp_inds = all_inds[invalid_in_range]
            valid_inds = all_inds[valid_in_range]
            valid_values = values[valid_in_range]
            
            # Interpolate
            interpolated = np.interp(interp_inds, valid_inds, valid_values)
            result = values.copy()
            result[invalid_in_range] = interpolated
        else:
            result = values.copy()

    return result


def cleantobeginning(self: NDArray[np.floating]) -> NDArray[np.floating]:
    """Forward-fill NaN values at the beginning of a time series.
    
    Replaces all NaN values that occur before the first valid (non-NaN) value
    with a copy of that first valid value. This is useful for cleaning time
    series data where early historical values may be missing.
    
    Args:
        self: 1D numpy array containing time series data with potential NaN values
        
    Returns:
        NDArray[np.floating]: Array with leading NaN values replaced by the first
            valid value
            
    Example:
        >>> data = np.array([np.nan, np.nan, 100.0, 101.0, 102.0])
        >>> result = cleantobeginning(data)
        >>> # Result: [100.0, 100.0, 100.0, 101.0, 102.0]
        
    Note:
        - Only affects NaN values before the first valid value
        - NaN values after the first valid value are preserved
        - If all values are NaN, returns array unchanged
    """
    values = self.copy()
    invalid = isnan(values)
    valid = ~invalid  # Use ~ instead of deprecated - operator
    firstIndex = valid.argmax()
    for i in range(firstIndex):
        values[i] = values[firstIndex]
    return values


def cleantoend(self: NDArray[np.floating]) -> NDArray[np.floating]:
    """Backward-fill NaN values at the end of a time series.
    
    Replaces all NaN values that occur after the last valid (non-NaN) value
    with a copy of that last valid value. This is useful for cleaning time
    series data where recent values may be missing.
    
    Args:
        self: 1D numpy array containing time series data with potential NaN values
        
    Returns:
        NDArray[np.floating]: Array with trailing NaN values replaced by the last
            valid value
            
    Example:
        >>> data = np.array([100.0, 101.0, np.nan, 102.0, np.nan, np.nan])
        >>> result = cleantoend(data)
        >>> # Result: [100.0, 101.0, nan, 102.0, 102.0, 102.0]
        
    Note:
        - Only affects NaN values after the last valid value
        - NaN values before the last valid value are preserved
        - Implemented by reversing array, applying cleantobeginning, then reversing back
    """
    # reverse input 1D array and use cleantobeginning
    reverse = self[::-1]
    reverse = cleantobeginning(reverse)
    return reverse[::-1]


def clean_signal(array1D: NDArray[np.floating], symbol_name: str) -> NDArray[np.floating]:
    """Apply comprehensive cleaning to a price signal time series.
    
    Performs a three-step cleaning process on price data:
    1. Linear interpolation of missing values (after first valid value)
    2. Forward-fill NaN values at the beginning
    3. Backward-fill NaN values at the end
    
    Args:
        array1D: 1D numpy array containing price data with potential NaN values
        symbol_name: Stock symbol or identifier (used for logging)
        
    Returns:
        NDArray[np.floating]: Cleaned price array with all NaN values filled
        
    Example:
        >>> prices = np.array([np.nan, np.nan, 100.0, np.nan, 102.0, np.nan])
        >>> clean_prices = clean_signal(prices, "AAPL")
        >>> # Result: [100.0, 100.0, 100.0, 101.0, 102.0, 102.0]
        
    Note:
        - Prints a message if the cleaning process modified any values
        - The three cleaning steps are applied sequentially
        - Used to ensure complete price data for technical analysis
    """
    ### clean input signals (again)
    quotes_before_cleaning = array1D.copy()
    adjClose = interpolate(array1D)
    adjClose = cleantobeginning(adjClose)
    adjClose = cleantoend(adjClose)
    adjClose_changed = False in (adjClose == quotes_before_cleaning)
    logger.debug("   ... inside clean_signal ... symbol=%s, did cleaning change adjClose? %s", symbol_name, adjClose_changed)
    return adjClose


def cleanspikes(x: NDArray[np.floating], periods: int = 20, stddevThreshold: float = 5.0) -> NDArray[np.floating]:
    """Remove outlier spikes from price data based on gradient analysis.
    
    Identifies and removes price spikes by analyzing forward and backward
    gain/loss ratios. Points that show abnormal jumps in both directions
    (indicating a spike) are replaced with NaN.
    
    Args:
        x: 1D numpy array of price data
        periods: Number of periods for calculating moving statistics (currently unused)
        stddevThreshold: Number of standard deviations above median to flag as spike
        
    Returns:
        NDArray[np.floating]: Price array with spikes replaced by NaN
        
    Example:
        >>> prices = np.array([100, 101, 200, 102, 103])  # 200 is a spike
        >>> cleaned = cleanspikes(prices, stddevThreshold=3.0)
        >>> # Middle spike would be replaced with nan
        
    Note:
        - Computes forward ratio: x[i+1]/x[i] and reverse ratio: x[i]/x[i+1]
        - Spikes are detected when both forward and reverse ratios exceed threshold
        - Uses median-normalized deviations to identify outliers
    """
    # remove outliers from gradient of x (in 2 directions)
    x_clean = np.array(x).copy()
    test = np.zeros(x.shape[0], 'float')
    gainloss_f = x[1:] / x[:-1]
    gainloss_r = x[:-1] / x[1:]
    valid_f = gainloss_f[gainloss_f != 1.]
    valid_f = valid_f[~np.isnan(valid_f)]
    if len(valid_f) > 0:
        Stddev_f = np.std(valid_f) + 1.e-5
    else:
        Stddev_f = 1.e-5
    valid_r = gainloss_r[gainloss_r != 1.]
    valid_r = valid_r[~np.isnan(valid_r)]
    if len(valid_r) > 0:
        Stddev_r = np.std(valid_r) + 1.e-5
    else:
        Stddev_r = 1.e-5

    forward_test = gainloss_f / Stddev_f - np.median(gainloss_f / Stddev_f)
    reverse_test = gainloss_r / Stddev_r - np.median(gainloss_r / Stddev_r)

    test[:-1] += reverse_test
    test[1:] += forward_test
    test[np.isnan(test)] = 1.e-10

    x_clean[test > stddevThreshold] = np.nan

    return x_clean


def despike_2D(x: NDArray[np.floating], periods: int, stddevThreshold: float = 5.0) -> NDArray[np.floating]:
    """Remove outlier spikes from 2D multi-stock price data.
    
    Identifies and removes price spikes for multiple stocks simultaneously by
    analyzing gain/loss ratios and comparing them to rolling standard deviations.
    
    Args:
        x: 2D numpy array of shape (n_stocks, n_dates) containing price data
        periods: Lookback period for calculating rolling standard deviation
        stddevThreshold: Number of standard deviations threshold for spike detection
        
    Returns:
        NDArray[np.floating]: 2D array of despik price values (reconstructed from clipped gains)
        
    Example:
        >>> prices = np.random.rand(10, 100)  # 10 stocks, 100 days
        >>> clean_prices = despike_2D(prices, 20, stddevThreshold=5.0)
        
    Note:
        - Computes gain/loss ratios for each stock
        - Clips extreme gains exceeding threshold * rolling standard deviation
        - Reconstructs price series from clipped gains
        - More efficient than calling cleanspikes in a loop
    """
    # remove outliers from gradient of x (in 2nd dimension)
    gainloss = np.ones((x.shape[0], x.shape[1]), dtype=float)
    gainloss[:, 1:] = x[:, 1:] / x[:, :-1]
    for i in range(1, x.shape[1]):
        minx = max(0, i - periods)
        Stddev = np.std(gainloss[:, minx:i], axis=-1)
        Stddev *= stddevThreshold
        Stddev += 1.
        test = np.dstack((Stddev, gainloss[:, i]))
        gainloss[:, i] = np.min(test, axis=-1)
    gainloss[:, 0] = x[:, 0].copy()
    value = np.cumprod(gainloss, axis=1)
    return value
