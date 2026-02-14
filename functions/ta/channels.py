"""Channel-based technical indicators.

This module provides functions for calculating price channels using different
methods: percentile-based channels and min/max (DPG) channels. Channels help
identify support and resistance levels in price data.
"""

import numpy as np
from numpy.typing import NDArray


def percentileChannel(x: NDArray[np.floating], minperiod: int, maxperiod: int, 
                      incperiod: int, lowPct: float, hiPct: float) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Calculate percentile-based price channels for 1D time series.
    
    Computes upper and lower price channels by averaging percentiles across
    multiple lookback periods. This creates a smoothed envelope around the
    price data.
    
    Args:
        x: 1D numpy array of price data
        minperiod: Minimum lookback period for channel calculation
        maxperiod: Maximum lookback period for channel calculation
        incperiod: Increment step between periods
        lowPct: Lower percentile value (e.g., 10 for 10th percentile)
        hiPct: Upper percentile value (e.g., 90 for 90th percentile)
        
    Returns:
        tuple containing:
            - minchannel: Lower channel values (average of low percentiles)
            - maxchannel: Upper channel values (average of high percentiles)
            
    Example:
        >>> prices = np.array([100, 102, 101, 105, 103, 106])
        >>> low_ch, high_ch = percentileChannel(prices, 2, 5, 1, 20, 80)
        >>> # Returns 20th and 80th percentile channels
        
    Note:
        - Channels are averaged across all periods from minperiod to maxperiod
        - For early data points with insufficient history, uses available data
        - Useful for identifying support and resistance levels
    """
    periods = np.arange(minperiod, maxperiod, incperiod)
    minchannel = np.zeros(len(x), dtype=float)
    maxchannel = np.zeros(len(x), dtype=float)
    for i in range(len(x)):
        divisor = 0
        for j in range(len(periods)):
            minx = max(1, i-periods[j])
            if len(x[minx:i]) < 1:
                minchannel[i] = minchannel[i] + x[i]
                maxchannel[i] = maxchannel[i] + x[i]
                divisor += 1
            else:
                minchannel[i] = minchannel[i] + np.percentile(x[minx:i+1], lowPct)
                maxchannel[i] = maxchannel[i] + np.percentile(x[minx:i+1], hiPct)
                divisor += 1
        minchannel[i] /= divisor
        maxchannel[i] /= divisor
    return minchannel, maxchannel


def percentileChannel_2D(x: NDArray[np.floating], minperiod: int, maxperiod: int, 
                         incperiod: int, lowPct: float, hiPct: float) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Calculate percentile-based price channels for 2D multi-stock data.
    
    Computes upper and lower price channels for multiple stocks simultaneously
    by averaging percentiles across multiple lookback periods. Each stock gets
    its own channel calculated independently.
    
    Args:
        x: 2D numpy array of shape (n_stocks, n_dates) containing price data
        minperiod: Minimum lookback period for channel calculation
        maxperiod: Maximum lookback period for channel calculation
        incperiod: Increment step between periods
        lowPct: Lower percentile value (e.g., 10 for 10th percentile)
        hiPct: Upper percentile value (e.g., 90 for 90th percentile)
        
    Returns:
        tuple containing:
            - minchannel: 2D array of lower channel values
            - maxchannel: 2D array of upper channel values
            
    Example:
        >>> prices = np.random.rand(10, 100)  # 10 stocks, 100 days
        >>> low_ch, high_ch = percentileChannel_2D(prices, 5, 20, 5, 20, 80)
        >>> # Returns channels for all 10 stocks
        
    Note:
        - Prints diagnostic information about channel parameters and data range
        - For early data points with insufficient history, uses available data
        - More efficient than calling percentileChannel in a loop
    """
    print(" ... inside percentileChannel_2D ...  x min,mean,max = ", x.min(), x.mean(), x.max())
    periods = np.arange(minperiod, maxperiod, incperiod)
    minchannel = np.zeros((x.shape[0], x.shape[1]), dtype=float)
    maxchannel = np.zeros((x.shape[0], x.shape[1]), dtype=float)
    for i in range(x.shape[1]):
        divisor = 0
        for j in range(len(periods)):
            minx = int(max(1, i-periods[j])+.5)
            if len(x[0, minx:i]) < 1:
                minchannel[:, i] = minchannel[:, i] + x[:, i]
                maxchannel[:, i] = maxchannel[:, i] + x[:, i]
                divisor += 1
            else:
                minchannel[:, i] = minchannel[:, i] + np.percentile(x[:, minx:i+1], lowPct, axis=-1)
                maxchannel[:, i] = maxchannel[:, i] + np.percentile(x[:, minx:i+1], hiPct, axis=-1)
                divisor += 1
        minchannel[:, i] /= divisor
        maxchannel[:, i] /= divisor
    print(" minperiod,maxperiod,incperiod = ", minperiod, maxperiod, incperiod)
    print(" lowPct,hiPct = ", lowPct, hiPct)
    print(" x min,mean,max = ", x.min(), x.mean(), x.max())
    print(" divisor = ", divisor)
    return minchannel, maxchannel


def dpgchannel(x: NDArray[np.floating], minperiod: int, maxperiod: int, 
               incperiod: int) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Calculate min/max price channels for 1D time series (DPG method).
    
    Computes upper and lower price channels by averaging minimum and maximum
    values across multiple lookback periods. The DPG (dynamic price channel)
    method creates an envelope that tracks price extremes.
    
    Args:
        x: 1D numpy array of price data
        minperiod: Minimum lookback period for channel calculation
        maxperiod: Maximum lookback period for channel calculation
        incperiod: Increment step between periods
        
    Returns:
        tuple containing:
            - minchannel: Lower channel values (average of minimums)
            - maxchannel: Upper channel values (average of maximums)
            
    Example:
        >>> prices = np.array([100, 102, 101, 105, 103, 106])
        >>> low_ch, high_ch = dpgchannel(prices, 2, 5, 1)
        >>> # Returns min/max channels averaged over periods 2-5
        
    Note:
        - Channels are averaged across all periods from minperiod to maxperiod
        - For early data points with insufficient history, uses available data
        - Named 'dpg' for Donald P. Galanis (original developer)
    """
    periods = np.arange(minperiod, maxperiod, incperiod)
    minchannel = np.zeros(len(x), dtype=float)
    maxchannel = np.zeros(len(x), dtype=float)
    for i in range(len(x)):
        divisor = 0
        for j in range(len(periods)):
            minx = max(1, i-periods[j])
            if len(x[minx:i]) < 1:
                minchannel[i] = minchannel[i] + x[i]
                maxchannel[i] = maxchannel[i] + x[i]
                divisor += 1
            else:
                minchannel[i] = minchannel[i] + min(x[minx:i+1])
                maxchannel[i] = maxchannel[i] + max(x[minx:i+1])
                divisor += 1
        minchannel[i] /= divisor
        maxchannel[i] /= divisor
    return minchannel, maxchannel


def dpgchannel_2D(x: NDArray[np.floating], minperiod: int, maxperiod: int, 
                  incperiod: int) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Calculate min/max price channels for 2D multi-stock data (DPG method).
    
    Computes upper and lower price channels for multiple stocks simultaneously
    by averaging minimum and maximum values across multiple lookback periods.
    Each stock gets its own channel calculated independently.
    
    Args:
        x: 2D numpy array of shape (n_stocks, n_dates) containing price data
        minperiod: Minimum lookback period for channel calculation
        maxperiod: Maximum lookback period for channel calculation
        incperiod: Increment step between periods
        
    Returns:
        tuple containing:
            - minchannel: 2D array of lower channel values
            - maxchannel: 2D array of upper channel values
            
    Example:
        >>> prices = np.random.rand(10, 100)  # 10 stocks, 100 days
        >>> low_ch, high_ch = dpgchannel_2D(prices, 5, 20, 5)
        >>> # Returns min/max channels for all 10 stocks
        
    Note:
        - More efficient than calling dpgchannel in a loop
        - For early data points with insufficient history, uses available data
        - Named 'dpg' for Donald P. Galanis (original developer)
    """
    periods = np.arange(minperiod, maxperiod, incperiod)
    minchannel = np.zeros((x.shape[0], x.shape[1]), dtype=float)
    maxchannel = np.zeros((x.shape[0], x.shape[1]), dtype=float)
    for i in range(x.shape[1]):
        divisor = 0
        for j in range(len(periods)):
            minx = max(1, i-periods[j])
            if len(x[0, minx:i]) < 1:
                minchannel[:, i] = minchannel[:, i] + x[:, i]
                maxchannel[:, i] = maxchannel[:, i] + x[:, i]
                divisor += 1
            else:
                minchannel[:, i] = minchannel[:, i] + np.min(x[:, minx:i+1], axis=-1)
                maxchannel[:, i] = maxchannel[:, i] + np.max(x[:, minx:i+1], axis=-1)
                divisor += 1
        minchannel[:, i] /= divisor
        maxchannel[:, i] /= divisor
    return minchannel, maxchannel
