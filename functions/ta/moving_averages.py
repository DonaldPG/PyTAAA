"""Moving average implementations for technical analysis.

This module provides Simple Moving Averages (SMA), Hull Moving Averages (HMA),
and related moving window calculations for both 1D and 2D data.
"""

import numpy as np
from numpy.typing import NDArray


def SMA(x: NDArray[np.floating], periods: int) -> NDArray[np.floating]:
    """Calculate Simple Moving Average for 1D time series.
    
    Computes the simple moving average (arithmetic mean) over a rolling
    window of specified length.
    
    Args:
        x: 1D numpy array of price or indicator data
        periods: Number of periods (window size) for moving average
        
    Returns:
        NDArray[np.floating]: Array of simple moving average values
        
    Example:
        >>> prices = np.array([100, 102, 101, 105, 103])
        >>> sma_3 = SMA(prices, 3)
        >>> # Returns 3-period moving average
        
    Note:
        - Early values (before window fills) use all available data
        - Each point computes mean from max(0, i-periods) to i+1
    """
    SMA_result = np.zeros((x.shape[0]), dtype=float)
    for i in range(x.shape[0]):
        minx = np.max((0, i-periods))
        SMA_result[i] = np.mean(x[minx:i+1], axis=-1)
    return SMA_result


def SMA_2D(x: NDArray[np.floating], periods: int) -> NDArray[np.floating]:
    """Calculate Simple Moving Average for 2D multi-stock data.
    
    Computes the simple moving average for multiple stocks simultaneously.
    Each stock's moving average is calculated independently along the time axis.
    
    Args:
        x: 2D numpy array of shape (n_stocks, n_dates) containing price/indicator data
        periods: Number of periods (window size) for moving average
        
    Returns:
        NDArray[np.floating]: 2D array of simple moving average values
        
    Example:
        >>> prices = np.random.rand(10, 100)  # 10 stocks, 100 days
        >>> sma_20 = SMA_2D(prices, 20)
        >>> # Returns 20-day SMA for all 10 stocks
        
    Note:
        - Loops over dates (axis 1) computing mean across stocks
        - More efficient than calling SMA in a loop
        - Early values use partial windows (before window fills)
    """
    SMA_result = np.zeros((x.shape[0], x.shape[1]), dtype=float)
    for i in range(x.shape[1]):
        minx = np.max((0, i-periods))
        SMA_result[:, i] = np.mean(x[:, minx:i+1], axis=-1)
    return SMA_result


def SMS(x: NDArray[np.floating], periods: int) -> NDArray[np.floating]:
    """Calculate Simple Moving Sum for 1D time series.
    
    Computes the sum over a rolling window of specified length. This is
    the non-averaged version of SMA, useful for accumulation calculations.
    
    Args:
        x: 1D numpy array of values to sum
        periods: Number of periods (window size) for moving sum
        
    Returns:
        NDArray[np.floating]: Array of simple moving sum values
        
    Example:
        >>> values = np.array([1, 2, 3, 4, 5])
        >>> sms_3 = SMS(values, 3)
        >>> # Returns cumulative 3-period sums: [1, 3, 6, 9, 12]
        
    Note:
        - For each point i, returns sum(x[max(0,i-periods):i+1])
        - Early values use all available data
        - Related to SMA but without division by window size
    """
    _SMS = np.zeros((x.shape[0]), dtype=float)
    for i in range(x.shape[0]):
        minx = np.max((0, i-periods))
        _SMS[i] = np.sum(x[minx:i+1], axis=-1)
    return _SMS


def hma(x: NDArray[np.floating], period: int) -> NDArray[np.floating]:
    """Compute Hull Moving Average (HMA) for 2D multi-stock data.
    
    The Hull Moving Average is a sophisticated moving average that reduces lag
    while improving smoothness. It uses weighted moving averages and a square
    root period for the final smoothing.
    
    Formula: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    where WMA is weighted moving average and n is the period.
    
    Args:
        x: 2D numpy array of shape (n_stocks, n_dates) containing price data
        period: Period length for HMA calculation
        
    Returns:
        NDArray[np.floating]: 2D array of HMA values with same shape as input
        
    Example:
        >>> prices = np.random.rand(10, 100)  # 10 stocks, 100 days
        >>> hma_50 = hma(prices, 50)
        >>> # Returns 50-period HMA for all stocks
        
    Note:
        - Converts numpy array to pandas DataFrame for rolling window operations
        - Uses weighted moving average with linearly increasing weights
        - Final smoothing uses sqrt(period) as the window
        - More responsive than SMA while filtering noise better than EMA
        
    References:
        Hull, Alan (2005). "Hull Moving Average". Alanhull.com
    """
    # convert ndarray to pandas dataframe
    # x should have shape x[num_companies, n_days]
    import pandas as pd
    col_labels = ['stock'+str(i) for i in range(x.shape[0])]
    df = pd.DataFrame(
        data=x.T,    # values
        index=range(x.shape[1]),    # 1st column as index
        columns=col_labels
    )
    nday_half_range = np.arange(1, period//2+1)
    nday_range = np.arange(1, period + 1)
    _x = df
    _func1 = lambda _x: np.sum(_x * nday_half_range) / np.sum(nday_half_range)
    _func2 = lambda _x: np.sum(_x * nday_range) / np.sum(nday_range)
    wma_1 = _x.rolling(period//2).apply(_func1, raw=True)
    wma_2 = _x.rolling(period).apply(_func2, raw=True)
    diff = 2 * wma_1 - wma_2
    hma_result = diff.rolling(int(np.sqrt(period))).mean()
    hma_result = hma_result.values.T
    return hma_result


def hma_pd(data: NDArray[np.floating], period: int) -> NDArray[np.floating]:
    """Calculate Hull Moving Average for a 2D dataset (alternative implementation).
    
    This is an alternative HMA implementation that processes each stock separately.
    It's approximately 3x slower than the vectorized hma() function but may be
    more readable.
    
    Args:
        data: numpy array of shape (n_stocks, n_dates) containing price data
        period: integer representing the HMA period
    
    Returns:
        NDArray[np.floating]: 2D array of HMA values
        
    Note: 
        The elapsed time for this function is 3x that for function "hma"
    """
    import pandas as pd

    hma_result = np.zeros_like(data)
    for icompany in range(data.shape[0]):
        pd_data = pd.Series(list(data[icompany, :]))

        wma1 = pd_data.rolling(int(period/2)).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x)+1)),
            raw=True
        )
        wma2 = pd_data.rolling(period).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x)+1)),
            raw=True
        )

        hma_non_smooth = 2 * wma1 - wma2
        hma_result[icompany, :] = hma_non_smooth.rolling(int(np.sqrt(period))).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x)+1)),
            raw=True
        )

    return hma_result


def SMA_filtered_2D(x: NDArray[np.floating], periods: int, 
                    filt_min: float = -0.0125, filt_max: float = 0.0125) -> NDArray[np.floating]:
    """Calculate filtered Simple Moving Average for 2D data.
    
    Computes SMA only on dates where the gain/loss is outside the filter range.
    Interpolates between filtered points to create a smoother average that
    ignores small daily fluctuations.
    
    Args:
        x: 2D numpy array of shape (n_stocks, n_dates) containing price data
        periods: Number of periods for moving average calculation
        filt_min: Minimum gain/loss threshold (as decimal, e.g., -0.0125 for -1.25%)
        filt_max: Maximum gain/loss threshold (as decimal, e.g., 0.0125 for 1.25%)
        
    Returns:
        NDArray[np.floating]: 2D array of filtered SMA values
        
    Note:
        - Only computes SMA on points where daily gain/loss exceeds thresholds
        - Interpolates between significant points
        - Useful for filtering out noise from small daily movements
    """
    fsma = np.zeros((x.shape[0], x.shape[1]), dtype=float)
    gainloss = np.zeros_like(x)
    gainloss[:, 1:] = x[:, 1:] / x[:, :-1]
    gainloss[np.isnan(x)] = 1.
    gainloss -= 1.0
    ii, jj = np.where((gainloss <= filt_min) | (gainloss >= filt_max))
    x_count = np.zeros_like(x, dtype='int')
    x_count[ii, jj] = 1
    x_count = np.cumsum(x_count, axis=-1) - 1
    for i in range(x.shape[0]):
        indices = x_count[i, :]
        iii = np.where(indices[1:] != indices[:-1])[0] + 1
        SMA_sparse = SMA(x[i, iii], periods)
        if SMA_sparse.size > periods:
            fsma[i, iii[0]:iii[-1]+1] = np.interp(
                np.arange(iii[0], iii[-1]+1),
                iii,
                SMA_sparse
            )
            fsma[i, :iii[0]+1] = SMA_sparse[0]
            fsma[i, iii[-1]:] = SMA_sparse[-1]
    return fsma


def MoveMax(x: NDArray[np.floating], periods: int) -> NDArray[np.floating]:
    """Calculate moving maximum for 1D time series.
    
    Computes the maximum value over a rolling window of specified length.
    
    Args:
        x: 1D numpy array of price or indicator data
        periods: Number of periods (window size) for moving maximum
        
    Returns:
        NDArray[np.floating]: Array of moving maximum values
        
    Example:
        >>> prices = np.array([100, 102, 101, 105, 103])
        >>> move_max = MoveMax(prices, 3)
        >>> # Returns highest value in each 3-period window
        
    Note:
        - For each point i, returns max(x[max(0,i-periods):i+1])
        - Early values use all available data
        - Useful for tracking recent highs in technical analysis
    """
    MMax = np.zeros((x.shape[0]), dtype=float)
    for i in range(x.shape[0]):
        minx = max(0, i-periods)
        MMax[i] = np.max(x[minx:i+1], axis=-1)
    return MMax


def MoveMax_2D(x: NDArray[np.floating], periods: int) -> NDArray[np.floating]:
    """Calculate moving maximum for 2D multi-stock data.
    
    Computes the maximum value over a rolling window for multiple stocks
    simultaneously. Each stock's moving maximum is calculated independently.
    
    Args:
        x: 2D numpy array of shape (n_stocks, n_dates) containing price/indicator data
        periods: Number of periods (window size) for moving maximum
        
    Returns:
        NDArray[np.floating]: 2D array of moving maximum values
        
    Example:
        >>> prices = np.random.rand(10, 100)  # 10 stocks, 100 days
        >>> move_max = MoveMax_2D(prices, 20)
        >>> # Returns 20-day rolling maximum for all 10 stocks
        
    Note:
        - For each date i, returns max(x[:,max(0,i-periods):i+1])
        - Early values use all available data
        - Useful for tracking recent highs across multiple stocks
        - More efficient than calling MoveMax in a loop
    """
    MMax = np.zeros((x.shape[0], x.shape[1]), dtype=float)
    for i in range(x.shape[1]):
        minx = max(0, i-periods)
        MMax[:, i] = np.max(x[:, minx:i+1], axis=-1)
    return MMax


def MoveMin(x: NDArray[np.floating], periods: int) -> NDArray[np.floating]:
    """Calculate moving minimum for 1D time series.
    
    Computes the minimum value over a rolling window of specified length.
    
    Args:
        x: 1D numpy array of price or indicator data
        periods: Number of periods (window size) for moving minimum
        
    Returns:
        NDArray[np.floating]: Array of moving minimum values
        
    Example:
        >>> prices = np.array([100, 102, 101, 105, 103])
        >>> move_min = MoveMin(prices, 3)
        >>> # Returns lowest value in each 3-period window
        
    Note:
        - For each point i, returns min(x[max(0,i-periods):i+1])
        - Early values use all available data
        - Useful for tracking recent lows in technical analysis
    """
    MMin = np.zeros((x.shape[0]), dtype=float)
    for i in range(x.shape[0]):
        minx = max(0, i-periods)
        MMin[i] = np.min(x[minx:i+1], axis=-1)
    return MMin
