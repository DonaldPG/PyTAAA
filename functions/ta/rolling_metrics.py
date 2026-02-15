"""Rolling window performance metrics.

This module provides functions for calculating rolling performance metrics
such as Sharpe ratios, Martin ratios, and information ratios over moving windows.
"""

import logging
import numpy as np
from numpy import isnan
from numpy.typing import NDArray
from math import sqrt

# Import from sibling modules
from functions.ta.moving_averages import SMA_2D, MoveMax_2D

logger = logging.getLogger(__name__)


def move_sharpe_2D(adjClose: NDArray[np.floating], dailygainloss: NDArray[np.floating], 
                   period: int) -> NDArray[np.floating]:
    """Compute the moving Sharpe ratio for multiple stocks.
    
    The Sharpe ratio measures risk-adjusted returns. This function calculates
    it over a rolling window for multiple stocks simultaneously.
    
    Formula: sharpe_ratio = (gmean(gains)^252 - 1) / (std(gains) * sqrt(252))
    Assumes 252 trading days per year.
    
    Args:
        adjClose: 2D array of shape (n_stocks, n_dates) with adjusted closing prices
        dailygainloss: 2D array of shape (n_stocks, n_dates) with daily gain/loss ratios
        period: Number of days in rolling window for Sharpe calculation
        
    Returns:
        NDArray[np.floating]: 2D array of moving Sharpe ratios
        
    Example:
        >>> prices = np.random.rand(10, 100)  # 10 stocks, 100 days
        >>> gains = prices[:, 1:] / prices[:, :-1]
        >>> gains_padded = np.ones_like(prices)
        >>> gains_padded[:, 1:] = gains
        >>> sharpe = move_sharpe_2D(prices, gains_padded, 60)
        
    Note:
        - Uses geometric mean for returns calculation
        - Annualizes both returns and volatility (252 trading days)
        - NaN gain/loss values are treated as 1.0 (no change)
        - Zero Sharpe values are replaced with 0.05 to avoid division issues
        
    References:
        Sharpe, William F. (1966). "Mutual Fund Performance". Journal of Business.
    """
    from scipy.stats import gmean

    sharpe = np.zeros((adjClose.shape[0], adjClose.shape[1]), dtype=float)
    for i in range(dailygainloss.shape[1]):
        minindex = max(i-period, 0)
        if i > minindex:
            sharpeValues = dailygainloss[:, minindex:i+1]
            sharpeValues[np.isnan(sharpeValues)] = 1.0
            numerator = gmean(sharpeValues, axis=-1)**252 - 1.
            denominator = np.std(sharpeValues, axis=-1)*sqrt(252)
            denominator[denominator == 0.] = 1.e-5
            sharpe[:, i] = numerator / denominator
        else:
            sharpe[:, i] = 0.

    sharpe[sharpe == 0] = .05
    sharpe[isnan(sharpe)] = .05

    return sharpe


def move_martin_2D(adjClose: NDArray[np.floating], period: int) -> NDArray[np.floating]:
    """Compute the moving Martin ratio (Ulcer Performance Index) for multiple stocks.
    
    The Martin ratio is based on the Ulcer Index, which measures downside volatility
    using RMS drawdown. It provides a more conservative risk measure than standard
    deviation as it only penalizes downside moves.
    
    Formula: Martin ratio uses RMS drawdown instead of standard deviation
    Reference: http://www.tangotools.com/ui/ui.htm
    
    Args:
        adjClose: 2D array of shape (n_stocks, n_dates) with adjusted closing prices
        period: Number of days in rolling window for Martin ratio calculation
        
    Returns:
        NDArray[np.floating]: 2D array of Martin ratios (RMS drawdown values)
        
    Example:
        >>> prices = np.random.rand(10, 100)  # 10 stocks, 100 days
        >>> martin = move_martin_2D(prices, 60)
        
    Note:
        - Computes drawdown from rolling maximum price
        - Uses RMS (root mean square) of drawdowns
        - NaN values are reset to zero
        - Lower Martin ratio indicates less downside volatility
        
    References:
        Martin, Peter (1987). "The Investor's Guide to Fidelity Funds"
    """
    MoveMax = MoveMax_2D(adjClose, period)
    pctDrawDown = adjClose / MoveMax - 1.
    pctDrawDown = pctDrawDown ** 2

    martin = np.sqrt(SMA_2D(pctDrawDown, period))

    # reset NaN's to zero
    martin[np.isnan(martin)] = 0.

    return martin


def move_informationRatio(dailygainloss_portfolio: NDArray[np.floating], 
                          dailygainloss_index: NDArray[np.floating], 
                          period: int) -> NDArray[np.floating]:
    """Compute the moving information ratio for portfolios vs benchmark.
    
    The information ratio measures excess returns relative to a benchmark
    per unit of tracking error. It's useful for evaluating active management.
    
    Formula: information_ratio = ExcessReturn / TrackingError
    where:
        ExcessReturn = mean(portfolio_returns - index_returns)
        TrackingError = RMS(portfolio_returns - index_returns)
    
    Assumes 252 trading days per year.
    
    Args:
        dailygainloss_portfolio: 2D array (n_portfolios, n_dates) of portfolio daily gains
        dailygainloss_index: 1D array (n_dates,) of benchmark index daily gains
        period: Number of days in rolling window for calculation
        
    Returns:
        NDArray[np.floating]: 2D array of information ratios
        
    Example:
        >>> portfolio_gains = np.random.rand(5, 100)  # 5 portfolios, 100 days
        >>> index_gains = np.random.rand(100)
        >>> info_ratio = move_informationRatio(portfolio_gains, index_gains, 60)
        
    Note:
        - Higher information ratio indicates better risk-adjusted outperformance
        - NaN values are treated as 0.0
        - Typical good values are above 0.5
        
    References:
        Grinold, Richard C. (1989). "The Fundamental Law of Active Management"
    """
    from bottleneck import nanmean
    from functions.ta.utils import nanrms

    infoRatio = np.zeros((dailygainloss_portfolio.shape[0], dailygainloss_portfolio.shape[1]), dtype=float)

    for i in range(dailygainloss_portfolio.shape[1]):
        minindex = max(i-period, 0)

        if i > minindex:
            returns_portfolio = dailygainloss_portfolio[:, minindex:i+1] - 1.
            returns_index = dailygainloss_index[minindex:i+1] - 1.
            excessReturn = nanmean(returns_portfolio - returns_index, axis=-1)
            trackingError = nanrms(dailygainloss_portfolio[:, minindex:i+1] - dailygainloss_index[minindex:i+1], axis=-1)

            infoRatio[:, i] = excessReturn / trackingError

            if i == dailygainloss_portfolio.shape[1]-1:
                print(" returns_portfolio = ", returns_portfolio)
                print(" returns_index = ", returns_index)
                print(" excessReturn = ", excessReturn)
                print(" infoRatio[:,i] = ", infoRatio[:, i])
        else:
            infoRatio[:, i] *= 0.

    infoRatio[infoRatio == 0] = .0
    infoRatio[isnan(infoRatio)] = .0

    return infoRatio
