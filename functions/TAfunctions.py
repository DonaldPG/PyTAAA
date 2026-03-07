"""Technical analysis and signal computation functions.

This module contains the core technical analysis functions for PyTAAA,
including signal generation, ranking algorithms, moving averages, and
portfolio optimization routines.

Key Functions:
    computeSignal2D: Generate buy/sell signals from price data
    sharpeWeightedRank_2D: Rank stocks by Sharpe ratio
    interpolate: Fill missing values with linear interpolation
    cleantobeginning: Forward-fill missing values at start of series
    cleantoend: Backward-fill missing values at end of series
    strip_accents: Remove accent marks from Unicode text
    normcorrcoef: Compute normalized correlation coefficient
    
The module uses matplotlib for plotting and supports both interactive
and non-interactive (Agg) backends depending on display availability.
"""

import os
import numpy as np
from numpy import isnan
import datetime
from typing import Any, Optional, Tuple, Union
from numpy.typing import NDArray
#from yahooFinance import getQuote
from functions.quotes_adjClose import get_pe
# from functions.readSymbols import readSymbolList
from functions.readSymbols import read_symbols_list_local
from functions.GetParams import get_webpage_store, get_performance_store


class TradingConstants:
    """
    Trading day constants used for performance calculations.

    These represent standard trading day counts for various time periods,
    based on approximately 252 trading days per year.
    """

    TRADING_DAYS_PER_YEAR: int = 252
    TRADING_DAYS_1_YEAR: int = 252
    TRADING_DAYS_2_YEARS: int = 504
    TRADING_DAYS_3_YEARS: int = 756
    TRADING_DAYS_5_YEARS: int = 1260


def format_ranking_line(
    symbol: str,
    rank: int,
    weight: float,
    price: float,
    trend: str,
    company_name: str = ""
) -> str:
    """Format a single line for ranking output with standardized column widths.
    
    Args:
        symbol: Stock symbol (6 chars, left-justified)
        rank: Ranking position (6 chars, integer)
        weight: Portfolio weight (6 chars, 4 decimals)
        price: Stock price (8 chars, 2 decimals with comma separator)
        trend: Trend indicator ('up  ' or 'down')
        company_name: Company name (15 chars, left-justified)
    
    Returns:
        Formatted string with aligned columns
    """
    return (f"{symbol:<6s}  {rank:6d}  {weight:6.4f}  "
            f"{price:8,.2f}  {trend}  {company_name:15s}")


#############################################################################
# NOTE: Phase 5+ Refactoring - Functions re-exported from functions/ta/*
# 
# Common technical analysis functions have been extracted to modular subpackages
# and are re-exported here for backward compatibility.
#
# - functions/ta/utils.py: strip_accents, normcorrcoef, nanrms
# - functions/ta/data_cleaning.py: interpolate, cleantobeginning, cleantoend, etc.
# - functions/ta/moving_averages.py: SMA, SMA_2D, hma, MoveMax, MoveMin, etc.
# - functions/ta/channels.py: percentileChannel, dpgchannel (both 1D and 2D)
# - functions/ta/signal_generation.py: computeSignal2D
# - functions/ta/rolling_metrics.py: move_sharpe_2D, move_martin_2D, etc.
#
# New code should import directly from functions.ta.* modules for clarity.
#############################################################################

# Re-export functions from modular ta/ subpackage
from functions.ta.utils import (
    strip_accents,
    normcorrcoef,
    nanrms
)

from functions.ta.data_cleaning import (
    interpolate,
    cleantobeginning,
    cleantoend,
    clean_signal,
    cleanspikes,
    despike_2D
)

from functions.ta.moving_averages import (
    SMA,
    SMA_2D,
    SMS,
    hma,
    hma_pd,
    SMA_filtered_2D,
    MoveMax,
    MoveMax_2D,
    MoveMin
)

from functions.ta.channels import (
    percentileChannel,
    percentileChannel_2D,
    dpgchannel,
    dpgchannel_2D
)

from functions.ta.signal_generation import (
    computeSignal2D
)

from functions.ta.rolling_metrics import (
    move_sharpe_2D,
    move_martin_2D,
    move_informationRatio
)

#############################################################################

#############################################################################
# Unique Functions (not re-exported from ta/ modules)
#############################################################################

#----------------------------------------------
def selfsimilarity(hi: np.ndarray, lo: np.ndarray) -> np.ndarray:
    """Compute a self-similarity metric for a price bar series.

    Calculates a rolling measure of how self-similar recent price bar
    ranges are relative to the 10-day high–low spread.  The result is
    expressed as a moving percentile rank over the same rolling window,
    making it dimensionless and comparable across instruments.

    Args:
        hi: 1D array of intraday high prices.
        lo: 1D array of intraday low prices corresponding to ``hi``.

    Returns:
        1D NumPy array of percentile-rank values in [0, 100] representing
        self-similarity at each date.

    Notes:
        - Internally computes a 10-day sum of (hi - lo), normalises by
          the 10-day high–low spread, smooths with a 60-day SMA, and
          finally converts to a rolling percentile rank.
        - Values near 100 indicate that recent bar ranges are large
          relative to historical ranges (high self-similarity in trend).
    """
    HminusL = hi-lo

    periods = 10
    SMS = np.zeros( (hi.shape[0]), dtype=float)
    for i in range( hi.shape[0] ):
        minx = max(0,i-periods)
        SMS[i] = np.sum(HminusL[minx:i+1],axis=-1)

    # find the 10-day range (incl highest high and lowest low)
    range10day = MoveMax(hi,10) - MoveMin(lo,10)

    # normalize
    SMS /= range10day

    # compute quarterly (60-day) SMA
    SMS = SMA(SMS,60)

    # find percentile rank
    movepctrank = np.zeros( (hi.shape[0]), dtype=float)
    for i in range( hi.shape[0] ):
        minx = max(0,i-periods)
        movepctrank[i] = percentileofscore(SMS[minx:i+1],SMS[i])

    return movepctrank

#----------------------------------------------
def jumpTheChannelTest(
        x: np.ndarray,
        minperiod: int = 4,
        maxperiod: int = 12,
        incperiod: int = 3,
        numdaysinfit: int = 28,
        offset: int = 3
) -> Tuple[float, float, float, float]:
    """Test whether the most recent price has jumped outside its trend channel.

    Computes a linear trend through the upper and lower price channels
    over a recent fitting window (with a forward offset) and measures
    where the current price falls within that channel.  Also computes
    the cumulative gain/loss and volatility over the fitting period.

    Args:
        x: 1D array of price data for a single instrument.
        minperiod: Minimum period (in days) for channel calculation.
            Defaults to 4.
        maxperiod: Maximum period (in days) for channel calculation.
            Defaults to 12.
        incperiod: Step size between min and max periods. Defaults to 3.
        numdaysinfit: Number of historical days used to fit the linear
            trend through the channels. Defaults to 28.
        offset: Number of days by which the fitting window is offset
            back from the current date.  Acts as a small gap to avoid
            look-ahead. Defaults to 3.

    Returns:
        A 4-tuple of:
            - pctChannel (float): Position of the current price within
              the channel as a fraction; >1 means above, <0 means below.
            - gainloss_cumu (float): Cumulative multiplicative gain/loss
              over the fitting period minus 1.
            - gainloss_std (float): Standard deviation of daily
              gain/loss values over the fitting period.
            - numStdDevs (float): Current price expressed as the number
              of standard deviations above or below the channel midpoint.
    """

    # calculate dpgchannel for all stocks in x
    # - x[stock_number,date]
    # - 'numdaysinfit' describes number of days over which to calculate a linear trend
    # - 'offset'  describes number days to forecast channel trends forward

    # import warnings
    # warnings.simplefilter('ignore', np.RankWarning)

    pctChannel = np.zeros( (x.shape[0]), 'float' )
    # calculate linear trend over 'numdaysinfit' with 'offset'
    minchannel,maxchannel = dpgchannel(x,minperiod,maxperiod,incperiod)
    minchannel_trenddata = minchannel[-(numdaysinfit+offset):-offset]
    regression = np.polyfit(list(range(-(numdaysinfit+offset),-offset)), minchannel_trenddata, 1)
    minchannel_trend = regression[-1]
    maxchannel_trenddata = maxchannel[-(numdaysinfit+offset):-offset]
    regression = np.polyfit(list(range(-(numdaysinfit+offset),-offset)), maxchannel_trenddata, 1)
    maxchannel_trend = regression[-1]
    pctChannel = (x[-1]-minchannel_trend) / (maxchannel_trend-minchannel_trend)

    # calculate the stdev over the period
    gainloss_period = x[-(numdaysinfit+offset)+1:-offset+1] / x[-(numdaysinfit+offset):-offset]
    gainloss_period[np.isnan(gainloss_period)] = 1.
    gainloss_cumu = np.cumprod( gainloss_period )[-1] -1.
    gainloss_std = np.std( gainloss_period )

    # calculate the current quote as number of stdevs above or below trend
    currentMidChannel = (maxchannel_trenddata+minchannel_trend)/2.
    numStdDevs = (x[-1]/currentMidChannel[-1]-1.) / gainloss_std

    '''
    print "pctChannel = ", pctChannel
    print "gainloss_period = ", gainloss_period
    print "gainloss_cumu = ", gainloss_cumu
    print "gainloss_std = ", gainloss_std
    print "currentMidChannel = ", currentMidChannel[-1]
    print "numStdDevs = ", numStdDevs
    '''

    return pctChannel, gainloss_cumu, gainloss_std, numStdDevs

#----------------------------------------------
def recentChannelFit(
        x: np.ndarray,
        minperiod: int = 4,
        maxperiod: int = 12,
        incperiod: int = 3,
        numdaysinfit: int = 28,
        offset: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit linear trends to the recent upper and lower price channels.

    Computes the upper and lower price channels via :func:`dpgchannel`
    and then fits first-order polynomials through the channel values
    over a recent fitting window.  Supports both gapped (``offset > 0``)
    and non-gapped (``offset == 0``) fits.

    Args:
        x: 1D array of price data for a single instrument.
        minperiod: Minimum period (in days) for channel calculation.
            Defaults to 4.
        maxperiod: Maximum period (in days) for channel calculation.
            Defaults to 12.
        incperiod: Step size between min and max periods. Defaults to 3.
        numdaysinfit: Number of historical days used for the linear
            regression. Defaults to 28.
        offset: Number of days by which the fitting window is offset
            back from the current date.  Use 0 for a no-gap fit.
            Defaults to 3.

    Returns:
        A 2-tuple of 1D NumPy arrays ``(lower_fit, upper_fit)``, each
        containing the polynomial coefficients ``[slope, intercept]``
        for the lower and upper channel trend lines respectively.
    """

    # calculate dpgchannel for all stocks in x
    # - x[stock_number,date]
    # - 'numdaysinfit' describes number of days over which to calculate a linear trend
    # - 'offset'  describes number days to forecast channel trends forward

    # import warnings
    # warnings.simplefilter('ignore', np.RankWarning)

    ##pctChannel = np.zeros( (x.shape[0]), 'float' )
    # calculate linear trend over 'numdaysinfit' with 'offset'
    minchannel,maxchannel = dpgchannel(x,minperiod,maxperiod,incperiod)
    if offset == 0:
        minchannel_trenddata = minchannel[-(numdaysinfit+offset):]
        '''
        print "numdaysinfit = ", numdaysinfit
        print "offset = ", offset
        print "len(x) = ", len(x)
        print "len(minchannel) = ", len(minchannel)
        print "most recent quote = ", x[-1]
        print "quote[-offset:] = ", x[-offset:]
        print "quote[-(numdaysinfit+offset)+1:-offset+1] = ", x[-(numdaysinfit+offset):]
        print "len(quote[-(numdaysinfit+offset)+1:-offset+1]) = ", len(x[-(numdaysinfit+offset):])
        print "length of days = ", len(range(-(numdaysinfit+offset)+1,-offset+1))
        print "relative days = ", range(-(numdaysinfit+offset)+1,-offset+1)
        print "length of quotes = ", len(minchannel_trenddata)
        print "quotes = ", minchannel_trenddata
        '''
        regression1 = np.polyfit(list(range(-(numdaysinfit+offset)+1,-offset+1)), minchannel_trenddata, 1)
        minchannel_trend = regression1[-1]
        maxchannel_trenddata = maxchannel[-(numdaysinfit+offset):]
        regression2 = np.polyfit(list(range(-(numdaysinfit+offset)+1,-offset+1)), maxchannel_trenddata, 1)
    else:
        minchannel_trenddata = minchannel[-(numdaysinfit+offset)+1:-offset+1]
        '''
        print "numdaysinfit = ", numdaysinfit
        print "offset = ", offset
        print "len(x) = ", len(x)
        print "len(minchannel) = ", len(minchannel)
        print "most recent quote = ", x[-1]
        print "quote[-offset:] = ", x[-offset:]
        print "quote[-(numdaysinfit+offset)+1:-offset+1] = ", x[-(numdaysinfit+offset)+1:-offset+1]
        print "len(quote[-(numdaysinfit+offset)+1:-offset+1]) = ", len(x[-(numdaysinfit+offset)+1:-offset+1])
        print "length of days = ", len(range(-(numdaysinfit+offset)+1,-offset+1))
        print "relative days = ", range(-(numdaysinfit+offset)+1,-offset+1)
        print "length of quotes = ", len(minchannel_trenddata)
        print "quotes = ", minchannel_trenddata
        '''
        regression1 = np.polyfit(list(range(-(numdaysinfit+offset)+1,-offset+1)), minchannel_trenddata, 1)
        minchannel_trend = regression1[-1]
        maxchannel_trenddata = maxchannel[-(numdaysinfit+offset)+1:-offset+1]
        regression2 = np.polyfit(list(range(-(numdaysinfit+offset)+1,-offset+1)), maxchannel_trenddata, 1)
    ##maxchannel_trend = regression2[-1]
    ##pctChannel = (x[-1]-minchannel_trend) / (maxchannel_trend-minchannel_trend)

    return regression1, regression2

#----------------------------------------------
def recentTrendAndStdDevs(
        x: np.ndarray,
        datearray: np.ndarray,
        minperiod: int = 4,
        maxperiod: int = 12,
        incperiod: int = 3,
        numdaysinfit: int = 28,
        offset: int = 3
) -> Tuple[float, float, float]:
    """Compute gain, channel-relative position, and dispersion for recent prices.

    Fits a linear trend through the upper and lower price channels over
    a recent gapped window and summarises where the current price sits
    relative to that trend.

    Args:
        x: 1D array of price data for a single instrument.
        datearray: 1D array of dates corresponding to ``x``.
        minperiod: Minimum period (in days) for channel calculation.
            Defaults to 4.
        maxperiod: Maximum period (in days) for channel calculation.
            Defaults to 12.
        incperiod: Step size between min and max periods. Defaults to 3.
        numdaysinfit: Number of historical days used for the linear
            regression. Defaults to 28.
        offset: Number of days by which the fitting window is offset
            back from the current date. Defaults to 3.

    Returns:
        A 3-tuple of:
            - gainloss_cumu (float): Cumulative gain/loss of the trend
              midpoint over the fitting period minus 1.
            - numStdDevs (float): Current price expressed as the number
              of half-channel widths above (positive) or below (negative)
              the trend midpoint.
            - pctChannel (float): Current price position relative to
              the upper channel boundary.
    """

    # calculate dpgchannel for all stocks in x
    # - x[stock_number,date]
    # - 'numdaysinfit' describes number of days over which to calculate a linear trend
    # - 'offset'  describes number days to forecast channel trends forward

    # fit short-term recent trend channel for plotting
    lowerFit, upperFit = recentChannelFit( x,
                                           minperiod=minperiod,
                                           maxperiod=maxperiod,
                                           incperiod=incperiod,
                                           numdaysinfit=numdaysinfit,
                                           offset=offset)
    recentFitDates = datearray[-numdaysinfit-offset:-offset+1]
    relativedates = list(range(-numdaysinfit-offset,-offset+1))
    p = np.poly1d(upperFit)
    upperTrend = p(relativedates)
    currentUpper = p(0) * 1.
    p = np.poly1d(lowerFit)
    lowerTrend = p(relativedates)
    currentLower = p(0) * 1.
    midTrend = (upperTrend+lowerTrend)/2.
    #residuals = x[-numdaysinfit-offset:-offset+1] - midTrend
    #fitStdDev = np.std(residuals)
    fitStdDev = np.mean( upperTrend - lowerTrend )/2.
    #print ".....lowerFit, upperFit = ", lowerFit, upperFit
    #print ".....fitStdDev,currentUpper,currentLower,x[-1] = ", fitStdDev, currentUpper,currentLower,x[-1]
    currentResidual = x[-1] - (currentUpper + currentLower)/2.
    if fitStdDev != 0.:
        numStdDevs = currentResidual / fitStdDev
    else:
        numStdDevs = 0.

    # calculate gain or loss over the period
    gainloss_period = x[-(numdaysinfit+offset)+1:-offset+1] / x[-(numdaysinfit+offset):-offset]
    gainloss_period[np.isnan(gainloss_period)] = 1.
    gainloss_cumu = np.cumprod( gainloss_period )[-1] -1.

    # different method for gainloss over period using slope
    gainloss_cumu = midTrend[-1] / midTrend[0] -1.

    if currentUpper != currentLower:
        pctChannel = (x[-1]-currentUpper) / (currentUpper-currentLower)
    else:
        pctChannel = 0.

    return gainloss_cumu, numStdDevs, pctChannel


def recentSharpeWithAndWithoutGap(
        x: np.ndarray,
        numdaysinfit: int = 504,
        offset_factor: float = .4
) -> float:
    """Compute a multi-period Sharpe ratio with and without recent-data gaps.

    Iteratively halves the lookback window starting from ``numdaysinfit``
    days, computing the Sharpe ratio for both a gapped sub-period (the
    first 60 % of days) and a non-gapped sub-period (the most recent
    portion), then combines all period Sharpe ratios into a single scalar
    using a weighted angular combination.

    Args:
        x: 1D array of price data for a single instrument.
        numdaysinfit: Starting lookback period in days. Defaults to 504.
        offset_factor: Fraction of the lookback period used as the gap
            (i.e., the number of recent days to skip). Defaults to 0.4.

    Returns:
        A scalar Sharpe-ratio combination that blends multiple lookback
        periods, giving slightly more weight to shorter (more recent)
        windows via an angular combination at 33 degrees.

    Notes:
        - The iteration terminates when the halved window drops below
          20 days.
        - NaN Sharpe values in intermediate periods are replaced with
          0.0; the final period NaN is replaced with -999 as a sentinel.
        - Print statements are used to trace each iteration.
    """

    # calculate dpgchannel for all stocks in x
    # - x[stock_number,date]
    # - 'numdaysinfit' describes number of days over which to calculate sharpe with a gap
    # - 'numdaysinfit2' describes number of days over which to calculate sharpe without a gap
    # - 'offset'  describes number recent days to skip (e.g. the gap)

    # calculate number of loops
    sharpeList = []
    for i in range(1,25):
        if i == 1:
            numdaysStart = numdaysinfit
            numdaysEnd = int(numdaysStart * offset_factor + .5)
        else:
            numdaysStart /= 2
            if numdaysStart/2 > 20:
                numdaysEnd = int(numdaysStart * offset_factor + .5)
            else:
                numdaysEnd = 0

        # calculate gain or loss over the gapped period
        numdaysStart = int(numdaysStart)
        numdaysEnd = int(numdaysEnd)
        numdays = numdaysStart - numdaysEnd
        offset = numdaysEnd
        if offset > 0:
            print("i,start,end = ", i, -(numdays+offset)+1, -offset+1)
            gainloss_period = x[-(numdays+offset)+1:-offset+1] / x[-(numdays+offset):-offset]
            gainloss_period[np.isnan(gainloss_period)] = 1.

            # sharpe ratio in period with a gap
            sharpe = ( gmean(gainloss_period)**252 -1. ) / ( np.std(gainloss_period)*sqrt(252) )
        else:
            print("i,start,end = ", i, -numdays+1, 0)
            # calculate gain or loss over the period without a gap
            gainloss_period = x[-numdays+1:] / x[-numdays:-1]
            gainloss_period[np.isnan(gainloss_period)] = 1.

            # sharpe ratio in period wihtout a gap
            sharpe = ( gmean(gainloss_period)**252 -1. ) / ( np.std(gainloss_period)*sqrt(252) )
        sharpeList.append(sharpe)
        if numdaysStart/2 < 20:
            break

    print("sharpeList = ", sharpeList)
    sharpeList = np.array(sharpeList)
    for i,isharpe in enumerate(sharpeList):
        if i == len(sharpeList)-1:
            if np.isnan(isharpe):
                sharpeList[i] = -999.
        else:
            if np.isnan(isharpe) or isharpe == 0.:
                sharpeList[i] = 0.
    print("sharpeList = ", sharpeList)

    crossplot_rotationAngle = 33. * np.pi/180.
    for i,isharpe in enumerate(sharpeList):
        # combine sharpe ratios compouted over 2 different periods
        # - use an angle of 33 degrees instead of 45 to give slightly more weight the the "no gap" sharpe
        if i==0:
            continue
        elif i==1:
            sharpe_pair = [sharpeList[i-1],sharpeList[i]]
        else:
            sharpe_pair = [sharpe2periods,sharpeList[i]]
        sharpe2periods = sharpe_pair[0]*np.sin(crossplot_rotationAngle) + sharpe_pair[1]*np.cos(crossplot_rotationAngle)
        print("i, sharpe_pair, combined = " + str((i, sharpe_pair, sharpe2periods)))

    return sharpe2periods

#----------------------------------------------

def recentTrendAndMidTrendChannelFitWithAndWithoutGap(
        x: np.ndarray,
        minperiod: int = 4,
        maxperiod: int = 12,
        incperiod: int = 3,
        numdaysinfit: int = 28,
        numdaysinfit2: int = 20,
        offset: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit channel trends over two windows — one gapped and one recent.

    Computes two sets of linear trend lines for the price channel:
    a longer gapped window (to identify the prevailing trend) and a
    shorter no-gap window (to capture the most recent momentum).  The
    two midpoint trend lines can be compared to detect short-term
    deviations from the longer-term trend.

    Args:
        x: 1D array of price data for a single instrument.
        minperiod: Minimum period (in days) for channel calculation.
            Defaults to 4.
        maxperiod: Maximum period (in days) for channel calculation.
            Defaults to 12.
        incperiod: Step size between min and max periods. Defaults to 3.
        numdaysinfit: Number of days for the gapped (longer) fitting
            window. Defaults to 28.
        numdaysinfit2: Number of days for the no-gap (shorter) fitting
            window. Defaults to 20.
        offset: Number of recent days excluded from the gapped fit to
            avoid look-ahead. Defaults to 3.

    Returns:
        A 4-tuple of 1D NumPy arrays:
            - lowerTrend: Lower channel values evaluated over the gapped
              fitting window.
            - upperTrend: Upper channel values evaluated over the gapped
              fitting window.
            - NoGapLowerTrend: Lower channel values evaluated over the
              no-gap (recent) window.
            - NoGapUpperTrend: Upper channel values evaluated over the
              no-gap (recent) window.
    """

    # calculate dpgchannel for all stocks in x
    # - x[stock_number,date]
    # - 'numdaysinfit' describes number of days over which to calculate a linear trend
    # - 'offset'  describes number days to forecast channel trends forward

    # fit short-term recent trend channel with offset from current date for plotting
    gappedLowerFit, gappedUpperFit = recentChannelFit( x,
                                           minperiod=minperiod,
                                           maxperiod=maxperiod,
                                           incperiod=incperiod,
                                           numdaysinfit=numdaysinfit,
                                           offset=offset)
    #recentFitDates = datearray[-numdaysinfit-offset:-offset+1]
    relativedates = list(range(-numdaysinfit-offset,-offset+1))
    p = np.poly1d(gappedUpperFit)
    upperTrend = p(relativedates)
    currentUpper = p(0) * 1.
    p = np.poly1d(gappedLowerFit)
    lowerTrend = p(relativedates)
    currentLower = p(0) * 1.
    midTrend = (upperTrend+lowerTrend)/2.
    #residuals = x[-numdaysinfit-offset:-offset+1] - midTrend
    #fitStdDev = np.std(residuals)
    fitStdDev = np.mean( upperTrend - lowerTrend )/2.
    #print ".....gappedLowerFit, gappedUpperFit = ", gappedLowerFit, gappedUpperFit
    #print ".....fitStdDev,currentUpper,currentLower,x[-1] = ", fitStdDev, currentUpper,currentLower,x[-1]
    currentResidual = x[-1] - (currentUpper + currentLower)/2.
    if fitStdDev != 0.:
        numStdDevs = currentResidual / fitStdDev
    else:
        numStdDevs = 0.

    # calculate gain or loss over the period (with offset)
    gainloss_period = x[-(numdaysinfit+offset)+1:-offset+1] / x[-(numdaysinfit+offset):-offset]
    gainloss_period[np.isnan(gainloss_period)] = 1.
    gainloss_cumu = np.cumprod( gainloss_period )[-1] -1.

    if currentUpper!=currentLower:
        pctChannel = (x[-1]-currentUpper) / (currentUpper-currentLower)
    else:
        pctChannel = 0.

    # fit shorter trend without offset
    NoGapLowerFit, NoGapUpperFit = recentChannelFit( x,
                                           minperiod=minperiod,
                                           maxperiod=maxperiod,
                                           incperiod=incperiod,
                                           numdaysinfit=numdaysinfit2,
                                           offset=0)
    #recentFitDates = datearray[-numdaysinfit2:]
    relativedates = list(range(-numdaysinfit2+1,1))
    p = np.poly1d(NoGapUpperFit)
    NoGapUpperTrend = p(relativedates)
    NoGapCurrentUpper = p(0) * 1.
    p = np.poly1d(NoGapLowerFit)
    NoGapLowerTrend = p(relativedates)
    NoGapCurrentLower = p(0) * 1.
    NoGapMidTrend = (NoGapUpperTrend+NoGapLowerTrend)/2.

    return lowerTrend, upperTrend, NoGapLowerTrend, NoGapUpperTrend

#----------------------------------------------

def recentTrendAndMidTrendWithGap(
        x: np.ndarray,
        datearray: np.ndarray,
        minperiod: int = 4,
        maxperiod: int = 12,
        incperiod: int = 3,
        numdaysinfit: int = 28,
        numdaysinfit2: int = 20,
        offset: int = 3
) -> Tuple[float, float, float, float]:
    """Compute trend metrics using two channel fits and generate a diagnostic plot.

    Fits price-channel linear trends over a gapped window and a shorter
    no-gap window, summarises the trend metrics, and renders a matplotlib
    plot showing both channel fits.  This function is primarily used for
    diagnostic and visualisation purposes.

    Args:
        x: 1D array of price data for a single instrument.
        datearray: 1D array of dates corresponding to ``x``.
        minperiod: Minimum period (in days) for channel calculation.
            Defaults to 4.
        maxperiod: Maximum period (in days) for channel calculation.
            Defaults to 12.
        incperiod: Step size between min and max periods. Defaults to 3.
        numdaysinfit: Number of days for the gapped (longer) window.
            Defaults to 28.
        numdaysinfit2: Number of days for the no-gap (shorter) window.
            Defaults to 20.
        offset: Days excluded from the end of the gapped window.
            Defaults to 3.

    Returns:
        A 4-tuple of:
            - gainloss_cumu (float): Cumulative gain/loss of the gapped
              midpoint trend over the fitting period minus 1.
            - gainloss_cumu2 (float): Gain/loss of the no-gap midpoint
              relative to the start of the gapped midpoint.
            - numStdDevs (float): Current price in half-channel-width
              units above or below the gapped midpoint.
            - relative_GainLossRatio (float): Ratio of the no-gap
              channel midpoint to the gapped channel midpoint at the
              current date.

    Notes:
        Uses the ``Agg`` Matplotlib backend to allow rendering in
        headless environments.  The plot is displayed with
        ``plt.show()`` which may be a no-op in non-interactive sessions.
    """

    # calculate dpgchannel for all stocks in x
    # - x[stock_number,date]
    # - 'numdaysinfit' describes number of days over which to calculate a linear trend
    # - 'offset'  describes number days to forecast channel trends forward

    # fit short-term recent trend channel with offset from current date for plotting
    gappedLowerFit, gappedUpperFit = recentChannelFit( x,
                                           minperiod=minperiod,
                                           maxperiod=maxperiod,
                                           incperiod=incperiod,
                                           numdaysinfit=numdaysinfit,
                                           offset=offset)
    recentFitDates = datearray[-numdaysinfit-offset:-offset+1]
    relativedates = list(range(-numdaysinfit-offset,-offset+1))
    p = np.poly1d(gappedUpperFit)
    upperTrend = p(relativedates)
    currentUpper = p(0) * 1.
    p = np.poly1d(gappedLowerFit)
    lowerTrend = p(relativedates)
    currentLower = p(0) * 1.
    midTrend = (upperTrend+lowerTrend)/2.

    fitStdDev = np.mean( upperTrend - lowerTrend )/2.
    currentResidual = x[-1] - (currentUpper + currentLower)/2.
    numStdDevs = currentResidual / fitStdDev

    # calculate gain or loss over the period (with offset)
    gainloss_period = x[-(numdaysinfit+offset)+1:-offset+1] / x[-(numdaysinfit+offset):-offset]
    gainloss_period[np.isnan(gainloss_period)] = 1.
    gainloss_cumu = np.cumprod( gainloss_period )[-1] -1.

    pctChannel = (x[-1]-currentUpper) / (currentUpper-currentLower)

    # fit shorter trend without offset
    NoGapLowerFit, NoGapUpperFit = recentChannelFit(
        x,
        minperiod=minperiod,
        maxperiod=maxperiod,
        incperiod=incperiod,
        numdaysinfit=numdaysinfit2,
        offset=0
    )
    recentFitDates = datearray[-numdaysinfit2:]
    relativedates = list(range(-numdaysinfit2,1))
    p = np.poly1d(NoGapUpperFit)
    NoGapUpperTrend = p(relativedates)
    NoGapCurrentUpper = p(0) * 1.
    p = np.poly1d(NoGapLowerFit)
    NoGapLowerTrend = p(relativedates)
    NoGapCurrentLower = p(0) * 1.
    NoGapMidTrend = (NoGapUpperTrend+NoGapLowerTrend)/2.

    # calculate relative gain or loss over entire period
    gainloss_cumu2 = NoGapMidTrend[-1]/midTrend[0] -1.
    relative_GainLossRatio = (NoGapCurrentUpper + NoGapCurrentLower)/(currentUpper + currentLower)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pylab as plt
    plt.figure(1)
    plt.clf()
    plt.grid(True)
    plt.plot(datearray[-(numdaysinfit+offset+20):],x[-(numdaysinfit+offset+20):],'k-')
    relativedates = list(range(-numdaysinfit-offset,-offset+1))
    plt.plot(datearray[np.array(relativedates)],upperTrend,'y-')
    plt.plot(datearray[np.array(relativedates)],lowerTrend,'y-')
    plt.plot([datearray[-1]],[(upperTrend[-1]+lowerTrend[-1])/2.],'y.',ms=30)
    relativedates = list(range(-numdaysinfit2,0))
    plt.plot(datearray[np.array(relativedates)],NoGapUpperTrend,'c-')
    plt.plot(datearray[np.array(relativedates)],NoGapLowerTrend,'c-')
    plt.plot([datearray[-1]],[(NoGapUpperTrend[-1]+NoGapLowerTrend[-1])/2.],'c.',ms=30)
    plt.show()

    return gainloss_cumu, gainloss_cumu2, numStdDevs, relative_GainLossRatio

#----------------------------------------------

def recentTrendComboGain(
        x: np.ndarray,
        datearray: np.ndarray,
        minperiod: int = 4,
        maxperiod: int = 12,
        incperiod: int = 3,
        numdaysinfit: int = 28,
        numdaysinfit2: int = 20,
        offset: int = 3
) -> float:
    """Compute a combined annualised gain signal from two channel trend fits.

    Calculates annualised geometric mean gains for both a gapped and a
    no-gap price-channel midpoint trend, then combines them into a single
    "combo gain" that rewards recent acceleration relative to the
    prevailing trend.

    Args:
        x: 1D array of price data for a single instrument.
        datearray: 1D array of dates corresponding to ``x``.
        minperiod: Minimum period (in days) for channel calculation.
            Defaults to 4.
        maxperiod: Maximum period (in days) for channel calculation.
            Defaults to 12.
        incperiod: Step size between min and max periods. Defaults to 3.
        numdaysinfit: Number of days for the gapped (longer) window.
            Defaults to 28.
        numdaysinfit2: Number of days for the no-gap (shorter) window.
            Defaults to 20.
        offset: Days excluded from the end of the gapped window.
            Defaults to 3.

    Returns:
        A scalar ``comboGain`` value computed as the average of the two
        annualised gains, scaled by the ratio of the short-window gain
        to the long-window gain.  Higher values indicate stronger recent
        upward acceleration.
    """

    from scipy.stats import gmean

    # calculate dpgchannel for all stocks in x
    # - x[stock_number,date]
    # - 'numdaysinfit' describes number of days over which to calculate a linear trend
    # - 'offset'  describes number days to forecast channel trends forward

    # fit short-term recent trend channel with offset from current date for plotting
    gappedLowerFit, gappedUpperFit = recentChannelFit(
        x,
        minperiod=minperiod,
        maxperiod=maxperiod,
        incperiod=incperiod,
        numdaysinfit=numdaysinfit,
        offset=offset
    )
    recentFitDates = datearray[-numdaysinfit-offset:-offset+1]
    relativedates = list(range(-numdaysinfit-offset,-offset+1))
    p = np.poly1d(gappedUpperFit)
    upperTrend = p(relativedates)
    p = np.poly1d(gappedLowerFit)
    lowerTrend = p(relativedates)
    midTrend = (upperTrend+lowerTrend)/2.

    # calculate gain or loss over the period (no offset)
    gainloss_period = midTrend[1:] / midTrend[:-1]
    gainloss_period[np.isnan(gainloss_period)] = 1.
    gainloss_cumu = gmean( gainloss_period )**252 -1.

    # fit shorter trend without offset
    NoGapLowerFit, NoGapUpperFit = recentChannelFit(
        x,
        minperiod=minperiod,
        maxperiod=maxperiod,
        incperiod=incperiod,
        numdaysinfit=numdaysinfit2,
        offset=0
    )
    recentFitDates = datearray[-numdaysinfit2:]
    relativedates = list(range(-numdaysinfit2,1))
    p = np.poly1d(NoGapUpperFit)
    NoGapUpperTrend = p(relativedates)
    p = np.poly1d(NoGapLowerFit)
    NoGapLowerTrend = p(relativedates)
    NoGapMidTrend = (NoGapUpperTrend+NoGapLowerTrend)/2.

    # calculate gain or loss over the period (no offset)
    gainloss_period_nogap = NoGapMidTrend[1:] / NoGapMidTrend[:-1]
    gainloss_period_nogap[np.isnan(gainloss_period_nogap)] = 1.
    gainloss_cumu_nogap = gmean( gainloss_period_nogap )**252 -1.

    # calculate "combo gain" (defined as sum of gains rewarded for improvement, penalized for decline
    comboGain = (gainloss_cumu + gainloss_cumu_nogap)/2.
    comboGain *= (gainloss_cumu_nogap+1) / (gainloss_cumu+1)

    return comboGain

def sharpeWeightedRank_2D(
    json_fn: str,
    datearray: np.ndarray,
    symbols: list,
    adjClose: np.ndarray,
    signal2D: np.ndarray,
    signal2D_daily: np.ndarray,
    LongPeriod: int,
    numberStocksTraded: int,
    riskDownside_min: float,
    riskDownside_max: float,
    rankThresholdPct: float,
    stddevThreshold: float = 5.0,
    makeQCPlots: bool = False,
    # Parameters for backtest (refactored version).
    max_weight_factor: float = 3.0,
    min_weight_factor: float = 0.3,
    absolute_max_weight: float = 0.9,
    apply_constraints: bool = True,
    # Parameters for production (PortfolioPerformanceCalcs).
    is_backtest: bool = True,
    verbose: bool = False,
    stockList: str = "SP500",  # Add stockList parameter for early period logic
    **kwargs: Any  # Accept any additional keyword arguments for compatibility.
) -> np.ndarray:
    """Compute Sharpe-ratio-weighted portfolio allocation for all dates.

    Computes rolling Sharpe ratios for each stock and returns a 2D array
    of portfolio weights ``(n_stocks, n_days)``.  Stocks are selected
    based on: (1) having an uptrend signal (``signal2D > 0``), (2)
    ranking in the top N by Sharpe ratio, and (3) passing data-quality
    checks.  Weights are assigned proportionally to Sharpe ratios with
    optional constraints, and are forward-filled so every day has
    valid weights.

    Args:
        json_fn: Path to JSON configuration file (not used but kept
            for API compatibility).
        datearray: 1D array of dates corresponding to ``adjClose``
            columns.
        symbols: List of stock ticker symbols corresponding to rows
            of ``adjClose``.
        adjClose: 2D array of adjusted close prices
            ``(n_stocks, n_days)``.
        signal2D: 2D monthly uptrend signal array
            ``(n_stocks, n_days)``; 1 = uptrending.
        signal2D_daily: 2D daily uptrend signal array
            ``(n_stocks, n_days)`` (not used; kept for API
            compatibility).
        LongPeriod: Rolling lookback period in days for Sharpe ratio
            calculation.
        numberStocksTraded: Number of top-ranked stocks to include in
            the portfolio at each date (clamped to a maximum of 20).
        riskDownside_min: Minimum portfolio weight per position.
        riskDownside_max: Maximum portfolio weight per position.
        rankThresholdPct: Percentile threshold for rank filtering
            (currently unused; reserved for future use).
        stddevThreshold: Standard-deviation multiplier for spike
            detection. Defaults to 5.0.
        makeQCPlots: If ``True``, generate quality-control plots.
            Defaults to ``False``.
        max_weight_factor: Maximum weight as a multiple of the equal
            weight. Defaults to 3.0.
        min_weight_factor: Minimum weight as a multiple of the equal
            weight. Defaults to 0.3.
        absolute_max_weight: Absolute cap on any single position
            weight. Defaults to 0.9.
        apply_constraints: If ``True``, enforce weight constraints.
            Defaults to ``True``.
        is_backtest: If ``True``, run in backtest mode (uses
            historical data).  If ``False``, run in production mode.
            Defaults to ``True``.
        verbose: If ``True``, print detailed progress messages.
            Defaults to ``False``.
        stockList: Stock-list identifier (``"SP500"`` or
            ``"Naz100"``).  Early-period CASH logic only applies to
            ``"SP500"``. Defaults to ``"SP500"``.
        **kwargs: Additional keyword arguments accepted for forward
            compatibility.

    Returns:
        2D NumPy array of portfolio weights, shape
        ``(n_stocks, n_days)``.  Each column sums to 1.0.
    """
    from math import sqrt

    # Clamp numberStocksTraded to maximum of 20
    numberStocksTraded = min(numberStocksTraded, 20)

    n_stocks = adjClose.shape[0]
    n_days = adjClose.shape[1]

    print(" ... inside sharpeWeightedRank_2D (Sharpe-based selection) ...")
    print(f" ... n_stocks={n_stocks}, n_days={n_days}")
    print(f" ... LongPeriod={LongPeriod}")
    print(f" ... numberStocksTraded={numberStocksTraded} (clamped to max 20)")
    print(f" ... is_backtest={is_backtest}")
    print(f" ... max_weight_factor={max_weight_factor}")
    print(f" ... min_weight_factor={min_weight_factor}")
    print(f" ... stockList={stockList}")

    #########################################################################
    # Initialize output weight matrix.
    #########################################################################
    monthgainlossweight = np.zeros((n_stocks, n_days), dtype=float)

    #########################################################################
    # Compute daily gain/loss for Sharpe calculation.
    #########################################################################
    dailygainloss = np.ones((n_stocks, n_days), dtype=float)
    dailygainloss[:, 1:] = adjClose[:, 1:] / adjClose[:, :-1]
    dailygainloss[np.isnan(dailygainloss)] = 1.0
    dailygainloss[np.isinf(dailygainloss)] = 1.0

    # Create a binary mask from `signal2D` to ensure upstream filters
    # (like rolling_window_filter) that set zeros are respected.
    # Define it early so it's available to all subsequent logic.
    signal_mask = (signal2D > 0).astype(float)

    assert signal2D.shape == signal_mask.shape, (
        f"signal2D shape {signal2D.shape} != signal_mask shape {signal_mask.shape}"
    )
    assert signal2D_daily.shape == signal2D.shape, (
        f"signal2D_daily shape {signal2D_daily.shape} != signal2D shape {signal2D.shape}"
    )

    #########################################################################
    # Compute rolling Sharpe ratio for all stocks and all dates.
    #########################################################################
    print(" ... Computing rolling Sharpe ratios for all stocks...")
    sharpe_2d = np.zeros((n_stocks, n_days), dtype=float)

    # Rolling window size for dynamic threshold calculation.
    window_size = LongPeriod  # Use full LongPeriod for consistency with display

    for i in range(n_stocks):
        # Skip Sharpe calculation for CASH - it should always have Sharpe = 0.0
        if symbols[i] == 'CASH':
            sharpe_2d[i, :] = 0.0
            continue
            
        for j in range(window_size, n_days):
            # Extract window of returns (same as display function).
            start_idx = max(0, j - window_size)
            returns_window = dailygainloss[i, start_idx:j + 1] - 1.0

            # Skip if insufficient valid data.
            valid_returns = returns_window[~np.isnan(returns_window)]
            if len(valid_returns) < window_size // 2:
                sharpe_2d[i, j] = 0.0
                continue

            # Calculate mean and std of returns.
            mean_return = np.mean(valid_returns)
            std_return = np.std(valid_returns, ddof=1)

            # Calculate Sharpe ratio (annualized).
            if std_return > 0:
                sharpe_2d[i, j] = (mean_return / std_return) * sqrt(TradingConstants.TRADING_DAYS_PER_YEAR)
            else:
                sharpe_2d[i, j] = 0.0

    print(" ... Sharpe ratio calculation complete.")

    #########################################################################
    # For each date, select top stocks by Sharpe ratio and assign weights.
    #########################################################################
    print(" ... Selecting stocks and assigning weights...")

    for j in range(n_days):
        # Check if all signals are zero (no stocks have valid signals)
        # Use the binary mask so upstream filters (rolling window) are respected.
        all_signals_zero = np.sum(signal_mask[:, j] > 0) == 0
        
        if all_signals_zero:
            # Check if we're in the early period (2000-2002) where signals are expected to be zero
            # due to warm-up period. In this case, put everything in CASH.
            # This logic only applies to SP500 backtests, not Naz100.
            current_date = datearray[j]
            is_early_period = (current_date.year >= 2000 and current_date.year <= 2002) and (stockList == "SP500")
            
            if is_early_period:
                # For early period: assign weight 1.0 to CASH, 0.0 to all other stocks
                cash_idx = symbols.index('CASH') if 'CASH' in symbols else None
                if cash_idx is not None:
                    monthgainlossweight[:, j] = 0.0  # Zero all weights first
                    monthgainlossweight[cash_idx, j] = 1.0  # 100% in cash
                    if verbose:
                        print(
                            f" ... Date {datearray[j]}: "
                            "Early period (2000-2002), "
                            "all signals zero, assigning 100% to CASH"
                        )
                else:
                    # Fallback: equal weights to all stocks if no CASH symbol
                    equal_weight = 1.0 / n_stocks
                    monthgainlossweight[:, j] = equal_weight
                    if verbose:
                        print(
                            f" ... Date {datearray[j]}: "
                            "Early period but no CASH symbol, "
                            f"assigning equal weights to all {n_stocks} stocks"
                        )
            else:
                # Non-early period with all signals zero: the rolling window filter
                # has excluded all stocks (e.g. infilled prices). Assign 100% to
                # CASH to stay flat, or leave weights at 0.0 if CASH is absent.
                # Do NOT spread equal weight to all stocks - that would re-enable
                # filtered-out symbols (e.g. JEF during its 2015-2018 infill period).
                cash_idx = symbols.index('CASH') if 'CASH' in symbols else None
                if cash_idx is not None:
                    monthgainlossweight[:, j] = 0.0
                    monthgainlossweight[cash_idx, j] = 1.0
                    print(f" ... Date {datearray[j]}: All signals zero (non-early period), assigning 100% to CASH")
                else:
                    # monthgainlossweight[:, j] already 0.0 from initialization.
                    print(f" ... Date {datearray[j]}: All signals zero (non-early period), no CASH symbol, leaving weights at 0.0")
            continue

        # Get stocks with valid signals for this date (from mask).
        valid_signals = signal_mask[:, j] > 0
        valid_sharpe = ~np.isnan(sharpe_2d[:, j])

        # Combine signal and Sharpe criteria.
        eligible_stocks = valid_signals & valid_sharpe

        # Check for early period logic BEFORE checking if eligible stocks exist
        current_date = datearray[j]
        # Handle different date formats (datetime.date, datetime.datetime, numpy.datetime64)
        if hasattr(current_date, 'year'):
            year = current_date.year
        elif hasattr(current_date, 'item'):  # numpy datetime64
            year = current_date.item().year
        else:
            year = int(str(current_date)[:4])  # fallback for string dates
        
        is_early_period = (year >= 2000 and year <= 2002) and (stockList == "SP500")

        if np.sum(eligible_stocks) == 0:
            # No eligible stocks found
            if is_early_period:
                # For early period: assign weight 1.0 to CASH, 0.0 to all other stocks
                cash_idx = symbols.index('CASH') if 'CASH' in symbols else None
                if cash_idx is not None:
                    monthgainlossweight[:, j] = 0.0  # Zero all weights first
                    monthgainlossweight[cash_idx, j] = 1.0  # 100% in cash
                    if verbose:
                        print(
                            f" ... Date {datearray[j]}: "
                            "Early period (2000-2002), "
                            "no eligible stocks, assigning 100% to CASH"
                        )
                else:
                    # Fallback: equal weights to all stocks if no CASH symbol
                    equal_weight = 1.0 / n_stocks
                    monthgainlossweight[:, j] = equal_weight
                    if verbose:
                        print(
                            f" ... Date {datearray[j]}: "
                            "Early period but no CASH symbol, "
                            f"assigning equal weights to all {n_stocks} stocks"
                        )
            else:
                # Non-early period with no eligible stocks (signals present but
                # Sharpe filtering excluded all candidates). Assign 100% to CASH
                # or leave weights at 0.0. Do NOT spread equal weight to all stocks.
                cash_idx = symbols.index('CASH') if 'CASH' in symbols else None
                if cash_idx is not None:
                    monthgainlossweight[:, j] = 0.0
                    monthgainlossweight[cash_idx, j] = 1.0
                    if verbose:
                        print(
                            f" ... Date {datearray[j]}: "
                            "No eligible stocks (non-early period), "
                            "assigning 100% to CASH")
                else:
                    # monthgainlossweight[:, j] already 0.0 from initialization.
                    if verbose:
                        print(
                            f" ... Date {datearray[j]}: "
                            "No eligible stocks (non-early period), "
                            "no CASH symbol, leaving weights at 0.0"
                        )
            continue

        # If we're in the early period, force 100% CASH even if there are eligible stocks
        # (algorithm warm-up period)
        if is_early_period:
            cash_idx = symbols.index('CASH') if 'CASH' in symbols else None
            if cash_idx is not None:
                monthgainlossweight[:, j] = 0.0  # Zero all weights first
                monthgainlossweight[cash_idx, j] = 1.0  # 100% in cash
                continue
            else:
                if verbose:
                    print(
                        f" ... Date {datearray[j]}: "
                        "Early period but no CASH symbol, "
                        "proceeding with normal selection"
                    )

        # Get indices of eligible stocks
        eligible_indices = np.where(eligible_stocks)[0]
        
        # Sort by Sharpe descending
        sorted_indices = eligible_indices[np.argsort(-sharpe_2d[eligible_indices, j])]
        
        # Select top min(numberStocksTraded, len) stocks
        num_to_select = min(numberStocksTraded, len(sorted_indices))
        selected_stocks = sorted_indices[:num_to_select]

        if len(selected_stocks) == 0:
            continue

        # Get Sharpe ratios for selected stocks.
        selected_sharpe = sharpe_2d[selected_stocks, j]

        # Handle NaN and zero sum cases
        selected_sharpe = np.nan_to_num(selected_sharpe, nan=0.0)
        
        # Weight proportionally to Sharpe ratios
        if np.sum(selected_sharpe) == 0:
            raw_weights = np.ones(len(selected_stocks)) / len(selected_stocks)
        else:
            raw_weights = selected_sharpe / np.sum(selected_sharpe)

        if apply_constraints:
            # Apply weight constraints relative to equal weight
            equal_weight = 1.0 / len(selected_stocks)
            max_weight = min(max_weight_factor * equal_weight, absolute_max_weight)
            min_weight = min_weight_factor * equal_weight

            # Clip weights to bounds
            constrained_weights = np.clip(raw_weights, min_weight, max_weight)
            # Renormalize to sum to 1.0
            if np.sum(constrained_weights) > 0:
                constrained_weights /= np.sum(constrained_weights)
            else:
                constrained_weights = np.ones(len(selected_stocks)) / len(selected_stocks)

            # Assign constrained weights
            monthgainlossweight[selected_stocks, j] = constrained_weights
        else:
            # Simple proportional weighting without constraints
            monthgainlossweight[selected_stocks, j] = raw_weights

    #########################################################################
    # Forward-fill weights to ensure every day has valid weights.
    #########################################################################
    print(" ... Forward-filling weights...")

    for j in range(1, n_days):
        # Only forward-fill if the entire day has zero weights
        if np.sum(monthgainlossweight[:, j]) == 0:
            monthgainlossweight[:, j] = monthgainlossweight[:, j - 1]

    # Normalize weights to sum to 1.0 for each day.
    for j in range(n_days):
        daily_sum = monthgainlossweight[:, j].sum()
        if daily_sum > 0:
            monthgainlossweight[:, j] /= daily_sum

    print(" ... Forward-filling and normalization complete.")
    #########################################################################

    print(" ... sharpeWeightedRank_2D computation complete")
    print(f" ... Non-zero weights in output: {np.sum(monthgainlossweight > 0)}")
    
    # Verify first and last dates have valid weights.
    print(f" ... Weights sum on day 0: {monthgainlossweight[:, 0].sum():.4f}")
    print(f" ... Weights sum on day {n_days-1}: {monthgainlossweight[:, -1].sum():.4f}")
    
    return monthgainlossweight

    # temporarily skip this!!!!!!
    #return

    import datetime
    from functions.GetParams import get_json_params, get_holdings, GetEdition
    from functions.CheckMarketOpen import get_MarketOpenOrClosed
    #from functions.SendEmail import SendTextMessage
    from functions.SendEmail import SendEmail

    # send text message for held stocks if the lastest quote is outside
    # (to downside) the established channel

    # Get Credentials for sending email
    params = get_json_params(json_fn)
    print("")

    #print "params = ", params
    print("")
    username = str(params['fromaddr']).split("@")[0]
    emailpassword = str(params['PW'])

    subjecttext = "PyTAAA update - Pct Trend Channel"
    boldtext = "time is "+datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
    headlinetext = "market status: " + get_MarketOpenOrClosed()

    # Get Holdings from file
    # holdings = GetHoldings()
    holdings = get_holdings(json_fn)
    holdings_symbols = holdings['stocks']
    edition = GetEdition()

    # process symbols in current holdings
    downtrendSymbols = []
    channelPercent = []
    channelGainsLossesHoldings = []
    channelStdsHoldings = []
    channelGainsLosses = []
    channelStds = []
    currentNumStdDevs = []
    for i, symbol in enumerate(symbols):
        pctChannel,channelGainLoss,channelStd,numStdDevs = \
            jumpTheChannelTest(
                adjClose[i,:],
                minperiod=params['minperiod'],
                maxperiod=params['maxperiod'],
                incperiod=params['incperiod'],
                numdaysinfit=params['numdaysinfit'],
                offset=params['offset']
            )
        channelGainsLosses.append(channelGainLoss)
        channelStds.append(channelStd)
        if symbol in holdings_symbols:
            #pctChannel = jumpTheChannelTest(adjClose[i,:],minperiod=4,maxperiod=12,incperiod=3,numdaysinfit=28, offset=3)
            print(" ... performing PctChannelTest: symbol = ",format(symbol,'5s'), "  pctChannel = ", format(pctChannel-1.,'6.1%'))

            # send textmessage alert of current trend
            downtrendSymbols.append(symbol)
            channelPercent.append(format(pctChannel-1.,'6.1%'))
            channelGainsLossesHoldings.append(format(channelGainLoss,'6.1%'))
            channelStdsHoldings.append(format(channelStd,'6.1%'))
            currentNumStdDevs.append(format(numStdDevs,'6.1f'))

    print("\n ... downtrending symbols are ", downtrendSymbols, "\n")

    if len(downtrendSymbols) > 0:
        #--------------------------------------------------
        # send text message
        #--------------------------------------------------
        #text_message = "PyTAAA/"+edition+" shows "+str(downtrendSymbols)+" in possible downtrend... \n"+str(channelPercent)+" % of trend channel."
        text_message = "PyTAAA/"+edition+" shows "+str(downtrendSymbols)+" current trend... "+\
                       "\nPct of trend channel  = "+str(channelPercent)+\
                       "\nperiod gainloss     = "+str(channelGainsLossesHoldings)+\
                       "\nperiod gainloss std = "+str(channelStdsHoldings)+\
                       "\ncurrent # std devs  = "+str(currentNumStdDevs)

        print(text_message +"\n\n")

        # send text message if market is open
        if 'close in' in get_MarketOpenOrClosed():
            #SendTextMessage( username,emailpassword,params['toSMS'],params['fromaddr'],text_message )
            SendEmail(username,emailpassword,params['toSMS'],params['fromaddr'],subjecttext,text_message,boldtext,headlinetext)

    return


def multiSharpe(
        datearray: np.ndarray,
        adjClose: np.ndarray,
        periods: list
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute average Sharpe ratios across multiple lookback periods.

    For each lookback period in ``periods``, calculates the rolling
    Sharpe ratio for every stock in ``adjClose`` at every date after the
    longest period.  The per-stock Sharpe values are averaged across
    stocks and across periods to produce a ``medianSharpe`` and a
    ``signal`` time series.

    Args:
        datearray: 1D array of dates corresponding to ``adjClose`` columns.
        adjClose: 2D array of adjusted close prices with shape
            ``(n_stocks, n_dates)``.
        periods: List of integer lookback periods (in days) over which
            Sharpe ratios are computed.

    Returns:
        A 3-tuple of:
            - dates (np.ndarray): Date array starting after the longest
              period.
            - medianSharpe (np.ndarray): Clipped median of the adjusted
              Sharpe values over time, in [−0.1, 1.1].
            - signal (np.ndarray): Blended signal combining median and
              mean Sharpe contributions, clipped to [−0.05, 1.05].

    Notes:
        - Raw Sharpe values are shifted by +0.3 and scaled by 1/1.25
          before aggregation, effectively normalising them to a roughly
          [0, 1] range.
        - Progress is printed to stdout every 1 000 dates.
    """

    maxPeriod = np.max( periods )

    dates = datearray[maxPeriod:]
    sharpesPeriod = np.zeros( (len(periods),len(dates)), 'float' )
    #adjCloseSubset = adjClose[:,-len(dates):]

    for iperiod,period in enumerate(periods) :
        lenSharpe = period
        for idate in range( maxPeriod,adjClose.shape[1] ):
            sharpes = []
            for ii in range(adjClose.shape[0]):
                sharpes.append( allstats( adjClose[ii,idate-lenSharpe:idate] ).sharpe() )
            sharpes = np.array( sharpes )
            sharpes = sharpes[np.isfinite( sharpes )]
            if len(sharpes) > 0:
                sharpesAvg = np.mean(sharpes)
                if idate%1000 == 0:
                    print(period, datearray[idate],len(sharpes), sharpesAvg)
            else:
                sharpesAvg = 0.
            sharpesPeriod[iperiod,idate-maxPeriod] = sharpesAvg

    plotSharpe = sharpesPeriod[:,-len(dates):].copy()
    plotSharpe += .3
    plotSharpe /= 1.25
    signal = np.median(plotSharpe,axis=0)
    for i in range( plotSharpe.shape[0] ):
        signal += (np.clip( plotSharpe[i,:], -1., 2.) - signal)

    medianSharpe = np.median(plotSharpe,axis=0)
    signal = np.median(plotSharpe,axis=0) + 1.5 * (np.mean(plotSharpe,axis=0) - np.median(plotSharpe,axis=0))

    medianSharpe = np.clip( medianSharpe, -.1, 1.1 )
    signal = np.clip( signal, -.05, 1.05 )

    return dates, medianSharpe, signal


def MAA_WeightedRank_2D(
        json_fn: str,
        datearray: np.ndarray,
        symbols: list,
        adjClose: np.ndarray,
        signal2D: np.ndarray,
        signal2D_daily: np.ndarray,
        LongPeriod: int,
        numberStocksTraded: int,
        wR: float,
        wC: float,
        wV: float,
        wS: float,
        stddevThreshold: float = 4.
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute MAA (Minimum Acceptable Absolute) portfolio weights for all dates.

    Ranks stocks using a composite score based on period return,
    inverse correlation to the equal-weighted index (EWI), and
    volatility, weighted by the exponents ``wR``, ``wC``, ``wV``, and
    ``wS``.  The top ``numberStocksTraded`` stocks are selected at each
    date and their weights are normalised to sum to 1.  A separate
    crash-protection (CP) weight series based on the fraction of
    down-trending stocks is also computed.

    Args:
        json_fn: Path to the JSON configuration file.
        datearray: 1D array of dates corresponding to ``adjClose`` columns.
        symbols: List of stock ticker symbols corresponding to rows of
            ``adjClose``.
        adjClose: 2D array of adjusted close prices ``(n_stocks, n_dates)``.
        signal2D: 2D binary uptrend signal array ``(n_stocks, n_dates)``
            (1 = uptrending).
        signal2D_daily: 2D daily uptrend signal array ``(n_stocks, n_dates)``
            used for reporting.
        LongPeriod: Lookback period in days for return and correlation
            calculations.
        numberStocksTraded: Maximum number of stocks to hold at any date.
        wR: Exponent applied to the period return in the weight score.
        wC: Exponent applied to ``(1 - correlation_to_EWI)`` in the
            weight score.
        wV: Exponent applied to the inverse volatility in the weight
            score.
        wS: Outer exponent applied to the combined score.
        stddevThreshold: Number of standard deviations used to detect
            price spikes before calculation. Defaults to 4.0.

    Returns:
        A 2-tuple of 2D NumPy arrays ``(weights, CPweights)``, both of
        shape ``(n_stocks, n_dates)``, where each column sums to 1.0.
        ``weights`` reflects the MAA composite score ranking;
        ``CPweights`` incorporates crash-protection cash allocation.

    Notes:
        - Requires ``bottleneck`` for fast ranking; falls back to
          ``scipy.stats.mstats`` if unavailable.
        - The CASH symbol weight is overridden by the crash-protection
          fraction at each date.
        - Weights are held constant within each calendar month.
        - Requires ``nose`` (imported internally); this dependency may
          be absent in modern environments.

    # TODO(human-review): The ``import nose`` statement inside the function
    # body appears to be a legacy test-runner dependency that should be removed
    # as ``nose`` is no longer maintained.
    """

    import numpy as np
    import nose
    import os
    import sys
    from matplotlib import pylab as plt
    import matplotlib.gridspec as gridspec
    try:
        import bottleneck as bn
        from bottleneck import rankdata as rd
    except ImportError:
        import scipy.stats.mstats as bn

    from functions.GetParams import get_json_params
    params = get_json_params(json_fn)
    stockList = params['stockList']

    adjClose_despike = despike_2D( adjClose, LongPeriod, stddevThreshold=stddevThreshold )

    gainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    #gainloss[:,1:] = adjClose[:,1:] / adjClose[:,:-1]
    gainloss[:,1:] = adjClose_despike[:,1:] / adjClose_despike[:,:-1]  ## experimental
    gainloss[isnan(gainloss)]=1.

    # Create a binary mask from `signal2D` and do not modify the input array.
    signal_mask = (signal2D > 0.5).astype(float)

    ############################
    ###
    ### filter universe of stocks to exclude all that have return < 0
    ### - needed for correlation to "equal weight index" (EWI)
    ### - EWI is daily gain/loss percentage
    ###
    ############################

    EWI  = np.zeros( adjClose.shape[1], 'float' )
    EWI_count  = np.zeros( adjClose.shape[1], 'int' )
    for jj in np.arange(LongPeriod,adjClose.shape[1]) :
        for ii in range(adjClose.shape[0]):
            if signal2D_daily[ii,jj] == 1:
                EWI[jj] += gainloss[ii,jj]
                EWI_count[jj] += 1
    EWI = EWI/EWI_count
    EWI[np.isnan(EWI)] = 1.0

    ############################
    ###
    ### compute correlation to EWI
    ### - each day, for each stock
    ### - not needed for stocks on days with return < 0
    ###
    ############################

    corrEWI  = np.zeros( adjClose.shape, 'float' )
    for jj in np.arange(LongPeriod,adjClose.shape[1]) :
        for ii in range(adjClose.shape[0]):
            start_date = max( jj - LongPeriod, 0 )
            if adjClose_despike[ii,jj] > adjClose_despike[ii,start_date]:
                corrEWI[ii,jj] = normcorrcoef(gainloss[ii,start_date:jj]-1.,EWI[start_date:jj]-1.)
                if corrEWI[ii,jj] <0:
                    corrEWI[ii,jj] = 0.

    ############################
    ###
    ### compute weights
    ### - each day, for each stock
    ### - set to 0. for stocks on days with return < 0
    ###
    ############################

    weights  = np.zeros( adjClose.shape, 'float' )
    for jj in np.arange(LongPeriod,adjClose.shape[1]) :
        for ii in range(adjClose.shape[0]):
            start_date = max( jj - LongPeriod, 0 )
            returnForPeriod = (adjClose_despike[ii,jj]/adjClose_despike[ii,start_date])-1.
            if returnForPeriod  < 0.:
                returnForPeriod = 0.
            volatility = np.std(adjClose_despike[ii,start_date:jj])
            weights[ii,jj] = ( returnForPeriod**wR * (1.-corrEWI[ii,jj])**wC / volatility**wV ) **wS

    weights[np.isnan(weights)] = 0.0

    # make duplicate of weights for adjusting using crashProtection
    CPweights = weights.copy()
    CP_cashWeight = np.zeros(adjClose.shape[1], 'float' )
    for jj in np.arange(adjClose.shape[1]) :
        weightsToday = weights[:,jj]
        CP_cashWeight[jj] = float(len(weightsToday[weightsToday==0.])) / len(weightsToday)

    ############################
    ###
    ### compute weights ranking and keep best
    ### 'best' are numberStocksTraded*%risingStocks
    ### - weights need to sum to 100%
    ###
    ############################

    weightRank = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=int)

    weightRank = bn.rankdata(weights,axis=0)
    # reverse the ranks (low ranks are biggest gainers)
    maxrank = np.max(weightRank)
    weightRank -= maxrank-1
    weightRank *= -1
    weightRank += 2

    # set top 'numberStocksTraded' to have weights sum to 1.0
    for jj in np.arange(adjClose.shape[1]) :
        ranksToday = weightRank[:,jj].copy()
        weightsToday = weights[:,jj].copy()
        weightsToday[ranksToday > numberStocksTraded] = 0.
        if np.sum(weightsToday) > 0.:
            weights[:,jj] = weightsToday / np.sum(weightsToday)
        else:
            weights[:,jj] = 1./len(weightsToday)

    # set CASH to have weight based on CrashProtection
    cash_index = symbols.index("CASH")
    for jj in np.arange(adjClose.shape[1]) :
        CPweights[ii,jj] = CP_cashWeight[jj]
        weightRank[ii,jj] = 0
        ranksToday = weightRank[:,jj].copy()
        weightsToday = CPweights[:,jj].copy()
        weightsToday[ranksToday > numberStocksTraded] = 0.
        if np.sum(weightsToday) > 0.:
            CPweights[:,jj] = weightsToday / np.sum(weightsToday)
        else:
            CPweights[:,jj] = 1./len(weightsToday)

    # hold weights constant for month
    for jj in np.arange(LongPeriod,adjClose.shape[1]) :
        start_date = max( jj - LongPeriod, 0 )
        yesterdayMonth = datearray[jj-1].month
        todayMonth = datearray[jj].month
        if todayMonth == yesterdayMonth:
            weights[:,jj] = weights[:,jj-1]
            CPweights[:,jj] = CPweights[:,jj-1]

    # input symbols and company names from text file
    json_dir = os.path.split(json_fn)[0]
    if stockList == 'Naz100':
        companyName_file = os.path.join( json_dir, "symbols",  "companyNames.txt" )
    elif stockList == 'SP500':
        companyName_file = os.path.join( json_dir, "symbols",  "SP500_companyNames.txt" )
    with open( companyName_file, "r" ) as f:
        companyNames = f.read()

    print("\n\n\n")
    companyNames = companyNames.split("\n")
    ii = companyNames.index("")
    del companyNames[ii]
    companySymbolList  = []
    companyNameList = []
    for iname,name in enumerate(companyNames):
        name = name.replace("amp;", "")
        testsymbol, testcompanyName = name.split(";")
        companySymbolList.append(format(testsymbol,'5s'))
        companyNameList.append(testcompanyName)

    # print list showing current rankings and weights
    # - symbol
    # - rank (at begining of month)
    # - rank (most recent trading day)
    # - weight from sharpe ratio
    # - price
    import os
    rank_text = "<div id='rank_table_container'><h3>"+"<p>Current stocks, with ranks, weights, and prices are :</p></h3><font face='courier new' size=3><table border='1'> \
               </td><td>Rank (today) \
               </td><td>Symbol \
               </td><td>Company \
               </td><td>Weight \
               </td><td>CP Weight \
               </td><td>Price  \
               </td><td>Trend  \
               </td></tr>\n"
    for i, isymbol in enumerate(symbols):
        for j in range(len(symbols)):
            if int( weightRank[j,-1] ) == i :
                if signal2D_daily[j,-1] == 1.:
                    trend = 'up'
                else:
                    trend = 'down'

                # search for company name
                try:
                    symbolIndex = companySymbolList.index(format(symbols[j],'5s'))
                    companyName = companyNameList[symbolIndex]
                except (ValueError, IndexError):
                    companyName = ""

                rank_text = rank_text + \
                       "<tr><td>" + format(weightRank[j,-1],'6.0f')  + \
                       "<td>" + format(symbols[j],'5s')  + \
                       "<td>" + format(companyName,'15s')  + \
                       "<td>" + format(weights[j,-1],'5.03f') + \
                       "<td>" + format(CPweights[j,-1],'5.03f') + \
                       "<td>" + format(adjClose[j,-1],'6.2f')  + \
                       "<td>" + trend  + \
                       "</td></tr>  \n"
    rank_text = rank_text + "</table></div>\n"

    print("leaving function MAA_WeightedRank_2D...")

    return weights, CPweights


#----------------------------------------------
def UnWeightedRank_2D(
        datearray: np.ndarray,
        adjClose: np.ndarray,
        signal2D: np.ndarray,
        LongPeriod: int,
        rankthreshold: int,
        riskDownside_min: float,
        riskDownside_max: float,
        rankThresholdPct: float
) -> np.ndarray:
    """Compute equal-weighted portfolio allocations via rank-change scoring.

    Ranks stocks by their period gain/loss, computes a delta-rank
    score that rewards improving momentum, and assigns equal weight to
    the top ``rankthreshold`` stocks at each date.  Weights are held
    constant within each calendar month to reduce transaction costs.

    Unlike the previous version of this function, this function uses
    equal weights for selected stocks rather than risk-adjusted weights.

    Args:
        datearray: 1D array of dates corresponding to ``adjClose`` columns.
        adjClose: 2D array of adjusted close prices ``(n_stocks, n_dates)``.
        signal2D: 2D binary uptrend signal array ``(n_stocks, n_dates)``
            (1 = uptrending); non-uptrending stocks receive a neutral
            daily gain of 1.0.
        LongPeriod: Lookback period in days for period gain/loss
            calculation.
        rankthreshold: Number of top-ranked stocks to include in the
            portfolio at each date.
        riskDownside_min: Unused; kept for API compatibility.
        riskDownside_max: Unused; kept for API compatibility.
        rankThresholdPct: Fraction used to exclude stocks with extreme
            gain/loss ranks from selection.

    Returns:
        2D NumPy array of portfolio weights with shape
        ``(n_stocks, n_dates)``.  Each column sums to approximately
        1.0 (minor floating-point deviations may occur).

    Notes:
        - Requires ``bottleneck`` for fast ranking; falls back to
          ``scipy.stats.mstats`` if unavailable.
        - When fewer than ``rankthreshold + 2`` stocks are active a
          greedy fallback iteratively assigns the highest delta-rank
          stocks until the threshold is filled.
        - Look-ahead bias is present in rank computation (global
          ranking across all dates) — see :func:`sharpeWeightedRank_2D`
          for a point-in-time alternative.
    """

    import numpy as np
    # import nose
    try:
        import bottleneck as bn
        from bottleneck import rankdata as rd
    except ImportError:
        import scipy.stats.mstats as bn


    gainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    gainloss[:,1:] = adjClose[:,1:] / adjClose[:,:-1]
    gainloss[isnan(gainloss)]=1.

    # Create a binary mask from `signal2D` and do not modify the input array.
    signal_mask = (signal2D > 0.5).astype(float)

    # apply signal mask to daily gainloss
    gainloss = gainloss * signal_mask
    gainloss[gainloss == 0] = 1.0

    value = 10000. * np.cumprod(gainloss,axis=1)

    # calculate gainloss over period of "LongPeriod" days
    monthgainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    monthgainloss[:,LongPeriod:] = adjClose[:,LongPeriod:] / adjClose[:,:-LongPeriod]
    monthgainloss[isnan(monthgainloss)]=1.

    monthgainlossweight = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)

    rankweight = 1./rankthreshold

    ########################################################################
    ## Calculate change in rank of active stocks each day (without duplicates as ties)
    ########################################################################
    monthgainlossRank = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=int)
    monthgainlossPrevious = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    monthgainlossPreviousRank = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=int)

    monthgainlossRank = bn.rankdata(monthgainloss,axis=0)
    # reverse the ranks (low ranks are biggest gainers)
    maxrank = np.max(monthgainlossRank)
    monthgainlossRank -= maxrank-1
    monthgainlossRank *= -1
    monthgainlossRank += 2

    monthgainlossPrevious[:,LongPeriod:] = monthgainloss[:,:-LongPeriod]
    monthgainlossPreviousRank = bn.rankdata(monthgainlossPrevious,axis=0)
    # reverse the ranks (low ranks are biggest gainers)
    maxrank = np.max(monthgainlossPreviousRank)
    monthgainlossPreviousRank -= maxrank-1
    monthgainlossPreviousRank *= -1
    monthgainlossPreviousRank += 2

    # weight deltaRank for best and worst performers differently
    rankoffsetchoice = rankthreshold
    delta = -(monthgainlossRank - monthgainlossPreviousRank ) / (monthgainlossRank + rankoffsetchoice)

    # if rank is outside acceptable threshold, set deltarank to zero so stock will not be chosen
    #  - remember that low ranks are biggest gainers
    rankThreshold = (1. - rankThresholdPct) * ( monthgainlossRank.max() - monthgainlossRank.min() )
    for ii in range(monthgainloss.shape[0]):
        for jj in range(monthgainloss.shape[1]):
            if monthgainloss[ii,jj] > rankThreshold :
                delta[ii,jj] = -monthgainloss.shape[0]/2

    deltaRank = bn.rankdata(delta,axis=0)
    # reverse the ranks (low deltaRank have the fastest improving rank)
    maxrank = np.max(deltaRank)
    deltaRank -= maxrank-1
    deltaRank *= -1
    deltaRank += 2

    for ii in range(monthgainloss.shape[1]):
        if deltaRank[:,ii].min() == deltaRank[:,ii].max():
            deltaRank[:,ii] = 0.

    ########################################################################
    ## Hold values constant for calendar month (gains, ranks, deltaRanks)
    ########################################################################

    for ii in np.arange(1,monthgainloss.shape[1]):
        if datearray[ii].month == datearray[ii-1].month:
            monthgainloss[:,ii] = monthgainloss[:,ii-1]
            deltaRank[:,ii] = deltaRank[:,ii-1]

    ########################################################################
    ## Calculate number of active stocks each day
    ########################################################################

    # TODO: activeCount can be computed before loop to save CPU cycles
    # count number of unique values
    activeCount = np.zeros(adjClose.shape[1],dtype=float)
    for ii in np.arange(0,monthgainloss.shape[0]):
        firsttradedate = np.argmax( np.clip( np.abs( gainloss[ii,:]-1. ), 0., .00001 ) )
        activeCount[firsttradedate:] += 1

    minrank = np.min(deltaRank,axis=0)
    maxrank = np.max(deltaRank,axis=0)
    # convert rank threshold to equivalent percent of rank range

    rankthresholdpercentequiv = np.round(float(rankthreshold)*(activeCount-minrank+1)/adjClose.shape[0])
    ranktest = deltaRank <= rankthresholdpercentequiv

    ########################################################################
    ### calculate equal weights for ranks below threshold
    ########################################################################

    elsecount = 0
    elsedate  = 0
    for ii in np.arange(1,monthgainloss.shape[1]) :
        if activeCount[ii] > minrank[ii] and rankthresholdpercentequiv[ii] > 0:
            for jj in range(value.shape[0]):
                test = deltaRank[jj,ii] <= rankthresholdpercentequiv[ii]
                if test == True :
                    monthgainlossweight[jj,ii]  = 1./rankthresholdpercentequiv[ii]
                else:
                    monthgainlossweight[jj,ii]  = 0.
        elif activeCount[ii] == 0 :
            monthgainlossweight[:,ii]  *= 0.
            monthgainlossweight[:,ii]  += 1./adjClose.shape[0]
        else :
            elsedate = datearray[ii]
            elsecount += 1
            monthgainlossweight[:,ii]  = 1./activeCount[ii]

    aaa = np.sum(monthgainlossweight,axis=0)

    print("")
    print(" invoking correction to monthgainlossweight.....")
    print("")
    # find first date with number of stocks trading (rankthreshold) + 2
    activeCountAboveMinimum = activeCount
    activeCountAboveMinimum += -rankthreshold + 2
    firstTradeDate = np.argmax( np.clip( activeCountAboveMinimum, 0 , 1 ) )
    for ii in np.arange(firstTradeDate,monthgainloss.shape[1]) :
        if np.sum(monthgainlossweight[:,ii]) == 0:
            for kk in range(rankthreshold):
                indexHighDeltaRank = np.argmin(deltaRank[:,ii]) # remember that best performance is lowest deltaRank
                monthgainlossweight[indexHighDeltaRank,ii]  = 1./rankthreshold
                deltaRank[indexHighDeltaRank,ii] = 1000.


    print(" weights calculation else clause encountered :",elsecount," times. last date encountered is ",elsedate)
    rankweightsum = np.sum(monthgainlossweight,axis=0)

    monthgainlossweight[isnan(monthgainlossweight)] = 0.  # changed result from 1 to 0

    monthgainlossweight = monthgainlossweight / np.sum(monthgainlossweight,axis=0)
    monthgainlossweight[isnan(monthgainlossweight)] = 0.  # changed result from 1 to 0

    return monthgainlossweight


def hurst(X: np.ndarray) -> float:
    """Compute the Hurst exponent of a time series.

    The Hurst exponent characterises the long-range dependency of a
    series.  H ≈ 0.5 indicates random-walk behaviour; H < 0.5 indicates
    mean-reversion; H > 0.5 indicates trending (persistent) behaviour.

    Args:
        X: 1D array representing the time series.

    Returns:
        Hurst exponent H as a float.

    Example:
        >>> import numpy as np
        >>> from functions.TAfunctions import hurst
        >>> a = np.random.randn(4096)
        >>> hurst(a)  # doctest: +SKIP
        0.506...

    Notes:
        Ported from the PyEEG library (Forrest Sheng Bao, 2010).
        See https://code.google.com/p/pyeeg/ for the original source.
    """

    from numpy import zeros, log, array, cumsum, std
    from numpy.linalg import lstsq

    N = len(X)

    T = array([float(i) for i in range(1,N+1)])
    Y = cumsum(X)
    Ave_T = Y/T

    S_T = zeros((N))
    R_T = zeros((N))
    for i in range(N):
        S_T[i] = std(X[:i+1])
        X_T = Y - T * Ave_T[i]
        R_T[i] = max(X_T[:i + 1]) - min(X_T[:i + 1])

    R_S = R_T / S_T
    R_S = log(R_S)
    n = log(T).reshape(N, 1)
    H = lstsq(n[1:], R_S[1:])[0]
    return H[0]


def textmessageOutsideTrendChannel(
        symbols: list,
        adjClose: np.ndarray,
        json_fn: str
) -> None:
    """Send email alerts for stocks with extreme single-day price moves.

    Checks for stocks in ``symbols`` that have moved more than 5 % in a
    single day (up or down) and sends an email alert when the market is
    open.  At most 5 stocks are reported per call to avoid alert spam.

    Args:
        symbols: List of ticker symbols corresponding to rows of
            ``adjClose``.
        adjClose: 2D array of adjusted close prices ``(n_stocks, n_dates)``
            with at least 2 date columns.
        json_fn: Path to the JSON configuration file containing email
            credentials (``fromaddr``, ``PW``, ``toSMS``/``toaddrs``).

    Returns:
        None

    Notes:
        - The function only sends alerts when the market is open
          (as determined by :func:`~functions.CheckMarketOpen.get_MarketOpenOrClosed`).
        - All exceptions are caught and logged rather than re-raised so
          that alert failures do not abort the main pipeline.

    # TODO(human-review): The original intent was to detect stocks that
    # have moved outside their price-channel bands (not just a fixed 5 %
    # threshold).  The current implementation is a simplified placeholder
    # and may need to be updated to use recentTrendAndStdDevs for proper
    # channel-based detection.
    """
    # Import required modules
    from functions.GetParams import get_json_params
    from functions.CheckMarketOpen import get_MarketOpenOrClosed
    from functions.SendEmail import SendEmail
    
    try:
        # Get parameters
        params = get_json_params(json_fn)
        username = str(params['fromaddr']).split("@")[0]
        emailpassword = str(params['PW'])
        
        # Check if market is open
        market_status = get_MarketOpenOrClosed()
        if 'Market Open' not in market_status:
            return
            
        # Basic implementation - check for stocks with extreme moves
        # This is a simplified version that sends alerts for stocks
        # that have moved more than 5% in the last day
        
        if adjClose.shape[1] >= 2:  # Need at least 2 days of data
            daily_returns = (adjClose[:, -1] - adjClose[:, -2]) / adjClose[:, -2]
            
            # Find stocks with extreme moves (>5% or <-5%)
            extreme_moves = np.abs(daily_returns) > 0.05
            extreme_indices = np.where(extreme_moves)[0]
            
            if len(extreme_indices) > 0:
                message_lines = []
                for idx in extreme_indices[:5]:  # Limit to 5 stocks to avoid spam
                    symbol = symbols[idx]
                    ret = daily_returns[idx]
                    direction = "UP" if ret > 0 else "DOWN"
                    pct = abs(ret) * 100
                    message_lines.append(f"{symbol}: {direction} {pct:.1f}%")
                
                if message_lines:
                    subject = "PyTAAA: Stocks Outside Trend Channel"
                    body = "Stocks with extreme daily moves:\n" + "\n".join(message_lines)
                    
                    # Send email (text message functionality commented out)
                    SendEmail(username, emailpassword, params.get('toSMS', params.get('toaddrs', '')), 
                             params['fromaddr'], subject, body, "", "")
                    
                    print(f"Sent alert for {len(message_lines)} stocks outside trend channel")
    
    except Exception as e:
        print(f"Warning: textmessageOutsideTrendChannel failed: {e}")
        # Don't raise exception to avoid breaking the main flow
