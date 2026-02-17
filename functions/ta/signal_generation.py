"""Signal generation functions for trading decisions.

This module provides the core computeSignal2D function that generates buy/sell
signals based on various technical indicators including moving averages and channels.
"""

import logging
import numpy as np
from numpy import isnan
from numpy.typing import NDArray
from typing import Union

# Import from sibling modules
from functions.ta.moving_averages import SMA_2D, hma
from functions.ta.channels import dpgchannel_2D, percentileChannel_2D

logger = logging.getLogger(__name__)


def computeSignal2D(adjClose: NDArray[np.floating], gainloss: NDArray[np.floating], 
                    params: dict) -> Union[NDArray[np.floating], tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]]:
    """Compute buy/sell signals for multiple stocks using technical indicators.
    
    This is a core function that generates trading signals (1=buy, 0=sell) for each
    stock on each date based on price trends. Supports multiple signal generation
    methods: simple moving averages (SMAs), Hull moving averages (HMAs), min/max
    channels, and percentile channels.
    
    Args:
        adjClose: 2D array of shape (n_stocks, n_dates) with adjusted closing prices
        gainloss: 2D array of shape (n_stocks, n_dates) with daily gain/loss ratios
        params: Dictionary containing signal parameters including:
            - MA1 (int): Longest moving average period
            - MA2 (int): Shortest moving average period
            - MA2offset (int): Offset for middle moving average
            - MA2factor (float): Multiplier for longest MA
            - uptrendSignalMethod (str): Method name ('SMAs', 'HMAs', 
              'minmaxChannels', 'percentileChannels')
            - narrowDays, mediumDays, wideDays (list): Channel periods
            - lowPct, hiPct (float): Percentile levels for channels
            
    Returns:
        For 'SMAs', 'HMAs', 'minmaxChannels': NDArray of shape (n_stocks, n_dates)
            with 1.0 for buy signal, 0.0 for sell signal
        For 'percentileChannels': Tuple of (signal2D, lowChannel, hiChannel)
        
    Example:
        >>> adjClose = np.random.rand(100, 500)  # 100 stocks, 500 days
        >>> gainloss = np.ones_like(adjClose)
        >>> params = {'MA1': 200, 'MA2': 50, 'MA2offset': 50, 
        ...           'MA2factor': 1.0, 'uptrendSignalMethod': 'SMAs'}
        >>> signals = computeSignal2D(adjClose, gainloss, params)
        >>> # Returns 2D array of buy/sell signals
        
    Note:
        - Stocks with NaN prices on last date get signal set to 0
        - Constant prices at series beginning get signal set to 0
        - SMAs method: Signal=1 if price > long MA OR (price > min(short,mid) AND short rising)
        - HMAs method: Same logic as SMAs but uses Hull Moving Averages
        - minmaxChannels: Signal=1 if combined medium+wide channel signals > 0
        - percentileChannels: Signal=1 on crossing above low channel or above high channel
        
    Raises:
        No explicit raises, but will fail if params dict is missing required keys
    """
    logger.debug(" ... inside computeSignal2D ... ")
    logger.debug(" params = %s", params)
    MA1 = int(params['MA1'])
    MA2 = int(params['MA2'])
    MA2offset = int(params['MA2offset'])

    narrowDays = params['narrowDays']
    mediumDays = params['mediumDays']
    wideDays = params['wideDays']

    lowPct = float(params['lowPct'])
    hiPct = float(params['hiPct'])
    sma2factor = float(params['MA2factor'])
    uptrendSignalMethod = params['uptrendSignalMethod']
    
    # Parameters for percentile channels
    minperiod = int(params.get('minperiod', 4))
    maxperiod = int(params.get('maxperiod', 12))
    incperiod = int(params.get('incperiod', 3))

    if uptrendSignalMethod == 'SMAs':
        logger.info("Using 3 SMA's for signal2D")
        logger.info("Calculating signal2D using '%s' method", uptrendSignalMethod)
        #############################################################################
        # Calculate signal for all stocks based on 3 simple moving averages (SMA's)
        #############################################################################
        sma0 = SMA_2D(adjClose, MA2)               # MA2 is shortest
        sma1 = SMA_2D(adjClose, MA2 + MA2offset)
        sma2 = sma2factor * SMA_2D(adjClose, MA1)  # MA1 is longest

        signal2D = np.zeros((adjClose.shape[0], adjClose.shape[1]), dtype=float)
        for ii in range(adjClose.shape[0]):
            for jj in range(adjClose.shape[1]):
                if adjClose[ii, jj] > sma2[ii, jj] or ((adjClose[ii, jj] > min(sma0[ii, jj], sma1[ii, jj]) and sma0[ii, jj] > sma0[ii, jj-1])):
                    signal2D[ii, jj] = 1
                    if jj == adjClose.shape[1]-1 and isnan(adjClose[ii, -1]):
                        signal2D[ii, jj] = 0  # added to avoid choosing stocks no longer in index
            # take care of special case where constant share price is inserted at beginning of series
            index = np.argmax(np.clip(np.abs(gainloss[ii, :]-1), 0, 1e-8)) - 1
            signal2D[ii, 0:index] = 0

        dailyNumberUptrendingStocks = np.sum(signal2D, axis=0)
        return signal2D

    if uptrendSignalMethod == 'HMAs':
        logger.info("Using 3 HMA's (hull moving average) for signal2D")
        logger.info("Calculating signal2D using '%s' method", uptrendSignalMethod)
        #############################################################################
        # Calculate signal for all stocks based on 3 Hull moving averages (HMA's)
        #############################################################################
        sma0 = hma(adjClose, MA2)               # MA2 is shortest
        sma1 = hma(adjClose, MA2 + MA2offset)
        sma2 = sma2factor * hma(adjClose, MA1)  # MA1 is longest

        signal2D = np.zeros((adjClose.shape[0], adjClose.shape[1]), dtype=float)
        for ii in range(adjClose.shape[0]):
            for jj in range(adjClose.shape[1]):
                if adjClose[ii, jj] > sma2[ii, jj] or ((adjClose[ii, jj] > min(sma0[ii, jj], sma1[ii, jj]) and sma0[ii, jj] > sma0[ii, jj-1])):
                    signal2D[ii, jj] = 1
                    if jj == adjClose.shape[1]-1 and isnan(adjClose[ii, -1]):
                        signal2D[ii, jj] = 0  # added to avoid choosing stocks no longer in index
            # take care of special case where constant share price is inserted at beginning of series
            index = np.argmax(np.clip(np.abs(gainloss[ii, :]-1), 0, 1e-8)) - 1
            signal2D[ii, 0:index] = 0

        dailyNumberUptrendingStocks = np.sum(signal2D, axis=0)
        return signal2D

    elif uptrendSignalMethod == 'minmaxChannels':
        logger.info("Using 3 minmax channels for signal2D")
        logger.info("Calculating signal2D using '%s' method", uptrendSignalMethod)

        #############################################################################
        # Calculate signal for all stocks based on 3 minmax channels (dpgchannels)
        #############################################################################

        # narrow channel is designed to remove day-to-day variability
        logger.debug("narrow days min=%d, max=%d, inc=%f", narrowDays[0], narrowDays[-1], (narrowDays[-1]-narrowDays[0])/7.)
        narrow_minChannel, narrow_maxChannel = dpgchannel_2D(adjClose, narrowDays[0], narrowDays[-1], (narrowDays[-1]-narrowDays[0])/7.)
        narrow_midChannel = (narrow_minChannel+narrow_maxChannel)/2.

        medium_minChannel, medium_maxChannel = dpgchannel_2D(adjClose, mediumDays[0], mediumDays[-1], (mediumDays[-1]-mediumDays[0])/7.)
        medium_midChannel = (medium_minChannel+medium_maxChannel)/2.
        mediumSignal = ((narrow_midChannel-medium_minChannel)/(medium_maxChannel-medium_minChannel)-0.5)*2.0

        wide_minChannel, wide_maxChannel = dpgchannel_2D(adjClose, wideDays[0], wideDays[-1], (wideDays[-1]-wideDays[0])/7.)
        wide_midChannel = (wide_minChannel+wide_maxChannel)/2.
        wideSignal = ((narrow_midChannel-wide_minChannel)/(wide_maxChannel-wide_minChannel)-0.5)*2.0

        signal2D = np.zeros((adjClose.shape[0], adjClose.shape[1]), dtype=float)
        for ii in range(adjClose.shape[0]):
            for jj in range(adjClose.shape[1]):
                if mediumSignal[ii, jj] + wideSignal[ii, jj] > 0:
                    signal2D[ii, jj] = 1
                    if jj == adjClose.shape[1]-1 and isnan(adjClose[ii, -1]):
                        signal2D[ii, jj] = 0  # added to avoid choosing stocks no longer in index
            # take care of special case where constant share price is inserted at beginning of series
            index = np.argmax(np.clip(np.abs(gainloss[ii, :]-1), 0, 1e-8)) - 1
            signal2D[ii, 0:index] = 0

        return signal2D

    elif uptrendSignalMethod == 'percentileChannels':
        logger.info("Calculating signal2D using '%s' method", uptrendSignalMethod)
        signal2D = np.zeros((adjClose.shape[0], adjClose.shape[1]), dtype=float)
        lowChannel, hiChannel = percentileChannel_2D(adjClose, minperiod, maxperiod, incperiod, lowPct, hiPct)
        for ii in range(adjClose.shape[0]):
            for jj in range(1, adjClose.shape[1]):
                if (adjClose[ii, jj] > lowChannel[ii, jj] and adjClose[ii, jj-1] <= lowChannel[ii, jj-1]) or adjClose[ii, jj] > hiChannel[ii, jj]:
                    signal2D[ii, jj] = 1
                elif (adjClose[ii, jj] < hiChannel[ii, jj] and adjClose[ii, jj-1] >= hiChannel[ii, jj-1]) or adjClose[ii, jj] < lowChannel[ii, jj]:
                    signal2D[ii, jj] = 0
                else:
                    signal2D[ii, jj] = signal2D[ii, jj-1]

                if jj == adjClose.shape[1]-1 and isnan(adjClose[ii, -1]):
                    signal2D[ii, jj] = 0  # added to avoid choosing stocks no longer in index
            # take care of special case where constant share price is inserted at beginning of series
            index = np.argmax(np.clip(np.abs(gainloss[ii, :]-1), 0, 1e-8)) - 1
            signal2D[ii, 0:index] = 0

        logger.debug("Finished calculating signal2D... mean signal2D = %f", signal2D.mean())

        return signal2D, lowChannel, hiChannel
