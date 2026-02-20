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
from typing import Union
from numpy.typing import NDArray
#from yahooFinance import getQuote
from functions.quotes_adjClose import get_pe
# from functions.readSymbols import readSymbolList
from functions.readSymbols import read_symbols_list_local
from functions.GetParams import get_webpage_store, get_performance_store

import matplotlib
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')


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
    TRADING_DAYS_10_YEARS: int = 2520


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
def selfsimilarity(hi,lo):

    from scipy.stats import percentileofscore
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
def jumpTheChannelTest(x,minperiod=4,maxperiod=12,incperiod=3,numdaysinfit=28, offset=3):
    ###
    ### compute linear trend in upper and lower channels and compare
    ### actual stock price to forecast range
    ### return pctChannel for each stock
    ### calling function will use pctChannel as signal.
    ### - e.g. negative pctChannel is signal that down-trend begins
    ### - e.g. more than 100% pctChanel is sgnal of new up-trend beginning

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
def recentChannelFit(x,minperiod=4,maxperiod=12,incperiod=3,numdaysinfit=28, offset=3):
    ###
    ### compute cumulative gain over fitting period and number of
    ### ratio of current quote to fitted trend. Rescale based on std dev
    ### of residuals during fitting period.
    ### - e.g. negative pctChannel is signal that down-trend begins
    ### - e.g. more than 100% pctChanel is sgnal of new up-trend beginning

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
def recentTrendAndStdDevs(x,datearray,minperiod=4,maxperiod=12,incperiod=3,numdaysinfit=28, offset=3):

    ###
    ### compute linear trend in upper and lower channels and compare
    ### actual stock price to forecast range
    ### return pctChannel for each stock
    ### calling function will use pctChannel as signal.
    ### - e.g. numStdDevs < -1. is signal that down-trend begins
    ### - e.g. whereas  > 1.0 is signal of new up-trend beginning

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


def recentSharpeWithAndWithoutGap(x,numdaysinfit=504,offset_factor=.4):

    from math import sqrt
    from scipy.stats import gmean

    ###
    ### - Cmpute sharpe ratio for recent prices with gap of 'offset' recent days
    ### - Compute 2nd sharpe ratio for recent prices recent days

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

def recentTrendAndMidTrendChannelFitWithAndWithoutGap(x,minperiod=4,maxperiod=12,incperiod=3,numdaysinfit=28,numdaysinfit2=20, offset=3):
    ###
    ### - Cmpute linear trend in upper and lower channels and compare
    ###   actual stock price to forecast range
    ### - Compute 2nd linear trend in upper and lower channels only for
    ###   small number of recent prices without gap
    ### - return pctChannel for each stock
    ### - calling function will use pctChannel as signal.
    ###   * e.g. numStdDevs < -1. is signal that down-trend begins
    ###   * e.g. whereas  > 1.0 is signal of new up-trend beginning

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

def recentTrendAndMidTrendWithGap(x,datearray,minperiod=4,maxperiod=12,incperiod=3,numdaysinfit=28,numdaysinfit2=20, offset=3):
    ###
    ### - Cmpute linear trend in upper and lower channels and compare
    ###   actual stock price to forecast range
    ### - Compute 2nd linear trend in upper and lower channels only for
    ###   small number of recent prices without gap
    ### - return pctChannel for each stock
    ### - calling function will use pctChannel as signal.
    ###   * e.g. numStdDevs < -1. is signal that down-trend begins
    ###   * e.g. whereas  > 1.0 is signal of new up-trend beginning

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
        x,
        datearray,
        minperiod=4,
        maxperiod=12,
        incperiod=3,
        numdaysinfit=28,
        numdaysinfit2=20,
        offset=3
    ):
    ###
    ### - Cmpute linear trend in upper and lower channels and compare
    ###   actual stock price to forecast range
    ### - Compute 2nd linear trend in upper and lower channels only for
    ###   small number of recent prices without gap
    ### - return pctChannel for each stock
    ### - calling function will use pctChannel as signal.
    ###   * e.g. numStdDevs < -1. is signal that down-trend begins
    ###   * e.g. whereas  > 1.0 is signal of new up-trend beginning

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
    json_fn,
    datearray,
    symbols,
    adjClose,
    signal2D,
    signal2D_daily,
    LongPeriod,
    numberStocksTraded,
    riskDownside_min,
    riskDownside_max,
    rankThresholdPct,
    stddevThreshold=5.0,
    makeQCPlots=False,
    # Parameters for backtest (refactored version).
    max_weight_factor=3.0,
    min_weight_factor=0.3,
    absolute_max_weight=0.9,
    apply_constraints=True,
    # Parameters for production (PortfolioPerformanceCalcs).
    is_backtest=True,
    verbose=False,
    stockList="SP500",  # Add stockList parameter for early period logic
    **kwargs  # Accept any additional keyword arguments for compatibility.
):
    """
    Compute Sharpe-ratio-weighted portfolio allocation for all dates.

    This function computes rolling Sharpe ratios for each stock and
    returns a 2D array of portfolio weights [n_stocks, n_days].
    Stocks are selected based on:
    1. Having an uptrend signal (signal2D > 0)
    2. Ranking in the top N by Sharpe ratio
    3. Passing data quality checks

    Weights are assigned proportionally to Sharpe ratios, with constraints
    applied via max_weight_factor, min_weight_factor, and absolute_max_weight.

    IMPORTANT: Weights are forward-filled so every day has valid weights.
    This ensures portfolio calculations don't result in zero values.

    Parameters
    ----------
    json_fn : str
        Path to JSON configuration file (not used but kept for API).
    datearray : np.ndarray
        Array of dates corresponding to adjClose columns.
    symbols : list
        List of stock symbols corresponding to adjClose rows.
    adjClose : np.ndarray
        2D array of adjusted close prices [n_stocks, n_days].
    signal2D : np.ndarray
        2D array of uptrend signals [n_stocks, n_days], 1=uptrend.
    signal2D_daily : np.ndarray
        Daily signals before monthly hold logic (not used but kept).
    LongPeriod : int
        Lookback period for Sharpe ratio calculation.
    numberStocksTraded : int
        Number of top stocks to select for the portfolio.
    riskDownside_min : float
        Minimum weight constraint per position.
    riskDownside_max : float
        Maximum weight constraint per position.
    rankThresholdPct : float
        Percentile threshold for rank filtering (not used currently).
    stddevThreshold : float
        Threshold for spike detection (default 5.0).
    makeQCPlots : bool
        If True, generate QC plots (default False).
    max_weight_factor : float
        Maximum weight as multiple of equal weight (default 3.0).
    min_weight_factor : float
        Minimum weight as multiple of equal weight (default 0.3).
    absolute_max_weight : float
        Absolute maximum weight cap (default 0.9).
    apply_constraints : bool
        Whether to apply weight constraints (default True).
    is_backtest : bool
        If True, running in backtest mode (default True).
        If False, running in production mode.
    verbose : bool
        If True, print progress information (default False).
    stockList : str
        Stock list identifier ("SP500" or "Naz100"). Early period logic only applies to "SP500" (default "SP500").
    **kwargs : dict
        Additional keyword arguments for forward compatibility.

    Returns
    -------
    np.ndarray
        2D array of portfolio weights [n_stocks, n_days].
        Weights sum to 1.0 for each day (column).
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
    print(f" ... stockList={stockList}")  # Debug stockList

    # Identity and symbol-order diagnostics to help track whether the
    # array returned by the rolling filter is the same object received
    # here by the selector.
    try:
        print(f"DEBUG ids in selector: id(signal2D)={id(signal2D)}, id(signal2D_daily)={id(signal2D_daily)}")
        print(f"DEBUG ids in selector: id(signal_mask)={id(signal_mask)}")
        if symbols is not None and len(symbols) > 0:
            print(f"DEBUG symbols[0:10]={symbols[:10]}")
            if 'JEF' in symbols:
                print(f"DEBUG: JEF index in selector = {symbols.index('JEF')}")
    except Exception:
        pass

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

    # Sanity assertions / diagnostics: ensure shapes align and that
    # the mask reflects the filtered daily/monthly signals. Print
    # JEF-specific diagnostics at month starts to track where zeros
    # should have been applied by the rolling filter.
    try:
        assert signal2D.shape == signal_mask.shape, (
            f"signal2D shape {signal2D.shape} != signal_mask shape {signal_mask.shape}"
        )
        assert signal2D_daily.shape == signal2D.shape, (
            f"signal2D_daily shape {signal2D_daily.shape} != signal2D shape {signal2D.shape}"
        )
    except AssertionError:
        print("DIAG ASSERTION FAILED: shape mismatch between signals and mask")
        raise

    jef_idx = symbols.index("JEF") if "JEF" in symbols else None
    if jef_idx is not None:
        # Print a short diagnostic line at the start of every month
        for _j in range(n_days):
            try:
                if _j == 0 or datearray[_j].month != datearray[_j-1].month:
                    daily_val = float(signal2D_daily[jef_idx, _j])
                    month_val = float(signal2D[jef_idx, _j])
                    mask_val = float(signal_mask[jef_idx, _j])
                    print(f" ... DIAG: Date {datearray[_j]} JEF daily={daily_val:.6f} month={month_val:.6f} mask={mask_val:.1f}")
            except Exception:
                # Be robust to date types and continue
                try:
                    print(f" ... DIAG: Date index {_j} JEF entries: daily={signal2D_daily[jef_idx,_j]} month={signal2D[jef_idx,_j]} mask={signal_mask[jef_idx,_j]}")
                except Exception:
                    pass

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
                    print(f" ... Date {datearray[j]}: Early period (2000-2002), all signals zero, assigning 100% to CASH")
                else:
                    # Fallback: equal weights to all stocks if no CASH symbol
                    equal_weight = 1.0 / n_stocks
                    monthgainlossweight[:, j] = equal_weight
                    print(f" ... Date {datearray[j]}: Early period but no CASH symbol, assigning equal weights to all {n_stocks} stocks")
            else:
                # For non-early period: assign equal weights to all stocks as fallback
                equal_weight = 1.0 / n_stocks
                monthgainlossweight[:, j] = equal_weight
                print(f" ... Date {datearray[j]}: All signals zero (non-early period), assigning equal weights to all {n_stocks} stocks")
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
        
        if j < 5:  # Debug first 5 dates
            print(f" ... Date {datearray[j]}: year={year}, stockList={stockList}, is_early_period={is_early_period}")

        # Debug: print number of eligible stocks at beginning of each month
        if j == 0 or datearray[j].month != datearray[j-1].month:
            print(f" ... DEBUG: Date {datearray[j]}: {np.sum(eligible_stocks)} eligible stocks")

        if np.sum(eligible_stocks) == 0:
            # No eligible stocks found
            if is_early_period:
                # For early period: assign weight 1.0 to CASH, 0.0 to all other stocks
                cash_idx = symbols.index('CASH') if 'CASH' in symbols else None
                if cash_idx is not None:
                    monthgainlossweight[:, j] = 0.0  # Zero all weights first
                    monthgainlossweight[cash_idx, j] = 1.0  # 100% in cash
                    print(f" ... Date {datearray[j]}: Early period (2000-2002), no eligible stocks, assigning 100% to CASH")
                else:
                    # Fallback: equal weights to all stocks if no CASH symbol
                    equal_weight = 1.0 / n_stocks
                    monthgainlossweight[:, j] = equal_weight
                    print(f" ... Date {datearray[j]}: Early period but no CASH symbol, assigning equal weights to all {n_stocks} stocks")
            else:
                # For non-early period: assign equal weights to all stocks as fallback
                equal_weight = 1.0 / n_stocks
                monthgainlossweight[:, j] = equal_weight
                print(f" ... Date {datearray[j]}: No eligible stocks (non-early period), assigning equal weights to all {n_stocks} stocks")
            continue

        # If we're in the early period, force 100% CASH even if there are eligible stocks
        # (algorithm warm-up period)
        if is_early_period:
            cash_idx = symbols.index('CASH') if 'CASH' in symbols else None
            if cash_idx is not None:
                monthgainlossweight[:, j] = 0.0  # Zero all weights first
                monthgainlossweight[cash_idx, j] = 1.0  # 100% in cash
                if j < 5:  # Debug first 5 dates
                    print(f" ... Date {datearray[j]}: Early period (2000-2002), forcing 100% to CASH despite eligible stocks")
                continue
            else:
                print(f" ... Date {datearray[j]}: Early period but no CASH symbol, proceeding with normal selection")

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

        # Debug print for last date
        if j == n_days - 1:
            print(f"Debug: Date {datearray[j]}, eligible stocks: {len(eligible_indices)}, num_to_select: {num_to_select}")
            print(f"Debug: Selected Sharpe range: min={selected_sharpe.min():.4f}, max={selected_sharpe.max():.4f}")
            if len(selected_sharpe) > 1:
                print(f"Debug: All selected Sharpe equal: {np.allclose(selected_sharpe, selected_sharpe[0])}")
                print(f"Debug: Raw weights before constraints: {raw_weights}")

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

            # Debug print constrained weights
            if j == n_days - 1:
                print(f"Debug: Equal weight: {equal_weight:.4f}, min_weight: {min_weight:.4f}, max_weight: {max_weight:.4f}")
                print(f"Debug: Constrained weights: {constrained_weights}")
                print(f"Debug: Weight range: min={constrained_weights.min():.4f}, max={constrained_weights.max():.4f}")

            # Assign constrained weights
            monthgainlossweight[selected_stocks, j] = constrained_weights
            # Debug: print selected symbols and their assigned weights at
            # the beginning of each month (selection/rebalance points).
            try:
                if j == 0 or datearray[j].month != datearray[j-1].month:
                    sel_list = [(symbols[idx], float(w)) for idx, w in zip(selected_stocks, constrained_weights)]
                    sel_text = ", ".join([f"{s}:{wt:.4f}" for s, wt in sel_list])
                    print(f" ... SELECTION DEBUG: Date {datearray[j]} -> {len(sel_list)} selected: {sel_text}")
                    # Also print JEF specifically if present
                    if "JEF" in symbols:
                        jef_idx = symbols.index("JEF")
                        jef_w = float(monthgainlossweight[jef_idx, j])
                        if jef_w > 0:
                            print(f" ... SELECTION DEBUG: JEF selected with weight {jef_w:.4f} on {datearray[j]}")
            except Exception:
                pass
        else:
            # Simple proportional weighting without constraints
            monthgainlossweight[selected_stocks, j] = raw_weights
            # Debug: print selected symbols and their assigned weights at
            # the beginning of each month (selection/rebalance points).
            try:
                if j == 0 or datearray[j].month != datearray[j-1].month:
                    sel_list = [(symbols[idx], float(w)) for idx, w in zip(selected_stocks, raw_weights)]
                    sel_text = ", ".join([f"{s}:{wt:.4f}" for s, wt in sel_list])
                    print(f" ... SELECTION DEBUG: Date {datearray[j]} -> {len(sel_list)} selected: {sel_text}")
                    if "JEF" in symbols:
                        jef_idx = symbols.index("JEF")
                        jef_w = float(monthgainlossweight[jef_idx, j])
                        if jef_w > 0:
                            print(f" ... SELECTION DEBUG: JEF selected with weight {jef_w:.4f} on {datearray[j]}")
            except Exception:
                pass

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

    print(" ... DEBUG: Show non-zero weights at beginning of each month...")
    for j in range(n_days):
        if j == 0 or datearray[j].month != datearray[j-1].month:
            print(f" ... DEBUG: Date {datearray[j]}: {np.sum(monthgainlossweight[:, j] > 0)} non-zero weights")

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


def multiSharpe( datearray, adjClose, periods ):

    from allstats import allstats

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


def sharpeWeightedRank_2D_old(
        json_fn: str, 
        datearray: NDArray[np.datetime64],
        symbols: list[str],
        adjClose: NDArray[np.floating],
        signal2D: NDArray[np.floating],
        signal2D_daily: NDArray[np.floating],
        LongPeriod: int,
        rankthreshold: int,
        riskDownside_min: float,
        riskDownside_max: float,
        rankThresholdPct: float,
        stddevThreshold: float = 4.,
        is_backtest: bool = True, 
        makeQCPlots: bool = False, 
        verbose: bool = False
) -> tuple:
    """Rank stocks by Sharpe ratio and performance metrics for portfolio selection.
    
    This is one of the most critical functions in PyTAAA. It ranks stocks based on
    risk-adjusted returns (Sharpe ratio) over a lookback period, applies uptrend
    signals, and selects the top performers for portfolio allocation. Also tracks
    uptrending stock counts and can generate quality control plots.
    
    The function:
    1. Removes price spikes from data
    2. Calculates daily and period returns
    3. Applies trend signals to filter stocks
    4. Ranks stocks by performance metrics
    5. Selects top performers for portfolio
    6. Tracks and logs daily uptrending stock counts
    
    Args:
        json_fn: Path to JSON configuration file containing parameters
        datearray: 1D array of dates corresponding to price data
        symbols: List of stock ticker symbols
        adjClose: 2D array (n_stocks, n_dates) of adjusted closing prices
        signal2D: 2D array (n_stocks, n_dates) of trend signals (1=uptrend, 0=downtrend)
        signal2D_daily: 2D array used for daily uptrending count tracking
        LongPeriod: Lookback period in days for performance calculation
        rankthreshold: Number of top-ranked stocks to select for portfolio
        riskDownside_min: Minimum acceptable downside risk level
        riskDownside_max: Maximum acceptable downside risk level
        rankThresholdPct: Percentage threshold for rank-based selection
        stddevThreshold: Number of standard deviations for spike detection (default 4.0)
        is_backtest: Whether this is a backtest run (affects certain behaviors)
        makeQCPlots: Whether to generate quality control diagnostic plots
        verbose: Whether to print detailed debug information
        
    Returns:
        tuple containing multiple portfolio metrics and selections (exact structure
        depends on the full implementation - returns various arrays and dataframes
        related to stock rankings, weights, and performance metrics)
        
    Example:
        >>> # Typically called from main portfolio optimization routine
        >>> results = sharpeWeightedRank_2D(
        ...     json_fn=\"config.json\",
        ...     datearray=dates,
        ...     symbols=[\"AAPL\", \"MSFT\", \"GOOGL\"],
        ...     adjClose=price_data,
        ...     signal2D=signals,
        ...     signal2D_daily=daily_signals,
        ...     LongPeriod=60,
        ...     rankthreshold=10,
        ...     riskDownside_min=0.8,
        ...     riskDownside_max=1.2,
        ...     rankThresholdPct=0.1
        ... )
        
    Note:
        - Uses despike_2D to remove outliers before calculations
        - Tracks daily uptrending stock counts in a text file
        - Applies signal2D as a filter (non-uptrending stocks get 1.0 gain/loss)
        - Uses bottleneck.rankdata for efficient ranking
        - Prints extensive diagnostic information during execution
        - Reads additional parameters from JSON configuration file
        
    Raises:
        FileNotFoundError: If JSON configuration file not found
        KeyError: If required parameters missing from JSON config
        
    See Also:
        computeSignal2D: Generates the signal2D input
        despike_2D: Used internally for outlier removal
        MAA_WeightedRank_2D: Alternative ranking method
    """

    # adjClose      --     # 2D array with adjusted closing prices (axes are stock number, date)
    # rankthreshold --     # select this many funds with best recent performance

    import numpy as np
    # import nose
    import os
    import sys
    from matplotlib import pylab as plt
    import matplotlib.gridspec as gridspec
    try:
        import bottleneck as bn
        from bn import rankdata as rd
    except:
        import scipy.stats.mstats as bn
    # from functions.quotes_for_list_adjClose import get_Naz100List, get_SP500List, get_ETFList
    from functions.GetParams import get_json_params, get_symbols_file
    from functions.calculateTrades import trade_today

    start_time = datetime.datetime.now()

    # Get params for sending textmessage and email
    params = get_json_params(json_fn)
    symbols_file = get_symbols_file(json_fn)
    stockList = params['stockList']

    # print params (one line per value)
    print(" ... parameters used in sharpeWeightedRank_2D: ")
    for key, value in params.items():
        print(f"{key}: {value}")

    adjClose_despike = despike_2D( adjClose, LongPeriod, stddevThreshold=stddevThreshold )

    gainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    #gainloss[:,1:] = adjClose[:,1:] / adjClose[:,:-1]
    gainloss[:,1:] = adjClose_despike[:,1:] / adjClose_despike[:,:-1]  ## experimental
    gainloss[isnan(gainloss)]=1.

    # Create a binary mask from `signal2D` and do not modify the input array.
    # This preserves any zeros set by upstream filters (e.g. rolling window).
    signal_mask = (signal2D > 0).astype(float)

    # apply signal mask to daily gainloss
    print("\n\n\n######################\n...gainloss min,median,max = ",gainloss.min(),gainloss.mean(),np.median(gainloss),gainloss.max())
    print("...signal_mask min,median,max = ",signal_mask.min(),signal_mask.mean(),np.median(signal_mask),signal_mask.max(),"\n\n\n")
    gainloss = gainloss * signal_mask
    gainloss[gainloss == 0] = 1.0

    # update file with daily count of uptrending symbols in index universe
    json_dir = os.path.split(json_fn)[0]
    webpage_dir = get_webpage_store(json_fn)
    filepath = os.path.join(webpage_dir, "pyTAAAweb_dailyNumberUptrendingSymbolsList.txt" )
    print("\n\nfile for daily number of uptrending symbols = ", filepath)
    if os.path.exists( os.path.abspath(filepath) ):
        numberUptrendingSymbols = 0
        for i in range(len(symbols)):
            if signal2D_daily[i,-1] == 1.:
                numberUptrendingSymbols += 1
                #print "numberUptrendingSymbols,i,symbol,signal2D = ",numberUptrendingSymbols,i,symbols[i],signal2D_daily[i,-1]

        dailyUptrendingCount_text = "\n"+str(datearray[-1])+", "+str(numberUptrendingSymbols)
        with open( filepath, "a" ) as f:
            f.write(dailyUptrendingCount_text)
    else:
        dailyUptrendingCount_text = "date, daily count of uptrending symbols"
        with open( filepath, "w" ) as f:
            f.write(dailyUptrendingCount_text)
        numberUptrendingSymbols = np.zeros( signal2D_daily.shape[1], 'int' )
        for k in range(signal2D_daily.shape[1]):
            for i in range(len(symbols)):
                if signal2D_daily[i,k] == 1.:
                    numberUptrendingSymbols[k] += 1
                    #print "numberUptrendingSymbols,i,symbol,signal2D = ",numberUptrendingSymbols[k],datearray[k],symbols[i],signal2D_daily[i,k]

            dailyUptrendingCount_text = "\n"+str(datearray[k])+", "+str(numberUptrendingSymbols[k])
            with open( filepath, "a" ) as f:
                f.write(dailyUptrendingCount_text)

    value = 10000. * np.cumprod(gainloss,axis=1)
    print("TAFunctions.sharpeWeightedRank_2D ... value computed ...")

    # calculate gainloss over period of "LongPeriod" days
    monthgainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    #monthgainloss[:,LongPeriod:] = adjClose[:,LongPeriod:] / adjClose[:,:-LongPeriod]
    monthgainloss[:,LongPeriod:] = adjClose_despike[:,LongPeriod:] / adjClose_despike[:,:-LongPeriod]  ## experimental
    monthgainloss[isnan(monthgainloss)]=1.

    # apply signal mask to period monthgainloss
    monthgainloss = monthgainloss * signal_mask
    monthgainloss[monthgainloss == 0] = 1.0

    monthgainlossweight = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)

    rankweight = 1./rankthreshold
    print("TAFunctions.sharpeWeightedRank_2D ... rankweight computed ...")


    ########################################################################
    ## Calculate change in rank of active stocks each day (without duplicates as ties)
    ## FIX: Rank stocks independently at each date to avoid look-ahead bias
    ########################################################################
    monthgainlossRank = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=int)
    monthgainlossPrevious = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    monthgainlossPreviousRank = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=int)

    # FIX: Rank each date independently using only data available at that date
    for jj in range(monthgainloss.shape[1]):
        monthgainlossRank[:, jj] = bn.rankdata(monthgainloss[:, jj])
        # reverse the ranks (low ranks are biggest gainers)
        # Use maxrank from THIS date only, not global maxrank
        maxrank_jj = np.max(monthgainlossRank[:, jj])
        monthgainlossRank[:, jj] -= maxrank_jj - 1
        monthgainlossRank[:, jj] *= -1
        monthgainlossRank[:, jj] += 2
    print("TAFunctions.sharpeWeightedRank_2D ... monthgainlossRank computed (point-in-time) ...")

    monthgainlossPrevious[:,LongPeriod:] = monthgainloss[:,:-LongPeriod]
    
    # FIX: Rank each date independently for previous period as well
    for jj in range(monthgainlossPrevious.shape[1]):
        monthgainlossPreviousRank[:, jj] = bn.rankdata(monthgainlossPrevious[:, jj])
        # reverse the ranks (low ranks are biggest gainers)
        # Use maxrank from THIS date only, not global maxrank
        maxrank_jj = np.max(monthgainlossPreviousRank[:, jj])
        monthgainlossPreviousRank[:, jj] -= maxrank_jj - 1
        monthgainlossPreviousRank[:, jj] *= -1
        monthgainlossPreviousRank[:, jj] += 2
    print("TAFunctions.sharpeWeightedRank_2D ... monthgainlossPreviousRank computed (point-in-time) ...")

    # weight deltaRank for best and worst performers differently
    rankoffsetchoice = rankthreshold
    delta = -( monthgainlossRank.astype('float') - monthgainlossPreviousRank.astype('float') ) / ( monthgainlossRank.astype('float') + float(rankoffsetchoice) )
    print("TAFunctions.sharpeWeightedRank_2D ... delta computed ...")

    # if rank is outside acceptable threshold, set deltarank to zero so stock will not be chosen
    #  - remember that low ranks are biggest gainers
    rankThreshold = (1. - rankThresholdPct) * ( monthgainlossRank.max() - monthgainlossRank.min() )
    print("TAFunctions.sharpeWeightedRank_2D ... monthgainloss.shape[0] = "+str(monthgainloss.shape[0]))
    print("TAFunctions.sharpeWeightedRank_2D ... monthgainloss.shape[1] = "+str(monthgainloss.shape[1]))
    for ii in range(monthgainloss.shape[0]):
        for jj in range(monthgainloss.shape[1]):
            if monthgainloss[ii,jj] > rankThreshold :
                delta[ii,jj] = -monthgainloss.shape[0]/2
                if verbose:
                    if jj == monthgainloss.shape[1]:
                        print(
                            "*******setting delta (Rank) low... " +\
                            "Stock has rank outside acceptable range... ",
                            ii, symbols[ii], monthgainloss[ii,jj]
                        )
    print("TAFunctions.sharpeWeightedRank_2D ... looping completed ...")

    # if symbol is not in current stock index universe, set deltarank to zero so stock will not be chosen
    #  - remember that low ranks are biggest gainers
    json_dir = os.path.split(json_fn)[0]
    symbol_directory = os.path.join(json_dir, "symbols")

    if stockList == 'Naz100':
        # currentSymbolList,_,_ = get_Naz100List(symbols_file)
        currentSymbolList = read_symbols_list_local(json_fn)
    elif stockList == 'SP500':
        # currentSymbolList,_,_ = get_SP500List(symbols_file)
        currentSymbolList = read_symbols_list_local(json_fn)
    else:
        # Default fallback for other stock lists
        currentSymbolList = read_symbols_list_local(json_fn)
    # elif stockList == 'ETF':
    #     currentSymbolList,_,_ = get_ETFList()

    print("TAFunctions.sharpeWeightedRank_2D ... monthgainloss.shape[0] = "+str(monthgainloss.shape[0]))
    rankThreshold = (1. - rankThresholdPct) * ( monthgainlossRank.max() - monthgainlossRank.min() )
    for ii in range(monthgainloss.shape[0]):
        if symbols[ii] not in currentSymbolList and symbols[ii] != 'CASH' :
            delta[ii,:] = -monthgainloss.shape[0]/2
            numisnans = adjClose[ii,:]
            # NaN in last value usually means the stock is removed from the index so is not updated, but history is still in HDF file
            if verbose:
                print(
                    "*******setting delta (Rank) low... " +\
                    "Stock is no longer in stock index universe... ",
                    ii, symbols[ii]
                )
    print("TAFunctions.sharpeWeightedRank_2D ... 2nd loop completed ...")


    # FIX: Rank deltaRank independently per date to avoid look-ahead bias
    deltaRank = np.zeros_like(delta, dtype=int)
    for jj in range(delta.shape[1]):
        deltaRank[:, jj] = bn.rankdata(delta[:, jj])
        # reverse the ranks (low deltaRank have the fastest improving rank)
        # Use maxrank from THIS date only
        maxrank_jj = np.max(deltaRank[:, jj])
        deltaRank[:, jj] -= maxrank_jj - 1
        deltaRank[:, jj] *= -1
        deltaRank[:, jj] += 2
    print("TAFunctions.sharpeWeightedRank_2D ... deltaRank computed (point-in-time) ...")

    for ii in range(monthgainloss.shape[1]):
        if deltaRank[:,ii].min() == deltaRank[:,ii].max():
            deltaRank[:,ii] = 0.

    ########################################################################
    ## Copy current day rankings deltaRankToday
    ########################################################################

    deltaRankToday = deltaRank[:,-1].copy()

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
    ### Calculate downside risk measure for weighting stocks.
    ### Use 1./ movingwindow_sharpe_ratio for risk measure.
    ### Modify weights with 1./riskDownside and scale so they sum to 1.0
    ########################################################################

    riskDownside = 1. / move_sharpe_2D(adjClose,gainloss,LongPeriod)
    riskDownside = np.clip( riskDownside, riskDownside_min, riskDownside_max)

    riskDownside[isnan(riskDownside)] = np.max(riskDownside[~isnan(riskDownside)])
    for ii in range(riskDownside.shape[0]) :
        riskDownside[ii] = riskDownside[ii] / np.sum(riskDownside,axis=0)

    ########################################################################
    ### calculate equal weights for ranks below threshold
    ########################################################################

    # Define minimum active stocks threshold for portfolio allocation
    # Use different thresholds based on index size
    if stockList == 'SP500':
        min_active_stocks_threshold = 250
    elif stockList == 'Naz100':
        min_active_stocks_threshold = 50
    else:
        min_active_stocks_threshold = 50  # Default fallback

    # Early period logic: For SP500 backtests, if all signals are zero during 2000-2002
    # (algorithm warm-up period), allocate 100% to CASH to maintain portfolio value
    for ii in np.arange(1, monthgainloss.shape[1]):
        # Check if all signals are zero (no stocks have valid signals)
        # Use the binary mask so upstream filters are respected.
        all_signals_zero = np.sum(signal_mask[:, ii] > 0) == 0
        
        if all_signals_zero:
            # Check if we're in the early period (2000-2002) where signals are expected to be zero
            # due to warm-up period. In this case, put everything in CASH.
            # This logic only applies to SP500, not Naz100.
            current_date = datearray[ii]
            # Handle different date formats
            if hasattr(current_date, 'year'):
                year = current_date.year
            elif hasattr(current_date, 'item'):  # numpy datetime64
                year = current_date.item().year
            else:
                year = int(str(current_date)[:4])  # fallback
            
            is_early_period = (year >= 2000 and year <= 2002) and (stockList == "SP500")
            
            if is_early_period:
                # For early period: assign weight 1.0 to CASH, 0.0 to all other stocks
                try:
                    cash_index = symbols.index('CASH')
                    monthgainlossweight[:, ii] = 0.0  # Zero all weights first
                    monthgainlossweight[cash_index, ii] = 1.0  # 100% in cash
                    print(f" ... Date {datearray[ii]}: Early period (2000-2002), all signals zero, assigning 100% to CASH")
                    continue  # Skip the normal weight assignment logic
                except ValueError:
                    # Fallback: equal weights to all stocks if no CASH symbol
                    equal_weight = 1.0 / monthgainloss.shape[0]
                    monthgainlossweight[:, ii] = equal_weight
                    print(f" ... Date {datearray[ii]}: Early period but no CASH symbol, assigning equal weights to all {monthgainloss.shape[0]} stocks")
                    continue

    elsecount = 0
    elsedate  = 0
    for ii in np.arange(1,monthgainloss.shape[1]) :
        if activeCount[ii] > minrank[ii] and rankthresholdpercentequiv[ii] > 0:
            for jj in range(value.shape[0]):
                test = deltaRank[jj,ii] <= rankthresholdpercentequiv[ii]
                if test == True :
                    monthgainlossweight[jj,ii]  = 1./rankthresholdpercentequiv[ii]
                    monthgainlossweight[jj,ii]  = monthgainlossweight[jj,ii] / riskDownside[jj,ii]
                else:
                    monthgainlossweight[jj,ii]  = 0.
        elif activeCount[ii] < min_active_stocks_threshold :
            # When insufficient active stocks, allocate 100% to CASH
            monthgainlossweight[:,ii]  = 0.
            cash_index = symbols.index("CASH")
            monthgainlossweight[cash_index,ii]  = 1.0
            print(f"Insufficient active stocks ({activeCount[ii]} < {min_active_stocks_threshold}) on {datearray[ii]}, allocating 100% to CASH")
        else :
            elsedate = datearray[ii]
            elsecount += 1
            monthgainlossweight[:,ii]  = 1./activeCount[ii]

    aaa = np.sum(monthgainlossweight,axis=0)


    allzerotest = np.sum(monthgainlossweight,axis=0)
    sumallzerotest = allzerotest[allzerotest == 0].size
    if sumallzerotest > 0:
        print("")
        print(" invoking correction to monthgainlossweight.....")
        print("")
        for ii in np.arange(1,monthgainloss.shape[1]) :
            if np.sum(monthgainlossweight[:,ii]) == 0:
                monthgainlossweight[:,ii]  = 1./activeCount[ii]

    print(" weights calculation else clause encountered :",elsecount," times. last date encountered is ",elsedate)
    rankweightsum = np.sum(monthgainlossweight,axis=0)

    monthgainlossweight[isnan(monthgainlossweight)] = 10.e15  # changed result from 1 to 0, changed again to 10.e15
    monthgainlossweight[monthgainlossweight==0.] = 1.e-15

    monthgainlossweight = monthgainlossweight / np.sum(monthgainlossweight,axis=0)
    monthgainlossweight[isnan(monthgainlossweight)] = 0.  # changed result from 1 to 0
    monthgainlossweight[monthgainlossweight<1.e-3] = 0.  # changed result from 1 to 0


    ### TODO: make sure re-rodering this loop works. If not. delete repeated function -- start
    monthgainlossweightToday = monthgainlossweight[:,-1].copy()

    ########################################################################
    ## Hold values constant for calendar month (gains, ranks, deltaRanks)
    ########################################################################

    try:
        from functions.readSymbols import read_company_names_local
        companySymbolList, companyNameList = read_company_names_local(
            json_fn, verbose=False
        )
    except Exception as e:
        try:
            from functions.readSymbols import read_symbols_list_web    
            companyNameList, companySymbolList = read_symbols_list_web(
                json_fn, verbose=False
            )
        except Exception as e2:
            # If both methods fail, create empty lists to continue
            print(f"Warning: Could not fetch company symbols or names: {e2}")
            companySymbolList = []
            companyNameList = []

    for ii in range(1,monthgainloss.shape[1]):
        if datearray[ii].month == datearray[ii-1].month:
            monthgainloss[:,ii] = monthgainloss[:,ii-1]
            delta[:,ii] = delta[:,ii-1]
            deltaRank[:,ii] = deltaRank[:,ii-1]

    rank_today_text = "<div id='rank_table_container'><h3>"+ \
        "<p>Current stocks, with ranks, weights, and prices are :</p></h3> + \
        <font face='courier new' size=3><table border='1'> \
        <tr> \
        <td>Symbol \
        </td><td>Rank (today) \
        </td><td>Weight (today) \
        </td><td>Price  \
        </td><td>Trend  \
        </td><td>Company \
        </td></tr>\n"

    print("\n\n ... Tdoay's rankings and weights:")
    symbols_today = []
    weight_today = []
    price_today = []
    for i, isymbol in enumerate(symbols):
        for j in range(len(symbols)):

            if (
                    int( deltaRankToday[j] ) == i and \
                    monthgainlossweightToday[j] > 1.0e-3
            ):

                if signal2D_daily[j,-1] == 1.:
                    trend = 'up  '
                else:
                    trend = 'down'

                # search for company name
                try:
                    symbolIndex = companySymbolList.index(symbols[j])
                    companyName = companyNameList[symbolIndex]
                except:
                    companyName = ""

                rank_today_text = rank_today_text + \
                       "<tr><td>" + symbols[j].ljust(5)  + \
                       "<td>" + format(deltaRankToday[j],'6.0f')  + \
                       "<td>" + format(monthgainlossweightToday[j],'5.03f') + \
                       "<td>" + format(adjClose[j,-1],'7.2f')  + \
                       "<td>" + trend  + \
                       "<td>" + format(companyName,'15s')  + \
                       "</td></tr>  \n"
                row_text = symbols[j].ljust(6) + "  " + \
                format(deltaRankToday[j],'6.0f') + "  " + \
                format(monthgainlossweightToday[j],'6.04f') + "  " + \
                format(adjClose[j,-1],'6.2f') + "  " + \
                trend  + "  " + \
                format(companyName,'15s')
                print(row_text)
                symbols_today.append(symbols[j])
                weight_today.append(monthgainlossweightToday[j])
                price_today.append(adjClose[j,-1])
    print("\n\n\n")
    if not is_backtest:
        trade_today(json_fn, symbols_today, weight_today, price_today)
    ### TODO: make sure re-rodering this loop works. If not. delete repeated function -- end

    if makeQCPlots==True:
        json_dir = os.path.split(json_fn)[0]
        symbols_file = get_symbols_file(json_fn)
        symbols_dir = os.path.split(symbols_file)[0]
        # input symbols and company names from text file
        if stockList == 'Naz100':
            companyName_file = os.path.join(symbols_dir, "companyNames.txt")
        elif stockList == 'SP500':
            companyName_file = os.path.join(symbols_dir, "SP500_companyNames.txt")
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
                   <tr><td>Rank (start of month) \
                   </td><td>Rank (today) \
                   </td><td>Symbol \
                   </td><td>Company \
                   </td><td>Weight \
                   </td><td>Price  \
                   </td><td>Trend  \
                   </td><td>recent Gain or Loss (excludes a few days)  \
                   </td><td>stdDevs above or below trend  \
                   </td><td>trends ratio (%) with & wo gap  \
                   </td><td>P/E ratio \
                   </td></tr>\n"
        ChannelPct_text = "channelPercent:"
        channelPercent = []
        channelGainsLosses = []
        channelComboGainsLosses = []
        stdevsAboveChannel = []
        floatChannelGainsLosses = []
        floatChannelComboGainsLosses = []
        floatStdevsAboveChannel = []
        trendsRatio = []
        sharpeRatio = []
        floatSharpeRatio = []
        for i, isymbol in enumerate(symbols):
            import time
            symbol_start_time = time.time()

            xrange = range(params['minperiod'], params['maxperiod']+1, params['incperiod'])
            print(
                f"\nDEBUG: symbol={symbols[i]}, "
                f"x.shape={adjClose[i,:].shape}, "
                f"x.dtype={adjClose[i,:].dtype}, "
                f"xrange={(list(xrange))}"
            )

            ### save current projected position in price channel calculated without recent prices
            channelGainLoss, numStdDevs, pctChannel = recentTrendAndStdDevs(
                adjClose[i,:],
                datearray,
                minperiod=params['minperiod'],
                maxperiod=params['maxperiod'],
                incperiod=params['incperiod'],
                numdaysinfit=params['numdaysinfit'],
                offset=params['offset']
            )
            time1 = time.time()
            symbol_elapsed_time_01 = time1 - symbol_start_time

            print("\nsymbol = ", symbols[i])
            sharpe2periods = recentSharpeWithAndWithoutGap(adjClose[i,:])
            time2 = time.time()
            symbol_elapsed_time_12 = time2 - time1

            print(" ... performing PctChannelTest: symbol = ",format(isymbol,'5s'), "  numStdDevs = ", format(numStdDevs,'6.1f'))
            channelGainsLosses.append(format(channelGainLoss,'6.1%'))
            stdevsAboveChannel.append(format(numStdDevs,'6.1f'))
            floatChannelGainsLosses.append(channelGainLoss)
            floatStdevsAboveChannel.append(numStdDevs)
            ChannelPct_text = ChannelPct_text + format(pctChannel-1.,'6.1%')
            sharpeRatio.append(format(sharpe2periods,'6.1f'))
            floatSharpeRatio.append(sharpe2periods)
            print(
                "isymbol,floatSharpeRatio = ",
                isymbol,floatSharpeRatio[-1],
                flush=True
            )

            channelComboGainLoss = recentTrendComboGain(
                adjClose[i,:],
                datearray,
                minperiod=params['minperiod'],
                maxperiod=params['maxperiod'],
                incperiod=params['incperiod'],
                numdaysinfit=params['numdaysinfit'],
                offset=params['offset']
            )
            time3 = time.time()
            symbol_elapsed_time_23 = time3 - time2

            #print " companyName, channelComboGainLoss = ", companyNameList[i], channelComboGainLoss
            channelComboGainsLosses.append(format(channelComboGainLoss,'6.1%'))
            floatChannelComboGainsLosses.append(channelComboGainLoss)

            lowerTrend, upperTrend, NoGapLowerTrend, NoGapUpperTrend = \
                recentTrendAndMidTrendChannelFitWithAndWithoutGap( \
                    adjClose[i,:], \
                    minperiod=params['minperiod'], \
                    maxperiod=params['maxperiod'], \
                    incperiod=params['incperiod'], \
                    numdaysinfit=params['numdaysinfit'], \
                    numdaysinfit2=params['numdaysinfit2'], \
                    offset=params['offset']
                )
            midTrendEndPoint = (lowerTrend[-1]+upperTrend[-1])/2.
            noGapMidTrendEndPoint = (NoGapLowerTrend[-1]+NoGapUpperTrend[-1])/2.
            trendsRatio.append( noGapMidTrendEndPoint/midTrendEndPoint - 1. )

            time4 = time.time()
            symbol_elapsed_time_34 = time4 - time3
            symbol_elapsed_time = time4 - symbol_start_time
            print(
                f". inside functions/TAfunctions/sharpeWeightedRank_2D\n"
                f"[TIMING] Symbol {isymbol} processing steps: "
                f"[{symbol_elapsed_time_01}, {symbol_elapsed_time_12}, "
                f"{symbol_elapsed_time_23}, {symbol_elapsed_time_34}]\n"
                f"[TIMING] Symbol {isymbol} processing completed in "
                f"{symbol_elapsed_time:.4f} seconds"
            )
        print(" ... finished computing price positions within trend channels ...")

        json_dir = os.path.split(json_fn)[0]

        path_symbolChartsSort_byRankBeginMonth = os.path.join(webpage_dir, "pyTAAAweb_symbolCharts_MonthStartRank.html" )
        path_symbolChartsSort_byRankToday = os.path.join(webpage_dir, "pyTAAAweb_symbolCharts_TodayRank.html" )
        path_symbolChartsSort_byRecentGainRank = os.path.join(webpage_dir, "pyTAAAweb_symbolCharts_recentGainRank.html" )
        path_symbolChartsSort_byRecentComboGainRank = os.path.join(webpage_dir, "pyTAAAweb_symbolCharts_recentComboGainRank.html" )
        path_symbolChartsSort_byRecentTrendsRatioRank = os.path.join(webpage_dir, "pyTAAAweb_symbolCharts_recentTrendRatioRank.html" )
        path_symbolChartsSort_byRecentSharpeRatioRank = os.path.join(webpage_dir, "pyTAAAweb_symbolCharts_recentSharpeRatioRank.html" )

        pagetext_byRankBeginMonth = "<!DOCTYPE html>+\n"  +\
                               "<html>+\n"  +\
                               "<head>+\n"  +\
                               "<title>pyTAAA web</title>+\n"  +\
                               "</head>+\n"  +\
                               "<br><h1>Symbol Charts Ordered by Ranking at Start of Month</h1>+\n"
        pagetext_byRankToday = "<!DOCTYPE html>+\n"  +\
                               "<html>+\n"  +\
                               "<head>+\n"  +\
                               "<title>pyTAAA web</title>+\n"  +\
                               "</head>+\n"  +\
                               "<br><h1>Symbol Charts Ordered by Ranking Today</h1>+\n"
        pagetext_byRecentGainRank = "<!DOCTYPE html>+\n"  +\
                               "<html>+\n"  +\
                               "<head>+\n"  +\
                               "<title>pyTAAA web</title>+\n"  +\
                               "</head>+\n"  +\
                               "<br><h1>Symbol Charts Ordered by Recent Gain Ranking</h1>+\n"
        pagetext_byRecentComboGainRank = "<!DOCTYPE html>+\n"  +\
                               "<html>+\n"  +\
                               "<head>+\n"  +\
                               "<title>pyTAAA web</title>+\n"  +\
                               "</head>+\n"  +\
                               "<br><h1>Symbol Charts Ordered by Recent Combo Gain Ranking</h1>+\n"
        pagetext_byRecentTrendRatioRank = "<!DOCTYPE html>+\n"  +\
                               "<html>+\n"  +\
                               "<head>+\n"  +\
                               "<title>pyTAAA web</title>+\n"  +\
                               "</head>+\n"  +\
                               "<br><h1>Symbol Charts Ordered by Recent Trend Ratio Ranking</h1>+\n"
        pagetext_byRecentSharpeRatioRank = "<!DOCTYPE html>+\n"  +\
                               "<html>+\n"  +\
                               "<head>+\n"  +\
                               "<title>pyTAAA web</title>+\n"  +\
                               "</head>+\n"  +\
                               "<br><h1>Symbol Charts Ordered by Recent Sharpe Ratio Ranking</h1>+\n"

        floatChannelGainsLosses = np.array(floatChannelGainsLosses)
        floatChannelGainsLosses[np.isinf(floatChannelGainsLosses)] = -999.
        floatChannelGainsLosses[np.isneginf(floatChannelGainsLosses)] = -999.
        floatChannelGainsLosses[np.isnan(floatChannelGainsLosses)] = -999.
        floatChannelComboGainsLosses = np.array(floatChannelComboGainsLosses)
        floatChannelComboGainsLosses[np.isinf(floatChannelComboGainsLosses)] = -999.
        floatChannelComboGainsLosses[np.isneginf(floatChannelComboGainsLosses)] = -999.
        floatChannelComboGainsLosses[np.isnan(floatChannelComboGainsLosses)] = -999.
        floatStdevsAboveChannel = np.array(floatStdevsAboveChannel)
        floatStdevsAboveChannel[np.isinf(floatStdevsAboveChannel)] = -999.
        floatStdevsAboveChannel[np.isneginf(floatStdevsAboveChannel)] = -999.
        floatStdevsAboveChannel[np.isnan(floatStdevsAboveChannel)] = -999.
        floatTrendsRatio = np.array(trendsRatio)
        floatTrendsRatio[np.isinf(floatTrendsRatio)] = -999.
        floatTrendsRatio[np.isneginf(floatTrendsRatio)] = -999.
        floatTrendsRatio[np.isnan(floatTrendsRatio)] = -999.
        floatSharpeRatio = np.array(floatSharpeRatio)
        floatSharpeRatio[np.isinf(floatSharpeRatio)] = -999.
        floatSharpeRatio[np.isneginf(floatSharpeRatio)] = -999.
        floatSharpeRatio[np.isnan(floatSharpeRatio)] = -999.

        RecentGainRank = len(floatChannelGainsLosses) - bn.rankdata( floatChannelGainsLosses )
        RecentComboGainRank = len(floatChannelComboGainsLosses) - bn.rankdata( floatChannelComboGainsLosses )
        RecentGainStdDevRank = len(floatStdevsAboveChannel)- bn.rankdata( floatStdevsAboveChannel )
        RecentOrder = np.argsort( RecentGainRank + RecentGainStdDevRank )
        RecentRank = np.argsort( RecentOrder )
        RecentTrendsRatioRank = len(floatTrendsRatio) - bn.rankdata( floatTrendsRatio )
        RecentSharpeRatioRank = len(floatSharpeRatio) - bn.rankdata( floatSharpeRatio )

        print(" ... checking P/E ratios ...")

        peList = []
        floatPE_list = []
        for ticker in symbols:
            floatPE_list.append( get_pe(ticker) )
            peList.append( str(floatPE_list[-1]) )

        print(" ... P/E ratios downloaded...")

        # Initialize averaged variables in case no valid company names are found
        avgChannelGainsLosses = 0.0
        avgStdevsAboveChannel = 0.0
        avgTrendsRatio = 0.0
        validCompanyCount = 0

        for i, isymbol in enumerate(symbols):
            for j in range(len(symbols)):

                if int( deltaRank[j,-1] ) == i :
                    if signal2D_daily[j,-1] == 1.:
                        trend = 'up'
                    else:
                        trend = 'down'

                    # search for company name
                    try:
                        symbolIndex = companySymbolList.index(format(symbols[j],'5s'))
                        companyName = companyNameList[symbolIndex]
                    except:
                        companyName = ""

                    pe = peList[j]
                    rank_text = rank_text + \
                           "<tr><td>" + format(deltaRank[j,-1],'6.0f')  + \
                           "<td>" + format(deltaRankToday[j],'6.0f')  + \
                           "<td>" + format(symbols[j],'5s')  + \
                           "<td>" + format(companyName,'15s')  + \
                           "<td>" + format(monthgainlossweight[j,-1],'5.03f') + \
                           "<td>" + format(adjClose[j,-1],'6.2f')  + \
                           "<td>" + trend  + \
                           "<td>" + channelGainsLosses[j]  + \
                           "<td>" + stdevsAboveChannel[j]  + \
                           "<td>" + format(floatTrendsRatio[j],'4.1%') + \
                           "<td>" + pe  + \
                           "</td></tr>  \n"

                    if companyName != "":
                        if validCompanyCount == 0:
                            avgChannelGainsLosses = floatChannelGainsLosses[j]
                            avgStdevsAboveChannel = floatStdevsAboveChannel[j]
                        else:
                            avgChannelGainsLosses = (avgChannelGainsLosses*validCompanyCount+floatChannelGainsLosses[j])/(validCompanyCount+1)
                            avgStdevsAboveChannel = (avgStdevsAboveChannel*validCompanyCount+floatStdevsAboveChannel[j])/(validCompanyCount+1)
                        validCompanyCount += 1


                if i == deltaRank[j,-1]:
                    if signal2D_daily[j,-1] == 1.:
                        trend = 'up'
                    else:
                        trend = 'down'
                    # search for company name
                    try:
                        symbolIndex = companySymbolList.index(format(symbols[j],'5s'))
                        companyName = companyNameList[symbolIndex]
                    except:
                        companyName = ""
                    #pe = str(getQuote(symbols[j])['PE'][0])
                    pe = peList[j]
                    pagetext_byRankBeginMonth = pagetext_byRankBeginMonth +"<br><p> </p><p> </p><p> </p>"+\
                       "<font face='courier new' size=3><table border='1'>" +\
                       "<tr><td>Rank<br>(start of month)" +\
                       "</td><td>Rank<br>(today)" +\
                       "</td><td>Symbol" +\
                       "</td><td>Company" +\
                       "</td><td>Weight" +\
                       "</td><td>Price"  +\
                       "</td><td>Trend"  +\
                       "</td><td>recent<br>Gain or Loss<br>(excludes a few days)"  +\
                       "</td><td>stdDevs above<br>or below trend"  +\
                       "</td><td>trends ratio (%)<br>with & wo gap"  +\
                       "</td><td>P/E ratio" +\
                       "</td></tr>\n"+\
                       "<tr><td>" + format(deltaRank[j,-1],'6.0f')  + \
                       "<td>" + format(deltaRankToday[j],'6.0f')  + \
                       "<td>" + format(symbols[j],'5s')  + \
                       "<td>" + format(companyName,'15s')  + \
                       "<td>" + format(monthgainlossweight[j,-1],'5.03f') + \
                       "<td>" + format(adjClose[j,-1],'6.2f')  + \
                       "<td>" + trend  + \
                       "<td>" + channelGainsLosses[j]  + \
                       "<td>" + stdevsAboveChannel[j]  + \
                       "<td>" + format(floatTrendsRatio[j],'4.1%') + \
                       "<td>" + pe  + \
                       "</td></tr>  \n"+\
                       "<br><img src='0_recent_" +symbols[j]+ ".png' alt='PyTAAA by DonaldPG' width='1000' height='500'>"

                if i == deltaRankToday[j]:
                    if signal2D_daily[j,-1] == 1.:
                        trend = 'up'
                    else:
                        trend = 'down'
                    # search for company name
                    try:
                        symbolIndex = companySymbolList.index(format(symbols[j],'5s'))
                        companyName = companyNameList[symbolIndex]
                    except:
                        companyName = ""
                    #pe = str(getQuote(symbols[j])['PE'][0])
                    pe = peList[j]
                    pagetext_byRankToday = pagetext_byRankToday +"<br><p> </p><p> </p><p> </p><br>"+\
                       "<font face='courier new' size=3><table border='1'>" +\
                       "<tr><td>Rank <br>(start of month)" +\
                       "</td><td>Rank<br>(today)" +\
                       "</td><td>Symbol" +\
                       "</td><td>Company" +\
                       "</td><td>Weight" +\
                       "</td><td>Price"  +\
                       "</td><td>Trend"  +\
                       "</td><td>recent<br>Gain or Loss<br>(excludes a few days)"  +\
                       "</td><td>stdDevs above<br>or below trend"  +\
                       "</td><td>trends ratio (%)<br>with & wo gap"  +\
                       "</td><td>P/E ratio" +\
                       "</td></tr>\n"+\
                       "<tr><td>" + format(deltaRank[j,-1],'6.0f')  + \
                       "<td>" + format(deltaRankToday[j],'6.0f')  + \
                       "<td>" + format(symbols[j],'5s')  + \
                       "<td>" + format(companyName,'15s')  + \
                       "<td>" + format(monthgainlossweight[j,-1],'5.03f') + \
                       "<td>" + format(adjClose[j,-1],'6.2f')  + \
                       "<td>" + trend  + \
                       "<td>" + channelGainsLosses[j]  + \
                       "<td>" + stdevsAboveChannel[j]  + \
                       "<td>" + format(floatTrendsRatio[j],'4.1%') + \
                       "<td>" + pe  + \
                       "</td></tr>  \n"+\
                       "<br><img src='0_recent_" +symbols[j]+ ".png' alt='PyTAAA by DonaldPG' width='1000' height='500'>"

                if i == RecentRank[j]:
                    if signal2D_daily[j,-1] == 1.:
                        trend = 'up'
                    else:
                        trend = 'down'
                    # search for company name
                    try:
                        symbolIndex = companySymbolList.index(format(symbols[j],'5s'))
                        companyName = companyNameList[symbolIndex]
                    except:
                        companyName = ""
                    print((" ...at line 2193: companyName = "+companyName))
                    #pe = str(getQuote(symbols[j])['PE'][0])
                    pe = peList[j]
                    pagetext_byRecentGainRank = pagetext_byRecentGainRank +"<br><p> </p><p> </p><p> </p><br>"+\
                       "<font face='courier new' size=3><table border='1'>" +\
                       "<tr><td>Rank<br>(start of month)" +\
                       "</td><td>Rank<br>(today)" +\
                       "</td><td>Symbol" +\
                       "</td><td>Company" +\
                       "</td><td>Weight" +\
                       "</td><td>Price"  +\
                       "</td><td>Trend"  +\
                       "</td><td>recent<br>Gain or Loss<br>(excludes a few days)"  +\
                       "</td><td>stdDevs above<br>or below trend"  +\
                       "</td><td>trends ratio (%)<br>with & wo gap"  +\
                       "</td><td>P/E ratio" +\
                       "</td></tr>\n"+\
                       "<tr><td>" + format(deltaRank[j,-1],'6.0f')  + \
                       "<td>" + format(deltaRankToday[j],'6.0f')  + \
                       "<td>" + format(symbols[j],'5s')  + \
                       "<td>" + format(companyName,'15s')  + \
                       "<td>" + format(monthgainlossweight[j,-1],'5.03f') + \
                       "<td>" + format(adjClose[j,-1],'6.2f')  + \
                       "<td>" + trend  + \
                       "<td>" + channelGainsLosses[j]  + \
                       "<td>" + stdevsAboveChannel[j]  + \
                       "<td>" + format(floatTrendsRatio[j],'4.1%') + \
                       "<td>" + pe  + \
                       "</td></tr>  \n"+\
                       "<br><img src='0_recent_" +symbols[j]+ ".png' alt='PyTAAA by DonaldPG' width='1000' height='500'>"

                if i == RecentComboGainRank[j]:
                    if signal2D_daily[j,-1] == 1.:
                        trend = 'up'
                    else:
                        trend = 'down'
                    # search for company name
                    try:
                        symbolIndex = companySymbolList.index(format(symbols[j],'5s'))
                        companyName = companyNameList[symbolIndex]
                    except:
                        companyName = ""
                    # print((" ...at line 2236: companyName = "+companyName))
                    #pe = str(getQuote(symbols[j])['PE'][0])
                    pe = peList[j]
                    pagetext_byRecentComboGainRank = pagetext_byRecentComboGainRank +"<br><p> </p><p> </p><p> </p><br>"+\
                       "<font face='courier new' size=3><table border='1'>" +\
                       "<tr><td>Rank<br>(start of month)" +\
                       "</td><td>Rank<br>(today)" +\
                       "</td><td>Symbol" +\
                       "</td><td>Company" +\
                       "</td><td>Weight" +\
                       "</td><td>Price"  +\
                       "</td><td>Trend"  +\
                       "</td><td>recent<br>Combo Gain or Loss<br>(w and wo gap)"  +\
                       "</td><td>stdDevs above<br>or below trend"  +\
                       "</td><td>trends ratio (%)<br>with & wo gap"  +\
                       "</td><td>P/E ratio" +\
                       "</td></tr>\n"+\
                       "<tr><td>" + format(deltaRank[j,-1],'6.0f')  + \
                       "<td>" + format(deltaRankToday[j],'6.0f')  + \
                       "<td>" + format(symbols[j],'5s')  + \
                       "<td>" + format(companyName,'15s')  + \
                       "<td>" + format(monthgainlossweight[j,-1],'5.03f') + \
                       "<td>" + format(adjClose[j,-1],'6.2f')  + \
                       "<td>" + trend  + \
                       "<td>" + channelComboGainsLosses[j]  + \
                       "<td>" + stdevsAboveChannel[j]  + \
                       "<td>" + format(floatTrendsRatio[j],'4.1%') + \
                       "<td>" + pe  + \
                       "</td></tr>  \n"+\
                       "<br><img src='0_recent_" +symbols[j]+ ".png' alt='PyTAAA by DonaldPG' width='1000' height='500'>"

                if i == RecentTrendsRatioRank[j]:
                    if signal2D_daily[j,-1] == 1.:
                        trend = 'up'
                    else:
                        trend = 'down'
                    # search for company name
                    try:
                        symbolIndex = companySymbolList.index(format(symbols[j],'5s'))
                        companyName = companyNameList[symbolIndex]
                    except:
                        companyName = ""
                    #pe = str(getQuote(symbols[j])['PE'][0])
                    pe = peList[j]
                    pagetext_byRecentTrendRatioRank = pagetext_byRecentTrendRatioRank +"<br><p> </p><p> </p><p> </p><br>"+\
                       "<font face='courier new' size=3><table border='1'>" +\
                       "<tr><td>Rank<br>(start of month)" +\
                       "</td><td>Rank<br>(today)" +\
                       "</td><td>Symbol" +\
                       "</td><td>Company" +\
                       "</td><td>Weight" +\
                       "</td><td>Price"  +\
                       "</td><td>Trend"  +\
                       "</td><td>recent<br>Gain or Loss<br>(excludes a few days)"  +\
                       "</td><td>stdDevs above<br>or below trend"  +\
                       "</td><td>trends ratio (%)<br>with & wo gap"  +\
                       "</td><td>P/E ratio" +\
                       "</td></tr>\n"+\
                       "<tr><td>" + format(deltaRank[j,-1],'6.0f')  + \
                       "<td>" + format(deltaRankToday[j],'6.0f')  + \
                       "<td>" + format(symbols[j],'5s')  + \
                       "<td>" + format(companyName,'15s')  + \
                       "<td>" + format(monthgainlossweight[j,-1],'5.03f') + \
                       "<td>" + format(adjClose[j,-1],'6.2f')  + \
                       "<td>" + trend  + \
                       "<td>" + channelGainsLosses[j]  + \
                       "<td>" + stdevsAboveChannel[j]  + \
                       "<td>" + format(floatTrendsRatio[j],'4.1%') + \
                       "<td>" + pe  + \
                       "</td></tr>  \n"+\
                       "<br><img src='0_recent_" +symbols[j]+ ".png' alt='PyTAAA by DonaldPG' width='1000' height='500'>"

                if i == RecentSharpeRatioRank[j]:
                    if signal2D_daily[j,-1] == 1.:
                        trend = 'up'
                    else:
                        trend = 'down'
                    # search for company name
                    try:
                        symbolIndex = companySymbolList.index(format(symbols[j],'5s'))
                        companyName = companyNameList[symbolIndex]
                    except:
                        companyName = ""
                    #pe = str(getQuote(symbols[j])['PE'][0])
                    pe = peList[j]
                    pagetext_byRecentSharpeRatioRank = pagetext_byRecentSharpeRatioRank +"<br><p> </p><p> </p><p> </p><br>"+\
                       "<font face='courier new' size=3><table border='1'>" +\
                       "<tr><td>Rank<br>(start of month)" +\
                       "</td><td>Rank<br>(today)" +\
                       "</td><td>Symbol" +\
                       "</td><td>Company" +\
                       "</td><td>Weight" +\
                       "</td><td>Price"  +\
                       "</td><td>Sharpe"  +\
                       "</td><td>recent<br>Gain or Loss<br>(excludes a few days)"  +\
                       "</td><td>stdDevs above<br>or below trend"  +\
                       "</td><td>sharpe ratio (%)<br>multiple periods"  +\
                       "</td><td>P/E ratio" +\
                       "</td></tr>\n"+\
                       "<tr><td>" + format(deltaRank[j,-1],'6.0f')  + \
                       "<td>" + format(deltaRankToday[j],'6.0f')  + \
                       "<td>" + format(symbols[j],'5s')  + \
                       "<td>" + format(companyName,'15s')  + \
                       "<td>" + format(monthgainlossweight[j,-1],'5.0f') + \
                       "<td>" + format(adjClose[j,-1],'6.2f')  + \
                       "<td>" + trend  + \
                       "<td>" + channelGainsLosses[j]  + \
                       "<td>" + stdevsAboveChannel[j]  + \
                       "<td>" + format(floatSharpeRatio[j],'5.1f') + \
                       "<td>" + pe  + \
                       "</td></tr>  \n"+\
                       "<br><img src='0_recent_" +symbols[j]+ ".png' alt='PyTAAA by DonaldPG' width='1000' height='500'>"

        print(" ... html for web pages containing charts with various rankings created ...")

        medianChannelGainsLosses = np.median(floatChannelGainsLosses)
        medianTrendsRatio = np.median(floatTrendsRatio)
        avgTrendsRatio = np.mean(floatTrendsRatio)
        medianStdevsAboveChannel = np.median(floatStdevsAboveChannel)

        print("peList = ", floatPE_list)
        floatPE_list = np.array(floatPE_list)
        floatPE_list = floatPE_list[~np.isinf(floatPE_list)]
        floatPE_list = floatPE_list[~np.isneginf(floatPE_list)]
        floatPE_list = floatPE_list[~np.isnan(floatPE_list)]
        averagePE = np.mean(floatPE_list)
        medianPE = np.median(floatPE_list)

        print(" ... html created for hypothetical trades if executed today to re-align holdings with today's weights")
        p_store = get_performance_store(json_fn)
        filepath = os.path.join(p_store, "PyTAAA_hypothetical_trades.txt" )
        with open(filepath, "r" ) as f:
            lines = f.readlines()
        lines = [x for x in lines if x != "\n"]
        hypo_trade_text = ""
        for item in lines:
            _item = item.replace("\n", "<br>")
            for i in range(10):
                _item_ = _item.replace("  ", "&nbsp; ")
                if _item == _item_:
                    break
                else:
                    _item = _item_
            hypo_trade_text = hypo_trade_text + _item
        hypothetical_trades_html = "\n\n\n<font face='courier new' size=5>" +\
            "<p>PyTAAA_hypothetical_trades (if performed today):" +\
            "</p></h3><font face='courier new' size=4>"+\
            "<pre>" + hypo_trade_text + "</pre>" + \
            "</p></h3><font face='courier new' size=4>\n\n"

        avg_performance_text = "\n\n\n<font face='courier new' size=5><p>Average recent performance:</p></h3><font face='courier new' size=4>"+\
                               "<p>average trend excluding several days  = "+format(avgChannelGainsLosses,'6.1%')+"<br>"+\
                               "median trend excluding several days  = "+format(medianChannelGainsLosses,'6.1%')+"</p></h3><font face='courier new' size=4>"+\
                               "<p>average ratio of trend wo & with last several days  = "+format(avgTrendsRatio,'6.1%')+"<br>"+\
                               "median ratio of trend wo & with last several days  = "+format(medianTrendsRatio,'6.1%')+"</p></h3><font face='courier new' size=4>"+\
                               "<p>average number stds above/below trend = "+format(avgStdevsAboveChannel,'5.1f')+"<br>"+\
                               "median number stds above/below trend = "+format(medianStdevsAboveChannel,'5.1f')+"</p></h3><font face='courier new' size=4>"+\
                               "<p>average P/E = "+format(averagePE,'5.1f')+"<br>"+\
                               "median P/E = "+format(medianPE,'5.1f')+"</p></h3><font face='courier new' size=4>\n\n"

        rank_text = hypothetical_trades_html + avg_performance_text + rank_text + "</table></div>\n"

        print(" ... rank_text = " + str(rank_text))

        json_dir = os.path.split(json_fn)[0]
        p_store = get_performance_store(json_fn)

        filepath = os.path.join(webpage_dir, "pyTAAAweb_RankList.txt" )
        with open( filepath, "w" ) as f:
            f.write(rank_text)

        filepath = path_symbolChartsSort_byRankBeginMonth
        with open( filepath, "w" ) as f:
            f.write(pagetext_byRankBeginMonth)

        filepath = path_symbolChartsSort_byRankToday
        with open( filepath, "w" ) as f:
            f.write(pagetext_byRankToday)

        filepath = path_symbolChartsSort_byRecentGainRank
        with open( filepath, "w" ) as f:
            f.write(pagetext_byRecentGainRank)

        filepath = path_symbolChartsSort_byRecentComboGainRank
        with open( filepath, "w" ) as f:
            f.write(pagetext_byRecentComboGainRank)

        filepath = path_symbolChartsSort_byRecentTrendsRatioRank
        with open( filepath, "w" ) as f:
            f.write(pagetext_byRecentTrendRatioRank)

        filepath = path_symbolChartsSort_byRecentSharpeRatioRank
        with open( filepath, "w" ) as f:
            f.write(pagetext_byRecentSharpeRatioRank)

        ########################################################################
        ### save current ranks to params file
        ########################################################################

        lastdate_text = "lastdate: " + str(datearray[-1])
        symbol_text = "symbols: "
        rank_text = "ranks:"
        #####ChannelPct_text = "channelPercent:"

        for i, isymbol in enumerate(symbols):
            for j in range(len(symbols)):
                if int( deltaRank[j,-1] ) == i :
                    symbol_text = symbol_text + format(symbols[j],'6s')
                    rank_text = rank_text + format(deltaRankToday[j],'6.0f')

        filepath = os.path.join(p_store, "PyTAAA_ranks.params" )
        with open( filepath, "a" ) as f:
            f.write(lastdate_text)
            f.write("\n")
            f.write(symbol_text)
            f.write("\n")
            f.write(rank_text)
            f.write("\n")
            f.write(ChannelPct_text)
            f.write("\n")

    elapsed_ = datetime.datetime.now() - start_time
    print("\n   . elapsed time in sharpeWeightedRank_2D = " + str(elapsed_))
    print("leaving function sharpeWeightedRank_2D...")

    return monthgainlossweight


def MAA_WeightedRank_2D(
        json_fn, datearray, symbols, adjClose ,signal2D ,signal2D_daily,
        LongPeriod,numberStocksTraded,
        wR, wC, wV, wS, stddevThreshold=4.
):

    # adjClose      --     # 2D array with adjusted closing prices (axes are stock number, date)
    # rankthreshold --     # select this many funds with best recent performance

    import numpy as np
    import nose
    import os
    import sys
    from matplotlib import pylab as plt
    import matplotlib.gridspec as gridspec
    try:
        import bottleneck as bn
        from bn import rankdata as rd
    except:
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
    signal_mask = (signal2D > 0).astype(float)

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
                except:
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
def UnWeightedRank_2D(datearray,adjClose,signal2D,LongPeriod,rankthreshold,riskDownside_min,riskDownside_max,rankThresholdPct):

    # adjClose      --     # 2D array with adjusted closing prices (axes are stock number, date)
    # rankthreshold --     # select this many funds with best recent performance

    import numpy as np
    # import nose
    try:
        import bottleneck as bn
        from bn import rankdata as rd
    except:
        import scipy.stats.mstats as bn


    gainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    gainloss[:,1:] = adjClose[:,1:] / adjClose[:,:-1]
    gainloss[isnan(gainloss)]=1.

    # Create a binary mask from `signal2D` and do not modify the input array.
    signal_mask = (signal2D > 0).astype(float)

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


def hurst(X):
    """ Compute the Hurst exponent of X. If the output H=0.5,the behavior
    of the time-series is similar to random walk. If H<0.5, the time-series
    cover less "distance" than a random walk, vice verse.

    Parameters
    ----------
    X
        list
        a time series

    Returns
    -------
    H
        float
        Hurst exponent

    Examples
    --------
    >>> import pyeeg
    >>> from numpy.random import randn
    >>> a = randn(4096)
    >>> pyeeg.hurst(a)
    >>> 0.5057444

    ######################## Function contributed by Xin Liu #################
    https://code.google.com/p/pyeeg/source/browse/pyeeg.py
    Copyleft 2010 Forrest Sheng Bao http://fsbao.net
    PyEEG, a Python module to extract EEG features, v 0.02_r2
    Project homepage: http://pyeeg.org

    **Naming convention**

    Constants: UPPER_CASE_WITH_UNDERSCORES, e.g., SAMPLING_RATE, LENGTH_SIGNAL.
    Function names: lower_case_with_underscores, e.g., spectrum_entropy.
    Variables (global and local): CapitalizedWords or CapWords, e.g., Power.
    If a variable name consists of one letter, I may use lower case, e.g., x, y.

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


def textmessageOutsideTrendChannel(symbols, adjClose, json_fn):
    """
    Send text messages for stocks that are outside their trend channels.
    
    This function checks for stocks that have moved significantly outside
    their trend channels and sends text alerts if the market is open.
    
    Parameters
    ----------
    symbols : list
        List of stock symbols
    adjClose : np.ndarray
        Adjusted close prices [n_stocks, n_days]
    json_fn : str
        Path to JSON configuration file
        
    Returns
    -------
    None
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
