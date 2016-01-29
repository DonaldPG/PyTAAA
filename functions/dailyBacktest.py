
import time, threading

import numpy as np
import os
import datetime
from scipy.stats import rankdata
import nose
from scipy.stats import gmean
from math import sqrt

## local imports
from functions.quotes_for_list_adjClose import *
from functions.TAfunctions import *
from functions.UpdateSymbols_inHDF5 import UpdateHDF5, loadQuotes_fromHDF

def computeDailyBacktest( datearray, \
                         symbols, \
                         adjClose, \
                         numberStocksTraded=7, \
                         trade_cost=7.95, \
                         monthsToHold=4, \
                         LongPeriod=104, \
                         MA1=207, \
                         MA2=26, \
                         MA2offset=3, \
                         sma2factor=.911, \
                         rankThresholdPct=.02, \
                         riskDownside_min=.272, \
                         riskDownside_max=4.386, \
                         narrowDays=[6.,40.2], \
                         mediumDays=[25.2,38.3],\
                         wideDays=[75.2,512.3],\
                         stddevThreshold=4.0,
                         lowPct=17, \
                         hiPct=84, \
                         uptrendSignalMethod='uptrendSignalMethod' ):

    # put params in a dictionary
    params = {}
    params['numberStocksTraded'] = numberStocksTraded
    params['monthsToHold'] = monthsToHold
    params['LongPeriod'] = LongPeriod
    params['stddevThreshold'] = stddevThreshold
    params['MA1'] = MA1
    params['MA2'] = MA2
    params['MA2offset'] = MA2offset
    params['sma2factor'] = sma2factor
    params['MA2factor'] = sma2factor
    params['rankThresholdPct'] = rankThresholdPct
    params['riskDownside_min'] = riskDownside_min
    params['riskDownside_max'] = riskDownside_max
    params['narrowDays'] = narrowDays
    params['mediumDays'] = mediumDays
    params['wideDays'] = wideDays
    params['uptrendSignalMethod'] = uptrendSignalMethod
    params['lowPct'] = lowPct
    params['hiPct'] = hiPct

    MaxPortfolioValue = np.zeros(adjClose.shape[1],dtype=float)
    MaxBuyHoldPortfolioValue = np.zeros(adjClose.shape[1],dtype=float)
    numberStocksUpTrendingNearHigh = np.zeros( adjClose.shape[1], dtype=float)
    numberStocksUpTrendingBeatBuyHold = np.zeros( adjClose.shape[1], dtype=float)

    gainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    activeCount = np.zeros(adjClose.shape[1],dtype=float)

    gainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    activeCount = np.zeros(adjClose.shape[1],dtype=float)

    numberSharesCalc = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    gainloss[:,1:] = adjClose[:,1:] / adjClose[:,:-1]
    gainloss[isnan(gainloss)]=1.
    value = 10000. * np.cumprod(gainloss,axis=1)
    BuyHoldFinalValue = np.average(value,axis=0)[-1]

    print " gainloss check: ",gainloss[isnan(gainloss)].shape
    print " value check: ",value[isnan(value)].shape
    lastEmptyPriceIndex = np.zeros(adjClose.shape[0],dtype=int)


    for ii in range(adjClose.shape[0]):
        # take care of special case where constant share price is inserted at beginning of series
        index = np.argmax(np.clip(np.abs(gainloss[ii,:]-1),0,1e-8)) - 1
        print "first valid price and date = ",symbols[ii]," ",index," ",datearray[index]
        lastEmptyPriceIndex[ii] = index
        activeCount[lastEmptyPriceIndex[ii]+1:] += 1

    monthgainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    monthgainloss[:,LongPeriod:] = adjClose[:,LongPeriod:] / adjClose[:,:-LongPeriod]
    monthgainloss[isnan(monthgainloss)]=1.

    ####################################################################
    ###
    ### calculate signal for uptrending stocks (in signal2D)
    ### - method depends on params uptrendSignalMethod
    ###
    ####################################################################
    ###
    ### Use either 3 SMA's or channels to comput uptrending signal


    """
    # check the operatiing system to determine whether to move files or use ftp
    import platform

    operatingSystem = platform.system()
    architecture = platform.uname()[4]
    computerName = platform.uname()[1]

    print "  ...platform: ", operatingSystem, architecture, computerName

    if operatingSystem == 'Linux' and architecture == 'armv6l' :
        print "  ...using 3 SMA's for signal2D"
        ########################################################################
        ## Calculate signal for all stocks based on 3 simple moving averages (SMA's)
        ########################################################################
        sma0 = SMA_2D( adjClose, MA2 )               # MA2 is shortest
        sma1 = SMA_2D( adjClose, MA2 + MA2offset )
        sma2 = sma2factor * SMA_2D( adjClose, MA1 )  # MA1 is longest

        signal2D = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
        for ii in range(adjClose.shape[0]):
            for jj in range(adjClose.shape[1]):
                if adjClose[ii,jj] > sma2[ii,jj] or ((adjClose[ii,jj] > min(sma0[ii,jj],sma1[ii,jj]) and sma0[ii,jj] > sma0[ii,jj-1])):
                    signal2D[ii,jj] = 1
                    if jj== adjClose.shape[1]-1 and isnan(adjClose[ii,-1]):
                        signal2D[ii,jj] = 0                #### added to avoid choosing stocks no longer in index
            # take care of special case where constant share price is inserted at beginning of series
            index = np.argmax(np.clip(np.abs(gainloss[ii,:]-1),0,1e-8)) - 1

            signal2D[ii,0:index] = 0

        dailyNumberUptrendingStocks = np.sum(signal2D,axis = 0)

        # hold signal constant for each month
        for jj in np.arange(1,adjClose.shape[1]):
            #if not ((datearray[jj].month != datearray[jj-1].month) and (datearray[jj].month ==1 or datearray[jj].month == 5 or datearray[jj].month == 9)):
            if not ((datearray[jj].month != datearray[jj-1].month) and (datearray[jj].month - 1)%monthsToHold == 0 ):
                signal2D[:,jj] = signal2D[:,jj-1]
            else:
                if iter==0:
                    print "date, signal2D changed",datearray[jj]

        numberStocks = np.sum(signal2D,axis = 0)

    else:
        print "  ...using 3 minmax channels for signal2D"

        ########################################################################
        ## Calculate signal for all stocks based on 3 minmax channels (dpgchannels)
        ########################################################################

        # narrow channel is designed to remove day-to-day variability

        print "narrow days min,max,inc = ", narrowDays[0], narrowDays[-1], (narrowDays[-1]-narrowDays[0])/7.
        narrow_minChannel, narrow_maxChannel = dpgchannel_2D( adjClose, narrowDays[0], narrowDays[-1], (narrowDays[-1]-narrowDays[0])/7. )
        narrow_midChannel = (narrow_minChannel+narrow_maxChannel)/2.

        medium_minChannel, medium_maxChannel = dpgchannel_2D( adjClose, mediumDays[0], mediumDays[-1], (mediumDays[-1]-mediumDays[0])/7. )
        medium_midChannel = (medium_minChannel+medium_maxChannel)/2.
        mediumSignal = ((narrow_midChannel-medium_minChannel)/(medium_maxChannel-medium_minChannel)-0.5)*2.0

        wide_minChannel, wide_maxChannel = dpgchannel_2D( adjClose, wideDays[0], wideDays[-1], (wideDays[-1]-wideDays[0])/7. )
        wide_midChannel = (wide_minChannel+wide_maxChannel)/2.
        wideSignal = ((narrow_midChannel-wide_minChannel)/(wide_maxChannel-wide_minChannel)-0.5)*2.0

        signal2D = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
        for ii in range(adjClose.shape[0]):
            for jj in range(adjClose.shape[1]):
                if mediumSignal[ii,jj] + wideSignal[ii,jj] > 0:
                    signal2D[ii,jj] = 1
                    if jj== adjClose.shape[1]-1 and isnan(adjClose[ii,-1]):
                        signal2D[ii,jj] = 0                #### added to avoid choosing stocks no longer in index
            # take care of special case where constant share price is inserted at beginning of series
            index = np.argmax(np.clip(np.abs(gainloss[ii,:]-1),0,1e-8)) - 1

            signal2D[ii,0:index] = 0

            '''
            # take care of special case where mp quote exists at end of series
            if firstTrailingEmptyPriceIndex[ii] != 0:
                signal2D[ii,firstTrailingEmptyPriceIndex[ii]:] = 0
            '''

    # copy to daily signal
    signal2D_daily = signal2D.copy()

    # hold signal constant for each month
    for jj in np.arange(1,adjClose.shape[1]):
        #if not ((datearray[jj].month != datearray[jj-1].month) and (datearray[jj].month ==1 or datearray[jj].month == 5 or datearray[jj].month == 9)):
        if not ((datearray[jj].month != datearray[jj-1].month) and (datearray[jj].month - 1)%monthsToHold == 0 ):
            signal2D[:,jj] = signal2D[:,jj-1]
        else:
            if iter==0:
                print "date, signal2D changed",datearray[jj]

    numberStocks = np.sum(signal2D,axis = 0)

    print " signal2D check: ",signal2D[isnan(signal2D)].shape
    """

    """
    if uptrendSignalMethod == 'SMAs' :
        print "  ...using 3 SMA's for signal2D"
        print "\n\n ...calculating signal2D using '"+uptrendSignalMethod+"' method..."
        ########################################################################
        ## Calculate signal for all stocks based on 3 simple moving averages (SMA's)
        ########################################################################
        sma0 = SMA_2D( adjClose, MA2 )               # MA2 is shortest
        sma1 = SMA_2D( adjClose, MA2 + MA2offset )
        sma2 = sma2factor * SMA_2D( adjClose, MA1 )  # MA1 is longest

        signal2D = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
        for ii in range(adjClose.shape[0]):
            for jj in range(adjClose.shape[1]):
                if adjClose[ii,jj] > sma2[ii,jj] or ((adjClose[ii,jj] > min(sma0[ii,jj],sma1[ii,jj]) and sma0[ii,jj] > sma0[ii,jj-1])):
                    signal2D[ii,jj] = 1
                    if jj== adjClose.shape[1]-1 and isnan(adjClose[ii,-1]):
                        signal2D[ii,jj] = 0                #### added to avoid choosing stocks no longer in index
            # take care of special case where constant share price is inserted at beginning of series
            index = np.argmax(np.clip(np.abs(gainloss[ii,:]-1),0,1e-8)) - 1

            signal2D[ii,0:index] = 0

    elif uptrendSignalMethod == 'minmaxChannels' :
        print "  ...using 3 minmax channels for signal2D"
        print "\n\n ...calculating signal2D using '"+uptrendSignalMethod+"' method..."

        ########################################################################
        ## Calculate signal for all stocks based on 3 minmax channels (dpgchannels)
        ########################################################################

        # narrow channel is designed to remove day-to-day variability

        print "narrow days min,max,inc = ", narrowDays[0], narrowDays[-1], (narrowDays[-1]-narrowDays[0])/7.
        narrow_minChannel, narrow_maxChannel = dpgchannel_2D( adjClose, narrowDays[0], narrowDays[-1], (narrowDays[-1]-narrowDays[0])/7. )
        narrow_midChannel = (narrow_minChannel+narrow_maxChannel)/2.

        medium_minChannel, medium_maxChannel = dpgchannel_2D( adjClose, mediumDays[0], mediumDays[-1], (mediumDays[-1]-mediumDays[0])/7. )
        medium_midChannel = (medium_minChannel+medium_maxChannel)/2.
        mediumSignal = ((narrow_midChannel-medium_minChannel)/(medium_maxChannel-medium_minChannel)-0.5)*2.0

        wide_minChannel, wide_maxChannel = dpgchannel_2D( adjClose, wideDays[0], wideDays[-1], (wideDays[-1]-wideDays[0])/7. )
        wide_midChannel = (wide_minChannel+wide_maxChannel)/2.
        wideSignal = ((narrow_midChannel-wide_minChannel)/(wide_maxChannel-wide_minChannel)-0.5)*2.0

        signal2D = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
        for ii in range(adjClose.shape[0]):
            for jj in range(adjClose.shape[1]):
                if mediumSignal[ii,jj] + wideSignal[ii,jj] > 0:
                    signal2D[ii,jj] = 1
                    if jj== adjClose.shape[1]-1 and isnan(adjClose[ii,-1]):
                        signal2D[ii,jj] = 0                #### added to avoid choosing stocks no longer in index
            # take care of special case where constant share price is inserted at beginning of series
            index = np.argmax(np.clip(np.abs(gainloss[ii,:]-1),0,1e-8)) - 1

            signal2D[ii,0:index] = 0

            '''
            # take care of special case where mp quote exists at end of series
            if firstTrailingEmptyPriceIndex[ii] != 0:
                signal2D[ii,firstTrailingEmptyPriceIndex[ii]:] = 0
            '''

    elif uptrendSignalMethod == 'percentileChannels' :
        print "\n\n ...calculating signal2D using '"+uptrendSignalMethod+"' method..."
        signal2D = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
        lowChannel,hiChannel = percentileChannel_2D(adjClose,MA1,MA2+.01,MA2offset,lowPct,hiPct)
        for ii in range(adjClose.shape[0]):
            for jj in range(1,adjClose.shape[1]):
                if adjClose[ii,jj] > lowChannel[ii,jj] and adjClose[ii,jj-1] <= lowChannel[ii,jj-1]:
                    signal2D[ii,jj] = 1
                elif adjClose[ii,jj] < hiChannel[ii,jj] and adjClose[ii,jj-1] >= hiChannel[ii,jj-1]:
                    signal2D[ii,jj] = 0
                else:
                    signal2D[ii,jj] = signal2D[ii,jj-1]

                if jj== adjClose.shape[1]-1 and isnan(adjClose[ii,-1]):
                    signal2D[ii,jj] = 0                #### added to avoid choosing stocks no longer in index
            # take care of special case where constant share price is inserted at beginning of series
            index = np.argmax(np.clip(np.abs(gainloss[ii,:]-1),0,1e-8)) - 1
            signal2D[ii,0:index] = 0
    """

    if params['uptrendSignalMethod'] == 'percentileChannels':
        signal2D, lowChannel, hiChannel = computeSignal2D( adjClose, gainloss, params )
    else:
        signal2D = computeSignal2D( adjClose, gainloss, params )
    
    # copy to daily signal
    signal2D_daily = signal2D.copy()

    # hold signal constant for each month
    for jj in np.arange(1,adjClose.shape[1]):
        #if not ((datearray[jj].month != datearray[jj-1].month) and (datearray[jj].month ==1 or datearray[jj].month == 5 or datearray[jj].month == 9)):
        if not ((datearray[jj].month != datearray[jj-1].month) and (datearray[jj].month - 1)%monthsToHold == 0 ):
            signal2D[:,jj] = signal2D[:,jj-1]
        else:
            if iter==0:
                print "date, signal2D changed",datearray[jj]

    numberStocks = np.sum(signal2D,axis = 0)

    print " signal2D check: ",signal2D[isnan(signal2D)].shape


    """
    ########################################################################
    ## Calculate signal for all stocks based on 3 simple moving averages (SMA's)
    ########################################################################
    sma0 = SMA_2D( adjClose, MA2 )               # MA2 is shortest
    sma1 = SMA_2D( adjClose, MA2 + MA2offset )
    sma2 = sma2factor * SMA_2D( adjClose, MA1 )  # MA1 is longest

    signal2D = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    for ii in range(adjClose.shape[0]):
        for jj in range(adjClose.shape[1]):
            if adjClose[ii,jj] > sma2[ii,jj] or ((adjClose[ii,jj] > min(sma0[ii,jj],sma1[ii,jj]) and sma0[ii,jj] > sma0[ii,jj-1])):
                signal2D[ii,jj] = 1
                if jj== adjClose.shape[1]-1 and isnan(adjClose[ii,-1]):
                    signal2D[ii,jj] = 0                #### added to avoid choosing stocks no longer in index
        # take care of special case where constant share price is inserted at beginning of series
        index = np.argmax(np.clip(np.abs(gainloss[ii,:]-1),0,1e-8)) - 1

        signal2D[ii,0:index] = 0

    # hold signal constant for each month
    for jj in np.arange(1,adjClose.shape[1]):
        #if not ((datearray[jj].month != datearray[jj-1].month) and (datearray[jj].month ==1 or datearray[jj].month == 5 or datearray[jj].month == 9)):
        if not ((datearray[jj].month != datearray[jj-1].month) and (datearray[jj].month - 1)%monthsToHold == 0 ):
            signal2D[:,jj] = signal2D[:,jj-1]
        else:
            if iter==0:
                print "date, signal2D changed",datearray[jj]

    numberStocks = np.sum(signal2D,axis = 0)

    print " signal2D check: ",signal2D[isnan(signal2D)].shape


    ########################################################################
    ## Calculate signal for all stocks based on 3 minmax channels (dpgchannels)
    ########################################################################

    # narrow channel is designed to remove day-to-day variability

    print "narrow days min,max,inc = ", narrowDays[0], narrowDays[-1], (narrowDays[-1]-narrowDays[0])/7.
    narrow_minChannel, narrow_maxChannel = dpgchannel_2D( adjClose, narrowDays[0], narrowDays[-1], (narrowDays[-1]-narrowDays[0])/7. )
    narrow_midChannel = (narrow_minChannel+narrow_maxChannel)/2.

    medium_minChannel, medium_maxChannel = dpgchannel_2D( adjClose, mediumDays[0], mediumDays[-1], (mediumDays[-1]-mediumDays[0])/7. )
    medium_midChannel = (medium_minChannel+medium_maxChannel)/2.
    mediumSignal = ((narrow_midChannel-medium_minChannel)/(medium_maxChannel-medium_minChannel)-0.5)*2.0

    wide_minChannel, wide_maxChannel = dpgchannel_2D( adjClose, wideDays[0], wideDays[-1], (wideDays[-1]-wideDays[0])/7. )
    wide_midChannel = (wide_minChannel+wide_maxChannel)/2.
    wideSignal = ((narrow_midChannel-wide_minChannel)/(wide_maxChannel-wide_minChannel)-0.5)*2.0

    signal2D = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    for ii in range(adjClose.shape[0]):
        for jj in range(adjClose.shape[1]):
            if mediumSignal[ii,jj] + wideSignal[ii,jj] > 0:
                signal2D[ii,jj] = 1
                if jj== adjClose.shape[1]-1 and isnan(adjClose[ii,-1]):
                    signal2D[ii,jj] = 0                #### added to avoid choosing stocks no longer in index
        # take care of special case where constant share price is inserted at beginning of series
        index = np.argmax(np.clip(np.abs(gainloss[ii,:]-1),0,1e-8)) - 1

        signal2D[ii,0:index] = 0

        '''
        # take care of special case where mp quote exists at end of series
        if firstTrailingEmptyPriceIndex[ii] != 0:
            signal2D[ii,firstTrailingEmptyPriceIndex[ii]:] = 0
        '''

    # copy to daily signal
    signal2D_daily = signal2D.copy()

    # hold signal constant for each month
    for jj in np.arange(1,adjClose.shape[1]):
        #if not ((datearray[jj].month != datearray[jj-1].month) and (datearray[jj].month ==1 or datearray[jj].month == 5 or datearray[jj].month == 9)):
        if not ((datearray[jj].month != datearray[jj-1].month) and (datearray[jj].month - 1)%monthsToHold == 0 ):
            signal2D[:,jj] = signal2D[:,jj-1]
        else:
            if iter==0:
                print "date, signal2D changed",datearray[jj]

    numberStocks = np.sum(signal2D,axis = 0)

    print " signal2D check: ",signal2D[isnan(signal2D)].shape
    """
    ########################################################################
    ### compute weights for each stock based on:
    ### 1. uptrending signal in "signal2D"
    ### 1. delta-rank computed from gain/loss over "LongPeriod_random"
    ### 2. sharpe ratio computed from daily gains over "LongPeriod"
    ########################################################################

    monthgainlossweight = sharpeWeightedRank_2D(datearray,symbols,adjClose,signal2D,signal2D_daily,LongPeriod,numberStocksTraded,riskDownside_min,riskDownside_max,rankThresholdPct,stddevThreshold=stddevThreshold)


    ########################################################################
    ### compute traded value of stock for each month
    ########################################################################

    # initialize monthvalue (assume B&H)
    monthvalue = value.copy()

    starting_valid_symbols = value[:,0]
    starting_valid_symbols_count = starting_valid_symbols[starting_valid_symbols > 1.e-4].shape[0]

    '''
    # handle special case for buying index  
    monthvalue = value.copy() * 0.
    for jj in np.arange(monthgainloss.shape[0]):
        if value[jj,0] != 1.0:
            monthvalue[jj,0] = 10000.# / starting_valid_symbols_count
        else:
            monthvalue[jj,0] = 0.
    '''

    print "  \n\n\n"
    for ii in np.arange(1,monthgainloss.shape[1]):

        if (datearray[ii].month != datearray[ii-1].month) and \
                ( (datearray[ii].month - 1)%monthsToHold == 0) and \
                np.max(np.abs(monthgainlossweight[:,ii-1] - monthgainlossweight[:,ii])) > 1.e-4 :
            commission = 0
            symbol_changed_count = 0
            valuemean=np.mean(monthvalue[:,ii-1])
            valuesum=np.sum(monthvalue[:,ii-1])

            # compute yesterday's holdings value
            yesterdayValue = np.sum( monthgainlossweight[:,ii-1] * monthvalue[:,ii-1] )
            todayValue = np.sum( gainloss[:,ii] * yesterdayValue * monthgainlossweight[:,ii] )
            # reduce CASH by commission amount
            weight_changes = np.abs(monthgainlossweight[:,ii-1]-monthgainlossweight[:,ii])
            symbol_changed_count = weight_changes[weight_changes > 1.e-4].shape[0]
            # handle special case for buying index
            if symbol_changed_count > 2 * numberStocksTraded:
                # handle special case for buying index
                commission = trade_cost
            else:
                commission = symbol_changed_count * trade_cost
            commission_factor = (valuesum-commission*monthvalue.shape[0])/valuesum
            
            print "date,changed#, commission,valuemean,yesterdayValue,todayValue,commissionFactor(%)= ", \
                   datearray[ii], symbol_changed_count, commission, valuemean, yesterdayValue,todayValue,format(commission_factor-1.,'5.2%')
            ### Note: this is only approximate to what I really want to do in trades. This models all percentages changing
            ###       which implies trading all stocks. But I really want just to adjust CASH balance if monthgainlossweight is constant.
            for jj in range(value.shape[0]):
                monthvalue[jj,ii] = gainloss[jj,ii]*valuesum*monthgainlossweight[jj,ii]*commission_factor   # re-balance using weights (that sum to 1.0 less commissions)
                #monthvalue[jj,ii] = monthgainlossweight[jj,ii]*yesterdayValue*gainloss[jj,ii]   # re-balance using weights (that sum to 1.0 less commissions)
                """
                if ii==800:
                    print "\n\n\nprevious day --- "
                    print jj,symbols[jj],monthvalue[jj,ii],valuemean, monthgainlossweight[jj,ii]
                    print "this day ---"
                    print gainloss[jj,ii],valuemean,monthgainlossweight[jj,ii],commission_factor,gainloss[jj,ii]*valuesum*monthgainlossweight[jj,ii]*commission_factor
                
                if ii==800 and jj==112:
                    import pdb
                    pdb.set_trace()
                """


        else:
            for jj in range(value.shape[0]):
                monthvalue[jj,ii] = monthvalue[jj,ii-1]*gainloss[jj,ii]
                
                """
                if ii==799:
                    monthvalue[jj,ii] = 10000.
                if datearray[ii] == datetime.date(1991,2,1) or datearray[ii] == datetime.date(1991,3,1):
                    import pdb
                    pdb.set_trace()
                """
                

    numberSharesCalc = monthvalue / adjClose    # for info only
    
    print "  \n\n\n"

    ########################################################################
    ### gather statistics on number of uptrending stocks
    ########################################################################

    index = 3780
    if monthvalue.shape[1] < 3780: index = monthvalue.shape[1]

    PortfolioValue = np.average(monthvalue,axis=0)
    PortfolioDailyGains = PortfolioValue[1:] / PortfolioValue[:-1]
    Sharpe15Yr = ( gmean(PortfolioDailyGains[-index:])**252 -1. ) / ( np.std(PortfolioDailyGains[-index:])*sqrt(252) )
    Sharpe10Yr = ( gmean(PortfolioDailyGains[-2520:])**252 -1. ) / ( np.std(PortfolioDailyGains[-2520:])*sqrt(252) )
    Sharpe5Yr = ( gmean(PortfolioDailyGains[-1260:])**252 -1. ) / ( np.std(PortfolioDailyGains[-1260:])*sqrt(252) )
    Sharpe3Yr = ( gmean(PortfolioDailyGains[-756:])**252 -1. ) / ( np.std(PortfolioDailyGains[-756:])*sqrt(252) )
    Sharpe2Yr = ( gmean(PortfolioDailyGains[-504:])**252 -1. ) / ( np.std(PortfolioDailyGains[-504:])*sqrt(252) )
    Sharpe1Yr = ( gmean(PortfolioDailyGains[-252:])**252 -1. ) / ( np.std(PortfolioDailyGains[-252:])*sqrt(252) )

    print "15 year : ",index,PortfolioValue[-1], PortfolioValue[-index],datearray[-index]

    Return15Yr = (PortfolioValue[-1] / PortfolioValue[-index])**(252./index)
    Return10Yr = (PortfolioValue[-1] / PortfolioValue[-2520])**(1/10.)
    Return5Yr = (PortfolioValue[-1] / PortfolioValue[-1260])**(1/5.)
    Return3Yr = (PortfolioValue[-1] / PortfolioValue[-756])**(1/3.)
    Return2Yr = (PortfolioValue[-1] / PortfolioValue[-504])**(1/2.)
    Return1Yr = (PortfolioValue[-1] / PortfolioValue[-252])

    MaxPortfolioValue *= 0.
    for jj in range(PortfolioValue.shape[0]):
        MaxPortfolioValue[jj] = max(MaxPortfolioValue[jj-1],PortfolioValue[jj])
    PortfolioDrawdown = PortfolioValue / MaxPortfolioValue - 1.
    Drawdown15Yr = np.mean(PortfolioDrawdown[-index:])
    Drawdown10Yr = np.mean(PortfolioDrawdown[-2520:])
    Drawdown5Yr = np.mean(PortfolioDrawdown[-1260:])
    Drawdown3Yr = np.mean(PortfolioDrawdown[-756:])
    Drawdown2Yr = np.mean(PortfolioDrawdown[-504:])
    Drawdown1Yr = np.mean(PortfolioDrawdown[-252:])

    BuyHoldPortfolioValue = np.mean(value,axis=0)
    BuyHoldDailyGains = BuyHoldPortfolioValue[1:] / BuyHoldPortfolioValue[:-1]
    BuyHoldSharpe15Yr = ( gmean(BuyHoldDailyGains[-index:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-index:])*sqrt(252) )
    BuyHoldSharpe10Yr = ( gmean(BuyHoldDailyGains[-2520:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-2520:])*sqrt(252) )
    BuyHoldSharpe5Yr  = ( gmean(BuyHoldDailyGains[-1126:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-1260:])*sqrt(252) )
    BuyHoldSharpe3Yr  = ( gmean(BuyHoldDailyGains[-756:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-756:])*sqrt(252) )
    BuyHoldSharpe2Yr  = ( gmean(BuyHoldDailyGains[-504:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-504:])*sqrt(252) )
    BuyHoldSharpe1Yr  = ( gmean(BuyHoldDailyGains[-252:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-252:])*sqrt(252) )
    BuyHoldReturn15Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-index])**(252./index)
    BuyHoldReturn10Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-2520])**(1/10.)
    BuyHoldReturn5Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-1260])**(1/5.)
    BuyHoldReturn3Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-756])**(1/3.)
    BuyHoldReturn2Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-504])**(1/2.)
    BuyHoldReturn1Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-252])
    for jj in range(BuyHoldPortfolioValue.shape[0]):
        MaxBuyHoldPortfolioValue[jj] = max(MaxBuyHoldPortfolioValue[jj-1],BuyHoldPortfolioValue[jj])

    BuyHoldPortfolioDrawdown = BuyHoldPortfolioValue / MaxBuyHoldPortfolioValue - 1.
    BuyHoldDrawdown15Yr = np.mean(BuyHoldPortfolioDrawdown[-index:])
    BuyHoldDrawdown10Yr = np.mean(BuyHoldPortfolioDrawdown[-2520:])
    BuyHoldDrawdown5Yr = np.mean(BuyHoldPortfolioDrawdown[-1260:])
    BuyHoldDrawdown3Yr = np.mean(BuyHoldPortfolioDrawdown[-756:])
    BuyHoldDrawdown2Yr = np.mean(BuyHoldPortfolioDrawdown[-504:])
    BuyHoldDrawdown1Yr = np.mean(BuyHoldPortfolioDrawdown[-252:])


    print ""
    print ""
    print "Sharpe15Yr, BuyHoldSharpe15Yr = ", Sharpe15Yr, BuyHoldSharpe15Yr
    print "Sharpe10Yr, BuyHoldSharpe10Yr = ", Sharpe10Yr, BuyHoldSharpe10Yr
    print "Sharpe5Yr, BuyHoldSharpe5Yr =   ", Sharpe5Yr, BuyHoldSharpe5Yr
    print "Sharpe3Yr, BuyHoldSharpe3Yr =   ", Sharpe3Yr, BuyHoldSharpe3Yr
    print "Sharpe2Yr, BuyHoldSharpe2Yr =   ", Sharpe2Yr, BuyHoldSharpe2Yr
    print "Sharpe1Yr, BuyHoldSharpe1Yr =   ", Sharpe1Yr, BuyHoldSharpe1Yr
    print "Return15Yr, BuyHoldReturn15Yr = ", Return15Yr, BuyHoldReturn15Yr
    print "Return10Yr, BuyHoldReturn10Yr = ", Return10Yr, BuyHoldReturn10Yr
    print "Return5Yr, BuyHoldReturn5Yr =   ", Return5Yr, BuyHoldReturn5Yr
    print "Return3Yr, BuyHoldReturn3Yr =   ", Return3Yr, BuyHoldReturn3Yr
    print "Return2Yr, BuyHoldReturn2Yr =   ", Return2Yr, BuyHoldReturn2Yr
    print "Return1Yr, BuyHoldReturn1Yr =   ", Return1Yr, BuyHoldReturn1Yr
    print "Drawdown15Yr, BuyHoldDrawdown15Yr = ", Drawdown15Yr, BuyHoldDrawdown15Yr
    print "Drawdown10Yr, BuyHoldDrawdown10Yr = ", Drawdown10Yr, BuyHoldDrawdown10Yr
    print "Drawdown5Yr, BuyHoldDrawdown5Yr =   ", Drawdown5Yr, BuyHoldDrawdown5Yr
    print "Drawdown3Yr, BuyHoldDrawdown3Yr =   ", Drawdown3Yr, BuyHoldDrawdown3Yr
    print "Drawdown2Yr, BuyHoldDrawdown2Yr =   ", Drawdown2Yr, BuyHoldDrawdown2Yr
    print "Drawdown1Yr, BuyHoldDrawdown1Yr =   ", Drawdown1Yr, BuyHoldDrawdown1Yr


    beatBuyHoldTest = ( (Sharpe15Yr-BuyHoldSharpe15Yr)/15. + \
                        (Sharpe10Yr-BuyHoldSharpe10Yr)/10. + \
                        (Sharpe5Yr-BuyHoldSharpe5Yr)/5. + \
                        (Sharpe3Yr-BuyHoldSharpe3Yr)/3. + \
                        (Sharpe2Yr-BuyHoldSharpe2Yr)/2. + \
                        (Sharpe1Yr-BuyHoldSharpe1Yr)/1. ) / (1/15. + 1/10.+1/5.+1/3.+1/2.+1)


    beatBuyHoldTest2 = 0
    if Return15Yr > BuyHoldReturn15Yr: beatBuyHoldTest2 += 1
    if Return10Yr > BuyHoldReturn10Yr: beatBuyHoldTest2 += 1
    if Return5Yr  > BuyHoldReturn5Yr:  beatBuyHoldTest2 += 1
    if Return3Yr  > BuyHoldReturn3Yr:  beatBuyHoldTest2 += 1.5
    if Return2Yr  > BuyHoldReturn2Yr:  beatBuyHoldTest2 += 2
    if Return1Yr  > BuyHoldReturn1Yr:  beatBuyHoldTest2 += 2.5
    if Return15Yr > 0: beatBuyHoldTest2 += 1
    if Return10Yr > 0: beatBuyHoldTest2 += 1
    if Return5Yr  > 0: beatBuyHoldTest2 += 1
    if Return3Yr  > 0: beatBuyHoldTest2 += 1.5
    if Return2Yr  > 0: beatBuyHoldTest2 += 2
    if Return1Yr  > 0: beatBuyHoldTest2 += 2.5
    if Drawdown15Yr > BuyHoldDrawdown15Yr: beatBuyHoldTest2 += 1
    if Drawdown10Yr > BuyHoldDrawdown10Yr: beatBuyHoldTest2 += 1
    if Drawdown5Yr  > BuyHoldDrawdown5Yr:  beatBuyHoldTest2 += 1
    if Drawdown3Yr  > BuyHoldDrawdown3Yr:  beatBuyHoldTest2 += 1.5
    if Drawdown2Yr  > BuyHoldDrawdown2Yr:  beatBuyHoldTest2 += 2
    if Drawdown1Yr  > BuyHoldDrawdown1Yr:  beatBuyHoldTest2 += 2.5
    # make it a ratio ranging from 0 to 1
    beatBuyHoldTest2 /= 27

    if beatBuyHoldTest2 > .60 :
        print "found monte carlo trial that beats BuyHold (test2)..."
        print "shape of numberStocksUpTrendingBeatBuyHold = ",numberStocksUpTrendingBeatBuyHold.shape
        print "mean of numberStocksUpTrendingBeatBuyHold values = ",np.mean(numberStocksUpTrendingBeatBuyHold)

    print ""
    print " "
    print "The B&H portfolio final value is: ","{:,}".format(int(BuyHoldFinalValue))
    print " "
    print "Monthly re-balance based on ",LongPeriod, "days of recent performance."
    print "The portfolio final value is: ","{:,}".format(int(np.average(monthvalue,axis=0)[-1]))
    print " "
    print "Today's top ranking choices are: "
    last_symbols_text = []
    for ii in range(len(symbols)):
        if monthgainlossweight[ii,-1] > 0:
            # print symbols[ii]
            print datearray[-1], symbols[ii],monthgainlossweight[ii,-1]
            last_symbols_text.append( symbols[ii] )



    ###
    ### save backtest portfolio values ( B&H and system )
    ###
    try:
        filepath = os.path.join( os.getcwd(), "pyTAAAweb_backtestPortfolioValue.params" )
        textmessage = ""
        for idate in range(len(BuyHoldPortfolioValue)):
            textmessage = textmessage + str(datearray[idate])+"  "+str(BuyHoldPortfolioValue[idate])+"  "+str(np.average(monthvalue[:,idate]))+"\n"
        with open( filepath, "w" ) as f:
            f.write(textmessage)
    except:
        pass


    ########################################################################
    ### compute some portfolio performance statistics and print summary
    ########################################################################

    print "final value for portfolio ", "{:,}".format(np.average(monthvalue[:,-1]))


    print "portfolio annualized gains : ", ( gmean(PortfolioDailyGains)**252 )
    print "portfolio annualized StdDev : ", ( np.std(PortfolioDailyGains)*sqrt(252) )
    print "portfolio sharpe ratio : ", ( gmean(PortfolioDailyGains)**252 ) / ( np.std(PortfolioDailyGains)*sqrt(252) )

    # Compute trading days back to target start date
    targetdate = datetime.date(2008,1,1)
    lag = int((datearray[-1] - targetdate).days*252/365.25)

    # Print some stats for B&H and trading from target date to end_date
    print ""
    print ""
    BHValue = np.average(value,axis=0)
    BHdailygains = np.concatenate( (np.array([0.]), BHValue[1:]/BHValue[:-1]), axis = 0 )
    BHsharpefromtargetdate = ( gmean(BHdailygains[-lag:])**252 -1. ) / ( np.std(BHdailygains[-lag:])*sqrt(252) )
    BHannualgainfromtargetdate = ( gmean(BHdailygains[-lag:])**252 )
    print "start date for recent performance measures: ",targetdate
    print "BuyHold annualized gains & sharpe from target date:   ", BHannualgainfromtargetdate,BHsharpefromtargetdate

    Portfoliosharpefromtargetdate = ( gmean(PortfolioDailyGains[-lag:])**252 -1. ) / ( np.std(PortfolioDailyGains[-lag:])*sqrt(252) )
    Portfolioannualgainfromtargetdate = ( gmean(PortfolioDailyGains[-lag:])**252 )
    print "portfolio annualized gains & sharpe from target date: ", Portfolioannualgainfromtargetdate,Portfoliosharpefromtargetdate

    return

