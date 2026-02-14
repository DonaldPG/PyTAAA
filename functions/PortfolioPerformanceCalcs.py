import numpy as np
import os
#import nose
import datetime
from scipy.stats import gmean
from math import sqrt
from numpy import std
from numpy import isnan

import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
# Set DPI for inline plots and saved figures
plt.rcParams['figure.figsize'] = (9, 7)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

from functions.readSymbols import *
from functions.data_loaders import load_quotes_for_analysis
from functions.allstats import *
from functions.dailyBacktest import computeDailyBacktest
from functions.TAfunctions import (
    computeSignal2D,
    despike_2D,
    recentTrendAndMidTrendChannelFitWithAndWithoutGap,
    sharpeWeightedRank_2D
)
from functions.CheckMarketOpen import *
from functions.GetParams import get_webpage_store
from functions.CountNewHighsLows import newHighsAndLows

def PortfolioPerformanceCalcs(symbol_directory, symbol_file, params, json_fn):

    print("\n\n ... inside PortfolioPerformanceCalcs...")

    print("   . symbol_directory = " + symbol_directory)
    print("   . symbol_file = " + symbol_file)

    json_dir = os.path.split(json_fn)[0]
    print("   . json_dir = " + json_dir)

    ## update quotes from list of symbols
    filename = os.path.join(symbol_directory, symbol_file)
    print("   . filename for load_quotes_for_analysis = " + filename)
    adjClose, symbols, datearray = load_quotes_for_analysis(filename, json_fn, verbose=True)

    gainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    activeCount = np.zeros(adjClose.shape[1],dtype=float)

    numberSharesCalc = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    gainloss[:,1:] = adjClose[:,1:] / adjClose[:,:-1]
    gainloss[np.isnan(gainloss)]=1.
    gainloss[np.isinf(gainloss)]=1.
    value = 10000. * np.cumprod(gainloss,axis=1)

    BuyHoldFinalValue = np.average(value,axis=0)[-1]

    lastEmptyPriceIndex = np.zeros(adjClose.shape[0],dtype=int)

    for ii in range(adjClose.shape[0]):
        # take care of special case where constant share price is inserted at beginning of series
        index = np.argmax(np.clip(np.abs(gainloss[ii,:]-1),0,1e-8)) - 1
        lastEmptyPriceIndex[ii] = index
        activeCount[lastEmptyPriceIndex[ii]+1:] += 1

    # remove NaN's from count for each day
    for ii in range(adjClose.shape[1]):
        numNaNs = ( np.isnan( adjClose[:,ii] ) )
        numNaNs = numNaNs[ numNaNs == True ].shape[0]
        activeCount[ii] = activeCount[ii] - np.clip(numNaNs,0.,99999)

    monthsToHold = params['monthsToHold']
    numberStocksTraded = params['numberStocksTraded']
    LongPeriod = params['LongPeriod']
    stddevThreshold = float(params['stddevThreshold'])
    sma2factor = params['MA2factor']
    MA1 = int(params['MA1'])
    MA2 = int(params['MA2'])
    MA2offset = int(params['MA3']) - MA2
    riskDownside_min = float(params['riskDownside_min'])
    riskDownside_max = float(params['riskDownside_max'])
    rankThresholdPct = float(params['rankThresholdPct'])

    narrowDays = params['narrowDays']
    mediumDays = params['mediumDays']
    wideDays = params['wideDays']

    lowPct = float(params['lowPct'])
    hiPct = float(params['hiPct'])
    uptrendSignalMethod = params['uptrendSignalMethod']

    ##

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
    ### - method depends on where code is running

    # """
    # # check the operatiing system to determine whether to move files or use ftp
    # import platform

    # operatingSystem = platform.system()
    # architecture = platform.uname()[4]
    # computerName = platform.uname()[1]

    # print "  ...platform: ", operatingSystem, architecture, computerName

    # if uptrendSignalMethod == 'SMAs' :
    #     print "  ...using 3 SMA's for signal2D"
    #     print "\n\n ...calculating signal2D using '"+uptrendSignalMethod+"' method..."
    #     ########################################################################
    #     ## Calculate signal for all stocks based on 3 simple moving averages (SMA's)
    #     ########################################################################
    #     sma0 = SMA_2D( adjClose, MA2 )               # MA2 is shortest
    #     sma1 = SMA_2D( adjClose, MA2 + MA2offset )
    #     sma2 = sma2factor * SMA_2D( adjClose, MA1 )  # MA1 is longest

    #     signal2D = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    #     for ii in range(adjClose.shape[0]):
    #         for jj in range(adjClose.shape[1]):
    #             if adjClose[ii,jj] > sma2[ii,jj] or ((adjClose[ii,jj] > min(sma0[ii,jj],sma1[ii,jj]) and sma0[ii,jj] > sma0[ii,jj-1])):
    #                 signal2D[ii,jj] = 1
    #                 if jj== adjClose.shape[1]-1 and isnan(adjClose[ii,-1]):
    #                     signal2D[ii,jj] = 0                #### added to avoid choosing stocks no longer in index
    #         # take care of special case where constant share price is inserted at beginning of series
    #         index = np.argmax(np.clip(np.abs(gainloss[ii,:]-1),0,1e-8)) - 1

    #         signal2D[ii,0:index] = 0

    #     dailyNumberUptrendingStocks = np.sum(signal2D,axis = 0)


    # elif uptrendSignalMethod == 'minmaxChannels' :
    #     print "  ...using 3 minmax channels for signal2D"
    #     print "\n\n ...calculating signal2D using '"+uptrendSignalMethod+"' method..."

    #     ########################################################################
    #     ## Calculate signal for all stocks based on 3 minmax channels (dpgchannels)
    #     ########################################################################

    #     # narrow channel is designed to remove day-to-day variability

    #     print "narrow days min,max,inc = ", narrowDays[0], narrowDays[-1], (narrowDays[-1]-narrowDays[0])/7.
    #     narrow_minChannel, narrow_maxChannel = dpgchannel_2D( adjClose, narrowDays[0], narrowDays[-1], (narrowDays[-1]-narrowDays[0])/7. )
    #     narrow_midChannel = (narrow_minChannel+narrow_maxChannel)/2.

    #     medium_minChannel, medium_maxChannel = dpgchannel_2D( adjClose, mediumDays[0], mediumDays[-1], (mediumDays[-1]-mediumDays[0])/7. )
    #     medium_midChannel = (medium_minChannel+medium_maxChannel)/2.
    #     mediumSignal = ((narrow_midChannel-medium_minChannel)/(medium_maxChannel-medium_minChannel)-0.5)*2.0

    #     wide_minChannel, wide_maxChannel = dpgchannel_2D( adjClose, wideDays[0], wideDays[-1], (wideDays[-1]-wideDays[0])/7. )
    #     wide_midChannel = (wide_minChannel+wide_maxChannel)/2.
    #     wideSignal = ((narrow_midChannel-wide_minChannel)/(wide_maxChannel-wide_minChannel)-0.5)*2.0

    #     signal2D = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    #     for ii in range(adjClose.shape[0]):
    #         for jj in range(adjClose.shape[1]):
    #             if mediumSignal[ii,jj] + wideSignal[ii,jj] > 0:
    #                 signal2D[ii,jj] = 1
    #                 if jj== adjClose.shape[1]-1 and isnan(adjClose[ii,-1]):
    #                     signal2D[ii,jj] = 0                #### added to avoid choosing stocks no longer in index
    #         # take care of special case where constant share price is inserted at beginning of series
    #         index = np.argmax(np.clip(np.abs(gainloss[ii,:]-1),0,1e-8)) - 1

    #         signal2D[ii,0:index] = 0

    #         '''
    #         # take care of special case where mp quote exists at end of series
    #         if firstTrailingEmptyPriceIndex[ii] != 0:
    #             signal2D[ii,firstTrailingEmptyPriceIndex[ii]:] = 0
    #         '''

    # elif uptrendSignalMethod == 'percentileChannels' :
    #     print "\n\n ...calculating signal2D using '"+uptrendSignalMethod+"' method..."
    #     signal2D = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    #     lowChannel,hiChannel = percentileChannel_2D(adjClose,MA1,MA2+.01,MA2offset,lowPct,hiPct)
    #     for ii in range(adjClose.shape[0]):
    #         for jj in range(1,adjClose.shape[1]):
    #             if adjClose[ii,jj] > lowChannel[ii,jj] and adjClose[ii,jj-1] <= lowChannel[ii,jj-1]:
    #                 signal2D[ii,jj] = 1
    #             elif adjClose[ii,jj] < hiChannel[ii,jj] and adjClose[ii,jj-1] >= hiChannel[ii,jj-1]:
    #                 signal2D[ii,jj] = 0
    #             else:
    #                 signal2D[ii,jj] = signal2D[ii,jj-1]

    #             if jj== adjClose.shape[1]-1 and isnan(adjClose[ii,-1]):
    #                 signal2D[ii,jj] = 0                #### added to avoid choosing stocks no longer in index
    #         # take care of special case where constant share price is inserted at beginning of series
    #         index = np.argmax(np.clip(np.abs(gainloss[ii,:]-1),0,1e-8)) - 1
    #         signal2D[ii,0:index] = 0
    # """

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
                # print("date, signal2D changed",datearray[jj])
                pass

    numberStocks = np.sum(signal2D,axis = 0)
    dailyNumberUptrendingStocks = np.sum(signal2D,axis = 0)

    print(" signal2D check: ",signal2D[isnan(signal2D)].shape)
    print(" signal2D min, mean,max: ",signal2D.min(),signal2D.mean(),signal2D.max())
    print(" numberStocks (uptrending) min, mean,max: ",numberStocks.min(),numberStocks.mean(),numberStocks.max())

    ##########################################
    # Write daily backtest portfolio and even-weighted B&H values to file for web page
    ##########################################

    computeDailyBacktest(
        json_fn,
        datearray,
        symbols,
        adjClose,
        numberStocksTraded = numberStocksTraded,
        trade_cost=params['trade_cost'],
        monthsToHold = monthsToHold,
        LongPeriod = LongPeriod,
        MA1 = MA1,
        MA2 = MA2,
        MA2offset = MA2offset,
        sma2factor = sma2factor,
        rankThresholdPct = rankThresholdPct,
        riskDownside_min = riskDownside_min,
        riskDownside_max = riskDownside_max,
        narrowDays = narrowDays,
        mediumDays = mediumDays,
        wideDays = wideDays,
        stddevThreshold = stddevThreshold,
        lowPct=lowPct,
        hiPct=hiPct,
        uptrendSignalMethod=uptrendSignalMethod
    )

    print("\n\n Successfully updated daily backtest at in 'pyTAAAweb_backtestPortfolioValue.params'. Completed on ", datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print("")

    print(".....PortfolioPermanceCalcs.py line 233.... adjClose[5,-1] = ", adjClose[5,-1])

    ##########################################
    # Write date and Percent of up-trending stocks to file for web page
    ##########################################
    
    # Phase 4b2: File writing extracted to output_generators.py
    from functions.output_generators import write_portfolio_status_files
    
    web_dir = get_webpage_store(json_fn)
    write_portfolio_status_files(
        dailyNumberUptrendingStocks, activeCount, datearray, web_dir
    )


    # '''
    # ##########################################
    # # Write date and trend dispersion value to file for web page
    # ##########################################
    # ### this version will be deleted after it works successfully once

    # try:
    #     print "\n\n diagnostics for updating pyTAAAweb_MeanTrendDispersion_status.params....."
    #     dailyTrendDispersionMeans = []
    #     dailyTrendDispersionMedians = []

    #     for ii in range(adjClose.shape[0]):
    #         try:
    #             ii_zscore = np.mean( np.abs(allstats( adjClose[ii,-20:] ).z_score() ))
    #         except:
    #             ii_zscore = np.nan
    #         try:
    #             ii_sharpe = np.mean( allstats( adjClose[ii,-20:].sharpe() ))
    #         except:
    #             ii_sharpe = np.nan

    #         dailyTrendDispersionMeans.append( ii_zscore )
    #         dailyTrendDispersionMedians.append( ii_sharpe )

    #     dailyTrendDispersionMeans = np.array( dailyTrendDispersionMeans )
    #     dailyTrendDispersionMeans = dailyTrendDispersionMeans[ dailyTrendDispersionMeans > 0. ]

    #     dailyTrendDispersionMedians = np.array( dailyTrendDispersionMedians )
    #     dailyTrendDispersionMedians = dailyTrendDispersionMedians[ np.isfinite( dailyTrendDispersionMedians ) ]


    #     filepath = os.path.join( os.getcwd(), "pyTAAAweb_MeanTrendDispersion_status.params" )
    #     print "filepath in writeWebPage = ", filepath
    #     textmessage = ""
    #     textmessage = textmessage + str(datearray[-1])+"  "+str( dailyTrendDispersionMeans.mean() )+"  "+str( dailyTrendDispersionMedians.mean() )+"\n"

    #     with open( filepath, "a" ) as f:
    #         f.write(textmessage)
    #     print " Successfully updated to pyTAAAweb_MeanTrendDispersion_status.params at ", datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
    #     print ""
    # except :
    #     print " Error: unable to update pyTAAAweb_MeanTrendDispersion_status.params"
    #     print ""
    # '''

    ########################################################################
    ### 1. make plots for all stocks of adjusted price history
    ########################################################################
    
    # Phase 4b1: Plot generation extracted to output_generators.py
    from functions.output_generators import generate_portfolio_plots
    
    filepath = get_webpage_store(json_fn)
    
    # Generate plots (conditional on time of day, handled inside function)
    if params['uptrendSignalMethod'] == 'percentileChannels':
        generate_portfolio_plots(
            adjClose, symbols, datearray, signal2D, signal2D_daily,
            params, filepath, lowChannel=lowChannel, hiChannel=hiChannel
        )
    else:
        generate_portfolio_plots(
            adjClose, symbols, datearray, signal2D, signal2D_daily,
            params, filepath
        )
    
    # End of plot generation (Phase 4b1: extracted to output_generators.py)

    # print list of currently uptrending stocks
    if uptrendSignalMethod == 'percentileChannels' :
        print("\n\n\nCurrently up-trending symbols ("+str(datearray[-1])+"):")
        uptrendCount = 0
        for i in range(len(symbols)):
            if signal2D_daily[i,-1] > 0:
                uptrendCount += 1
                print(uptrendCount, symbols[i], adjClose[i,-1], " uptrend", lowChannel[i,-1], hiChannel[i,-1])
            else:
                print(uptrendCount, symbols[i], adjClose[i,-1], "        ", lowChannel[i,-1], hiChannel[i,-1])
    print("\n\n\n")

    ####################################################################################3



    ########################################################################
    ### compute weights for each stock based on:
    ### 1. uptrending signal in "signal2D"
    ### 1. delta-rank computed from gain/loss over "LongPeriod_random"
    ### 2. sharpe ratio computed from daily gains over "LongPeriod_random"
    ########################################################################

    monthgainlossweight = sharpeWeightedRank_2D(
        json_fn, datearray, symbols, adjClose,
        signal2D, signal2D_daily, LongPeriod, numberStocksTraded,
        riskDownside_min, riskDownside_max, rankThresholdPct,
        stddevThreshold=stddevThreshold,
        is_backtest=False, makeQCPlots=True
    )

    # """
    # # moved to MakeValuePlot.py
    # ########################################################################
    # ### compute stats on new 52-week highs and lows
    # ########################################################################

    # _, _ = newHighsAndLows( num_days_highlow=(84,252),\
    #                         num_days_cumu=(63,126),\
    #                         HighLowRatio=(2.,2.),\
    #                         makeQCPlots=True)
    # """

    ########################################################################
    ### compute traded value of stock for each month
    ########################################################################

    monthvalue = value.copy()
    for ii in np.arange(1,monthgainloss.shape[1]):
        if (datearray[ii].month != datearray[ii-1].month) and ( (datearray[ii].month - 1)%monthsToHold == 0):
            valuesum=np.sum(monthvalue[:,ii-1])
            for jj in range(value.shape[0]):
                monthvalue[jj,ii] = monthgainlossweight[jj,ii]*valuesum*gainloss[jj,ii]   # re-balance using weights (that sum to 1.0)
        else:
            for jj in range(value.shape[0]):
                monthvalue[jj,ii] = monthvalue[jj,ii-1]*gainloss[jj,ii]

    numberSharesCalc = monthvalue / adjClose    # for info only


    print(" ")
    print("The B&H portfolio final value is: ","{:,}".format(int(BuyHoldFinalValue)))
    print(" ")
    print("Monthly re-balance based on ",LongPeriod, "days of recent performance.")
    print("The portfolio final value is: ","{:,}".format(int(np.average(monthvalue,axis=0)[-1])))
    print(" ")
    print("Today's top ranking choices are: ")
    last_symbols_text = []
    last_symbols_weight = []
    last_symbols_price = []
    for ii in range(len(symbols)):
        if monthgainlossweight[ii,-1] > 0:
            print(datearray[-1], format(symbols[ii],'5s'), format(monthgainlossweight[ii,-1],'5.3f'))
            last_symbols_text.append( symbols[ii] )
            last_symbols_weight.append( float(round(monthgainlossweight[ii,-1],4)))
            last_symbols_price.append( float(round(adjClose[ii,-1],2)))

    print("\n ... inside portfolioPerformanceCalcs")
    print("   . datearray[-1] = " +str(datearray[-1]))
    print("   . last_symbols_text = " +str(last_symbols_text))
    print("   . last_symbols_weight = " + str(last_symbols_weight))
    print("   . last_symbols_price = " + str(last_symbols_price))

    # send text message for held stocks breaking to downside outside trend channel
    # - if markets are currently open
    marketStatus = get_MarketOpenOrClosed()
    if 'Market Open' in marketStatus:
        textmessageOutsideTrendChannel( symbols, adjClose )
    #textmessageOutsideTrendChannel( symbols, adjClose )

    return datearray[-1], last_symbols_text, last_symbols_weight, last_symbols_price

