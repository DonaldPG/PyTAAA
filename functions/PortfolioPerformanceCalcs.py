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

    #############################################################################
    # Phase 4b3: Call pure computation function (no side effects)
    #############################################################################
    
    from functions.output_generators import compute_portfolio_metrics
    
    metrics = compute_portfolio_metrics(adjClose, symbols, datearray, params, json_fn)
    
    # Extract computed values from metrics dictionary
    gainloss = metrics['gainloss']
    value = metrics['value']
    BuyHoldFinalValue = metrics['BuyHoldFinalValue']
    activeCount = metrics['activeCount']
    monthgainloss = metrics['monthgainloss']
    signal2D = metrics['signal2D']
    signal2D_daily = metrics['signal2D_daily']
    numberStocks = metrics['numberStocks']
    dailyNumberUptrendingStocks = metrics['dailyNumberUptrendingStocks']
    monthgainlossweight = metrics['monthgainlossweight']
    monthvalue = metrics['monthvalue']
    numberSharesCalc = metrics['numberSharesCalc']
    last_symbols_text = metrics['last_symbols_text']
    last_symbols_weight = metrics['last_symbols_weight']
    last_symbols_price = metrics['last_symbols_price']
    
    # Extract optional channels if present
    lowChannel = metrics.get('lowChannel', None)
    hiChannel = metrics.get('hiChannel', None)
    
    #############################################################################
    # End of Phase 4b3 computation
    #############################################################################
    
    print(" signal2D check: ", signal2D[isnan(signal2D)].shape)
    print(" signal2D min, mean,max: ", signal2D.min(), signal2D.mean(), signal2D.max())
    print(" numberStocks (uptrending) min, mean,max: ", numberStocks.min(), numberStocks.mean(), numberStocks.max())

    ##########################################
    # Write daily backtest portfolio and even-weighted B&H values to file for web page
    ##########################################

    # Extract parameters for computeDailyBacktest call
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
    if uptrendSignalMethod == 'percentileChannels':
        print("\n\n\nCurrently up-trending symbols ("+str(datearray[-1])+"):")
        uptrendCount = 0
        for i in range(len(symbols)):
            if signal2D_daily[i,-1] > 0:
                uptrendCount += 1
                print(uptrendCount, symbols[i], adjClose[i,-1], " uptrend", lowChannel[i,-1], hiChannel[i,-1])
            else:
                print(uptrendCount, symbols[i], adjClose[i,-1], "        ", lowChannel[i,-1], hiChannel[i,-1])
    print("\n\n\n")

    ####################################################################################

    print(" ")
    print("The B&H portfolio final value is: ","{:,}".format(int(BuyHoldFinalValue)))
    print(" ")
    print("Monthly re-balance based on ",LongPeriod, "days of recent performance.")
    print("The portfolio final value is: ","{:,}".format(int(np.average(monthvalue,axis=0)[-1])))
    print(" ")
    print("Today's top ranking choices are: ")
    for ii in range(len(symbols)):
        if monthgainlossweight[ii,-1] > 0:
            print(datearray[-1], format(symbols[ii],'5s'), format(monthgainlossweight[ii,-1],'5.3f'))

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

