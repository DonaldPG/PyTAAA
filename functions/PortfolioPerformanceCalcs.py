import numpy as np
import os
import nose
import bottleneck as bn
import la
import h5py
import datetime
from scipy.stats import gmean
from math import sqrt
from numpy import std
from numpy import isnan
from functions.readSymbols import *
from functions.UpdateSymbols_inHDF5 import *
from functions.allstats import *
from functions.dailyBacktest import *
from functions.TAfunctions import *

def PortfolioPerformanceCalcs( symbol_directory, symbol_file, params ) :

    ## update quotes from list of symbols
    filename = os.path.join(symbol_directory, symbol_file)
    adjClose, symbols, datearray, _, _ = loadQuotes_fromHDF( symbol_file )

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
    sma2factor = params['MA2factor']
    MA1 = params['MA1']
    MA2 = params['MA2']
    MA2offset = params['MA3'] - params['MA2']
    riskDownside_min = params['riskDownside_min']
    riskDownside_max = params['riskDownside_max']
    rankThresholdPct = params['rankThresholdPct']
    ##

    monthgainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    monthgainloss[:,LongPeriod:] = adjClose[:,LongPeriod:] / adjClose[:,:-LongPeriod]
    monthgainloss[isnan(monthgainloss)]=1.

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

    ##########################################
    # Write daily backtest portfolio and even-weighted B&H values to file for web page
    ##########################################

    computeDailyBacktest( datearray, \
                     symbols, \
                     adjClose, \
                     numberStocksTraded = numberStocksTraded, \
                     monthsToHold = monthsToHold, \
                     LongPeriod = LongPeriod, \
                     MA1 = MA1, \
                     MA2 = MA2, \
                     MA2offset = MA2offset, \
                     sma2factor = sma2factor, \
                     rankThresholdPct = rankThresholdPct, \
                     riskDownside_min = riskDownside_min, \
                     riskDownside_max = riskDownside_max )
                     
    print "\n\n Successfully updated daily backtest at in 'pyTAAAweb_backtestPortfolioValue.params'. Completed on ", datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
    print ""


    ##########################################
    # Write date and Percent of up-trending stocks to file for web page
    ##########################################
    try:
        filepath = os.path.join( os.getcwd(), "pyTAAAweb_numberUptrendingStocks_status.params" )
        textmessage = ""
        for jj in range(dailyNumberUptrendingStocks.shape[0]):
            textmessage = textmessage + str(datearray[jj])+"  "+str(dailyNumberUptrendingStocks[jj])+"  "+str(activeCount[jj])+"\n"
        with open( filepath, "w" ) as f:
            f.write(textmessage)
        print " Successfully updated to pyTAAAweb_numberUptrendingStocks_status.params at ", datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
        print ""
    except :
        print " Error: unable to update pyTAAAweb_numberUptrendingStocks_status.params"
        print ""



    '''
    ##########################################
    # Write date and trend dispersion value to file for web page
    ##########################################
    ### this version will be deleted after it works successfully once

    try:
        print "\n\n diagnostics for updating pyTAAAweb_MeanTrendDispersion_status.params....."
        dailyTrendDispersionMeans = []
        dailyTrendDispersionMedians = []
        
        for ii in range(adjClose.shape[0]):
            try:
                ii_zscore = np.mean( np.abs(allstats( adjClose[ii,-20:] ).z_score() ))
            except:
                ii_zscore = np.nan
            try:
                ii_sharpe = np.mean( allstats( adjClose[ii,-20:].sharpe() ))
            except:
                ii_sharpe = np.nan
            
            dailyTrendDispersionMeans.append( ii_zscore )
            dailyTrendDispersionMedians.append( ii_sharpe )
        
        dailyTrendDispersionMeans = np.array( dailyTrendDispersionMeans )
        dailyTrendDispersionMeans = dailyTrendDispersionMeans[ dailyTrendDispersionMeans > 0. ]
        
        dailyTrendDispersionMedians = np.array( dailyTrendDispersionMedians )
        dailyTrendDispersionMedians = dailyTrendDispersionMedians[ np.isfinite( dailyTrendDispersionMedians ) ]

        
        filepath = os.path.join( os.getcwd(), "pyTAAAweb_MeanTrendDispersion_status.params" )
        print "filepath in writeWebPage = ", filepath
        textmessage = ""
        textmessage = textmessage + str(datearray[-1])+"  "+str( dailyTrendDispersionMeans.mean() )+"  "+str( dailyTrendDispersionMedians.mean() )+"\n"
        
        with open( filepath, "a" ) as f:
            f.write(textmessage)
        print " Successfully updated to pyTAAAweb_MeanTrendDispersion_status.params at ", datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
        print ""
    except :
        print " Error: unable to update pyTAAAweb_MeanTrendDispersion_status.params"
        print ""
    '''

    ########################################################################
    ### 1. make plots for all stocks of adjusted price history
    ########################################################################

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pylab as plt
    filepath = os.path.join( os.getcwd(), "pyTAAA_web" )

    today = datetime.datetime.now()
    hourOfDay = today.hour

    if  hourOfDay >= 22 :
        for i in range( len(symbols) ) :
            plt.clf()
            plt.grid(True)
            plt.plot(datearray,adjClose[i,:])
            plt.plot(datearray,signal2D[i,:]*adjClose[i,-1])
            plot_text = str(adjClose[i,-7:])
            plt.text(datearray[50],0,plot_text)
            plt.title(symbols[i])
            plotfilepath = os.path.join( filepath, "0_"+symbols[i]+".png" )
            plt.savefig( plotfilepath )

    ####################################################################################3



    ########################################################################
    ### compute weights for each stock based on:
    ### 1. uptrending signal in "signal2D"
    ### 1. delta-rank computed from gain/loss over "LongPeriod_random"
    ### 2. sharpe ratio computed from daily gains over "LongPeriod_random"
    ########################################################################

    monthgainlossweight = sharpeWeightedRank_2D(datearray,symbols,adjClose,signal2D,LongPeriod,numberStocksTraded,riskDownside_min,riskDownside_max,rankThresholdPct)


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



    print " "
    print "The B&H portfolio final value is: ","{:,}".format(int(BuyHoldFinalValue))
    print " "
    print "Monthly re-balance based on ",LongPeriod, "days of recent performance."
    print "The portfolio final value is: ","{:,}".format(int(np.average(monthvalue,axis=0)[-1]))
    print " "
    print "Today's top ranking choices are: "
    last_symbols_text = []
    last_symbols_weight = []
    last_symbols_price = []
    for ii in range(len(symbols)):
        if monthgainlossweight[ii,-1] > 0:
            # print symbols[ii]
            print datearray[-1], format(symbols[ii],'5s'), format(monthgainlossweight[ii,-1],'5.3f')
            last_symbols_text.append( symbols[ii] )
            last_symbols_weight.append( monthgainlossweight[ii,-1] )
            last_symbols_price.append( round(adjClose[ii,-1],2) )

    return datearray[-1], last_symbols_text, last_symbols_weight, last_symbols_price
