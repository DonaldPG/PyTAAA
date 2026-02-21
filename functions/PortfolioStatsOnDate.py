import numpy as np
import os
import nose
import bottleneck as bn
import la
import h5py
import datetime
from typing import List
from scipy.stats import gmean
from math import sqrt
from numpy import std
from numpy import isnan
from functions.readSymbols import *

def PortfolioStatsOnDate(directory_name: str, file_name: str, params: dict, StatDate) -> list:

    ######################
    ### Input parameters
    ######################

    (shortname, extension) = os.path.splitext( file_name )

    #print "file name for symbols = ","_"+shortname+"_"
    #print "file type for symbols = ",extension

    # associate directory_name and file_name with HDF5 file on disk.
    if shortname == "symbols" :
        listname = "TAA-Symbols"
    elif shortname == "cmg_symbols" :
        listname = "CMG-Symbols"
    elif shortname == "Naz100_Symbols" :
        listname = "Naz100-Symbols"
    elif shortname == "biglist" :
        listname = "biglist-Symbols"
    elif shortname == "ETF_symbols" :
        listname = "ETF-Symbols"
    elif shortname == "ProvidentFundSymbols" :
        listname = "ProvidentFund-Symbols"
    elif shortname == "sp500_symbols" :
        listname = "SP500-Symbols"
    else :
        listname = shortname

    #hdf5_directory = os.getcwd()+"\\symbols"
    hdf5_directory = os.path.join( os.getcwd(), "symbols" )
    hdf5filename = os.path.join( hdf5_directory, listname + "_.hdf5" )

    # get current list of symbols
    filename = os.path.join( directory_name, file_name )
    symbols = readSymbolList( filename,verbose=True )

    # loop through symbols to create list of stats
    # - use list (with length same as symbols length) of lists (stats per symbol)
    allSymbolStats = []
    # get 2D quotes from labelled array on disk
    io = la.IO(hdf5filename)
    x = io[listname]
    ##print "x shape = ",x.shape
    # get date labels from labelled array on disk
    date = x.getlabel(1)
    dates=[]
    for i in range(len(date)):
        datestr = date[i]
        date_newformat = datetime.date(*[int(val) for val in datestr.split('-')])
        dates.append(date_newformat)
    datearray = np.asarray(dates)
    # find index to desired date
    ##print " *******************"
    ##print " ******************* StatDate, and final date in hdf5 file = ", StatDate, date[-1], datearray[-1], type(StatDate), type(date[-1]), type(datearray[-1])
    ##print " *******************"
    dateIndex = x.labelindex(str(StatDate),axis=1)
    ##print " FOUND? index for date & date = ", dateIndex, datearray[dateIndex]
    ##print "last 5 days in datearray = ", datearray[-5:]
    # loop through current symbols and generate stats
    for i, isymbol in enumerate( symbols ) :
        symbolStats = []
        # find index to current symbol (the smart way) and copy quotes to 'quote' array
        ##print "symbols[isymbol], isymbol = ", symbols[i], isymbol
        hdf5SymbolIndex = x.labelindex(isymbol,axis=0)
        ##print " FOUND? index for symbol & symbol = ", hdf5SymbolIndex, isymbol

        quote = x[hdf5SymbolIndex,:].copyx()

        # use 3 moving averages (MA's) to determine if uptrending on desired date
        SMA1 = params['MA2factor'] * np.mean( quote[dateIndex-int(params['MA1']):dateIndex] )    #longest MA
        previousSMA2 = np.mean( quote[dateIndex-int(params['MA2'])-1:dateIndex-1] )              #shortest MA
        SMA2 = np.mean( quote[dateIndex-int(params['MA2']):dateIndex] )                          #shortest MA
        SMA3 = np.mean( quote[dateIndex-int(params['MA3']):dateIndex] )
        if quote[dateIndex] > SMA1 or ( quote[dateIndex] > min(SMA2,SMA3) and SMA2 > previousSMA2 ):
            uptrend = 1
        else:
            uptrend = 0
        #print " date, symbol, uptrend flag = ", str(StatDate), isymbol, uptrend

        # Only output data for uptrending stocks
        if uptrend == 1:
            symbolStats.append(isymbol)
            symbolStats.append(uptrend)

            # Calculate Gain (Loss) over LongPeriod ending on StatDate and also LongPeriod days prior
            LongPeriod = params['LongPeriod']
            GainLoss =  quote[dateIndex] / quote[dateIndex-LongPeriod]
            GainLossPrevious =  quote[dateIndex-LongPeriod] / quote[dateIndex-2*LongPeriod]
            symbolStats.append(GainLoss)
            symbolStats.append(GainLossPrevious)

            ########################################################################
            ### Calculate downside risk measure for weighting stocks.
            ### Use 1./ movingwindow_sharpe_ratio for risk measure.
            ### Modify weights with 1./riskDownside and scale so they sum to 1.0
            ########################################################################

            riskDownside_min = params['riskDownside_min']
            riskDownside_max = params['riskDownside_max']
            gainloss = np.ones((quote.shape[0]),dtype=float)
            gainloss[1:] = quote[1:] / quote[:-1]
            gainloss[isnan(gainloss)]=1.
            sharpe = ( gmean(gainloss[dateIndex-LongPeriod:dateIndex])**252 -1. )     \
                       / ( np.std(gainloss[dateIndex-LongPeriod:dateIndex])*sqrt(252) )
            riskDownside = 1. / sharpe
            riskDownside = np.clip( riskDownside, riskDownside_min, riskDownside_max)
            symbolStats.append(riskDownside)

            # add this stock's ranking data to master list
            # - data includes:  symbol, uptrend state, period gain, previous period gain, riskDownside
            allSymbolStats.append(symbolStats)

    # Use collected data to generate ranks
    allSymbolStatsGainLoss = []
    for i in range(len(allSymbolStats)):
        allSymbolStatsGainLoss.append( allSymbolStats[i][2] )
    ranksGainLossIndices = np.argsort( allSymbolStatsGainLoss )
    #print "ranksGainLossIndices = ", ranksGainLossIndices

    for i in range(len(ranksGainLossIndices)):
        allSymbolStats[ranksGainLossIndices[i]].append(i)

    allSymbolStatsGainLossPrevious = []
    for i in range(len(allSymbolStats)):
        allSymbolStatsGainLossPrevious.append( allSymbolStats[i][3] )
    ranksGainLossIndicesPrevious = np.argsort( allSymbolStatsGainLossPrevious )

    for i in range(len(ranksGainLossIndicesPrevious)):
        allSymbolStats[ranksGainLossIndicesPrevious[i]].append(i)


    ###
    ###
    """
    for i in range(len(allSymbolStats)):
        print "i, allSymbolStats[i] = ", i, allSymbolStats[i]

    from time import sleep
    sleep(30)
    """
    ###
    ###


    # reverse the ranks (make low ranks are biggest gainers)
    # - once finished, remember that lower rank is best performer
    ranksGainLoss = []
    for i in range(len(allSymbolStats)):
        ranksGainLoss.append( allSymbolStats[i][5] )
    ranksGainLoss = np.array( ranksGainLoss )
    """
    for i in range(len(ranksGainLoss)):
        print "i, ranksGainLoss[i] = ", i, ranksGainLoss[i]
    """
    maxRank = np.max(ranksGainLoss)  # use later to reverse ranks so that smallest rank is best performer
    ranksGainLoss -= maxRank-1
    ranksGainLoss *= -1
    ranksGainLoss += 2

    ranksGainLossPrevious = []
    for i in range(len(allSymbolStats)):
        ranksGainLossPrevious.append( allSymbolStats[i][6] )
    ranksGainLossPrevious = np.array( ranksGainLossPrevious )
    maxRank = np.max(ranksGainLossPrevious)  # use later to reverse ranks so that smallest rank is best performer
    ranksGainLossPrevious -= maxRank-1
    ranksGainLossPrevious *= -1
    ranksGainLossPrevious += 2

    # weight deltaRank for best and worst performers differently
    rankoffsetchoice = params['numberStocksTraded']
    rankThresholdPct = params['rankThresholdPct']
    delta = -1.0*(ranksGainLoss - ranksGainLossPrevious ) / (ranksGainLoss + rankoffsetchoice)

    """
    # if rank is outside acceptable threshold, set deltarank to zero so stock will not be chosen
    #  - remember that low ranks are biggest gainers
    rankThreshold = (1. - rankThresholdPct) * ( ranksGainLoss.max() - ranksGainLoss.min() )
    for ii in range(ranksGainLoss.shape[0]):
        if monthgainloss[ii] > rankThreshold :
            delta[ii,jj] = -monthgainloss.shape[0]/2
    """

    #deltaRank = bn.rankdata(delta,axis=0)
    deltaRank = np.argsort( delta )
    # reverse the ranks (low deltaRank have the fastest improving rank)
    maxrank = np.max(deltaRank)
    deltaRank -= maxrank-1
    deltaRank *= -1
    deltaRank += 2

    ###
    """
    print ""
    print "delta = ", delta
    print ""
    print "deltaRank = ", deltaRank
    print ""
    for i in range(len(delta)):
        print "i, allSymbolStatsGainLossPrevious, allSymbolStatsGainLoss, ranksGainLossPrevious, ranksGainLoss, delta, deltaRank = ",i, allSymbolStatsGainLossPrevious[i], allSymbolStatsGainLoss[i], ranksGainLossPrevious[i], ranksGainLoss[i], delta[i], deltaRank[i]
    print "shapes of allSymbolStats, delta, deltaRank = ", len(allSymbolStats), len(delta), len(deltaRank)
    """
    ###
    for i in range(len(deltaRank)):
        allSymbolStats[deltaRank[i]-1].append(i)

    """
    for i in range(len(allSymbolStats)):
        print "i, allSymbolStats[i] = ", i, allSymbolStats[i]
        """

    from time import sleep
    #sleep(30)

    return allSymbolStats


