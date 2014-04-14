


def dailyBacktest_pctLong():

    import time, threading
    
    import numpy as np
    from matplotlib.pylab import *
    import matplotlib.gridspec as gridspec
    import os
    
    import datetime
    from scipy import random
    from scipy import ndimage
    from random import choice
    from scipy.stats import rankdata
    
    
    import pandas as pd
    
    from scipy.stats import gmean
    
    ## local imports
    from functions.quotes_for_list_adjClose import *
    from functions.TAfunctions import *
    from functions.UpdateSymbols_inHDF5 import UpdateHDF5, loadQuotes_fromHDF

    #---------------------------------------------

    # number of monte carlo scenarios
    randomtrials = 31

    ##
    ##  Import list of symbols to process.
    ##

    # read list of symbols from disk.
    filename = os.path.join( os.getcwd(), 'symbols', 'Naz100_Symbols.txt' )

    ###############################################################################################
    ###  UpdateHDF5( symbols_directory, symbols_file )  ### assume hdf is already up to date
    adjClose, symbols, datearray, _, _ = loadQuotes_fromHDF( filename )
    firstdate = datearray[0]


    # Clean up missing values in input quotes
    #  - infill interior NaN values using nearest good values to linearly interpolate
    #  - copy first valid quote from valid date to all earlier positions
    #  - copy last valid quote from valid date to all later positions
    for ii in range(adjClose.shape[0]):
        adjClose[ii,:] = interpolate(adjClose[ii,:])
        adjClose[ii,:] = cleantobeginning(adjClose[ii,:])
        adjClose[ii,:] = cleantoend(adjClose[ii,:])


    import os
    basename = os.path.split( filename )[-1]
    print "basename = ", basename

    # set up to write monte carlo results to disk.
    if basename == "symbols.txt" :
        runnum = 'run2501'
        plotmax = 1.e5     # maximum value for plot (figure 3)
        holdMonths = [1,2,3,4,6,12]
    elif basename == "Naz100_Symbols.txt" :
        runnum = 'run2502'
        plotmax = 1.e9     # maximum value for plot (figure 3)
        holdMonths = [1,1,1,2,2,3,4,6,12]
    elif basename == "biglist.txt" :
        runnum = 'run2503'
        plotmax = 1.e9     # maximum value for plot (figure 3)
        holdMonths = [1,2,3,4,6,12]
    elif basename == "ProvidentFundSymbols.txt" :
        runnum = 'run2504'
        plotmax = 1.e7     # maximum value for plot (figure 3)
        holdMonths = [4,6,12]
    elif basename == "sp500_symbols.txt" :
        runnum = 'run2505'
        plotmax = 1.e8     # maximum value for plot (figure 3)
        holdMonths = [1,2,3,4,6,12]
    elif basename == "cmg_symbols.txt" :
        runnum = 'run2507'
        plotmax = 1.e7     # maximum value for plot (figure 3)
        holdMonths = [3,4,6,12]
    else :
        runnum = 'run2506'
        plotmax = 1.e7     # maximum value for plot (figure 3)
        holdMonths = [1,2,3,4,6,12]

    if firstdate == (2003,1,1):
        runnum=runnum+"short"
        plotmax /= 100
        plotmax = max(plotmax,100000)
    elif firstdate == (2007,1,1):
        runnum=runnum+"vshort"
        plotmax /= 250
        plotmax = max(plotmax,100000)

    print " security values check: ",adjClose[isnan(adjClose)].shape

    ########################################################################
    # take inverse of quotes for declines
    ########################################################################
    for iCompany in range( adjClose.shape[0] ):
        tempQuotes = adjClose[iCompany,:]
        tempQuotes[ np.isnan(tempQuotes) ] = 1.0
        index = np.argmax(np.clip(np.abs(tempQuotes[::-1]-1),0,1e-8)) - 1
        if index == -1:
            lastquote = adjClose[iCompany,-1]
            lastquote = lastquote ** 2
        else:
            lastQuoteIndex = -index-1
            lastquote = adjClose[iCompany,lastQuoteIndex]
            print "\nlast quote index and quote for ", symbols[iCompany],lastQuoteIndex,adjClose[iCompany,lastQuoteIndex]
            lastquote = lastquote ** 2
            adjClose[iCompany,lastQuoteIndex:] = adjClose[iCompany,lastQuoteIndex-1]
            print adjClose[iCompany,lastQuoteIndex-3:]

    ########################################################################

    gainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    activeCount = np.zeros(adjClose.shape[1],dtype=float)

    numberSharesCalc = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    gainloss[:,1:] = adjClose[:,1:] / adjClose[:,:-1]
    gainloss[isnan(gainloss)]=1.
    value = 10000. * np.cumprod(gainloss,axis=1)   ### used bigger constant for inverse of quotes
    BuyHoldFinalValue = np.average(value,axis=0)[-1]

    print " gainloss check: ",gainloss[isnan(gainloss)].shape
    print " value check: ",value[isnan(value)].shape
    lastEmptyPriceIndex = np.zeros(adjClose.shape[0],dtype=int)
    firstTrailingEmptyPriceIndex = np.zeros(adjClose.shape[0],dtype=int)

    for ii in range(adjClose.shape[0]):
        # take care of special case where constant share price is inserted at beginning of series
        index = np.argmax(np.clip(np.abs(gainloss[ii,:]-1),0,1e-8)) - 1
        print "fist valid price and date = ",symbols[ii]," ",index," ",datearray[index]
        lastEmptyPriceIndex[ii] = index
        activeCount[lastEmptyPriceIndex[ii]+1:] += 1

    for ii in range(adjClose.shape[0]):
        # take care of special case where no quote exists at end of series
        tempQuotes = adjClose[ii,:]
        tempQuotes[ np.isnan(tempQuotes) ] = 1.0
        index = np.argmax(np.clip(np.abs(tempQuotes[::-1]-1),0,1e-8)) - 1
        if index != -1:
            firstTrailingEmptyPriceIndex[ii] = -index
            print "first trailing invalid price: index and date = ",symbols[ii]," ",firstTrailingEmptyPriceIndex[ii]," ",datearray[index]
            activeCount[firstTrailingEmptyPriceIndex[ii]:] -= 1


    FinalTradedPortfolioValue = np.zeros(randomtrials,dtype=float)
    PortfolioReturn = np.zeros(randomtrials,dtype=float)
    PortfolioSharpe = np.zeros(randomtrials,dtype=float)
    MaxPortfolioValue = np.zeros(adjClose.shape[1],dtype=float)
    MaxBuyHoldPortfolioValue = np.zeros(adjClose.shape[1],dtype=float)
    periodForSignal = np.zeros(randomtrials,dtype=float)
    numberStocksUpTrending = np.zeros( (randomtrials,adjClose.shape[1]), dtype=float)
    numberStocksUpTrendingNearHigh = np.zeros( adjClose.shape[1], dtype=float)
    numberStocksUpTrendingBeatBuyHold = np.zeros( adjClose.shape[1], dtype=float)

    LP_montecarlo = np.zeros(randomtrials,dtype=float)
    MA1_montecarlo = np.zeros(randomtrials,dtype=float)
    MA2_montecarlo = np.zeros(randomtrials,dtype=float)
    MA2offset_montecarlo = np.zeros(randomtrials,dtype=float)
    numberStocksTraded_montecarlo = np.zeros(randomtrials,dtype=float)
    monthsToHold_montecarlo = np.zeros(randomtrials,dtype=float)
    riskDownside_min_montecarlo = np.zeros(randomtrials,dtype=float)
    riskDownside_max_montecarlo = np.zeros(randomtrials,dtype=float)
    sma2factor_montecarlo = np.zeros(randomtrials,dtype=float)
    rankThresholdPct_montecarlo = np.zeros(randomtrials,dtype=float)

    for iter in range(randomtrials):

        if iter%1==0:
            print ""
            print ""
            print " random trial:  ",iter

        LongPeriod_random = int(random.uniform(149,180)+.5)
        LongPeriod_random = int(random.uniform(55,280)+.5)
        MA1 = int(random.uniform(15,250)+.5)
        MA2 = int(random.uniform(7,30)+.5)
        MA2offset = int(random.uniform(.6,5)+.5)
        numberStocksTraded = int(random.uniform(1.9,8.9)+.5)
        monthsToHold = choice(holdMonths)
        print ""
        print "months to hold = ",holdMonths,monthsToHold
        print ""
        riskDownside_min = random.triangular(.2,.25,.3)
        riskDownside_max = random.triangular(3.5,4.25,5)
        sma2factor = .99
        rankThresholdPct = int(random.triangular(0,2,25)) / 100.

        ##
        ## these ranges are for naz100
        ##


        sma2factor = random.triangular(.85,.91,.999)
        ##
        ##
        ##

        LongPeriod = LongPeriod_random

        if iter >= randomtrials/2 :
            print "\n\n\n"
            print "*********************************\nUsing pyTAAA parameters .....\n"
            ### original values

            numberStocksTraded = 9
            monthsToHold = 1
            LongPeriod = 256
            MA1 = 120
            MA2 = 16
            MA3 = 18
            sma2factor = .909
            rankThresholdPct = .03
            riskDownside_min = .288
            riskDownside_max = 4.449

            paramNumberToVary = choice([0,1,2,3,4,5,6,7,8,9])

            if paramNumberToVary == 0 :
                numberStocksTraded += choice([-1,0,1])
            if paramNumberToVary == 1 :
                for kk in range(15):
                    temp = choice(holdMonths)
                    if temp != monthsToHold:
                        monthsToHold = temp
                        break
            if paramNumberToVary == 2 :
                LongPeriod = LongPeriod * np.around(random.uniform(-.01*LongPeriod, .01*LongPeriod))
            if paramNumberToVary == 3 :
                MA1 = MA1 * np.around(random.uniform(-.01*MA1, .01*MA1))
            if paramNumberToVary == 4 :
                MA2 = MA2 * np.around(random.uniform(-.01*MA2, .01*MA2))
            if paramNumberToVary == 5 :
                MA3 = MA3 * np.around(random.uniform(-.01*MA3, .01*MA3))
            if paramNumberToVary == 6 :
                sma2factor = sma2factor * np.around(random.uniform(-.01*sma2factor, .01*sma2factor),-3)
            if paramNumberToVary == 7 :
                rankThresholdPct = rankThresholdPct * np.around(random.uniform(-.01*rankThresholdPct, .01*rankThresholdPct),-2)
            if paramNumberToVary == 8 :
                riskDownside_min = riskDownside_min * np.around(random.uniform(-.01*riskDownside_min, .01*riskDownside_min),-3)
            if paramNumberToVary == 9 :
                riskDownside_max = riskDownside_max * np.around(random.uniform(-.01*riskDownside_max, .01*riskDownside_max),-3)


        if iter < randomtrials/2 :
            paramNumberToVary = -999
            print "\n\n\n"
            print "*********************************\nUsing pyTAAA parameters .....\n"
            numberStocksTraded = choice([5,6,7,8])
            monthsToHold = choice([1,2])
            LongPeriod = int( 251 * random.uniform(.8,1.2) )
            MA1 = int( 116 * random.uniform(.8,1.2) )
            MA2 = int( 18 * random.uniform(.8,1.2) )
            MA3 = int( 21 * random.uniform(.8,1.2) )
            sma2factor = .882 * random.uniform(.9,1.1)
            rankThresholdPct = .08 * random.uniform(.9,1.1)
            riskDownside_min = .27903 * random.uniform(.9,1.1)
            riskDownside_max = 4.30217 * random.uniform(.9,1.1)

        if iter == randomtrials-1 :
            print "\n\n\n"
            print "*********************************\nUsing pyTAAApi linux edition parameters .....\n"
            numberStocksTraded = 7
            monthsToHold = 4
            monthsToHold = 1
            LongPeriod = 104
            MA1 = 207
            MA2 = 26
            MA3 = 29
            sma2factor = .911
            rankThresholdPct = .02
            riskDownside_min = .272
            riskDownside_max = 4.386

            numberStocksTraded= 7
            monthsToHold=       1
            LongPeriod=       244
            MA1=              249
            MA2=               18
            MA3=               20
            sma2factor=          .961
            rankThresholdPct=    .20
            riskDownside_min=    .234
            riskDownside_max=   4.694

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

            # take care of special case where mp quote exists at end of series
            if firstTrailingEmptyPriceIndex[ii] != 0:
                signal2D[ii,firstTrailingEmptyPriceIndex[ii]:] = 0

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
        ### compute weights for each stock based on:
        ### 1. uptrending signal in "signal2D"
        ### 1. delta-rank computed from gain/loss over "LongPeriod_random"
        ### 2. sharpe ratio computed from daily gains over "LongPeriod"
        ########################################################################

        monthgainlossweight = sharpeWeightedRank_2D(datearray,symbols,adjClose,signal2D,LongPeriod,numberStocksTraded,riskDownside_min,riskDownside_max,rankThresholdPct)

        print "here I am........"

        ########################################################################
        ### compute traded value of stock for each month
        ########################################################################

        monthvalue = value.copy()
        print " 1 - monthvalue check: ",monthvalue[isnan(monthvalue)].shape
        #print '1 - monthvalue',monthvalue[:,-50]   #### diagnostic
        for ii in np.arange(1,monthgainloss.shape[1]):
            #if datearray[ii].month <> datearray[ii-1].month:
            #if iter==0:
            #   print " date,test = ", datearray[ii], (datearray[ii].month != datearray[ii-1].month) and (datearray[ii].month ==1 or datearray[ii].month == 5 or datearray[ii].month == 9)
            if (datearray[ii].month != datearray[ii-1].month) and ( (datearray[ii].month - 1)%monthsToHold == 0):
                valuesum=np.sum(monthvalue[:,ii-1])
                #print " re-balancing ",datearray[ii],valuesum
                for jj in range(value.shape[0]):
                    #monthvalue[jj,ii] = signal2D[jj,ii]*valuesum*gainloss[jj,ii]   # re-balance using weights (that sum to 1.0)
                    monthvalue[jj,ii] = monthgainlossweight[jj,ii]*valuesum*gainloss[jj,ii]   # re-balance using weights (that sum to 1.0)
                    ###if ii>0 and ii<30:print "      ",jj,ii," previous value, new value:  ",monthvalue[jj,ii-1],monthvalue[jj,ii],valuesum,monthgainlossweight[jj,ii],valuesum,gainloss[jj,ii],monthgainlossweight[jj,ii]
            else:
                for jj in range(value.shape[0]):
                    monthvalue[jj,ii] = monthvalue[jj,ii-1]*gainloss[jj,ii]


        numberSharesCalc = monthvalue / adjClose    # for info only


        ########################################################################
        ### gather statistics on number of uptrending stocks
        ########################################################################

        numberStocksUpTrending[iter,:] = numberStocks
        numberStocksUpTrendingMedian = np.median(numberStocksUpTrending[:iter,:],axis=0)
        numberStocksUpTrendingMean   = np.mean(numberStocksUpTrending[:iter,:],axis=0)

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
        PortfolioSharpe[iter] = ( gmean(PortfolioDailyGains)**252 -1. ) / ( np.std(PortfolioDailyGains)*sqrt(252) )

        print "15 year : ",index,PortfolioValue[-1], PortfolioValue[-index],datearray[-index]

        Return15Yr = (PortfolioValue[-1] / PortfolioValue[-index])**(252./index)
        Return10Yr = (PortfolioValue[-1] / PortfolioValue[-2520])**(1/10.)
        Return5Yr = (PortfolioValue[-1] / PortfolioValue[-1260])**(1/5.)
        Return3Yr = (PortfolioValue[-1] / PortfolioValue[-756])**(1/3.)
        Return2Yr = (PortfolioValue[-1] / PortfolioValue[-504])**(1/2.)
        Return1Yr = (PortfolioValue[-1] / PortfolioValue[-252])
        PortfolioReturn[iter] = gmean(PortfolioDailyGains)**252 -1.

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

        if iter == 0:
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

        if iter == 0:
            beatBuyHoldCount = 0
            beatBuyHold2Count = 0
        beatBuyHoldTest = ( (Sharpe15Yr-BuyHoldSharpe15Yr)/15. + \
                            (Sharpe10Yr-BuyHoldSharpe10Yr)/10. + \
                            (Sharpe5Yr-BuyHoldSharpe5Yr)/5. + \
                            (Sharpe3Yr-BuyHoldSharpe3Yr)/3. + \
                            (Sharpe2Yr-BuyHoldSharpe2Yr)/2. + \
                            (Sharpe1Yr-BuyHoldSharpe1Yr)/1. ) / (1/15. + 1/10.+1/5.+1/3.+1/2.+1)
        if beatBuyHoldTest > 0. :
            beatBuyHoldCount += 1

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
            beatBuyHold2Count += 1
            numberStocksUpTrendingBeatBuyHold = (numberStocksUpTrendingBeatBuyHold * (beatBuyHold2Count -1) + numberStocks) / beatBuyHold2Count


        print "beatBuyHoldTest = ", beatBuyHoldTest, beatBuyHoldTest2
        print "countof trials that BeatBuyHold  = ", beatBuyHoldCount
        print "countof trials that BeatBuyHold2 = ", beatBuyHold2Count
        print ""
        print ""

        from scipy.stats import scoreatpercentile
        if iter > 1:
            for jj in range(adjClose.shape[1]):
                numberStocksUpTrendingNearHigh[jj]   = scoreatpercentile(numberStocksUpTrending[:iter,jj], 90)

        if iter == 0:
            from time import sleep
            for i in range(len(symbols)):
                clf()
                grid()
                ##plot(datearray,signal2D[i,:]*np.mean(adjClose[i,:])*numberStocksTraded/2)
                plot(datearray,adjClose[i,:])
                aaa = signal2D[i,:]
                NaNcount = aaa[np.isnan(aaa)].shape[0]
                title("signal2D before figure3 ... "+symbols[i]+"   "+str(NaNcount))
                draw()
                #time.sleep(.2)

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


        ########################################################################
        ### compute traded value of stock for each month (using varying percent invested)
        ########################################################################

        ###
        ### gather sum of all quotes minus SMA
        ###
        QminusSMADays = int(random.uniform(252,5*252)+.5)
        QminusSMAFactor = random.triangular(.88,.91,.999)

        # re-calc constant monthPctInvested
        uptrendConst = random.uniform(0.45,0.75)
        PctInvestSlope = random.triangular(2.,5.,7.)
        PctInvestIntercept = -random.triangular(-.05,0.0,.07)
        maxPctInvested = choice([1.0,1.0,1.0,1.2,1.33,1.5])

        if iter == randomtrials-1 :
            print "\n\n\n"
            print "*********************************\nUsing pyTAAA parameters .....\n"
            QminusSMADays = 355
            QminusSMAFactor = .90
            PctInvestSlope = 5.45
            PctInvestIntercept = -.01
            maxPctInvested = 1.25

        adjCloseSMA = QminusSMAFactor * SMA_2D( adjClose, QminusSMADays )  # MA1 is longest
        QminusSMA = np.zeros( adjClose.shape[1], 'float' )
        for ii in range( 1,adjClose.shape[1] ):
            ajdClose_date = adjClose[:,ii]
            ajdClose_prevdate = adjClose[:,ii-1]
            adjCloseSMA_date = adjCloseSMA[:,ii]
            ajdClose_date_edit = ajdClose_date[ajdClose_date != ajdClose_prevdate]
            adjCloseSMA_date_edit = adjCloseSMA_date[ajdClose_date != ajdClose_prevdate]
            QminusSMA[ii] = np.sum( ajdClose_date_edit - adjCloseSMA_date_edit  ) / np.sum( adjCloseSMA_date_edit )

        ###
        ### do MACD on monthPctInvested
        ###
        monthPctInvestedDaysMAshort = int(random.uniform(5,35)+.5)

        monthPctInvestedSMAshort = SMA( QminusSMA, monthPctInvestedDaysMAshort )
        monthPctInvestedDaysMAlong = int(random.uniform(3,100)+.5) + monthPctInvestedDaysMAshort
        monthPctInvestedSMAlong = SMA( QminusSMA, monthPctInvestedDaysMAlong )
        monthPctInvestedMACD = monthPctInvestedSMAshort - monthPctInvestedSMAlong

        aa = ( QminusSMA + PctInvestIntercept ) * PctInvestSlope
        monthPctInvested = np.clip( aa, 0., maxPctInvested )


        print " NaNs in value = ", (value[np.isnan(value)]).shape

        monthvalueVariablePctInvest = value.copy()
        print " 1 - monthvalueVariablePctInvest check: ",monthvalueVariablePctInvest[isnan(monthvalueVariablePctInvest)].shape
        for ii in np.arange(1,monthgainloss.shape[1]):
            if (datearray[ii].month != datearray[ii-1].month) and ( (datearray[ii].month - 1)%monthsToHold == 0):
                valuesum=np.sum(monthvalueVariablePctInvest[:,ii-1])
                for jj in range(value.shape[0]):
                    monthvalueVariablePctInvest[jj,ii] = monthgainlossweight[jj,ii]*valuesum*(1.0+(gainloss[jj,ii]-1.0)*monthPctInvested[ii])   # re-balance using weights (that sum to 1.0)
            else:
                monthPctInvested[ii] = monthPctInvested[ii-1]
                for jj in range(value.shape[0]):
                    monthvalueVariablePctInvest[jj,ii] = monthvalueVariablePctInvest[jj,ii-1]*(1.0+(gainloss[jj,ii]-1.0)*monthPctInvested[ii])

        ########################################################################
        ### gather statistics on number of uptrending stocks (using varying percent invested)
        ########################################################################

        index = 3780
        if monthvalueVariablePctInvest.shape[1] < 3780: index = monthvalueVariablePctInvest.shape[1]

        PortfolioValue = np.average(monthvalueVariablePctInvest,axis=0)
        PortfolioDailyGains = PortfolioValue[1:] / PortfolioValue[:-1]
        VarPctSharpe15Yr = ( gmean(PortfolioDailyGains[-index:])**252 -1. ) / ( np.std(PortfolioDailyGains[-index:])*sqrt(252) )
        VarPctSharpe10Yr = ( gmean(PortfolioDailyGains[-2520:])**252 -1. ) / ( np.std(PortfolioDailyGains[-2520:])*sqrt(252) )
        VarPctSharpe5Yr = ( gmean(PortfolioDailyGains[-1260:])**252 -1. ) / ( np.std(PortfolioDailyGains[-1260:])*sqrt(252) )
        VarPctSharpe3Yr = ( gmean(PortfolioDailyGains[-756:])**252 -1. ) / ( np.std(PortfolioDailyGains[-756:])*sqrt(252) )
        VarPctSharpe2Yr = ( gmean(PortfolioDailyGains[-504:])**252 -1. ) / ( np.std(PortfolioDailyGains[-504:])*sqrt(252) )
        VarPctSharpe1Yr = ( gmean(PortfolioDailyGains[-252:])**252 -1. ) / ( np.std(PortfolioDailyGains[-252:])*sqrt(252) )

        print "15 year : ",index,PortfolioValue[-1], PortfolioValue[-index],datearray[-index]

        VarPctReturn15Yr = (PortfolioValue[-1] / PortfolioValue[-index])**(252./index)
        VarPctReturn10Yr = (PortfolioValue[-1] / PortfolioValue[-2520])**(1/10.)
        VarPctReturn5Yr = (PortfolioValue[-1] / PortfolioValue[-1260])**(1/5.)
        VarPctReturn3Yr = (PortfolioValue[-1] / PortfolioValue[-756])**(1/3.)
        VarPctReturn2Yr = (PortfolioValue[-1] / PortfolioValue[-504])**(1/2.)
        VarPctReturn1Yr = (PortfolioValue[-1] / PortfolioValue[-252])

        MaxPortfolioValue *= 0.
        for jj in range(PortfolioValue.shape[0]):
            MaxPortfolioValue[jj] = max(MaxPortfolioValue[jj-1],PortfolioValue[jj])
        PortfolioDrawdown = PortfolioValue / MaxPortfolioValue - 1.
        VarPctDrawdown15Yr = np.mean(PortfolioDrawdown[-index:])
        VarPctDrawdown10Yr = np.mean(PortfolioDrawdown[-2520:])
        VarPctDrawdown5Yr = np.mean(PortfolioDrawdown[-1260:])
        VarPctDrawdown3Yr = np.mean(PortfolioDrawdown[-756:])
        VarPctDrawdown2Yr = np.mean(PortfolioDrawdown[-504:])
        VarPctDrawdown1Yr = np.mean(PortfolioDrawdown[-252:])


        beatBuyHoldTestVarPct = ( (VarPctSharpe15Yr-BuyHoldSharpe15Yr)/15. + \
                                (VarPctSharpe10Yr-BuyHoldSharpe10Yr)/10. + \
                                (VarPctSharpe5Yr-BuyHoldSharpe5Yr)/5. + \
                                (VarPctSharpe3Yr-BuyHoldSharpe3Yr)/3. + \
                                (VarPctSharpe2Yr-BuyHoldSharpe2Yr)/2. + \
                                (VarPctSharpe1Yr-BuyHoldSharpe1Yr)/1. ) / (1/15. + 1/10.+1/5.+1/3.+1/2.+1)


        beatBuyHoldTest2VarPct = 0
        if VarPctReturn15Yr > BuyHoldReturn15Yr: beatBuyHoldTest2VarPct += 1
        if VarPctReturn10Yr > BuyHoldReturn10Yr: beatBuyHoldTest2VarPct += 1
        if VarPctReturn5Yr  > BuyHoldReturn5Yr:  beatBuyHoldTest2VarPct += 1
        if VarPctReturn3Yr  > BuyHoldReturn3Yr:  beatBuyHoldTest2VarPct += 1.5
        if VarPctReturn2Yr  > BuyHoldReturn2Yr:  beatBuyHoldTest2VarPct += 2
        if VarPctReturn1Yr  > BuyHoldReturn1Yr:  beatBuyHoldTest2VarPct += 2.5
        if VarPctReturn15Yr > 0: beatBuyHoldTest2VarPct += 1
        if VarPctReturn10Yr > 0: beatBuyHoldTest2VarPct += 1
        if VarPctReturn5Yr  > 0: beatBuyHoldTest2VarPct += 1
        if VarPctReturn3Yr  > 0: beatBuyHoldTest2VarPct += 1.5
        if VarPctReturn2Yr  > 0: beatBuyHoldTest2VarPct += 2
        if VarPctReturn1Yr  > 0: beatBuyHoldTest2VarPct += 2.5
        if VarPctDrawdown15Yr > BuyHoldDrawdown15Yr: beatBuyHoldTest2VarPct += 1
        if VarPctDrawdown10Yr > BuyHoldDrawdown10Yr: beatBuyHoldTest2VarPct += 1
        if VarPctDrawdown5Yr  > BuyHoldDrawdown5Yr:  beatBuyHoldTest2VarPct += 1
        if VarPctDrawdown3Yr  > BuyHoldDrawdown3Yr:  beatBuyHoldTest2VarPct += 1.5
        if VarPctDrawdown2Yr  > BuyHoldDrawdown2Yr:  beatBuyHoldTest2VarPct += 2
        if VarPctDrawdown1Yr  > BuyHoldDrawdown1Yr:  beatBuyHoldTest2VarPct += 2.5
        # make it a ratio ranging from 0 to 1
        beatBuyHoldTest2VarPct /= 27


        ########################################################################
        ### plot results
        ########################################################################


        matplotlib.rcParams['figure.edgecolor'] = 'grey'
        rc('savefig',edgecolor = 'grey')
        fig = figure(1)
        clf()
        subplotsize = gridspec.GridSpec(3,1,height_ratios=[4,1,1])
        subplot(subplotsize[0])
        grid()
        ##
        ## make plot of all stocks' individual prices
        ##
        if iter == 0:
            yscale('log')
            ylim([1000,max(10000,plotmax)])
            ymin, ymax = emath.log10(1e3), emath.log10(max(10000,plotmax))
            bin_width = (ymax - ymin) / 50
            y_bins = np.arange(ymin, ymax+.0000001, bin_width)
            AllStocksHistogram = np.ones((len(y_bins)-1, len(datearray),3))
            HH = np.zeros((len(y_bins)-1, len(datearray)))
            mm = np.zeros(len(datearray))
            xlocs = []
            xlabels = []
            for i in xrange(1,len(datearray)):
                ValueOnDate = value[:,i]
                if ValueOnDate[ValueOnDate == 10000].shape[0] > 1:
                    ValueOnDate[ValueOnDate == 10000] = 0.
                    ValueOnDate[np.argmin(ValueOnDate)] = 10000.
                h, _ = np.histogram(np.log10(ValueOnDate), bins=y_bins, density=True)
                # reverse so big numbers become small(and print out black)
                h = 1. - h
                # set range to [.5,1.]
                h /= 2.
                h += .5
                HH[:,i] = h
                mm[i] = np.median(value[-1,:])
                if datearray[i].year != datearray[i-1].year:
                    print " inside histogram evaluation for date = ", datearray[i]
                    xlocs.append(i)
                    xlabels.append(str(datearray[i].year))
            AllStocksHistogram[:,:,2] = HH
            AllStocksHistogram[:,:,1] = AllStocksHistogram[:,:,2]
            AllStocksHistogram = np.clip(AllStocksHistogram,AllStocksHistogram.max()*.05,1)
            AllStocksHistogram /= AllStocksHistogram.max()

        plt.imshow(AllStocksHistogram, cmap='Reds_r', aspect='auto', origin='lower', extent=(0, len(datearray), 10**ymin, 10**ymax))
        plt.grid()
        ##
        ## cumulate final values for grayscale histogram overlay
        ##
        if iter == 0:
            MonteCarloPortfolioValues = np.zeros( (randomtrials, len(datearray) ), float )
        MonteCarloPortfolioValues[iter,:] = np.average(monthvalue,axis=0)

        ##
        ## cumulate final values for grayscale histogram overlay
        ##
        if iter == 0:
            MonteCarloPortfolioValues = np.zeros( (randomtrials, len(datearray) ), float )
        MonteCarloPortfolioValues[iter,:] = np.average(monthvalue,axis=0)


        if iter > 9 and iter%10 == 0:
            yscale('log')
            ylim([1000,max(10000,plotmax)])
            ymin, ymax = emath.log10(1e3), emath.log10(max(10000,plotmax))
            bin_width = (ymax - ymin) / 50
            y_bins = np.arange(ymin, ymax+.0000001, bin_width)
            H = np.zeros((len(y_bins)-1, len(datearray)))
            m = np.zeros(len(datearray))
            hb = np.zeros((len(y_bins)-1, len(datearray),3))
            for i in xrange(1,len(datearray)):
                h, _ = np.histogram(np.log10(MonteCarloPortfolioValues[:iter,i]), bins=y_bins, density=True)
                # reverse so big numbers become small(and print out black)
                h = 1. - h
                # set range to [.5,1.]
                h = np.clip( h, .05, 1. )
                h /= 2.
                h += .5
                H[:,i] = h
                m[i] = np.median(value[-1,:])
                if datearray[i].year != datearray[i-1].year:
                    print " inside histogram evaluation for date = ", datearray[i]
            hb[:,:,0] = H
            hb[:,:,1] = H
            hb[:,:,2] = H
            hb = .5 * AllStocksHistogram + .5 * hb

        if iter > 10  :
            yscale('log')
            plt.imshow(hb, cmap='gray', aspect='auto', origin='lower', extent=(0, len(datearray), 10**ymin, 10**ymax))

        yscale('log')
        plot( np.average(monthvalue,axis=0), lw=3, c='k' )
        grid()
        draw()

        ##
        ## continue
        ##
        FinalTradedPortfolioValue[iter] = np.average(monthvalue[:,-1])
        fFinalTradedPortfolioValue = "{:,}".format(int(FinalTradedPortfolioValue[iter]))
        PortfolioDailyGains = np.average(monthvalue,axis=0)[1:] / np.average(monthvalue,axis=0)[:-1]
        PortfolioSharpe[iter] = ( gmean(PortfolioDailyGains)**252 -1. ) / ( np.std(PortfolioDailyGains)*sqrt(252) )
        fPortfolioSharpe = format(PortfolioSharpe[iter],'5.2f')

        FinalTradedVarPctPortfolioValue = np.average(monthvalueVariablePctInvest[:,-1])
        fVFinalTradedPortfolioValue = "{:,}".format(int(FinalTradedVarPctPortfolioValue))
        PortfolioDailyGains = np.average(monthvalueVariablePctInvest,axis=0)[1:] / np.average(monthvalueVariablePctInvest,axis=0)[:-1]
        PortVarPctfolioSharpe = ( gmean(PortfolioDailyGains)**252 -1. ) / ( np.std(PortfolioDailyGains)*sqrt(252) )
        fVPortfolioSharpe = format(PortVarPctfolioSharpe,'5.2f')

        print ""
        print " value 2 yrs ago, 1 yr ago, last = ",np.average(monthvalue[:,-504]),np.average(monthvalue[:,-252]),np.average(monthvalue[:,-1])
        print " one year gain, daily geom mean, stdev = ",np.average(monthvalue,axis=0)[-1] / np.average(monthvalue,axis=0)[-252],gmean(PortfolioDailyGains[-252:])**252 -1.,np.std(PortfolioDailyGains[-252:])*sqrt(252)
        print " two year gain, daily geom mean, stdev = ",np.average(monthvalue,axis=0)[-1] / np.average(monthvalue,axis=0)[-504],gmean(PortfolioDailyGains[-504:])**252 -1.,np.std(PortfolioDailyGains[-504:])*sqrt(252)

        title_text = str(iter)+":  "+ \
                      str(int(numberStocksTraded))+"__"+   \
                      str(int(monthsToHold))+"__"+   \
                      str(int(LongPeriod))+"-"+   \
                      str(int(MA1))+"-"+   \
                      str(int(MA2))+"-"+   \
                      str(int(MA2+MA2offset))+"-"+   \
                      format(sma2factor,'5.3f')+"_"+   \
                      format(rankThresholdPct,'.1%')+"__"+   \
                      format(riskDownside_min,'6.3f')+"-"+  \
                      format(riskDownside_max,'6.3f')+"__"+   \
                      fFinalTradedPortfolioValue+'__'+   \
                      fPortfolioSharpe+'\n{'+   \
                      str(QminusSMADays)+"-"+   \
                      format(QminusSMAFactor,'6.2f')+"_-"+   \
                      format(PctInvestSlope,'6.2f')+"_"+   \
                      format(PctInvestIntercept,'6.2f')+"_"+   \
                      format(maxPctInvested,'4.2f')+"}__"+   \
                      fVFinalTradedPortfolioValue+'__'+   \
                      fVPortfolioSharpe

        title( title_text, fontsize = 9 )
        fSharpe15Yr = format(Sharpe15Yr,'5.2f')
        fSharpe10Yr = format(Sharpe10Yr,'5.2f')
        fSharpe5Yr = format(Sharpe5Yr,'5.2f')
        fSharpe3Yr = format(Sharpe3Yr,'5.2f')
        fSharpe2Yr = format(Sharpe2Yr,'5.2f')
        fSharpe1Yr = format(Sharpe1Yr,'5.2f')
        fReturn15Yr = format(Return15Yr,'5.2f')
        fReturn10Yr = format(Return10Yr,'5.2f')
        fReturn5Yr = format(Return5Yr,'5.2f')
        fReturn3Yr = format(Return3Yr,'5.2f')
        fReturn2Yr = format(Return2Yr,'5.2f')
        fReturn1Yr = format(Return1Yr,'5.2f')
        fDrawdown15Yr = format(Drawdown15Yr,'.1%')
        fDrawdown10Yr = format(Drawdown10Yr,'.1%')
        fDrawdown5Yr = format(Drawdown5Yr,'.1%')
        fDrawdown3Yr = format(Drawdown3Yr,'.1%')
        fDrawdown2Yr = format(Drawdown2Yr,'.1%')
        fDrawdown1Yr = format(Drawdown1Yr,'.1%')
        print " one year sharpe = ",fSharpe1Yr
        print ""
        plotrange = log10(plotmax / 1000.)
        text( 50, 1500, filename, fontsize=8 )
        text( 50, 2500, "Backtested on "+datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"), fontsize=7.5 )
        text(50,1000*10**(.95*plotrange),'Period Sharpe AvgProfit  Avg DD',fontsize=7.5)
        text(50,1000*10**(.91*plotrange),'15 Yr '+fSharpe15Yr+'  '+fReturn15Yr+'  '+fDrawdown15Yr,fontsize=8)
        text(50,1000*10**(.87*plotrange),'10 Yr '+fSharpe10Yr+'  '+fReturn10Yr+'  '+fDrawdown10Yr,fontsize=8)
        text(50,1000*10**(.83*plotrange),' 5 Yr  '+fSharpe5Yr+'  '+fReturn5Yr+'  '+fDrawdown5Yr,fontsize=8)
        text(50,1000*10**(.79*plotrange),' 3 Yr  '+fSharpe3Yr+'  '+fReturn3Yr+'  '+fDrawdown3Yr,fontsize=8)
        text(50,1000*10**(.75*plotrange),' 2 Yr  '+fSharpe2Yr+'  '+fReturn2Yr+'  '+fDrawdown2Yr,fontsize=8)
        text(50,1000*10**(.71*plotrange),' 1 Yr  '+fSharpe1Yr+'  '+fReturn1Yr+'  '+fDrawdown1Yr,fontsize=8)

        fVSharpe15Yr = format(VarPctSharpe15Yr,'5.2f')
        fVSharpe10Yr = format(VarPctSharpe10Yr,'5.2f')
        fVSharpe5Yr = format(VarPctSharpe5Yr,'5.2f')
        fVSharpe3Yr = format(VarPctSharpe3Yr,'5.2f')
        fVSharpe2Yr = format(VarPctSharpe2Yr,'5.2f')
        fVSharpe1Yr = format(VarPctSharpe1Yr,'5.2f')
        fVReturn15Yr = format(VarPctReturn15Yr,'5.2f')
        fVReturn10Yr = format(VarPctReturn10Yr,'5.2f')
        fVReturn5Yr = format(VarPctReturn5Yr,'5.2f')
        fVReturn3Yr = format(VarPctReturn3Yr,'5.2f')
        fVReturn2Yr = format(VarPctReturn2Yr,'5.2f')
        fVReturn1Yr = format(VarPctReturn1Yr,'5.2f')
        fVDrawdown15Yr = format(VarPctDrawdown15Yr,'.1%')
        fVDrawdown10Yr = format(VarPctDrawdown10Yr,'.1%')
        fVDrawdown5Yr = format(VarPctDrawdown5Yr,'.1%')
        fVDrawdown3Yr = format(VarPctDrawdown3Yr,'.1%')
        fVDrawdown2Yr = format(VarPctDrawdown2Yr,'.1%')
        fVDrawdown1Yr = format(VarPctDrawdown1Yr,'.1%')

        text(1500,1000*10**(.95*plotrange),'Period Sharpe AvgProfit  Avg DD',fontsize=7.5,color='b')
        text(1500,1000*10**(.91*plotrange),'15 Yr '+fVSharpe15Yr+'  '+fVReturn15Yr+'  '+fVDrawdown15Yr,fontsize=8,color='b')
        text(1500,1000*10**(.87*plotrange),'10 Yr '+fVSharpe10Yr+'  '+fVReturn10Yr+'  '+fVDrawdown10Yr,fontsize=8,color='b')
        text(1500,1000*10**(.83*plotrange),' 5 Yr  '+fVSharpe5Yr+'  '+fVReturn5Yr+'  '+fVDrawdown5Yr,fontsize=8,color='b')
        text(1500,1000*10**(.79*plotrange),' 3 Yr  '+fVSharpe3Yr+'  '+fVReturn3Yr+'  '+fVDrawdown3Yr,fontsize=8,color='b')
        text(1500,1000*10**(.75*plotrange),' 2 Yr  '+fVSharpe2Yr+'  '+fVReturn2Yr+'  '+fVDrawdown2Yr,fontsize=8,color='b')
        text(1500,1000*10**(.71*plotrange),' 1 Yr  '+fVSharpe1Yr+'  '+fVReturn1Yr+'  '+fVDrawdown1Yr,fontsize=8,color='b')

        if beatBuyHoldTest > 0. :
            text(50,1000*10**(.65*plotrange),format(beatBuyHoldTest2,'.2%')+'  beats BuyHold...')
        else:
            text(50,1000*10**(.65*plotrange),format(beatBuyHoldTest2,'.2%'))

        if beatBuyHoldTestVarPct > 0. :
            text(50,1000*10**(.59*plotrange),format(beatBuyHoldTest2VarPct,'.2%')+'  beats BuyHold...',color='b')
        else:
            text(50,1000*10**(.59*plotrange),format(beatBuyHoldTest2VarPct,'.2%'),color='b'
            )

        text(50,1000*10**(.54*plotrange),last_symbols_text,fontsize=8)
        plot(BuyHoldPortfolioValue,lw=3,c='r')
        plot(np.average(monthvalue,axis=0),lw=4,c='k')
        plot(np.average(monthvalueVariablePctInvest,axis=0),lw=2,c='b')
        # set up to use dates for labels
        xlocs = []
        xlabels = []
        for i in xrange(1,len(datearray)):
            if datearray[i].year != datearray[i-1].year:
                xlocs.append(i)
                xlabels.append(str(datearray[i].year))
        print "xlocs,xlabels = ", xlocs, xlabels
        if len(xlocs) < 12 :
            xticks(xlocs, xlabels)
        else:
            xticks(xlocs[::2], xlabels[::2])
        xlim(0,len(datearray))
        subplot(subplotsize[1])
        grid()
        ##ylim(0, value.shape[0])
        ylim(0, 1.2)
        plot(datearray,numberStocksUpTrendingMedian / activeCount,'g-',lw=1)
        plot(datearray,numberStocksUpTrendingNearHigh / activeCount,'b-',lw=1)
        plot(datearray,numberStocksUpTrendingBeatBuyHold / activeCount,'k-',lw=2)
        plot(datearray,numberStocks  / activeCount,'r-')



        subplot(subplotsize[2])
        grid()
        plot(datearray,QminusSMA,'m-',lw=.8)
        plot(datearray,monthPctInvested,'r-',lw=.8)
        draw()
        # save figure to disk, but only if trades produce good results
        if iter==randomtrials-1:
            outputplotname = os.path.join( os.getcwd(), 'pyTAAA_web', 'PyTAAA_monteCarloBacktest.png' )
            savefig(outputplotname, format='png', edgecolor='gray' )

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
        print "portfolio sharpe ratio : ",PortfolioSharpe[iter]

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

        csv_text = runnum+","+str(iter)+","+    \
                      str(numberStocksTraded)+","+   \
                      str(monthsToHold)+","+  \
                      str(LongPeriod)+","+   \
                      str(MA1)+","+   \
                      str(MA2)+","+   \
                      str(MA2+MA2offset)+","+   \
                      str(riskDownside_min)+","+str(riskDownside_max)+","+   \
                      str(FinalTradedPortfolioValue[iter])+','+   \
                      format(sma2factor,'5.3f')+","+  \
                      format(rankThresholdPct,'.1%')+","+  \
                      str(np.std(PortfolioDailyGains)*sqrt(252))+','+   \
                      str(PortfolioSharpe[iter])+','+   \
                      str(targetdate)+','+   \
                      str(Portfolioannualgainfromtargetdate)+','+   \
                      str(Portfoliosharpefromtargetdate)+','+   \
                      str(BHannualgainfromtargetdate)+','+   \
                      str(BHsharpefromtargetdate)+","+   \
                      fSharpe15Yr+","+   \
                      fSharpe10Yr+","+   \
                      fSharpe5Yr+","+   \
                      fSharpe3Yr+","+   \
                      fSharpe2Yr+","+   \
                      fSharpe1Yr+","+   \
                      fReturn15Yr+","+   \
                      fReturn10Yr+","+   \
                      fReturn5Yr+","+   \
                      fReturn3Yr+","+   \
                      fReturn2Yr+","+   \
                      fReturn1Yr+","+   \
                      fDrawdown15Yr+","+   \
                      fDrawdown10Yr+","+   \
                      fDrawdown5Yr+","+   \
                      fDrawdown3Yr+","+   \
                      fDrawdown2Yr+","+   \
                      fDrawdown1Yr+","+   \
                      format(beatBuyHoldTest,'5.3f')+","+\
                      format(beatBuyHoldTest2,'.2%')+","+\
                      str(paramNumberToVary)+\
                      " \n"


        periodForSignal[iter] = LongPeriod


        # create and update counter for holding period
        # with number of random trials choosing this symbol on last date times Sharpe ratio for trial in last year
        print ""
        print ""
        print "cumulative tally of holding periods for last date"
        if iter == 0:
            print "initializing cumulative talley of holding periods chosen for last date..."
            holdmonthscount = np.zeros(len(holdMonths),dtype=float)
        if beatBuyHoldTest > 0 :
            numdays1Yr = 252
            Sharpe1Yr = ( gmean(PortfolioDailyGains[-numdays1Yr:])**252 -1. ) / ( np.std(PortfolioDailyGains[-numdays1Yr:])*sqrt(252) )
            Sharpe2Yr = ( gmean(PortfolioDailyGains[-2*numdays1Yr:])**252 -1. ) / ( np.std(PortfolioDailyGains[-2*numdays1Yr:])*sqrt(252) )
            Sharpe3Yr = ( gmean(PortfolioDailyGains[-3*numdays1Yr:])**252 -1. ) / ( np.std(PortfolioDailyGains[-3*numdays1Yr:])*sqrt(252) )
            Sharpe5Yr = ( gmean(PortfolioDailyGains[-5*numdays1Yr:])**252 -1. ) / ( np.std(PortfolioDailyGains[-5*numdays1Yr:])*sqrt(252) )
            Sharpe10Yr = ( gmean(PortfolioDailyGains[-10*numdays1Yr:])**252 -1. ) / ( np.std(PortfolioDailyGains[-10*numdays1Yr:])*sqrt(252) )
            for ii in range(len(holdMonths)):
                if monthsToHold == holdMonths[ii]:
                    #print symbols[ii],"  weight = ",max( 0., ( .666*Sharpe1Yr + .333*Sharpe2Yr ) * monthgainlossweight[ii,-1] )
                    #symbolscount[ii] += max( 0., ( .666*Sharpe1Yr + .333*Sharpe2Yr ) * monthgainlossweight[ii,-1] )
                    holdmonthscount[ii] += ( 1.0*Sharpe1Yr + 1./2*Sharpe2Yr + 1./3.*Sharpe3Yr + 1./5.*Sharpe5Yr + 1./10.*Sharpe10Yr ) * (1+2+3+5+10)
            bestchoicethreshold = 3. * np.median(holdmonthscount[holdmonthscount > 0.])
            holdmonthscountnorm = holdmonthscount*1.
            if holdmonthscountnorm[holdmonthscountnorm > 0].shape[0] > 0:
                holdmonthscountnorm -= holdmonthscountnorm[holdmonthscountnorm > 0].min()
                holdmonthscountnorm /= holdmonthscountnorm.max()
            holdmonthscountint = np.round(holdmonthscountnorm*40)
            holdmonthscountint[holdmonthscountint == NaN] =0
            for ii in range(len(holdMonths)):
                if holdmonthscountint[ii] > 0:
                    tagnorm = "*"* holdmonthscountint[ii]
                    print format(str(holdMonths[ii]),'7s')+   \
                          str(datearray[-1])+         \
                          format(holdmonthscount[ii],'7.2f'), tagnorm

    return
