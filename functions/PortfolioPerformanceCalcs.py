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

def PortfolioPerformanceCalcs( symbol_directory, symbol_file, params ) :

    ## update quotes from list of symbols
    #(symbols_directory, symbols_file) = os.path.split(filename)
    #print "symbols_directory, symbols.file = ", symbols_directory, symbols_file
    ###############################################################################################      UpdateHDF5( symbols_directory, symbols_file )
    filename = os.path.join(symbol_directory, symbol_file)
    adjClose, symbols, datearray, _, _ = loadQuotes_fromHDF( symbol_file )

    gainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    activeCount = np.zeros(adjClose.shape[1],dtype=float)

    numberSharesCalc = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    gainloss[:,1:] = adjClose[:,1:] / adjClose[:,:-1]
    gainloss[isnan(gainloss)]=1.
    value = 10000. * np.cumprod(gainloss,axis=1)
    BuyHoldFinalValue = np.average(value,axis=0)[-1]

    #print " gainloss check: ",gainloss[isnan(gainloss)].shape
    #print " value check: ",value[isnan(value)].shape
    lastEmptyPriceIndex = np.zeros(adjClose.shape[0],dtype=int)

    for ii in range(adjClose.shape[0]):
        # take care of special case where constant share price is inserted at beginning of series
        index = np.argmax(np.clip(np.abs(gainloss[ii,:]-1),0,1e-8)) - 1
        #print "fist valid price and date = ",symbols[ii]," ",index," ",datearray[index]
        lastEmptyPriceIndex[ii] = index
        activeCount[lastEmptyPriceIndex[ii]+1:] += 1

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
        """
        if iter == 0 :
            print "fist valid price and date = ",symbols[ii]," ",index," ",datearray[index]
            lastEmptyPriceIndex[ii] = index
        """
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

    """
    # make signals sum to 1.0
    for jj in np.arange(1,adjClose.shape[1]):
        signal2D[:,jj] = signal2D[:,jj] / np.sum(signal2D[:,jj],axis=0)
    """

    """
    if iter == 0:
        from time import sleep
        for i in range(len(symbols)):
            clf()
            grid()
            plot(datearray,signal2D[i,:]*np.mean(adjClose[i,:])*numberStocksTraded)
            plot(datearray,adjClose[i,:])
            draw()
            #time.sleep(5)
    """

    #print " signal2D check: ",signal2D[isnan(signal2D)].shape

    ########################################################################
    ### compute weights for each stock based on:
    ### 1. uptrending signal in "signal2D"
    ### 1. delta-rank computed from gain/loss over "LongPeriod_random"
    ### 2. sharpe ratio computed from daily gains over "LongPeriod_random"
    ########################################################################

    monthgainlossweight = sharpeWeightedRank_2D(datearray,symbols,adjClose,signal2D,LongPeriod,numberStocksTraded,riskDownside_min,riskDownside_max,rankThresholdPct)

    #print "here I am........"
    
    '''# plot if it is NFLX
    for ii in range(len(symbols)):
        if symbols[ii] == 'NFLX':    
            print "********** uptrending symbols found: symbol, lastprice = ", symbols[ii], adjClose[ii,-1]
            plt.title(symbols[ii])
            plt.grid()
            plt.text(datearray[10],1,"price has NaN")
            plt.plot( datearray, adjClose[ii,:])
            plt.show()
            for jj in range(-25,0):
				print ii,jj,adjClose.shape[1]+jj,len(datearray), adjClose[ii,jj]
				print "NFLX price: date, adjClose = ",jj, datearray[len(datearray)+jj], adjClose[ii,adjClose.shape[1]+jj], monthgainlossweight[ii,adjClose.shape[1]+jj]
    '''
    
    ########################################################################
    ### compute traded value of stock for each month
    ########################################################################

    monthvalue = value.copy()
    #print " 1 - monthvalue check: ",monthvalue[isnan(monthvalue)].shape
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


    """
    # adjust numberSharesCalc to zero if activeCount is zero or len(symbols)
    for jj in range(value.shape[1]):
        #activeCount[jj] = monthgainlossweight[:,jj][monthgainlossweight[:,jj] > 0].shape[0]
        if activeCount[jj] == 0 :
            numberSharesCalc[:,jj] *= 0
    """

    """
    #print 'montvalue',monthvalue[:,-50]   #### diagnostic
    #print 'montvalue # zeros',monthvalue[monthvalue==0].shape   #### diagnostic
    #print 'montvalue # NaNs ',monthvalue[isnan(monthvalue)].shape   #### diagnostic

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

    """
    """
    # drawdown diagnostic plots
    clf()
    subplot(111)
    grid()
    plot(datearray,MaxBuyHoldPortfolioValue)
    plot(datearray,BuyHoldPortfolioValue)
    plot(datearray,MaxPortfolioValue)
    plot(datearray,PortfolioValue)
    savefig("pngs\Naz100-tripleMAs_montecarlo_"+str(dateForFilename)+runnum+"_"+str(iter)+"MaxValueForDrawdownQC1.png", format='png')
    clf()
    subplot(211)
    grid()
    plot(datearray,PortfolioDrawdown)
    plot(datearray,BuyHoldPortfolioDrawdown)
    subplot(212)
    grid()
    plot(datearray,PortfolioValue)
    plot(datearray,BuyHoldPortfolioValue)
    savefig("pngs\Naz100-tripleMAs_montecarlo_"+str(dateForFilename)+runnum+"_"+str(iter)+"MaxValueForDrawdownQC2.png", format='png')
    """

    """
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
        #print "found monte carlo trial that beats BuyHold..."
        #print "shape of numberStocksUpTrendingBeatBuyHold = ",numberStocksUpTrendingBeatBuyHold.shape
        #print "mean of numberStocksUpTrendingBeatBuyHold values = ",np.mean(numberStocksUpTrendingBeatBuyHold)
        beatBuyHoldCount += 1
        #numberStocksUpTrendingBeatBuyHold = (numberStocksUpTrendingBeatBuyHold * (beatBuyHoldCount -1) + numberStocks) / beatBuyHoldCount

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
    """

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
