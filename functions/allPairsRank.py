import numpy as np
import os
import datetime
from typing import List

from functions.TAfunctions import *

'''
symbol_file = os.path.join( os.getcwd(), 'symbols', 'Naz100_Symbols.txt' )
adjClose, symbols, datearray, _, _ = loadQuotes_fromHDF( symbol_file )
'''

def allPairsRanking(
        adjClose: np.ndarray,
        symbols: list,
        datearray: np.ndarray,
        span: int = 150,
) -> np.ndarray:

    period = span

    rank = np.zeros_like( adjClose )
    ratioAvg = np.zeros( len(symbols), 'float' )

    for d in range(period,len(adjClose[0,:])-1):
        if datearray[d].month != datearray[d-1].month:
            print " date = ", datearray[d],
            for i in range(len(symbols)):
                a = adjClose[i,:d] / adjClose[i,d-period]
                for j in range(len(symbols)):
                    b = adjClose[j,:d] / adjClose[j,d-period]
                    ratio = a / b
                    if j==0 :
                        ratioSum = ratio
                    else:
                        ratioSum += ratio

                ratioSum /= len(symbols)
                minchannel, maxchannel = dpgchannel(ratioSum,9,51,9)
                percentChannel = (ratioSum-minchannel) / (maxchannel-minchannel+1e-10)

                ratioSumChan = ratioSum[-span:] * percentChannel[-span:]
                ratioAvg[i] = np.mean( ratioSumChan )

            order = np.argsort( ratioAvg )[::-1]

            rank[:,d] = order
            print [ symbols[i] for i in [ rank[0:7,d].astype('int') ][0] ]
        else:
            rank[:,d] = rank[:,d-1]
    return rank



def allPairs_sharpeWeightedRank_2D(
        datearray: np.ndarray,
        symbols: list,
        adjClose: np.ndarray,
        signal2D: np.ndarray,
        LongPeriod: int,
        rankthreshold: int,
        riskDownside_min: float,
        riskDownside_max: float,
        rankThresholdPct: float,
        stockList: str = 'Naz100',
) -> np.ndarray:

    # adjClose      --     # 2D array with adjusted closing prices (axes are stock number, date)
    # rankthreshold --     # select this many funds with best recent performance

    import numpy as np
    import nose
    import os
    import sys
    try:
        import bottleneck as bn
        from bn import rankdata as rd
    except:
        import scipy.stats.mstats as bn

    gainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    gainloss[:,1:] = adjClose[:,1:] / adjClose[:,:-1]
    gainloss[isnan(gainloss)]=1.

    # convert signal2D to contain either 1 or 0 for weights
    signal2D -= signal2D.min()
    signal2D *= signal2D.max()

    # apply signal to daily gainloss
    gainloss = gainloss * signal2D
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

    
    ###
    ###
    ###
    
    monthgainlossRank = allPairsRanking( adjClose, symbols, datearray, span=LongPeriod )
    
    ###
    ###
    ###
    
    
    ########monthgainlossRank = bn.rankdata(monthgainloss,axis=0)
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
    delta = -( monthgainlossRank.astype('float') - monthgainlossPreviousRank.astype('float') ) / ( monthgainlossRank.astype('float') + float(rankoffsetchoice) )

    # if rank is outside acceptable threshold, set deltarank to zero so stock will not be chosen
    #  - remember that low ranks are biggest gainers
    rankThreshold = (1. - rankThresholdPct) * ( monthgainlossRank.max() - monthgainlossRank.min() )
    for ii in range(monthgainloss.shape[0]):
        for jj in range(monthgainloss.shape[1]):
            if monthgainloss[ii,jj] > rankThreshold :
                delta[ii,jj] = -monthgainloss.shape[0]/2
                if jj == monthgainloss.shape[1]:
                    print "*******setting delta (Rank) low... Stock has rank outside acceptable range... ",ii, symbols[ii], monthgainloss[ii,jj]

    # if adjClose is nan, set deltarank to zero so stock will not be chosen
    #  - remember that low ranks are biggest gainers
    rankThreshold = (1. - rankThresholdPct) * ( monthgainlossRank.max() - monthgainlossRank.min() )
    for ii in range(monthgainloss.shape[0]):
        if isnan( adjClose[ii,-1] )  :
            delta[ii,:] = -monthgainloss.shape[0]/2
            numisnans = adjClose[ii,:]
            # NaN in last value usually means the stock is removed from the index so is not updated, but history is still in HDF file
            print "*******setting delta (Rank) low... Stock has NaN for last value... ",ii, symbols[ii], numisnans[np.isnan(numisnans)].shape

    deltaRank = bn.rankdata( delta, axis=0 )

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

    for ii in range(1,monthgainloss.shape[1]):
        if datearray[ii].month == datearray[ii-1].month:
            monthgainloss[:,ii] = monthgainloss[:,ii-1]
            delta[:,ii] = delta[:,ii-1]
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
    sumallzerotest = allzerotest[allzerotest == 0].shape
    if sumallzerotest > 0:
        print ""
        print " invoking correction to monthgainlossweight....."
        print ""
        for ii in np.arange(1,monthgainloss.shape[1]) :
            if np.sum(monthgainlossweight[:,ii]) == 0:
                monthgainlossweight[:,ii]  = 1./activeCount[ii]

    print " weights calculation else clause encountered :",elsecount," times. last date encountered is ",elsedate
    rankweightsum = np.sum(monthgainlossweight,axis=0)

    monthgainlossweight[isnan(monthgainlossweight)] = 0.  # changed result from 1 to 0

    monthgainlossweight = monthgainlossweight / np.sum(monthgainlossweight,axis=0)
    monthgainlossweight[isnan(monthgainlossweight)] = 0.  # changed result from 1 to 0

    # input symbols and company names from text file
    companyName_file = os.path.join( os.getcwd(), "symbols",  "companyNames.txt" )
    with open( companyName_file, "r" ) as f:
        companyNames = f.read()

    print "\n\n\n"
    companyNames = companyNames.split("\n")
    ii = companyNames.index("")
    del companyNames[ii]
    companySymbolList  = []
    companyNameList = []
    for iname,name in enumerate(companyNames):
        name = name.replace("amp;", "")
        testsymbol, testcompanyName = name.split(";")
        companySymbolList.append(testsymbol)
        companyNameList.append(testcompanyName)

    # print list showing current rankings and weights
    # - symbol
    # - rank
    # - weight from sharpe ratio
    # - price
    import os
    rank_text = "<div id='rank_table_container'><h3>"+"<p>Current stocks, with ranks, weights, and prices are :</p></h3><font face='courier new' size=3><table border='1'> \
               <tr><td>Rank \
               </td><td>Symbol \
               </td><td>Company \
               </td><td>Weight \
               </td><td>Price  \
               </td><td>Trend  \
               </td></tr>\n"
    for i, isymbol in enumerate(symbols):
        for j in range(len(symbols)):
            if int( deltaRank[j,-1] ) == i :
                if signal2D[j,-1] == 1.:
                    trend = 'up'
                else:
                    trend = 'down'

                # search for company name
                try:
                    symbolIndex = companySymbolList.index(symbols[j])
                    companyName = companyNameList[symbolIndex]
                except:
                    companyName = ""

                rank_text = rank_text + \
                       "<tr><td>" + format(deltaRank[j,-1],'6.0f')  + \
                       "<td>" + format(symbols[j],'5s')  + \
                       "<td>" + format(companyName,'15s')  + \
                       "<td>" + format(monthgainlossweight[j,-1],'5.03f') + \
                       "<td>" + format(adjClose[j,-1],'6.2f')  + \
                       "<td>" + trend  + \
                       "</td></tr>  \n"
    rank_text = rank_text + "</table></div>\n"

    filepath = os.path.join( os.getcwd(), "pyTAAA_web", "pyTAAAweb_RankList.txt" )
    with open( filepath, "w" ) as f:
        f.write(rank_text)

    print "leaving function sharpeWeightedRank_2D..."

    return monthgainlossweight
