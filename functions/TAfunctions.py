import numpy as np
from numpy import isnan

def interpolate(self, method='linear'):
    """
    Interpolate missing values (after the first valid value)
    Parameters
    ----------
    method : {'linear'}
    Interpolation method.
    Time interpolation works on daily and higher resolution
    data to interpolate given length of interval
    Returns
    -------
    interpolated : Series
    from-- https://github.com/wesm/pandas/blob/master/pandas/core/series.py
    edited to keep only 'linear' method
    Usage: infill NaN values with linear interpolated values
    """
    #import numpy as np
    inds = np.arange(len(self))
    values = self.copy()
    invalid = isnan(values)
    valid = -invalid
    firstIndex = valid.argmax()
    valid = valid[firstIndex:]
    invalid = invalid[firstIndex:]
    inds = inds[firstIndex:]
    result = values.copy()
    result[firstIndex:][invalid] = np.interp(inds[invalid], inds[valid],values[firstIndex:][valid])
    return result
#----------------------------------------------
def cleantobeginning(self):
    """
    Copy missing values (to all dates prior the first valid value)

    Usage: infill NaN values at beginning with copy of first valid value
    """
    #import numpy as np
    inds = np.arange(len(self))
    values = self.copy()
    invalid = isnan(values)
    valid = -invalid
    firstIndex = valid.argmax()
    for i in range(firstIndex):
        values[i]=values[firstIndex]
    return values
#----------------------------------------------
def dpgchannel(x,minperiod,maxperiod,incperiod):
    #import numpy as np
    periods = np.arange(minperiod,maxperiod,incperiod)
    minchannel = np.zeros(len(x),dtype=float)
    maxchannel = np.zeros(len(x),dtype=float)
    for i in range(len(x)):
        divisor = 0
        for j in range(len(periods)):
            minx = max(1,i-periods[j])
            #print "i,j,periods[j],minx,x[minx:i]   :",i,j,periods[j],minx,x[i],x[minx:i]
            if len(x[minx:i]) < 1:
                #print "short   ",i,j,x[i]
                minchannel[i] = minchannel[i] + x[i]
                maxchannel[i] = maxchannel[i] + x[i]
                divisor += 1
            else:
                minchannel[i] = minchannel[i] + min(x[minx:i+1])
                maxchannel[i] = maxchannel[i] + max(x[minx:i+1])
                divisor += 1
        minchannel[i] /= divisor
        maxchannel[i] /= divisor
    return minchannel,maxchannel
#----------------------------------------------
def dpgchannel_2D(x,minperiod,maxperiod,incperiod):
    #import numpy as np
    periods = np.arange(minperiod,maxperiod,incperiod)
    minchannel = np.zeros( (x.shape[0],x.shape[1]), dtype=float)
    maxchannel = np.zeros( (x.shape[0],x.shape[1]), dtype=float)
    for i in range( x.shape[1] ):
        divisor = 0
        for j in range(len(periods)):
            minx = max(1,i-periods[j])
            #print "i,j,periods[j],minx,x[minx:i]   :",i,j,periods[j],minx,x[i],x[minx:i]
            if len(x[0,minx:i]) < 1:
                #print "short   ",i,j,x[i]
                minchannel[:,i] = minchannel[:,i] + x[:,i]
                maxchannel[:,i] = maxchannel[:,i] + x[:,i]
                divisor += 1
            else:
                minchannel[:,i] = minchannel[:,i] + np.min(x[:,minx:i+1],axis=-1)
                maxchannel[:,i] = maxchannel[:,i] + np.max(x[:,minx:i+1],axis=-1)
                divisor += 1
        minchannel[:,i] /= divisor
        maxchannel[:,i] /= divisor
    return minchannel,maxchannel
#----------------------------------------------
def SMA_2D(x,periods):
    #import numpy as np
    SMA = np.zeros( (x.shape[0],x.shape[1]), dtype=float)
    for i in range( x.shape[1] ):
        minx = max(0,i-periods)
        SMA[:,i] = np.mean(x[:,minx:i+1],axis=-1)
    return SMA
#----------------------------------------------
def move_sharpe_2D(adjClose,dailygainloss,period):
    """
    Compute the moving sharpe ratio
      sharpe_ratio = ( gmean(PortfolioDailyGains[-lag:])**252 -1. )
                   / ( np.std(PortfolioDailyGains[-lag:])*sqrt(252) )
      formula assume 252 trading days per year

    Geometric mean is simplified as follows:
    where the geometric mean is being used to determine the average
    growth rate of some quantity, and the initial and final values
    of that quantity are known, the product of the measured growth
    rate at every step need not be taken. Instead, the geometric mean
    is simply ( a(n)/a(0) )**(1/n), where n is the number of steps
    """
    from scipy.stats import gmean
    from math import sqrt
    from numpy import std
    #
    print "period,adjClose.shape  :",period,adjClose.shape
    #
    gmeans = np.ones( (adjClose.shape[0],adjClose.shape[1]), dtype=float)
    stds   = np.ones( (adjClose.shape[0],adjClose.shape[1]), dtype=float)
    sharpe = np.zeros( (adjClose.shape[0],adjClose.shape[1]), dtype=float)
    for i in range( dailygainloss.shape[1] ):
        minindex = max( i-period, 0 )
        #if i < period*1.2:
            #print "i,minindex,period,adjClose.shape  :",i,minindex,period,adjClose.shape
        if i > minindex :
            #gmeans[:,i] = (adjClose[:,i]/adjClose[:,minindex])**(1./(i-minindex))
            #stds[:,i] = np.std(dailygainloss[:,minindex:i+1],axis=-1)
            sharpe[:,i] = ( gmean(dailygainloss[:,minindex:i+1],axis=-1)**252 -1. )     \
                   / ( np.std(dailygainloss[:,minindex:i+1],axis=-1)*sqrt(252) )
        else :
            #gmeans[:,i] = 1.
            #stds[:,i] = 10.
            sharpe[:,i] = 0.
    #gmeans = gmeans**252 -1.
    #stds  *= sqrt(252)
    #return gmeans/stds
    #print 'sharpe[:,-50] inside move_sharpe_2D  ',sharpe[:,-50]   #### diagnostic
    #print '# zero values inside move_sharpe_2D  ',sharpe[sharpe==0.].shape[0]   #### diagnostic
    sharpe[sharpe==0]=.05
    sharpe[isnan(sharpe)] =.05
    #print '# zero values inside move_sharpe_2D  ',sharpe[sharpe==0.].shape[0]   #### diagnostic
    return sharpe

#----------------------------------------------
def sharpeWeightedRank_2D(datearray,adjClose,signal2D,LongPeriod,rankthreshold,riskDownside_min,riskDownside_max,rankThresholdPct):

    # adjClose      --     # 2D array with adjusted closing prices (axes are stock number, date)
    # rankthreshold --     # select this many funds with best recent performance
    
    print "starting function sharpeWeightedRank_2D..."

    import numpy as np
    import nose
    import bottleneck as bn

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

    # if adjClose is nan, set deltarank to zero so stock will not be chosen
    #  - remember that low ranks are biggest gainers
    rankThreshold = (1. - rankThresholdPct) * ( monthgainlossRank.max() - monthgainlossRank.min() )
    for ii in range(monthgainloss.shape[0]):
		if isnan(adjClose[ii,-1])  :
			delta[ii,:] = -monthgainloss.shape[0]/2
			print "*******setting delta (Rank) low...",ii

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
    """
    print "rankthreshold = ",rankthreshold
    print "activeCount = ",activeCount
    print "minrank = ",minrank
    print "adjClose.shape[0] = ",adjClose.shape[0]
    print "formula =",float(rankthreshold)*(activeCount-minrank+1)/adjClose.shape[0]
    #rankthresholdpercentequiv = rint(float(rankthreshold)*(activeCount-minrank+1)/adjClose.shape[0])
    """
    rankthresholdpercentequiv = np.round(float(rankthreshold)*(activeCount-minrank+1)/adjClose.shape[0])
    ranktest = deltaRank <= rankthresholdpercentequiv

    ########################################################################
    ### Calculate downside risk measure for weighting stocks.
    ### Use 1./ movingwindow_sharpe_ratio for risk measure.
    ### Modify weights with 1./riskDownside and scale so they sum to 1.0
    ########################################################################

    riskDownside = 1. / move_sharpe_2D(adjClose,gainloss,LongPeriod)
    #riskDownside = np.clip( riskDownside, .1, 2.5)
    riskDownside = np.clip( riskDownside, riskDownside_min, riskDownside_max)

    #print 'riskDownside [:,-50]',riskDownside[:,-50]   #### diagnostic
    #print 'riskDownside [:,-1]  ',riskDownside[:,-1]   #### diagnostic

    riskDownside[isnan(riskDownside)] = np.max(riskDownside[~isnan(riskDownside)])
    for ii in range(riskDownside.shape[0]) :
        riskDownside[ii] = riskDownside[ii] / np.sum(riskDownside,axis=0)

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
                    monthgainlossweight[jj,ii]  = monthgainlossweight[jj,ii] / riskDownside[jj,ii]
                else:
                    monthgainlossweight[jj,ii]  = 0.
        elif activeCount[ii] == 0 :
            monthgainlossweight[:,ii]  *= 0.
            monthgainlossweight[:,ii]  += 1./adjClose.shape[0]
        else :
            elsedate = datearray[ii]
            elsecount += 1
            monthgainlossweight[:,ii]  = 1./activeCount[ii]

    """
    import pylab as plt
    plt.figure(2)
    plt.clf()
    plt.title('rankthresholdpercentequiv')
    plt.plot(datearray,rankthresholdpercentequiv,'bo')
    plt.plot(datearray,rankthresholdpercentequiv,'b-')
    plt.plot(datearray,activeCount,'r.')
    plt.plot(datearray,activeCount,'r-')
    for iii in range(monthgainlossweight.shape[0]):
        plt.plot(datearray,monthgainlossweight[iii,:])
    plt.show()
    from time import sleep
    sleep(10)
    plt.close(2)
    """


    print " 3 - monthgainlossweight ontains NaN's???",monthgainlossweight[isnan(monthgainlossweight)].shape
    aaa = np.sum(monthgainlossweight,axis=0)
    print " 3 - monthgainlossweight sums to zero ???",aaa[aaa == 0].shape

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

    print "leaving function sharpeWeightedRank_2D..."

    return monthgainlossweight

#----------------------------------------------
def UnWeightedRank_2D(datearray,adjClose,signal2D,LongPeriod,rankthreshold,riskDownside_min,riskDownside_max,rankThresholdPct):

    # adjClose      --     # 2D array with adjusted closing prices (axes are stock number, date)
    # rankthreshold --     # select this many funds with best recent performance

    import numpy as np
    import nose
    import bottleneck as bn

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
    """
    print "rankthreshold = ",rankthreshold
    print "activeCount = ",activeCount
    print "minrank = ",minrank
    print "adjClose.shape[0] = ",adjClose.shape[0]
    print "formula =",float(rankthreshold)*(activeCount-minrank+1)/adjClose.shape[0]
    #rankthresholdpercentequiv = rint(float(rankthreshold)*(activeCount-minrank+1)/adjClose.shape[0])
    """
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

    print " 3 - monthgainlossweight ontains NaN's???",monthgainlossweight[isnan(monthgainlossweight)].shape
    aaa = np.sum(monthgainlossweight,axis=0)
    print " 3 - monthgainlossweight sums to zero ???",aaa[aaa == 0].shape

    print ""
    print " invoking correction to monthgainlossweight....."
    print ""
    # find first date with number of stocks trading (rankthreshold) + 2
    activeCountAboveMinimum = activeCount
    activeCountAboveMinimum += -rankthreshold + 2
    firstTradeDate = np.argmax( np.clip( activeCountAboveMinimum, 0 , 1 ) )
    for ii in np.arange(firstTradeDate,monthgainloss.shape[1]) :
        if np.sum(monthgainlossweight[:,ii]) == 0:
            for kk in range(rankthreshold):
                #print "invoking correction to monthgainlosswight  ----- ii, kk, np.argmin(deltaRank[:,ii]) ", ii, kk, np.argmin(deltaRank[:,ii]) , 1./rankthreshold
                indexHighDeltaRank = np.argmin(deltaRank[:,ii]) # remember that best performance is lowest deltaRank
                #print "             before: ", monthgainlossweight[:,ii]
                monthgainlossweight[indexHighDeltaRank,ii]  = 1./rankthreshold
                #print "             after : ", monthgainlossweight[:,ii]
                deltaRank[indexHighDeltaRank,ii] = 1000.

    '''import pylab as plt
    plt.figure(2)
    plt.clf()
    plt.title('rankthresholdpercentequiv')
    plt.plot(datearray,rankthresholdpercentequiv,'bo')
    plt.plot(datearray,rankthresholdpercentequiv,'b-')
    plt.plot(datearray,activeCount,'r.')
    plt.plot(datearray,activeCount,'r-')
    plt.plot(datearray,aaa,'r-',lw=7)
    plt.plot(datearray,np.sum(monthgainlossweight,axis=0),'g-',lw=4)
    for iii in range(monthgainlossweight.shape[0]):
        plt.plot(datearray,monthgainlossweight[iii,:])
        #print "deltaRank min and max = ", iii, deltaRank.min(),deltaRank.max()
        #plt.plot(datearray,deltaRank[iii,:])
    plt.show()
    from time import sleep
    sleep(10)
    plt.close(2)
    '''

    print " weights calculation else clause encountered :",elsecount," times. last date encountered is ",elsedate
    rankweightsum = np.sum(monthgainlossweight,axis=0)

    monthgainlossweight[isnan(monthgainlossweight)] = 0.  # changed result from 1 to 0

    monthgainlossweight = monthgainlossweight / np.sum(monthgainlossweight,axis=0)
    monthgainlossweight[isnan(monthgainlossweight)] = 0.  # changed result from 1 to 0

    return monthgainlossweight
