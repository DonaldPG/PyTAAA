import os
# Force matplotlib to not use any Xwindows backend.
import matplotlib
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
# Set DPI for inline plots and saved figures
plt.rcParams['figure.figsize'] = (16*.75, 9*.75)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
from math import sqrt
from functions.GetParams import (
    get_json_params, get_symbols_file, get_webpage_store, get_performance_store
)
#from functions.quotes_for_list_adjClose import *
from functions.TAfunctions import (
    cleantobeginning, cleantoend,interpolate
)
from functions.UpdateSymbols_inHDF5 import loadQuotes_fromHDF
from functions.CountNewHighsLows import newHighsAndLows


def plotRecentPerfomance3(
        indexRealtimeStart, _datearray, symbols,
        _value, _monthvalue,
        _AllStocksHistogram,
        _MonteCarloPortfolioValues,
        _FinalTradedPortfolioValue,
        _TradedPortfolioValue,
        _BuyHoldPortfolioValue,
        _numberStocksUpTrending,
        last_symbols_text,
        _activeCount,
        _numberStocks,
        _numberStocksUpTrendingNearHigh,
        _numberStocksUpTrendingBeatBuyHold,
        png_fn,
        json_fn
):

    # make local copies of arrays after indexRealtimeStart (2013,1,1)
    # adjust all starting values to 10,000 on (2013,1,1)
    datearray = _datearray[indexRealtimeStart:]
    value = _value[:,indexRealtimeStart:].copy()
    monthvalue = _monthvalue[:,indexRealtimeStart:].copy()
    AllStocksHistogram = _AllStocksHistogram[:,indexRealtimeStart:,:].copy()
    MonteCarloPortfolioValues = _MonteCarloPortfolioValues[:,indexRealtimeStart:].copy()
    for i in range(MonteCarloPortfolioValues.shape[0]):
        MonteCarloPortfolioValues[i,:] =MonteCarloPortfolioValues[i,:] * 10000. / MonteCarloPortfolioValues[i,0]
    FinalTradedPortfolioValue = _FinalTradedPortfolioValue.copy()
    TradedPortfolioValue = _TradedPortfolioValue[indexRealtimeStart:].copy()
    TradedPortfolioValue = TradedPortfolioValue  * 10000./ TradedPortfolioValue[0]
    BuyHoldPortfolioValue = _BuyHoldPortfolioValue[indexRealtimeStart:].copy()
    BuyHoldPortfolioValue = BuyHoldPortfolioValue * 10000./ BuyHoldPortfolioValue[0]
    numberStocksUpTrending = _numberStocksUpTrending[:,indexRealtimeStart:].copy()
    activeCount = _activeCount[indexRealtimeStart:].copy()
    numberStocks = _numberStocks[indexRealtimeStart:].copy()
    numberStocksUpTrendingNearHigh = _numberStocksUpTrendingNearHigh[indexRealtimeStart:].copy()
    numberStocksUpTrendingBeatBuyHold = _numberStocksUpTrendingBeatBuyHold[indexRealtimeStart:].copy()

    import numpy as np
    from matplotlib import pylab as plt
    import matplotlib.gridspec as gridspec
    import os

    import datetime

    from scipy.stats import gmean
    from math import sqrt

    ## local imports

    #from functions.UpdateSymbols_inHDF5 import UpdateHDF5, loadQuotes_fromHDF
    # from functions.UpdateSymbols_inHDF5 import UpdateHDF_yf, loadQuotes_fromHDF

    print("*************************************************************")
    print("*************************************************************")
    print("***                                                       ***")
    print("***                                                       ***")
    print("***  daily montecarlo backtest                            ***")
    print("***  recent performance update                            ***")
    print("***                                                       ***")
    print("***                                                       ***")
    print("*************************************************************")
    print("*************************************************************")
    from time import sleep
    sleep(5)

    params = get_json_params(json_fn)
    trade_cost = params['trade_cost']
    monthsToHold = params['monthsToHold']
    numberStocksTraded = params['numberStocksTraded']
    LongPeriod = params['LongPeriod']
    stddevThreshold = params['stddevThreshold']
    MA1 = params['MA1']
    MA2 = params['MA2']
    MA3 = params['MA3']
    MA2offset = params['MA3'] - params['MA2']
    sma2factor = params['MA2factor']
    rankThresholdPct = params['rankThresholdPct']
    riskDownside_min = params['riskDownside_min']
    riskDownside_max = params['riskDownside_max']

    ##
    ##  Import list of symbols to process.
    ##
    params = get_json_params(json_fn)
    symbols_file = get_symbols_file(json_fn)
    stockList = params['stockList']

    # # read list of symbols from disk.
    # symbol_directory = os.path.join( os.getcwd(), "symbols" )
    # if stockList == 'Naz100':
    #     symbol_file = "Naz100_Symbols.txt"
    # elif stockList == 'SP500':
    #     symbol_file = "SP500_Symbols.txt"
    # symbols_file = os.path.join( symbol_directory, symbol_file )
    symbol_directory, symbol_file = os.path.split(symbols_file)

    # """
    # # read list of symbols from disk.
    # filename = os.path.join( os.getcwd(), 'symbols', 'Naz100_Symbols.txt' )
    # """

    ########################################################################
    ### clean up data for plotting
    ########################################################################

    # import pdb
    #pdb.set_trace()

    print(" number of nans in value = ", value[np.isnan(value)].shape)
    print(" number of nans in monthvalue = ", monthvalue[np.isnan(monthvalue)].shape)

    print(" number of infs in value = ", value[np.isinf(value)].shape)
    print(" number of infs in monthvalue = ", monthvalue[np.isinf(monthvalue)].shape)

    '''
    value[value==0.] = np.nan
    value[np.isinf(value)] = np.nan
    monthvalue[monthvalue==0.] = np.nan
    monthvalue[np.isinf(monthvalue)] = np.nan
    '''

    ########################################################################
    ### normalize to starting value of $10,000
    ########################################################################

    # '''
    # for ii in range( MonteCarloPortfolioValues.shape[0] ):

    #     #print "ii, ",ii,
    #     aa = MonteCarloPortfolioValues[ii,:].copy()
    #     aa[aa==0.] = np.nan
    #     MonteCarloPortfolioValues[ii,:] = cleantobeginning( aa )
    #     Factor = 10000. / MonteCarloPortfolioValues[ii,0]
    #     #print Factor,
    #     MonteCarloPortfolioValues[ii,:] *= Factor
    #     FinalTradedPortfolioValue[ii] *= Factor
    # '''

    for ii in range( value.shape[0] ):

        #print "ii, symbols[ii], value[ii,0] = ",ii,symbols[ii], value[ii,0],
        #value[ii,:] = interpolate( value[ii,:] )
        if value[ii,0] == 0.:
            aa = value[ii,:].copy()
            aa[aa==0.] = np.nan
            value[ii,:] = cleantobeginning( aa )
        #value[ii,:] = cleantobeginning( value[ii,:] )
        value[ii,:] = cleantoend( value[ii,:] )
        Factor = 10000. / value[ii,0]
        #print 'Factor =', Factor
        value[ii,:] *= Factor

        # '''
        # monthvalue[ii,:] = interpolate( monthvalue[ii,:] )
        # monthvalue[ii,:] = cleantobeginning( monthvalue[ii,:] )
        # monthvalue[ii,:] = cleantoend( monthvalue[ii,:] )
        # '''
    Factor = 10000. / np.average( monthvalue[:,0] )
    #print Factor
    monthvalue *= Factor

    plotmax = 10. ** np.around( np.log10( value.max() ))
    MonteCarloPortfolioValues_NoNaNs = MonteCarloPortfolioValues[~np.isnan(MonteCarloPortfolioValues)]
    plotmax = np.around( 1.2 * MonteCarloPortfolioValues_NoNaNs.max(), decimals=-2 )
    print("\n\n ... plotmax = "+ str(plotmax))

    ########################################################################
    ### gather statistics for recent performance
    ########################################################################

    numberStocksUpTrendingMedian = np.median(numberStocksUpTrending,axis=0)
    numberStocksUpTrendingMean   = np.mean(numberStocksUpTrending,axis=0)

    index = monthvalue.shape[1]-1
    minindex = -np.min([756,len(datearray)-1])
    print("index = ", index, monthvalue.shape, len(datearray),minindex)

    PortfolioValue = np.nanmean(monthvalue,axis=0)
    PortfolioValue = np.mean(monthvalue,axis=0)
    PortfolioDailyGains = PortfolioValue[1:] / PortfolioValue[:-1]
    SharpeLife = ( gmean(PortfolioDailyGains[-index:])**252 -1. ) / ( np.std(PortfolioDailyGains[-index:])*sqrt(252) )
    Sharpe3Yr = ( gmean(PortfolioDailyGains[minindex:])**252 -1. ) / ( np.std(PortfolioDailyGains[minindex:])*sqrt(252) )
    Sharpe1Yr = ( gmean(PortfolioDailyGains[-252:])**252 -1. ) / ( np.std(PortfolioDailyGains[-252:])*sqrt(252) )
    Sharpe6Mo = ( gmean(PortfolioDailyGains[-126:])**252 -1. ) / ( np.std(PortfolioDailyGains[-126:])*sqrt(252) )
    Sharpe3Mo = ( gmean(PortfolioDailyGains[-63:])**252 -1. ) / ( np.std(PortfolioDailyGains[-63:])*sqrt(252) )
    Sharpe1Mo = ( gmean(PortfolioDailyGains[-21:])**252 -1. ) / ( np.std(PortfolioDailyGains[-21:])*sqrt(252) )
    PortfolioSharpe = ( gmean(PortfolioDailyGains)**252 -1. ) / ( np.std(PortfolioDailyGains)*sqrt(252) )

    print("Lifetime : ",index,PortfolioValue[-1], PortfolioValue[-index],datearray[-index])

    ReturnLife = (PortfolioValue[-1] / PortfolioValue[-index])**(252./index)
    Return3Yr = (PortfolioValue[-1] / PortfolioValue[minindex])**(1/3.)
    Return1Yr = (PortfolioValue[-1] / PortfolioValue[-252])
    Return6Mo = (PortfolioValue[-1] / PortfolioValue[-126])**(1/.5)
    Return3Mo = (PortfolioValue[-1] / PortfolioValue[-63])**(1/.25)
    Return1Mo = (PortfolioValue[-1] / PortfolioValue[-21])**(1/(1./12.))
    PortfolioReturn = gmean(PortfolioDailyGains)**252 -1.

    MaxPortfolioValue = np.zeros( PortfolioValue.shape[0], 'float' )
    for jj in range(PortfolioValue.shape[0]):
        MaxPortfolioValue[jj] = max(MaxPortfolioValue[jj-1],PortfolioValue[jj])
    PortfolioDrawdown = PortfolioValue / MaxPortfolioValue - 1.
    DrawdownLife = np.mean(PortfolioDrawdown[-index:])
    Drawdown3Yr = np.mean(PortfolioDrawdown[minindex:])
    Drawdown1Yr = np.mean(PortfolioDrawdown[-252:])
    Drawdown6Mo = np.mean(PortfolioDrawdown[-126:])
    Drawdown3Mo = np.mean(PortfolioDrawdown[-63:])
    Drawdown1Mo = np.mean(PortfolioDrawdown[-21:])

    #BuyHoldPortfolioValue = np.nanmean(value,axis=0)   # now in function call
    BuyHoldDailyGains = BuyHoldPortfolioValue[1:] / BuyHoldPortfolioValue[:-1]
    BuyHoldSharpeLife = ( gmean(BuyHoldDailyGains[-index:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-index:])*sqrt(252) )
    BuyHoldSharpe3Yr = ( gmean(BuyHoldDailyGains[minindex:])**252 -1. ) / ( np.std(BuyHoldDailyGains[minindex:])*sqrt(252) )
    BuyHoldSharpe1Yr  = ( gmean(BuyHoldDailyGains[-252:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-252:])*sqrt(252) )
    BuyHoldSharpe6Mo  = ( gmean(BuyHoldDailyGains[-126:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-126:])*sqrt(252) )
    BuyHoldSharpe3Mo  = ( gmean(BuyHoldDailyGains[-63:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-63:])*sqrt(252) )
    BuyHoldSharpe1Mo  = ( gmean(BuyHoldDailyGains[-21:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-21:])*sqrt(252) )
    BuyHoldReturnLife = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-index])**(252./index)
    BuyHoldReturn3Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[minindex:])**(1/13.)
    BuyHoldReturn1Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-252])
    BuyHoldReturn6Mo = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-126])**(1/.5)
    BuyHoldReturn3Mo = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-63])**(1/.25)
    BuyHoldReturn1Mo = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-21])**(1/(1./12.))
    MaxBuyHoldPortfolioValue = np.zeros( BuyHoldPortfolioValue.shape, 'float' )
    for jj in range(BuyHoldPortfolioValue.shape[0]):
        MaxBuyHoldPortfolioValue[jj] = max(MaxBuyHoldPortfolioValue[jj-1],BuyHoldPortfolioValue[jj])

    BuyHoldPortfolioDrawdown = BuyHoldPortfolioValue / MaxBuyHoldPortfolioValue - 1.
    BuyHoldDrawdownLife = np.mean(BuyHoldPortfolioDrawdown[-index:])
    BuyHoldDrawdown3Yr = np.mean(BuyHoldPortfolioDrawdown[minindex:])
    BuyHoldDrawdown1Yr = np.mean(BuyHoldPortfolioDrawdown[-252:])
    BuyHoldDrawdown6Mo = np.mean(BuyHoldPortfolioDrawdown[-126:])
    BuyHoldDrawdown3Mo = np.mean(BuyHoldPortfolioDrawdown[-63:])
    BuyHoldDrawdown1Mo = np.mean(BuyHoldPortfolioDrawdown[-21:])

    ########################################################################
    ### plot recent results
    ########################################################################

    plt.rcParams['figure.edgecolor'] = 'grey'
    plt.rc('savefig',edgecolor = 'grey')
    fig = plt.figure(1)
    plt.clf()
    subplotsize = gridspec.GridSpec(2,1,height_ratios=[5,1])
    plt.subplot(subplotsize[0])
    plt.grid()
    ##
    ## make plot of all stocks' individual prices
    ##
    #yscale('log')
    #ylim([1000,max(100000,plotmax)])
    plt.ylim([7000,plotmax])
    #ymin, ymax = emath.log10(1e3), emath.log10(max(10000,plotmax))
    ymin, ymax = 7000,plotmax
    #bin_width = (np.log10(ymax) - np.log10(ymin)) / 50
    #y_bins = np.arange(np.log10(ymin), np.log10(ymax)+.0000001, bin_width)
    n_bins = 150
    #bin_width = (ymax - ymin) / n_bins
    #y_bins = np.arange(ymin, ymax+.0000001, bin_width)
    y_bins = np.linspace(ymin, ymax, n_bins)

    AllStocksHistogram = np.ones((len(y_bins)-1, len(datearray),3))
    HH = np.zeros((len(y_bins)-1, len(datearray)))
    mm = np.zeros(len(datearray))
    xlocs = []
    xlabels = []
    for i in range(1,len(datearray)):
        #ValueOnDate = np.log10(value[:,i])
        ValueOnDate = value[:,i]
        '''
        if ValueOnDate[ValueOnDate == 10000].shape[0] > 1:
            ValueOnDate[ValueOnDate == 10000] = 0.
            ValueOnDate[np.argmin(ValueOnDate)] = 10000.
        '''
        #h, _ = np.histogram(np.log10(ValueOnDate), bins=y_bins, density=True)
        h, _ = np.histogram(ValueOnDate, bins=y_bins, density=True)
        h /= h.sum()
        # reverse so big numbers become small(and print out black)
        h = 1. - h
        # set range to [.5,1.]
        h /= 2.
        h += .5
        #print "idatearray[i],h min,mean,max = ", h.min(),h.mean(),h.max()
        HH[:,i] = h
        mm[i] = np.median(value[-1,:])
        if datearray[i].year != datearray[i-1].year:
            # print(" inside histogram evaluation for date = ", datearray[i])
            xlocs.append(i)
            xlabels.append(str(datearray[i].year))
    HH -= np.percentile(HH.flatten(),2)
    HH /= HH.max()
    HH = np.clip( HH, 0., 1. )
    #print "HH min,mean,max = ", HH.min(),HH.mean(),HH.max()
    AllStocksHistogram[:,:,2] = HH
    AllStocksHistogram[:,:,1] = AllStocksHistogram[:,:,2]
    AllStocksHistogram = np.clip(AllStocksHistogram,AllStocksHistogram.max()*.05,1)
    AllStocksHistogram /= AllStocksHistogram.max()

    #plt.imshow(AllStocksHistogram, cmap='Reds_r', aspect='auto', origin='lower', extent=(0, len(datearray), 10**ymin, 10**ymax))
    ##plt.imshow(AllStocksHistogram, cmap='Reds_r', aspect='auto', origin='lower', extent=(0, len(datearray), ymin, ymax))
    plt.grid()
    ##
    ## cumulate final values for grayscale histogram overlay
    ##
    '''
    if iter == 0:
        MonteCarloPortfolioValues = np.zeros( (randomtrials, len(datearray) ), float )
    MonteCarloPortfolioValues[iter,:] = np.average(monthvalue,axis=0)
    '''

    ##
    ## cumulate final values for grayscale histogram overlay
    ##

    '''
    #yscale('log')
    #ylim([1000,max(10000,plotmax)])
    plt.ylim([7000,plotmax])
    #ymin, ymax = emath.log10(1e3), emath.log10(max(10000,plotmax))
    ymin, ymax = 7000,plotmax
    #bin_width = (np.log10(ymax) - np.log10(ymin)) / 50
    #y_bins = np.arange(np.log10(ymin), np.log10(ymax)+.0000001, bin_width)
    n_bins = 150
    #bin_width = (np.log10(ymax) - np.log10(ymin)) / n_bins
    #y_bins = np.arange(np.log10(ymin), np.log10(ymax)+.0000001, bin_width)
    y_bins = np.linspace(np.log10(ymin), np.log10(ymax), n_bins)

    H = np.zeros((len(y_bins)-1, len(datearray)))
    m = np.zeros(len(datearray))
    hb = np.zeros((len(y_bins)-1, len(datearray),3))

    ## TODO: remove the following lines after QC and de-bugging
    print("\n\n ... inside dailyBacktest_pctLong.py near line 338")
    print(" ... MonteCarloPortfolioValues[:,:].min() = "+str(MonteCarloPortfolioValues[:,:].min()))
    print(" ... MonteCarloPortfolioValues[:,:].max() = "+str(MonteCarloPortfolioValues[:,:].max()))
    print(" ... MonteCarloPortfolioValues[0,:] = "+str(MonteCarloPortfolioValues[0,:]))
    print(" ... y_bins[0], y_bins[-1] = "+str((y_bins[0], y_bins[-1])))

    for i in range(1,len(datearray)):
        #h, _ = np.histogram(np.log10(MonteCarloPortfolioValues[:,i]), bins=y_bins, density=True)
        h, _ = np.histogram(np.log10(MonteCarloPortfolioValues[:,i]), bins=y_bins, density=True)
        # reverse so big numbers become small(and print out black)
        h = 1. - h
        # set range to [.5,1.]
        h = np.clip( h, .05, 1. )
        h /= 2.
        h += .5
        H[:,i] = h
        m[i] = np.median(value[-1,:])
        if datearray[i].year != datearray[i-1].year:
            print(" inside histogram evaluation for date = ", datearray[i])
    H -= np.percentile(H.flatten(),2)
    H /= H.max()
    H = np.clip( H, 0., 1. )
    hb[:,:,0] = H
    hb[:,:,1] = H
    hb[:,:,2] = H
    hb = .5 * AllStocksHistogram + .5 * hb
    '''

    plt.ylim([7000,plotmax])
    #ymin, ymax = emath.log10(1e3), emath.log10(max(10000,plotmax))
    ymin, ymax = 7000,plotmax
    #bin_width = (np.log10(ymax) - np.log10(ymin)) / 50
    #y_bins = np.arange(np.log10(ymin), np.log10(ymax)+.0000001, bin_width)
    n_bins = 150
    #bin_width = (np.log10(ymax) - np.log10(ymin)) / n_bins
    #y_bins = np.arange(np.log10(ymin), np.log10(ymax)+.0000001, bin_width)
    #y_bins = np.linspace(np.log10(ymin), np.log10(ymax), n_bins)
    #bin_width = (ymax - ymin) / n_bins
    #y_bins = np.arange(ymin, ymax+.0000001, bin_width)
    y_bins = np.linspace(ymin, ymax, n_bins)

    H = np.zeros((len(y_bins)-1, len(datearray)))
    m = np.zeros(len(datearray))
    hb = np.zeros((len(y_bins)-1, len(datearray),3))

    ## TODO: remove the following lines after QC and de-bugging
    print("\n\n ... inside dailyBacktest_pctLong.py near line 338")
    print(" ... MonteCarloPortfolioValues[:,:].min() = "+str(MonteCarloPortfolioValues[:,:].min()))
    print(" ... MonteCarloPortfolioValues[:,:].max() = "+str(MonteCarloPortfolioValues[:,:].max()))
    print(" ... MonteCarloPortfolioValues[0,:] = "+str(MonteCarloPortfolioValues[0,:]))
    print(" ... y_bins[0], y_bins[-1] = "+str((y_bins[0], y_bins[-1])))

    for i in range(1,len(datearray)):
        ValueOnDate = MonteCarloPortfolioValues[:,i]
        #h, _ = np.histogram(np.log10(MonteCarloPortfolioValues[:,i]), bins=y_bins, density=True)
        #h, _ = np.histogram(np.log10(MonteCarloPortfolioValues[:,i]), bins=y_bins, density=True)
        h, _ = np.histogram(ValueOnDate, bins=y_bins, density=True)
        h /= h.sum()
        # reverse so big numbers become small(and print out black)
        h = 1. - h
        # set range to [.5,1.]
        h = np.clip( h, .05, 1. )
        h /= 2.
        h += .5
        H[:,i] = h
        m[i] = np.median(value[-1,:])
        # if datearray[i].year != datearray[i-1].year:
        #     print(" inside histogram evaluation for date = ", datearray[i])
    H -= np.percentile(H.flatten(),2)
    H /= H.max()
    H = np.clip( H, 0., 1. )
    hb[:,:,0] = H
    hb[:,:,1] = H
    hb[:,:,2] = H
    hb = .5 * AllStocksHistogram + .5 * hb

    #yscale('log')
    #plt.imshow(hb, cmap='gray', aspect='auto', origin='lower', extent=(0, len(datearray), 10**ymin, 10**ymax))
    ##plt.imshow(hb, cmap='gray', aspect='auto', origin='lower', extent=(0, len(datearray), ymin, ymax))

    #yscale('log')
    plt.yscale('log')   ## TODO: check this
    plt.plot( np.average(monthvalue,axis=0), lw=3, c='k' )
    plt.grid()
    plt.draw()

    ##
    ## continue
    ##
    #FinalTradedPortfolioValue[iter] = np.average(monthvalue[:,-1])
    fFinalTradedPortfolioValue = "{:,}".format(int(FinalTradedPortfolioValue[-1]))
    PortfolioDailyGains = np.average(monthvalue,axis=0)[1:] / np.average(monthvalue,axis=0)[:-1]
    PortfolioSharpe = ( gmean(PortfolioDailyGains)**252 -1. ) / ( np.std(PortfolioDailyGains)*sqrt(252) )
    fPortfolioSharpe = format(PortfolioSharpe,'5.2f')

    print("")
    print(" value 3 months yrs ago, 1 month ago, last = ",np.average(monthvalue[:,-63]),np.average(monthvalue[:,-21]),np.average(monthvalue[:,-1]))
    print(" one month gain, daily geom mean, stdev = ",np.average(monthvalue,axis=0)[-1] / np.average(monthvalue,axis=0)[-21],gmean(PortfolioDailyGains[-21:])**252 -1.,np.std(PortfolioDailyGains[-252:])*sqrt(252))
    print(" three month gain, daily geom mean, stdev = ",np.average(monthvalue,axis=0)[-1] / np.average(monthvalue,axis=0)[-63],gmean(PortfolioDailyGains[-63:])**252 -1.,np.std(PortfolioDailyGains[-504:])*sqrt(252))

    title_text = str(0)+":  "+ \
                  str(int(numberStocksTraded))+"__"+   \
                  str(int(monthsToHold))+"__"+   \
                  str(int(LongPeriod))+"-"+   \
                  format(stddevThreshold,'4.2f')+"-"+   \
                  str(int(MA1))+"-"+   \
                  str(int(MA2))+"-"+   \
                  str(int(MA2+MA2offset))+"-"+   \
                  format(sma2factor,'5.3f')+"_"+   \
                  format(rankThresholdPct,'.1%')+"__"+   \
                  format(riskDownside_min,'6.3f')+"-"+  \
                  format(riskDownside_max,'6.3f')+"__"+   \
                  fFinalTradedPortfolioValue+'__'+   \
                  fPortfolioSharpe

    plt.title( title_text, fontsize = 9 )
    fSharpeLife = format(SharpeLife,'5.2f')
    fSharpe3Yr = format(Sharpe3Yr,'5.2f')
    fSharpe1Yr = format(Sharpe1Yr,'5.2f')
    fSharpe6Mo = format(Sharpe6Mo,'5.2f')
    fSharpe3Mo = format(Sharpe3Mo,'5.2f')
    fSharpe1Mo = format(Sharpe1Mo,'5.2f')
    fReturnLife = format(ReturnLife,'5.2f')
    fReturn3Yr = format(Return3Yr,'5.2f')
    fReturn1Yr = format(Return1Yr,'5.2f')
    fReturn6Mo = format(Return6Mo,'5.2f')
    fReturn3Mo = format(Return3Mo,'5.2f')
    fReturn1Mo = format(Return1Mo,'5.2f')
    fDrawdownLife = format(DrawdownLife,'.1%')
    fDrawdown3Yr = format(Drawdown3Yr,'.1%')
    fDrawdown1Yr = format(Drawdown1Yr,'.1%')
    fDrawdown6Mo = format(Drawdown6Mo,'.1%')
    fDrawdown3Mo = format(Drawdown3Mo,'.1%')
    fDrawdown1Mo = format(Drawdown1Mo,'.1%')
    print(" one year sharpe = ",fSharpe1Mo)
    print("")
    #plotrange = log10(plotmax / 1000.)
    #plotrange = (plotmax -7000.)
    plotrange = np.log10(plotmax) - np.log10(7000.)
    plt.text( 50,10.**(np.log10(7000)+(.47*plotrange)), symbols_file, fontsize=8 )
    plt.text( 50,10.**(np.log10(7000)+(.43*plotrange)), "Backtested on "+datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"), fontsize=7.5 )
    # '''
    # text(50,1000*10**(.95*plotrange),'Period Sharpe AvgProfit  Avg DD',fontsize=7.5)
    # text(50,1000*10**(.91*plotrange),' Life  '+fSharpeLife+'  '+fReturnLife+'  '+fDrawdownLife,fontsize=8)
    # text(50,1000*10**(.87*plotrange),' 3 Yr  '+fSharpe3Yr+'  '+fReturn3Yr+'  '+fDrawdown3Yr,fontsize=8)
    # text(50,1000*10**(.83*plotrange),' 1 Yr  '+fSharpe1Yr+'  '+fReturn1Yr+'  '+fDrawdown1Yr,fontsize=8)
    # text(50,1000*10**(.79*plotrange),' 6 Mo  '+fSharpe6Mo+'  '+fReturn6Mo+'  '+fDrawdown6Mo,fontsize=8)
    # text(50,1000*10**(.75*plotrange),' 3 Mo  '+fSharpe3Mo+'  '+fReturn3Mo+'  '+fDrawdown3Mo,fontsize=8)
    # text(50,1000*10**(.71*plotrange),' 1 Mo  '+fSharpe1Mo+'  '+fReturn1Mo+'  '+fDrawdown1Mo,fontsize=8)

    # text(50,1000*10**(.54*plotrange),last_symbols_text,fontsize=8)
    # '''

    # '''
    # plt.text(50,7000+(.95*plotrange),'Period Sharpe AvgProfit  Avg DD',fontsize=7.5)
    # plt.text(50,7000+(.91*plotrange),' Life  '+fSharpeLife+'  '+fReturnLife+'  '+fDrawdownLife,fontsize=8)
    # plt.text(50,7000+(.87*plotrange),' 3 Yr  '+fSharpe3Yr+'  '+fReturn3Yr+'  '+fDrawdown3Yr,fontsize=8)
    # plt.text(50,7000+(.83*plotrange),' 1 Yr  '+fSharpe1Yr+'  '+fReturn1Yr+'  '+fDrawdown1Yr,fontsize=8)
    # plt.text(50,7000+(.79*plotrange),' 6 Mo  '+fSharpe6Mo+'  '+fReturn6Mo+'  '+fDrawdown6Mo,fontsize=8)
    # plt.text(50,7000+(.75*plotrange),' 3 Mo  '+fSharpe3Mo+'  '+fReturn3Mo+'  '+fDrawdown3Mo,fontsize=8)
    # plt.text(50,7000+(.71*plotrange),' 1 Mo  '+fSharpe1Mo+'  '+fReturn1Mo+'  '+fDrawdown1Mo,fontsize=8)

    # plt.text(50,7000+(.54*plotrange),last_symbols_text,fontsize=8)
    # '''
    # '''
    # plt.text(50,10.**(np.log10(7000)+(.95*plotrange)),'Period Sharpe AvgProfit  Avg DD',fontsize=7.5)
    # plt.text(50,10.**(np.log10(7000)+(.91*plotrange)),' Life  '+fSharpeLife+'  '+fReturnLife+'  '+fDrawdownLife,fontsize=8)
    # plt.text(50,10.**(np.log10(7000)+(.87*plotrange)),' 3 Yr  '+fSharpe3Yr+'  '+fReturn3Yr+'  '+fDrawdown3Yr,fontsize=8)
    # plt.text(50,10.**(np.log10(7000)+(.83*plotrange)),' 1 Yr  '+fSharpe1Yr+'  '+fReturn1Yr+'  '+fDrawdown1Yr,fontsize=8)
    # plt.text(50,10.**(np.log10(7000)+(.79*plotrange)),' 6 Mo  '+fSharpe6Mo+'  '+fReturn6Mo+'  '+fDrawdown6Mo,fontsize=8)
    # plt.text(50,10.**(np.log10(7000)+(.75*plotrange)),' 3 Mo  '+fSharpe3Mo+'  '+fReturn3Mo+'  '+fDrawdown3Mo,fontsize=8)
    # plt.text(50,10.**(np.log10(7000)+(.71*plotrange)),' 1 Mo  '+fSharpe1Mo+'  '+fReturn1Mo+'  '+fDrawdown1Mo,fontsize=8)

    # plt.text(50,10.**(np.log10(7000)+(.54*plotrange)),last_symbols_text,fontsize=8)
    # '''

    plt.text(50,10.**(np.log10(7000)+(.95*plotrange)),'Period Sharpe AvgProfit  Avg DD',fontsize=7.5)
    plt.text(50,10.**(np.log10(7000)+(.91*plotrange)),' Life  '+fSharpeLife+'  '+fReturnLife+'  '+fDrawdownLife,fontsize=8)
    plt.text(50,10.**(np.log10(7000)+(.87*plotrange)),' 3 Yr  '+fSharpe3Yr+'  '+fReturn3Yr+'  '+fDrawdown3Yr,fontsize=8)
    plt.text(50,10.**(np.log10(7000)+(.83*plotrange)),' 1 Yr  '+fSharpe1Yr+'  '+fReturn1Yr+'  '+fDrawdown1Yr,fontsize=8)
    plt.text(50,10.**(np.log10(7000)+(.79*plotrange)),' 6 Mo  '+fSharpe6Mo+'  '+fReturn6Mo+'  '+fDrawdown6Mo,fontsize=8)
    plt.text(50,10.**(np.log10(7000)+(.75*plotrange)),' 3 Mo  '+fSharpe3Mo+'  '+fReturn3Mo+'  '+fDrawdown3Mo,fontsize=8)
    plt.text(50,10.**(np.log10(7000)+(.71*plotrange)),' 1 Mo  '+fSharpe1Mo+'  '+fReturn1Mo+'  '+fDrawdown1Mo,fontsize=8)

    plt.text(50,10.**(np.log10(7000)+(.54*plotrange)),last_symbols_text,fontsize=8)

    plt.plot(BuyHoldPortfolioValue,lw=3,c='r')
    plt.plot(np.average(monthvalue,axis=0),lw=4,c='k')
    # set up to use dates for labels
    xlocs = []
    xlabels = []
    for i in range(1,len(datearray)):
        if datearray[i].year != datearray[i-1].year:
            xlocs.append(i)
            xlabels.append(str(datearray[i].year))
    print("xlocs,xlabels = ", xlocs, xlabels)
    if len(xlocs) < 12 :
        plt.xticks(xlocs, xlabels)
    else:
        plt.xticks(xlocs[::2], xlabels[::2])
    #plt.xlim(0,len(datearray))
    plt.xlim(0,len(datearray)+25)

    for ii in range( MonteCarloPortfolioValues.shape[0] ):
        plt.plot( MonteCarloPortfolioValues[ii,:], c='k', lw=.1, alpha=.75 )
    for ii in range( value.shape[0] ):
        plt.plot( value[ii,:], c=(1.,0.5,0.5), lw=.1, alpha=.5 )

    # '''
    # import pdb
    # pdb.set_trace()
    # '''

    plt.subplot(subplotsize[1])
    plt.grid()
    plt.xlim(datearray[0],datearray[-1]+datetime.timedelta(days=25))
    ##ylim(0, value.shape[0])
    plt.ylim(0, 1.2)
    plt.plot(datearray,numberStocksUpTrendingMedian / activeCount,'g-',lw=1)
    plt.plot(datearray,numberStocksUpTrendingNearHigh / activeCount,'b-',lw=1)
    plt.plot(datearray,numberStocksUpTrendingBeatBuyHold / activeCount,'k-',lw=2)
    plt.plot(datearray,numberStocks  / activeCount,'r-')

    # '''
    # for i,idate in enumerate(datearray):
    #     print "i,datearray[i],activeCount[i] = ",i,idate,activeCount[i]
    # '''

    plt.draw()
    # save figure to disk
    # json_dir = os.path.split(json_fn)[0]
    webpage_dir = get_webpage_store(json_fn)
    outputplotname = os.path.join(webpage_dir, png_fn+'.png' )
    plt.savefig(outputplotname, format='png', edgecolor='gray' )

    return



def plotRecentPerfomance2(
        indexRealtimeStart, _datearray, symbols,
        _value, _monthvalue,
        _AllStocksHistogram,
        _MonteCarloPortfolioValues,
        _FinalTradedPortfolioValue,
        _TradedPortfolioValue,
        _BuyHoldPortfolioValue,
        _numberStocksUpTrending,
        last_symbols_text,
        _activeCount,
        _numberStocks,
        _numberStocksUpTrendingNearHigh,
        _numberStocksUpTrendingBeatBuyHold,
        json_fn
):

    # make local copies of arrays after indexRealtimeStart (2013,1,1)
    # adjust all starting values to 10,000 on (2013,1,1)
    datearray = _datearray[indexRealtimeStart:]
    value = _value[:,indexRealtimeStart:].copy()
    monthvalue = _monthvalue[:,indexRealtimeStart:].copy()
    AllStocksHistogram = _AllStocksHistogram[:,indexRealtimeStart:,:].copy()
    MonteCarloPortfolioValues = _MonteCarloPortfolioValues[:,indexRealtimeStart:].copy()
    for i in range(MonteCarloPortfolioValues.shape[0]):
        MonteCarloPortfolioValues[i,:] =MonteCarloPortfolioValues[i,:] * 10000. / MonteCarloPortfolioValues[i,0]
    FinalTradedPortfolioValue = _FinalTradedPortfolioValue.copy()
    TradedPortfolioValue = _TradedPortfolioValue[indexRealtimeStart:].copy()
    TradedPortfolioValue = TradedPortfolioValue  * 10000./ TradedPortfolioValue[0]
    BuyHoldPortfolioValue = _BuyHoldPortfolioValue[indexRealtimeStart:].copy()
    BuyHoldPortfolioValue = BuyHoldPortfolioValue * 10000./ BuyHoldPortfolioValue[0]
    numberStocksUpTrending = _numberStocksUpTrending[:,indexRealtimeStart:].copy()
    activeCount = _activeCount[indexRealtimeStart:].copy()
    numberStocks = _numberStocks[indexRealtimeStart:].copy()
    numberStocksUpTrendingNearHigh = _numberStocksUpTrendingNearHigh[indexRealtimeStart:].copy()
    numberStocksUpTrendingBeatBuyHold = _numberStocksUpTrendingBeatBuyHold[indexRealtimeStart:].copy()

    # import time, threading

    import numpy as np
    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec
    import os

    import datetime
    # from numpy import random
    # from scipy import ndimage
    # from random import choice
    # from scipy.stats import rankdata


    # import pandas as pd

    from scipy.stats import gmean
    from math import sqrt

    ## local imports

    #from functions.UpdateSymbols_inHDF5 import UpdateHDF5, loadQuotes_fromHDF
    # from functions.UpdateSymbols_inHDF5 import UpdateHDF_yf, loadQuotes_fromHDF

    print("*************************************************************")
    print("*************************************************************")
    print("***                                                       ***")
    print("***                                                       ***")
    print("***  daily montecarlo backtest                            ***")
    print("***  recent performance update                            ***")
    print("***                                                       ***")
    print("***                                                       ***")
    print("*************************************************************")
    print("*************************************************************")
    from time import sleep
    sleep(5)

    params = get_json_params(json_fn)
    trade_cost = params['trade_cost']
    monthsToHold = params['monthsToHold']
    numberStocksTraded = params['numberStocksTraded']
    LongPeriod = params['LongPeriod']
    stddevThreshold = params['stddevThreshold']
    MA1 = params['MA1']
    MA2 = params['MA2']
    MA3 = params['MA3']
    MA2offset = params['MA3'] - params['MA2']
    sma2factor = params['MA2factor']
    rankThresholdPct = params['rankThresholdPct']
    riskDownside_min = params['riskDownside_min']
    riskDownside_max = params['riskDownside_max']

    ##
    ##  Import list of symbols to process.
    ##
    params = get_json_params(json_fn)
    symbols_file = get_symbols_file(json_fn)
    stockList = params['stockList']

    # # read list of symbols from disk.
    # symbol_directory = os.path.join( os.getcwd(), "symbols" )
    # if stockList == 'Naz100':
    #     symbol_file = "Naz100_Symbols.txt"
    # elif stockList == 'SP500':
    #     symbol_file = "SP500_Symbols.txt"
    # symbols_file = os.path.join( symbol_directory, symbol_file )
    symbol_directory, symbol_file = os.path.split(symbols_file)

    # """
    # # read list of symbols from disk.
    # filename = os.path.join( os.getcwd(), 'symbols', 'Naz100_Symbols.txt' )
    # """

    ########################################################################
    ### clean up data for plotting
    ########################################################################

    import pdb
    #pdb.set_trace()

    print(" number of nans in value = ", value[np.isnan(value)].shape)
    print(" number of nans in monthvalue = ", monthvalue[np.isnan(monthvalue)].shape)

    print(" number of infs in value = ", value[np.isinf(value)].shape)
    print(" number of infs in monthvalue = ", monthvalue[np.isinf(monthvalue)].shape)

    '''
    value[value==0.] = np.nan
    value[np.isinf(value)] = np.nan
    monthvalue[monthvalue==0.] = np.nan
    monthvalue[np.isinf(monthvalue)] = np.nan
    '''

    ########################################################################
    ### normalize to starting value of $10,000
    ########################################################################

    # '''
    # for ii in range( MonteCarloPortfolioValues.shape[0] ):

    #     #print "ii, ",ii,
    #     aa = MonteCarloPortfolioValues[ii,:].copy()
    #     aa[aa==0.] = np.nan
    #     MonteCarloPortfolioValues[ii,:] = cleantobeginning( aa )
    #     Factor = 10000. / MonteCarloPortfolioValues[ii,0]
    #     #print Factor,
    #     MonteCarloPortfolioValues[ii,:] *= Factor
    #     FinalTradedPortfolioValue[ii] *= Factor
    # '''

    for ii in range( value.shape[0] ):

        #print "ii, symbols[ii], value[ii,0] = ",ii,symbols[ii], value[ii,0],
        #value[ii,:] = interpolate( value[ii,:] )
        if value[ii,0] == 0.:
            aa = value[ii,:].copy()
            aa[aa==0.] = np.nan
            value[ii,:] = cleantobeginning( aa )
        #value[ii,:] = cleantobeginning( value[ii,:] )
        value[ii,:] = cleantoend( value[ii,:] )
        Factor = 10000. / value[ii,0]
        #print 'Factor =', Factor
        value[ii,:] *= Factor

        # '''
        # monthvalue[ii,:] = interpolate( monthvalue[ii,:] )
        # monthvalue[ii,:] = cleantobeginning( monthvalue[ii,:] )
        # monthvalue[ii,:] = cleantoend( monthvalue[ii,:] )
        # '''
    Factor = 10000. / np.average( monthvalue[:,0] )
    #print Factor
    monthvalue *= Factor

    plotmax = 10. ** np.around( np.log10( value.max() ))
    MonteCarloPortfolioValues_NoNaNs = MonteCarloPortfolioValues[~np.isnan(MonteCarloPortfolioValues)]
    plotmax = np.around( 1.2 * MonteCarloPortfolioValues_NoNaNs.max(), decimals=-2 )
    print("\n\n ... plotmax = "+ str(plotmax))

    ########################################################################
    ### gather statistics for recent performance
    ########################################################################

    numberStocksUpTrendingMedian = np.median(numberStocksUpTrending,axis=0)
    numberStocksUpTrendingMean   = np.mean(numberStocksUpTrending,axis=0)

    index = monthvalue.shape[1]-1
    minindex = -np.min([756,len(datearray)-1])
    print("index = ", index, monthvalue.shape, len(datearray),minindex)

    PortfolioValue = np.nanmean(monthvalue,axis=0)
    PortfolioValue = np.mean(monthvalue,axis=0)
    PortfolioDailyGains = PortfolioValue[1:] / PortfolioValue[:-1]
    SharpeLife = ( gmean(PortfolioDailyGains[-index:])**252 -1. ) / ( np.std(PortfolioDailyGains[-index:])*sqrt(252) )
    Sharpe3Yr = ( gmean(PortfolioDailyGains[minindex:])**252 -1. ) / ( np.std(PortfolioDailyGains[minindex:])*sqrt(252) )
    Sharpe1Yr = ( gmean(PortfolioDailyGains[-252:])**252 -1. ) / ( np.std(PortfolioDailyGains[-252:])*sqrt(252) )
    Sharpe6Mo = ( gmean(PortfolioDailyGains[-126:])**252 -1. ) / ( np.std(PortfolioDailyGains[-126:])*sqrt(252) )
    Sharpe3Mo = ( gmean(PortfolioDailyGains[-63:])**252 -1. ) / ( np.std(PortfolioDailyGains[-63:])*sqrt(252) )
    Sharpe1Mo = ( gmean(PortfolioDailyGains[-21:])**252 -1. ) / ( np.std(PortfolioDailyGains[-21:])*sqrt(252) )
    PortfolioSharpe = ( gmean(PortfolioDailyGains)**252 -1. ) / ( np.std(PortfolioDailyGains)*sqrt(252) )

    print("Lifetime : ",index,PortfolioValue[-1], PortfolioValue[-index],datearray[-index])

    ReturnLife = (PortfolioValue[-1] / PortfolioValue[-index])**(252./index)
    Return3Yr = (PortfolioValue[-1] / PortfolioValue[minindex])**(1/3.)
    Return1Yr = (PortfolioValue[-1] / PortfolioValue[-252])
    Return6Mo = (PortfolioValue[-1] / PortfolioValue[-126])**(1/.5)
    Return3Mo = (PortfolioValue[-1] / PortfolioValue[-63])**(1/.25)
    Return1Mo = (PortfolioValue[-1] / PortfolioValue[-21])**(1/(1./12.))
    PortfolioReturn = gmean(PortfolioDailyGains)**252 -1.

    MaxPortfolioValue = np.zeros( PortfolioValue.shape[0], 'float' )
    for jj in range(PortfolioValue.shape[0]):
        MaxPortfolioValue[jj] = max(MaxPortfolioValue[jj-1],PortfolioValue[jj])
    PortfolioDrawdown = PortfolioValue / MaxPortfolioValue - 1.
    DrawdownLife = np.mean(PortfolioDrawdown[-index:])
    Drawdown3Yr = np.mean(PortfolioDrawdown[minindex:])
    Drawdown1Yr = np.mean(PortfolioDrawdown[-252:])
    Drawdown6Mo = np.mean(PortfolioDrawdown[-126:])
    Drawdown3Mo = np.mean(PortfolioDrawdown[-63:])
    Drawdown1Mo = np.mean(PortfolioDrawdown[-21:])

    #BuyHoldPortfolioValue = np.nanmean(value,axis=0)   # now in function call
    BuyHoldDailyGains = BuyHoldPortfolioValue[1:] / BuyHoldPortfolioValue[:-1]
    BuyHoldSharpeLife = ( gmean(BuyHoldDailyGains[-index:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-index:])*sqrt(252) )
    BuyHoldSharpe3Yr = ( gmean(BuyHoldDailyGains[minindex:])**252 -1. ) / ( np.std(BuyHoldDailyGains[minindex:])*sqrt(252) )
    BuyHoldSharpe1Yr  = ( gmean(BuyHoldDailyGains[-252:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-252:])*sqrt(252) )
    BuyHoldSharpe6Mo  = ( gmean(BuyHoldDailyGains[-126:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-126:])*sqrt(252) )
    BuyHoldSharpe3Mo  = ( gmean(BuyHoldDailyGains[-63:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-63:])*sqrt(252) )
    BuyHoldSharpe1Mo  = ( gmean(BuyHoldDailyGains[-21:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-21:])*sqrt(252) )
    BuyHoldReturnLife = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-index])**(252./index)
    BuyHoldReturn3Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[minindex:])**(1/13.)
    BuyHoldReturn1Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-252])
    BuyHoldReturn6Mo = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-126])**(1/.5)
    BuyHoldReturn3Mo = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-63])**(1/.25)
    BuyHoldReturn1Mo = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-21])**(1/(1./12.))
    MaxBuyHoldPortfolioValue = np.zeros( BuyHoldPortfolioValue.shape, 'float' )
    for jj in range(BuyHoldPortfolioValue.shape[0]):
        MaxBuyHoldPortfolioValue[jj] = max(MaxBuyHoldPortfolioValue[jj-1],BuyHoldPortfolioValue[jj])

    BuyHoldPortfolioDrawdown = BuyHoldPortfolioValue / MaxBuyHoldPortfolioValue - 1.
    BuyHoldDrawdownLife = np.mean(BuyHoldPortfolioDrawdown[-index:])
    BuyHoldDrawdown3Yr = np.mean(BuyHoldPortfolioDrawdown[minindex:])
    BuyHoldDrawdown1Yr = np.mean(BuyHoldPortfolioDrawdown[-252:])
    BuyHoldDrawdown6Mo = np.mean(BuyHoldPortfolioDrawdown[-126:])
    BuyHoldDrawdown3Mo = np.mean(BuyHoldPortfolioDrawdown[-63:])
    BuyHoldDrawdown1Mo = np.mean(BuyHoldPortfolioDrawdown[-21:])

    ########################################################################
    ### plot recent results
    ########################################################################

    plt.rcParams['figure.edgecolor'] = 'grey'
    plt.rc('savefig',edgecolor = 'grey')
    fig = plt.figure(1)
    plt.clf()
    subplotsize = gridspec.GridSpec(2,1,height_ratios=[5,1])
    plt.subplot(subplotsize[0])
    plt.grid()
    ##
    ## make plot of all stocks' individual prices
    ##
    #yscale('log')
    #ylim([1000,max(100000,plotmax)])
    plt.ylim([7000,plotmax])
    #ymin, ymax = emath.log10(1e3), emath.log10(max(10000,plotmax))
    ymin, ymax = 7000,plotmax
    bin_width = (np.log10(ymax) - np.log10(ymin)) / 50
    y_bins = np.arange(np.log10(ymin), np.log10(ymax)+.0000001, bin_width)
    AllStocksHistogram = np.ones((len(y_bins)-1, len(datearray),3))
    HH = np.zeros((len(y_bins)-1, len(datearray)))
    mm = np.zeros(len(datearray))
    xlocs = []
    xlabels = []
    for i in range(1,len(datearray)):
        ValueOnDate = np.log10(value[:,i])
        '''
        if ValueOnDate[ValueOnDate == 10000].shape[0] > 1:
            ValueOnDate[ValueOnDate == 10000] = 0.
            ValueOnDate[np.argmin(ValueOnDate)] = 10000.
        '''
        #h, _ = np.histogram(np.log10(ValueOnDate), bins=y_bins, density=True)
        h, _ = np.histogram(ValueOnDate, bins=y_bins, density=True)
        # reverse so big numbers become small(and print out black)
        h = 1. - h
        # set range to [.5,1.]
        h /= 2.
        h += .5
        #print "idatearray[i],h min,mean,max = ", h.min(),h.mean(),h.max()
        HH[:,i] = h
        mm[i] = np.median(value[-1,:])
        if datearray[i].year != datearray[i-1].year:
            # print(" inside histogram evaluation for date = ", datearray[i])
            xlocs.append(i)
            xlabels.append(str(datearray[i].year))
    HH -= np.percentile(HH.flatten(),2)
    HH /= HH.max()
    HH = np.clip( HH, 0., 1. )
    #print "HH min,mean,max = ", HH.min(),HH.mean(),HH.max()
    AllStocksHistogram[:,:,2] = HH
    AllStocksHistogram[:,:,1] = AllStocksHistogram[:,:,2]
    AllStocksHistogram = np.clip(AllStocksHistogram,AllStocksHistogram.max()*.05,1)
    AllStocksHistogram /= AllStocksHistogram.max()

    #plt.imshow(AllStocksHistogram, cmap='Reds_r', aspect='auto', origin='lower', extent=(0, len(datearray), 10**ymin, 10**ymax))
    plt.imshow(AllStocksHistogram, cmap='Reds_r', aspect='auto', origin='lower', extent=(0, len(datearray), ymin, ymax))
    plt.grid()
    ##
    ## cumulate final values for grayscale histogram overlay
    ##
    # '''
    # if iter == 0:
    #     MonteCarloPortfolioValues = np.zeros( (randomtrials, len(datearray) ), float )
    # MonteCarloPortfolioValues[iter,:] = np.average(monthvalue,axis=0)
    # '''

    ##
    ## cumulate final values for grayscale histogram overlay
    ##

    #yscale('log')
    #ylim([1000,max(10000,plotmax)])
    plt.ylim([7000,plotmax])
    #ymin, ymax = emath.log10(1e3), emath.log10(max(10000,plotmax))
    ymin, ymax = 7000,plotmax
    bin_width = (np.log10(ymax) - np.log10(ymin)) / 50
    y_bins = np.arange(np.log10(ymin), np.log10(ymax)+.0000001, bin_width)
    H = np.zeros((len(y_bins)-1, len(datearray)))
    m = np.zeros(len(datearray))
    hb = np.zeros((len(y_bins)-1, len(datearray),3))

    ## TODO: remove the following lines after QC and de-bugging
    print("\n\n ... inside dailyBacktest_pctLong.py near line 338")
    print(" ... MonteCarloPortfolioValues[:,:].min() = "+str(MonteCarloPortfolioValues[:,:].min()))
    print(" ... MonteCarloPortfolioValues[:,:].max() = "+str(MonteCarloPortfolioValues[:,:].max()))
    print(" ... MonteCarloPortfolioValues[0,:] = "+str(MonteCarloPortfolioValues[0,:]))
    print(" ... y_bins[0], y_bins[-1] = "+str((y_bins[0], y_bins[-1])))

    for i in range(1,len(datearray)):
        #h, _ = np.histogram(np.log10(MonteCarloPortfolioValues[:,i]), bins=y_bins, density=True)
        h, _ = np.histogram(np.log10(MonteCarloPortfolioValues[:,i]), bins=y_bins, density=True)
        # reverse so big numbers become small(and print out black)
        h = 1. - h
        # set range to [.5,1.]
        h = np.clip( h, .05, 1. )
        h /= 2.
        h += .5
        H[:,i] = h
        m[i] = np.median(value[-1,:])
        # if datearray[i].year != datearray[i-1].year:
        #     print(" inside histogram evaluation for date = ", datearray[i])
    H -= np.percentile(H.flatten(),2)
    H /= H.max()
    H = np.clip( H, 0., 1. )
    hb[:,:,0] = H
    hb[:,:,1] = H
    hb[:,:,2] = H
    hb = .5 * AllStocksHistogram + .5 * hb

    #yscale('log')
    #plt.imshow(hb, cmap='gray', aspect='auto', origin='lower', extent=(0, len(datearray), 10**ymin, 10**ymax))
    plt.imshow(hb, cmap='gray', aspect='auto', origin='lower', extent=(0, len(datearray), ymin, ymax))

    #yscale('log')
    plt.yscale('log')   ## TODO: check this
    plt.plot( np.average(monthvalue,axis=0), lw=3, c='k' )
    plt.grid()
    plt.draw()

    ##
    ## continue
    ##
    #FinalTradedPortfolioValue[iter] = np.average(monthvalue[:,-1])
    fFinalTradedPortfolioValue = "{:,}".format(int(FinalTradedPortfolioValue[-1]))
    PortfolioDailyGains = np.average(monthvalue,axis=0)[1:] / np.average(monthvalue,axis=0)[:-1]
    PortfolioSharpe = ( gmean(PortfolioDailyGains)**252 -1. ) / ( np.std(PortfolioDailyGains)*sqrt(252) )
    fPortfolioSharpe = format(PortfolioSharpe,'5.2f')

    print("")
    print(" value 3 months yrs ago, 1 month ago, last = ",np.average(monthvalue[:,-63]),np.average(monthvalue[:,-21]),np.average(monthvalue[:,-1]))
    print(" one month gain, daily geom mean, stdev = ",np.average(monthvalue,axis=0)[-1] / np.average(monthvalue,axis=0)[-21],gmean(PortfolioDailyGains[-21:])**252 -1.,np.std(PortfolioDailyGains[-252:])*sqrt(252))
    print(" three month gain, daily geom mean, stdev = ",np.average(monthvalue,axis=0)[-1] / np.average(monthvalue,axis=0)[-63],gmean(PortfolioDailyGains[-63:])**252 -1.,np.std(PortfolioDailyGains[-504:])*sqrt(252))

    title_text = str(0)+":  "+ \
                  str(int(numberStocksTraded))+"__"+   \
                  str(int(monthsToHold))+"__"+   \
                  str(int(LongPeriod))+"-"+   \
                  format(stddevThreshold,'4.2f')+"-"+   \
                  str(int(MA1))+"-"+   \
                  str(int(MA2))+"-"+   \
                  str(int(MA2+MA2offset))+"-"+   \
                  format(sma2factor,'5.3f')+"_"+   \
                  format(rankThresholdPct,'.1%')+"__"+   \
                  format(riskDownside_min,'6.3f')+"-"+  \
                  format(riskDownside_max,'6.3f')+"__"+   \
                  fFinalTradedPortfolioValue+'__'+   \
                  fPortfolioSharpe

    plt.title( title_text, fontsize = 9 )
    fSharpeLife = format(SharpeLife,'5.2f')
    fSharpe3Yr = format(Sharpe3Yr,'5.2f')
    fSharpe1Yr = format(Sharpe1Yr,'5.2f')
    fSharpe6Mo = format(Sharpe6Mo,'5.2f')
    fSharpe3Mo = format(Sharpe3Mo,'5.2f')
    fSharpe1Mo = format(Sharpe1Mo,'5.2f')
    fReturnLife = format(ReturnLife,'5.2f')
    fReturn3Yr = format(Return3Yr,'5.2f')
    fReturn1Yr = format(Return1Yr,'5.2f')
    fReturn6Mo = format(Return6Mo,'5.2f')
    fReturn3Mo = format(Return3Mo,'5.2f')
    fReturn1Mo = format(Return1Mo,'5.2f')
    fDrawdownLife = format(DrawdownLife,'.1%')
    fDrawdown3Yr = format(Drawdown3Yr,'.1%')
    fDrawdown1Yr = format(Drawdown1Yr,'.1%')
    fDrawdown6Mo = format(Drawdown6Mo,'.1%')
    fDrawdown3Mo = format(Drawdown3Mo,'.1%')
    fDrawdown1Mo = format(Drawdown1Mo,'.1%')
    print(" one year sharpe = ",fSharpe1Mo)
    print("")
    #plotrange = log10(plotmax / 1000.)
    plotrange = (plotmax -7000.)
    plt.text( 50, 7050, symbols_file, fontsize=8 )
    plt.text( 50, 8000, "Backtested on "+datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"), fontsize=7.5 )
    # '''
    # text(50,1000*10**(.95*plotrange),'Period Sharpe AvgProfit  Avg DD',fontsize=7.5)
    # text(50,1000*10**(.91*plotrange),' Life  '+fSharpeLife+'  '+fReturnLife+'  '+fDrawdownLife,fontsize=8)
    # text(50,1000*10**(.87*plotrange),' 3 Yr  '+fSharpe3Yr+'  '+fReturn3Yr+'  '+fDrawdown3Yr,fontsize=8)
    # text(50,1000*10**(.83*plotrange),' 1 Yr  '+fSharpe1Yr+'  '+fReturn1Yr+'  '+fDrawdown1Yr,fontsize=8)
    # text(50,1000*10**(.79*plotrange),' 6 Mo  '+fSharpe6Mo+'  '+fReturn6Mo+'  '+fDrawdown6Mo,fontsize=8)
    # text(50,1000*10**(.75*plotrange),' 3 Mo  '+fSharpe3Mo+'  '+fReturn3Mo+'  '+fDrawdown3Mo,fontsize=8)
    # text(50,1000*10**(.71*plotrange),' 1 Mo  '+fSharpe1Mo+'  '+fReturn1Mo+'  '+fDrawdown1Mo,fontsize=8)

    # text(50,1000*10**(.54*plotrange),last_symbols_text,fontsize=8)
    # '''
    plt.text(50,7000+(.95*plotrange),'Period Sharpe AvgProfit  Avg DD',fontsize=7.5)
    plt.text(50,7000+(.91*plotrange),' Life  '+fSharpeLife+'  '+fReturnLife+'  '+fDrawdownLife,fontsize=8)
    plt.text(50,7000+(.87*plotrange),' 3 Yr  '+fSharpe3Yr+'  '+fReturn3Yr+'  '+fDrawdown3Yr,fontsize=8)
    plt.text(50,7000+(.83*plotrange),' 1 Yr  '+fSharpe1Yr+'  '+fReturn1Yr+'  '+fDrawdown1Yr,fontsize=8)
    plt.text(50,7000+(.79*plotrange),' 6 Mo  '+fSharpe6Mo+'  '+fReturn6Mo+'  '+fDrawdown6Mo,fontsize=8)
    plt.text(50,7000+(.75*plotrange),' 3 Mo  '+fSharpe3Mo+'  '+fReturn3Mo+'  '+fDrawdown3Mo,fontsize=8)
    plt.text(50,7000+(.71*plotrange),' 1 Mo  '+fSharpe1Mo+'  '+fReturn1Mo+'  '+fDrawdown1Mo,fontsize=8)

    plt.text(50,7000+(.54*plotrange),last_symbols_text,fontsize=8)
    plt.plot(BuyHoldPortfolioValue,lw=3,c='r')
    plt.plot(np.average(monthvalue,axis=0),lw=4,c='k')
    # set up to use dates for labels
    xlocs = []
    xlabels = []
    for i in range(1,len(datearray)):
        if datearray[i].year != datearray[i-1].year:
            xlocs.append(i)
            xlabels.append(str(datearray[i].year))
    print("xlocs,xlabels = ", xlocs, xlabels)
    if len(xlocs) < 12 :
        plt.xticks(xlocs, xlabels)
    else:
        plt.xticks(xlocs[::2], xlabels[::2])
    plt.xlim(0,len(datearray))

    for ii in range( MonteCarloPortfolioValues.shape[0] ):
        plt.plot( datearray, MonteCarloPortfolioValues[ii,:], c=(.1,.1,.1), lw=.1 )
    for ii in range( value.shape[0] ):
        plt.plot( datearray, value[ii,:], c=(.1,0.,0.), lw=.1 )

    # '''
    # import pdb
    # pdb.set_trace()
    # '''

    plt.subplot(subplotsize[1])
    plt.grid()
    ##ylim(0, value.shape[0])
    plt.ylim(0, 1.2)
    plt.plot(datearray,numberStocksUpTrendingMedian / activeCount,'g-',lw=1)
    plt.plot(datearray,numberStocksUpTrendingNearHigh / activeCount,'b-',lw=1)
    plt.plot(datearray,numberStocksUpTrendingBeatBuyHold / activeCount,'k-',lw=2)
    plt.plot(datearray,numberStocks  / activeCount,'r-')

    # '''
    # for i,idate in enumerate(datearray):
    #     print "i,datearray[i],activeCount[i] = ",i,idate,activeCount[i]
    # '''

    plt.draw()
    # save figure to disk
    # json_dir = os.path.split(json_fn)[0]
    # outputplotname = os.path.join( json_dir, 'pyTAAA_web', 'PyTAAA_monteCarloBacktestRecent.png' )
    webpage_dir = get_webpage_store(json_fn)
    png_fn = 'PyTAAA_monteCarloBacktestRecent'
    outputplotname = os.path.join(webpage_dir, png_fn+'.png' )
    plt.savefig(outputplotname, format='png', edgecolor='gray' )

    return



# '''
# def plotRecentPerfomance( datearray, symbols, value, monthvalue,
#                           ValueOnDate, AllStocksHistogram,
#                           MonteCarloPortfolioValues,
#                           FinalTradedPortfolioValue,
#                           numberStocksUpTrending,
#                           last_symbols_text,
#                           activeCount,
#                           numberStocks,
#                           numberStocksUpTrendingNearHigh,
#                           numberStocksUpTrendingBeatBuyHold ):
# '''
def plotRecentPerfomance(
        datearray, symbols, value, monthvalue,
        AllStocksHistogram,
        MonteCarloPortfolioValues,
        FinalTradedPortfolioValue,
        numberStocksUpTrending,
        last_symbols_text,
        activeCount,
        numberStocks,
        numberStocksUpTrendingNearHigh,
        numberStocksUpTrendingBeatBuyHold,
        json_fn
):

    import time, threading

    import numpy as np
    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec
    import os

    import datetime
    from numpy import random
    from scipy import ndimage
    from random import choice
    from scipy.stats import rankdata


    import pandas as pd

    from scipy.stats import gmean

    ## local imports

    # from functions.UpdateSymbols_inHDF5 import UpdateHDF5, loadQuotes_fromHDF
    # from functions.UpdateSymbols_inHDF5 import UpdateHDF_yf, loadQuotes_fromHDF

    print("*************************************************************")
    print("*************************************************************")
    print("***                                                       ***")
    print("***                                                       ***")
    print("***  daily montecarlo backtest                            ***")
    print("***  recent performance update                            ***")
    print("***                                                       ***")
    print("***                                                       ***")
    print("*************************************************************")
    print("*************************************************************")
    from time import sleep
    sleep(5)

    params = get_json_params(json_fn)
    json_dir = os.path.split(json_fn)[0]

    trade_cost = params['trade_cost']
    monthsToHold = params['monthsToHold']
    numberStocksTraded = params['numberStocksTraded']
    LongPeriod = params['LongPeriod']
    stddevThreshold = params['stddevThreshold']
    MA1 = params['MA1']
    MA2 = params['MA2']
    MA3 = params['MA3']
    MA2offset = params['MA3'] - params['MA2']
    sma2factor = params['MA2factor']
    rankThresholdPct = params['rankThresholdPct']
    riskDownside_min = params['riskDownside_min']
    riskDownside_max = params['riskDownside_max']

    ##
    ##  Import list of symbols to process.
    ##
    params = get_json_params(json_fn)
    symbols_file = get_symbols_file(json_fn)

    stockList = params['stockList']

    # read list of symbols from disk.
    # symbol_directory = os.path.join( os.getcwd(), "symbols" )
    # if stockList == 'Naz100':
    #     symbol_file = "Naz100_Symbols.txt"
    # elif stockList == 'SP500':
    #     symbol_file = "SP500_Symbols.txt"
    # symbols_file = os.path.join( symbol_directory, symbol_file )
    symbol_direcotyr, symbol_file = os.path.split(symbols_file)

    # """
    # # read list of symbols from disk.
    # filename = os.path.join( os.getcwd(), 'symbols', 'Naz100_Symbols.txt' )
    # """

    ########################################################################
    ### clean up data for plotting
    ########################################################################

    import pdb
    #pdb.set_trace()

    print(" number of nans in value = ", value[np.isnan(value)].shape)
    print(" number of nans in monthvalue = ", monthvalue[np.isnan(monthvalue)].shape)

    print(" number of infs in value = ", value[np.isinf(value)].shape)
    print(" number of infs in monthvalue = ", monthvalue[np.isinf(monthvalue)].shape)

    '''
    value[value==0.] = np.nan
    value[np.isinf(value)] = np.nan
    monthvalue[monthvalue==0.] = np.nan
    monthvalue[np.isinf(monthvalue)] = np.nan
    '''

    ########################################################################
    ### normalize to starting value of $10,000
    ########################################################################

    for ii in range( MonteCarloPortfolioValues.shape[0] ):

        #print "ii, ",ii,
        aa = MonteCarloPortfolioValues[ii,:].copy()
        aa[aa==0.] = np.nan
        MonteCarloPortfolioValues[ii,:] = cleantobeginning( aa )
        Factor = 10000. / MonteCarloPortfolioValues[ii,0]
        #print Factor,
        MonteCarloPortfolioValues[ii,:] *= Factor
        FinalTradedPortfolioValue[ii] *= Factor

    for ii in range( value.shape[0] ):

        #print "ii, symbols[ii], value[ii,0] = ",ii,symbols[ii], value[ii,0],
        #value[ii,:] = interpolate( value[ii,:] )
        if value[ii,0] == 0.:
            aa = value[ii,:].copy()
            aa[aa==0.] = np.nan
            value[ii,:] = cleantobeginning( aa )
        #value[ii,:] = cleantobeginning( value[ii,:] )
        value[ii,:] = cleantoend( value[ii,:] )
        Factor = 10000. / value[ii,0]
        #print 'Factor =', Factor
        value[ii,:] *= Factor

        monthvalue[ii,:] = interpolate( monthvalue[ii,:] )
        monthvalue[ii,:] = cleantobeginning( monthvalue[ii,:] )
        monthvalue[ii,:] = cleantoend( monthvalue[ii,:] )
    Factor = 10000. / np.average( monthvalue[:,0] )
    #print Factor
    monthvalue *= Factor

    plotmax = 10. ** np.around( np.log10( value.max() ))
    MonteCarloPortfolioValues_NoNaNs = MonteCarloPortfolioValues[~np.isnan(MonteCarloPortfolioValues)]
    plotmax = np.around( 1.2 * MonteCarloPortfolioValues_NoNaNs.max(), decimals=-2 )
    #print "\n\n ... plotmax = ", plotmax

    ########################################################################
    ### gather statistics for recent performance
    ########################################################################

    numberStocksUpTrendingMedian = np.median(numberStocksUpTrending,axis=0)
    numberStocksUpTrendingMean   = np.mean(numberStocksUpTrending,axis=0)

    index = monthvalue.shape[1]-1
    minindex = -np.min([756,len(datearray)-1])
    print("index = ", index, monthvalue.shape, len(datearray),minindex)

    PortfolioValue = np.nanmean(monthvalue,axis=0)
    PortfolioDailyGains = PortfolioValue[1:] / PortfolioValue[:-1]
    SharpeLife = ( gmean(PortfolioDailyGains[-index:])**252 -1. ) / ( np.std(PortfolioDailyGains[-index:])*sqrt(252) )
    Sharpe3Yr = ( gmean(PortfolioDailyGains[minindex:])**252 -1. ) / ( np.std(PortfolioDailyGains[minindex:])*sqrt(252) )
    Sharpe1Yr = ( gmean(PortfolioDailyGains[-252:])**252 -1. ) / ( np.std(PortfolioDailyGains[-252:])*sqrt(252) )
    Sharpe6Mo = ( gmean(PortfolioDailyGains[-126:])**252 -1. ) / ( np.std(PortfolioDailyGains[-126:])*sqrt(252) )
    Sharpe3Mo = ( gmean(PortfolioDailyGains[-63:])**252 -1. ) / ( np.std(PortfolioDailyGains[-63:])*sqrt(252) )
    Sharpe1Mo = ( gmean(PortfolioDailyGains[-21:])**252 -1. ) / ( np.std(PortfolioDailyGains[-21:])*sqrt(252) )
    PortfolioSharpe = ( gmean(PortfolioDailyGains)**252 -1. ) / ( np.std(PortfolioDailyGains)*sqrt(252) )

    print("Lifetime : ",index,PortfolioValue[-1], PortfolioValue[-index],datearray[-index])

    ReturnLife = (PortfolioValue[-1] / PortfolioValue[-index])**(252./index)
    Return3Yr = (PortfolioValue[-1] / PortfolioValue[minindex])**(1/3.)
    Return1Yr = (PortfolioValue[-1] / PortfolioValue[-252])
    Return6Mo = (PortfolioValue[-1] / PortfolioValue[-126])**(1/.5)
    Return3Mo = (PortfolioValue[-1] / PortfolioValue[-63])**(1/.25)
    Return1Mo = (PortfolioValue[-1] / PortfolioValue[-21])**(1/(1./12.))
    PortfolioReturn = gmean(PortfolioDailyGains)**252 -1.

    MaxPortfolioValue = np.zeros( PortfolioValue.shape[0], 'float' )
    for jj in range(PortfolioValue.shape[0]):
        MaxPortfolioValue[jj] = max(MaxPortfolioValue[jj-1],PortfolioValue[jj])
    PortfolioDrawdown = PortfolioValue / MaxPortfolioValue - 1.
    DrawdownLife = np.mean(PortfolioDrawdown[-index:])
    Drawdown3Yr = np.mean(PortfolioDrawdown[minindex:])
    Drawdown1Yr = np.mean(PortfolioDrawdown[-252:])
    Drawdown6Mo = np.mean(PortfolioDrawdown[-126:])
    Drawdown3Mo = np.mean(PortfolioDrawdown[-63:])
    Drawdown1Mo = np.mean(PortfolioDrawdown[-21:])

    BuyHoldPortfolioValue = np.nanmean(value,axis=0)
    BuyHoldDailyGains = BuyHoldPortfolioValue[1:] / BuyHoldPortfolioValue[:-1]
    BuyHoldSharpeLife = ( gmean(BuyHoldDailyGains[-index:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-index:])*sqrt(252) )
    BuyHoldSharpe3Yr = ( gmean(BuyHoldDailyGains[minindex:])**252 -1. ) / ( np.std(BuyHoldDailyGains[minindex:])*sqrt(252) )
    BuyHoldSharpe1Yr  = ( gmean(BuyHoldDailyGains[-252:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-252:])*sqrt(252) )
    BuyHoldSharpe6Mo  = ( gmean(BuyHoldDailyGains[-126:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-126:])*sqrt(252) )
    BuyHoldSharpe3Mo  = ( gmean(BuyHoldDailyGains[-63:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-63:])*sqrt(252) )
    BuyHoldSharpe1Mo  = ( gmean(BuyHoldDailyGains[-21:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-21:])*sqrt(252) )
    BuyHoldReturnLife = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-index])**(252./index)
    BuyHoldReturn3Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[minindex:])**(1/13.)
    BuyHoldReturn1Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-252])
    BuyHoldReturn6Mo = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-126])**(1/.5)
    BuyHoldReturn3Mo = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-63])**(1/.25)
    BuyHoldReturn1Mo = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-21])**(1/(1./12.))
    MaxBuyHoldPortfolioValue = np.zeros( BuyHoldPortfolioValue.shape, 'float' )
    for jj in range(BuyHoldPortfolioValue.shape[0]):
        MaxBuyHoldPortfolioValue[jj] = max(MaxBuyHoldPortfolioValue[jj-1],BuyHoldPortfolioValue[jj])

    BuyHoldPortfolioDrawdown = BuyHoldPortfolioValue / MaxBuyHoldPortfolioValue - 1.
    BuyHoldDrawdownLife = np.mean(BuyHoldPortfolioDrawdown[-index:])
    BuyHoldDrawdown3Yr = np.mean(BuyHoldPortfolioDrawdown[minindex:])
    BuyHoldDrawdown1Yr = np.mean(BuyHoldPortfolioDrawdown[-252:])
    BuyHoldDrawdown6Mo = np.mean(BuyHoldPortfolioDrawdown[-126:])
    BuyHoldDrawdown3Mo = np.mean(BuyHoldPortfolioDrawdown[-63:])
    BuyHoldDrawdown1Mo = np.mean(BuyHoldPortfolioDrawdown[-21:])

    ########################################################################
    ### plot recent results
    ########################################################################

    plt.rcParams['figure.edgecolor'] = 'grey'
    plt.rc('savefig',edgecolor = 'grey')
    fig = plt.figure(1)
    plt.clf()
    subplotsize = gridspec.GridSpec(2,1,height_ratios=[5,1])
    plt.subplot(subplotsize[0])
    plt.grid()
    ##
    ## make plot of all stocks' individual prices
    ##
    #yscale('log')
    #ylim([1000,max(100000,plotmax)])
    plt.ylim([7000,plotmax])
    #ymin, ymax = emath.log10(1e3), emath.log10(max(10000,plotmax))
    ymin, ymax = 7000,plotmax
    n_bins = 150
    #bin_width = (ymax - ymin) / n_bins
    #y_bins = np.arange(ymin, ymax+.0000001, bin_width)
    y_bins = np.linspace(ymin, ymax, n_bins)
    AllStocksHistogram = np.ones((len(y_bins)-1, len(datearray),3))
    HH = np.zeros((len(y_bins)-1, len(datearray)))
    mm = np.zeros(len(datearray))
    xlocs = []
    xlabels = []
    for i in range(1,len(datearray)):
        ValueOnDate = value[:,i]
        '''
        if ValueOnDate[ValueOnDate == 10000].shape[0] > 1:
            ValueOnDate[ValueOnDate == 10000] = 0.
            ValueOnDate[np.argmin(ValueOnDate)] = 10000.
        '''
        #h, _ = np.histogram(np.log10(ValueOnDate), bins=y_bins, density=True)
        h, _ = np.histogram(ValueOnDate, bins=y_bins, density=True)
        h /= h.sum()
        # reverse so big numbers become small(and print out black)
        h = 1. - h
        # set range to [.5,1.]
        h /= 2.
        h += .5
        #print "idatearray[i],h min,mean,max = ", h.min(),h.mean(),h.max()
        HH[:,i] = h
        mm[i] = np.median(value[-1,:])
        if datearray[i].year != datearray[i-1].year:
            # print(" inside histogram evaluation for date = ", datearray[i])
            xlocs.append(i)
            xlabels.append(str(datearray[i].year))
    HH -= np.percentile(HH.flatten(),2)
    HH /= HH.max()
    HH = np.clip( HH, 0., 1. )
    #print "HH min,mean,max = ", HH.min(),HH.mean(),HH.max()
    AllStocksHistogram[:,:,2] = HH
    AllStocksHistogram[:,:,1] = AllStocksHistogram[:,:,2]
    AllStocksHistogram = np.clip(AllStocksHistogram,AllStocksHistogram.max()*.05,1)
    AllStocksHistogram /= AllStocksHistogram.max()

    #plt.imshow(AllStocksHistogram, cmap='Reds_r', aspect='auto', origin='lower', extent=(0, len(datearray), 10**ymin, 10**ymax))
    plt.imshow(AllStocksHistogram, cmap='Reds_r', aspect='auto', origin='lower', extent=(0, len(datearray), ymin, ymax))
    plt.grid()
    ##
    ## cumulate final values for grayscale histogram overlay
    ##
    '''
    if iter == 0:
        MonteCarloPortfolioValues = np.zeros( (randomtrials, len(datearray) ), float )
    MonteCarloPortfolioValues[iter,:] = np.average(monthvalue,axis=0)
    '''

    ##
    ## cumulate final values for grayscale histogram overlay
    ##

    #yscale('log')
    #ylim([1000,max(10000,plotmax)])
    plt.ylim([7000,plotmax])
    #ymin, ymax = emath.log10(1e3), emath.log10(max(10000,plotmax))
    ymin, ymax = 7000,plotmax
    n_bins = 150
    #bin_width = (np.log10(ymax) - np.log10(ymin)) / n_bins
    #y_bins = np.arange(np.log10(ymin), np.log10(ymax)+.0000001, bin_width)
    y_bins = np.linspace(np.log10(ymin), np.log10(ymax), n_bins)
    H = np.zeros((len(y_bins)-1, len(datearray)))
    m = np.zeros(len(datearray))
    hb = np.zeros((len(y_bins)-1, len(datearray),3))
    for i in range(1,len(datearray)):
        #h, _ = np.histogram(np.log10(MonteCarloPortfolioValues[:,i]), bins=y_bins, density=True)
        h, _ = np.histogram(np.log10(MonteCarloPortfolioValues[:,i]), bins=y_bins, density=True)
        h /= h.sum()
        # reverse so big numbers become small(and print out black)
        h = 1. - h
        # set range to [.5,1.]
        h = np.clip( h, .05, 1. )
        h /= 2.
        h += .5
        H[:,i] = h
        m[i] = np.median(value[-1,:])
        # if datearray[i].year != datearray[i-1].year:
        #     print(" inside histogram evaluation for date = ", datearray[i])
    H -= np.percentile(H.flatten(),2)
    H /= H.max()
    H = np.clip( H, 0., 1. )
    hb[:,:,0] = H
    hb[:,:,1] = H
    hb[:,:,2] = H
    hb = .5 * AllStocksHistogram + .5 * hb

    #yscale('log')
    #plt.imshow(hb, cmap='gray', aspect='auto', origin='lower', extent=(0, len(datearray), 10**ymin, 10**ymax))
    plt.imshow(hb, cmap='gray', aspect='auto', origin='lower', extent=(0, len(datearray), ymin, ymax))

    #yscale('log')
    plt.plot( np.average(monthvalue,axis=0), lw=3, c='k' )
    plt.grid()
    plt.draw()

    ##
    ## continue
    ##
    #FinalTradedPortfolioValue[iter] = np.average(monthvalue[:,-1])
    fFinalTradedPortfolioValue = "{:,}".format(int(FinalTradedPortfolioValue[-1]))
    PortfolioDailyGains = np.average(monthvalue,axis=0)[1:] / np.average(monthvalue,axis=0)[:-1]
    PortfolioSharpe = ( gmean(PortfolioDailyGains)**252 -1. ) / ( np.std(PortfolioDailyGains)*sqrt(252) )
    fPortfolioSharpe = format(PortfolioSharpe,'5.2f')

    print("")
    print(" value 3 months yrs ago, 1 month ago, last = ",np.average(monthvalue[:,-63]),np.average(monthvalue[:,-21]),np.average(monthvalue[:,-1]))
    print(" one month gain, daily geom mean, stdev = ",np.average(monthvalue,axis=0)[-1] / np.average(monthvalue,axis=0)[-21],gmean(PortfolioDailyGains[-21:])**252 -1.,np.std(PortfolioDailyGains[-252:])*sqrt(252))
    print(" three month gain, daily geom mean, stdev = ",np.average(monthvalue,axis=0)[-1] / np.average(monthvalue,axis=0)[-63],gmean(PortfolioDailyGains[-63:])**252 -1.,np.std(PortfolioDailyGains[-504:])*sqrt(252))

    title_text = str(0)+":  "+ \
                  str(int(numberStocksTraded))+"__"+   \
                  str(int(monthsToHold))+"__"+   \
                  str(int(LongPeriod))+"-"+   \
                  format(stddevThreshold,'4.2f')+"-"+   \
                  str(int(MA1))+"-"+   \
                  str(int(MA2))+"-"+   \
                  str(int(MA2+MA2offset))+"-"+   \
                  format(sma2factor,'5.3f')+"_"+   \
                  format(rankThresholdPct,'.1%')+"__"+   \
                  format(riskDownside_min,'6.3f')+"-"+  \
                  format(riskDownside_max,'6.3f')+"__"+   \
                  fFinalTradedPortfolioValue+'__'+   \
                  fPortfolioSharpe

    plt.title( title_text, fontsize = 9 )
    fSharpeLife = format(SharpeLife,'5.2f')
    fSharpe3Yr = format(Sharpe3Yr,'5.2f')
    fSharpe1Yr = format(Sharpe1Yr,'5.2f')
    fSharpe6Mo = format(Sharpe6Mo,'5.2f')
    fSharpe3Mo = format(Sharpe3Mo,'5.2f')
    fSharpe1Mo = format(Sharpe1Mo,'5.2f')
    fReturnLife = format(ReturnLife,'5.2f')
    fReturn3Yr = format(Return3Yr,'5.2f')
    fReturn1Yr = format(Return1Yr,'5.2f')
    fReturn6Mo = format(Return6Mo,'5.2f')
    fReturn3Mo = format(Return3Mo,'5.2f')
    fReturn1Mo = format(Return1Mo,'5.2f')
    fDrawdownLife = format(DrawdownLife,'.1%')
    fDrawdown3Yr = format(Drawdown3Yr,'.1%')
    fDrawdown1Yr = format(Drawdown1Yr,'.1%')
    fDrawdown6Mo = format(Drawdown6Mo,'.1%')
    fDrawdown3Mo = format(Drawdown3Mo,'.1%')
    fDrawdown1Mo = format(Drawdown1Mo,'.1%')
    print(" one year sharpe = ",fSharpe1Mo)
    print("")
    #plotrange = log10(plotmax / 1000.)
    plotrange = (plotmax -7000.)
    plt.text( 50, 7050, symbols_file, fontsize=8 )
    plt.text( 50, 8000, "Backtested on "+datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"), fontsize=7.5 )
    # '''
    # text(50,1000*10**(.95*plotrange),'Period Sharpe AvgProfit  Avg DD',fontsize=7.5)
    # text(50,1000*10**(.91*plotrange),' Life  '+fSharpeLife+'  '+fReturnLife+'  '+fDrawdownLife,fontsize=8)
    # text(50,1000*10**(.87*plotrange),' 3 Yr  '+fSharpe3Yr+'  '+fReturn3Yr+'  '+fDrawdown3Yr,fontsize=8)
    # text(50,1000*10**(.83*plotrange),' 1 Yr  '+fSharpe1Yr+'  '+fReturn1Yr+'  '+fDrawdown1Yr,fontsize=8)
    # text(50,1000*10**(.79*plotrange),' 6 Mo  '+fSharpe6Mo+'  '+fReturn6Mo+'  '+fDrawdown6Mo,fontsize=8)
    # text(50,1000*10**(.75*plotrange),' 3 Mo  '+fSharpe3Mo+'  '+fReturn3Mo+'  '+fDrawdown3Mo,fontsize=8)
    # text(50,1000*10**(.71*plotrange),' 1 Mo  '+fSharpe1Mo+'  '+fReturn1Mo+'  '+fDrawdown1Mo,fontsize=8)

    # text(50,1000*10**(.54*plotrange),last_symbols_text,fontsize=8)
    # '''

    # '''
    # plt.text(50,7000+(.95*plotrange),'Period Sharpe AvgProfit  Avg DD',fontsize=7.5)
    # plt.text(50,7000+(.91*plotrange),' Life  '+fSharpeLife+'  '+fReturnLife+'  '+fDrawdownLife,fontsize=8)
    # plt.text(50,7000+(.87*plotrange),' 3 Yr  '+fSharpe3Yr+'  '+fReturn3Yr+'  '+fDrawdown3Yr,fontsize=8)
    # plt.text(50,7000+(.83*plotrange),' 1 Yr  '+fSharpe1Yr+'  '+fReturn1Yr+'  '+fDrawdown1Yr,fontsize=8)
    # plt.text(50,7000+(.79*plotrange),' 6 Mo  '+fSharpe6Mo+'  '+fReturn6Mo+'  '+fDrawdown6Mo,fontsize=8)
    # plt.text(50,7000+(.75*plotrange),' 3 Mo  '+fSharpe3Mo+'  '+fReturn3Mo+'  '+fDrawdown3Mo,fontsize=8)
    # plt.text(50,7000+(.71*plotrange),' 1 Mo  '+fSharpe1Mo+'  '+fReturn1Mo+'  '+fDrawdown1Mo,fontsize=8)

    # plt.text(50,7000+(.54*plotrange),last_symbols_text,fontsize=8)
    # '''
    plt.text(50,np.log10(7000+(.95*plotrange)),'Period Sharpe AvgProfit  Avg DD',fontsize=7.5)
    plt.text(50,np.log10(7000+(.91*plotrange)),' Life  '+fSharpeLife+'  '+fReturnLife+'  '+fDrawdownLife,fontsize=8)
    plt.text(50,np.log10(7000+(.87*plotrange)),' 3 Yr  '+fSharpe3Yr+'  '+fReturn3Yr+'  '+fDrawdown3Yr,fontsize=8)
    plt.text(50,np.log10(7000+(.83*plotrange)),' 1 Yr  '+fSharpe1Yr+'  '+fReturn1Yr+'  '+fDrawdown1Yr,fontsize=8)
    plt.text(50,np.log10(7000+(.79*plotrange)),' 6 Mo  '+fSharpe6Mo+'  '+fReturn6Mo+'  '+fDrawdown6Mo,fontsize=8)
    plt.text(50,np.log10(7000+(.75*plotrange)),' 3 Mo  '+fSharpe3Mo+'  '+fReturn3Mo+'  '+fDrawdown3Mo,fontsize=8)
    plt.text(50,np.log10(7000+(.71*plotrange)),' 1 Mo  '+fSharpe1Mo+'  '+fReturn1Mo+'  '+fDrawdown1Mo,fontsize=8)

    plt.text(50,np.log10(7000+(.54*plotrange)),last_symbols_text,fontsize=8)


    plt.plot(BuyHoldPortfolioValue,lw=3,c='r')
    plt.plot(np.average(monthvalue,axis=0),lw=4,c='k')
    # set up to use dates for labels
    xlocs = []
    xlabels = []
    for i in range(1,len(datearray)):
        if datearray[i].year != datearray[i-1].year:
            xlocs.append(i)
            xlabels.append(str(datearray[i].year))
    print("xlocs,xlabels = ", xlocs, xlabels)
    if len(xlocs) < 12 :
        plt.xticks(xlocs, xlabels)
    else:
        plt.xticks(xlocs[::2], xlabels[::2])
    plt.xlim(0,len(datearray)+25)

    for ii in range( MonteCarloPortfolioValues.shape[0] ):
        plt.plot( datearray, MonteCarloPortfolioValues[ii,:], c=(.1,.1,.1), lw=.1 )
    for ii in range( value.shape[0] ):
        plt.plot( datearray, value[ii,:], c=(.1,0.,0.), lw=.1 )

    # '''
    # import pdb
    # pdb.set_trace()
    # '''

    plt.subplot(subplotsize[1])
    plt.grid()
    plt.xlim(0,len(datearray)+25)
    ##ylim(0, value.shape[0])
    plt.ylim(0, 1.2)
    plt.plot(datearray,numberStocksUpTrendingMedian / activeCount,'g-',lw=1)
    plt.plot(datearray,numberStocksUpTrendingNearHigh / activeCount,'b-',lw=1)
    plt.plot(datearray,numberStocksUpTrendingBeatBuyHold / activeCount,'k-',lw=2)
    plt.plot(datearray,numberStocks  / activeCount,'r-')


    plt.draw()
    # save figure to disk
    from functions.GetParams import get_webpage_store
    webpage_dir = get_webpage_store(json_fn)
    outputplotname = os.path.join(webpage_dir, 'PyTAAA_monteCarloBacktestRecent.png' )
    plt.savefig(outputplotname, format='png', edgecolor='gray' )

    return


def dailyBacktest_pctLong(json_fn, verbose=False):

    # import time, threading

    import numpy as np
    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec
    import os

    import datetime
    from numpy import random
    from random import choice

    from scipy.stats import gmean

    ## local imports
    from functions.GetParams import GetEdition, get_symbols_file
    #from functions.quotes_for_list_adjClose import *
    from functions.TAfunctions import (SMA_2D,
                                       SMA,
                                       dpgchannel_2D,
                                       computeSignal2D,
                                       percentileChannel_2D,
                                       sharpeWeightedRank_2D)
    #from functions.UpdateSymbols_inHDF5 import UpdateHDF5, loadQuotes_fromHDF
    #---------------------------------------------

    print("*************************************************************")
    print("*************************************************************")
    print("***                                                       ***")
    print("***                                                       ***")
    print("***  daily montecarlo backtest                            ***")
    print("***                                                       ***")
    print("***                                                       ***")
    print("*************************************************************")
    print("*************************************************************")
    from time import sleep
    sleep(5)

    # number of monte carlo scenarios
    randomtrials = 31
    randomtrials = 12
    randomtrials = 4
    edition = GetEdition()
    if edition == 'pi' or edition == 'pine64':
        randomtrials = 12
    elif edition == 'Windows32':
        randomtrials = 25
    elif edition == 'Windows64':
        randomtrials = 51
        randomtrials = 15
    elif edition == 'MacOS':
        randomtrials = 13

    ########################################################################
    ### compute count of new highs and lows over various time periods
    ### - do this once outside the loop over monte carlo trials -- used later
    ### - don't create a plot
    ########################################################################

    from functions.GetParams import get_json_params
    from functions.CountNewHighsLows import newHighsAndLows

    params = get_json_params(json_fn)

    print("\n\n\n ... inside dailyBacktest_pctLong.py/dailyBacktest_pctLong ...")
    print("\n   . params = " + str(params))
    print("\n   . params.get('stockList') = " + str(params.get('stockList')))
    print("\n   . params['stockList'] = " + str(params['stockList']))

    if params['stockList'] == 'Naz100':
        sumNewHighs, sumNewLows, mean_TradedValue = newHighsAndLows(
            json_fn, num_days_highlow=(73,293),
            num_days_cumu=(50,159),
            HighLowRatio=(1.654,2.019),
            HighPctile=(8.499,8.952),
            HGamma=(1.,1.),
            LGamma=(1.176,1.223),
            makeQCPlots=False
        )

    elif params['stockList'] == 'SP500':
        sumNewHighs, sumNewLows, mean_TradedValue = newHighsAndLows(
            json_fn, num_days_highlow=(73,146),
            num_days_cumu=(76,108),
            HighLowRatio=(2.293,1.573),
            HighPctile=(12.197,11.534),
            HGamma=(1.157,.568),
            LGamma=(.667,1.697),
            makeQCPlots=False
        )

    # # Sum across stocks (axis=0) to get array of shape (num_dates, num_param_sets) or (num_dates,)
    # sumNewHighs = np.sum(newHighs_2D, axis=0)
    # sumNewLows = np.sum(newLows_2D, axis=0)
    # If multiple parameter sets were used, sum across parameter sets (axis=-1) to get 1D array
    if sumNewHighs.ndim > 1:
        sumNewHighs = np.sum(sumNewHighs, axis=-1)
    if sumNewLows.ndim > 1:
        sumNewLows = np.sum(sumNewLows, axis=-1)

    ##
    ##  Import list of symbols to process.
    ##
    symbols_file = get_symbols_file(json_fn)
    json_dir = os.path.split(json_fn)[0]
    p_store = get_performance_store(json_fn)

    ###############################################################################################
    ###  UpdateHDF5( symbols_directory, symbols_file )  ### assume hdf is already up to date
    adjClose, symbols, datearray, _, _ = loadQuotes_fromHDF(symbols_file, json_fn)
    firstdate = datearray[0]

    # for iii in range(len(symbols)):
    #     print(
    #         " i,symbols[i],datearray[-1],adjClose[i,-1] = ",
    #         iii, symbols[iii], datearray[-1], adjClose[iii,-1]
    #     )

    # Clean up missing values in input quotes
    #  - infill interior NaN values using nearest good values to linearly interpolate
    #  - copy first valid quote from valid date to all earlier positions
    #  - copy last valid quote from valid date to all later positions
    for ii in range(adjClose.shape[0]):
        adjClose[ii,:] = interpolate(adjClose[ii,:])
        adjClose[ii,:] = cleantobeginning(adjClose[ii,:])
        adjClose[ii,:] = cleantoend(adjClose[ii,:])

    import os
    basename = os.path.split( symbols_file )[-1]
    print("basename = ", basename)

    # set up to write monte carlo results to disk.
    if basename == "symbols.txt" :
        runnum = 'run2501'
        plotmax = 1.e5     # maximum value for plot (figure 3)
        holdMonths = [1,2,3,4,6,12]
    elif basename.lower() == "Naz100_Symbols.txt".lower() :
        runnum = 'run2502'
        plotmax = 1.e10     # maximum value for plot (figure 3)
        holdMonths = [1,1,1,2,2,3,4,6,12]
    elif basename == "biglist.txt" :
        runnum = 'run2503'
        plotmax = 1.e9     # maximum value for plot (figure 3)
        holdMonths = [1,2,3,4,6,12]
    elif basename == "ProvidentFundSymbols.txt" :
        runnum = 'run2504'
        plotmax = 1.e7     # maximum value for plot (figure 3)
        holdMonths = [4,6,12]
    elif basename == "SP500_Symbols.txt" :
        runnum = 'run2505'
        plotmax = 1.e9     # maximum value for plot (figure 3)
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

    print(" security values check: ",adjClose[np.isnan(adjClose)].shape)

    print("\n\n ... inside dailyBacktest_pctLong")
    print("   . basename = " + basename)
    print("   . runnum = " + runnum)
    print("   . plotmax = " + str(plotmax))

    ########################################################################

    gainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    activeCount = np.zeros(adjClose.shape[1],dtype=float)

    numberSharesCalc = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    gainloss[:,1:] = adjClose[:,1:] / adjClose[:,:-1]
    gainloss[np.isnan(gainloss)]=1.
    value = 10000. * np.cumprod(gainloss,axis=1)   ### used bigger constant for inverse of quotes
    BuyHoldFinalValue = np.average(value,axis=0)[-1]

    print(" gainloss check: ",gainloss[np.isnan(gainloss)].shape)
    print(" value check: ",value[np.isnan(value)].shape)
    lastEmptyPriceIndex = np.zeros(adjClose.shape[0],dtype=int)
    firstTrailingEmptyPriceIndex = np.zeros(adjClose.shape[0],dtype=int)

    for ii in range(adjClose.shape[0]):
        # take care of special case where constant share price is inserted at beginning of series
        index = np.argmax(np.clip(np.abs(gainloss[ii,:]-1),0,1e-8)) - 1
        if verbose:
            print(
                "first valid price and date = ",
                symbols[ii]," ",index," ",datearray[index]
            )
        lastEmptyPriceIndex[ii] = index
        activeCount[lastEmptyPriceIndex[ii]+1:] += 1

    for ii in range(adjClose.shape[0]):
        # take care of special case where no quote exists at end of series
        tempQuotes = adjClose[ii,:]
        tempQuotes[ np.isnan(tempQuotes) ] = 1.0
        index = np.argmax(np.clip(np.abs(tempQuotes[::-1]-1),0,1e-8)) - 1
        if index != -1:
            firstTrailingEmptyPriceIndex[ii] = -index
            print("first trailing invalid price: index and date = ",symbols[ii]," ",firstTrailingEmptyPriceIndex[ii]," ",datearray[index])
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

    # params = get_json_params(json_fn)

    trade_cost = params['trade_cost']

    for iter in range(randomtrials):

        if iter%1==0:
            print("")
            print("")
            print(" random trial:  ",iter)

        LongPeriod_random = int(random.uniform(149,180)+.5)
        LongPeriod_random = int(random.uniform(55,280)+.5)
        stddevThreshold = random.uniform(3.0,7.)
        MA1 = int(random.uniform(15,250)+.5)
        MA2 = int(random.uniform(7,30)+.5)
        MA2offset = int(random.uniform(.6,5)+.5)
        numberStocksTraded = int(random.uniform(1.9,8.9)+.5)
        monthsToHold = choice(holdMonths)
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

        if iter == 0 or iter == randomtrials-1:
            print("\n\n\n")
            print("*********************************\nUsing pyTAAApi linux edition parameters .....\n")
            params = get_json_params(json_fn)
            monthsToHold = params['monthsToHold']
            numberStocksTraded = params['numberStocksTraded']
            LongPeriod = params['LongPeriod']
            stddevThreshold = params['stddevThreshold']
            MA1 = int(params['MA1'])
            MA2 = int(params['MA2'])
            MA3 = int(params['MA3'])
            MA2offset = int(params['MA3']) - MA2
            sma2factor = params['MA2factor']
            rankThresholdPct = params['rankThresholdPct']
            riskDownside_min = params['riskDownside_min']
            riskDownside_max = params['riskDownside_max']

            narrowDays = params['narrowDays']
            mediumDays = params['mediumDays']
            wideDays = params['wideDays']

            lowPct = float(params['lowPct'])
            hiPct = float(params['hiPct'])
            uptrendSignalMethod = params['uptrendSignalMethod']
            paramNumberToVary = ""


        # 2 random trial use the parameters from GetParams(), namely indices 0 and randomtrials-1.
        # for iter==0, results are saved to compare against and determine if random params provide better recent results.
        # for iter==randomtrials-1, results are used for backtest plots on webpage.
        if 0 < iter < randomtrials-1 :
            print("\n\n\n")
            print("*********************************\nUsing pyTAAA parameters +/- 20% .....\n")
            params = get_json_params(json_fn)
            monthsToHold = params['monthsToHold']
            numberStocksTraded = params['numberStocksTraded']
            LongPeriod = params['LongPeriod']
            stddevThreshold = params['stddevThreshold']
            MA1 = int(params['MA1'])
            MA2 = int(params['MA2'])
            MA3 = int(params['MA3'])
            MA2offset = params['MA3'] - params['MA2']
            sma2factor = params['MA2factor']
            rankThresholdPct = params['rankThresholdPct']
            riskDownside_min = params['riskDownside_min']
            riskDownside_max = params['riskDownside_max']
            uptrendSignalMethod = params['uptrendSignalMethod']

            paramNumberToVary = choice([0,1,2,3,4,5,6,7,8,9])

            numberStocksTraded += choice([-1,0,1])
            for kk in range(15):
                temp = choice(holdMonths)
                if temp != monthsToHold:
                    monthsToHold = temp
                    break
            monthsToHold = choice(holdMonths)
            LongPeriod = int(LongPeriod + np.around(random.uniform(-.20*LongPeriod, .20*LongPeriod))+.5)
            stddevThreshold = stddevThreshold * random.uniform(.80, 1.20)
            MA1 = MA1 + int(np.around(random.uniform(-.20*MA1, .20*MA1))+.5)
            MA2 = MA2 + int(np.around(random.uniform(-.20*MA2, .20*MA2))+.5)
            if MA2< MA1:
                MAtemp = [MA2,MA1]
                MA1 = MAtemp[0]
                MA2 = MAtemp[1]
            MA3 = MA3 + int(np.around(random.uniform(-.20*MA3, .20*MA3))+.5)
            MA2offset = params['MA3'] - params['MA2']
            sma2factor = sma2factor + np.around(random.uniform(-.20*sma2factor, .20*sma2factor),-3)
            rankThresholdPct = rankThresholdPct + np.around(random.uniform(-.20*rankThresholdPct, .20*rankThresholdPct),-2)
            riskDownside_min = riskDownside_min + np.around(random.uniform(-.20*riskDownside_min, .20*riskDownside_min),-3)
            riskDownside_max = riskDownside_max + np.around(random.uniform(-.20*riskDownside_max, .20*riskDownside_max),-3)

            days_NarrowChannel = random.triangular(4,5,9) # 4.6
            factor_NarrowChannel = random.triangular(1.16,1.3,1.36) # 1.16
            #middle = 1.163
            #factor_NarrowChannel = random.uniform(.85*middle,1.15*middle) # 1.16

            days_MediumChannel = random.triangular(17,22,27) # 21.6
            middle = 1.09
            factor_MediumChannel = random.uniform(1.04,1.15) # 1.08
            factor_MediumChannel = random.uniform(.85*middle,1.15*middle) # 1.08

            days_WideChannel = random.triangular(45,70,125) # 67.08
            middle = 1.215
            factor_WideChannel = random.uniform(1.12,1.28) # 1.2
            factor_WideChannel = random.uniform(.85*middle,1.15*middle)

            narrowDays = []
            mediumDays = []
            wideDays = []
            for iii in range(8):
                if iii>0:
                    narrowDays.append( narrowDays[-1] * factor_NarrowChannel )
                    mediumDays.append( mediumDays[-1] * factor_MediumChannel )
                    wideDays.append( wideDays[-1] * factor_WideChannel )
                else:
                    narrowDays.append( days_NarrowChannel )
                    mediumDays.append( days_MediumChannel )
                    wideDays.append( days_WideChannel )

            lowPct = float(params['lowPct'])*random.uniform(.85,1.15)
            hiPct = float(params['hiPct'])*random.uniform(.85,1.15)


        print(" LongPeriod = ", LongPeriod)
        print("")
        print(" months to hold = ",holdMonths,monthsToHold)
        print("")
        monthgainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
        monthgainloss[:,LongPeriod:] = adjClose[:,LongPeriod:] / adjClose[:,:-LongPeriod]
        monthgainloss[np.isnan(monthgainloss)]=1.

        ####################################################################
        ###
        ### calculate signal for uptrending stocks (in signal2D)
        ### - method depends on params uptrendSignalMethod
        ###
        ####################################################################

        if uptrendSignalMethod == 'SMAs' :
            print("  ...using 3 SMA's for signal2D")
            print("\n\n ...calculating signal2D using '"+uptrendSignalMethod+"' method...")
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
                        if jj== adjClose.shape[1]-1 and np.isnan(adjClose[ii,-1]):
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
                        # print("date, signal2D changed",datearray[jj])
                        pass

            numberStocks = np.sum(signal2D,axis = 0)

        elif uptrendSignalMethod == 'minmaxChannels' :
            print("  ...using 3 minmax channels for signal2D")
            print("\n\n ...calculating signal2D using '"+uptrendSignalMethod+"' method...")

            ########################################################################
            ## Calculate signal for all stocks based on 3 minmax channels (dpgchannels)
            ########################################################################

            # narrow channel is designed to remove day-to-day variability

            print("narrow days min,max,inc = ", narrowDays[0], narrowDays[-1], (narrowDays[-1]-narrowDays[0])/7.)
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
                        if jj== adjClose.shape[1]-1 and np.isnan(adjClose[ii,-1]):
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
            print("\n\n ...calculating signal2D using '"+uptrendSignalMethod+"' method...")
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

                    if jj== adjClose.shape[1]-1 and np.isnan(adjClose[ii,-1]):
                        signal2D[ii,jj] = 0                #### added to avoid choosing stocks no longer in index
                # take care of special case where constant share price is inserted at beginning of series
                index = np.argmax(np.clip(np.abs(gainloss[ii,:]-1),0,1e-8)) - 1
                signal2D[ii,0:index] = 0


        if params['uptrendSignalMethod'] == 'percentileChannels':
            signal2D, lowChannel, hiChannel = computeSignal2D( adjClose, gainloss, params )
        else:
            signal2D = computeSignal2D( adjClose, gainloss, params )

        # SP500 pre-2002 condition: Force 100% CASH allocation (overrides rolling window filter)
        if params.get('stockList') == 'SP500':
            print("\n   . DEBUG: Applying SP500 pre-2002 condition: Forcing 100% CASH allocation for all stocks before 2002-01-01")
            cutoff_date = datetime.date(2002, 1, 1)
            for date_idx in range(len(datearray)):
                if datearray[date_idx] < cutoff_date:
                    # Zero all stock signals for 100% CASH allocation
                    signal2D[:, date_idx] = 0.0
        # else:
        # Apply rolling window data quality filter if enabled (only for non-SP500)
        # if params.get('enable_rolling_filter', False):  # Default disabled for performance
        from functions.rolling_window_filter import apply_rolling_window_filter
        signal2D = apply_rolling_window_filter(
            adjClose, signal2D, params.get('window_size', 50),
            symbols=symbols, datearray=datearray
        )

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

        print(" signal2D check: ",signal2D[np.isnan(signal2D)].shape)
        print(" numberStocks (uptrending) min, mean,max: ",numberStocks.min(),numberStocks.mean(),numberStocks.max())


        ########################################################################
        ### compute weights for each stock based on:
        ### 1. uptrending signal in "signal2D"
        ### 1. delta-rank computed from gain/loss over "LongPeriod_random"
        ### 2. sharpe ratio computed from daily gains over "LongPeriod"
        ########################################################################

        monthgainlossweight = sharpeWeightedRank_2D(
            json_fn, datearray, symbols, adjClose,
            signal2D, signal2D_daily,
            LongPeriod, numberStocksTraded, riskDownside_min, riskDownside_max,
            rankThresholdPct,
            stddevThreshold=stddevThreshold,
            is_backtest=True,
            makeQCPlots=False,
            stockList=params.get('stockList', 'SP500')
        )

        print("here I am........")

        # """
        # ########################################################################
        # ### compute traded value of stock for each month
        # ########################################################################

        # monthvalue = value.copy()
        # print " 1 - monthvalue check: ",monthvalue[isnan(monthvalue)].shape
        # for ii in np.arange(1,monthgainloss.shape[1]):
        #     if (datearray[ii].month != datearray[ii-1].month) and ( (datearray[ii].month - 1)%monthsToHold == 0):
        #         valuesum=np.sum(monthvalue[:,ii-1])
        #         for jj in range(value.shape[0]):
        #             monthvalue[jj,ii] = monthgainlossweight[jj,ii]*valuesum*gainloss[jj,ii]   # re-balance using weights (that sum to 1.0)
        #     else:
        #         for jj in range(value.shape[0]):
        #             monthvalue[jj,ii] = monthvalue[jj,ii-1]*gainloss[jj,ii]


        # numberSharesCalc = monthvalue / adjClose    # for info only
        # """

        ########################################################################
        ### compute traded value of stock for each month
        ########################################################################

        # initialize monthvalue (assume B&H)
        monthvalue = value.copy()

        starting_valid_symbols = value[:,0]
        starting_valid_symbols_count = starting_valid_symbols[starting_valid_symbols > 1.e-4].shape[0]

        print("  \n\n\n")
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

                #print "date,changed#, commission,valuemean,yesterdayValue,todayValue,commissionFactor(%)= ", \
                #       datearray[ii], symbol_changed_count, commission, valuemean, yesterdayValue,todayValue,format(commission_factor-1.,'5.2%')
                ### Note: this is only approximate to what I really want to do in trades. This models all percentages changing
                ###       which implies trading all stocks. But I really want just to adjust CASH balance if monthgainlossweight is constant.
                for jj in range(value.shape[0]):
                    monthvalue[jj,ii] = gainloss[jj,ii]*valuesum*monthgainlossweight[jj,ii]*commission_factor   # re-balance using weights (that sum to 1.0 less commissions)

            else:
                for jj in range(value.shape[0]):
                    monthvalue[jj,ii] = monthvalue[jj,ii-1]*gainloss[jj,ii]

        numberSharesCalc = monthvalue / adjClose    # for info only

        print("  \n\n\n")

        ########################################################################
        ### gather statistics on number of uptrending stocks
        ########################################################################

        try:
            print(" iter = ", iter)
            print(" numberStocks = ", numberStocks)
            print(" numberStocksUpTrending[:iter,:].shape = ", numberStocksUpTrending[:iter,:].shape)
            print(" np.median(numberStocksUpTrending[:iter,:],axis=0) = ", np.median(numberStocksUpTrending[:iter,:],axis=0))

            numberStocksUpTrending[iter,:] = numberStocks
            numberStocksUpTrendingMedian = np.median(numberStocksUpTrending[:iter,:],axis=0)
            numberStocksUpTrendingMean   = np.mean(numberStocksUpTrending[:iter,:],axis=0)
        except:
            numberStocksUpTrending[iter,:] = numberStocks
            numberStocksUpTrendingMedian = numberStocks
            numberStocksUpTrendingMean   = numberStocks

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
        Sharpe6Mo = ( gmean(PortfolioDailyGains[-126:])**252 -1. ) / ( np.std(PortfolioDailyGains[-126:])*sqrt(252) )
        Sharpe3Mo = ( gmean(PortfolioDailyGains[-63:])**252 -1. ) / ( np.std(PortfolioDailyGains[-63:])*sqrt(252) )
        Sharpe1Mo = ( gmean(PortfolioDailyGains[-21:])**252 -1. ) / ( np.std(PortfolioDailyGains[-21:])*sqrt(252) )
        PortfolioSharpe[iter] = ( gmean(PortfolioDailyGains)**252 -1. ) / ( np.std(PortfolioDailyGains)*sqrt(252) )

        print("15 year : ",index,PortfolioValue[-1], PortfolioValue[-index],datearray[-index])

        Return15Yr = (PortfolioValue[-1] / PortfolioValue[-index])**(252./index)
        Return10Yr = (PortfolioValue[-1] / PortfolioValue[-2520])**(1/10.)
        Return5Yr = (PortfolioValue[-1] / PortfolioValue[-1260])**(1/5.)
        Return3Yr = (PortfolioValue[-1] / PortfolioValue[-756])**(1/3.)
        Return2Yr = (PortfolioValue[-1] / PortfolioValue[-504])**(1/2.)
        Return1Yr = (PortfolioValue[-1] / PortfolioValue[-252])
        Return6Mo = (PortfolioValue[-1] / PortfolioValue[-126])**(2.)
        Return3Mo = (PortfolioValue[-1] / PortfolioValue[-63])**(4.)
        Return1Mo = (PortfolioValue[-1] / PortfolioValue[-21])**(12.)
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
        Drawdown6Mo = np.mean(PortfolioDrawdown[-126:])
        Drawdown3Mo = np.mean(PortfolioDrawdown[-63:])
        Drawdown1Mo = np.mean(PortfolioDrawdown[-21:])

        if iter == 0:

            Sharpe3Yr_PyTAAA = Sharpe3Yr
            Sharpe2Yr_PyTAAA = Sharpe2Yr
            Sharpe1Yr_PyTAAA = Sharpe1Yr
            Sharpe6Mo_PyTAAA = Sharpe6Mo
            Sharpe3Mo_PyTAAA = Sharpe3Mo
            Sharpe1Mo_PyTAAA = Sharpe1Mo
            Return3Yr_PyTAAA = Return3Yr
            Return2Yr_PyTAAA = Return2Yr
            Return1Yr_PyTAAA = Return1Yr
            Return6Mo_PyTAAA = Return6Mo
            Return3Mo_PyTAAA = Return3Mo
            Return1Mo_PyTAAA = Return1Mo
            Drawdown3Yr_PyTAAA = Drawdown3Yr
            Drawdown2Yr_PyTAAA = Drawdown2Yr
            Drawdown1Yr_PyTAAA = Drawdown1Yr
            Drawdown6Mo_PyTAAA = Drawdown6Mo
            Drawdown3Mo_PyTAAA = Drawdown3Mo
            Drawdown1Mo_PyTAAA = Drawdown1Mo

            BuyHoldPortfolioValue = np.mean(value,axis=0)
            BuyHoldDailyGains = BuyHoldPortfolioValue[1:] / BuyHoldPortfolioValue[:-1]
            BuyHoldSharpe15Yr = ( gmean(BuyHoldDailyGains[-index:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-index:])*sqrt(252) )
            BuyHoldSharpe10Yr = ( gmean(BuyHoldDailyGains[-2520:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-2520:])*sqrt(252) )
            BuyHoldSharpe5Yr  = ( gmean(BuyHoldDailyGains[-1126:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-1260:])*sqrt(252) )
            BuyHoldSharpe3Yr  = ( gmean(BuyHoldDailyGains[-756:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-756:])*sqrt(252) )
            BuyHoldSharpe2Yr  = ( gmean(BuyHoldDailyGains[-504:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-504:])*sqrt(252) )
            BuyHoldSharpe1Yr  = ( gmean(BuyHoldDailyGains[-252:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-252:])*sqrt(252) )
            BuyHoldSharpe6Mo  = ( gmean(BuyHoldDailyGains[-126:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-126:])*sqrt(252) )
            BuyHoldSharpe3Mo  = ( gmean(BuyHoldDailyGains[-63:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-63:])*sqrt(252) )
            BuyHoldSharpe1Mo  = ( gmean(BuyHoldDailyGains[-21:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-21:])*sqrt(252) )
            BuyHoldReturn15Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-index])**(252./index)
            BuyHoldReturn10Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-2520])**(1/10.)
            BuyHoldReturn5Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-1260])**(1/5.)
            BuyHoldReturn3Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-756])**(1/3.)
            BuyHoldReturn2Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-504])**(1/2.)
            BuyHoldReturn1Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-252])
            BuyHoldReturn6Mo = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-126])**(2.)
            BuyHoldReturn3Mo = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-63])**(4.)
            BuyHoldReturn1Mo = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-21])**(12.)
            for jj in range(BuyHoldPortfolioValue.shape[0]):
                MaxBuyHoldPortfolioValue[jj] = max(MaxBuyHoldPortfolioValue[jj-1],BuyHoldPortfolioValue[jj])

            BuyHoldPortfolioDrawdown = BuyHoldPortfolioValue / MaxBuyHoldPortfolioValue - 1.
            BuyHoldDrawdown15Yr = np.mean(BuyHoldPortfolioDrawdown[-index:])
            BuyHoldDrawdown10Yr = np.mean(BuyHoldPortfolioDrawdown[-2520:])
            BuyHoldDrawdown5Yr = np.mean(BuyHoldPortfolioDrawdown[-1260:])
            BuyHoldDrawdown3Yr = np.mean(BuyHoldPortfolioDrawdown[-756:])
            BuyHoldDrawdown2Yr = np.mean(BuyHoldPortfolioDrawdown[-504:])
            BuyHoldDrawdown1Yr = np.mean(BuyHoldPortfolioDrawdown[-252:])
            BuyHoldDrawdown6Mo = np.mean(BuyHoldPortfolioDrawdown[-126:])
            BuyHoldDrawdown3Mo = np.mean(BuyHoldPortfolioDrawdown[-63:])
            BuyHoldDrawdown1Mo = np.mean(BuyHoldPortfolioDrawdown[-21:])

        print("")
        print("")
        print("Sharpe15Yr, BuyHoldSharpe15Yr = ", Sharpe15Yr, BuyHoldSharpe15Yr)
        print("Sharpe10Yr, BuyHoldSharpe10Yr = ", Sharpe10Yr, BuyHoldSharpe10Yr)
        print("Sharpe5Yr, BuyHoldSharpe5Yr =   ", Sharpe5Yr, BuyHoldSharpe5Yr)
        print("Sharpe3Yr, BuyHoldSharpe3Yr =   ", Sharpe3Yr, BuyHoldSharpe3Yr)
        print("Sharpe2Yr, BuyHoldSharpe2Yr =   ", Sharpe2Yr, BuyHoldSharpe2Yr)
        print("Sharpe1Yr, BuyHoldSharpe1Yr =   ", Sharpe1Yr, BuyHoldSharpe1Yr)
        print("Return15Yr, BuyHoldReturn15Yr = ", Return15Yr, BuyHoldReturn15Yr)
        print("Return10Yr, BuyHoldReturn10Yr = ", Return10Yr, BuyHoldReturn10Yr)
        print("Return5Yr, BuyHoldReturn5Yr =   ", Return5Yr, BuyHoldReturn5Yr)
        print("Return3Yr, BuyHoldReturn3Yr =   ", Return3Yr, BuyHoldReturn3Yr)
        print("Return2Yr, BuyHoldReturn2Yr =   ", Return2Yr, BuyHoldReturn2Yr)
        print("Return1Yr, BuyHoldReturn1Yr =   ", Return1Yr, BuyHoldReturn1Yr)
        print("Drawdown15Yr, BuyHoldDrawdown15Yr = ", Drawdown15Yr, BuyHoldDrawdown15Yr)
        print("Drawdown10Yr, BuyHoldDrawdown10Yr = ", Drawdown10Yr, BuyHoldDrawdown10Yr)
        print("Drawdown5Yr, BuyHoldDrawdown5Yr =   ", Drawdown5Yr, BuyHoldDrawdown5Yr)
        print("Drawdown3Yr, BuyHoldDrawdown3Yr =   ", Drawdown3Yr, BuyHoldDrawdown3Yr)
        print("Drawdown2Yr, BuyHoldDrawdown2Yr =   ", Drawdown2Yr, BuyHoldDrawdown2Yr)
        print("Drawdown1Yr, BuyHoldDrawdown1Yr =   ", Drawdown1Yr, BuyHoldDrawdown1Yr)

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
            print("found monte carlo trial that beats BuyHold (test2)...")
            print("shape of numberStocksUpTrendingBeatBuyHold = ",numberStocksUpTrendingBeatBuyHold.shape)
            print("mean of numberStocksUpTrendingBeatBuyHold values = ",np.mean(numberStocksUpTrendingBeatBuyHold))
            beatBuyHold2Count += 1
            numberStocksUpTrendingBeatBuyHold = (numberStocksUpTrendingBeatBuyHold * (beatBuyHold2Count -1) + numberStocks) / beatBuyHold2Count


        print("beatBuyHoldTest = ", beatBuyHoldTest, beatBuyHoldTest2)
        print("countof trials that BeatBuyHold  = ", beatBuyHoldCount)
        print("countof trials that BeatBuyHold2 = ", beatBuyHold2Count)
        print("")
        print("")

        if iter > 1:
            for jj in range(adjClose.shape[1]):
                numberStocksUpTrendingNearHigh[jj]   = np.percentile(numberStocksUpTrending[:iter,jj], 90)

        if iter == 0:
            from time import sleep
            for i in range(len(symbols)):
                plt.clf()
                plt.grid()
                ##plot(datearray,signal2D[i,:]*np.mean(adjClose[i,:])*numberStocksTraded/2)
                plt.plot(datearray,adjClose[i,:])
                aaa = signal2D[i,:]
                NaNcount = aaa[np.isnan(aaa)].shape[0]
                plt.title("signal2D before figure3 ... "+symbols[i]+"   "+str(NaNcount))
                plt.draw()
                #time.sleep(.2)

        print(" ")
        print("The B&H portfolio final value is: ","{:,}".format(int(BuyHoldFinalValue)))
        print(" ")
        print("Monthly re-balance based on ",LongPeriod, "days of recent performance.")
        print("The portfolio final value is: ","{:,}".format(int(np.average(monthvalue,axis=0)[-1])))
        print(" ")
        print("Today's top ranking choices are: ")
        last_symbols_text = []
        for ii in range(len(symbols)):
            if monthgainlossweight[ii,-1] > 0:
                # print symbols[ii]
                print(datearray[-1], symbols[ii],monthgainlossweight[ii,-1])
                last_symbols_text.append( symbols[ii] )


        ########################################################################
        ### compute traded value of stock for each month (using varying percent invested)
        ########################################################################

        ###
        ### gather sum of all quotes minus SMA
        ###
        QminusSMADays = int(random.uniform(252,5*252)+.5)
        QminusSMADays = int(random.uniform(352,2*352)+.5)
        QminusSMAFactor = random.triangular(.88,.91,.999)

        # re-calc constant monthPctInvested
        uptrendConst = random.uniform(0.45,0.75)
        PctInvestSlope = random.triangular(2.,5.,7.)
        PctInvestIntercept = -random.triangular(-.05,0.0,.07)
        maxPctInvested = choice([1.0,1.0,1.0,1.2,1.33,1.5])
        maxPctInvested = 1.5

        if iter == 0 :
            print("\n\n\n")
            print("*********************************\nUsing pyTAAA parameters .....\n")
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


        print(" NaNs in value = ", (value[np.isnan(value)]).shape)
        print(" Infs in value = ", (value[np.isinf(value)]).shape)

        monthvalueVariablePctInvest = value.copy()
        print(" 1 - monthvalueVariablePctInvest check: ",monthvalueVariablePctInvest[np.isnan(monthvalueVariablePctInvest)].shape)
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

        print("15 year : ",index,PortfolioValue[-1], PortfolioValue[-index],datearray[-index])

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

        plt.rcParams['figure.edgecolor'] = 'grey'
        plt.rc('savefig',edgecolor = 'grey')
        fig = plt.figure(1)
        plt.clf()
        subplotsize = gridspec.GridSpec(3,1,height_ratios=[4,1,1])
        plt.subplot(subplotsize[0])
        plt.grid()
        ##
        ## make plot of all stocks' individual prices
        ##
        plt.plot( np.average(value,axis=0), lw=.3, c='r', alpha=.5 )

        print("\n\n ... inside daily_backtest")

        if iter == 0:

            plt.yscale('log')
            plt.ylim([1000,max(10000,plotmax)])
            ymin, ymax = np.log10(1e3), np.log10(max(10000,plotmax))
            bin_width = (ymax - ymin) / 50
            print("ymin = ", ymin)
            print("ymax = ", ymax)
            print("bin_width = ", bin_width)

            y_bins = np.arange(ymin, ymax+.0000001, bin_width)
            print("y_bins = ", y_bins)

            AllStocksHistogram = np.ones((len(y_bins)-1, len(datearray),3))
            HH = np.zeros((len(y_bins)-1, len(datearray)))
            mm = np.zeros(len(datearray))
            xlocs = []
            xlabels = []
            for i in range(1,len(datearray)):
                ValueOnDate = value[:,i].copy()
                ValueOnDate = ValueOnDate[~np.isnan(ValueOnDate)]
                ValueOnDate = ValueOnDate[~np.isinf(ValueOnDate)]
                #print " i,datearray[i],ValueOnDate.min(),ValueOnDate.max() = ", i,datearray[i],ValueOnDate.min(),ValueOnDate.max()

                if ValueOnDate[ValueOnDate == 10000].shape[0] > 1:
                    ValueOnDate[ValueOnDate == 10000] = 1.
                    ValueOnDate[np.argmin(ValueOnDate)] = 10000.
                h, _ = np.histogram(np.log10(ValueOnDate/10000.), bins=y_bins, density=True)
                # reverse so big numbers become small(and print out black)
                h = 1. - h
                # set range to [.5,1.]
                h /= 2.
                h += .5
                HH[:,i] = h
                mm[i] = np.median(value[-1,:])
                if datearray[i].year != datearray[i-1].year:
                    # print(" inside histogram evaluation for date = ", datearray[i])
                    xlocs.append(i)
                    xlabels.append(str(datearray[i].year))
            AllStocksHistogram[:,:,2] = HH
            AllStocksHistogram[:,:,1] = AllStocksHistogram[:,:,2]
            AllStocksHistogram = np.clip(AllStocksHistogram,AllStocksHistogram.max()*.05,1)
            AllStocksHistogram /= AllStocksHistogram.max()


        #plt.imshow(AllStocksHistogram, cmap='Reds_r', aspect='auto', origin='lower', extent=(0, len(datearray), 10**ymin, 10**ymax))
        plt.grid(True)
        ##
        ## cumulate final values for grayscale histogram overlay
        ##
        if iter == 0:
            MonteCarloPortfolioValues = np.zeros( (randomtrials, len(datearray) ), float )
        MonteCarloPortfolioValues[iter,:] = np.average(monthvalue,axis=0)
        plt.plot( datearray, MonteCarloPortfolioValues[iter,:], lw=.3, c='k', alpha=.5 )

        if iter > 9 and iter%10 == 0:
            plt.yscale('log')
            plt.ylim([1000,max(10000,plotmax)])
            ymin, ymax = np.log10(1e3), np.log10(max(10000,plotmax))
            bin_width = (ymax - ymin) / 50
            y_bins = np.arange(ymin, ymax+.0000001, bin_width)
            H = np.zeros((len(y_bins)-1, len(datearray)))
            m = np.zeros(len(datearray))
            hb = np.zeros((len(y_bins)-1, len(datearray),3))
            for i in range(1,len(datearray)):
                h, _ = np.histogram(np.log10(MonteCarloPortfolioValues[:iter,i]/10000.), bins=y_bins, density=True)
                # reverse so big numbers become small(and print out black)
                h = 1. - h
                # set range to [.5,1.]
                h = np.clip( h, .05, 1. )
                h /= 2.
                h += .5
                H[:,i] = h
                m[i] = np.median(value[-1,:])
                # if datearray[i].year != datearray[i-1].year:
                #     print(" inside histogram evaluation for date = ", datearray[i])
            hb[:,:,0] = H
            hb[:,:,1] = H
            hb[:,:,2] = H
            hb = .5 * AllStocksHistogram + .5 * hb

        if iter > 10  :
            plt.yscale('log')
            ##plt.imshow(hb, cmap='gray', aspect='auto', origin='lower', extent=(0, len(datearray), 10**ymin, 10**ymax))

        for ii in range( MonteCarloPortfolioValues.shape[0] ):
            plt.plot( MonteCarloPortfolioValues[ii,:], c='k', lw=.3, alpha=.5 )
        for ii in range( value.shape[0] ):
            plt.plot( value[ii,:], c=(1.,0.5,0.5), lw=.1, alpha=.5 )

        plt.yscale('log')
        plt.plot( np.average(monthvalue,axis=0), lw=3, c='k' )
        plt.grid(True)
        plt.draw()

        ##
        ## continue
        ##
        #FinalTradedPortfolioValue[iter] = np.average(monthvalue[:,-1])
        FinalTradedPortfolioValue[iter] = np.average(monthvalue, axis=0)[-1]
        fFinalTradedPortfolioValue = "{:,}".format(int(FinalTradedPortfolioValue[iter]))
        PortfolioDailyGains = np.average(monthvalue,axis=0)[1:] / np.average(monthvalue,axis=0)[:-1]
        PortfolioSharpe[iter] = ( gmean(PortfolioDailyGains)**252 -1. ) / ( np.std(PortfolioDailyGains)*sqrt(252) )
        fPortfolioSharpe = format(PortfolioSharpe[iter],'5.2f')

        FinalTradedVarPctPortfolioValue = np.average(monthvalueVariablePctInvest[:,-1])
        fVFinalTradedPortfolioValue = "{:,}".format(int(FinalTradedVarPctPortfolioValue))
        PortfolioDailyGains = np.average(monthvalueVariablePctInvest,axis=0)[1:] / np.average(monthvalueVariablePctInvest,axis=0)[:-1]
        PortVarPctfolioSharpe = ( gmean(PortfolioDailyGains)**252 -1. ) / ( np.std(PortfolioDailyGains)*sqrt(252) )
        fVPortfolioSharpe = format(PortVarPctfolioSharpe,'5.2f')

        print("")
        print(" value 2 yrs ago, 1 yr ago, last = ",np.average(monthvalue[:,-504]),np.average(monthvalue[:,-252]),np.average(monthvalue[:,-1]))
        print(" one year gain, daily geom mean, stdev = ",np.average(monthvalue,axis=0)[-1] / np.average(monthvalue,axis=0)[-252],gmean(PortfolioDailyGains[-252:])**252 -1.,np.std(PortfolioDailyGains[-252:])*sqrt(252))
        print(" two year gain, daily geom mean, stdev = ",np.average(monthvalue,axis=0)[-1] / np.average(monthvalue,axis=0)[-504],gmean(PortfolioDailyGains[-504:])**252 -1.,np.std(PortfolioDailyGains[-504:])*sqrt(252))

        title_text = str(iter)+":  "+ \
                      str(int(numberStocksTraded))+"__"+   \
                      str(int(monthsToHold))+"__"+   \
                      str(int(LongPeriod))+"-"+   \
                      format(stddevThreshold,'4.2f')+"-"+   \
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

        plt.title( title_text, fontsize = 9 )
        fSharpe15Yr = format(Sharpe15Yr,'5.2f')
        fSharpe10Yr = format(Sharpe10Yr,'5.2f')
        fSharpe5Yr = format(Sharpe5Yr,'5.2f')
        fSharpe3Yr = format(Sharpe3Yr,'5.2f')
        fSharpe2Yr = format(Sharpe2Yr,'5.2f')
        fSharpe1Yr = format(Sharpe1Yr,'5.2f')
        fSharpe6Mo = format(Sharpe6Mo,'5.2f')
        fSharpe3Mo = format(Sharpe3Mo,'5.2f')
        fSharpe1Mo = format(Sharpe1Mo,'5.2f')
        fReturn15Yr = format(Return15Yr,'5.2f')
        fReturn10Yr = format(Return10Yr,'5.2f')
        fReturn5Yr = format(Return5Yr,'5.2f')
        fReturn3Yr = format(Return3Yr,'5.2f')
        fReturn2Yr = format(Return2Yr,'5.2f')
        fReturn1Yr = format(Return1Yr,'5.2f')
        fReturn6Mo = format(Return6Mo,'5.2f')
        fReturn3Mo = format(Return3Mo,'5.2f')
        fReturn1Mo = format(Return1Mo,'5.2f')
        fDrawdown15Yr = format(Drawdown15Yr,'.1%')
        fDrawdown10Yr = format(Drawdown10Yr,'.1%')
        fDrawdown5Yr = format(Drawdown5Yr,'.1%')
        fDrawdown3Yr = format(Drawdown3Yr,'.1%')
        fDrawdown2Yr = format(Drawdown2Yr,'.1%')
        fDrawdown1Yr = format(Drawdown1Yr,'.1%')
        fDrawdown6Mo = format(Drawdown6Mo,'.1%')
        fDrawdown3Mo = format(Drawdown3Mo,'.1%')
        fDrawdown1Mo = format(Drawdown1Mo,'.1%')

        print(" one year sharpe = ",fSharpe1Yr)
        print("")
        plotrange = np.log10(plotmax / 1000.)
        plt.text( 50, 1500, symbols_file, fontsize=8 )
        plt.text( 50, 2500, "Backtested on "+datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"), fontsize=7.5 )
        plt.text(50,1000*10**(.95*plotrange),'Period Sharpe AvgProfit  Avg DD',fontsize=7.5)
        plt.text(50,1000*10**(.91*plotrange),'15 Yr '+fSharpe15Yr+'  '+fReturn15Yr+'  '+fDrawdown15Yr,fontsize=8)
        plt.text(50,1000*10**(.87*plotrange),'10 Yr '+fSharpe10Yr+'  '+fReturn10Yr+'  '+fDrawdown10Yr,fontsize=8)
        plt.text(50,1000*10**(.83*plotrange),' 5 Yr  '+fSharpe5Yr+'  '+fReturn5Yr+'  '+fDrawdown5Yr,fontsize=8)
        plt.text(50,1000*10**(.79*plotrange),' 3 Yr  '+fSharpe3Yr+'  '+fReturn3Yr+'  '+fDrawdown3Yr,fontsize=8)
        plt.text(50,1000*10**(.75*plotrange),' 2 Yr  '+fSharpe2Yr+'  '+fReturn2Yr+'  '+fDrawdown2Yr,fontsize=8)
        plt.text(50,1000*10**(.71*plotrange),' 1 Yr  '+fSharpe1Yr+'  '+fReturn1Yr+'  '+fDrawdown1Yr,fontsize=8)

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

        plt.text(1500,1000*10**(.95*plotrange),'Period Sharpe AvgProfit  Avg DD',fontsize=7.5,color='b')
        plt.text(1500,1000*10**(.91*plotrange),'15 Yr '+fVSharpe15Yr+'  '+fVReturn15Yr+'  '+fVDrawdown15Yr,fontsize=8,color='b')
        plt.text(1500,1000*10**(.87*plotrange),'10 Yr '+fVSharpe10Yr+'  '+fVReturn10Yr+'  '+fVDrawdown10Yr,fontsize=8,color='b')
        plt.text(1500,1000*10**(.83*plotrange),' 5 Yr  '+fVSharpe5Yr+'  '+fVReturn5Yr+'  '+fVDrawdown5Yr,fontsize=8,color='b')
        plt.text(1500,1000*10**(.79*plotrange),' 3 Yr  '+fVSharpe3Yr+'  '+fVReturn3Yr+'  '+fVDrawdown3Yr,fontsize=8,color='b')
        plt.text(1500,1000*10**(.75*plotrange),' 2 Yr  '+fVSharpe2Yr+'  '+fVReturn2Yr+'  '+fVDrawdown2Yr,fontsize=8,color='b')
        plt.text(1500,1000*10**(.71*plotrange),' 1 Yr  '+fVSharpe1Yr+'  '+fVReturn1Yr+'  '+fVDrawdown1Yr,fontsize=8,color='b')

        if beatBuyHoldTest > 0. :
            plt.text(50,1000*10**(.65*plotrange),format(beatBuyHoldTest2,'.2%')+'  beats BuyHold...')
        else:
            plt.text(50,1000*10**(.65*plotrange),format(beatBuyHoldTest2,'.2%'))

        if beatBuyHoldTestVarPct > 0. :
            plt.text(50,1000*10**(.59*plotrange),format(beatBuyHoldTest2VarPct,'.2%')+'  beats BuyHold...',color='b')
        else:
            plt.text(50,1000*10**(.59*plotrange),format(beatBuyHoldTest2VarPct,'.2%'),color='b'
            )

        plt.text(50,1000*10**(.54*plotrange),last_symbols_text,fontsize=8)
        plt.plot(BuyHoldPortfolioValue,lw=3,c='r')
        plt.plot(np.mean(monthvalue,axis=0),lw=4,c='k')
        plt.plot(np.mean(monthvalueVariablePctInvest,axis=0),lw=2,c='b')
        # set up to use dates for labels
        xlocs = []
        xlabels = []
        for i in range(1,len(datearray)):
            if datearray[i].year != datearray[i-1].year:
                xlocs.append(i)
                xlabels.append(str(datearray[i].year))
        print("xlocs,xlabels = ", xlocs, xlabels)
        if len(xlocs) < 12 :
            plt.xticks(xlocs, xlabels)
        else:
            plt.xticks(xlocs[::2], xlabels[::2])
        plt.xlim(0,len(datearray))
        plt.ylim(1000,plotmax)

        plt.subplot(subplotsize[1])
        plt.grid(True)
        ##ylim(0, value.shape[0])
        plt.ylim(0, 1.2)

        n_stocks_uptrending_median = np.zeros((activeCount.size), 'float32')
        n_stocks_uptrending_near_high = np.zeros_like(n_stocks_uptrending_median)
        n_stocks_uptrending_beat_BH = np.zeros_like(n_stocks_uptrending_median)
        n_stocks_uptrending = np.zeros_like(n_stocks_uptrending_median)

        ii = np.where(activeCount > 0)[0]
        _dates = np.array(datearray)[ii]
        n_stocks_uptrending_median[ii] = numberStocksUpTrendingMedian[ii] / activeCount[ii]
        n_stocks_uptrending_near_high[ii] = numberStocksUpTrendingNearHigh[ii] / activeCount[ii]
        n_stocks_uptrending_beat_BH[ii] = numberStocksUpTrendingBeatBuyHold[ii] / activeCount[ii]
        n_stocks_uptrending[ii] = numberStocks[ii] / activeCount[ii]

        plt.plot(_dates, n_stocks_uptrending_median[ii],'g-',lw=1)
        plt.plot(_dates, n_stocks_uptrending_near_high[ii],'b-',lw=1)
        plt.plot(_dates, n_stocks_uptrending_beat_BH[ii],'k-',lw=2)
        plt.plot(_dates, n_stocks_uptrending[ii] ,'r-')

        plt.xlim(datearray[0],datearray[len(datearray)-1])

        plt.subplot(subplotsize[2])
        plt.grid(True)
        plt.plot(datearray,QminusSMA,'m-',lw=.8)
        plt.plot(datearray,monthPctInvested,'r-',lw=.8)
        plt.xlim(datearray[0],datearray[len(datearray)-1])
        plt.draw()
        # save figure to disk, but only if trades produce good results
        webpage_dir = get_webpage_store(json_fn)
        if iter==randomtrials-1:
            outputplotname = os.path.join(webpage_dir, 'PyTAAA_monteCarloBacktest.png' )
            plt.savefig(outputplotname, format='png', edgecolor='gray' )

        ###
        ### save backtest portfolio values ( B&H and system )
        ###
        try:
            filepath = os.path.join(p_store, "pyTAAAweb_backtestPortfolioValue.params" )
            textmessage = ""
            # Only write data for dates where new highs/lows are valid (>= 0)
            # This typically skips the first ~500 days where there isn't enough historical data
            for idate in range(len(BuyHoldPortfolioValue)):
                textmessage = textmessage + \
                            str(datearray[idate]) + " " + \
                            str(BuyHoldPortfolioValue[idate]) + " " + \
                            str(np.average(monthvalue[:,idate]))  + " " + \
                            f"{float(sumNewHighs[idate]):.1f}" + " " + \
                            f"{float(sumNewLows[idate]):.1f}" + "\n"
            with open( filepath, "w" ) as f:
                f.write(textmessage)
        except:
            _fn = os.path.join(p_store, "pyTAAAweb_backtestPortfolioValue.params")
            print(" ERROR: unable to update file " + _fn)


        ########################################################################
        ### compute some portfolio performance statistics and print summary
        ########################################################################

        print("final value for portfolio ", "{:,}".format(np.average(monthvalue[:,-1])))


        print("portfolio annualized gains : ", ( gmean(PortfolioDailyGains)**252 ))
        print("portfolio annualized StdDev : ", ( np.std(PortfolioDailyGains)*sqrt(252) ))
        print("portfolio sharpe ratio : ",PortfolioSharpe[iter])

        # Compute trading days back to target start date
        targetdate = datetime.date(2008,1,1)
        targetdate = datetime.date.today()-datetime.timedelta(365.25*3)
        lag = int((datearray[-1] - targetdate).days*252/365.25)

        # Print some stats for B&H and trading from target date to end_date
        print("")
        print("")
        BHValue = np.average(value,axis=0)
        BHdailygains = np.concatenate( (np.array([0.]), BHValue[1:]/BHValue[:-1]), axis = 0 )
        BHsharpefromtargetdate = ( gmean(BHdailygains[-lag:])**252 -1. ) / ( np.std(BHdailygains[-lag:])*sqrt(252) )
        BHannualgainfromtargetdate = ( gmean(BHdailygains[-lag:])**252 )
        print("start date for recent performance measures: ",targetdate)
        print("BuyHold annualized gains & sharpe from target date:   ", BHannualgainfromtargetdate,BHsharpefromtargetdate)

        Portfoliosharpefromtargetdate = ( gmean(PortfolioDailyGains[-lag:])**252 -1. ) / ( np.std(PortfolioDailyGains[-lag:])*sqrt(252) )
        Portfolioannualgainfromtargetdate = ( gmean(PortfolioDailyGains[-lag:])**252 )
        print("portfolio annualized gains & sharpe from target date: ", Portfolioannualgainfromtargetdate,Portfoliosharpefromtargetdate)

        csv_text = str(datearray[-1])+","+str(iter)+","+    \
                      str(numberStocksTraded)+","+   \
                      str(monthsToHold)+","+  \
                      str(LongPeriod)+","+   \
                      format(stddevThreshold,'4.2f')+","+   \
                      str(MA1)+","+   \
                      str(MA2)+","+   \
                      str(MA2+MA2offset)+","+   \
                      params['uptrendSignalMethod']+","+   \
                      str(lowPct)+","+   \
                      str(hiPct)+","+   \
                      str(riskDownside_min)+","+str(riskDownside_max)+","+   \
                      format(np.average(monthvalue[:,-1]), "10.0f")+','+   \
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
                      fSharpe6Mo+","+   \
                      fSharpe3Mo+","+   \
                      fSharpe1Mo+","+   \
                      fReturn15Yr+","+   \
                      fReturn10Yr+","+   \
                      fReturn5Yr+","+   \
                      fReturn3Yr+","+   \
                      fReturn2Yr+","+   \
                      fReturn1Yr+","+   \
                      fReturn6Mo+","+   \
                      fReturn3Mo+","+   \
                      fReturn1Mo+","+   \
                      fDrawdown15Yr+","+   \
                      fDrawdown10Yr+","+   \
                      fDrawdown5Yr+","+   \
                      fDrawdown3Yr+","+   \
                      fDrawdown2Yr+","+   \
                      fDrawdown1Yr+","+   \
                      fDrawdown6Mo+","+   \
                      fDrawdown3Mo+","+   \
                      fDrawdown1Mo+","+   \
                      format(beatBuyHoldTest,'5.3f')+","+\
                      format(beatBuyHoldTest2,'.2%')+","+\
                      str(paramNumberToVary)+\
                      " \n"


        print("\n\ncsv_text = ", csv_text)
        periodForSignal[iter] = LongPeriod

        ###
        ### save today's monte carlo backtest results to file ( B&H and system )
        ###
        try:
            filepath = os.path.join(p_store, "pyTAAAweb_backtestTodayMontecarloPortfolioValue.csv" )
            textmessage = ""
            for idate in range(len(BuyHoldPortfolioValue)):
                textmessage = textmessage + str(datearray[idate])+"  "+str(BuyHoldPortfolioValue[idate])+"  "+str(np.average(monthvalue[:,idate]))+"\n"
            if iter == 0 :
                csv_text = csv_text.replace(" \n","  ** Using pyTAAA parameters\n")
                with open( filepath, "a" ) as f:
                    f.write(csv_text)
            else:
                beatBuyHoldRecent = ( (Sharpe3Yr-BuyHoldSharpe3Yr)/15. + \
                                      (Sharpe2Yr-BuyHoldSharpe2Yr)/10. + \
                                      (Sharpe1Yr-BuyHoldSharpe1Yr)/5. + \
                                      (Sharpe6Mo-BuyHoldSharpe6Mo)/3. + \
                                      (Sharpe3Mo-BuyHoldSharpe3Mo)/2. + \
                                      (Sharpe1Mo-BuyHoldSharpe1Mo)/1. + \
                                      (Return3Yr-BuyHoldReturn3Yr)/15. + \
                                      (Return2Yr-BuyHoldReturn2Yr)/10. + \
                                      (Return1Yr-BuyHoldReturn1Yr)/5. + \
                                      (Return6Mo-BuyHoldReturn6Mo)/3. + \
                                      (Return3Mo-BuyHoldReturn3Mo)/2. + \
                                      (Return1Mo-BuyHoldReturn1Mo)/1. + \
                                      (Drawdown3Yr-BuyHoldDrawdown3Yr)/15. + \
                                      (Drawdown2Yr-BuyHoldDrawdown2Yr)/10. + \
                                      (Drawdown1Yr-BuyHoldDrawdown1Yr)/5. + \
                                      (Drawdown6Mo-BuyHoldDrawdown6Mo)/3. + \
                                      (Drawdown3Mo-BuyHoldDrawdown3Mo)/2. + \
                                      (Drawdown1Mo-BuyHoldDrawdown1Mo)/1. ) \
                                      / (3.*(1/15. + 1/10.+1/5.+1/3.+1/2.+1))
                beatPyTAARecent =   ( (Sharpe3Yr-Sharpe3Yr_PyTAAA)/15. + \
                                      (Sharpe2Yr-Sharpe2Yr_PyTAAA)/10. + \
                                      (Sharpe1Yr-Sharpe1Yr_PyTAAA)/5. + \
                                      (Sharpe6Mo-Sharpe6Mo_PyTAAA)/3. + \
                                      (Sharpe3Mo-Sharpe3Mo_PyTAAA)/2. + \
                                      (Sharpe1Mo-Sharpe1Mo_PyTAAA)/1. + \
                                      (Return3Yr-Return3Yr_PyTAAA)/15. + \
                                      (Return2Yr-Return2Yr_PyTAAA)/10. + \
                                      (Return1Yr-Return1Yr_PyTAAA)/5. + \
                                      (Return6Mo-Return6Mo_PyTAAA)/3. + \
                                      (Return3Mo-Return3Mo_PyTAAA)/2. + \
                                      (Return1Mo-Return1Mo_PyTAAA)/1. + \
                                      (Drawdown3Yr-Drawdown3Yr_PyTAAA)/15. + \
                                      (Drawdown2Yr-Drawdown2Yr_PyTAAA)/10. + \
                                      (Drawdown1Yr-Drawdown1Yr_PyTAAA)/5. + \
                                      (Drawdown6Mo-Drawdown6Mo_PyTAAA)/3. + \
                                      (Drawdown3Mo-Drawdown3Mo_PyTAAA)/2. + \
                                      (Drawdown1Mo-Drawdown1Mo_PyTAAA)/1. ) \
                                      / (3.*(1/15. + 1/10.+1/5.+1/3.+1/2.+1))
                print("\n\n ...beatBuyHoldRecent = ", beatBuyHoldRecent, "\n ...beatPyTAARecent = ", beatPyTAARecent, "\n\n")
                if beatBuyHoldRecent > 0. and beatPyTAARecent > 0.:
                    with open( filepath, "a" ) as f:
                        csv_text = csv_text.replace("\n",","+str(beatBuyHoldRecent)+","+str(beatPyTAARecent)+"\n")
                        csv_text = csv_text.replace("\n",","+str(last_symbols_text)+"\n")
                        f.write(csv_text)
        except:
            filepath = os.path.join(p_store, "pyTAAAweb_backtestTodayMontecarloPortfolioValue.csv" )
            print(" Error: unable to update file " + filepath)


        # create and update counter for holding period
        # with number of random trials choosing this symbol on last date times Sharpe ratio for trial in last year
        print("")
        print("")
        print("cumulative tally of holding periods for last date")
        if iter == 0:
            print("initializing cumulative talley of holding periods chosen for last date...")
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
                #############
                # Normalize holdmonthscountnorm, avoiding division by zero warnings.
                #############
                max_val = holdmonthscountnorm.max()
                holdmonthscountnorm = np.divide(
                    holdmonthscountnorm,
                    max_val,
                    out=np.zeros_like(holdmonthscountnorm),
                    where=max_val != 0
                )
            holdmonthscountint = np.round(holdmonthscountnorm*40).astype('int')
            holdmonthscountint[np.isnan(holdmonthscountint)] =0
            for ii in range(len(holdMonths)):
                if holdmonthscountint[ii] > 0:
                    tagnorm = "*"* holdmonthscountint[ii]
                    print(format(str(holdMonths[ii]),'7s')+   \
                          str(datearray[-1])+         \
                          format(holdmonthscount[ii],'7.2f'), tagnorm)

    ###
    ### make plot of performance since 1/1/2013 (date that real-time tracking was started)
    ###
    realtimestartdate = datetime.date(2013,1,1)
    import pdb
    daysCount =  np.zeros( len(datearray), 'int' )
    for ii in range( len(datearray) ):
        daysCount[ii] = (datearray[ii]-realtimestartdate).days
    #pdb.set_trace()
    indexRealtimeStart = np.argmax( np.clip(daysCount,-5000,1) ) - 1

    TradedPortfolioValue = np.average(monthvalue,axis=0)

    png_fn = 'PyTAAA_monteCarloBacktestFull'
    plotRecentPerfomance3( 0, datearray,
                          symbols,
                          value,
                          monthvalue,
                          AllStocksHistogram,
                          MonteCarloPortfolioValues,
                          FinalTradedPortfolioValue,
                          TradedPortfolioValue,
                          BuyHoldPortfolioValue,
                          numberStocksUpTrending,
                          last_symbols_text,
                          activeCount,
                          numberStocks,
                          numberStocksUpTrendingNearHigh,
                          numberStocksUpTrendingBeatBuyHold,
                          png_fn, json_fn)

    png_fn = 'PyTAAA_monteCarloBacktestRecent'
    plotRecentPerfomance3( indexRealtimeStart, datearray,
                          symbols,
                          value,
                          monthvalue,
                          AllStocksHistogram,
                          MonteCarloPortfolioValues,
                          FinalTradedPortfolioValue,
                          TradedPortfolioValue,
                          BuyHoldPortfolioValue,
                          numberStocksUpTrending,
                          last_symbols_text,
                          activeCount,
                          numberStocks,
                          numberStocksUpTrendingNearHigh,
                          numberStocksUpTrendingBeatBuyHold,
                          png_fn, json_fn)
    return
