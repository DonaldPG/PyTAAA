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
from functions.UpdateSymbols_inHDF5 import (
    loadQuotes_fromHDF
)
from functions.allstats import *
from functions.dailyBacktest import computeDailyBacktest
from functions.TAfunctions import (
    computeSignal2D,
    interpolate,
    cleantobeginning,
    cleantoend,
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
    print("   . filename for loadQuotes_fromHDF = " + filename)
    adjClose, symbols, datearray, _, _ = loadQuotes_fromHDF(filename, json_fn)
    for i in range(adjClose.shape[0]):
        adjClose[i,:] = interpolate(adjClose[i,:])
        adjClose[i,:] = cleantobeginning(adjClose[i,:])
        adjClose[i,:] = cleantoend(adjClose[i,:])

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
                print("date, signal2D changed",datearray[jj])

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
    
    web_dir = get_webpage_store(json_fn)
    try:
        filepath = os.path.join(web_dir, "pyTAAAweb_numberUptrendingStocks_status.params")
        textmessage = ""
        for jj in range(dailyNumberUptrendingStocks.shape[0]):
            textmessage = textmessage + str(datearray[jj])+"  "+str(dailyNumberUptrendingStocks[jj])+"  "+str(activeCount[jj])+"\n"
        with open( filepath, "w" ) as f:
            f.write(textmessage)
        print(" Successfully updated to pyTAAAweb_numberUptrendingStocks_status.params at ", datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
        print("")
    except :
        print(" Error: unable to update pyTAAAweb_numberUptrendingStocks_status.params")
        print("")


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

    import functions.ystockquote as ysq
    # import matplotlib
    # matplotlib.use('Agg')
    # from matplotlib import pylab as plt
    filepath = get_webpage_store(json_fn)
    # filepath = os.path.join(json_dir, "pyTAAA_web")

    today = datetime.datetime.now()
    hourOfDay = today.hour

    if hourOfDay >= 1  or 11<hourOfDay<13:
        for ii in range( len(datearray) ):
            if datearray[ii].year > datearray[ii-1].year and datearray[ii].year == 2013:
                firstdate_index = ii
                break
        for i in range( len(symbols) ) :
            # get 'despiked' quotes for this symbol

            # check recency of plot file and skip if less than 20 hours old
            # Get the modification time in seconds since the epoch
            plotfilepath = os.path.join(
                filepath, "0_"+symbols[i]+".png"
            )
            if os.path.isfile(plotfilepath):
                mtime = datetime.datetime.fromtimestamp(
                    os.path.getmtime(plotfilepath)
                )
                # Convert to elpased time in hours
                # modified_time = (datetime.datetime.now() - mtime).seconds
                # modified_hours = modified_time / (60. * 60.)
                modified_time = (datetime.datetime.now() - mtime)
                modified_hours = modified_time.days * 24 + modified_time.seconds / 3600
                if modified_hours < 20.0:
                    continue

            quotes = adjClose[i,:].copy()
            quotes = quotes.reshape(1,len(quotes))
            quotes_despike = despike_2D( quotes, LongPeriod, stddevThreshold=stddevThreshold )
            # '''
            # try:
            #     plt.clf()
            #     plt.grid(True)
            #     plt.plot(datearray,adjClose[i,:])
            #     plt.plot(datearray,signal2D[i,:]*adjClose[i,-1],lw=.2)
            #     plt.plot(datearray,quotes_despike[0,:])
            #     if params['uptrendSignalMethod'] == 'percentileChannels':
            #         plt.plot(datearray,lowChannel[i,:],'m-')
            #         plt.plot(datearray,hiChannel[i,:],'m-')
            #     plot_text = str(adjClose[i,-7:])
            #     plt.text(datearray[50],0,plot_text)
            #     # put text line with most recent date at bottom of plot
            #     # - get 7.5% of x-scale and y-scale for text location
            #     x_range = datearray[-1] - datearray[0]
            #     text_x = datearray[0] + datetime.timedelta( x_range.days / 20. )
            #     text_y = ( np.max(adjClose[i,:]) - np.min(adjClose[i,:]) )* .085 + np.min(adjClose[i,:])
            #     plt.text( text_x,text_y, "most recent value from "+str(datearray[-1])+"\nplotted at "+today.strftime("%A, %d. %B %Y %I:%M%p")+"\nvalue = "+str(adjClose[i,-1]), fontsize=8 )
            #     plt.title(symbols[i]+"\n"+ysq.get_company_name(symbols[i]))
            #     plotfilepath = os.path.join( filepath, "0_"+symbols[i]+".png" )
            #     print " ...inside PortfolioPerformancealcs... plotfilepath = ", plotfilepath
            #     plt.savefig( plotfilepath, format='png' )
            # except:
            #     pass
            # '''
            plt.clf()
            plt.grid(True)
            plt.plot(datearray,adjClose[i,:])
            plt.plot(datearray,signal2D[i,:]*adjClose[i,-1],lw=.2)
            despiked_quotes = quotes_despike[0,:]
            number_nans = despiked_quotes[np.isnan(despiked_quotes)].shape[0]
            if number_nans == 0:
                plt.plot(datearray,quotes_despike[0,:])
            if params['uptrendSignalMethod'] == 'percentileChannels':
                plt.plot(datearray,lowChannel[i,:],'m-')
                plt.plot(datearray,hiChannel[i,:],'m-')
            plot_text = str(adjClose[i,-7:])
            plt.text(datearray[50],0,plot_text)
            # put text line with most recent date at bottom of plot
            # - get 7.5% of x-scale and y-scale for text location
            x_range = datearray[-1] - datearray[0]
            text_x = datearray[0] + datetime.timedelta( x_range.days / 20. )
            adjClose_noNaNs = adjClose[i,:].copy()
            adjClose_noNaNs = adjClose_noNaNs[~np.isnan(adjClose_noNaNs)]
            text_y = ( np.max(adjClose_noNaNs) - np.min(adjClose_noNaNs) )* .085 + np.min(adjClose_noNaNs)
            plt.text( text_x,text_y, "most recent value from "+str(datearray[-1])+"\nplotted at "+today.strftime("%A, %d. %B %Y %I:%M%p")+"\nvalue = "+str(adjClose[i,-1]), fontsize=8 )
            plt.title(symbols[i]+"\n"+ysq.get_company_name(symbols[i]))
            plt.yscale('log')
            plotfilepath = os.path.join( filepath, "0_"+symbols[i]+".png" )
            print(" ...inside PortfolioPerformancealcs... plotfilepath = ", plotfilepath)
            plt.savefig( plotfilepath, format='png' )


        for i in range( len(symbols) ) :

            # check recency of plot file and skip if less than 20 hours old
            # Get the modification time in seconds since the epoch
            plotfilepath = os.path.join(
                filepath, "0_recent_"+symbols[i]+".png"
            )
            if os.path.isfile(plotfilepath):
                mtime = datetime.datetime.fromtimestamp(
                    os.path.getmtime(plotfilepath)
                )
                # Convert to elpased time in hours
                # modified_time = (datetime.datetime.now() - mtime).seconds
                # modified_hours = modified_time / (60. * 60.)
                modified_time = (datetime.datetime.now() - mtime)
                modified_hours = modified_time.days * 24 + modified_time.seconds / 3600
                if modified_hours < 20.0:
                    continue
            # mtime = datetime.datetime.fromtimestamp(
            #     os.path.getmtime(plotfilepath)
            # )
            # # Convert to elpased time in hours
            # modified_time = (datetime.datetime.now() - mtime).seconds
            # modified_hours = modified_time / (60. * 60.)
            # if modified_hours < 20.0:
            #     continue

            # fit short-term recent trend channel for plotting
            quotes = adjClose[i,:].copy()
            quotes = quotes.reshape(1,len(quotes))
            quotes_despike = despike_2D( quotes, LongPeriod, stddevThreshold=stddevThreshold )
            # re-scale to have same value at beginning of plot
            quotes_despike *= quotes[0,firstdate_index]/quotes_despike[0,firstdate_index]

            # '''
            # upperFit, lowerFit = recentChannelFit( adjClose[i,:],
            #                                        minperiod=params['minperiod'],
            #                                        maxperiod=params['maxperiod'],
            #                                        incperiod=params['incperiod'],
            #                                        numdaysinfit=params['numdaysinfit'],
            #                                        offset=params['offset'])
            # '''

            lowerTrend, upperTrend, NoGapLowerTrend, NoGapUpperTrend = \
                     recentTrendAndMidTrendChannelFitWithAndWithoutGap( \
                                   adjClose[i,:], \
                                   minperiod=params['minperiod'], \
                                   maxperiod=params['maxperiod'], \
                                   incperiod=params['incperiod'], \
                                   numdaysinfit=params['numdaysinfit'], \
                                   numdaysinfit2=params['numdaysinfit2'], \
                                   offset=params['offset'])


            #recentFitDates = datearray[-params['numdaysinfit']-params['offset']:-params['offset']+1]
            #recentFitDates2 = datearray[-params['numdaysinfit']:]
            # '''
            # upperTrend = []
            # lowerTrend = []
            # for iii in range(-params['numdaysinfit']-params['offset'],-params['offset']+1):
            #     p = np.poly1d(upperFit)
            #     upperTrend.append( p(iii) )
            #     p = np.poly1d(lowerFit)
            #     lowerTrend.append( p(iii) )
            # '''


            try:
                # plot recent (about 2 years) performance for each symbol in stock universe
                # plt.figure(10,figsize=(9,5))
                plt.figure(10)
                plt.clf()
                plt.grid(True)
                #plt.plot(datearray[firstdate_index:],adjClose[i,firstdate_index:])
                plt.plot(datearray[firstdate_index:],signal2D[i,firstdate_index:]*adjClose[i,-1],lw=.25,alpha=.6)
                plt.plot(datearray[firstdate_index:],signal2D_daily[i,firstdate_index:]*adjClose[i,-1],lw=.25,alpha=.6)
                plt.plot(datearray[firstdate_index:],quotes_despike[0,firstdate_index:],lw=.15)
                adjClose_noNaNs = adjClose[i,:].copy()
                adjClose_noNaNs = adjClose_noNaNs[~np.isnan(adjClose_noNaNs)]
                ymax = np.around(np.max(adjClose_noNaNs[firstdate_index:]) * 1.1)
                if params['uptrendSignalMethod'] == 'percentileChannels':
                    ymin = np.around(np.min(lowChannel[i,firstdate_index:]) * 0.85)
                else:
                    ymin = np.around(np.min(adjClose[i,firstdate_index:]) * 0.90)
                plt.ylim((ymin,ymax))
                xmin = datearray[firstdate_index]
                xmax = datearray[-1] + datetime.timedelta( 10 )
                plt.xlim((xmin,xmax))
                if params['uptrendSignalMethod'] == 'percentileChannels':
                    plt.plot(datearray[firstdate_index:],lowChannel[i,firstdate_index:],'m-')
                    plt.plot(datearray[firstdate_index:],hiChannel[i,firstdate_index:],'m-')
                #plt.plot(recentFitDates,upperTrend,'y-')
                #plt.plot(recentFitDates,lowerTrend,'y-')

                relativedates = list(range(-params['numdaysinfit']-params['offset'],-params['offset']+1))
                plt.plot(np.array(datearray)[relativedates],upperTrend,'y-',lw=.5)
                plt.plot(np.array(datearray)[relativedates],lowerTrend,'y-',lw=.5)
                #plt.plot(datearray[np.array(relativedates)],upperTrend,'y-')
                #plt.plot(datearray[np.array(relativedates)],lowerTrend,'y-')
                plt.plot([datearray[-1]],[(upperTrend[-1]+lowerTrend[-1])/2.],'y.',ms=10,alpha=.6)
                plt.plot(np.array(datearray)[-params['numdaysinfit2']:],NoGapUpperTrend,ls='-',c=(0,0,1),lw=1.)
                plt.plot(np.array(datearray)[-params['numdaysinfit2']:],NoGapLowerTrend,ls='-',c=(0,0,1),lw=1.)
                plt.plot([datearray[-1]],[(NoGapUpperTrend[-1]+NoGapLowerTrend[-1])/2.],'.',c=(0,0,1),ms=10,alpha=.6)

                plt.plot(datearray[firstdate_index:],adjClose[i,firstdate_index:],'k-',lw=.5)

                plot_text = str(adjClose[i,-7:])
                plt.text(datearray[firstdate_index+10],ymin,plot_text, fontsize=10)
                # put text line with most recent date at bottom of plot
                # - get 7.5% of x-scale and y-scale for text location
                x_range = datearray[-1] - datearray[firstdate_index]
                text_x = datearray[firstdate_index] + datetime.timedelta( x_range.days / 20. )
                text_y = ( np.max(adjClose_noNaNs) - np.min(adjClose_noNaNs) )* .085 + np.min(adjClose_noNaNs)
                text_y = ( ymax - ymin )* .085 + ymin
                plt.text( text_x,text_y, "most recent value from "+str(datearray[-1])+"\nplotted at "+today.strftime("%A, %d. %B %Y %I:%M%p")+"\nvalue = "+str(adjClose[i,-1]), fontsize=8 )
                plt.title(symbols[i]+"\n"+ysq.get_company_name(symbols[i]))
                # We change the fontsize of minor ticks label
                plt.tick_params(axis='both', which='major', labelsize=8)
                plt.tick_params(axis='both', which='minor', labelsize=6)
                plotfilepath = os.path.join( filepath, "0_recent_"+symbols[i]+".png" )
                print(" ...inside PortfolioPerformancealcs... plotfilepath = ", plotfilepath)
                plt.savefig( plotfilepath, format='png' )
            except:
                print(" ERROR in PortfoloioPerformanceCalcs -- no plot generated for symbol ", symbols[i])
                pass

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
        stddevThreshold=stddevThreshold, makeQCPlots=True
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

