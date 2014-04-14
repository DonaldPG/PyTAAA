import os
import numpy as np
import datetime
import platform
#from matplotlib import pylab as plt
# Force matplotlib to not use any Xwindows backend.
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import matplotlib.gridspec as gridspec
from functions.GetParams import GetEdition

def makeValuePlot(  ):

    ##########################################
    # read valuations status file and make plot
    ##########################################

    filepath = os.path.join( os.getcwd(), "PyTAAA_status.params" )
    figurepath = os.path.join( os.getcwd(), "pyTAAA_web", "PyTAAA_value.png" )

    # get edition from where software is running
    edition = GetEdition()

    date = []
    value = []
    try:
        with open( filepath, "r" ) as f:
            # get number of lines in file
            lines = f.read().split("\n")
            numlines = len (lines)
            for i in range(numlines):
                try:
                    statusline = lines[i]
                    statusline_list = (statusline.split("\r")[0]).split(" ")
                    if len( statusline_list ) == 4:
                        date.append( datetime.datetime.strptime( statusline_list[1], '%Y-%m-%d') )
                        value.append( float(statusline_list[3]) )
                except:
                    break

            #print "rankingMessage -----"
            #print rankingMessage
    except:
        print " Error: unable to read updates from PyTAAA_status.params"
        print ""

    value = np.array( value ).astype('float')

    for i in range(0,len(value),500 ):
        print "   ...inside makeValuePlot - i, date[i], value[i] = ", i, date[i], value[i]

    plt.figure(1,figsize=(9,7))
    plt.clf()
    plt.grid(True)
    plt.plot( date, value )
    plt.xlim((date[0],date[-1]+datetime.timedelta(1) ))
    plt.title("pyTAAA Value History Plot ("+edition+" edition)")
    # put text line with most recent date at bottom of plot
    # - get 5% of x-scale and y-scale for text location
    x_range = date[-1] - date[0]
    text_x = date[0] + datetime.timedelta( x_range.days / 20. )
    text_y = ( np.max(value) - np.min(value) )* .05 + np.min(value)
    plt.text( text_x,text_y, "most recent value from "+str(date[-1].date())+"\nplotted at "+datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"), fontsize=8 )
    plt.savefig(figurepath)
    try:
        plt.close(1)
    except:
        pass

    figurepath = 'PyTAAA_value.png'  # re-set to name without full path
    ( _, figurepath ) = os.path.split( figurepath )  # re-set to name without full path
    figure_htmlText = "\n<br><h3>Plot of model performance since inception</h3>\n"
    figure_htmlText = figure_htmlText + "\nDoes not include slippage, fees, dividends, etc\n"
    figure_htmlText = figure_htmlText + '''<br><img src="'''+figurepath+'''" alt="PyTAAA by DonaldPG" width="850" height="500"><br>\n'''

    return figure_htmlText


def makeUptrendingPlot( ):

    from functions.TAfunctions import SMA, MoveMax, dpgchannel

    ##########################################
    # read uptrending stocks status file
    ##########################################

    file2path = os.path.join( os.getcwd(), "pyTAAAweb_numberUptrendingStocks_status.params" )

    date = []
    value = []
    active = []
    try:
        with open( file2path, "r" ) as f:
            # get number of lines in file
            lines = f.read().split("\n")
            numlines = len (lines)
            for i in range(numlines):
                try:
                    statusline = lines[i]
                    statusline_list = statusline.split(" ")
                    if len( statusline_list ) == 5:
                        date.append( datetime.datetime.strptime( statusline_list[0], '%Y-%m-%d') )
                        value.append( float(statusline_list[2]) )
                        active.append( float(statusline_list[4]) )
                except:
                    break
    except:
        print " Error: unable to read updates from pyTAAAweb_numberUptrendingStocks_status.params"
        print ""

    ##########################################
    # read multi-Sharpe signal status file
    ##########################################

    file2path = os.path.join( os.getcwd(), "pyTAAAweb_multiSharpeIndicator_status.params" )

    dates = []
    medianSharpe = []
    signalSharpe = []
    try:
        with open( file2path, "r" ) as f:
            # get number of lines in file
            lines = f.read().split("\n")
            numlines = len (lines)
            for i in range(numlines):
                try:
                    statusline = lines[i]
                    statusline_list = statusline.split(" ")
                    if len( statusline_list ) == 5:
                        dates.append( datetime.datetime.strptime( statusline_list[0], '%Y-%m-%d') )
                        medianSharpe.append( float(statusline_list[2]) )
                        signalSharpe.append( float(statusline_list[4]) )
                except:
                    break
    except:
        print " Error: unable to read updates from pyTAAAweb_multiSharpeIndicator_status.params"
        print ""

    ##########################################
    # make plot
    ##########################################

    figure3path = os.path.join( os.getcwd(), "pyTAAA_web", "PyTAAA_numberUptrendingStocks.png" )

    value = np.array( value ).astype('float') / np.array( active ).astype('float')
    valueSMA = SMA( value, 100 )
    valueMMA = MoveMax( valueSMA, 252 )
    valueSMA2 = SMA( value, 500 )
    valueMMA2 = MoveMax( valueSMA2, 252 )
    value_channelMin, value_channelMax = dpgchannel( value, 3,13,2 )
    value_MidChannel = ( value_channelMin + value_channelMax ) /2.
    PctInvested = 1.0+(value_MidChannel-valueSMA+value-valueSMA2)-(valueMMA2-valueSMA2 + valueMMA-valueSMA)

    plt.figure(2,figsize=(9,7))
    plt.clf()
    plt.grid(True)
    numDaysToPlot = 252*10
    plt.plot( date[-numDaysToPlot:], .97 * valueSMA[-numDaysToPlot:], 'r-', lw=1, label='pct in uptrend - short trend 1')
    plt.plot( date[-numDaysToPlot:], .97 * valueMMA[-numDaysToPlot:], 'r-', lw=.25, label='pct in uptrend - short trend 1')
    plt.plot( date[-numDaysToPlot:], valueSMA2[-numDaysToPlot:], 'b-', lw=1, label='pct in uptrend - short trend 2')
    plt.plot( date[-numDaysToPlot:], valueMMA2[-numDaysToPlot:], 'b-', lw=.25, label='pct in uptrend - short trend 2')
    plt.plot( date[-numDaysToPlot:], value[-numDaysToPlot:], 'k-', lw=.35, label='pct of stocks in uptrend')
    plt.plot( date[-numDaysToPlot:], np.clip( np.around( PctInvested[-numDaysToPlot:], decimals=1), 0., 1.2 ) , 'g-', lw=1.5, label='Percent to Invest (rounded)')
    plt.plot( date[-numDaysToPlot:], np.clip( np.around( PctInvested[-numDaysToPlot:], decimals=1), 0., 1.2 ) , '-', lw=1.5, c=(1,1,1,.8), label='Percent to Invest (rounded)')
    plt.plot( date[-numDaysToPlot:], PctInvested[-numDaysToPlot:], 'g-', lw=.5, label='Percent to Invest')
    plt.ylim( 0.0, 1.25 )
    plt.legend(loc=3,prop={'size':9})
    plt.title("pyTAAA History Plot\nPercent Uptrending Stocks ")
    plt.savefig(figure3path)
    figure3path = 'PyTAAA_numberUptrendingStocks.png'  # re-set to name without full path
    figure3_htmlText = "\n<br><h3>Percentage of stocks uptrending</h3>\n"
    figure3_htmlText = figure3_htmlText + "\nPlot shows percent of uptrending stocks in Nasdaq 100 (black line), 97% of 100 day moving average (red line), and 500 day moving average (blue line)\n"
    figure3_htmlText = figure3_htmlText + '''<br><img src="'''+figure3path+'''" alt="PyTAAA by DonaldPG" width="850" height="500"><br>\n'''

    return figure3_htmlText



def makeTrendDispersionPlot( ):

    from functions.TAfunctions import SMA, MoveMax
    import functions.allstats

    ##########################################
    # read uptrending stocks status file and make plot
    ##########################################
    file4path = os.path.join( os.getcwd(), "pyTAAAweb_MeanTrendDispersion_status.params" )
    figure4path = os.path.join( os.getcwd(), "pyTAAA_web", "PyTAAA_MeanTrendDispersion.png" )

    dateMedians = []
    valueMeans = []
    valueMedians = []
    try:
        with open( file4path, "r" ) as f:
            # get number of lines in file
            lines = f.read().split("\n")
            numlines = len (lines)
            for i in range(numlines):
                try:
                    statusline = lines[i]
                    statusline_list = statusline.split(" ")
                    if len( statusline_list ) == 5:
                        dateMedians.append( datetime.datetime.strptime( statusline_list[0], '%Y-%m-%d') )
                        valueMeans.append( float(statusline_list[2]) )
                        valueMedians.append( float(statusline_list[4]) )
                except:
                    break
    except:
        print " Error: unable to read updates from pyTAAAweb_MeanTrendDispersion_status.params"
        print ""

    valueMeans = np.array( valueMeans ).astype('float')
    valueMeansSMA = SMA( valueMeans, 100 )
    valueMedians = np.array( np.clip(valueMedians,-25.,25. ) ).astype('float')
    valueMediansSMA = SMA( valueMedians, 100 )

    plt.figure(4,figsize=(9,7))
    plt.clf()
    plt.grid(True)
    numDaysToPlot = 252*10
    numDaysToPlot = len( valueMeans )
    plt.plot( dateMedians[-numDaysToPlot:], valueMediansSMA[-numDaysToPlot:], 'r-', lw=1)
    plt.plot( dateMedians[-numDaysToPlot:], valueMedians[-numDaysToPlot:], 'r-', lw=.25)
    plt.title("pyTAAA History Plot\n Average Trend Dispersion")
    plt.savefig(figure4path)
    figure4path = 'PyTAAA_MeanTrendDispersion.png'  # re-set to name without full path
    figure4_htmlText = "\n<br><h3>Average Trend Dispersion</h3>\n"
    figure4_htmlText = figure4_htmlText + "\nPlot shows dispersion of trend for stocks in Nasdaq 100 (thin line), 100 day moving average (thick line).\n"
    figure4_htmlText = figure4_htmlText + "\nBlack lines use means, red lines use average sharpe.\n"
    figure4_htmlText = figure4_htmlText + '''<br><img src="'''+figure4path+'''" alt="PyTAAA by DonaldPG" width="850" height="500"><br>\n'''

    ###
    ### make a combined plot
    ### 1. get percent of uptrending stocks
    ###
    file2path = os.path.join( os.getcwd(), "pyTAAAweb_numberUptrendingStocks_status.params" )
    date = []
    value = []
    active = []
    try:
        with open( file2path, "r" ) as f:
            # get number of lines in file
            lines = f.read().split("\n")
            numlines = len (lines)
            for i in range(numlines):
                try:
                    statusline = lines[i]
                    statusline_list = statusline.split(" ")
                    if len( statusline_list ) == 5:
                        date.append( datetime.datetime.strptime( statusline_list[0], '%Y-%m-%d') )
                        value.append( float(statusline_list[2]) )
                        active.append( float(statusline_list[4]) )
                except:
                    break
    except:
        print " Error: unable to read updates from pyTAAAweb_numberUptrendingStocks_status.params"
        print ""

    value = np.array( value ).astype('float') / np.array( active ).astype('float')
    valueSMA = SMA( value, 100 )
    valueMMA = MoveMax( valueSMA, 252 )
    valueSMA2 = SMA( value, 500 )
    valueMMA2 = MoveMax( valueSMA2, 252 )
    PctInvested = 1.0+(value-valueSMA+value-valueSMA2)-(valueMMA2-valueSMA2 + valueMMA-valueSMA)

    valueMediansSMA65 = SMA( valueMedians, 65 )
    valueMediansSMA100 = SMA( valueMedians, 100 )
    valueMediansSMA65 = ( valueMediansSMA65 - valueMediansSMA65.mean() ) * 8. + .7
    valueMediansSMA100 = ( valueMediansSMA100 - valueMediansSMA100.mean() ) * 8. + .7

    file3path = os.path.join( os.getcwd(), "pyTAAAweb_backtestPortfolioValue.params" )
    backtestDate = []
    backtestBHvalue = []
    backtestSystemvalue = []
    try:
        with open( file3path, "r" ) as f:
            # get number of lines in file
            lines = f.read().split("\n")
            numlines = len (lines)
            for i in range(numlines):
                try:
                    statusline = lines[i]
                    statusline_list = statusline.split(" ")
                    if len( statusline_list ) == 5:
                        backtestDate.append( datetime.datetime.strptime( statusline_list[0], '%Y-%m-%d') )
                        backtestBHvalue.append( float(statusline_list[2]) )
                        backtestSystemvalue.append( float(statusline_list[4]) )
                except:
                    break
    except:
        print " Error: unable to read updates from pyTAAAweb_backtestPortfolioValue.params"
        print ""


    ###
    ### make a combined plot
    ### 2. make plot showing trend below B&H and trade-system Value
    ###
    figure5path = os.path.join( os.getcwd(), "pyTAAA_web", "PyTAAA_backtestWithTrend.png" )
    plt.figure(5,figsize=(9,7))
    plt.clf()
    subplotsize = gridspec.GridSpec(2,1,height_ratios=[5,3])
    plt.subplot(subplotsize[0])
    plt.grid(True)
    plt.yscale('log')
    plotmax = 1.e9
    plt.ylim([1000,max(10000,plotmax)])
    numDaysToPlot = 252*10
    numDaysToPlot = len( backtestBHvalue )
    plt.plot( backtestDate[-numDaysToPlot:], backtestBHvalue[-numDaysToPlot:], 'r-', lw=1.25, label='Buy & Hold')
    plt.plot( backtestDate[-numDaysToPlot:], backtestSystemvalue[-numDaysToPlot:], 'k-', lw=1.25, label='Trading System')
    plt.legend(loc=2,prop={'size':9})
    plt.title("pyTAAA History Plot\n Portfolio Value")
    plt.text( backtestDate[-numDaysToPlot+50], 2500, "Backtest updated "+datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"), fontsize=7.5 )
    plt.subplot(subplotsize[1])
    plt.grid(True)
    plt.ylim(0, 1.2)
    numDaysToPlot = 252*10
    numDaysToPlot = len( value )
    plt.plot( date[-numDaysToPlot:], value[-numDaysToPlot:], 'k-', lw=.25, label='Percent Uptrending')
    plt.plot( date[-numDaysToPlot:], np.clip(PctInvested[-numDaysToPlot:],0.,1.2), 'g-', alpha=.65, lw=.5, label='Percent to Invest')
    numDaysToPlot = len( valueMedians )
    plt.legend(loc=3,prop={'size':6})
    plt.savefig(figure5path)
    figure5path = 'PyTAAA_backtestWithTrend.png'  # re-set to name without full path
    figure5_htmlText = "\n<br><h3>Daily backtest with trend indicators</h3>\n"
    figure5_htmlText = figure5_htmlText + "\nCombined backtest with Trend indicators.\n"
    figure5_htmlText = figure5_htmlText + '''<br><img src="'''+figure5path+'''" alt="PyTAAA by DonaldPG" width="850" height="500"><br>\n'''

    return figure5_htmlText


def makeDailyMonteCarloBacktest( ):

    import datetime
    from functions.dailyBacktest_pctLong import *

    ##########################################
    # make plot with daily monte carlo backtest
    ##########################################
    figure6path = os.path.join( os.getcwd(), 'pyTAAA_web', 'PyTAAA_monteCarloBacktest.png' )
    
    ###
    ### make a combined plot
    ### - only update between midnight and 2 a.m.
    ### - make plot showing trend below B&H and trade-system Value
    ###
    
    today = datetime.datetime.now()
    hourOfDay = today.hour
    
    if hourOfDay < 3:
		dailyBacktest_pctLong()
    
    figure6path = 'PyTAAA_monteCarloBacktest.png'  # re-set to name without full path
    figure6_htmlText = "\n<br><h3>Daily backtest with trend indicators and measure of invested percent</h3>\n"
    figure6_htmlText = figure6_htmlText + "\nCombined backtest with Trend indicators.\n"
    figure6_htmlText = figure6_htmlText + '''<br><img src="'''+figure6path+'''" alt="PyTAAA by DonaldPG" width="850" height="500"><br>\n'''

    return figure6_htmlText
