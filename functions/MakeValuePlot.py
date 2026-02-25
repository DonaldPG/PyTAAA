import os
import subprocess
import sys
import numpy as np
import datetime
#import platform
# Force matplotlib to not use any Xwindows backend.
import matplotlib
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
from matplotlib import pylab as plt
# Set DPI for inline plots and saved figures
plt.rcParams['figure.figsize'] = (9, 7)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
import matplotlib.gridspec as gridspec
from functions.GetParams import (
    get_json_params, get_symbols_file, GetEdition, get_performance_store
)
from functions.TAfunctions import dpgchannel, SMA
from functions.GetParams import get_webpage_store, get_web_output_dir


def makeValuePlot(json_fn: str) -> None:

    ##########################################
    # read valuations status file and make plot
    ##########################################

    json_folder = os.path.split(json_fn)[0]
    p_store = get_performance_store(json_fn)
    webpage_dir = get_webpage_store(json_fn)

    filepath = os.path.join(p_store, "PyTAAA_status.params" )
    figurepath = os.path.join(webpage_dir, "PyTAAA_value.png" )

    print(" ... inside makeValuePlot. '.params' filepath = " + filepath)

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
                    if len( statusline_list ) >= 4:
                        date.append( datetime.datetime.strptime( statusline_list[1], '%Y-%m-%d') )
                        value.append( float(statusline_list[3]) )
                except Exception as e:
                    if statusline.strip():  # Only print if line is not empty
                        print(f" Warning: Error parsing line {i} in PyTAAA_status.params: {e}")
                        print(f"   Line content: {statusline[:100]}")
                    break

            #print "rankingMessage -----"
            #print rankingMessage
    except:
        print(" Error: unable to read updates from PyTAAA_status.params")
        print("")

    value = np.array( value ).astype('float')

    # calculate mid-channel and compare to MA
    dailyValue = [ value[-1] ]
    dailyDate = [ date[-1] ]
    for ii in range( len(value)-2, 0, -1 ):
        if date[ii] != date[ii+1]:
            dailyValue.append( value[ii] )
            dailyDate.append( date[ii] )
    sortindices = (np.array( dailyDate )).argsort()
    sortedDailyValue = (np.array( dailyValue ))[ sortindices ]
    sortedDailyDate = (np.array( dailyDate ))[ sortindices ]

    minchannel, maxchannel = dpgchannel( sortedDailyValue, 5, 18, 4 )
    midchannel = ( minchannel + maxchannel )/2.
    MA_midchannel = SMA( midchannel, 5 )

    # create signal 11,000 for 'long' and 10,001 for 'cash'
    signal = np.ones_like( sortedDailyValue ) * 11000.
    signal[ MA_midchannel > midchannel ] = 10001
    signal[0] = 11000.

    # calculate value only when signal is 11,000
    valueSignal = np.zeros_like( signal )
    valueSignal[:5] = sortedDailyValue[:5]
    for ii in range( 4, len(signal) ):
        # delay signal usage by one day compared to computation
        if signal[ii-1] == 11000.:
            valueSignal[ii] = sortedDailyValue[ii]/sortedDailyValue[ii-1]*valueSignal[ii-1]
        else:
            valueSignal[ii] = valueSignal[ii-1]

    # for i in range(0,len(value),500 ):
    #     print("   ...inside makeValuePlot - i, date[i], value[i] = ", i, date[i], value[i])

    plt.figure(1,figsize=(9,7))
    plt.clf()
    plt.grid(True)

    # set up to use dates for labels
    xlocs = []
    xlabels = []
    for i in range(1,len(date)):
        if date[i].year != date[i-1].year:
            xlocs.append(date[i])
            xlabels.append(str(date[i].year))
    if len(xlocs) < 12 :
        plt.xticks(np.array(xlocs), np.array(xlabels))
    else:
        plt.xticks(xlocs[::2], xlabels[::2])

    plt.plot( sortedDailyDate, sortedDailyValue )
    plt.plot( sortedDailyDate, midchannel )
    plt.plot( sortedDailyDate, MA_midchannel )
    plt.plot( sortedDailyDate, signal )
    plt.plot( sortedDailyDate, valueSignal, 'k-', lw=.75)

    params = get_json_params(json_fn)
    if params['stockList'] == 'Naz100' and params['uptrendSignalMethod'] == "HMAs":
        switch_date = datetime.date(2024,9,7)
        switch_date = datetime.datetime(
            switch_date.year, switch_date.month, switch_date.day
        )
        switch_value_index = np.argmin(np.abs(sortedDailyDate-switch_date))
        plt.plot(
            [switch_date, switch_date],
            [sortedDailyValue[switch_value_index]//2., sortedDailyValue.max()],
            'g-', lw=.75
        )
        plt.text(
            switch_date, sortedDailyValue[switch_value_index]//2.,
            "switch to HMAs",
            rotation=90,
            horizontalalignment = 'left',
            verticalalignment='top'
        )

    if params['stockList'] == 'SP500' and params['uptrendSignalMethod'] == "HMAs":
        switch_date = datetime.date(2025,1,1)
        switch_date = datetime.datetime(
            switch_date.year, switch_date.month, switch_date.day
        )
        switch_value_index = np.argmin(np.abs(sortedDailyDate-switch_date))
        plt.plot(
            [switch_date, switch_date],
            [sortedDailyValue[switch_value_index]//2., sortedDailyValue.max()],
            'g-', lw=.75
        )
        plt.text(
            switch_date, sortedDailyValue[switch_value_index]//2.,
            "switch to HMAs",
            rotation=90,
            horizontalalignment = 'left',
            verticalalignment='top'
        )

    if params['stockList'] == 'SP500' and params['uptrendSignalMethod'] == "percentileChannels":
        switch_date = datetime.date(2025,12,12)
        switch_date = datetime.datetime(
            switch_date.year, switch_date.month, switch_date.day
        )
        switch_value_index = np.argmin(np.abs(sortedDailyDate-switch_date))
        plt.plot(
            [switch_date, switch_date],
            [sortedDailyValue[switch_value_index]//2., sortedDailyValue.max()],
            'g-', lw=.75
        )
        plt.text(
            switch_date, sortedDailyValue[switch_value_index]//2.,
            "switch to percentileChannels",
            rotation=90,
            horizontalalignment = 'left',
            verticalalignment='top'
        )

    plt.xlim((date[0],date[-1]+datetime.timedelta(10) ))
    plt.title("pyTAAA Value History Plot ("+edition+" edition)")
    # put text line with most recent date at bottom of plot
    # - get 5% of x-scale and y-scale for text location
    x_range = date[-1] - date[0]
    text_x = date[0] + datetime.timedelta( x_range.days / 20. )
    text_y = ( np.max(value) - np.min(value) )* .05 + np.min(value)
    plt.text( text_x,text_y, "most recent value from "+str(date[-1].date())+"\nplotted at "+datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")+"\nCurrent signal = "+format(int(signal[-1]/11000.),'-2d'), fontsize=8 )
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


def makeUptrendingPlot(json_fn: str) -> None:

    from functions.TAfunctions import SMA, MoveMax, dpgchannel

    #############
    # Read uptrending stocks status file
    #############

    json_folder = os.path.split(json_fn)[0]
    p_store = get_performance_store(json_fn)
    webpage_dir = get_webpage_store(json_fn)
    file2path = os.path.join(webpage_dir, "pyTAAAweb_numberUptrendingStocks_status.params" )
    
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
                except Exception as e:
                    if statusline.strip():
                        print(f" Warning: Error parsing line {i} in numberUptrendingStocks (2nd): {e}")
                        print(f"   Line content: {statusline[:100]}")
                    break
    except:
        print(" Error: unable to read updates from pyTAAAweb_numberUptrendingStocks_status.params")
        print("")

    ##########################################
    # read multi-Sharpe signal status file
    ##########################################

    file2path = os.path.join(p_store, "pyTAAAweb_multiSharpeIndicator_status.params" )

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
                except Exception as e:
                    if statusline.strip():
                        print(f" Warning: Error parsing line {i} in MeanTrendDispersion (file2): {e}")
                        print(f"   Line content: {statusline[:100]}")
                    break
    except:
        print(" Error: unable to read updates from pyTAAAweb_multiSharpeIndicator_status.params")
        print("")

    ##########################################
    # make plot
    ##########################################

    webpage_dir = get_webpage_store(json_fn)
    figure3path = os.path.join(webpage_dir, "PyTAAA_numberUptrendingStocks.png" )

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


def makeNewHighsAndLowsPlot(json_fn: str) -> None:

    #from functions.TAfunctions import SMA, MoveMax, dpgchannel
    from functions.GetParams import get_json_params
    from functions.CountNewHighsLows import newHighsAndLows

    ########################################################################
    ### compute plot showing new highs and lows over various time periods
    ########################################################################

    params = get_json_params(json_fn)

    if params['stockList'] == 'Naz100':
        _, _, _ = newHighsAndLows(
            json_fn, num_days_highlow=(73,293),
            num_days_cumu=(50,159),
            HighLowRatio=(1.654,2.019),
            HighPctile=(8.499,8.952),
            HGamma=(1.,1.),
            LGamma=(1.176,1.223),
            makeQCPlots=True
        )

    elif params['stockList'] == 'SP500':
        _, _, _ = newHighsAndLows(
            json_fn, num_days_highlow=(73,146),
            num_days_cumu=(76,108),
            HighLowRatio=(2.293,1.573),
            HighPctile=(12.197,11.534),
            HGamma=(1.157,.568),
            LGamma=(.667,1.697),
            makeQCPlots=True
        )

    ########################################################################
    ### write html for web page
    ########################################################################

    figurepath = 'PyTAAA_newHighs_newLows_count.png'  # re-set to name without full path
    figure_htmlText = "\n<br><h3>Counts of stocks with new highs and new lows over various time periods</h3>\n"
    figure_htmlText = figure_htmlText + "\nPlot shows number of stocks that had new highs (green line) and/or new lows (red line) over a combination of time periods. New Highs are plotted with a scale factor since their influence qualitatively seems to be about double that of new lows. There is a good, although qualitative, agreement between overall valuations of average stock movement and the relative count of new highs and new lows. When new lows dominate, consider a smaller investment level.\n"
    figure_htmlText = figure_htmlText + '''<br><img src="'''+figurepath+'''" alt="PyTAAA by DonaldPG" width="850" height="500"><br>\n'''

    return figure_htmlText


def makeTrendDispersionPlot(json_fn: str) -> None:

    from functions.TAfunctions import SMA, MoveMax
    #import functions.allstats

    ##########################################
    # read uptrending stocks status file and make plot
    ##########################################
    json_folder = os.path.split(json_fn)[0]
    p_store = get_performance_store(json_fn)
    file4path = os.path.join(p_store, "pyTAAAweb_MeanTrendDispersion_status.params" )
    webpage_dir = get_webpage_store(json_fn)
    figure4path = os.path.join(webpage_dir, "PyTAAA_MeanTrendDispersion.png" )

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
                except Exception as e:
                    if statusline.strip():
                        print(f" Warning: Error parsing line {i} in DailyChannelOffsetSignal: {e}")
                        print(f"   Line content: {statusline[:100]}")
                    break
    except:
        print(" Error: unable to read updates from pyTAAAweb_MeanTrendDispersion_status.params")
        print("")

    valueMeans = np.array( valueMeans ).astype('float')
    #valueMeansSMA = SMA( valueMeans, 100 )
    valueMedians = np.array( np.clip(valueMedians,-25.,25. ) ).astype('float')
    valueMediansSMA = SMA( valueMedians, 100 )

    plt.figure(3,figsize=(9,7))
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
    webpage_dir = get_webpage_store(json_fn)
    file2path = os.path.join(webpage_dir, "pyTAAAweb_numberUptrendingStocks_status.params" )
    date = []
    value = []
    active = []
    try:
        with open( file2path, "r" ) as f:
            ## get number of lines in file
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
        print(" Error: unable to read updates from pyTAAAweb_numberUptrendingStocks_status.params")
        print("")

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

    file3path = os.path.join(p_store, "pyTAAAweb_backtestPortfolioValue.params" )
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
                    statusline_list = [x for x in statusline.split(" ") if x]  # Filter empty strings
                    if len( statusline_list ) >= 3:
                        backtestDate.append( datetime.datetime.strptime( statusline_list[0], '%Y-%m-%d') )
                        backtestBHvalue.append( float(statusline_list[1]) )
                        backtestSystemvalue.append( float(statusline_list[2]) )
                except Exception as e:
                    if statusline.strip():
                        print(f" Warning: Error parsing line {i} in backtestPortfolioValue: {e}")
                        print(f"   Line content: {statusline[:100]}")
                        print(f"   Split result length: {len(statusline_list)}, content: {statusline_list}")
                    break
    except:
        print(" Error: unable to read updates from pyTAAAweb_backtestPortfolioValue.params")
        print("")


    ###
    ### make a combined plot
    ### 2. make plot showing trend below B&H and trade-system Value
    ###
    webpage_dir = get_webpage_store(json_fn)

    figure5path = os.path.join(webpage_dir, "PyTAAA_backtestWithTrend.png" )
    plt.figure(4,figsize=(9,7))
    plt.clf()
    subplotsize = gridspec.GridSpec(2,1,height_ratios=[5,3])
    plt.subplot(subplotsize[0])
    plt.grid(True)
    plt.yscale('log')
    plotmax = 1.e10
    plt.ylim([1000,max(10000,plotmax)])
    numDaysToPlot = 252*10
    numDaysToPlot = len( backtestBHvalue )
    plt.plot(
        backtestDate[-numDaysToPlot:], backtestBHvalue[-numDaysToPlot:],
        'r-', lw=1.25, label='Buy & Hold'
    )
    plt.plot(
        backtestDate[-numDaysToPlot:], backtestSystemvalue[-numDaysToPlot:],
        'k-', lw=1.25, label='Trading System'
    )
    plt.legend(loc=2,prop={'size':9})
    plt.title("pyTAAA History Plot\n Portfolio Value")
    plt.text(
        backtestDate[-numDaysToPlot+50], 2500,
        "Backtest updated "+datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"),
        fontsize=7.5
    )
    plt.subplot(subplotsize[1])
    plt.grid(True)
    plt.ylim(0, 1.2)
    numDaysToPlot = 252*10
    numDaysToPlot = len( value )
    plt.plot(
        date[-numDaysToPlot:], value[-numDaysToPlot:],
        'k-', lw=.25, label='Percent Uptrending'
    )
    plt.plot(
        date[-numDaysToPlot:], np.clip(PctInvested[-numDaysToPlot:],0.,1.2),
        'g-', alpha=.65, lw=.5, label='Percent to Invest'
    )
    numDaysToPlot = len( valueMedians )
    plt.legend(loc=3,prop={'size':6})
    plt.savefig(figure5path)
    #figure5path = 'PyTAAA_backtestWithTrend.png'  # re-set to name without full path
    figure5path = 'PyTAAA_backtestWithTrend.png'  # re-set to name without full path
    figure5_htmlText = "\n<br><h3>Monthly backtest</h3>\n"
    figure5_htmlText = figure5_htmlText + "\nMost recent backtest for PyTAAA. "
    figure5_htmlText = figure5_htmlText + "PyTAAA is "
    figure5_htmlText = figure5_htmlText + "based on recent performance metrics. "
    figure5_htmlText = figure5_htmlText + "These metrics are computed conventionally, not using DL. "
    figure5_htmlText = figure5_htmlText + "Choices are made from these metrics to "
    figure5_htmlText = figure5_htmlText + "choose the number of stocks to hold, "
    figure5_htmlText = figure5_htmlText + "and what performance metric is used "
    figure5_htmlText = figure5_htmlText + "to judge recent performance.\n"
    figure5_htmlText = figure5_htmlText + '''<br><img src="'''+figure5path+'''" alt="PyTAAA by DonaldPG" width="850" height="500"><br>\n'''

    return figure5_htmlText


def _kill_existing_montecarlo_processes(json_fn: str) -> None:
    """Kill any existing background Monte Carlo processes for the same config.
    
    Searches for running processes matching the background_montecarlo_runner
    pattern with the same JSON file argument and terminates them.
    
    Args:
        json_fn: Path to the JSON configuration file to match.
    
    Returns:
        None
    """
    try:
        # Get list of all processes
        ps_output = subprocess.check_output(
            ["ps", "aux"],
            text=True,
            stderr=subprocess.DEVNULL
        )
        
        # Normalize the json_fn path for comparison
        json_fn_normalized = os.path.abspath(json_fn)
        
        # Find matching processes
        killed_count = 0
        for line in ps_output.splitlines():
            # Look for background_montecarlo_runner processes
            if "background_montecarlo_runner" in line and "--json-file" in line:
                # Extract PID (second column in ps aux output)
                parts = line.split()
                if len(parts) < 2:
                    continue
                pid_str = parts[1]
                
                # Check if this process is using the same JSON file
                if json_fn in line or json_fn_normalized in line:
                    try:
                        pid = int(pid_str)
                        # Don't kill our own process
                        if pid != os.getpid():
                            os.kill(pid, 15)  # SIGTERM
                            killed_count += 1
                            print(f" [async] Killed existing Monte Carlo process (PID {pid})")
                    except (ValueError, ProcessLookupError, PermissionError) as e:
                        # Process may have already exited or permission denied
                        pass
        
        if killed_count == 0:
            print(" [async] No existing Monte Carlo processes found for this config")
    
    except Exception as e:
        # Don't fail the spawn if process detection fails
        print(f" [async] Warning: Could not check for existing processes: {e}")


def _spawn_background_montecarlo(json_fn: str, web_dir: str) -> None:
    """Spawn a detached background process for Monte Carlo backtest generation.

    Launches ``functions/background_montecarlo_runner.py`` as a fully
    detached subprocess (new session, stdout/stderr redirected to a log
    file).  Returns immediately; the caller does not wait for the backtest
    to complete.
    
    Before spawning, checks for and terminates any existing background
    Monte Carlo processes for the same configuration to prevent duplicate
    computations.

    Args:
        json_fn: Path to the JSON configuration file.
        web_dir: Directory where the log file will be written.

    Returns:
        None

    Side Effects:
        - Kills any existing background Monte Carlo processes for this config.
        - Spawns a detached background process.
        - Writes subprocess stdout/stderr to ``montecarlo_backtest.log``
          in *web_dir* (file is recreated, not appended).
    """
    # Kill any existing Monte Carlo processes for this config
    _kill_existing_montecarlo_processes(json_fn)
    
    project_root = os.path.dirname(os.path.dirname(__file__))

    env = os.environ.copy()
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = (
            f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
        )
    else:
        env["PYTHONPATH"] = project_root

    cmd = [
        sys.executable,
        "-m",
        "functions.background_montecarlo_runner",
        "--json-file",
        json_fn,
    ]

    # Ensure the web directory exists
    os.makedirs(web_dir, exist_ok=True)
    
    log_file = os.path.join(web_dir, "montecarlo_backtest.log")
    with open(log_file, "w") as log_fh:
        log_fh.write(
            f"[{datetime.datetime.now().isoformat()}] "
            f"Spawning background Monte Carlo backtest\n"
        )

    with open(log_file, "a") as log_fh:
        subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=log_fh,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )

    print(" [async] Background Monte Carlo backtest started.")
    print(f"   Log: {log_file}\n\n\n")


def makeDailyMonteCarloBacktest(
    json_fn: str, async_mode: bool = False
) -> str:
    """Generate the daily Monte Carlo backtest plots and return HTML markup.

    Performs a freshness check: if the backtest PNG is less than 20 hours
    old the heavy computation is skipped entirely.  When *async_mode* is
    ``True`` and the plot needs refreshing, a detached background process
    is spawned instead of running the backtest inline; the main program
    returns immediately.

    Args:
        json_fn: Path to the JSON configuration file.
        async_mode: When ``True``, run the Monte Carlo backtest in a
            detached background process and return immediately.
            When ``False`` (default), run synchronously.

    Returns:
        HTML fragment string containing ``<img>`` tags for the two Monte
        Carlo backtest plots.
    """

    from functions.UpdateSymbols_inHDF5 import loadQuotes_fromHDF
    from functions.dailyBacktest_pctLong import dailyBacktest_pctLong

    ##########################################
    # make plot with daily monte carlo backtest
    ##########################################
    webpage_dir = get_webpage_store(json_fn)
    figure6path = os.path.join(
        webpage_dir, "PyTAAA_monteCarloBacktest.png"
    )

    ###
    ### make a combined plot
    ### - only update between midnight and 2 a.m.
    ### - make plot showing trend below B&H and trade-system Value
    ###

    """
    today = datetime.datetime.now()
    hourOfDay = today.hour

    if hourOfDay < 3:
        dailyBacktest_pctLong()
    """

    ##########################################
    # perform backtest if this is first time PyTAAA is computed today
    ##########################################

    ###
    ### retrieve quotes with symbols and dates
    ###

    # params = get_json_params(json_fn)
    symbols_file = get_symbols_file(json_fn)
    # stockList = params['stockList']

    # # read list of symbols from disk.
    # symbol_directory = os.path.join( json_folder, "symbols" )
    # if stockList == 'Naz100':
    #     symbol_file = "Naz100_Symbols.txt"
    # elif stockList == 'SP500':
    #     symbol_file = "SP500_Symbols.txt"
    # symbols_file = os.path.join( symbol_directory, symbol_file )
    symbol_directory, symbol_file = os.path.split(symbols_file)

    _, _, datearray, _, _ = loadQuotes_fromHDF(symbols_file, json_fn)

    # get day when output plot was last modified
    try:
        mtime = os.path.getmtime(figure6path)
    except OSError:
        mtime = 0
    # last_modified_date = datetime.date.fromtimestamp(mtime)
    last_modified_date = datetime.datetime.fromtimestamp(mtime)

    print(
        "\n\n\nBacktest check:   last_modified_date (day) = ",
        last_modified_date.day,
        " datearray[-1].day = ",
        datearray[-1].day,
    )

    #if last_modified_date.day <= datearray[-1].day:
    # if (last_modified_date - datearray[-1]).total_seconds() < 0:
    # if modified_hours < 20.0:
    #     dailyBacktest_pctLong(json_fn)
    modified_time = datetime.datetime.now() - last_modified_date
    modified_hours = (
        modified_time.days * 24 + modified_time.seconds / 3600
    )

    if modified_hours > 20.0:
        if async_mode:
            # Fire-and-forget: spawn detached background process.
            _spawn_background_montecarlo(json_fn, webpage_dir)
        else:
            dailyBacktest_pctLong(json_fn)

    if "abacus" in json_fn.lower():
        # Update abacus backtest portfolio values (column 3) with
        # model-switching results.
        from functions.abacus_backtest import (  # noqa: PLC0415
            write_abacus_backtest_portfolio_values,
        )
        write_abacus_backtest_portfolio_values(json_fn)

    # for abacus, re-populate the saved abacus backtest portfolio values


    ##########################################
    # create html markup for backtest plot
    ##########################################

    figure6path = 'PyTAAA_monteCarloBacktest.png'  # re-set to name without full path
    figure6_htmlText = "\n<br><h3>Daily backtest with trend indicators and measure of invested percent</h3>\n"
    figure6_htmlText = figure6_htmlText + "\nCombined backtest with Trend indicators.\n"
    figure6_htmlText = figure6_htmlText + '''<br><img src="'''+figure6path+'''" alt="PyTAAA by DonaldPG" width="850" height="500"><br>\n'''

    figure6Recentpath = 'PyTAAA_monteCarloBacktestRecent.png'  # re-set to name without full path
    figure6_htmlText = figure6_htmlText + "\n<br><h3>Recent portion of daily backtest with trend indicators and measure of invested percent</h3>\n"
    figure6_htmlText = figure6_htmlText + "\nRecent part of combined backtest with Trend indicators.\n"
    figure6_htmlText = figure6_htmlText + '''<br><img src="'''+figure6Recentpath+'''" alt="PyTAAA by DonaldPG" width="850" height="500"><br>\n'''

    return figure6_htmlText



def makeStockCluster(json_fn: str) -> None:

    import datetime
    from functions.stock_cluster import dailyStockClusters

    ##########################################
    # make plot with daily monte carlo backtest
    ##########################################
    webpage_dir = get_webpage_store(json_fn)
    figure7path = os.path.join(webpage_dir, 'Clustered_companyNames.png' )

    ###
    ### make a combined plot
    ### - only update between midnight and 2 a.m.
    ### - make plot showing stock performance clustering
    ###

    today = datetime.datetime.now()
    hourOfDay = today.hour

    if hourOfDay < 3:
        figure7_htmlText = dailyStockClusters(json_fn)
    else:
        figure7path = 'Clustered_companyNames.png'  # re-set to name without full path
        figure7_htmlText = "\n<br><h3>Daily stock clustering analyis. Based on one year performance correlations.</h3>\n"
        figure7_htmlText = figure7_htmlText + "\nClustering based on daily variation in Nasdaq 100 quotes.\n"
        figure7_htmlText = figure7_htmlText + '''<br><img src="'''+figure7path+'''" alt="PyTAAA by DonaldPG" width="850" height="500"><br>\n'''

    return figure7_htmlText


def makeMinimumSpanningTree(json_fn: str) -> None:

    from functions.make_stock_xcorr_network_plots import make_networkx_spanning_tree_plot

    ##########################################
    # make plot with daily monte carlo backtest
    ##########################################
    webpage_dir = get_webpage_store(json_fn)
    figure7apath = os.path.join(webpage_dir, 'minimum_spanning_tree.png' )

    ###
    ### make plot of minimum spanning tree based on correlations between
    ### stock's recent (22 day) performance
    ###

    make_networkx_spanning_tree_plot(figure7apath, json_fn)

    figure7apath = 'minimum_spanning_tree.png'  # re-set to name without full path
    figure7a_htmlText = "\n<br><h3>Daily stock minimum-spanning tree analyis. Based on 22 day performance correlations.</h3>\n"
    figure7a_htmlText = figure7a_htmlText + "\nCorrelations for graph network based on daily variation quotes for the stock universe.\n"
    figure7a_htmlText = figure7a_htmlText + "\nUse to visually observe if patterns are related to (desirable attributes from) portfolio diversity\n"
    figure7a_htmlText = figure7a_htmlText + '''<br><img src="'''+figure7apath+'''" alt="PyTAAA by DonaldPG" width="850" height="500"><br>\n'''

    return figure7a_htmlText



def makeDailyChannelOffsetSignal(json_fn: str) -> None:
    from functions.TAfunctions import recentTrendAndStdDevs
    #import functions.allstats
    from functions.UpdateSymbols_inHDF5 import loadQuotes_fromHDF

    json_folder = os.path.split(json_fn)[0]
    p_store = get_performance_store(json_fn)
    webpage_dir = get_webpage_store(json_fn)
    file4path = os.path.join(p_store, "pyTAAAweb_DailyChannelOffsetSignal_status.params" )
    figure4path = os.path.join(webpage_dir, "PyTAAA_DailyChannelOffsetSignalV.png" )

    print("   . json_fn = " + str(json_fn))
    params = get_json_params(json_fn)
    symbols_file = get_symbols_file(json_fn)

    # # read list of symbols from disk.
    # symbol_directory = os.path.join( json_folder, "symbols" )
    # if stockList == 'Naz100':
    #     symbol_file = "Naz100_Symbols.txt"
    # elif stockList == 'SP500':
    #     symbol_file = "SP500_Symbols.txt"
    # symbols_file = os.path.join( symbol_directory, symbol_file )
    symbol_directory, symbol_file = os.path.split(symbols_file)

    adjClose, symbols, datearray, _, _ = loadQuotes_fromHDF(symbols_file, json_fn)

    ###
    ### get last date already processed
    ###
    _dates = []
    avgPctChannel = []
    numAboveBelowChannel = []
    try:
        with open( file4path, "r" ) as f:
            # get number of lines in file
            lines = f.read().split("\n")
            numlines = len (lines)
            for i in range(numlines):
                statusline = lines[i]
                statusline_list = statusline.split(" ")
                statusline_list = [_f for _f in statusline_list if _f]
                if len( statusline_list ) == 3:
                    _dates.append( datetime.datetime.strptime( statusline_list[0], '%Y-%m-%d') )
                    avgPctChannel.append( float(statusline_list[1].split('%')[0])/100. )
                    numAboveBelowChannel.append( float(statusline_list[2]) )
    except:
        print(" Error: unable to read updates from pyTAAAweb_numberUptrendingStocks_status.params")
        print("")
    # print(" ... inside MakeValuePlot. .line 615. _dates = " + str( _dates))
    # last_date = _dates[-1].date()
    last_date = datetime.datetime.strptime(str(datearray[-1]), '%Y-%m-%d')
    print("   ...inside makeDailyChannelOffsetSignal... last_date = ", last_date)

    # parameters for signal
    params = get_json_params(json_fn)
    minperiod = params['minperiod']
    maxperiod = params['maxperiod']
    incperiod = params['incperiod']
    numdaysinfit = params['numdaysinfit']
    offset = params['offset']
    send_texts = bool(params['send_texts'])
    print(" ... in MakeValuePlot.makeDailyChannelOffsetSignal: send_texts = "+str(send_texts) + ", type = " + str(type(send_texts)))

    print("minperiod,maxperiod,incperiod,numdaysinfit,offset = ", minperiod,maxperiod,incperiod,numdaysinfit,offset)

    # process for each date
    print("\n  ... inside makeDailyChannelOffsetSignal ...")
    #dailyChannelOffsetSignal = np.zeros( adjClose.shape[1], 'float' )
    #dailyCountDowntrendChannelOffsetSignal = np.zeros( adjClose.shape[1], 'float' )
    #for idate in range(numdaysinfit+incperiod,adjClose.shape[1])
    for idate in range(adjClose.shape[1]):
        if send_texts and datearray[idate] >= last_date :
            #if datearray[idate] > datetime.date(1992,1,1) :
            #if datearray[idate] > datetime.date(1992,1,1) :
            if idate%10 == 0:
                print("   ...idate, datearray[idate] = ", idate, datearray[idate])
            # process all symbols
            #numberDowntrendSymbols = 0
            #dailyChannelPct = []
            #print "     ... symbols = ", symbols
            floatChannelGainsLosses = []
            floatStdevsAboveChannel = []
            for i, symbol in enumerate(symbols):
                #print "     ... symbol = ", symbol
                quotes = adjClose[i,idate-numdaysinfit-offset-1:idate].copy()

                channelGainLoss, numStdDevs, pctChannel = \
                                                recentTrendAndStdDevs( \
                                                quotes, \
                                                datearray,\
                                                minperiod=minperiod,\
                                                maxperiod=maxperiod,\
                                                incperiod=incperiod,\
                                                numdaysinfit=numdaysinfit,\
                                                offset=offset)


                floatChannelGainsLosses.append(channelGainLoss)
                floatStdevsAboveChannel.append(numStdDevs)

            floatChannelGainsLosses = np.array(floatChannelGainsLosses)
            floatChannelGainsLosses[np.isinf(floatChannelGainsLosses)] = -999.
            floatChannelGainsLosses[np.isneginf(floatChannelGainsLosses)] = -999.
            floatChannelGainsLosses[np.isnan(floatChannelGainsLosses)] = -999.
            floatChannelGainsLosses = floatChannelGainsLosses[floatChannelGainsLosses != -999.]
            floatStdevsAboveChannel = np.array(floatStdevsAboveChannel)
            floatStdevsAboveChannel[np.isinf(floatStdevsAboveChannel)] = -999.
            floatStdevsAboveChannel[np.isneginf(floatStdevsAboveChannel)] = -999.
            floatStdevsAboveChannel[np.isnan(floatStdevsAboveChannel)] = -999.
            floatStdevsAboveChannel = floatStdevsAboveChannel[floatStdevsAboveChannel != -999.]
            ##print "floatChannelGainsLosses.shape = ", floatChannelGainsLosses.shape
            trimmeanGains = np.mean(floatChannelGainsLosses[np.logical_and(\
                                    floatChannelGainsLosses>np.percentile(floatChannelGainsLosses,5),\
                                    floatChannelGainsLosses<np.percentile(floatChannelGainsLosses,95)\
                                    )])
            trimmeanStdevsAboveChannel = np.mean(floatStdevsAboveChannel[np.logical_and(\
                                    floatStdevsAboveChannel>np.percentile(floatStdevsAboveChannel,5),\
                                    floatStdevsAboveChannel<np.percentile(floatStdevsAboveChannel,95)\
                                    )])

            #print "idate= ",idate,str(datearray[idate])
            textmessage2 = ''
            with open( file4path, "a" ) as ff:
                textmessage2 = "\n"+str(datearray[idate])+"  "+\
                              format(trimmeanGains,"8.2%")+"  "+\
                              format(trimmeanStdevsAboveChannel,"7.1f")
                ff.write(textmessage2)
                print("textmessage2 = ", textmessage2)
            #print "idate= ",idate, str(datearray[idate])


    ##########################################
    # make plot
    ##########################################

    ###
    ### make a combined plot
    ### 1. get percent of uptrending stocks
    ###
    _dates = np.array(_dates)
    avgPctChannel = np.array(avgPctChannel)
    numAboveBelowChannel = np.array(numAboveBelowChannel)
    
    # Debug: Show what we have before trying to use these arrays
    print("\n=== makeDailyChannelOffsetSignal Debug Info ===")
    print(f"file4path (file just read): {file4path}")
    print(f"File exists: {os.path.exists(file4path)}")
    if os.path.exists(file4path):
        file_size = os.path.getsize(file4path)
        print(f"File size: {file_size} bytes")
        if file_size > 0:
            with open(file4path, "r") as f:
                lines = f.read().split("\n")
                print(f"Number of lines in file: {len(lines)}")
                print(f"First 3 lines:")
                for i, line in enumerate(lines[:3], 1):
                    print(f"  Line {i}: '{line}'")
    print(f"_dates array length: {len(_dates)}")
    print(f"avgPctChannel array length: {len(avgPctChannel)}")
    print(f"numAboveBelowChannel array length: {len(numAboveBelowChannel)}")
    print("=" * 50)
    
    # Check if arrays are empty before calling .min(), .mean(), .max()
    if len(avgPctChannel) == 0:
        print("WARNING: avgPctChannel is empty! Cannot compute min/mean/max.")
        print("This likely means the status file has no valid data.")
        # Return early to avoid crash
        figure4_htmlText = "\n<br><h3>Channel Offset Signal</h3>\n"
        figure4_htmlText = figure4_htmlText + "\n<p>No data available for Channel Offset Signal plot.</p>\n"
        figure5_htmlText = "\n<br><h3>Daily backtest with offset Channel trend signal</h3>\n"
        figure5_htmlText = figure5_htmlText + "\n<p>No data available for backtest plot.</p>\n"
        return figure4_htmlText, figure5_htmlText
    
    print(" avgPctChannel shape = " + str(avgPctChannel.shape))
    print(" avgPctChannel min, mean, max = ", avgPctChannel.min(),avgPctChannel.mean(),avgPctChannel.max())
    print("\n\n numAboveBelowChannel = ", numAboveBelowChannel)
    print(" numAboveBelowChannel min, mean, max = ", numAboveBelowChannel.min(),numAboveBelowChannel.mean(),numAboveBelowChannel.max())
    plt.figure(5,figsize=(9,7))
    plt.clf()
    plt.grid(True)
    numDaysToPlot = 252*3
    plt.plot( _dates[-numDaysToPlot:], np.clip(avgPctChannel[-numDaysToPlot:]*100.,-200.,200.), 'r-', lw=.1)
    plt.plot( _dates[-numDaysToPlot:], numAboveBelowChannel[-numDaysToPlot:], 'b-', lw=.25)
    plt.title("pyTAAA History Plot\nChannel Offset Signal")
    plt.savefig(figure4path)
    figure4path = 'PyTAAA_DailyChannelOffsetSignalV2.png'  # re-set to name without full path
    figure4_htmlText = "\n<br><h3>Channel Offset Signal</h3>\n"
    figure4_htmlText = figure4_htmlText + "\nPlot shows up/down trending in last few days compared to trend for stocks in Nasdaq 100.\n"
    figure4_htmlText = figure4_htmlText + '''<br><img src="'''+figure4path+'''" alt="PyTAAA by DonaldPG" width="850" height="500"><br>\n'''

    ###
    ### make a combined plot
    ### 2. make plot showing trend below B&H and trade-system Value
    ###
    file3path = os.path.join(p_store, "pyTAAAweb_backtestPortfolioValue.params" )
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
                    statusline_list = [x for x in statusline.split(" ") if x]  # Filter empty strings
                    if len( statusline_list ) >= 3:
                        backtestDate.append( datetime.datetime.strptime( statusline_list[0], '%Y-%m-%d') )
                        backtestBHvalue.append( float(statusline_list[1]) )
                        backtestSystemvalue.append( float(statusline_list[2]) )
                except Exception as e:
                    if statusline.strip():
                        print(f" Warning: Error parsing line {i} in backtestPortfolioValue (2nd plot): {e}")
                        print(f"   Line content: {statusline[:100]}")
                        print(f"   Split result length: {len(statusline_list)}, content: {statusline_list}")
                    break
    except:
        print(" Error: unable to read updates from pyTAAAweb_backtestPortfolioValue.params")
        print("")

    webpage_dir = get_webpage_store(json_fn)
    figure5path = os.path.join(webpage_dir, "PyTAAA_backtestWithOffsetChannelSignal.png" )
    plt.figure(6,figsize=(9,7))
    plt.clf()
    subplotsize = gridspec.GridSpec(2,1,height_ratios=[5,3])
    plt.subplot(subplotsize[0])
    plt.grid(True)
    plt.yscale('log')
    plotmax = 1.e10
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
    plt.ylim(-100, 100)
    # test that array lengths are the same

    plt.plot( _dates[-numDaysToPlot:], np.clip(avgPctChannel[-numDaysToPlot:]*100.,-200.,200.), 'r-', lw=.1, label='avg Pct offset channel')
    plt.plot( _dates[-numDaysToPlot:], numAboveBelowChannel[-numDaysToPlot:], 'b-', lw=.25, label='number above/below offset channel')
    plt.legend(loc=3,prop={'size':6})
    plt.savefig(figure5path)
    figure5path = 'PyTAAA_backtestWithOffsetChannelSignal.png'  # re-set to name without full path
    figure5_htmlText = "\n<br><h3>Daily backtest with offset Channel trend signal</h3>\n"
    figure5_htmlText = figure5_htmlText + "\nCombined backtest with offset Channel trend signal.\n"
    figure5_htmlText = figure5_htmlText + '''<br><img src="'''+figure5path+'''" alt="PyTAAA by DonaldPG" width="850" height="500"><br>\n'''

    return figure4_htmlText, figure5_htmlText