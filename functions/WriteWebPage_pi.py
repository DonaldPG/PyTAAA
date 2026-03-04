"""Web page generation and file deployment utilities for PyTAAA.

This module handles the creation of the PyTAAA HTML dashboard and the
deployment of web assets to remote servers.  It supports two transfer
methods: SFTP via ``paramiko`` for remote Linux servers and local file
copy for Raspberry Pi deployments.

Key Functions:
    writeWebPage: Assemble and write the main HTML dashboard.
    piMoveDirectory: Copy web assets to a local or network path.
    ftpMoveDirectory: Transfer web assets to a remote host via SFTP.

Example:
    from functions.WriteWebPage_pi import writeWebPage
    writeWebPage(
        regulartext="Allocation: 80% AAPL, 20% CASH",
        boldtext="Updated 2025-01-01",
        headlinetext="Market is open",
        lastdate="2025-01-01",
        last_symbols_text=["AAPL", "CASH"],
        last_symbols_weight=[0.8, 0.2],
        last_symbols_price=[150.0, 1.0],
        json_fn="pytaaa_model_switching_params.json",
    )
"""

from typing import List
from functions.GetParams import get_webpage_store
from functions.graph_plots import makeMinimumSpanningTree
from functions.deploy import ftpMoveDirectory, piMoveDirectory  # re-exported



def writeWebPage(
        regulartext: str,
        boldtext: str,
        headlinetext: str,
        lastdate: str,
        last_symbols_text: list,
        last_symbols_weight: list,
        last_symbols_price: list,
        json_fn: str,
) -> None:
    """Assemble and write the main PyTAAA HTML dashboard page.

    Generates the ``pyTAAAweb.html`` file in the configured web output
    directory.  The page includes:

    - Current portfolio valuation text and headline
    - Portfolio value plot (from :func:`~functions.MakeValuePlot.makeValuePlot`)
    - Abacus model-switching recommendation plot (when available)
    - Up-trending stocks plot
    - Trend dispersion, new-highs/lows, channel-offset, and Monte Carlo
      backtest plots
    - Minimum spanning tree graph
    - Current stock rankings table
    - Recent index composition changes table
    - Original backtest reference plots

    After writing the HTML file the function copies the static banner
    and backtest images to the web directory, then deploys the entire
    web directory to the configured remote server using either
    :func:`piMoveDirectory` or :func:`ftpMoveDirectory` depending on
    the host platform.

    Args:
        regulartext: Plain-text description of the current portfolio
            allocation, displayed in the ``<p>`` body tag.
        boldtext: Short summary shown in bold below the headline.
        headlinetext: ``<h1>`` heading text (e.g., market status).
        lastdate: Date string of the most recent data point, used for
            display purposes.
        last_symbols_text: List of ticker symbols in the current
            portfolio.
        last_symbols_weight: List of portfolio weights corresponding to
            ``last_symbols_text``.
        last_symbols_price: List of current prices corresponding to
            ``last_symbols_text``.
        json_fn: Path to the JSON configuration file used to locate
            output directories, symbols file, and transfer credentials.

    Returns:
        None

    Notes:
        - The function suppresses most exceptions internally and prints
          diagnostic messages instead of raising, so a single failed
          plot or file copy does not abort the page build.
        - File deployment is platform-dependent: ARM Linux triggers
          ``piMoveDirectory``; Windows hosts use ``ftpMoveDirectory``;
          other platforms log a warning and skip deployment.
    """
    import datetime
    import os
    #import numpy as np
    # Local imports
    from functions.MakeValuePlot import (
        makeValuePlot,
        makeUptrendingPlot,
        makeNewHighsAndLowsPlot,
        makeTrendDispersionPlot,
        makeDailyChannelOffsetSignal,
        makeDailyMonteCarloBacktest,
    )

    from functions.GetParams import get_json_params, get_symbols_file, get_web_output_dir

    # message body preliminaries
    message = """<!DOCTYPE html>
<html>
<head>
<title>pyTAAA web</title>
</head>

<body id="w3s" bgcolor=#F2F2F2>

<style>
    body
    {
    background-image: -ms-linear-gradient(top left, #F2F2F2 0%, #94B0B3 80%);

    background-image: -moz-linear-gradient(top left, #F2F2F2 0%, #94B0B3 80%);

    background-image: -webkit-linear-gradient(top left, #F2F2F2 0%, #94B0B3 80%);

    background-image: linear-gradient(to bottom right, #F2F2F2 0%, #94B0B3 80%);
    }
    #rank_table_container
    {
      float:left;
      width:825px;
    }
    #indexchanges_table_container
    {
      float:left;
      width:150px;
    }
</style>


<img src="PyTAAA_stock-chart-blue.png" alt="PyTAAA by DonaldPG" width="1000" height="350">

"""
    ##########################################
    # add current valuation table to message
    ##########################################
    message = message+"""
<h1>"""+headlinetext+"""</h1>
<b>"""+boldtext+"""</b>
<p> </p>
<p>"""+regulartext+"""</p>

"""

    """
    #print "last_symbols_price, type = ", last_symbols_price, type(last_symbols_price)

    # add current valuation table to message
    rankingsMessage = "<br><p>Symbol Rank Wght  Price"
    for i, symbol in enumerate( last_symbols_text ):
        #print "last_symbols_price, type = ", last_symbols_price[i], type(last_symbols_price[i]), format(symbol,'6s'),format(i,'5.0f'),format(last_symbols_weight[i],'5.3f'), format(last_symbols_price[i],'7.2f')
        rankingsMessage = rankingsMessage + "<br>"+format(symbol,'6s') \
                                          + format(i,'5.0f') \
                                          + format(last_symbols_weight[i],'5.3f') \
                                          + format(last_symbols_price[i],'7.2f')
    rankingsMessage = rankingsMessage + "<br>"
    """
    print("... writeWebPage: current valuation table added to message ")

    ##########################################
    # read valuations status file and make plot
    ##########################################
    json_folder = os.path.split(json_fn)[0]
    figure_htmlText = makeValuePlot(json_fn)
    message = message + figure_htmlText

    ##########################################
    # Insert abacus model switching explanation and plot before uptrending plot
    ##########################################
    
    figure2path = "PyTAAA_backtest.png"
    figure2_htmlText = "<div id='rank_table_container'>\n<br><h3>Original Monte-carlo Backtest plot</h3>\n"
    figure2_htmlText = figure2_htmlText + "\nHeavy black line is back-tested performance for model. Black shaded area shows performance with different modeling parameters.\n"
    figure2_htmlText = figure2_htmlText + "\n<br>Heavy red line is for equal-weighted basket of current Naz100 stocks. Red shaded area shows performance of individual stocks.\n"
    figure2_htmlText = figure2_htmlText + "\n<br>Lower graph shows number of up-trending stocks.\n"
    figure2_htmlText = figure2_htmlText + '''<br><img src="'''+figure2path+'''" alt="PyTAAA backtest by DonaldPG" width="850" height="500"><br><br>\n'''
    # message = message + figure2_htmlText
    print("... writeWebPage: backtest plot added to message ")


    figure3path = "PyTAAA_backtest_updated.png"
    figure3_htmlText = "\n<br><h3>Monte-carlo Backtest plot after 1 year of 'Forward Testing'</h3>\n"
    figure3_htmlText = figure3_htmlText + '''<br><img src="'''+figure3path+'''" alt="PyTAAA backtest by DonaldPG" width="850" height="500"><br><br>\n</div>'''
    # message = message + figure3_htmlText
    print("... writeWebPage: backtest (updated) plot added to message ")


    ##########################################
    # add plot and explanation of abacus model-swithcing method
    ##########################################
    params = get_json_params(json_fn)
    # webpage_dir = params.get("web_output_dir", "./pyTAAA_web")
    webpage_dir = get_web_output_dir(json_fn)
    rec_plot_relpath = "recommendation_plot.png"
    rec_plot_abspath = os.path.abspath(os.path.join(webpage_dir, rec_plot_relpath))
    print(f" ... rec_plot_abspath = {rec_plot_abspath}")
    print(f" ... webpage_dir = {webpage_dir}")
    print(f" ... os.path.isfile(rec_plot_abspath) = {os.path.isfile(rec_plot_abspath)}")

    add_abacus_plot = False
    if (
        os.path.isfile(rec_plot_abspath)
    ):
        # Copy plot to web output dir if not already present
        try:
            if not os.path.isfile(os.path.join(webpage_dir, rec_plot_relpath)):
                shutil.copy2(rec_plot_abspath, webpage_dir)
            add_abacus_plot = True
        except Exception as e:
            print(f"Warning: Could not copy recommendation plot: {e}")
            add_abacus_plot = False
    print(f" ... add_abacus_plot = {add_abacus_plot}")
    # Explanatory text (2-3 sentences)
    abacus_explanation = (
        "<br><h3>Model Switching Recommendations</h3>\n"
        "PyTAAA's abacus model switching system dynamically selects the "
        "best-performing trading model each month by analyzing recent performance "
        "across multiple strategies. This approach aims to maximize returns and "
        "manage risk by automatically adapting to changing market conditions. "
        "The plot below shows the most recent model recommendations and their performance."\
        "<br><img src=\"recommendation_plot.png\" alt=\"Model Switching Recommendations\" "
        "width=\"850\" height=\"500\"><br>\n"
    )

    if add_abacus_plot:
        message = message + abacus_explanation
        print("... writeWebPage: abacus plot added ")


    ##########################################
    # read uptrending stocks status file and make plot
    ##########################################

    figure4_htmlText = makeUptrendingPlot(json_fn)
    print("... writeWebPage: uptrending stocks status plot created ")
    message = message + figure4_htmlText


    ##########################################
    # Combine plots and explanation
    ##########################################
    # if add_abacus_plot:
    #     message = message + figure_htmlText + abacus_explanation + figure4_htmlText
    # else:
    #     message = message + figure_htmlText + figure4_htmlText

    ##########################################
    # read performance dispersion status file and make plot
    ##########################################

    figure5_htmlText = makeTrendDispersionPlot(json_fn)
    print("... writeWebPage: performance dispersion plot created ")


    ##########################################
    # read performance dispersion status file and make plot
    ##########################################

    figure5aa_htmlText = makeNewHighsAndLowsPlot(json_fn)
    print("... writeWebPage: new highs and lows plot created ")


    ##########################################
    # compute stock value compared to offset trend and make plot
    ##########################################

    figure5a_htmlText, figure5b_htmlText = makeDailyChannelOffsetSignal(json_fn)
    print("... writeWebPage: daily channel offset signal plot created ")


    ##########################################
    # make plot showing monte carlo backtest using variable percent Long trades
    ##########################################

    # Read async flag from config; defaults to True (opt-out behavior).
    async_montecarlo = params.get("async_montecarlo_backtest", True)
    figure6_htmlText = makeDailyMonteCarloBacktest(
        json_fn, async_mode=async_montecarlo
    )
    print("... writeWebPage: monte carlo backtest plot created ")

    """
    ##########################################
    # make plot showing how stock performance clusters
    ##########################################

    try:
        figure7_htmlText = makeStockCluster( )
    except Exception:
        pass
    """

    ##########################################
    # make plot showing minimum spanning tree from graph analysis of stock performance
    ##########################################

    figure7a_htmlText = makeMinimumSpanningTree(json_fn)
    print("... writeWebPage: minimum spanning tree plot created ")


    ##########################################
    # add current rankings table to message
    ##########################################

    webpage_dir = get_webpage_store(json_fn)
    filepath = os.path.join(webpage_dir, "pyTAAAweb_RankList.txt" )
    try:
        with open( filepath, "r" ) as f:
            rankingMessage = f.read()
    except Exception:
        print(" Error: unable to read updates from pyTAAAweb_RankList.txt")
        print("")
    print("... writeWebPage: current rankings table added to message ")

    ##########################################
    # add table with Nasdaq100 index exchanges to message
    ##########################################

    params = get_json_params(json_fn)
    symbols_fn = get_symbols_file(json_fn)
    symbols_folder = os.path.split(symbols_fn)[0]
    stockList = params['stockList']

    if stockList == 'Naz100':
        filepath = os.path.join(symbols_folder, "Naz100_symbolsChanges.txt" )
    elif stockList == 'SP500':
        filepath = os.path.join(symbols_folder, "SP500_symbolsChanges.txt" )
    try:
        with open( filepath, "r" ) as f:
            input = f.read()
    except Exception:
        print(" Error: unable to read updates from *_symbolsChanges.txt")
        print("")

    # separate and remove empty lines
    inputList0 = input.split("\n")

    # remove lines with just "\r"
    inputList = []
    for item in inputList0:
        if item != "\r":
            inputList.append(item)

    # remove "\r" at end of each item in list
    for i,item in enumerate(inputList):
        inputList[i] = item.split("\r")[0]

    # final filtering of each item in list
    inputListFiltered = []
    for item in inputList:
        if item:
            inputListFiltered.append(str(item))

    # print preliminary html tags
    indexExchangesMessage = '<div id="indexchanges_table_container">\n'
    indexExchangesMessage = indexExchangesMessage + "<h3><p>Recent Index Changes :  ..........</p></h3>\n"
    indexExchangesMessage = indexExchangesMessage + "<font face='courier new' size=3><table border='1'>\n"
    indexExchangesMessage = indexExchangesMessage + "<tr><td>Date</td><td>Change</td><td>Ticker</td></tr>\n"

    # print html table entries
    for istring in inputListFiltered :
        print("istring = "+istring)
        date, action, ticker = istring.split()
        indexExchangesMessage = indexExchangesMessage + "<tr><td>"+date \
                                          + "</td><td>"+action \
                                          + "</td><td>"+ticker \
                                          + "</td></tr>\n"
    indexExchangesMessage = indexExchangesMessage + "</table></div>"
    print("... writeWebPage: current index exchanges table added to message ")

    ##########################################
    # Create an updated html page
    ##########################################
    try:
        webpage_dir = get_webpage_store(json_fn)
        filepath = os.path.join(webpage_dir, "pyTAAAweb.html")
        with open( filepath, "w" ) as f:
            f.write(message)
            # Note: figure_htmlText and figure4_htmlText already added to message above
            # Only write plots that weren't added to message
            f.write(figure5_htmlText)
            f.write(figure5aa_htmlText)
            f.write(figure5a_htmlText)
            f.write(figure5b_htmlText)
            f.write(figure6_htmlText)
            f.write(figure7a_htmlText)
            f.write(rankingMessage)
            f.write(indexExchangesMessage)
            f.write(figure2_htmlText)
            f.write(figure3_htmlText)
        print(" Successfully wrote updates to pyTAAAweb html ", datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
        print("")
    except Exception:
        print(" Error: unable to write updates to pyTAAAweb html")
        print("")


    ##########################################
    # Make sure banner is present
    ##########################################
    print("\n\n ... checking banner image is exists")
    try:
        banner_fn = os.path.abspath(os.path.join(
            os.path.abspath(__file__),
            "../..", "assets", "PyTAAA_stock-chart-blue.png"
        ))
        webpage_dir = get_webpage_store(json_fn)
        dest_fn = os.path.join(webpage_dir, "PyTAAA_stock-chart-blue.png")
        if not os.path.isfile(dest_fn):
            # Copy the file with metadata
            shutil.copy2(banner_fn, os.path.split(dest_fn)[0])
    except Exception:
        print("\n\n ... Error:   unable to copy banner image")
    print("   . banner_fn = " + banner_fn)
    print("   . dest_fn = " + dest_fn)


    ##########################################
    # Make sure original backtest plots (2014) are present
    ##########################################
    print("\n\n ... checking original backtest plots exist")
    try:
        banner_fn = os.path.abspath(os.path.join(
            os.path.abspath(__file__),
            "../..", "assets", "PyTAAA_backtest.png"
        ))
        webpage_dir = get_webpage_store(json_fn)
        dest_fn = os.path.join(webpage_dir, "PyTAAA_backtest.png")
        if not os.path.isfile(dest_fn):
            # Copy the file with metadata
            shutil.copy2(banner_fn, os.path.split(dest_fn)[0])
    except Exception:
        print("\n\n ... Error:   unable to copy backtest image 1")
    print("   . banner_fn = " + banner_fn)
    print("   . dest_fn = " + dest_fn)

    try:
        banner_fn = os.path.abspath(os.path.join(
            os.path.abspath(__file__),
            "../..", "assets", "PyTAAA_backtest_updated.png"
        ))
        webpage_dir = get_webpage_store(json_fn)
        dest_fn = os.path.join(webpage_dir, "PyTAAA_backtest_updated.png")
        if not os.path.isfile(dest_fn):
            # Copy the file with metadata
            shutil.copy2(banner_fn, os.path.split(dest_fn)[0])
    except Exception:
        print("\n\n ... Error:   unable to copy backtest image 2")
    print("   . banner_fn = " + banner_fn)
    print("   . dest_fn = " + dest_fn)

    ##########################################
    # Copy web pages to piDonaldPG using ftp (from pc) or using filecopy (from linux)
    ##########################################

    # check the operatiing system to determine whether to move files or use ftp
    import platform

    operatingSystem = platform.system()
    architecture = platform.uname()[4]
    computerName = platform.uname()[1]

    print("  ...platform: ", operatingSystem, architecture, computerName)

    if operatingSystem == 'Linux' and architecture == 'armv6l' :
        print("  ...using piMoveDirectory")
        piMoveDirectory(  )
        try:
            piMoveDirectory(  )
        except Exception:
            print("Could not ftp web files...")

    elif operatingSystem == 'Windows' and computerName == 'Don-XPS1530' :
        print("  ...using ftpMoveDirectory")
        try:
            ftpMoveDirectory(  )
        except Exception:
            print("Could not ftp web files...")

    elif operatingSystem == 'Windows' and computerName == 'DonEnvy' :
        print("  ...using ftpMoveDirectory")
        try:
            ftpMoveDirectory(  )
        except Exception:
            print("Could not ftp web files...")

    elif operatingSystem == 'Windows' and computerName == 'Spectre' :
        print("  ...using ftpMoveDirectory")
        try:
            ftpMoveDirectory(  )
        except Exception:
            print("Could not ftp web files...")

    elif operatingSystem == 'Linux' and computerName == 'pine64' :
        print("  ...using ftpMoveDirectory")
        #try:
        ftpMoveDirectory(  )
        '''
        except Exception:
            print "Could not ftp web files..."
        '''

    else:
        print("Could not place web files on server...")




