import scheduler
import datetime
import getpass
import numpy as np
import os
import time
import urllib.request, urllib.parse, urllib.error
import platform
from functions.SendEmail import SendEmail
from functions.WriteWebPage_pi import writeWebPage
from functions.GetParams import (GetParams,
                                GetHoldings,
                                GetStatus,
                                PutStatus,
                                GetIP,
                                GetEdition)
from functions.UpdateSymbols_inHDF5 import UpdateHDF_yf
from functions.CheckMarketOpen import (get_MarketOpenOrClosed,
                                       CheckMarketOpen)
from functions.PortfolioPerformanceCalcs import PortfolioPerformanceCalcs
from functions.quotes_for_list_adjClose import LastQuotesForSymbolList_hdf, get_SectorAndIndustry_google
from functions.calculateTrades import calculateTrades
from functions.quotes_for_list_adjClose import get_Naz100List, get_SP500List
from functions.stock_cluster import getClusterForSymbolsList
from functions.ftp_quotes import copy_updated_quotes
import sys
print(sys.path)

try:
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
except OSError:
    # Expected: file not found, permission denied, etc.
    os.chdir("C:\\Users\\Don\\Py3TAAA")
except Exception as e:
    # Safety fallback for unexpected exceptions
    import logging
    logging.getLogger(__name__).warning(f"Unexpected exception in os.chdir: {type(e).__name__}: {e}")
    os.chdir("C:\\Users\\Don\\Py3TAAA")


computerName = platform.uname()[1]

# Get Credentials for sending email
params = GetParams()
print("")
print("params = ", params)
print("")
username = str(params['fromaddr']).split("@")[0]
emailpassword = str(params['PW'])
stockList = params['stockList']
# get name of server used to download and serve quotes
quote_server = params['quote_server']
try:
    ip = GetIP()
except (urllib.error.URLError, OSError, TimeoutError) as e:
    # Expected: network unavailable, timeout, etc.
    import logging
    logging.getLogger(__name__).debug(f"Could not get external IP: {e}")
    ip = '0.0.0.0'
except Exception as e:
    # Safety fallback for unexpected exceptions
    import logging
    logging.getLogger(__name__).warning(f"Unexpected exception in GetIP: {type(e).__name__}: {e}")
    ip = '0.0.0.0'
print("Current ip address is ", ip)
print("An email with updated analysis will be sent to ", params['toaddrs'], " every ", params['pausetime'], " seconds")
print(params['pausetime'], " seconds is ", format(params['pausetime']/60/60., '2.1f'), " hours, or ",  \
                                            format(params['pausetime']/60/60/24., '3.1f'), " days.")
print("")


def IntervalTask():

    # keep track of total time to update everything
    start_time_total = time.time()

    # set value to compare with cumu_value as test of info content
    cumu_value_prior = GetStatus()

    # Get Holdings from file
    holdings = GetHoldings()
    print("")
    print("current Holdings :")
    print("stocks: ", holdings['stocks'])
    print("shares: ", holdings['shares'])
    print("buyprice: ", holdings['buyprice'])
    print("current ranks: ", holdings['ranks'])
    print("cumulativecashin: ", holdings['cumulativecashin'][0])
    print("")

    # Update prices in HDF5 file for symbols in list
    # - check web for current stocks in Naz100, update files if changes needed
    today = datetime.datetime.now()
    hourOfDay = today.hour
    start_time_updateStockList = time.time()
    if hourOfDay <= 17:
        if stockList == 'Naz100':
            _, removedTickers, addedTickers = get_Naz100List(verbose=True)
        elif stockList == 'SP500':
            _, removedTickers, addedTickers = get_SP500List(verbose=True)
    else:
        removedTickers, addedTickers = [], []
    elapsed_time_updateStockList = time.time() - start_time_updateStockList

    symbol_directory = os.path.join(os.getcwd(), "symbols")

    if stockList == 'Naz100':
        symbol_file = "Naz100_Symbols.txt"
    elif stockList == 'SP500':
        symbol_file = "SP500_Symbols.txt"
    symbols_file = os.path.join( symbol_directory, symbol_file )
    start_time = time.time()
    try:
        daily_update_done in locals()
        if hourOfDay <= 15:
            daily_update_done = False
    except NameError:
        # Expected: variable doesn't exist on first run
        daily_update_done = False
    except Exception as e:
        # Safety fallback for unexpected exceptions
        import logging
        logging.getLogger(__name__).warning(f"Unexpected exception checking daily_update_done: {type(e).__name__}: {e}")
        daily_update_done = False
    print("hourOfDay, daily_update_done =", hourOfDay, daily_update_done)
    if quote_server != computerName:
        copy_updated_quotes()
    if not daily_update_done :
        if  quote_server == computerName:
            UpdateHDF_yf( symbol_directory, symbols_file )
        if hourOfDay > 15:
            daily_update_done = True
    marketOpen, lastDayOfMonth = CheckMarketOpen()
    elapsed_time = time.time() - start_time

    # Re-compute stock ranks and weightings
    try:
        last_symbols_text in locals()
    except NameError:
        # Expected: variable doesn't exist on first run
        CalcsUpdateCount = 0
        not_Calculated = True
    except Exception as e:
        # Safety fallback for unexpected exceptions
        import logging
        logging.getLogger(__name__).warning(f"Unexpected exception checking last_symbols_text: {type(e).__name__}: {e}")
        CalcsUpdateCount = 0
        not_Calculated = True
    if (daily_update_done and CalcsUpdateCount == 0) or not_Calculated:
        lastdate, last_symbols_text, last_symbols_weight, last_symbols_price = PortfolioPerformanceCalcs( symbol_directory, symbol_file, params )
        CalcsUpdateCount += 1

    # Get updated Holdings from file (ranks are updated)
    holdings = GetHoldings()
    print("")
    print("current Holdings :")
    print("stocks: ", holdings['stocks'])
    print("shares: ", holdings['shares'])
    print("buyprice: ", holdings['buyprice'])
    print("current ranks: ", holdings['ranks'])
    print("cumulativecashin: ", holdings['cumulativecashin'][0])
    print("")

    # put holding data in lists
    holdings_symbols = holdings['stocks']
    holdings_shares = np.array(holdings['shares']).astype('float')
    holdings_buyprice = np.array(holdings['buyprice']).astype('float')
    holdings_ranks = np.array(holdings['ranks']).astype('int')

    #holdings_currentPrice = LastQuotesForList( holdings_symbols )
    #holdings_currentPrice = LastQuotesForSymbolList( holdings_symbols )
    holdings_currentPrice = LastQuotesForSymbolList_hdf( holdings_symbols, symbol_file )
    print("holdings_symbols = ", holdings_symbols)
    print("holdings_shares = ", holdings_shares)
    print("holdings_currentPrice = ", holdings_currentPrice)

    # retrieve cluster labels for holdings
    try:
        holdings_cluster_labels = getClusterForSymbolsList( holdings_symbols )
    except (FileNotFoundError, KeyError, OSError) as e:
        # Expected: cluster data missing or symbol not found
        import logging
        logging.getLogger(__name__).debug(f"Cluster data unavailable: {e}")
        holdings_cluster_labels = np.zeros((len(holdings_symbols)), 'int')
    except Exception as e:
        # Safety fallback for unexpected exceptions
        import logging
        logging.getLogger(__name__).warning(f"Unexpected exception in getClusterForSymbolsList: {type(e).__name__}: {e}")
        holdings_cluster_labels = np.zeros((len(holdings_symbols)), 'int')

    # calculate holdings value
    currentHoldingsValue = 0.
    for i in range(len(holdings_symbols)):
        #print "holdings_shares, holdings_currentPrice[i] = ", i, holdings_shares[i],holdings_currentPrice[i]
        #print "type of above = ",type(holdings_shares[i]),type(holdings_currentPrice[i])
        currentHoldingsValue += float(holdings_shares[i]) * float(holdings_currentPrice[i])

    # calculate lifetime profit
    print("holdings['cumulativecashin'] = ", holdings['cumulativecashin'][0])
    lifetimeProfit = currentHoldingsValue - float(holdings['cumulativecashin'][0])
    print("Lifetime profit = ", lifetimeProfit)
    # calculate elapsed time period -- starting 1/1/13 and ending today
    elapsedYears = ( (datetime.date( today.year, today.month, today.day ) - datetime.date( 2013,1,1)).days ) / 365.25
    lifetimeProfitAnnualized = (( 1. + lifetimeProfit ) ** 1./elapsedYears ) - 1.

    message_text = "<h3>Current stocks and weights are :</h3><font face='courier new' size=3><table border='1'> \
                   <tr><td>symbol  \
                   </td><td>shares  \
                   </td><td>purch price  \
                   </td><td>purch cost  \
                   </td><td>cumu purch  \
                   </td><td>last price  \
                   </td><td>% change  \
                   </td><td>Value ($)  \
                   </td><td>cumu Value ($)  \
                   </td><td>Curr Rank  \
                   </td><td>Cluster  \
                   </td><td>Sector  \
                   </td><td>Industry  \
                   </td></tr>\n"
    cumu_purchase_value = 0.
    cumu_value = 0.
    print("holdings_shares = ", holdings_shares)
    print("holdings_buyprice = ", holdings_buyprice)
    print("last_symbols_text = ", last_symbols_text)
    print("last_symbols_weight = ", last_symbols_weight)
    print("last_symbols_price = ", last_symbols_price)
    print("holdings_ranks = ", holdings_ranks)
    print("\n\n")

    #for i in range(len(holdings_shares)):
    for i in range(len(holdings_ranks)):
        purchase_value = holdings_buyprice[i]*holdings_shares[i]
        cumu_purchase_value += purchase_value
        value = float(holdings_currentPrice[i]) * float(holdings_shares[i])
        cumu_value += value
        profitPct = float(holdings_currentPrice[i]) / float(holdings_buyprice[i]) - 1.
        print(i, format(holdings_symbols[i],'5s'),\
              float(holdings_buyprice[i]),\
              format(holdings_shares[i],'6.0f'),\
              format(holdings_buyprice[i],'6.2f'),\
              format(purchase_value,'6.2f'), \
              format(cumu_purchase_value,'6.2f'), \
              format(float(holdings_currentPrice[i]),'6.2f'), \
              format(float(profitPct),'6.2%'), \
              format(value,'6.2f'), \
              format(cumu_value,'6.2f'), \
              format(holdings_ranks[i],'3d'),\
              str(holdings_cluster_labels[i]),\
              "\n")
        # get sector and industry for holdings symbol
        if holdings_symbols[i] != 'CASH':
            sector, industry = get_SectorAndIndustry_google( holdings_symbols[i] )
        else:
            sector, industry = "",""
        message_text = message_text+"<p><tr><td>"+format(holdings_symbols[i],'5s') \
                                   +"</td><td>"+format(holdings_shares[i],'6.0f') \
                                   +"</td><td>"+format(holdings_buyprice[i],'6.2f') \
                                   +"</td><td>"+format(purchase_value,'6.2f') \
                                   +"</td><td>"+format(cumu_purchase_value,'6.2f') \
                                   +"</td><td>"+format(float(holdings_currentPrice[i]),'6.2f') \
                                   +"</td><td>"+format(float(profitPct),'6.2%') \
                                   +"</td><td>"+format(value,'6.2f') \
                                   +"</td><td>"+format(cumu_value,'6.2f') \
                                   +"</td><td>"+format(holdings_ranks[i],'3d') \
                                   +"</td><td>"+str(holdings_cluster_labels[i]) \
                                   +"</td><td>"+str(sector) \
                                   +"</td><td>"+str(industry) \
                                   +"</td></tr>\n"
    print("")

    # Notify with buys/sells on trade dates
    month = datetime.datetime.now().month
    monthsToHold = params['monthsToHold']
    trade_message = "<br>"
    if 0 == 0 :
        trade_message = calculateTrades( holdings, last_symbols_text, last_symbols_weight, last_symbols_price )
        message_text = message_text + trade_message

    edition = GetEdition()
    message_text = message_text+"</table><br></font><p>Lifetime profit = $"+str(lifetimeProfit)+"   = "+format(lifetimeProfit/float(holdings['cumulativecashin'][0]),'6.1%')+"</p>"
    message_text = message_text+"</font><p>Lifetime profit (annualized rate of return) = "+format(lifetimeProfitAnnualized/float(holdings['cumulativecashin'][0]),'6.1%')+"</p>"


    # Update message for changes in  tickers removed from or added to the Nasdaq100 index
    if removedTickers != [] or addedTickers != []:
        message_text = message_text+"<br><p>There are changes in the stock list<p>"
        for i, ticker in enumerate( removedTickers ):
            message_text = message_text+"<p> ...Ticker "+ticker+" has been removed from the stock index list"
        message_text += "<p>"
        for i, ticker in enumerate( addedTickers ):
            message_text = message_text+"<p> ...Ticker "+ticker+" has been added to the stock index list"

    message_text = message_text+"<br><p>"+edition+" edition software running at "+str(ip)
    message_text = message_text+"<br>Stock universe is "+params['stockList']
    message_text = message_text+"<br>up-trending signal method is "+params['uptrendSignalMethod']+"<p>"

    elapsed_time_total = time.time() - start_time_total

    # send an email with status and updates (tries up to 10 times for each call).
    boldtext = "time is "+datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
    regulartext = message_text+"<br>elapsed time to update stock index companies from web "+format(elapsed_time_updateStockList,'6.2f')+" seconds"
    regulartext = regulartext+"<br>elapsed time to update stock index stock prices from web "+format(elapsed_time,'6.2f')+" seconds"
    if elapsed_time_total < 60 :
        regulartext = regulartext+"<br>elapsed time for web updates, computations, updating web page "+format(elapsed_time_total,'6.2f')+" seconds</p>"
    else:
        regulartext = regulartext+"<br>elapsed time for web updates, computations, updating web page "+format(elapsed_time_total/60.,'6.2f')+" minutes</p>"
    regulartext = regulartext+"<br><p>Links:"
    regulartext = regulartext+"<br><a href=pyTAAAweb_symbolCharts_MonthStartRank.html>Stock Charts Ordered by Ranking at Start of Month</a>"
    regulartext = regulartext+"<br><a href=pyTAAAweb_symbolCharts_TodayRank.html>Stock Charts Ordered by Ranking Today</a>"
    regulartext = regulartext+"<br><a href=pyTAAAweb_symbolCharts_recentGainRank.html>Stock Charts Ordered by Recent Gain Ranking</a>"
    regulartext = regulartext+"<br><a href=pyTAAAweb_symbolCharts_recentComboGainRank.html>Stock Charts Ordered by Recent Combo Gain Ranking</a>"
    regulartext = regulartext+"<br><a href=pyTAAAweb_symbolCharts_recentTrendRatioRank.html>Stock Charts Ordered by Ranking Today using ratio of recent trends without and with gap</a>"
    regulartext = regulartext+"<br><a href=pyTAAAweb_symbolCharts_recentSharpeRatioRank.html>Stock Charts Ordered by recent sharpe ranking</a>"
    regulartext = regulartext+"<br><p>Links to additional PyTAAA variants:"
    regulartext = regulartext+"<br><a href=../pyTAAA_piweb/pyTAAAweb.html>PyTAAA using Nasdaq 100 (3 MAs method)</a>"
    regulartext = regulartext+"<br><a href=../pyTAAA_web/pyTAAAweb.html>PyTAAA using Nasdaq 100 (3 minmax method)</a>"
    regulartext = regulartext+"<br><a href=../pyTAAA_SP500web/pyTAAAweb.html>PyTAAA using S&P 500</a>"
    regulartext = regulartext+"<br><a href=../pyTAAADL_web/pyTAAAweb.html>PyTAAADL using Nasdaq 100</a>"

    # Customize and send email
    # - based on day of month and whether market is open or closed
    if lastDayOfMonth:
        subjecttext = "PyTAAA holdings update and trade suggestions"
    else:
        subjecttext = "PyTAAA status update"

    print("cumu_value_prior, cumu_value = ", cumu_value_prior, cumu_value)
    print(np.round(float(cumu_value_prior),2) != np.round(cumu_value,2))
    print("trade_message = ", trade_message)
    print(trade_message != "<br>")
    if np.round(float(cumu_value_prior),2) != np.round(cumu_value,2) or trade_message != "<br>":
        headlinetext = "Regularly scheduled update. Market status: " + get_MarketOpenOrClosed()
        SendEmail(username,emailpassword,params['toaddrs'],params['fromaddr'],subjecttext,regulartext,boldtext,headlinetext)
    else:
        headlinetext = "Regularly scheduled update. Market status: " + get_MarketOpenOrClosed()
        print(" No email required or sent -- no new information since last email...")
        cumu_value_prior = cumu_value


    # If there are changes to Nasdaq100 stock list, add message

    # build the updated web page
    writeWebPage( regulartext,boldtext,headlinetext,lastdate, last_symbols_text, last_symbols_weight, last_symbols_price )
    # set value to compare with cumu_value as test of info content
    PutStatus( cumu_value )

    # print market status to terminal window
    get_MarketOpenOrClosed()
    print(" . finished at ", str(datetime.datetime.now()))



'''
Main program
'''
if __name__ == '__main__':

    # Create a scheduler
    my_scheduler = scheduler.Scheduler()

    # Add the mail task, a receipt is returned that can be used to drop the task from the scheduler
    mail_task = scheduler.Task("Interval_Task",
                          datetime.datetime.now(),
                          scheduler.every_x_secs(params['pausetime']),
                          scheduler.RunUntilSuccess( func=IntervalTask, num_tries=1 ) )

    mail_receipt = my_scheduler.schedule_task(mail_task)

    # Once started, the scheduler will identify the next task to run and execute it.
    my_scheduler.start()

    # Stop the scheduler after runtime
    from time import sleep
    sleep(params['runtime'])
    my_scheduler.halt()

    # Give it a timeout to halt any running tasks and stop gracefully
    my_scheduler.join(100)
