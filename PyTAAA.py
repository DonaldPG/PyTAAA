import scheduler
import datetime
import getpass
import os
import time
import urllib
from functions.SendEmail import *
from functions.WriteWebPage_pi import *
from functions.GetParams import *
from functions.UpdateSymbols_inHDF5 import *
from functions.CheckMarketOpen import *
from functions.PortfolioPerformanceCalcs import *
from functions.calculateTrades import *

# Get Credentials for sending email
params = GetParams()
print ""
print "params = ", params
print ""
username = str(params['fromaddr']).split("@")[0]
emailpassword = str(params['PW'])
ip = GetIP()
print "Current ip address is ", ip
print "An email with updated analysis will be sent to ", params['toaddrs'], " every ", params['pausetime'], " seconds"
print params['pausetime'], " seconds is ", format(params['pausetime']/60/60.,'2.1f'), " hours, or ",  \
                                            format(params['pausetime']/60/60/24.,'3.1f'), " days."
print ""


def IntervalTask( ) :

    # keep track of total time to update everything
    start_time_total = time.time()

    # set value to compare with cumu_value as test of info content
    cumu_value_prior = GetStatus()

    # Get Holdings from file
    holdings = GetHoldings()
    print ""
    print "current Holdings :"
    print "stocks: ", holdings['stocks']
    print "shares: ", holdings['shares']
    print "buyprice: ", holdings['buyprice']
    print "cumulativecashin: ", holdings['cumulativecashin'][0]
    print ""

    # Update prices in HDF5 file for symbols in list
    # - check web for current stocks in Naz100, update files if changes needed
    start_time_updateNaz100List = time.time()
    _, removedTickers, addedTickers = get_Naz100List( verbose = True )
    elapsed_time_updateNaz100List = time.time() - start_time_updateNaz100List

    symbol_directory = os.path.join( os.getcwd(), "symbols" )

    symbol_file = "Naz100_symbols.txt"
    symbols_file = os.path.join( symbol_directory, symbol_file )
    start_time = time.time()
    today = datetime.datetime.now()
    hourOfDay = today.hour
    try:
        daily_update_done in locals()
        if hourOfDay <= 15:
            daily_update_done = False
    except:
        daily_update_done = False
    print "hourOfDay, daily_update_done =", hourOfDay, daily_update_done
    if not daily_update_done :
        UpdateHDF5( symbol_directory, symbols_file )
        if hourOfDay > 15:
            daily_update_done = True
    marketOpen, lastDayOfMonth = CheckMarketOpen()
    elapsed_time = time.time() - start_time

    # Re-compute stock ranks and weightings
    try:
        last_symbols_text in locals()
    except:
        CalcsUpdateCount = 0
        not_Calculated = True
    if (daily_update_done and CalcsUpdateCount == 0) or not_Calculated:
        lastdate, last_symbols_text, last_symbols_weight, last_symbols_price = PortfolioPerformanceCalcs( symbol_directory, symbol_file, params )
        CalcsUpdateCount += 1

    # put holding data in lists
    holdings_symbols = holdings['stocks']
    holdings_shares = np.array(holdings['shares']).astype('float')
    holdings_buyprice = np.array(holdings['buyprice']).astype('float')

    #holdings_currentPrice = LastQuotesForList( holdings_symbols )
    holdings_currentPrice = LastQuotesForSymbolList( holdings_symbols )
    print "holdings_symbols = ", holdings_symbols
    print "holdings_shares = ", holdings_shares
    print "holdings_currentPrice = ", holdings_currentPrice

    # calculate holdings value
    currentHoldingsValue = 0.
    for i in range(len(holdings_symbols)):
        #print "holdings_shares, holdings_currentPrice[i] = ", i, holdings_shares[i],holdings_currentPrice[i]
        #print "type of above = ",type(holdings_shares[i]),type(holdings_currentPrice[i])
        currentHoldingsValue += float(holdings_shares[i]) * float(holdings_currentPrice[i])

    # calculate lifetime profit
    print "holdings['cumulativecashin'] = ", holdings['cumulativecashin'][0]
    lifetimeProfit = currentHoldingsValue - float(holdings['cumulativecashin'][0])
    print "Lifetime profit = ", lifetimeProfit

    message_text = "<h3>Current stocks and weights are :</h3><font face='courier new' size=3><table border='1'> \
                   <tr><td>symbol  \
                   </td><td>shares  \
                   </td><td>purch price  \
                   </td><td>purch cost  \
                   </td><td>cumu purch  \
                   </td><td>last price  \
                   </td><td>Value ($)  \
                   </td><td>cumu Value ($)  \
                   </td></tr>\n"
    cumu_purchase_value = 0.
    cumu_value = 0.
    print "holdings_shares = ", holdings_shares
    print "holdings_buyprice = ", holdings_buyprice
    print "last_symbols_text = ", last_symbols_text
    print "last_symbols_weight = ", last_symbols_weight
    print "last_symbols_price = ", last_symbols_price

    for i in range(len(holdings_shares)):
        purchase_value = holdings_buyprice[i]*holdings_shares[i]
        cumu_purchase_value += purchase_value
        value = float(holdings_currentPrice[i]) * float(holdings_shares[i])
        cumu_value += value

        message_text = message_text+"<p><tr><td>"+format(holdings_symbols[i],'5s') \
                                   +"</td><td>"+format(holdings_shares[i],'6.0f') \
                                   +"</td><td>"+format(holdings_buyprice[i],'6.2f') \
                                   +"</td><td>"+format(purchase_value,'6.2f') \
                                   +"</td><td>"+format(cumu_purchase_value,'6.2f') \
                                   +"</td><td>"+format(float(holdings_currentPrice[i]),'6.2f') \
                                   +"</td><td>"+format(value,'6.2f') \
                                   +"</td><td>"+format(cumu_value,'6.2f') \
                                   +"</td></tr>\n"
    print ""


    # Notify with buys/sells on trade dates
    month = datetime.datetime.now().month
    monthsToHold = params['monthsToHold']
    trade_message = "<br>"
    if 0 == 0 :
        trade_message = calculateTrades( holdings, last_symbols_text, last_symbols_weight, last_symbols_price )
        message_text = message_text + trade_message

    edition = GetEdition()
    message_text = message_text+"</table><br><"+"/font><p>Lifetime profit = $"+str(lifetimeProfit)+"   = "+format(lifetimeProfit/float(holdings['cumulativecashin'][0]),'6.1%')+"</p>"

    # Update message for changes in  tickers removed from or added to the Nasdaq100 index
    if removedTickers != [] or addedTickers != []:
        message_text = message_text+"<br><p>There are changes in the stock list<p>"
        for i, ticker in enumerate( removedTickers ):
            message_text = message_text+"<p> ...Ticker "+ticker+" has been removed from the Nasdaq100 index"
        message_text = message_text+"<p>"
        for i, ticker in enumerate( addedTickers ):
            message_text = message_text+"<p> ...Ticker "+ticker+" has been added to the Nasdaq100 index"

    message_text = message_text+"<br><p>"+edition+" ediition software running at "+str(ip)

    elapsed_time_total = time.time() - start_time_total

    # send an email with status and updates (tries up to 10 times for each call).
    boldtext = "time is "+datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
    regulartext = message_text+"<br>elapsed time to update Nasdaq 100 companies from web "+format(elapsed_time_updateNaz100List,'6.2f')+" seconds"
    regulartext = regulartext+"<br>elapsed time to update Nasdaq 100 stock prices from web "+format(elapsed_time,'6.2f')+" seconds"
    if elapsed_time_total < 60 :
        regulartext = regulartext+"<br>elapsed time for web updates, computations, updating web page "+format(elapsed_time_total,'6.2f')+" seconds</p>"
    else:
        regulartext = regulartext+"<br>elapsed time for web updates, computations, updating web page "+format(elapsed_time_total/60.,'6.2f')+" minutes</p>"


    # Customize and send email
    # - based on day of month and whether market is open or closed
    if lastDayOfMonth:
        subjecttext = "PyTAAA holdings update and trade suggestions"
    else:
        subjecttext = "PyTAAA status update"

    print cumu_value_prior, cumu_value
    print np.round(float(cumu_value_prior),2) != np.round(cumu_value,2)
    if np.round(float(cumu_value_prior),2) != np.round(cumu_value,2) or trade_message != "<br><br>":
        headlinetext = "Regularly scheduled update (market is open) " + get_MarketOpenOrClosed()
        SendEmail(username,emailpassword,params['toaddrs'],params['fromaddr'],subjecttext,regulartext,boldtext,headlinetext)
    else:
        headlinetext = "Regularly scheduled update (market is closed) " + get_MarketOpenOrClosed()
        print " No email required or sent -- no new information since last email..."
        cumu_value_prior = cumu_value


    # If there are changes to Nasdaq100 stock list, add message

    # build the updated web page
    writeWebPage( regulartext,boldtext,headlinetext,lastdate, last_symbols_text, last_symbols_weight, last_symbols_price )
    # set value to compare with cumu_value as test of info content
    PutStatus( cumu_value )

    # print market status to terminal window
    get_MarketOpenOrClosed()

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
                          scheduler.RunUntilSuccess( func=IntervalTask ) )

    mail_receipt = my_scheduler.schedule_task(mail_task)

    # Once started, the scheduler will identify the next task to run and execute it.
    my_scheduler.start()

    # Stop the scheduler after runtime
    from time import sleep
    sleep(params['runtime'])
    my_scheduler.halt()

    # Give it a timeout to halt any running tasks and stop gracefully
    my_scheduler.join(100)
