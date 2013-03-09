import datetime
from functions.quotes_for_list_adjCloseVol import *

###
### Perform a check to see if the stock market is open
### - purpose is to stop calculating and sending emails when nothing has changed
###

def calculateTrades( holdings, last_symbols_text, last_symbols_weight, last_symbols_price ) :

    # set up empty lists for trades
    # - will use trade_shares > 0 for buy, < 0 for sells
    trade_symbols = []
    trade_shares = []
    trade_message = "<br>"

    # put holding data in lists
    holdings_symbols = holdings['stocks']
    holdings_shares = np.array(holdings['shares']).astype('float')
    holdings_buyprice = holdings['buyprice']
    #date2 = datetime.date.today() + datetime.timedelta(+10)
    holdings_currentPrice = LastQuotesForList( holdings_symbols )

    # calculate holdings value
    currentHoldingsValue = 0.
    for i in range(len(holdings_symbols)):
        #print "holdings_shares, holdings_currentPrice[i] = ", i, holdings_shares[i],holdings_currentPrice[i]
        #print "type of above = ",type(holdings_shares[i]),type(holdings_currentPrice[i])
        currentHoldingsValue += float(holdings_shares[i]) * float(holdings_currentPrice[i])

    # check for sells -- stocks that were in last period and out now
    for i,symbol in enumerate( holdings_symbols ):
        if symbol not in last_symbols_text:
            trade_symbols.append( symbol )
            trade_shares.append( -holdings_shares[i] )

    # check for buys -- stocks that were out last period and in now
    for i,symbol in enumerate( last_symbols_text ):
        if symbol not in holdings_symbols:
            trade_symbols.append( symbol )
            new_shares = currentHoldingsValue * last_symbols_weight[i]
            trade_shares.append( new_shares )

    # check for adjustments to current holdings -- stocks that were in last period and in now
    for i,symbol in enumerate(last_symbols_text):
        if symbol in holdings_symbols:
            old_shares = holdings_shares[i]
            new_shares = currentHoldingsValue * last_symbols_weight[i] / last_symbols_price[i]
            deltaShares = new_shares - old_shares
            if deltaShares > 0 :
                trade_symbols.append( symbol )
                trade_shares.append( int(deltaShares) )
            else:
                trade_symbols.append( symbol )
                trade_shares.append( -int(-deltaShares) )

    # prepare message for stocks purchases and sales
    for i in range(len(trade_symbols)):
        if trade_shares[i] < 0:
            # append sells messages
            trade_message = trade_message + "<p>Sell  " + str(trade_symbols[i]) +" "+ str(trade_shares[i])+"</p>"
        else:
            # append buys messages
            trade_message = trade_message + "<p>Buy  " + str(trade_symbols[i]) +" "+ str(trade_shares[i])+"</p>"
    trade_message = trade_message + "<br>"
    # TODO
    # - write new buys to PyTAAA_holdings.params

    print ""
    print "holdings_symbols = ", holdings_symbols
    print "holdings_shares = ", holdings_shares
    print "last_symbols_text = ", last_symbols_text
    print "last_symbols_price = ", last_symbols_price
    print "trade_symbols = ",trade_symbols
    print "trade_shares = ", trade_shares

    return trade_message
