import os
import numpy as np
import datetime
#from functions.quotes_for_list_adjCloseVol import *
from functions.quotes_for_list_adjClose import LastQuotesForSymbolList_hdf
from functions.CheckMarketOpen import *
from functions.GetParams import (
    get_json_params, get_symbols_file, get_performance_store
)

###
### Perform a check to see if the stock market is open
### - purpose is to stop calculating and sending emails when nothing has changed
###

def calculateTrades(
        holdings, last_symbols_text,
        last_symbols_weight, last_symbols_price, json_fn
):
    """
    Function calculates updates to holdings based on input lists.
    1. Finds stocks from existing holding that will continue to be held.
       - Calculates adjustments to number of shares held, subject to $400 minimum exchange
    2. Finds stocks to be sold.
    3. Find stocks to be bought, subject to $800 minimum purchase.
    4. Calculates cash balance with symbol "CASH"

    Note. Exchanges and purchases that are below minimum have their value re-distributed
    among other exchanges and purchases, respectively

    Variables:

    holdings - dict with 'stocks', 'shares', 'buyprice' for currently held stocks
    last_symbols_text - list of symbols with highest ranking recent performance
    last_symbols_weight - list of weights to apply to protfolio value for 'last_symbols_text'
    last_symbols_price - list of current market price for 'last_symbols_text'

    trade_symbols - list with symbols for suggested trades (sells, buys, exchanges)
    trade_shares  - list with number of shares for suggested trades (sells, buys, exchanges)

    new_symbols - list with symbols after suggested trades (sells, buys, exchanges, and unchanged holdings)
    new_shares - list with shares after suggested trades (sells, buys, exchanges, and unchanged holdings)
    new_buyprice - list with market price after suggested trades (sells, buys, exchanges, and unchanged holdings)
    buySellCost - total comission for recommended buys, sells, and re-balance trades (excluding CASH)
    BuySellFee - per-trade comission (in dollars)

    """
    print(" ... inside calculateTrades/calculateTrades ...")
	# set up empty lists for trades
    # - will use trade_shares > 0 for buy, < 0 for sells
    buySellCost = 0.
    BuySellFee = 4.95
    new_symbols = []
    new_shares = []
    new_buyprice = []
    trade_symbols = []
    trade_shares = []

    trade_message = "<br>"

    cumuValueAfterExchanges = 0.
    today = datetime.datetime.now()

    # put holding data in lists
    holdingsParams_symbols = holdings['stocks']
    holdingsParams_shares = np.array(holdings['shares']).astype('float')
    holdingsParams_buyprice = np.array(holdings['buyprice']).astype('float')

    # get current prices for holdings
    #holdingsParams_currentPrice = LastQuotesForSymbolList( holdingsParams_symbols )
    # Get Credentials for sending email
    params = get_json_params(json_fn)
    json_dir = os.path.split(json_fn)[0]
    symbol_directory = os.path.join( json_dir, "symbols" )
    stockList = params['stockList']
    # if stockList == 'Naz100':
    #     symbol_file = "Naz100_Symbols.txt"
    # elif stockList == 'SP500':
    #     symbol_file = "SP500_Symbols.txt"
    # symbols_file = os.path.join( symbol_directory, symbol_file )
    symbols_file = get_symbols_file(json_fn)
    holdingsParams_currentPrice = LastQuotesForSymbolList_hdf(
        holdingsParams_symbols, symbols_file, json_fn
    )

    print(" ... holdingsParams_currentPrice = " + str(holdingsParams_currentPrice))

    '''
    # check for duplicate holdings. Combine duplicates if they exist.
    holdings_symbols = holdingsParams_symbols
    #holdings_symbols = []
    holdings_shares = []
    holdings_buyprice = []
    holdings_currentPrice = []

    for i,val in enumerate(holdingsParams_symbols):
        if holdingsParams_symbols.index(val) == i and holdingsParams_currentPrice[i] != np.nan:
            index = holdingsParams_symbols.index(val)
            #holdings_symbols.append( val )
            holdings_shares.append( holdingsParams_shares[index] )
            holdings_buyprice.append( holdingsParams_buyprice[index] )
            holdings_currentPrice.append( holdingsParams_currentPrice[index] )
        else:
            indexToAdjust = holdings_symbols.index(val)
            holdings_shares[indexToAdjust] += holdingsParams_shares[i]
            holdings_buyprice[indexToAdjust] =   \
                      ( holdingsParams_buyprice[indexToAdjust] * holdingsParams_shares[indexToAdjust] +   \
                      holdingsParams_buyprice[i] * holdingsParams_shares[i] ) /   \
                      holdings_shares[indexToAdjust]
    '''
    # skip combining...
    holdings_symbols = holdingsParams_symbols
    holdings_shares = holdingsParams_shares
    holdings_buyprice = holdingsParams_buyprice
    holdings_currentPrice = holdingsParams_currentPrice

    # parse symbols in current holdings and new selections into buys, sells, and stocks in both lists
    sells = [item for item in holdings_symbols if item not in last_symbols_text]
    buys = [item for item in last_symbols_text if item not in holdings_symbols]
    matches = [item for item in holdings_symbols if item in last_symbols_text]


    print(" holdings_symbols      = ", holdings_symbols)
    print(" holdings_shares       = ", holdings_shares)
    print(" holdings_buyprice     = ", holdings_buyprice)
    print(" holdings_currentPrice = ", holdings_currentPrice)
    profit = 0
    for i in range(len(holdings_symbols)):
        print(" ... i, holdings_symbols[i], holdings_shares[i], holdings_buyprice[i], holdings_currentPrice[i] = ", i, holdings_symbols[i], holdings_shares[i], holdings_buyprice[i], holdings_currentPrice[i])
        profit += float(holdings_shares[i]) * ( float(holdings_currentPrice[i]) - float(holdings_buyprice[i]) )
    print(" holdings profit = ", profit)

    # calculate holdings value
    currentHoldingsValue = 0.
    for i in range(len(holdings_symbols)):
        currentHoldingsValue += float(holdings_shares[i]) * float(holdings_currentPrice[i])

    ##### diagnostics ###################################################################################################
    json_dir = os.path.split(json_fn)[0]
    p_store = get_performance_store(json_fn)
    file_out = os.path.join(p_store, "PyTAAA_diagnostic.params")
    with open(file_out, "a") as holdingsfile:

        holdingsfile.write( str(today) + " \n" )
        holdingsfile.write( "currently held stocks:   " + str(holdings_symbols) +"\n")
        holdingsfile.write( "currently held shares:   " + str(holdings_shares) +"\n")
        holdingsfile.write( "currently held buyprice: " + str(holdings_buyprice) +"\n")
        holdingsfile.write( "currently held nowprice: " + str(holdings_currentPrice) +"\n")
        holdingsfile.write( "new stock selection: " + str(last_symbols_text) +"\n")
        holdingsfile.write( "new stock weight:    " + str(last_symbols_weight) +"\n")
        holdingsfile.write( "new stock nowprice:  " + str(last_symbols_price) +"\n")
    ##### end diagnostics ###############################################################################################


    ####################################################################
    ### check for adjustments to current holdings -- stocks that were in last period and are in now
    ### - apply $400 threshold to changes
    ### find symbols that are held from current holdings
    ####################################################################

    DeltaValue = []
    DeltaValueThresholded = []
    DeltaValueThresholdedNormed = []
    cumuAbsDeltaValue = 0.
    cumuAbsDeltaValueThresholded = 0.

    for i, symbol in enumerate( matches ):
        # calculate the change in number of shares and value
        holdings_index = holdings_symbols.index( matches[i] )
        last_symbols_index = last_symbols_text.index( matches[i] )

        old_numshares = holdings_shares[holdings_index]
        new_numshares = currentHoldingsValue* last_symbols_weight[last_symbols_index] / last_symbols_price[last_symbols_index]
        deltaShares = new_numshares - old_numshares
        DeltaValue.append( deltaShares * last_symbols_price[last_symbols_index] )

        cumuAbsDeltaValue += abs( DeltaValue[-1] )

        # - apply $400 threshold to changes.
        deltaValueTotal = 0
        cumuThresholdedValue = 0.
        if abs(DeltaValue[-1]) < 400 :
            DeltaValueThresholded.append( 0. )
        else:
            DeltaValueThresholded.append( deltaShares * last_symbols_price[last_symbols_index] )
        cumuAbsDeltaValueThresholded += abs( DeltaValueThresholded[-1] )

    print(" matches (symbols) =     ", matches)
    print(" DeltaValue =            ", DeltaValue)
    print(" DeltaValueThresholded = ", DeltaValueThresholded)
    print(" cumuAbsDeltaValue =     ", cumuAbsDeltaValue)
    print(" cumuAbsDeltaValueThresholded = ", cumuAbsDeltaValueThresholded)

    DeltaValueThresholded = np.array( DeltaValueThresholded )
    if DeltaValueThresholded.all() != 0:
        makeChanges = True
    else:
        makeChanges = False

    # get total amount of thresholded delta values
    thresholdingResidual = 0.
    for i, isymbol in enumerate( matches ):
        thresholdingResidual += DeltaValue[i] - DeltaValueThresholded[i]

    # get percent of total abs deltavalue after thresholding and normalize (so it sums to 100%)
    absDeltaPct = []
    cumuAbsDeltaPct = 0.
    for i, isymbol in enumerate( matches ):
        absDeltaPct.append( abs( DeltaValueThresholded[i] ) / cumuAbsDeltaValue )
        cumuAbsDeltaPct += absDeltaPct[-1]
    absDeltaPctNormed = []
    for i, isymbol in enumerate( matches ):
        absDeltaPctNormed.append( absDeltaPct[i] / cumuAbsDeltaPct )


    # Re-normalize deltaValue to have same total change for all held stocks. Convert to shares.
    for i, symbol in enumerate( matches ):
        if makeChanges :
            DeltaValueThresholdedNormed.append( DeltaValueThresholded[i] + absDeltaPctNormed[i] * thresholdingResidual )
            holdings_index = holdings_symbols.index( matches[i] )
            last_symbols_index = last_symbols_text.index( matches[i] )
            numDeltaShares = DeltaValueThresholdedNormed[i]/last_symbols_price[last_symbols_index]
            last_symbols_deltashares_normed = int( abs(numDeltaShares) ) * np.sign( numDeltaShares )
            cumuValueAfterExchanges += float( last_symbols_deltashares_normed + holdings_shares[holdings_index] ) * last_symbols_price[last_symbols_index]
            print(" symbol, numDeltaShares = ", last_symbols_text[last_symbols_index], numDeltaShares)
            print(" cumValueAfterExchanges parts = ", last_symbols_deltashares_normed, holdings_shares[holdings_index], last_symbols_price[last_symbols_index])

            # calculate effective (average) purchase price for all shares after exchange
            value = float(holdings_shares[holdings_index]) * float(holdings_buyprice[holdings_index]) + last_symbols_deltashares_normed*float(last_symbols_price[last_symbols_index])
            if symbol != "CASH" and last_symbols_deltashares_normed != 0:
                trade_symbols.append( symbol )
                trade_shares.append( last_symbols_deltashares_normed )
                buySellCost += BuySellFee
            if symbol != "CASH" and holdings_shares[holdings_index] + last_symbols_deltashares_normed != 0:
                shares = holdings_shares[holdings_index] + last_symbols_deltashares_normed
                shares = int( shares )
                new_symbols.append( symbol )
                new_shares.append( shares )
                buy_price = value / new_shares[-1]
                buy_price = round( buy_price, 2 )
                new_buyprice.append( buy_price )
                buySellCost += BuySellFee
        else:
            new_symbols.append( symbol )
            new_shares.append( holdings_shares[i] )
            new_buyprice.append( holdings_buyprice[i] )


    ####################################################################
    ### check for sells -- stocks that were in last period and out now
    ####################################################################

    for i, symbol in enumerate( sells ):
        holdings_index = holdings_symbols.index( sells[i] )
        if symbol != "CASH":
            trade_symbols.append( symbol )
            trade_shares.append( -holdings_shares[holdings_index] )
            buySellCost += BuySellFee


    ####################################################################
    ### check for buys -- stocks that were out last period and in now
    ### - apply $800 threshold
    ####################################################################
    cumuNewValue = 0.
    cumuNewValueThresholded = 0.
    for i, symbol in enumerate( buys ):
        last_symbols_index = last_symbols_text.index( buys[i] )
        new_value = currentHoldingsValue * last_symbols_weight[i]
        cumuNewValue += new_value
        if new_value < 800.:
            new_value = 0.
        cumuNewValueThresholded += new_value

    weightBuysNormed = []
    for i, symbol in enumerate( buys ):
        last_symbols_index = last_symbols_text.index( buys[i] )
        new_value = currentHoldingsValue * last_symbols_weight[last_symbols_index]
        weightBuysNormed = last_symbols_weight[last_symbols_index] * cumuNewValueThresholded / cumuNewValue
        new_valueNormed = currentHoldingsValue * weightBuysNormed
        if new_value > 800. and symbol != "CASH":
            #print " inside Buys .... symbol, new_value, new_valueNormed, shares = ", symbol, new_value, new_valueNormed, int( new_valueNormed / last_symbols_price[last_symbols_index] )
            trade_symbols.append( symbol )
            trade_shares.append( int( new_valueNormed / last_symbols_price[last_symbols_index] ) )
            cumuValueAfterExchanges += ( trade_shares[-1] * last_symbols_price[last_symbols_index] )
            new_symbols.append( symbol )
            shares = int( trade_shares[-1] )
            buy_price = last_symbols_price[last_symbols_index]
            buy_price = round( buy_price, 2 )
            new_shares.append( shares )
            new_buyprice.append( buy_price )
            buySellCost += BuySellFee

    ####################################################################
    ### adjust CASH balance
    ### - Sum value of all new holdings (after thresholding,
    ###   after sells and buys, after adjustments to stocks being held from last period)
    ####################################################################
    cumuValueAfterExchanges = 0.
    cashindex = holdings_symbols.index("CASH")
    for i, symbol in enumerate( new_symbols ):
        if symbol != "CASH":
            cumuValueAfterExchanges += float(new_shares[i]) * float(new_buyprice[i])
    cash_bal = currentHoldingsValue - cumuValueAfterExchanges

    if makeChanges :
        new_symbols.append( "CASH" )
        new_shares.append( round( cash_bal,2 ) )
        new_buyprice.append( 1.0 )

    holdings_cash_bal = holdings_shares[cashindex]
    if makeChanges :
        trade_symbols.append( "CASH" )
        trade_shares.append( round( cash_bal - holdings_cash_bal, 2 ) )

    cash_bal -= buySellCost

    ####################################################################
    ### prepare messages for stocks purchases and sales
    ### - put informational messages in email
    ### - if this is a trading day, put new holdings in file PyTAAA_holdings.params
    ####################################################################
    for i in range(len(trade_symbols)):
        if trade_shares[i] < 0:
            # append sells messages
            trade_message = trade_message + "<p>Sell  " + str(trade_symbols[i]) +" "+ str(trade_shares[i])+"</p>"
        else:
            # append buys messages
            trade_message = trade_message + "<p>Buy  " + str(trade_symbols[i]) +" "+ str(trade_shares[i])+"</p>"
    if 'Buy' in trade_message or 'Sell' in trade_message:
        trade_message = trade_message + "<br>"
        trade_message = trade_message + "<p>Transaction Fees Applied to Model  $" + str(buySellCost) +"</p>"
        trade_message = trade_message + "<br>"

    # Determine if this is a trade-date, and if so, write new buys to PyTAAA_holdings.params
    # - based on day of month and whether market is open or closed
    # - add to existing file without deleting old entries
    # - note that threshold already applied to ignore small changes to stocks held from prior period
    marketOpen, lastDayOfMonth = CheckMarketOpen()
    if lastDayOfMonth and makeChanges:
        if not marketOpen:

            p_store = get_performance_store(json_fn)
            holdings_fn = os.path.join(p_store, "PyTAAA_holdings.params")
            with open(holdings_fn, "a") as holdingsfile:
                new_symbols_str = ""
                new_shares_str = ""
                new_buyprice_str = ""
                for i in range( len(new_symbols) ):
                    new_symbols_str = new_symbols_str + str(new_symbols[i]) + " "
                    new_shares_str = new_shares_str + str(new_shares[i]) + " "
                    new_buyprice_str = new_buyprice_str + str(new_buyprice[i]) + " "

                holdingsfile.write( " \n" )
                holdingsfile.write( "TradeDate: " + str(today).split(" ")[0] +"\n")
                holdingsfile.write( "stocks: " + new_symbols_str +"\n")
                holdingsfile.write( "shares: " + new_shares_str +"\n")
                holdingsfile.write( "buyprice: " + new_buyprice_str +"\n")
                holdingsfile.write( "commissons: " + str(buySellCost) +"\n")


    print("")
    print("holdings_symbols = ", holdings_symbols)
    print("holdings_shares = ", holdings_shares)
    print("last_symbols_text = ", last_symbols_text)
    print("last_symbols_price = ", last_symbols_price)


    return trade_message


def trade_today(json_fn, symbols_today, weight_today, price_today):

    from functions.GetParams import (
        get_holdings, get_symbols_file, get_performance_store
    )

    # get holdings, p_store folder
    holdings = get_holdings(json_fn)
    symbols_file = get_symbols_file(json_fn)
    p_store = get_performance_store(json_fn)


    trade_date = str(datetime.datetime.now()).split(" ")[0]

    # Get updated Holdings from file (ranks are updated)
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

    unique_symbols = list(set(holdings_symbols))
    holdings_currentPrice = LastQuotesForSymbolList_hdf(
        unique_symbols, symbols_file, json_fn
    )
    print("holdings_symbols = ", holdings_symbols)
    print("holdings_shares = ", holdings_shares)
    print("holdings_currentPrice = ", holdings_currentPrice)

    # calculate holdings total value
    currentHoldingsValue = 0.
    for j, _symbol in enumerate(holdings_symbols):
        for i, held_symbol in enumerate(unique_symbols):
            if held_symbol == _symbol:
                currentHoldingsValue += (
                    float(holdings_shares[j]) * float(holdings_currentPrice[i])
                )

    # set up lists if trades executed today
    today_symbols = []
    today_shares = []
    today_buyprice = []

    # find unchanged holdings
    unchanged_symbols = [x for x in holdings_symbols if x in symbols_today]

    # find new symbols
    new_symbols = [x for x in symbols_today if x not in holdings_symbols]

    # find dropped symbols
    drop_symbols = [x for x in holdings_symbols if x not in symbols_today]


    print(" ... unchanged_symbols = " + str(unchanged_symbols))
    print(" ... new_symbols = " + str(new_symbols))
    print(" ... drop_symbols = " + str(drop_symbols))

    # compute shares_today
    shares_today = []
    for i, t_symbol in enumerate(symbols_today):
        today_value = weight_today[i] * currentHoldingsValue
        shares_today.append(int(today_value / price_today[i]))

    # process changes
    shares_residual = np.array(shares_today).astype('int')
    weight_residual = np.array(weight_today).astype('float32')
    buy_text = ""
    sell_text = ""
    cumu_value = 0.0
    for i, h_symbol in enumerate(holdings_symbols):
        # if i == 5:
        #     break
        if h_symbol in unchanged_symbols:
            today_index = symbols_today.index(h_symbol)
            today_value = weight_residual[today_index] * currentHoldingsValue
            if holdings_shares[i] == shares_residual[today_index]:
                # keep shares from last month. no transactions
                today_symbols.append(holdings_symbols[i])
                today_buyprice.append(holdings_buyprice[i])
                today_shares.append(int(holdings_shares[i]))
                shares_residual[today_index] -= holdings_shares[i]
                # cumu_value += today_buyprice[-1] * today_shares[-1]
                # weight_residual[today_index] = 0.0
                last_price = price_today[today_index]
                cumu_value += last_price * today_shares[-1]
            elif 0 < shares_residual[today_index] < holdings_shares[i]:
                # need to sell some, but not all shares
                today_symbols.append(holdings_symbols[i])
                today_buyprice.append(holdings_buyprice[i])
                # today_value_ = weight_residual[today_index] * currentHoldingsValue
                today_shares_ = int(today_value / price_today[today_index])
                sell_shares = int(holdings_shares[i] - today_shares_)
                shares_residual[today_index] = (holdings_shares[i] - sell_shares)
                weight_residual[today_index] -= (shares_residual[today_index] * today_buyprice[-1]) / currentHoldingsValue
                today_shares.append(today_shares_)
                shares_residual[today_index] -= today_shares_
                # cumu_value += today_buyprice[-1] * today_shares[-1]
                last_price = price_today[today_index]
                cumu_value += last_price * today_shares[-1]
                sell_text = sell_text + "info:  Sell " + trade_date + " " + holdings_symbols[i].ljust(6)
                sell_text = sell_text + format(-int(sell_shares), '7d')
                sell_text = sell_text + format(price_today[today_index], '8.2f')
                sell_text = sell_text + "\n"
            elif shares_residual[today_index] >= holdings_shares[i]:
                # keep shares from last month
                today_symbols.append(holdings_symbols[i])
                today_buyprice.append(holdings_buyprice[i])
                today_shares.append(int(holdings_shares[i]))
                shares_residual[today_index] -= holdings_shares[i]
                weight_residual[today_index] -= (int(holdings_shares[i]) * holdings_buyprice[today_index]) / currentHoldingsValue
                last_price = price_today[today_index]
                cumu_value += last_price * today_shares[-1]
            elif shares_residual[today_index] <= 0:
                sell_shares = holdings_shares[i] - shares_residual[today_index]
                sell_text = sell_text + "info:  Sell " + trade_date + " " + holdings_symbols[i].ljust(6)
                sell_text = sell_text + format(-sell_shares, '7d')
                sell_text = sell_text + format(price_today[today_index], '8.2f')
                sell_text = sell_text + "\n"

        if h_symbol in drop_symbols and h_symbol != "CASH":
            today_index = list(set(holdings_symbols)).index(h_symbol)
            sell_shares = int(holdings_shares[i])
            print(" ... sell holdings symbol " + holdings_symbols[i])
            print(" ... sell holdings shares " + str(sell_shares))
            sell_text = sell_text + "info:  Sell " + trade_date + " " + holdings_symbols[i].ljust(6)
            sell_text = sell_text + format(-sell_shares, '7d')
            sell_text = sell_text + format(holdings_currentPrice[today_index], '8.2f')
            sell_text = sell_text + "\n"

    # existing symbols having added shares today
    for i, t_symbol in enumerate(symbols_today):
        if shares_residual[i] > 0 and t_symbol in holdings_symbols:
            today_symbols.append(t_symbol)
            buy_shares = int(shares_residual[i])
            today_shares.append(buy_shares)
            today_buyprice.append(price_today[i])
            cumu_value += today_buyprice[-1] * today_shares[-1]
            buy_text = buy_text + "info:  Buy  " + trade_date + " " + t_symbol.ljust(6)
            buy_text = buy_text + format(buy_shares, '7d')
            buy_text = buy_text + format(price_today[i], '8.2f')
            buy_text = buy_text + "\n"

    for j, n_symbol in enumerate(new_symbols):

        today_index = symbols_today.index(n_symbol)
        today_symbols.append(n_symbol)
        buy_shares = int(weight_today[today_index] * currentHoldingsValue / price_today[today_index])
        today_shares.append(buy_shares)
        today_buyprice.append(price_today[today_index])
        cumu_value += today_buyprice[-1] * today_shares[-1]
        buy_text = buy_text + "info:  Buy  " + trade_date + " " + n_symbol.ljust(6)
        buy_text = buy_text + format(buy_shares, '7d')
        buy_text = buy_text + format(price_today[today_index], '8.2f')
        buy_text = buy_text + "\n"

    # add cash
    today_symbols.append("CASH")
    today_shares.append(int(currentHoldingsValue - cumu_value + 0.5))
    today_buyprice.append(1.00)

    # price new holdings text
    holdings_text = "stocks:      "
    for i, _symbol in enumerate(today_symbols):
        holdings_text = holdings_text + _symbol.ljust(8)
    holdings_text = holdings_text + "\nshares:      "
    for i, _share in enumerate(today_shares):
        holdings_text = holdings_text + str(_share).ljust(8)
    holdings_text = holdings_text + "\nbuyprice:    "
    for i, _buyprice in enumerate(today_buyprice):
        holdings_text = holdings_text + str(np.round(_buyprice,2)).ljust(8)

    # print and write to file
    hypthetical_trade_info = "\n\nTradeDate: " + trade_date + "\n"
    hypthetical_trade_info = hypthetical_trade_info + sell_text
    hypthetical_trade_info = hypthetical_trade_info + buy_text
    hypthetical_trade_info = hypthetical_trade_info + holdings_text
    
    print(hypthetical_trade_info)

    filepath = os.path.join(p_store, "PyTAAA_hypothetical_trades.txt" )
    with open( filepath, "w" ) as f:
        f.write(hypthetical_trade_info)
        f.write("\n")
