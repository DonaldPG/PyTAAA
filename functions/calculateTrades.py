import os
import numpy as np
import datetime
import pandas as pd
#from functions.quotes_for_list_adjCloseVol import *
from functions.quotes_for_list_adjClose import LastQuotesForSymbolList_hdf
from functions.CheckMarketOpen import *
from functions.GetParams import (
    get_json_params, get_symbols_file, get_performance_store
)

# Set print options to suppress scientific notation and control precision
np.set_printoptions(suppress=True, precision=2)

###
### Perform a check to see if the stock market is open
### - purpose is to stop calculating and updating when nothing has changed
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
    # Get parameters
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
    print( str(today))
    print( "currently held stocks:   " + str(holdings_symbols))
    print( "currently held shares:   " + str(holdings_shares))
    print( "currently held buyprice: " + str(holdings_buyprice))
    print( "currently held nowprice: " + str(np.array(holdings_currentPrice)))
    print( "new stock selection: " + str(last_symbols_text))
    print( "new stock weight:    " + str(last_symbols_weight))
    print( "new stock nowprice:  " + str(last_symbols_price))
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
        DeltaValue.append(
            float(np.round(deltaShares * last_symbols_price[last_symbols_index], 2))
        )

        cumuAbsDeltaValue += abs( DeltaValue[-1] )

        # - apply $400 threshold to changes.
        deltaValueTotal = 0
        cumuThresholdedValue = 0.
        if abs(DeltaValue[-1]) < 400 :
            DeltaValueThresholded.append(0.)
        else:
            DeltaValueThresholded.append(
                float(np.round(deltaShares * last_symbols_price[last_symbols_index], 2))
            )
        cumuAbsDeltaValueThresholded += abs( DeltaValueThresholded[-1] )

    print(" matches (symbols) =     ", matches)
    print(" DeltaValue =            ", DeltaValue)
    print(" DeltaValueThresholded = ", DeltaValueThresholded)
    print(" cumuAbsDeltaValue =     ", float(np.round(cumuAbsDeltaValue,2)))
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

    # for i, symbol in enumerate( sells ):
    #     holdings_index = holdings_symbols.index( sells[i] )
    #     if symbol != "CASH":
    #         trade_symbols.append( symbol )
    #         trade_shares.append( -holdings_shares[holdings_index] )
    #         buySellCost += BuySellFee

    holdings_indices = []
    for i, symbol in enumerate(holdings_symbols):
        if symbol in sells:
            holdings_indices.append(i)
    for i, symbol in enumerate( sells ):
        if symbol != "CASH":
            holdings_index = holdings_indices[i]
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
    ### - put informational messages in output
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


def trade_today(json_fn, symbols_today, weight_today, price_today, verbose=False):

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

    symbols_file = get_symbols_file(json_fn)
    # holdings_currentPrice = LastQuotesForSymbolList_hdf(
    #     holdingsParams_symbols, symbols_file, json_fn
    # )
    holdings_currentPrice = LastQuotesForSymbolList_hdf(
        holdings["stocks"], symbols_file, json_fn
    )

    # put holdings in pandas datafroame to get total shares for held symbols
    holdings_symbols = holdings['stocks']
    holdings_shares = np.array(holdings['shares']).astype("float32").astype("int")
    holdings_buyprice = np.array(holdings['buyprice']).astype("float32")
    current_holdings_dict = {
        "symbols": holdings_symbols,
        "shares": holdings_shares,
        "buyprice": holdings_buyprice,
        "price": holdings_currentPrice
    }
    held_df = pd.DataFrame(current_holdings_dict)
    # held_df = held_df[held_df['symbols'] != "CASH"]
    held_grouped_df = held_df.groupby('symbols')['shares'].sum().reset_index()

    # test if all buyprice are same as price. if so, don't compute trades
    price_diff = np.abs(held_df["buyprice"] - held_df["price"])
    print("   . abs price_diff = " + str(price_diff))
    print("   . abs price_diff.sum() = " + str(price_diff.sum()))
    if price_diff.sum() <= 0.10:
        return

    if verbose:
        print("\nheld_df:\n" + str(held_df))
        print("\nheld_grouped_df:\n" + str(held_grouped_df))

    # calculate holdings value
    currentHoldingsValue = 0.
    for i in range(len(holdings_symbols)):
        currentHoldingsValue += float(holdings_shares[i]) * float(holdings_currentPrice[i])

    # do same for target holdings
    new_shares = []
    new_buy_price = []
    cumu_new_value = 0.0
    for i, _weight in enumerate(weight_today):
        new_shares.append(
            int(currentHoldingsValue * _weight / price_today[i])
        )
        new_buy_price.append(price_today[i])
        cumu_new_value += new_shares[-1] * price_today[i]
    cash_value = int(currentHoldingsValue - cumu_new_value + 0.5)
    symbols_today.append("CASH")
    price_today.append(1.00)
    new_buy_price.append(1.00)
    new_shares.append(cash_value)

    new_holdings_dict = {
        "symbols": symbols_today,
        "shares": np.array(new_shares).astype("int"),
        "buyprice": new_buy_price,
        "price": new_buy_price
    }
    new_df = pd.DataFrame(new_holdings_dict)
    if verbose:
        print("\nnew_df:\n" + str(new_df))

    df1 = held_df
    df2 = new_df
    # Sum shares for each symbol in df1
    df1_sum = df1.groupby('symbols', as_index=False).agg(
        {'shares': 'sum', 'buyprice': 'first', 'price': 'first'}
    )

    # Merge the two DataFrames on 'symbols'
    merged_df = pd.merge(df1_sum, df2, on='symbols', how='outer', suffixes=('_df1', '_df2'))

    # Calculate the differences in shares
    merged_df['shares_diff'] = merged_df['shares_df2'].fillna(0) - merged_df['shares_df1'].fillna(0)
    if verbose:
        print("\nmerged_df:\n" + str(merged_df))

    ####################################################################
    ### check for sells -- stocks that were in last period and out now
    ####################################################################
    buySellCost = 0.
    BuySellFee = 4.95
    new_symbols = []
    new_shares = []
    new_buyprice = []
    trade_symbols = []
    trade_shares = []
    trade_price = []
    trade_activity = []
    trade_buyprice = []

    # sell entire holding
    sells = merged_df[merged_df['shares_df1'] > 0 ]
    sells = sells[np.isnan(sells['shares_df2'])]
    if len(list(sells.symbols)) > 0:
        for i, symbol in enumerate(list(sells.symbols)):
            trade_symbols.append( symbol )
            trade_shares.append(int(sells.shares_diff.values[i]))
            trade_price.append(float(np.round(sells.price_df1.values[i],2)))
            trade_buyprice.append(np.round(sells.buyprice_df1.values[i],2))
            trade_activity.append("sell")
            buySellCost += BuySellFee

    # decrease existing holding
    sellsome = merged_df[merged_df['shares_diff'] < 0 ]
    sellsome = sellsome[sellsome['shares_df1'] > 0]
    sellsome = sellsome[sellsome['shares_df2'] > 0]
    sellsome = sellsome[sellsome["symbols"] != "CASH"]

    sellsome_held = held_df[held_df['symbols'].isin(sellsome['symbols'])]
    sellsome = sellsome.set_index('symbols')

    if len(list(sellsome_held.symbols)) > 0:
        for i, symbol in enumerate(list(sellsome_held.symbols)):

            shares_to_sell = min(
                sellsome[
                    sellsome.index == symbol
                ].loc[:,"shares_diff"].values[0] * -1,
                sellsome_held.loc[:, "shares"].iloc[i]
            )

            sellsome.loc[symbol, "shares_diff"] = shares_to_sell + \
                sellsome.loc[symbol, "shares_diff"]

            trade_symbols.append( symbol )
            trade_shares.append(int(-1 * shares_to_sell))
            trade_buyprice.append(np.round(sellsome_held.buyprice.values[i],2))
            trade_price.append(float(np.round(sellsome_held.price.values[i],2)))
            trade_activity.append("reduce")

            buySellCost += BuySellFee

    # buy non-held company
    buys = merged_df[merged_df['shares_df2'] > 0 ]
    buys = buys[np.isnan(buys['shares_df1'])]
    if len(list(buys.symbols)):
        for i, symbol in enumerate(list(buys.symbols)):
            trade_symbols.append( symbol )
            trade_shares.append(int(buys.shares_diff.values[i]))
            trade_price.append(float(np.round(buys.price_df2.values[i],2)))
            trade_buyprice.append(float(np.round(buys.buyprice_df2.values[i],2)))
            trade_activity.append("buy")
            buySellCost += BuySellFee

    # increase existing holding
    buymore = merged_df[merged_df['shares_diff'] > 0 ]
    buymore = buymore[buymore['shares_df1'] > 0]
    buymore = buymore[buymore["symbols"] != "CASH"]
    if len(list(buymore.symbols)) > 0:
        for i, symbol in enumerate(list(buymore.symbols)):
            trade_symbols.append( symbol )
            trade_shares.append(int(buymore.shares_diff.values[i]))
            trade_price.append(float(np.round(buymore.price_df2.values[i],2)))
            trade_buyprice.append(float(np.round(buymore.buyprice_df2.values[i],2)))
            trade_activity.append("increase")
            buySellCost += BuySellFee

    # hold unchanged
    # hold = merged_df[merged_df['shares_diff'] == 0 ]
    # hold = hold[hold['shares_df1'] > 0]
    # hold = hold[hold["symbols"] != "CASH"]
    # if len(list(hold.symbols)) > 0:
    #     for i, symbol in enumerate(list(hold.symbols)):
    #         trade_symbols.append( symbol )
    #         trade_shares.append(0)
    #         trade_price.append(float(np.round(hold.price_df1.values[i],2)))
    #         trade_buyprice.append(float(np.round(hold.buyprice_df1.values[i],2)))
    #         trade_activity.append("hold")
    #         buySellCost += BuySellFee

    # hold existing holding
    holdsome = merged_df[merged_df['shares_diff'] == 0 ]
    holdsome = holdsome[holdsome['shares_df1'] > 0]
    holdsome = holdsome[holdsome['shares_df2'] > 0]
    holdsome = holdsome[holdsome["symbols"] != "CASH"]

    holdsome_held = held_df[held_df['symbols'].isin(holdsome['symbols'])]
    holdsome = holdsome.set_index('symbols')

    if len(list(holdsome_held.symbols)) > 0:
        for i, symbol in enumerate(list(held_df.symbols)):

            if symbol == "CASH":
                continue

            if symbol in list(holdsome.index):
                held_df_row = held_df.iloc[i]

                print("\n" + str(held_df_row))

                trade_symbols.append(symbol)
                trade_shares.append(0)
                trade_buyprice.append(np.round(held_df_row["buyprice"],2))
                trade_price.append(np.round(held_df_row["price"],2))
                trade_activity.append("hold")

                buySellCost += BuySellFee




    # handle cash
    symbol = "CASH"
    trade_symbols.append( symbol )
    trade_shares.append(int(merged_df[merged_df["symbols"] == symbol].shares_diff.values[0]))
    trade_price.append(1.0)
    trade_buyprice.append(1.0)
    if trade_shares[-1] > 0:
            trade_activity.append("increase")
    elif trade_shares[-1] < 0:
            trade_activity.append("reduce")
    elif trade_shares[-1] == 0:
            trade_activity.append("hold")

    # re-evaluate activity
    trade_activity = []
    for i in range(len(trade_shares)):
        if trade_shares[i] > 0:
                trade_activity.append("increase")
        elif trade_shares[i] < 0:
                trade_activity.append("reduce")
        elif trade_shares[i] == 0:
                trade_activity.append("hold")

    # put trades in dataframe
    trades_dict = {
        "symbols": trade_symbols,
        "shares": trade_shares,
        "buyprice": trade_buyprice,
        "price": trade_price,
        "activity": trade_activity
    }
    trades_df = pd.DataFrame(trades_dict)
    if verbose:
        print("\ntrades_df:\n" + str(trades_df))

    # # merge with held_df
    # held_df2 = held_df
    # activity_list = []
    # for _row_i in range(held_df.values.shape[0]):
    #     # row = held_df.values[_row_i,:]
    #     _symb = held_df.iloc[_row_i, :]["symbols"]
    #     trades_activity = trades_df.loc[trades_df["symbols"] == _symb]["activity"]
    #     activity_list.append(trades_activity.values[0])
    # held_df2['activity'] = activity_list


    trades_df_wo_buy = trades_df[trades_df["activity"] != "buy"]
    # trades_df_wo_buy = held_df2[held_df2["activity"] != "buy"]
    if verbose:
        print("\nheld_df.columns:\n" + str(held_df.columns))
        print("\nheld_df:\n" + str(held_df))
        print("\ntrades_df_wo_buy.columns:\n" + str(trades_df_wo_buy.columns))
        print("\ntrades_df_wo_buy:\n" + str(trades_df_wo_buy))

    try:
        # merged_trades_df = pd.merge(
        #     held_df, trades_df_wo_buy, on=['symbols','buyprice'], how='outer',
        #     suffixes=('_held', '_trade'), validate="one_to_one"
        # )

        # merged_trades_df = pd.merge(
        #     held_df, trades_df_wo_buy, on=['symbols','buyprice'], how='outer',
        #     suffixes=('_held', '_trade'), validate="one_to_one"
        # )

        merged_trades_df = pd.merge(
            held_df, trades_df_wo_buy, on=['symbols','buyprice'], how='outer',
            suffixes=('_held', '_trade')
        )

        # fill in NaN's in merged_trades_df
        nan_shares = merged_trades_df[pd.isna(merged_trades_df["activity"])]
        nan_shares = nan_shares[nan_shares["symbols"] != "CASH"]

        symbols_list = []
        shares_held_list = []
        buyprice_list = []
        price_held_list = []
        shares_trade_list = []
        price_trade_list = []
        activity_list = []
        if len(list(nan_shares.symbols)) > 0:
            for i, symbol in enumerate(list(nan_shares.symbols)):
                df_row = trades_df[trades_df["symbols"] == symbol]
                trades_df_shares = float(df_row["shares"].values[0])
                trades_df_buyprice = float(df_row["buyprice"].values[0])
                trades_df_price = float(df_row["price"].values[0])
                trades_df_activity = str(df_row["activity"].values[0])

                print(
                    " symbol, trades_df_price, trades_df_price = " + \
                    str((symbol, trades_df_shares, trades_df_activity, trades_df_buyprice, trades_df_price))
                )

                if trades_df_activity == "sell":
                    # sell
                    nan_row = nan_shares.iloc[i]
                    nan_df_shares = int(nan_row["shares_held"])
                    nan_df_buyprice = float(np.round(nan_row["buyprice"], 2))
                    nan_df_price_held = trades_df_price
                    nan_df_shares_trade = -nan_df_shares
                    nan_df_price_trade = trades_df_price

                    symbols_list.append(symbol)
                    shares_held_list.append(nan_df_shares)
                    buyprice_list.append(nan_df_buyprice)
                    price_held_list.append(nan_df_price_held)
                    shares_trade_list.append(nan_df_shares_trade)
                    price_trade_list.append(nan_df_price_trade)
                    activity_list.append(trades_df_activity)

                elif trades_df_activity == "increase":
                    # purchase more shares
                    nan_row = nan_shares.iloc[i]
                    nan_df_shares = int(nan_row["shares_held"])
                    nan_df_buyprice = float(np.round(nan_row["buyprice"], 2))
                    nan_df_price_held = trades_df_price
                    nan_df_shares_trade = 0.0
                    nan_df_price_trade = trades_df_price

                    symbols_list.append(symbol)
                    shares_held_list.append(nan_df_shares)
                    buyprice_list.append(nan_df_buyprice)
                    price_held_list.append(nan_df_price_held)
                    shares_trade_list.append(nan_df_shares_trade)
                    price_trade_list.append(nan_df_price_trade)
                    activity_list.append(trades_df_activity)

        # put in dict, then dataframe
        nan_dict = {
            "symbols": symbols_list,
            "shares_held": shares_held_list,
            "buyprice": buyprice_list,
            "price_held": price_held_list,
            "shares_trade": shares_trade_list,
            "price_trade": price_trade_list,
            "activity": activity_list,
        }
        yes_nan_df = pd.DataFrame(nan_dict)

        # iterate through df with no NaNs.
        # make "shares_trade" and "activity" consistent with values in df
        no_nan_shares = merged_trades_df[~pd.isna(merged_trades_df["activity"])]

        symbols_list = []
        shares_held_list = []
        buyprice_list = []
        price_held_list = []
        shares_trade_list = []
        price_trade_list = []
        activity_list = []
        if len(list(no_nan_shares.symbols)) > 0:
            for i, symbol in enumerate(list(no_nan_shares.symbols)):

                nonan_df_row = no_nan_shares.iloc[i]

                nonan_df_shares_held = float(nonan_df_row["shares_held"])
                nonan_df_buyprice = float(nonan_df_row["buyprice"])
                nonan_df_price_held = float(nonan_df_row["price_held"])
                nonan_df_shares_trade = float(nonan_df_row["shares_trade"])
                nonan_df_price_trade = str(nonan_df_row["price_trade"])
                nonan_df_activity = str(nonan_df_row["activity"])

                if nonan_df_activity == "reduce" and nonan_df_shares_trade == 0.0:
                    nonan_df_activity = "hold"

                if (
                    nonan_df_activity == "sell" and \
                    nonan_df_shares_trade < -nonan_df_shares_held
                ):
                    nonan_df_shares_trade = -nonan_df_shares_held

                # if (
                #     nonan_df_activity == "increase" and \
                #     not np.isnan(nonan_df_shares_held)
                # ):
                #     nonan_df_shares_trade = 0.0


                symbols_list.append(symbol)
                shares_held_list.append(nonan_df_shares_held)
                buyprice_list.append(nonan_df_buyprice)
                price_held_list.append(nonan_df_price_held)
                shares_trade_list.append(nonan_df_shares_trade)
                price_trade_list.append(nonan_df_price_trade)
                activity_list.append(nonan_df_activity)

        # put in dict, then dataframe
        no_nan_dict = {
            "symbols": symbols_list,
            "shares_held": shares_held_list,
            "buyprice": buyprice_list,
            "price_held": price_held_list,
            "shares_trade": shares_trade_list,
            "price_trade": price_trade_list,
            "activity": activity_list,
        }
        no_nan_df = pd.DataFrame(no_nan_dict)

        # merge the 2 processed df's
        merged_trades_df = pd.concat([yes_nan_df, no_nan_df], ignore_index=True)

    except:
        print("\nheld_df:\n" + str(held_df))
        print("\ntrades_df_wo_buy:\n" + str(trades_df_wo_buy))
        print(
            "\n\n\n ************************* Error\n" + \
            "   . calculateTrades.py line 606, in trade_today " + \
            "   . pandas.errors.MergeError: " + \
            "Merge keys are not unique in left dataset; not a one-to-one merge"
        )
        return
    # # set NaNs in shares to zero and compute number of shares after monthly update
    # merged_trades_df['shares_target'] = merged_trades_df[
    #     ['shares_held', 'shares_trade']
    # ].fillna(0).sum(axis=1)

    merged_trades_df2 = merged_trades_df.copy()
    merged_trades_df2['shares_target'] = merged_trades_df2[
        ['shares_held', 'shares_trade']
    ].fillna(0).sum(axis=1)

    # # check and make corrections for mutiple held entries
    # merged_trades_grouped_df = merged_trades_df.groupby(
    #         'symbols'
    #     )['shares_target'].sum().reset_index()
    # merge grouped and ungrouped versions
    # merged_trades_df2 = merged_trades_df.copy()
    # revised_target_list = []
    # for _row_i in range(merged_trades_df.values.shape[0]):
    #     _symb = merged_trades_df.iloc[_row_i, :]["symbols"]
    #     _shares_target = merged_trades_grouped_df.loc[merged_trades_grouped_df["symbols"] == _symb]["shares_target"]
    #     revised_target_list.append(_shares_target.values[0])
    # merged_trades_df2['shares_target'] = revised_target_list

    trades_df_only_buy = trades_df[trades_df["activity"] == "buy"]

    if verbose:
        print("\nmerged_trades_df:\n" + str(merged_trades_df2))
        print("\ntrades_df_only_buy:\n" + str(trades_df_only_buy))

    # price new holdings text
    holdings_text = "stocks:      "

    # stocks that were held both last month and this month
    for i, _symbol in enumerate(merged_trades_df2.loc[:]["symbols"]):
        if merged_trades_df2.loc[i]["shares_target"] > 0.0:
            holdings_text = holdings_text + _symbol.ljust(8)
    for i, _symbol in enumerate(trades_df_only_buy.loc[:]["symbols"]):
            holdings_text = holdings_text + _symbol.ljust(8)
    holdings_text = holdings_text + "\nshares:      "
    for i, _share in enumerate(merged_trades_df2.loc[:]["shares_target"]):
        if merged_trades_df2.loc[i]["shares_target"] > 0.0:
            holdings_text = holdings_text + str(_share).ljust(8)
    for i, _share in enumerate(trades_df_only_buy.loc[:]["shares"]):
            holdings_text = holdings_text + str(_share).ljust(8)
    holdings_text = holdings_text + "\nbuyprice:    "
    for i, _buyprice in enumerate(merged_trades_df2.loc[:]["buyprice"]):
        if merged_trades_df2.loc[i]["shares_target"] > 0.0:
            holdings_text = holdings_text + str(np.round(_buyprice,2)).ljust(8)
    for i, _buyprice in enumerate(trades_df_only_buy.loc[:]["buyprice"]):
            holdings_text = holdings_text + str(np.round(_buyprice,2)).ljust(8)

    print(holdings_text)


    # stocks that were held both last month and this month
    sell_text = ""
    for i, _symbol in enumerate(trades_df.loc[:]["symbols"]):
        if _symbol == "CASH":
            continue
        if trades_df.loc[i]["shares"] >= 0.0:
            continue

        _sell_shares = trades_df.loc[i, "shares"]
        _sell_price = trades_df.loc[i, "price"]
        if verbose:
            print(" ... sell holdings symbol " + _symbol)
            print(" ... sell holdings shares " + str(_sell_shares))
        sell_text = sell_text + "info:  Sell " + trade_date + " " + _symbol.ljust(6)
        sell_text = sell_text + format(_sell_shares, '7d')
        sell_text = sell_text + format(_sell_price, '8.2f')
        sell_text = sell_text + "\n"

    buy_text = ""
    for i, _symbol in enumerate(trades_df.loc[:]["symbols"]):
        if _symbol == "CASH":
            continue
        if trades_df.loc[i]["shares"] <= 0.0:
            continue
        _buy_shares = trades_df.loc[i]["shares"]
        _buy_price = trades_df.loc[i]["price"]
        if verbose:
            print(" ... buy holdings symbol " + _symbol)
            print(" ... buy holdings shares " + str(_buy_shares))
        buy_text = buy_text + "info:  Buy  " + trade_date + " " + _symbol.ljust(6)
        buy_text = buy_text + format(_buy_shares, '7d')
        buy_text = buy_text + format(_buy_price, '8.2f')
        buy_text = buy_text + "\n"

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


