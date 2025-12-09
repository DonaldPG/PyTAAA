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
    ### PRESERVE INDIVIDUAL HOLDINGS - DO NOT AGGREGATE
    ####################################################################

    # Process each individual holding separately to preserve buy prices and dates
    for i, symbol in enumerate(holdings_symbols):
        if symbol in matches and symbol != "CASH":
            # This individual holding is in both current and target
            last_symbols_index = last_symbols_text.index(symbol)
            
            # Calculate target shares for this symbol (total across all holdings)
            target_total_shares = int(currentHoldingsValue * last_symbols_weight[last_symbols_index] / last_symbols_price[last_symbols_index])
            
            # For now, keep this individual holding unchanged
            # The adjustment logic needs to be reworked to preserve individual transactions
            new_symbols.append(symbol)
            new_shares.append(int(holdings_shares[i]))
            new_buyprice.append(holdings_buyprice[i])
        
        elif symbol not in last_symbols_text and symbol != "CASH":
            # This holding should be sold
            trade_symbols.append(symbol)
            trade_shares.append(-int(holdings_shares[i]))
            buySellCost += BuySellFee

    ####################################################################
    ### check for sells -- stocks that were in last period and out now
    ####################################################################

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

    # Always make changes to preserve individual holdings
    makeChanges = len(trade_symbols) > 0 or len(new_symbols) > 0
    
    if makeChanges:
        new_symbols.append( "CASH" )
        new_shares.append( round( cash_bal,2 ) )
        new_buyprice.append( 1.0 )

    holdings_cash_bal = holdings_shares[cashindex]
    if makeChanges:
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
    
    # Do NOT aggregate holdings with same symbol but different buy prices
    # Each holding transaction must be tracked separately
    
    # For target holdings (df2), we need to calculate how to distribute
    # the target shares among existing holdings and new purchases
    
    # First, calculate total target shares per symbol from df2
    target_shares_by_symbol = df2.groupby('symbols')['shares'].sum().to_dict()
    
    # Now process each held position individually without aggregation
    merge_results = []
    
    # Track which symbols we've processed for target allocation
    processed_symbols = set()
    
    # Process each individual held position
    for idx, held_row in df1.iterrows():
        symbol = held_row['symbols']
        
        if symbol in target_shares_by_symbol:
            target_total = target_shares_by_symbol[symbol]
            
            # For now, keep existing positions as-is and handle changes separately
            # This preserves individual buy prices and dates
            merge_results.append({
                'symbols': symbol,
                'shares_held': held_row['shares'],
                'buyprice_held': held_row['buyprice'],
                'price_held': held_row['price'],
                'shares_target_total': target_total,
                'individual_holding': True
            })
            
        else:
            # This holding should be sold (symbol not in target)
            merge_results.append({
                'symbols': symbol,
                'shares_held': held_row['shares'],
                'buyprice_held': held_row['buyprice'],
                'price_held': held_row['price'],
                'shares_target_total': 0,
                'individual_holding': True
            })
    
    # Add new symbols that aren't currently held
    held_symbols = set(df1['symbols'])
    for idx, target_row in df2.iterrows():
        symbol = target_row['symbols']
        if symbol not in held_symbols:
            merge_results.append({
                'symbols': symbol,
                'shares_held': 0,
                'buyprice_held': target_row['price'],
                'price_held': target_row['price'],
                'shares_target_total': target_row['shares'],
                'individual_holding': False
            })
    
    merged_df = pd.DataFrame(merge_results)

    ####################################################################
    ### Generate trades based on merged results - ONLY NECESSARY TRADES
    ####################################################################
    buySellCost = 0.
    BuySellFee = 4.95
    trade_symbols = []
    trade_shares = []
    trade_price = []
    trade_activity = []
    trade_buyprice = []

    # For symbols NOT in target portfolio - sell ALL holdings
    for idx, held_row in held_df.iterrows():
        symbol = held_row['symbols']
        if symbol == "CASH":
            continue
            
        held_shares = held_row['shares']
        held_buyprice = held_row['buyprice']
        held_price = held_row['price']
        
        # If symbol not in target portfolio, sell this entire holding
        if symbol not in target_shares_by_symbol:
            trade_symbols.append(symbol)
            trade_shares.append(-held_shares)  # Negative for sell
            trade_price.append(float(np.round(held_price, 2)))
            trade_buyprice.append(np.round(held_buyprice, 2))
            trade_activity.append("sell")
            buySellCost += BuySellFee

    # For symbols IN target portfolio - only trade if needed
    for symbol, target_shares in target_shares_by_symbol.items():
        if symbol == "CASH":
            continue
            
        if target_shares > 0:
            current_total_shares = held_df[held_df['symbols'] == symbol]['shares'].sum()
            
            if current_total_shares == 0:
                # Symbol not currently held - buy the full amount
                target_price = new_df[new_df['symbols'] == symbol]['price'].iloc[0]
                trade_symbols.append(symbol)
                trade_shares.append(target_shares)  # Positive for buy
                trade_price.append(float(np.round(target_price, 2)))
                trade_buyprice.append(float(np.round(target_price, 2)))
                trade_activity.append("buy")
                buySellCost += BuySellFee
                
            elif target_shares > current_total_shares:
                # Need to buy more shares
                shares_to_buy = target_shares - current_total_shares
                target_price = new_df[new_df['symbols'] == symbol]['price'].iloc[0]
                trade_symbols.append(symbol)
                trade_shares.append(shares_to_buy)  # Positive for buy
                trade_price.append(float(np.round(target_price, 2)))
                trade_buyprice.append(float(np.round(target_price, 2)))
                trade_activity.append("buy")
                buySellCost += BuySellFee
                
            elif target_shares < current_total_shares:
                # Need to sell some shares - sell from holdings with highest buy price first (LIFO)
                shares_to_sell = current_total_shares - target_shares
                symbol_holdings = held_df[held_df['symbols'] == symbol].copy()
                symbol_holdings = symbol_holdings.sort_values('buyprice', ascending=False)  # Highest price first
                
                remaining_to_sell = shares_to_sell
                for idx, holding in symbol_holdings.iterrows():
                    if remaining_to_sell <= 0:
                        break
                        
                    shares_in_holding = holding['shares']
                    sell_from_this_holding = min(shares_in_holding, remaining_to_sell)
                    
                    if sell_from_this_holding > 0:
                        trade_symbols.append(symbol)
                        trade_shares.append(-sell_from_this_holding)  # Negative for sell
                        trade_price.append(float(np.round(holding['price'], 2)))
                        trade_buyprice.append(np.round(holding['buyprice'], 2))
                        trade_activity.append("reduce")
                        buySellCost += BuySellFee
                        
                        remaining_to_sell -= sell_from_this_holding
            # If target_shares == current_total_shares, no trades needed for this symbol

    # Handle cash separately
    cash_held = held_df[held_df['symbols'] == 'CASH']['shares'].sum() if 'CASH' in held_df['symbols'].values else 0
    cash_target = target_shares_by_symbol.get('CASH', 0)
    cash_diff = cash_target - cash_held
    
    if cash_diff != 0:
        trade_symbols.append("CASH")
        trade_shares.append(int(cash_diff))
        trade_price.append(1.0)
        trade_buyprice.append(1.0)
        if cash_diff > 0:
            trade_activity.append("increase")
        else:
            trade_activity.append("reduce")

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

    # Generate final holdings output text
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
        _buy_price = trades_df.loc[i, "price"]
        if verbose:
            print(" ... buy holdings symbol " + _symbol)
            print(" ... buy holdings shares " + str(_buy_shares))
        buy_text = buy_text + "info:  Buy  " + trade_date + " " + _symbol.ljust(6)
        buy_text = buy_text + format(_buy_shares, '7d')
        buy_text = buy_text + format(_buy_price, '8.2f')
        buy_text = buy_text + "\n"

    # Generate final holdings text showing resulting portfolio
    holdings_text = "stocks:      "
    for symbol in target_shares_by_symbol.keys():
        if target_shares_by_symbol[symbol] > 0:
            holdings_text = holdings_text + symbol.ljust(8)
    
    holdings_text = holdings_text + "\nshares:      "
    for symbol in target_shares_by_symbol.keys():
        if target_shares_by_symbol[symbol] > 0:
            holdings_text = holdings_text + str(target_shares_by_symbol[symbol]).ljust(8)
    
    holdings_text = holdings_text + "\nbuyprice:    "
    for symbol in target_shares_by_symbol.keys():
        if target_shares_by_symbol[symbol] > 0:
            if symbol == "CASH":
                buyprice_value = 1.0
            else:
                buyprice_value = new_df[new_df['symbols'] == symbol]['price'].iloc[0]
            holdings_text = holdings_text + str(np.round(buyprice_value, 2)).ljust(8)

    # Generate final holdings - preserve individual transactions, handle partial sells
    final_holdings = []
    
    # Create a copy of original holdings to track remaining shares after sells
    remaining_holdings = []
    for idx, held_row in held_df.iterrows():
        remaining_holdings.append({
            'symbol': held_row['symbols'],
            'shares': held_row['shares'],
            'buyprice': held_row['buyprice'],
            'original_index': idx  # Track original position for debugging
        })
    
    # Process all sell transactions to reduce holdings
    # Use LIFO (Last In, First Out) approach - sell from highest buy price first
    for i, trade_symbol in enumerate(trade_symbols):
        if trade_shares[i] < 0:  # This is a sell transaction
            shares_to_sell = abs(trade_shares[i])
            
            # Find all holdings of this symbol, sort by buy price (highest first for LIFO)
            symbol_holdings = [h for h in remaining_holdings 
                             if h['symbol'] == trade_symbol and h['shares'] > 0]
            symbol_holdings.sort(key=lambda x: x['buyprice'], reverse=True)
            
            # Sell from holdings until we've sold the required amount
            for holding in symbol_holdings:
                if shares_to_sell <= 0:
                    break
                
                # Sell from this specific holding
                shares_sold_from_holding = min(holding['shares'], shares_to_sell)
                holding['shares'] -= shares_sold_from_holding
                shares_to_sell -= shares_sold_from_holding
                
                if verbose:
                    print(f"  Reducing {holding['symbol']} holding: "
                          f"{shares_sold_from_holding} shares @ ${holding['buyprice']:.2f}")
    
    # Add remaining holdings (shares > 0) to final holdings
    for holding in remaining_holdings:
        if holding['shares'] > 0:
            final_holdings.append({
                'symbol': holding['symbol'],
                'shares': holding['shares'],
                'buyprice': holding['buyprice']
            })
    
    # Add new purchases from buy transactions
    for i, trade_symbol in enumerate(trade_symbols):
        if trade_shares[i] > 0 and trade_symbol != "CASH":
            final_holdings.append({
                'symbol': trade_symbol,
                'shares': trade_shares[i],
                'buyprice': trade_buyprice[i]
            })
    
    # Calculate final cash position - exclude transaction fees from cash calculation
    original_cash = held_df[held_df['symbols'] == 'CASH']['shares'].sum() if 'CASH' in held_df['symbols'].values else 0
    
    # Calculate net cash flow from all trades (excluding fees)
    total_sell_proceeds = 0.0
    total_buy_cost = 0.0
    
    for i in range(len(trade_symbols)):
        if trade_symbols[i] != "CASH":
            trade_value = abs(trade_shares[i]) * trade_price[i]
            if trade_shares[i] < 0:  # Sell transaction
                total_sell_proceeds += trade_value
            elif trade_shares[i] > 0:  # Buy transaction  
                total_buy_cost += trade_value
    
    # Final cash = original cash + sell proceeds - buy costs (NO transaction fees)
    final_cash = original_cash + total_sell_proceeds - total_buy_cost
    
    # Ensure cash is non-negative
    if final_cash < 0:
        print(f"Warning: Negative cash position {final_cash:.2f}, adjusting to 0")
        final_cash = 0
    
    # Update the final holdings cash entry
    for holding in final_holdings:
        if holding['symbol'] == "CASH":
            holding['shares'] = int(round(final_cash))
            break
    else:
        # If no cash holding exists, add one
        final_holdings.append({
            'symbol': "CASH",
            'shares': int(round(final_cash)),
            'buyprice': 1.0
        })
    
    if verbose:
        print(f"Cash calculation (fees excluded from holdings):")
        print(f"  Original cash: ${original_cash:,.2f}")
        print(f"  Sell proceeds: ${total_sell_proceeds:,.2f}")
        print(f"  Buy costs: ${total_buy_cost:,.2f}")
        print(f"  Transaction fees (info only): ${buySellCost:,.2f}")
        print(f"  Final cash: ${final_cash:,.2f}")
        
        # Verify portfolio value conservation (excluding fees)
        original_portfolio_value = currentHoldingsValue
        final_portfolio_value = 0.0
        for holding in final_holdings:
            if holding['symbol'] == "CASH":
                final_portfolio_value += holding['shares']
            else:
                # Use current market price for final valuation
                current_price = held_df[held_df['symbols'] == holding['symbol']]['price'].iloc[0] if holding['symbol'] in held_df['symbols'].values else holding['buyprice']
                final_portfolio_value += holding['shares'] * current_price
        
        print(f"Portfolio value check (excluding fees):")
        print(f"  Original: ${original_portfolio_value:,.2f}")
        print(f"  Final: ${final_portfolio_value:,.2f}")
        print(f"  Difference: ${abs(final_portfolio_value - original_portfolio_value):,.2f}")

    # Generate holdings text preserving individual transactions
    holdings_text = "stocks:      "
    for holding in final_holdings:
        if holding['shares'] > 0:
            holdings_text = holdings_text + holding['symbol'].ljust(8)
    
    holdings_text = holdings_text + "\nshares:      "
    for holding in final_holdings:
        if holding['shares'] > 0:
            holdings_text = holdings_text + str(int(holding['shares'])).ljust(8)
    
    holdings_text = holdings_text + "\nbuyprice:    "
    for holding in final_holdings:
        if holding['shares'] > 0:
            holdings_text = holdings_text + str(np.round(holding['buyprice'], 2)).ljust(8)

    print(holdings_text)

    # Print and write to file
    hypothetical_trade_info = "\n\nTradeDate: " + trade_date + "\n"
    hypothetical_trade_info = hypothetical_trade_info + sell_text
    hypothetical_trade_info = hypothetical_trade_info + buy_text
    hypothetical_trade_info = hypothetical_trade_info + holdings_text

    print(hypothetical_trade_info)

    filepath = os.path.join(p_store, "PyTAAA_hypothetical_trades.txt")
    with open(filepath, "w") as f:
        f.write(hypothetical_trade_info)
        f.write("\n")


