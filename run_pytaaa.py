import datetime
import logging
import numpy as np
import os
import time
import urllib.error
import platform
from functions.SendEmail import SendEmail
from functions.WriteWebPage_pi import writeWebPage
from functions.GetParams import (
    get_json_params, get_symbols_file,
    get_holdings, get_status, GetIP, GetEdition,
    put_status
)
from functions.UpdateSymbols_inHDF5 import UpdateHDF_yf
from functions.CheckMarketOpen import (get_MarketOpenOrClosed,
                                       CheckMarketOpen)
from functions.PortfolioPerformanceCalcs import run_portfolio_analysis
from functions.output_generators import write_rank_list_html
from functions.quotes_for_list_adjClose import LastQuotesForSymbolList_hdf
from functions.calculateTrades import calculateTrades, trade_today
# from functions.quotes_for_list_adjClose import get_Naz100List, get_SP500List
from functions.readSymbols import get_symbols_changes
from functions.stock_cluster import getClusterForSymbolsList
from functions.ftp_quotes import copy_updated_quotes

# Module-level sentinels that persist across repeated scheduler calls.
# These replace the broken `x in locals()` patterns that reset on every
# function invocation, making the hour-of-day guard non-functional.
_daily_update_done: bool = False
_calcs_update_count: int = 0
_cached_lastdate = None
_cached_last_symbols_text = None
_cached_last_symbols_weight = None
_cached_last_symbols_price = None
_cached_symbols = None
_cached_adjClose = None
_cached_signal2D_daily = None
_cached_monthgainlossweight = None
_cached_datearray = None


def run_pytaaa(json_fn):
    global _daily_update_done, _calcs_update_count
    global _cached_lastdate, _cached_last_symbols_text
    global _cached_last_symbols_weight, _cached_last_symbols_price
    global _cached_symbols, _cached_adjClose
    global _cached_signal2D_daily, _cached_monthgainlossweight, _cached_datearray

    # Resolve all paths relative to this file's directory so callers do
    # not need to set CWD before invoking run_pytaaa().
    os.chdir(os.path.abspath(os.path.dirname(__file__)))

    computerName = platform.uname()[1]

    # Get Credentials for sending email
    params = get_json_params(json_fn, verbose=True)
    symbols_file = get_symbols_file(json_fn)

    username = str(params['fromaddr']).split("@")[0]
    emailpassword = str(params['PW'])
    stockList = params['stockList']
    # get name of server used to download and serve quotes
    quote_server = params['quote_server']
    try:
        ip = GetIP()
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        # Expected: network unavailable, timeout, etc.
        logging.getLogger(__name__).debug(f"Could not get external IP: {e}")
        ip = '0.0.0.0'
    except Exception as e:
        # Safety fallback for unexpected exceptions
        logging.getLogger(__name__).warning(f"Unexpected exception in GetIP: {type(e).__name__}: {e}")
        ip = '0.0.0.0'
    print("Current ip address is ", ip)
    print(
        "An email with updated analysis will be sent to ",
        params['toaddrs'], " every ", params['pausetime'], " seconds"
    )

    # keep track of total time to update everything
    start_time_total = time.time()

    # set value to compare with cumu_value as test of info content
    cumu_value_prior = get_status(json_fn)

    # Get Holdings from file
    holdings = get_holdings(json_fn)
    print("")
    print("current Holdings :")
    print("stocks: ", holdings['stocks'])
    print("shares: ", holdings['shares'])
    print("buyprice: ", holdings['buyprice'])
    print("current ranks: ", holdings['ranks'])
    print("cumulativecashin: ", holdings['cumulativecashin'][0])
    print("")
    
    # Proactively update fundamentals cache for current holdings
    # This avoids rate limiting by batching updates and using cached data
    # Only update symbols that are currently in the active universe
    try:
        from functions.stock_fundamentals_cache import get_cache
        from functions.data_loaders import load_quotes_for_analysis
        
        # Load current universe symbols to validate holdings
        try:
            _, universe_symbols, _ = load_quotes_for_analysis(symbols_file, json_fn, verbose=False)
            valid_symbols = set(universe_symbols)
        except Exception as e:
            print(f"Warning: Could not load universe symbols, skipping validation: {e}")
            valid_symbols = None
        
        cache = get_cache()
        active_symbols = [s for s in holdings['stocks'] if s != 'CASH']
        
        # Only update symbols that are both in holdings AND in current universe
        if valid_symbols is not None:
            cache.update_for_current_symbols(active_symbols, force_refresh=False, valid_symbols=universe_symbols)
            print(f"Updated fundamentals cache for holdings validated against universe")
        else:
            cache.update_for_current_symbols(active_symbols, force_refresh=False)
            print(f"Updated fundamentals cache for {len(active_symbols)} holdings (no universe validation)")
    except Exception as e:
        print(f"Warning: Failed to update fundamentals cache: {e}")

    # Update prices in HDF5 file for symbols in list
    # - check web for current stocks in Naz100, update files if changes needed
    today = datetime.datetime.now()
    hourOfDay = today.hour
    start_time_updateStockList = time.time()
    if hourOfDay <= 17:
        if stockList == 'Naz100' or stockList == 'SP500':
            # _, removedTickers, addedTickers = get_SP500List(symbols_file, verbose=True)
            _, removedTickers, addedTickers = get_symbols_changes(json_fn)
    else:
        removedTickers, addedTickers = [], []
    elapsed_time_updateStockList = time.time() - start_time_updateStockList

    # symbol_directory = os.path.join(os.getcwd(), "symbols")

    # if stockList == 'Naz100':
    #     symbol_file = "Naz100_Symbols.txt"
    # elif stockList == 'SP500':
    #     symbol_file = "SP500_Symbols.txt"
    # symbols_file = os.path.join( symbol_directory, symbol_file )
    symbols_file = get_symbols_file(json_fn)
    symbol_directory, symbol_file = os.path.split(symbols_file)

    start_time = time.time()
    # Reset daily HDF5 update flag and calcs counter before the cutoff
    # hour so each new business day triggers a fresh update. Module-level
    # sentinels persist across repeated scheduler calls; local variables
    # cannot (new frame on every call).
    if hourOfDay <= 15:
        _daily_update_done = False
        _calcs_update_count = 0
    daily_update_done = _daily_update_done
    print("hourOfDay, daily_update_done =", hourOfDay, daily_update_done)
    if quote_server != computerName:
        copy_updated_quotes(json_fn)
    if not daily_update_done:
        # Check if quote server has _NO suffix to disable updates
        if quote_server == computerName and not quote_server.endswith('_NO'):
            UpdateHDF_yf(symbol_directory, symbols_file, json_fn)
        elif quote_server.endswith('_NO'):
            print("Stock quote updating disabled due to _NO suffix in quote_server setting")
        if hourOfDay > 15:
            _daily_update_done = True
            daily_update_done = True
    marketOpen, lastDayOfMonth = CheckMarketOpen()
    elapsed_time = time.time() - start_time

    print("\n   . inside run_taaa: UpdateHDF_yf completed...")
    print("  . inside run_taaa: symbol_dirctory" + symbol_directory)
    print("  . inside run_taaa: symbol_file" + symbol_file)

    # Re-compute stock ranks and weightings when HDF5 data is fresh or
    # results have never been computed. Module-level cached results avoid
    # redundant recomputation across repeated scheduler calls.
    not_Calculated = (_calcs_update_count == 0)
    if (daily_update_done and _calcs_update_count == 0) or not_Calculated:
        (
            _cached_lastdate,
            _cached_last_symbols_text,
            _cached_last_symbols_weight,
            _cached_last_symbols_price,
            _cached_symbols,
            _cached_adjClose,
            _cached_signal2D_daily,
            _cached_monthgainlossweight,
            _cached_datearray,
        ) = run_portfolio_analysis(
            symbol_directory, symbol_file, params, json_fn,
        )
        _calcs_update_count += 1
    lastdate = _cached_lastdate
    last_symbols_text = _cached_last_symbols_text
    last_symbols_weight = _cached_last_symbols_weight
    last_symbols_price = _cached_last_symbols_price

    # Get updated Holdings from file (ranks are updated)
    holdings = get_holdings(json_fn)
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
    holdings_currentPrice = LastQuotesForSymbolList_hdf(
        holdings_symbols, symbol_file, json_fn
    )
    print("holdings_symbols = ", holdings_symbols)
    print("holdings_shares = ", holdings_shares)
    print("holdings_currentPrice = ", holdings_currentPrice)

    # retrieve cluster labels for holdings
    try:
        holdings_cluster_labels = getClusterForSymbolsList(
            holdings_symbols, json_fn
        )
    except (FileNotFoundError, KeyError, OSError) as e:
        # Expected: cluster data missing or symbol not found
        logging.getLogger(__name__).debug(f"Cluster data unavailable: {e}")
        holdings_cluster_labels = np.zeros((len(holdings_symbols)), 'int')
    except Exception as e:
        # Safety fallback for unexpected exceptions
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

    # holding_rows is built inside the loop below; message_text is rendered
    # from the Jinja2 template after the loop (Item 9).
    holding_rows = []
    cumu_purchase_value = 0.
    cumu_value = 0.
    print("holdings_shares = ", holdings_shares)
    print("holdings_buyprice = ", holdings_buyprice)
    print("last_symbols_text = ", last_symbols_text)
    print("last_symbols_weight = ", last_symbols_weight)
    print("last_symbols_price = ", last_symbols_price)
    print("holdings_ranks = ", holdings_ranks)
    print("\n\n")

    # Pre-fetch sector/industry data for all holding symbols before the
    # reporting loop.  This eliminates N synchronous web requests from
    # the hot path; the fundamentals cache handles freshness and
    # network fallback transparently.
    from functions.stock_fundamentals_cache import prefetch_sector_industry
    sector_industry_map = prefetch_sector_industry(list(holdings_symbols))

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
        # Look up sector/industry from the pre-fetched map instead of
        # making a live web request for each symbol.
        sector, industry = sector_industry_map.get(
            holdings_symbols[i], ("", "")
        )
        # Build the row dict with pre-formatted values for the template.
        holding_rows.append({
            "symbol": format(holdings_symbols[i], "5s"),
            "shares": format(holdings_shares[i], "6.0f"),
            "buy_price": format(holdings_buyprice[i], "6.2f"),
            "purchase_cost": format(purchase_value, "6.2f"),
            "cumu_purchase": format(cumu_purchase_value, "6.2f"),
            "current_price": format(float(holdings_currentPrice[i]), "6.2f"),
            "profit_pct": format(float(profitPct), "6.2%"),
            "value": format(value, "6.2f"),
            "cumu_value": format(cumu_value, "6.2f"),
            "rank": format(holdings_ranks[i], "3d"),
            "cluster": str(holdings_cluster_labels[i]),
            "sector": str(sector),
            "industry": str(industry),
        })
    print("")

    # Notify with buys/sells on trade dates.
    trade_message = "<br>"
    trade_message = calculateTrades(
        holdings, last_symbols_text,
        last_symbols_weight, last_symbols_price, json_fn
    )
    print("   . returned from calculateTrades ...\n\n")

    # Generate hypothetical trade recommendations for stdout and write
    # PyTAAA_hypothetical_trades.txt in the performance store.  Then
    # immediately write pyTAAAweb_RankList.txt so the webpage shows
    # THIS run's trades (not the previous run's).
    try:
        trade_today(
            json_fn,
            list(last_symbols_text),
            list(last_symbols_weight),
            list(last_symbols_price),
        )
    except Exception as trade_today_exc:
        print(
            f" Warning: trade_today() failed: {trade_today_exc}"
        )

    # Write rank-list HTML now that the fresh trades file is on disk.
    if _cached_symbols is not None:
        try:
            write_rank_list_html(
                json_fn, _cached_symbols, _cached_adjClose,
                _cached_signal2D_daily, _cached_monthgainlossweight,
                _cached_datearray,
            )
        except Exception as rank_html_exc:
            print(
                f" Warning: write_rank_list_html() failed: {rank_html_exc}"
            )

    edition = GetEdition()

    # Render the holdings HTML report via Jinja2 replacing the previous
    # manual string concatenation (Item 9).
    from functions.report_builders import build_holdings_html_report
    message_text = build_holdings_html_report(
        holdings_rows=holding_rows,
        trade_message=trade_message,
        lifetime_profit=lifetimeProfit,
        cumulative_cash_in=float(holdings['cumulativecashin'][0]),
        lifetime_profit_annualized=lifetimeProfitAnnualized,
        removed_tickers=removedTickers,
        added_tickers=addedTickers,
        edition=edition,
        ip=ip,
        params=params,
    )

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

    # cumu_value_prior: portfolio value stored at end of previous run.
    # cumu_value: current holdings * today's HDF5 prices (pre-trade).
    # Values are identical when run twice in the same day with the same
    # HDF5 data — not a bug; email gate uses this to suppress duplicates.
    value_changed = np.round(float(cumu_value_prior), 2) != np.round(cumu_value, 2)
    message_changed = trade_message != "<br>"
    print(
        f"  Portfolio value (prior run)  : {float(cumu_value_prior):>14,.2f}"
    )
    print(
        f"  Portfolio value (this run)   : {cumu_value:>14,.2f}"
        f"  (changed={value_changed})"
    )
    print(f"  trade_message changed        : {message_changed}")
    if value_changed or message_changed:
        headlinetext = "Regularly scheduled update. Market status: " + get_MarketOpenOrClosed()
        SendEmail(username,emailpassword,params['toaddrs'],params['fromaddr'],subjecttext,regulartext,boldtext,headlinetext)
    else:
        headlinetext = "Regularly scheduled update. Market status: " + get_MarketOpenOrClosed()
        print(" No email required or sent -- no new information since last email...")
        cumu_value_prior = cumu_value


    # If there are changes to Nasdaq100 stock list, add message

    # build the updated web page
    writeWebPage(
        regulartext, boldtext, headlinetext, lastdate,
        last_symbols_text, last_symbols_weight, last_symbols_price,
        json_fn
    )
    # set value to compare with cumu_value as test of info content
    put_status(cumu_value, json_fn)

    # print market status to terminal window
    get_MarketOpenOrClosed()
    print(" . finished at ", str(datetime.datetime.now()))



'''
Main program
'''
if __name__ == '__main__':

    json_folder = os.getcwd()
    json_fn = os.path.join(json_folder, "pytaaa.json")

    run_pytaaa(json_fn)
