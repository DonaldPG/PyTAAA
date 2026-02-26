"""Portfolio Performance Calculations Module.

This module provides the main orchestration function for computing portfolio
performance metrics, generating plots, and writing output files.

Phase 4b refactoring: Separated concerns into load → compute → output pattern
- Phase 4b1: Extracted plot generation to output_generators.py
- Phase 4b2: Extracted file writing to output_generators.py
- Phase 4b3: Extracted pure computation to compute_portfolio_metrics()
- Phase 4b4: Clean orchestration (this module)
"""

import logging
import numpy as np
import os
import datetime
from numpy import isnan

from functions.data_loaders import load_quotes_for_analysis
from functions.dailyBacktest import computeDailyBacktest
from functions.CheckMarketOpen import get_MarketOpenOrClosed
from functions.TAfunctions import textmessageOutsideTrendChannel
from functions.GetParams import get_webpage_store
from functions.output_generators import (
    compute_portfolio_metrics,
    generate_portfolio_plots,
    write_portfolio_status_files
)

logger = logging.getLogger(__name__)


def PortfolioPerformanceCalcs(symbol_directory, symbol_file, params, json_fn):
    """Main orchestrator for portfolio performance analysis.
    
    Follows the pattern: Load Data → Compute Metrics → Generate Outputs
    
    Args:
        symbol_directory: Directory containing symbol file
        symbol_file: Name of file with stock symbols
        params: Dictionary of parameters from JSON config
        json_fn: Path to JSON configuration file
    
    Returns:
        Tuple of (date, symbols_list, weights_list, prices_list) for current holdings
    """
    
    #############################################################################
    # PHASE 1: LOAD DATA
    #############################################################################
    
    logger.info("Starting PortfolioPerformanceCalcs")
    logger.debug("symbol_directory = %s", symbol_directory)
    logger.debug("symbol_file = %s", symbol_file)

    json_dir = os.path.split(json_fn)[0]
    logger.debug("json_dir = %s", json_dir)

    filename = os.path.join(symbol_directory, symbol_file)
    logger.debug("filename for load_quotes_for_analysis = %s", filename)
    
    adjClose, symbols, datearray = load_quotes_for_analysis(
        filename, json_fn, verbose=True
    )

    #############################################################################
    # PHASE 2: COMPUTE PORTFOLIO METRICS (Pure Computation)
    #############################################################################
    
    metrics = compute_portfolio_metrics(adjClose, symbols, datearray, params, json_fn)
    
    # Extract computed values for output phase
    signal2D = metrics['signal2D']
    signal2D_daily = metrics['signal2D_daily']
    numberStocks = metrics['numberStocks']
    dailyNumberUptrendingStocks = metrics['dailyNumberUptrendingStocks']
    activeCount = metrics['activeCount']
    monthgainlossweight = metrics['monthgainlossweight']
    monthvalue = metrics['monthvalue']
    BuyHoldFinalValue = metrics['BuyHoldFinalValue']
    last_symbols_text = metrics['last_symbols_text']
    last_symbols_weight = metrics['last_symbols_weight']
    last_symbols_price = metrics['last_symbols_price']
    lowChannel = metrics.get('lowChannel', None)
    hiChannel = metrics.get('hiChannel', None)
    
    # Log diagnostic info
    logger.debug("signal2D check: %s", signal2D[isnan(signal2D)].shape)
    logger.debug("signal2D min=%f, mean=%f, max=%f", signal2D.min(), signal2D.mean(), signal2D.max())
    logger.debug("numberStocks (uptrending) min=%f, mean=%f, max=%f", 
          numberStocks.min(), numberStocks.mean(), numberStocks.max())

    #############################################################################
    # PHASE 3: GENERATE OUTPUTS (Files, Plots, Reports)
    #############################################################################
    
    web_dir = get_webpage_store(json_fn)
    
    # 3.1: Write daily backtest results
    _write_daily_backtest(json_fn, datearray, symbols, adjClose, params)
    
    # 3.2: Write portfolio status files
    write_portfolio_status_files(
        dailyNumberUptrendingStocks, activeCount, datearray, web_dir
    )
    
    # 3.3: Generate portfolio plots
    async_mode = bool(params.get('async_plot_generation', False))
    max_workers = int(params.get('plot_generation_workers', 2))
    if params['uptrendSignalMethod'] == 'percentileChannels':
        generate_portfolio_plots(
            adjClose, symbols, datearray, signal2D, signal2D_daily,
            params, web_dir, lowChannel=lowChannel, hiChannel=hiChannel,
            async_mode=async_mode, max_workers=max_workers,
        )
    else:
        generate_portfolio_plots(
            adjClose, symbols, datearray, signal2D, signal2D_daily,
            params, web_dir,
            async_mode=async_mode, max_workers=max_workers,
        )
    
    # 3.4: Print summary reports
    _print_portfolio_summary(
        datearray, symbols, adjClose, signal2D_daily, monthgainlossweight,
        monthvalue, BuyHoldFinalValue, last_symbols_text, last_symbols_weight,
        last_symbols_price, params, json_fn, lowChannel, hiChannel
    )
    
    # 3.5: Send text alerts if market is open
    marketStatus = get_MarketOpenOrClosed()
    if 'Market Open' in marketStatus:
        textmessageOutsideTrendChannel(symbols, adjClose, json_fn)

    return datearray[-1], last_symbols_text, last_symbols_weight, last_symbols_price


def _write_daily_backtest(json_fn, datearray, symbols, adjClose, params):
    """Write daily backtest portfolio and B&H values to file.
    
    Helper function to keep orchestrator clean.
    """
    computeDailyBacktest(
        json_fn, datearray, symbols, adjClose,
        numberStocksTraded=params['numberStocksTraded'],
        trade_cost=params['trade_cost'],
        monthsToHold=params['monthsToHold'],
        LongPeriod=params['LongPeriod'],
        MA1=int(params['MA1']),
        MA2=int(params['MA2']),
        MA2offset=int(params['MA3']) - int(params['MA2']),
        sma2factor=params['MA2factor'],
        rankThresholdPct=float(params['rankThresholdPct']),
        riskDownside_min=float(params['riskDownside_min']),
        riskDownside_max=float(params['riskDownside_max']),
        narrowDays=params['narrowDays'],
        mediumDays=params['mediumDays'],
        wideDays=params['wideDays'],
        stddevThreshold=float(params['stddevThreshold']),
        lowPct=float(params['lowPct']),
        hiPct=float(params['hiPct']),
        uptrendSignalMethod=params['uptrendSignalMethod']
    )
    
    print("\n\n Successfully updated daily backtest at in 'pyTAAAweb_backtestPortfolioValue.params'.")
    print(f" Completed on {datetime.datetime.now().strftime('%A, %d. %B %Y %I:%M%p')}")
    print("")


def _print_portfolio_summary(
    datearray, symbols, adjClose, signal2D_daily, monthgainlossweight,
    monthvalue, BuyHoldFinalValue, last_symbols_text, last_symbols_weight,
    last_symbols_price, params, json_fn, lowChannel=None, hiChannel=None
):
    """Print portfolio summary reports.
    
    Helper function to keep orchestrator clean.
    """
    # Print uptrending stocks list (if using percentileChannels)
    if params['uptrendSignalMethod'] == 'percentileChannels' and lowChannel is not None:
        print(f"\n\n\nCurrently up-trending symbols ({datearray[-1]}):")
        uptrendCount = 0
        for i in range(len(symbols)):
            if signal2D_daily[i, -1] > 0:
                uptrendCount += 1
                print(f"{uptrendCount:4d} {symbols[i]:5s} {adjClose[i, -1]:8,.2f} uptrend "
                      f"{lowChannel[i, -1]:8,.2f} {hiChannel[i, -1]:8,.2f}")
            else:
                print(f"{uptrendCount:4d} {symbols[i]:5s} {adjClose[i, -1]:8,.2f}         "
                      f"{lowChannel[i, -1]:8,.2f} {hiChannel[i, -1]:8,.2f}")
    print("\n\n\n")
    
    # Print portfolio performance summary
    print(" ")
    print("The B&H portfolio final value is: ", "{:,}".format(int(BuyHoldFinalValue)))
    print(" ")
    print("Monthly re-balance based on ", params['LongPeriod'], 
          "days of recent performance.")
    print("The portfolio final value is: ", 
          "{:,}".format(int(np.average(monthvalue, axis=0)[-1])))
    print(" ")
    
    # Print current top holdings with company names
    print("Today's top ranking choices are: ")
    # Fetch company names
    try:
        from functions.readSymbols import read_company_names_local
        companySymbolList, companyNameList = read_company_names_local(json_fn, verbose=False)
    except Exception:
        companySymbolList, companyNameList = [], []
    
    for ii in range(len(symbols)):
        if monthgainlossweight[ii, -1] > 0:
            # Look up company name
            try:
                symbolIndex = companySymbolList.index(symbols[ii])
                companyName = companyNameList[symbolIndex]
            except (ValueError, IndexError):
                companyName = ""
            
            print(f"{datearray[-1]}  {symbols[ii]:<6s}  "
                  f"{monthgainlossweight[ii, -1]:6.4f}  ({monthgainlossweight[ii, -1]*100:5.2f}%)  "
                  f"{companyName:20s}")
    
    print("\n ... inside portfolioPerformanceCalcs")
    print("   . datearray[-1] = " + str(datearray[-1]))
    print("   . last_symbols_text = " + str(last_symbols_text))
    print("   . last_symbols_weight = " + str(last_symbols_weight))
    print("   . last_symbols_price = " + str(last_symbols_price))

