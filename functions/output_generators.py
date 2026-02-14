"""Output generation functions for PyTAAA.

This module contains functions for generating plots and other output artifacts
from portfolio computation results. These functions have side effects (file I/O)
and are separated from pure computation logic for testability.

Phase 4b1: Extracted from PortfolioPerformanceCalcs.py
Phase 4b3: Added compute_portfolio_metrics (pure computation)
"""

import numpy as np
from numpy import isnan
import datetime
import os
from typing import Dict, List, Any, Optional

import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt

# Set DPI for inline plots and saved figures
plt.rcParams['figure.figsize'] = (9, 7)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

from functions.TAfunctions import (
    despike_2D,
    recentTrendAndMidTrendChannelFitWithAndWithoutGap
)
import functions.ystockquote as ysq


def write_portfolio_status_files(
    dailyNumberUptrendingStocks: np.ndarray,
    activeCount: np.ndarray,
    datearray: np.ndarray,
    output_dir: str
) -> None:
    """Write portfolio status files for web display.
    
    Writes the number of uptrending stocks and active count for each date
    to a .params file that can be displayed on the web page.
    
    Args:
        dailyNumberUptrendingStocks: Count of uptrending stocks per day (n_days,)
        activeCount: Count of active (non-NaN) stocks per day (n_days,)
        datearray: Array of datetime objects for each day (n_days,)
        output_dir: Directory to write status file
    
    Returns:
        None (writes file to output_dir)
    
    Side Effects:
        - Writes pyTAAAweb_numberUptrendingStocks_status.params to output_dir
        - Prints success/error messages to stdout
    """
    try:
        filepath = os.path.join(output_dir, "pyTAAAweb_numberUptrendingStocks_status.params")
        textmessage = ""
        for jj in range(dailyNumberUptrendingStocks.shape[0]):
            textmessage = textmessage + str(datearray[jj]) + "  " + \
                         str(dailyNumberUptrendingStocks[jj]) + "  " + \
                         str(activeCount[jj]) + "\n"
        with open(filepath, "w") as f:
            f.write(textmessage)
        print(
            f" Successfully updated to pyTAAAweb_numberUptrendingStocks_status.params at "
            f"{datetime.datetime.now().strftime('%A, %d. %B %Y %I:%M%p')}"
        )
        print("")
    except Exception as e:
        print(" Error: unable to update pyTAAAweb_numberUptrendingStocks_status.params")
        print(f" Exception: {e}")
        print("")


def generate_portfolio_plots(
    adjClose: np.ndarray,
    symbols: List[str],
    datearray: np.ndarray,
    signal2D: np.ndarray,
    signal2D_daily: np.ndarray,
    params: Dict[str, Any],
    output_dir: str,
    lowChannel: Optional[np.ndarray] = None,
    hiChannel: Optional[np.ndarray] = None
) -> None:
    """Generate all portfolio plots for web display.
    
    Creates two types of plots for each symbol:
    1. Full history plots (0_<symbol>.png)
    2. Recent 2-year plots with trend channels (0_recent_<symbol>.png)
    
    Plot generation is conditional on time of day to avoid excessive CPU usage
    during market hours. Plots are skipped if less than 20 hours old.
    
    Args:
        adjClose: Adjusted close prices (n_symbols x n_days)
        symbols: List of stock symbols
        datearray: Array of datetime objects for each day
        signal2D: Monthly uptrend signals (n_symbols x n_days)
        signal2D_daily: Daily uptrend signals (n_symbols x n_days)
        params: Dictionary of parameters including:
            - LongPeriod: Period for despiking
            - stddevThreshold: Threshold for despike detection
            - uptrendSignalMethod: 'percentileChannels' or other
            - minperiod, maxperiod, incperiod: Channel fit parameters
            - numdaysinfit, numdaysinfit2: Days in channel fit
            - offset: Offset for channel fit
        output_dir: Directory to write PNG files
        lowChannel: Optional low percentile channel (if percentileChannels method)
        hiChannel: Optional high percentile channel (if percentileChannels method)
    
    Returns:
        None (writes PNG files to output_dir)
    
    Side Effects:
        - Writes ~200 PNG files (2 per symbol) to output_dir
        - Prints progress messages to stdout
    """
    today = datetime.datetime.now()
    hourOfDay = today.hour
    
    # Only generate plots outside market hours to reduce load
    if not (hourOfDay >= 1 or 11 < hourOfDay < 13):
        return
    
    # Extract parameters
    LongPeriod = params['LongPeriod']
    stddevThreshold = float(params['stddevThreshold'])
    uptrendSignalMethod = params['uptrendSignalMethod']
    
    # Determine first date index for recent plots (2013+)
    firstdate_index = 0
    for ii in range(len(datearray)):
        if datearray[ii].year > datearray[ii-1].year and datearray[ii].year == 2013:
            firstdate_index = ii
            break
    
    ##########################################################################
    # 1. Generate full history plots for all symbols
    ##########################################################################
    for i in range(len(symbols)):
        plotfilepath = os.path.join(output_dir, f"0_{symbols[i]}.png")
        
        # Check recency of plot file and skip if less than 20 hours old
        if os.path.isfile(plotfilepath):
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(plotfilepath))
            modified_time = (datetime.datetime.now() - mtime)
            modified_hours = modified_time.days * 24 + modified_time.seconds / 3600
            if modified_hours < 20.0:
                continue
        
        # Get 'despiked' quotes for this symbol
        quotes = adjClose[i, :].copy()
        quotes = quotes.reshape(1, len(quotes))
        quotes_despike = despike_2D(quotes, LongPeriod, stddevThreshold=stddevThreshold)
        
        # Create plot
        plt.clf()
        plt.grid(True)
        plt.plot(datearray, adjClose[i, :])
        plt.plot(datearray, signal2D[i, :] * adjClose[i, -1], lw=.2)
        
        despiked_quotes = quotes_despike[0, :]
        number_nans = despiked_quotes[np.isnan(despiked_quotes)].shape[0]
        if number_nans == 0:
            plt.plot(datearray, quotes_despike[0, :])
        
        if uptrendSignalMethod == 'percentileChannels' and lowChannel is not None:
            plt.plot(datearray, lowChannel[i, :], 'm-')
            plt.plot(datearray, hiChannel[i, :], 'm-')
        
        plot_text = str(adjClose[i, -7:])
        plt.text(datearray[50], 0, plot_text)
        
        # Add text with most recent date at bottom of plot
        x_range = datearray[-1] - datearray[0]
        text_x = datearray[0] + datetime.timedelta(x_range.days / 20.)
        adjClose_noNaNs = adjClose[i, :].copy()
        adjClose_noNaNs = adjClose_noNaNs[~np.isnan(adjClose_noNaNs)]
        text_y = (np.max(adjClose_noNaNs) - np.min(adjClose_noNaNs)) * .085 + np.min(adjClose_noNaNs)
        
        plt.text(
            text_x, text_y,
            f"most recent value from {datearray[-1]}\n"
            f"plotted at {today.strftime('%A, %d. %B %Y %I:%M%p')}\n"
            f"value = {adjClose[i, -1]}",
            fontsize=8
        )
        plt.title(f"{symbols[i]}\n{ysq.get_company_name(symbols[i])}")
        plt.yscale('log')
        
        print(f" ...inside PortfolioPerformancealcs... plotfilepath = {plotfilepath}")
        plt.savefig(plotfilepath, format='png')
    
    ##########################################################################
    # 2. Generate recent (2-year) plots with trend channels
    ##########################################################################
    for i in range(len(symbols)):
        plotfilepath = os.path.join(output_dir, f"0_recent_{symbols[i]}.png")
        
        # Check recency of plot file and skip if less than 20 hours old
        if os.path.isfile(plotfilepath):
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(plotfilepath))
            modified_time = (datetime.datetime.now() - mtime)
            modified_hours = modified_time.days * 24 + modified_time.seconds / 3600
            if modified_hours < 20.0:
                continue
        
        # Fit short-term recent trend channel for plotting
        quotes = adjClose[i, :].copy()
        quotes = quotes.reshape(1, len(quotes))
        quotes_despike = despike_2D(quotes, LongPeriod, stddevThreshold=stddevThreshold)
        # Re-scale to have same value at beginning of plot
        quotes_despike *= quotes[0, firstdate_index] / quotes_despike[0, firstdate_index]
        
        lowerTrend, upperTrend, NoGapLowerTrend, NoGapUpperTrend = \
            recentTrendAndMidTrendChannelFitWithAndWithoutGap(
                adjClose[i, :],
                minperiod=params['minperiod'],
                maxperiod=params['maxperiod'],
                incperiod=params['incperiod'],
                numdaysinfit=params['numdaysinfit'],
                numdaysinfit2=params['numdaysinfit2'],
                offset=params['offset']
            )
        
        try:
            # Plot recent (about 2 years) performance for each symbol
            plt.figure(10)
            plt.clf()
            plt.grid(True)
            
            plt.plot(
                datearray[firstdate_index:],
                signal2D[i, firstdate_index:] * adjClose[i, -1],
                lw=.25, alpha=.6
            )
            plt.plot(
                datearray[firstdate_index:],
                signal2D_daily[i, firstdate_index:] * adjClose[i, -1],
                lw=.25, alpha=.6
            )
            plt.plot(
                datearray[firstdate_index:],
                quotes_despike[0, firstdate_index:],
                lw=.15
            )
            
            adjClose_noNaNs = adjClose[i, :].copy()
            adjClose_noNaNs = adjClose_noNaNs[~np.isnan(adjClose_noNaNs)]
            ymax = np.around(np.max(adjClose_noNaNs[firstdate_index:]) * 1.1)
            
            if uptrendSignalMethod == 'percentileChannels' and lowChannel is not None:
                ymin = np.around(np.min(lowChannel[i, firstdate_index:]) * 0.85)
            else:
                ymin = np.around(np.min(adjClose[i, firstdate_index:]) * 0.90)
            
            plt.ylim((ymin, ymax))
            xmin = datearray[firstdate_index]
            xmax = datearray[-1] + datetime.timedelta(10)
            plt.xlim((xmin, xmax))
            
            if uptrendSignalMethod == 'percentileChannels' and lowChannel is not None:
                plt.plot(datearray[firstdate_index:], lowChannel[i, firstdate_index:], 'm-')
                plt.plot(datearray[firstdate_index:], hiChannel[i, firstdate_index:], 'm-')
            
            # Plot trend channels
            relativedates = list(range(-params['numdaysinfit'] - params['offset'],
                                      -params['offset'] + 1))
            plt.plot(np.array(datearray)[relativedates], upperTrend, 'y-', lw=.5)
            plt.plot(np.array(datearray)[relativedates], lowerTrend, 'y-', lw=.5)
            plt.plot(
                [datearray[-1]],
                [(upperTrend[-1] + lowerTrend[-1]) / 2.],
                'y.', ms=10, alpha=.6
            )
            
            plt.plot(
                np.array(datearray)[-params['numdaysinfit2']:],
                NoGapUpperTrend,
                ls='-', c=(0, 0, 1), lw=1.
            )
            plt.plot(
                np.array(datearray)[-params['numdaysinfit2']:],
                NoGapLowerTrend,
                ls='-', c=(0, 0, 1), lw=1.
            )
            plt.plot(
                [datearray[-1]],
                [(NoGapUpperTrend[-1] + NoGapLowerTrend[-1]) / 2.],
                '.', c=(0, 0, 1), ms=10, alpha=.6
            )
            
            plt.plot(datearray[firstdate_index:], adjClose[i, firstdate_index:], 'k-', lw=.5)
            
            plot_text = str(adjClose[i, -7:])
            plt.text(datearray[firstdate_index + 10], ymin, plot_text, fontsize=10)
            
            # Add text with most recent date
            x_range = datearray[-1] - datearray[firstdate_index]
            text_x = datearray[firstdate_index] + datetime.timedelta(x_range.days / 20.)
            text_y = (ymax - ymin) * .085 + ymin
            
            plt.text(
                text_x, text_y,
                f"most recent value from {datearray[-1]}\n"
                f"plotted at {today.strftime('%A, %d. %B %Y %I:%M%p')}\n"
                f"value = {adjClose[i, -1]}",
                fontsize=8
            )
            plt.title(f"{symbols[i]}\n{ysq.get_company_name(symbols[i])}")
            
            # Change fontsize of tick labels
            plt.tick_params(axis='both', which='major', labelsize=8)
            plt.tick_params(axis='both', which='minor', labelsize=6)
            
            print(f" ...inside PortfolioPerformancealcs... plotfilepath = {plotfilepath}")
            plt.savefig(plotfilepath, format='png')
            
        except Exception as e:
            print(f" ERROR in PortfoloioPerformanceCalcs -- no plot generated for symbol {symbols[i]}")
            print(f" Exception: {e}")


def compute_portfolio_metrics(
    adjClose: np.ndarray,
    symbols: List[str],
    datearray: np.ndarray,
    params: Dict[str, Any],
    json_fn: str
) -> Dict[str, Any]:
    """Pure computation function for portfolio metrics.
    
    Computes all portfolio metrics including gains/losses, signals, weights,
    and portfolio values. This function has no side effects (no I/O, no prints)
    to enable testing and verification of computation logic.
    
    Args:
        adjClose: Adjusted closing prices (n_stocks, n_days)
        symbols: List of stock symbols (n_stocks,)
        datearray: Array of datetime objects (n_days,)
        params: Dictionary of parameters from JSON config
        json_fn: Path to JSON config file (for sharpeWeightedRank_2D)
    
    Returns:
        Dictionary containing all computed portfolio metrics:
            - gainloss: Daily gain/loss ratios (n_stocks, n_days)
            - value: Buy-and-hold values (n_stocks, n_days)
            - BuyHoldFinalValue: Final B&H portfolio value (scalar)
            - lastEmptyPriceIndex: Last index with constant price (n_stocks,)
            - activeCount: Count of active stocks per day (n_days,)
            - monthgainloss: Monthly gain/loss ratios (n_stocks, n_days)
            - signal2D: Uptrending signal (n_stocks, n_days)
            - signal2D_daily: Daily uptrending signal (n_stocks, n_days)
            - numberStocks: Count of uptrending stocks per day (n_days,)
            - dailyNumberUptrendingStocks: Same as numberStocks (n_days,)
            - lowChannel: Lower channel (optional, n_stocks, n_days)
            - hiChannel: Upper channel (optional, n_stocks, n_days)
            - monthgainlossweight: Portfolio weights (n_stocks, n_days)
            - monthvalue: Portfolio values (n_stocks, n_days)
            - numberSharesCalc: Number of shares (n_stocks, n_days)
            - last_symbols_text: List of currently held symbols
            - last_symbols_weight: List of current weights
            - last_symbols_price: List of current prices
    """
    
    #############################################################################
    # Phase 1: Compute basic gain/loss metrics
    #############################################################################
    
    gainloss = np.ones((adjClose.shape[0], adjClose.shape[1]), dtype=float)
    activeCount = np.zeros(adjClose.shape[1], dtype=float)
    numberSharesCalc = np.zeros((adjClose.shape[0], adjClose.shape[1]), dtype=float)
    
    gainloss[:, 1:] = adjClose[:, 1:] / adjClose[:, :-1]
    gainloss[np.isnan(gainloss)] = 1.
    gainloss[np.isinf(gainloss)] = 1.
    value = 10000. * np.cumprod(gainloss, axis=1)
    
    BuyHoldFinalValue = np.average(value, axis=0)[-1]
    
    lastEmptyPriceIndex = np.zeros(adjClose.shape[0], dtype=int)
    
    for ii in range(adjClose.shape[0]):
        # Take care of special case where constant share price is inserted at beginning
        index = np.argmax(np.clip(np.abs(gainloss[ii, :] - 1), 0, 1e-8)) - 1
        lastEmptyPriceIndex[ii] = index
        activeCount[lastEmptyPriceIndex[ii] + 1:] += 1
    
    # Remove NaN's from count for each day
    for ii in range(adjClose.shape[1]):
        numNaNs = (np.isnan(adjClose[:, ii]))
        numNaNs = numNaNs[numNaNs == True].shape[0]
        activeCount[ii] = activeCount[ii] - np.clip(numNaNs, 0., 99999)
    
    #############################################################################
    # Phase 2: Extract parameters
    #############################################################################
    
    monthsToHold = params['monthsToHold']
    numberStocksTraded = params['numberStocksTraded']
    LongPeriod = params['LongPeriod']
    stddevThreshold = float(params['stddevThreshold'])
    rankThresholdPct = float(params['rankThresholdPct'])
    riskDownside_min = float(params['riskDownside_min'])
    riskDownside_max = float(params['riskDownside_max'])
    
    #############################################################################
    # Phase 3: Compute monthly gain/loss
    #############################################################################
    
    monthgainloss = np.ones((adjClose.shape[0], adjClose.shape[1]), dtype=float)
    monthgainloss[:, LongPeriod:] = adjClose[:, LongPeriod:] / adjClose[:, :-LongPeriod]
    monthgainloss[isnan(monthgainloss)] = 1.
    
    #############################################################################
    # Phase 4: Compute signal for uptrending stocks
    #############################################################################
    
    # Import here to avoid circular dependency
    from functions.TAfunctions import computeSignal2D
    
    if params['uptrendSignalMethod'] == 'percentileChannels':
        signal2D, lowChannel, hiChannel = computeSignal2D(adjClose, gainloss, params)
    else:
        signal2D = computeSignal2D(adjClose, gainloss, params)
        lowChannel = None
        hiChannel = None
    
    # Copy to daily signal
    signal2D_daily = signal2D.copy()
    
    # Hold signal constant for each month
    for jj in np.arange(1, adjClose.shape[1]):
        if not ((datearray[jj].month != datearray[jj - 1].month) and 
                (datearray[jj].month - 1) % monthsToHold == 0):
            signal2D[:, jj] = signal2D[:, jj - 1]
    
    numberStocks = np.sum(signal2D, axis=0)
    dailyNumberUptrendingStocks = np.sum(signal2D, axis=0)
    
    #############################################################################
    # Phase 5: Compute weights for each stock
    #############################################################################
    
    # Import here to avoid circular dependency
    from functions.TAfunctions import sharpeWeightedRank_2D
    
    monthgainlossweight = sharpeWeightedRank_2D(
        json_fn, datearray, symbols, adjClose,
        signal2D, signal2D_daily, LongPeriod, numberStocksTraded,
        riskDownside_min, riskDownside_max, rankThresholdPct,
        stddevThreshold=stddevThreshold,
        is_backtest=False, makeQCPlots=True
    )
    
    #############################################################################
    # Phase 6: Compute traded value of stock for each month
    #############################################################################
    
    monthvalue = value.copy()
    for ii in np.arange(1, monthgainloss.shape[1]):
        if ((datearray[ii].month != datearray[ii - 1].month) and 
            ((datearray[ii].month - 1) % monthsToHold == 0)):
            valuesum = np.sum(monthvalue[:, ii - 1])
            for jj in range(value.shape[0]):
                # Re-balance using weights (that sum to 1.0)
                monthvalue[jj, ii] = monthgainlossweight[jj, ii] * valuesum * gainloss[jj, ii]
        else:
            for jj in range(value.shape[0]):
                monthvalue[jj, ii] = monthvalue[jj, ii - 1] * gainloss[jj, ii]
    
    numberSharesCalc = monthvalue / adjClose  # For info only
    
    #############################################################################
    # Phase 7: Extract current holdings
    #############################################################################
    
    last_symbols_text = []
    last_symbols_weight = []
    last_symbols_price = []
    for ii in range(len(symbols)):
        if monthgainlossweight[ii, -1] > 0:
            last_symbols_text.append(symbols[ii])
            last_symbols_weight.append(float(round(monthgainlossweight[ii, -1], 4)))
            last_symbols_price.append(float(round(adjClose[ii, -1], 2)))
    
    #############################################################################
    # Return all computed metrics
    #############################################################################
    
    result = {
        'gainloss': gainloss,
        'value': value,
        'BuyHoldFinalValue': BuyHoldFinalValue,
        'lastEmptyPriceIndex': lastEmptyPriceIndex,
        'activeCount': activeCount,
        'monthgainloss': monthgainloss,
        'signal2D': signal2D,
        'signal2D_daily': signal2D_daily,
        'numberStocks': numberStocks,
        'dailyNumberUptrendingStocks': dailyNumberUptrendingStocks,
        'monthgainlossweight': monthgainlossweight,
        'monthvalue': monthvalue,
        'numberSharesCalc': numberSharesCalc,
        'last_symbols_text': last_symbols_text,
        'last_symbols_weight': last_symbols_weight,
        'last_symbols_price': last_symbols_price
    }
    
    # Add optional channels if percentileChannels method
    if lowChannel is not None:
        result['lowChannel'] = lowChannel
    if hiChannel is not None:
        result['hiChannel'] = hiChannel
    
    return result
