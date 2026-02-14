"""Output generation functions for PyTAAA.

This module contains functions for generating plots and other output artifacts
from portfolio computation results. These functions have side effects (file I/O)
and are separated from pure computation logic for testability.

Phase 4b1: Extracted from PortfolioPerformanceCalcs.py
"""

import numpy as np
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
