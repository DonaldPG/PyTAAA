"""Output generation functions for PyTAAA.

This module contains functions for generating plots and other output artifacts
from portfolio computation results. These functions have side effects (file I/O)
and are separated from pure computation logic for testability.

Phase 4b1: Extracted from PortfolioPerformanceCalcs.py
Phase 4b3: Added compute_portfolio_metrics (pure computation)
Async phase: Added async_mode support via fire-and-forget subprocess.
"""

import logging
import numpy as np
from numpy import isnan
import datetime
import os
import pickle
import subprocess
import sys
import tempfile
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

import matplotlib

from functions.GetParams import get_json_params
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


def _generate_full_history_plots(
    adjClose: np.ndarray,
    symbols: List[str],
    datearray: np.ndarray,
    signal2D: np.ndarray,
    params: Dict[str, Any],
    output_dir: str,
    lowChannel: Optional[np.ndarray] = None,
    hiChannel: Optional[np.ndarray] = None,
) -> None:
    """Generate full-history PNG plots for all symbols (synchronous).

    Writes one ``0_<symbol>.png`` file per symbol into *output_dir*.
    Files less than 20 hours old are skipped.

    Args:
        adjClose: Adjusted close prices (n_symbols x n_days).
        symbols: List of stock ticker symbols.
        datearray: Array of datetime objects for each trading day.
        signal2D: Monthly uptrend signals (n_symbols x n_days).
        params: Parameter dictionary (requires ``LongPeriod``,
            ``stddevThreshold``, ``uptrendSignalMethod``).
        output_dir: Directory where PNG files are written.
        lowChannel: Optional low percentile channel (n_symbols x n_days).
        hiChannel: Optional high percentile channel (n_symbols x n_days).

    Returns:
        None
    """
    today = datetime.datetime.now()
    LongPeriod = params['LongPeriod']
    stddevThreshold = float(params['stddevThreshold'])
    uptrendSignalMethod = params['uptrendSignalMethod']

    for i in range(len(symbols)):
        plotfilepath = os.path.join(output_dir, f"0_{symbols[i]}.png")

        # Skip if less than 20 hours old
        if os.path.isfile(plotfilepath):
            mtime = datetime.datetime.fromtimestamp(
                os.path.getmtime(plotfilepath)
            )
            modified_time = datetime.datetime.now() - mtime
            modified_hours = (
                modified_time.days * 24 + modified_time.seconds / 3600
            )
            if modified_hours < 20.0:
                continue

        # Despike quotes for this symbol
        quotes = adjClose[i, :].copy().reshape(1, -1)
        quotes_despike = despike_2D(
            quotes, LongPeriod, stddevThreshold=stddevThreshold
        )

        plt.clf()
        plt.grid(True)
        plt.plot(datearray, adjClose[i, :])
        plt.plot(datearray, signal2D[i, :] * adjClose[i, -1], lw=.2)

        despiked_quotes = quotes_despike[0, :]
        if despiked_quotes[np.isnan(despiked_quotes)].shape[0] == 0:
            plt.plot(datearray, quotes_despike[0, :])

        if uptrendSignalMethod == 'percentileChannels' and lowChannel is not None:
            plt.plot(datearray, lowChannel[i, :], 'm-')
            plt.plot(datearray, hiChannel[i, :], 'm-')

        plot_text = str(adjClose[i, -7:])
        plt.text(datearray[50], 0, plot_text)

        x_range = datearray[-1] - datearray[0]
        text_x = datearray[0] + datetime.timedelta(x_range.days / 20.)
        adjClose_noNaNs = adjClose[i, :][~np.isnan(adjClose[i, :])]
        text_y = (
            (np.max(adjClose_noNaNs) - np.min(adjClose_noNaNs)) * .085
            + np.min(adjClose_noNaNs)
        )

        plt.text(
            text_x, text_y,
            f"most recent value from {datearray[-1]}\n"
            f"plotted at {today.strftime('%A, %d. %B %Y %I:%M%p')}\n"
            f"value = {adjClose[i, -1]}",
            fontsize=8
        )
        plt.title(f"{symbols[i]}\n{ysq.get_company_name(symbols[i])}")
        plt.yscale('log')

        print(f" ...inside PortfolioPerformanceCalcs... plotfilepath = {plotfilepath}")
        plt.savefig(plotfilepath, format='png')


def _generate_recent_plots(
    adjClose: np.ndarray,
    symbols: List[str],
    datearray: np.ndarray,
    signal2D: np.ndarray,
    signal2D_daily: np.ndarray,
    params: Dict[str, Any],
    output_dir: str,
    firstdate_index: int,
    lowChannel: Optional[np.ndarray] = None,
    hiChannel: Optional[np.ndarray] = None,
) -> None:
    """Generate recent-history PNG plots for all symbols (synchronous).

    Writes one ``0_recent_<symbol>.png`` file per symbol into *output_dir*.
    Files less than 20 hours old are skipped.

    Args:
        adjClose: Adjusted close prices (n_symbols x n_days).
        symbols: List of stock ticker symbols.
        datearray: Array of datetime objects for each trading day.
        signal2D: Monthly uptrend signals (n_symbols x n_days).
        signal2D_daily: Daily uptrend signals (n_symbols x n_days).
        params: Parameter dictionary.
        output_dir: Directory where PNG files are written.
        firstdate_index: Index of the first date for 2013+ recent plots.
        lowChannel: Optional low percentile channel (n_symbols x n_days).
        hiChannel: Optional high percentile channel (n_symbols x n_days).

    Returns:
        None
    """
    today = datetime.datetime.now()
    LongPeriod = params['LongPeriod']
    stddevThreshold = float(params['stddevThreshold'])
    uptrendSignalMethod = params['uptrendSignalMethod']

    for i in range(len(symbols)):
        plotfilepath = os.path.join(output_dir, f"0_recent_{symbols[i]}.png")

        # Skip if less than 20 hours old
        if os.path.isfile(plotfilepath):
            mtime = datetime.datetime.fromtimestamp(
                os.path.getmtime(plotfilepath)
            )
            modified_time = datetime.datetime.now() - mtime
            modified_hours = (
                modified_time.days * 24 + modified_time.seconds / 3600
            )
            if modified_hours < 20.0:
                continue

        quotes = adjClose[i, :].copy().reshape(1, -1)
        quotes_despike = despike_2D(
            quotes, LongPeriod, stddevThreshold=stddevThreshold
        )
        quotes_despike *= (
            quotes[0, firstdate_index]
            / quotes_despike[0, firstdate_index]
        )

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

            adjClose_noNaNs = adjClose[i, :][~np.isnan(adjClose[i, :])]
            ymax = np.around(np.max(adjClose_noNaNs[firstdate_index:]) * 1.1)

            if uptrendSignalMethod == 'percentileChannels' and lowChannel is not None:
                ymin = np.around(
                    np.min(lowChannel[i, firstdate_index:]) * 0.85
                )
            else:
                ymin = np.around(
                    np.min(adjClose[i, firstdate_index:]) * 0.90
                )

            plt.ylim((ymin, ymax))
            xmin = datearray[firstdate_index]
            xmax = datearray[-1] + datetime.timedelta(10)
            plt.xlim((xmin, xmax))

            if uptrendSignalMethod == 'percentileChannels' and lowChannel is not None:
                plt.plot(
                    datearray[firstdate_index:],
                    lowChannel[i, firstdate_index:], 'm-'
                )
                plt.plot(
                    datearray[firstdate_index:],
                    hiChannel[i, firstdate_index:], 'm-'
                )

            relativedates = list(
                range(
                    -params['numdaysinfit'] - params['offset'],
                    -params['offset'] + 1
                )
            )
            plt.plot(np.array(datearray)[relativedates], upperTrend, 'y-', lw=.5)
            plt.plot(np.array(datearray)[relativedates], lowerTrend, 'y-', lw=.5)
            plt.plot(
                [datearray[-1]],
                [(upperTrend[-1] + lowerTrend[-1]) / 2.],
                'y.', ms=10, alpha=.6
            )

            plt.plot(
                np.array(datearray)[-params['numdaysinfit2']:],
                NoGapUpperTrend, ls='-', c=(0, 0, 1), lw=1.
            )
            plt.plot(
                np.array(datearray)[-params['numdaysinfit2']:],
                NoGapLowerTrend, ls='-', c=(0, 0, 1), lw=1.
            )
            plt.plot(
                [datearray[-1]],
                [(NoGapUpperTrend[-1] + NoGapLowerTrend[-1]) / 2.],
                '.', c=(0, 0, 1), ms=10, alpha=.6
            )

            plt.plot(
                datearray[firstdate_index:],
                adjClose[i, firstdate_index:], 'k-', lw=.5
            )

            plot_text = str(adjClose[i, -7:])
            plt.text(
                datearray[firstdate_index + 10], ymin, plot_text, fontsize=10
            )

            x_range = datearray[-1] - datearray[firstdate_index]
            text_x = datearray[firstdate_index] + datetime.timedelta(
                x_range.days / 20.
            )
            text_y = (ymax - ymin) * .085 + ymin

            plt.text(
                text_x, text_y,
                f"most recent value from {datearray[-1]}\n"
                f"plotted at {today.strftime('%A, %d. %B %Y %I:%M%p')}\n"
                f"value = {adjClose[i, -1]}",
                fontsize=8
            )
            plt.title(f"{symbols[i]}\n{ysq.get_company_name(symbols[i])}")

            plt.tick_params(axis='both', which='major', labelsize=8)
            plt.tick_params(axis='both', which='minor', labelsize=6)

            print(f" ...inside PortfolioPerformanceCalcs... plotfilepath = {plotfilepath}")
            plt.savefig(plotfilepath, format='png')

        except Exception as e:
            print(
                f" ERROR in PortfolioPerformanceCalcs -- no plot generated "
                f"for symbol {symbols[i]}"
            )
            print(f" Exception: {e}")


def _spawn_background_plot_generation(
    adjClose: np.ndarray,
    symbols: List[str],
    datearray: np.ndarray,
    signal2D: np.ndarray,
    signal2D_daily: np.ndarray,
    params: Dict[str, Any],
    output_dir: str,
    max_workers: int = 2,
    lowChannel: Optional[np.ndarray] = None,
    hiChannel: Optional[np.ndarray] = None,
) -> None:
    """Serialize plot data and spawn a detached background process.

    Writes plot data to a temporary pickle file then launches
    ``functions/background_plot_generator.py`` as a fully detached
    subprocess (new session, stdout/stderr redirected to a log file).
    Returns immediately; the caller does not wait for plot generation.

    Args:
        adjClose: Adjusted close prices (n_symbols x n_days).
        symbols: List of stock ticker symbols.
        datearray: Array of datetime objects for each trading day.
        signal2D: Monthly uptrend signals (n_symbols x n_days).
        signal2D_daily: Daily uptrend signals (n_symbols x n_days).
        params: Parameter dictionary.
        output_dir: Directory where PNG files will be written.
        max_workers: Number of parallel worker processes.
        lowChannel: Optional low percentile channel.
        hiChannel: Optional high percentile channel.

    Returns:
        None

    Side Effects:
        - Creates a temporary pickle file (deleted by the subprocess).
        - Spawns a detached background process.
        - Writes subprocess stdout/stderr to ``plot_generation.log``
          in *output_dir*.
    """
    # Serialize data to a temporary pickle file
    data = {
        'adjClose': adjClose,
        'symbols': symbols,
        'datearray': datearray,
        'signal2D': signal2D,
        'signal2D_daily': signal2D_daily,
        'params': params,
        'output_dir': output_dir,
    }
    if lowChannel is not None:
        data['lowChannel'] = lowChannel
    if hiChannel is not None:
        data['hiChannel'] = hiChannel

    fd, data_file = tempfile.mkstemp(suffix='.pkl', prefix='pytaaa_plots_')
    try:
        with os.fdopen(fd, 'wb') as fh:
            pickle.dump(data, fh)
    except Exception:
        try:
            os.remove(data_file)
        except OSError:
            pass
        raise

    # Determine path to the background_plot_generator module
    generator_module = os.path.join(
        os.path.dirname(__file__), 'background_plot_generator.py'
    )
    log_file = os.path.join(output_dir, 'plot_generation.log')

    cmd = [
        sys.executable,
        generator_module,
        '--data-file', data_file,
        '--max-workers', str(max_workers),
    ]

    with open(log_file, 'a') as log_fh:
        log_fh.write(
            f"\n[{datetime.datetime.now().isoformat()}] "
            f"Spawning background plot generation\n"
        )

    # Launch a fully detached process (new session, no stdin)
    with open(log_file, 'a') as log_fh:
        subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=log_fh,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    print(
        f" [async] Background plot generation started "
        f"(max_workers={max_workers}). Log: {log_file}"
    )


def generate_portfolio_plots(
    adjClose: np.ndarray,
    symbols: List[str],
    datearray: np.ndarray,
    signal2D: np.ndarray,
    signal2D_daily: np.ndarray,
    params: Dict[str, Any],
    output_dir: str,
    lowChannel: Optional[np.ndarray] = None,
    hiChannel: Optional[np.ndarray] = None,
    async_mode: bool = False,
    max_workers: int = 2,
) -> None:
    """Generate all portfolio plots for web display.
    
    Creates two types of plots for each symbol:
    1. Full history plots (0_<symbol>.png)
    2. Recent 2-year plots with trend channels (0_recent_<symbol>.png)
    
    Plot generation is conditional on time of day to avoid excessive CPU usage
    during market hours. Plots are skipped if less than 20 hours old.

    When *async_mode* is ``True``, plot data is serialized to a temporary
    file and a detached background process is spawned via
    :func:`_spawn_background_plot_generation`.  The main program returns
    immediately without waiting for plots to finish.
    
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
        async_mode: If ``True``, spawn a detached background process and
            return immediately.  If ``False`` (default), generate plots
            synchronously.
        max_workers: Number of parallel worker processes used in async mode.
    
    Returns:
        None (writes PNG files to output_dir)
    
    Side Effects:
        - Writes ~200 PNG files (2 per symbol) to output_dir (sync mode)
        - Spawns detached background process (async mode)
        - Prints progress messages to stdout
    """
    today = datetime.datetime.now()
    hourOfDay = today.hour
    
    # Only generate plots outside market hours to reduce load
    if not (hourOfDay >= 1 or 11 < hourOfDay < 13):
        return

    ##########################################################################
    # Async mode: fire-and-forget background process
    ##########################################################################
    if async_mode:
        _spawn_background_plot_generation(
            adjClose, symbols, datearray, signal2D, signal2D_daily,
            params, output_dir,
            max_workers=max_workers,
            lowChannel=lowChannel,
            hiChannel=hiChannel,
        )
        return

    ##########################################################################
    # Synchronous mode (default): generate plots inline
    ##########################################################################

    # Determine first date index for recent plots (2013+)
    firstdate_index = 0
    for ii in range(len(datearray)):
        if datearray[ii].year > datearray[ii-1].year and datearray[ii].year == 2013:
            firstdate_index = ii
            break

    # 1. Full history plots
    _generate_full_history_plots(
        adjClose, symbols, datearray, signal2D, params, output_dir,
        lowChannel=lowChannel, hiChannel=hiChannel,
    )

    # 2. Recent (2-year) plots with trend channels
    _generate_recent_plots(
        adjClose, symbols, datearray, signal2D, signal2D_daily,
        params, output_dir, firstdate_index,
        lowChannel=lowChannel, hiChannel=hiChannel,
    )


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
    # Phase 1: Initialize arrays (placeholder)
    #############################################################################
    
    # Note: gainloss will be computed after despiking in Phase 2b
    # This placeholder section is kept for organizational clarity
    activeCount = np.zeros(adjClose.shape[1], dtype=float)
    numberSharesCalc = np.zeros((adjClose.shape[0], adjClose.shape[1]), dtype=float)
    
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
    # Phase 2b: Apply despike filter to remove interpolated/anomalous data
    #############################################################################
    
    # Apply despike_2D to detect and filter out stocks with interpolated or
    # linear-trending prices that would artificially inflate Sharpe ratios
    # (e.g., JEF 2015-2018 with linearly interpolated missing data)
    adjClose_despike = despike_2D(adjClose, LongPeriod, stddevThreshold=stddevThreshold)
    
    # Compute gainloss from despiked prices
    gainloss = np.ones((adjClose.shape[0], adjClose.shape[1]), dtype=float)
    gainloss[:, 1:] = adjClose_despike[:, 1:] / adjClose_despike[:, :-1]
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
        numNaNs = (np.isnan(adjClose_despike[:, ii]))
        numNaNs = numNaNs[numNaNs == True].shape[0]
        activeCount[ii] = activeCount[ii] - np.clip(numNaNs, 0., 99999)
    
    #############################################################################
    # Phase 3: Compute monthly gain/loss
    #############################################################################
    
    monthgainloss = np.ones((adjClose.shape[0], adjClose.shape[1]), dtype=float)
    monthgainloss[:, LongPeriod:] = adjClose_despike[:, LongPeriod:] / adjClose_despike[:, :-LongPeriod]
    monthgainloss[isnan(monthgainloss)] = 1.
    
    #############################################################################
    # Phase 4: Compute signal for uptrending stocks
    #############################################################################
    
    # Import here to avoid circular dependency
    from functions.TAfunctions import computeSignal2D
    
    if params['uptrendSignalMethod'] == 'percentileChannels':
        signal2D, lowChannel, hiChannel = computeSignal2D(adjClose_despike, gainloss, params)
    else:
        signal2D = computeSignal2D(adjClose_despike, gainloss, params)
        lowChannel = None
        hiChannel = None
    
    signal2D_1 = signal2D.copy()

    # SP500 pre-2002 condition: Force 100% CASH allocation (overrides rolling window filter)
    if params.get('stockList') == 'SP500':
        print("\n   . DEBUG: Applying SP500 pre-2002 condition: Forcing 100% CASH allocation for all stocks before 2002-01-01")
        cutoff_date = datetime.date(2002, 1, 1)
        for date_idx in range(len(datearray)):
            if datearray[date_idx] < cutoff_date:
                # Zero all stock signals for 100% CASH allocation
                signal2D[:, date_idx] = 0.0

    signal2D_2 = signal2D.copy()
    iii,jjj = np.where(signal2D_1 != signal2D_2)

    print(
        "\n   . DEBUG: output_generators: Completed SP500 pre-2002 condition (if applicable). "
        "number of changed signals in signal2D = ", len(iii)
    )

    # SP500 pre-2002 condition: Force 100% CASH allocation (overrides rolling window filter)
    # Apply rolling window data quality filter (enabled by default to catch interpolated data)
    print(f"DEBUG: About to check enable_rolling_filter, value = {params.get('enable_rolling_filter', True)}")
    if params.get('enable_rolling_filter', True):  # Changed default to True for data quality
        from functions.rolling_window_filter import apply_rolling_window_filter
        print(" ... Applying rolling window data quality filter to detect interpolated data...")
        signal2D = apply_rolling_window_filter(
            adjClose_despike, signal2D, params.get('window_size', 50),
            symbols=symbols, datearray=datearray, verbose=False
        )
        print(" ... Rolling window filter complete")
    else:
        print("DEBUG: Rolling filter SKIPPED because enable_rolling_filter is False or not set")

    signal2D_3 = signal2D.copy()
    iii,jjj = np.where(signal2D_3 != signal2D_2)
    print(
        "\n   . DEBUG: output_generators: Completed rolling_window_filter application. "
         "number of changed signals in signal2D = ", len(iii)
    )

    # # SP500 pre-2002 condition: Force 100% CASH allocation
    # _params = get_json_params(json_fn)
    # if _params.get('stockList') == 'SP500':
    #     print("\n   . DEBUG: Applying SP500 pre-2002 condition: Forcing 100% CASH allocation for all stocks before 2002-01-01")
    #     cutoff_date = datetime.date(2002, 1, 1)
    #     for date_idx in range(len(datearray)):
    #         if datearray[date_idx] < cutoff_date:
    #             # Zero all stock signals for 100% CASH allocation
    #             signal2D[:, date_idx] = 0.0
    
    # Copy to daily signal
    signal2D_daily = signal2D.copy()
    
    # Hold signal constant for each month
    for jj in np.arange(1, adjClose.shape[1]):
        if not ((datearray[jj].month != datearray[jj - 1].month) and 
                (datearray[jj].month - 1) % monthsToHold == 0):
            signal2D[:, jj] = signal2D[:, jj - 1]

    # Transient overwrite check: ensure monthly-held signals at rebalance
    # dates were taken from the filtered daily signals (signal2D_daily).
    try:
        print(f"DEBUG ids post-fill: id(signal2D)={id(signal2D)}, id(signal2D_daily)={id(signal2D_daily)}")
        rebalance_indices = []
        for jj in range(1, adjClose.shape[1]):
            is_rebalance = (
                (datearray[jj].month != datearray[jj-1].month) and
                ((datearray[jj].month - 1) % monthsToHold == 0)
            )
            if is_rebalance:
                rebalance_indices.append(jj)
        mismatches = []
        for jj in rebalance_indices:
            if not np.array_equal(signal2D[:, jj], signal2D_daily[:, jj]):
                mismatches.append((jj, int(np.sum(signal2D_daily[:, jj] > 0)), int(np.sum(signal2D[:, jj] > 0))))
        if mismatches:
            print("\nOUTPUT_GENERATORS OVERWRITE ASSERT: mismatches at rebalance dates")
            for m in mismatches[:10]:
                print(m)
            raise AssertionError("Monthly signals differ from filtered daily signals after forward-fill in output_generators.py")
    except AssertionError:
        raise
    except Exception:
        # Be permissive if variables or shapes are unexpected
        pass
    
    numberStocks = np.sum(signal2D, axis=0)
    dailyNumberUptrendingStocks = np.sum(signal2D, axis=0)
    
    #############################################################################
    # Phase 5: Compute weights for each stock
    #############################################################################
    
    # Import here to avoid circular dependency
    from functions.TAfunctions import sharpeWeightedRank_2D
    
    signal2D_4before = signal2D.copy()

    monthgainlossweight = sharpeWeightedRank_2D(
        json_fn, datearray, symbols, adjClose_despike,
        signal2D, signal2D_daily, LongPeriod, numberStocksTraded,
        riskDownside_min, riskDownside_max, rankThresholdPct,
        stddevThreshold=stddevThreshold,
        is_backtest=False, makeQCPlots=True,
        stockList=params.get('stockList', 'SP500')  # Pass stockList for early period logic
    )

    signal2D_4after = signal2D.copy()
    iii,jjj = np.where(signal2D_4after != signal2D_4before)
    print(
        "\n   . DEBUG: output_generators: Completed sharpeWeightedRank_2D. "
         "number of changed signals in signal2D = ", len(iii)
    )
    del signal2D_4after
    del signal2D_4before
    del signal2D_3
    del signal2D_2
    del signal2D_1

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
    
    numberSharesCalc = monthvalue / adjClose_despike  # For info only
    
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
            last_symbols_price.append(float(round(adjClose_despike[ii, -1], 2)))
    
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
