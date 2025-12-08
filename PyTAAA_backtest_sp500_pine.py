import time, threading

import numpy as np
# from matplotlib.pylab import *
from matplotlib import pylab as plt

import matplotlib.gridspec as gridspec
import os

import datetime
from numpy import random
from scipy import ndimage
from random import choice
from scipy.stats import rankdata

import pandas as pd
from scipy.stats import gmean

## local imports
from functions.quotes_for_list_adjClose import *
from functions.TAfunctions import *
from functions.TAfunctions import SMA, hma
from functions.TAfunctions import SMA_filtered_2D
from functions.TAfunctions import sharpeWeightedRank_2D
from functions.UpdateSymbols_inHDF5 import loadQuotes_fromHDF

import datetime
import numpy as np
import os
import time
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
from functions.PortfolioPerformanceCalcs import PortfolioPerformanceCalcs
from functions.quotes_for_list_adjClose import LastQuotesForSymbolList_hdf, get_SectorAndIndustry_google
from functions.calculateTrades import calculateTrades
# from functions.quotes_for_list_adjClose import get_Naz100List, get_SP500List
from functions.readSymbols import get_symbols_changes
from functions.stock_cluster import getClusterForSymbolsList
from functions.ftp_quotes import copy_updated_quotes

#############################################################################
# Helper functions for Monte Carlo parameter management
#############################################################################

#############################################################################
# Plot Display Configuration
#############################################################################
show_cagr_in_plot = True  # True = show CAGR in plots, False = show AvgProfit

def calculate_cagr(end_value, start_value, days):
    """
    Calculate Compound Annual Growth Rate (CAGR) with proper error handling.
    
    Args:
        end_value: Portfolio value at end of period
        start_value: Portfolio value at start of period  
        days: Number of trading days in period
        
    Returns:
        CAGR as decimal (e.g., 0.125 for 12.5% annual growth)
    """
    if start_value <= 0 or end_value <= 0 or days <= 0:
        return 0.0
    
    try:
        # Standard CAGR formula: (End/Start)^(252/days) - 1
        cagr = (end_value / start_value) ** (252.0 / days) - 1.0
        
        # Validate reasonable CAGR range (-50% to +100%)
        if cagr < -0.5 or cagr > 1.0:
            print(f" ... Warning: CAGR {cagr:.3f} outside reasonable range")
            
        return cagr
        
    except (ZeroDivisionError, ValueError, OverflowError) as e:
        print(f" ... Error calculating CAGR: {e}")
        return 0.0


def create_temporary_json(base_json_fn, realization_params, iter_num):
    """
    Create a temporary JSON file for a single Monte Carlo realization.
    
    Args:
        base_json_fn: Path to base JSON configuration file
        realization_params: Dictionary with parameters for this realization
        iter_num: Iteration number for unique temp file naming
        
    Returns:
        Path to temporary JSON file
    """
    import json
    import tempfile
    
    # Load base parameters
    try:
        with open(base_json_fn, 'r') as f:
            base_params = json.load(f)
    except Exception as e:
        print(f" ... Warning: Could not load base JSON file: {e}")
        # Create minimal base structure if file doesn't exist or is corrupt
        base_params = {
            "Email": {"To": "", "From": "", "PW": "", "IPaddress": ""},
            "Text": {"phoneEmail": "", "send_texts": False},
            "FTP": {"hostname": "", "remoteIP": "", "username": "", 
                   "password": "", "remotepath": ""},
            "Stock Server": {"quote_download_server": ""},
            "Setup": {"runtime": "15 days", "pausetime": "24 hours"},
            "Valuation": {}
        }
    
    # Update with realization-specific parameters
    updated_params = base_params.copy()
    
    # Ensure the Valuation section exists
    if "Valuation" not in updated_params:
        updated_params["Valuation"] = {}
    
    # Update Valuation section with our parameters
    updated_params["Valuation"].update(realization_params)
    
    # Create temporary file
    temp_dir = os.path.dirname(base_json_fn)
    temp_json_fn = os.path.join(temp_dir, f"temp_realization_{iter_num}.json")
    
    # Write temporary JSON file
    try:
        with open(temp_json_fn, 'w') as f:
            json.dump(updated_params, f, indent=2)
        print(f" ... Successfully created temp JSON: {temp_json_fn}")
    except Exception as e:
        print(f" ... Error writing temp JSON file: {e}")
        raise
        
    return temp_json_fn


def cleanup_temporary_json(temp_json_fn):
    """
    Clean up temporary JSON file.
    
    Args:
        temp_json_fn: Path to temporary JSON file to remove
    """
    try:
        if os.path.exists(temp_json_fn):
            os.remove(temp_json_fn)
            print(f"Cleaned up temporary file: {temp_json_fn}")
    except Exception as e:
        print(f"Warning: Could not remove temporary file {temp_json_fn}: {e}")


def run_single_monte_carlo_realization(
    base_json_fn, 
    realization_params, 
    iter_num,
    adjClose, 
    symbols, 
    datearray,
    gainloss,
    value,
    activeCount,
    holdMonths,
    verbose=False
):
    """
    Run a single Monte Carlo realization using temporary JSON configuration.
    
    Args:
        base_json_fn: Path to base JSON configuration file
        realization_params: Dictionary with parameters for this realization
        iter_num: Current iteration number
        adjClose: Stock price data
        symbols: Stock symbols
        datearray: Date array
        gainloss: Gain/loss data
        value: Portfolio values
        activeCount: Active stock counts
        holdMonths: Available holding periods
        verbose: Enable verbose output
        
    Returns:
        Dictionary with backtest results for this realization
    """
    temp_json_fn = None
    
    try:
        print(f" ... Creating temporary JSON for realization {iter_num}")
        
        # Create temporary JSON file
        temp_json_fn = create_temporary_json(base_json_fn, realization_params, iter_num)
        
        print(f" ... Created temp file: {temp_json_fn}")
        
        # Load parameters from temporary JSON
        params = get_json_params(temp_json_fn)
        
        if params is None:
            raise ValueError("Failed to load parameters from temporary JSON file")
        
        print(f" ... Loaded parameters successfully")
        
        # Extract parameters with validation
        monthsToHold = params.get('monthsToHold')
        numberStocksTraded = params.get('numberStocksTraded')
        LongPeriod = params.get('LongPeriod')
        stddevThreshold = params.get('stddevThreshold')
        MA1 = int(params.get('MA1', 100))
        MA2 = int(params.get('MA2', 20))
        MA2offset = int(params.get('MA2offset', params.get('MA3', MA2+2) - MA2))
        sma2factor = params.get('sma2factor', params.get('MA2factor', 0.91))
        rankThresholdPct = params.get('rankThresholdPct')
        riskDownside_min = params.get('riskDownside_min')
        riskDownside_max = params.get('riskDownside_max')
        lowPct = float(params.get('lowPct', 20.0))
        hiPct = float(params.get('hiPct', 80.0))
        uptrendSignalMethod = params.get('uptrendSignalMethod', 'percentileChannels')
        sma_filt_val = params.get('sma_filt_val', 0.02)
        
        # Validate critical parameters
        if None in [monthsToHold, numberStocksTraded, LongPeriod, stddevThreshold, 
                   rankThresholdPct, riskDownside_min, riskDownside_max]:
            raise ValueError("One or more critical parameters is None")
        
        # Validate array inputs
        if adjClose is None:
            raise ValueError("adjClose is None")
        if symbols is None:
            raise ValueError("symbols is None")
        if datearray is None:
            raise ValueError("datearray is None")
        if gainloss is None:
            raise ValueError("gainloss is None")
        if value is None:
            raise ValueError("value is None")
            
        print(f" ... Validated all inputs for realization {iter_num}")
        
        if verbose:
            print(f" ... Running realization {iter_num} with uptrendSignalMethod: {uptrendSignalMethod}")
            print(f" ... Parameters: lowPct={lowPct}, hiPct={hiPct}")
            print(f" ... Array shapes: adjClose={adjClose.shape}, value={value.shape}")
        
        # Run the core backtest logic using the temporary JSON
        results = execute_single_backtest(
            temp_json_fn,
            adjClose, symbols, datearray, gainloss, value, activeCount,
            monthsToHold, numberStocksTraded, LongPeriod, stddevThreshold,
            MA1, MA2, MA2offset, sma2factor, rankThresholdPct,
            riskDownside_min, riskDownside_max, lowPct, hiPct,
            uptrendSignalMethod, sma_filt_val, iter_num, verbose
        )
        
        if results is None:
            raise ValueError("execute_single_backtest returned None")
            
        print(f" ... Realization {iter_num} completed successfully")
        
        return results
        
    except Exception as e:
        print(f" ... ERROR in run_single_monte_carlo_realization for iter {iter_num}: {str(e)}")
        import traceback
        print(f" ... Traceback: {traceback.format_exc()}")
        
        # Return a minimal results dictionary to prevent further errors
        return {
            'iter': iter_num,
            'finalValue': 10000.0,  # Starting value
            'sharpeRatio': 0.0,
            'monthvalue': value.copy() if value is not None else np.ones((100, 1000)) * 10000,
            'signal2D': np.zeros((100, 1000)) if adjClose is None else np.zeros_like(adjClose),
            'numberStocks': np.zeros(1000) if datearray is None else np.zeros(len(datearray)),
            'monthgainlossweight': np.zeros((100, 1000)) if adjClose is None else np.zeros_like(adjClose),
            'parameters': realization_params.copy(),
            'error': str(e)
        }
        
    finally:
        # Always clean up temporary file
        if temp_json_fn:
            cleanup_temporary_json(temp_json_fn)


def execute_single_backtest(
    json_fn,
    adjClose, symbols, datearray, gainloss, value, activeCount,
    monthsToHold, numberStocksTraded, LongPeriod, stddevThreshold,
    MA1, MA2, MA2offset, sma2factor, rankThresholdPct,
    riskDownside_min, riskDownside_max, lowPct, hiPct,
    uptrendSignalMethod, sma_filt_val, iter_num, verbose=False
):
    """
    Execute the core backtest logic for a single realization.
    
    Returns:
        Dictionary with backtest results
    """
    print(f" ... Computing signals for realization {iter_num}")
    print(f" ... Using {uptrendSignalMethod} with lowPct={lowPct}, hiPct={hiPct}")
    
    # Create monthly gain/loss
    monthgainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    monthgainloss[:,LongPeriod:] = adjClose[:,LongPeriod:] / adjClose[:,:-LongPeriod]
    monthgainloss[np.isnan(monthgainloss)]=1.

    #############################################################################
    # Generate signals using percentile channels directly
    #############################################################################
    
    print(f" ... Calling percentileChannel_2D with MA1={MA1}, MA2={MA2}, MA2offset={MA2offset}")
    
    try:
        # Call percentileChannel_2D directly instead of relying on computeSignal2D
        lowChannel, hiChannel = percentileChannel_2D(
            adjClose, MA2, MA1+0.01, MA2offset, lowPct, hiPct
        )
        
        print(f" ... Generated channels with shapes: low={lowChannel.shape}, hi={hiChannel.shape}")
        
        # Initialize signal array
        signal2D = np.zeros_like(adjClose, dtype=float)
        
        # Generate buy/sell signals based on percentile channels
        for i in range(adjClose.shape[0]):  # For each stock
            for j in range(1, adjClose.shape[1]):  # For each date
                price = adjClose[i, j]
                low_thresh = lowChannel[i, j]
                hi_thresh = hiChannel[i, j]
                
                if not (np.isnan(price) or np.isnan(low_thresh) or np.isnan(hi_thresh)):
                    # Percentile channel signal logic
                    if price > hi_thresh:
                        signal2D[i, j] = 1.0  # Strong buy signal - price above upper channel
                    elif price > low_thresh:
                        signal2D[i, j] = signal2D[i, j-1]  # Hold previous signal - price between channels
                    else:
                        signal2D[i, j] = 0.0  # Sell signal - price below lower channel
                else:
                    # Handle NaN values by maintaining previous signal
                    signal2D[i, j] = signal2D[i, j-1] if j > 0 else 0.0
        
        print(f" ... Generated signal2D with shape: {signal2D.shape}")
        print(f" ... Signal2D stats: min={signal2D.min():.3f}, max={signal2D.max():.3f}, mean={signal2D.mean():.3f}")
        
    except Exception as e:
        print(f" ... Error generating percentile channels: {e}")
        print(" ... Using fallback signal generation")
        
        # Fallback to simple moving average signals
        try:
            # Use filtered SMAs as fallback
            sma_short = SMA_filtered_2D(adjClose, MA2, sma_filt_val)
            sma_long = SMA_filtered_2D(adjClose, MA1, sma_filt_val)
            
            signal2D = np.zeros_like(adjClose, dtype=float)
            signal2D[sma_short > sma_long] = 1.0
            
            print(" ... Generated fallback SMA-based signals")
            
        except Exception as e2:
            print(f" ... Error with SMA fallback: {e2}")
            # Final fallback - simple trend-following signals
            signal2D = np.ones_like(adjClose) * 0.5  # Neutral signal
            print(" ... Using neutral signals as final fallback")
    
    # Create signal2D_daily for daily signals (before monthly hold logic)
    signal2D_daily = signal2D.copy()
    
    # Hold signal constant for each month based on monthsToHold parameter
    for jj in np.arange(1, adjClose.shape[1]):
        if not ((datearray[jj].month != datearray[jj-1].month) and 
                (datearray[jj].month - 1) % monthsToHold == 0):
            signal2D[:, jj] = signal2D[:, jj-1]

    numberStocks = np.sum(signal2D, axis=0)
    print(f" ... Number of stocks with signals: min={numberStocks.min():.1f}, max={numberStocks.max():.1f}, mean={numberStocks.mean():.1f}")

    #############################################################################
    # Compute portfolio weights using sharpeWeightedRank_2D
    #############################################################################
    
    try:
        monthgainlossweight = sharpeWeightedRank_2D(
            json_fn, datearray, symbols, adjClose, signal2D, signal2D_daily,
            LongPeriod, numberStocksTraded,
            riskDownside_min, riskDownside_max, rankThresholdPct,
            stddevThreshold=stddevThreshold,
            makeQCPlots=False
        )
        print(f" ... Generated weights with shape: {monthgainlossweight.shape}")
        
    except Exception as e:
        print(f" ... Error in sharpeWeightedRank_2D: {e}")
        print(" ... Using fallback equal-weight allocation")
        
        # Fallback to equal-weight allocation for stocks with positive signals
        monthgainlossweight = np.zeros_like(adjClose, dtype=float)
        
        for j in range(adjClose.shape[1]):
            # Get stocks with positive signals
            signals_col = signal2D[:, j]
            positive_signals = signals_col > 0.5
            n_positive = np.sum(positive_signals)
            
            if n_positive > 0:
                # Limit to numberStocksTraded and assign equal weights
                n_to_trade = min(n_positive, numberStocksTraded)
                weight_per_stock = 1.0 / n_to_trade
                
                # Get indices of stocks with positive signals
                stock_indices = np.where(positive_signals)[0]
                
                # Assign equal weights to top signaling stocks
                for idx in stock_indices[:n_to_trade]:
                    monthgainlossweight[idx, j] = weight_per_stock

    #############################################################################
    # Compute portfolio values over time
    #############################################################################
    
    monthvalue = value.copy()
    
    for ii in np.arange(1, monthgainloss.shape[1]):
        # Check if this is a rebalancing date (monthly based on monthsToHold)
        if ((datearray[ii].month != datearray[ii-1].month) and 
            ((datearray[ii].month - 1) % monthsToHold == 0)):
            
            # Rebalancing date - apply new weights
            valuesum = np.sum(monthvalue[:, ii-1])
            for jj in range(value.shape[0]):
                monthvalue[jj, ii] = (monthgainlossweight[jj, ii] * valuesum * 
                                    gainloss[jj, ii])
        else:
            # Non-rebalancing date - maintain existing positions
            for jj in range(value.shape[0]):
                monthvalue[jj, ii] = monthvalue[jj, ii-1] * gainloss[jj, ii]

    #############################################################################
    # Calculate performance metrics
    #############################################################################
    
    PortfolioValue = np.average(monthvalue, axis=0)
    PortfolioDailyGains = PortfolioValue[1:] / PortfolioValue[:-1]
    FinalValue = np.average(monthvalue[:, -1])
    
    # Calculate Sharpe ratio with error handling
    try:
        daily_returns = PortfolioDailyGains
        annual_return = gmean(daily_returns)**252 - 1.0
        annual_volatility = np.std(daily_returns) * np.sqrt(252)
        PortfolioSharpe = annual_return / annual_volatility if annual_volatility > 0 else 0.0
    except Exception as e:
        print(f" ... Error calculating Sharpe ratio: {e}")
        PortfolioSharpe = 0.0
    
    print(f" ... Realization {iter_num} complete: Final Value = {FinalValue:,.0f}, Sharpe = {PortfolioSharpe:.3f}")

    # Return results dictionary
    results = {
        'iter': iter_num,
        'finalValue': FinalValue,
        'sharpeRatio': PortfolioSharpe,
        'monthvalue': monthvalue,
        'signal2D': signal2D,
        'numberStocks': numberStocks,
        'monthgainlossweight': monthgainlossweight,
        'parameters': {
            'monthsToHold': monthsToHold,
            'numberStocksTraded': numberStocksTraded,
            'LongPeriod': LongPeriod,
            'stddevThreshold': stddevThreshold,
            'MA1': MA1,
            'MA2': MA2,
            'MA2offset': MA2offset,
            'lowPct': lowPct,
            'hiPct': hiPct,
            'uptrendSignalMethod': uptrendSignalMethod
        }
    }
    
    return results

#---------------------------------------------

def random_triangle(low=0.0, mid=0.5, high=1.0, size=1):
    uni = np.random.uniform(low, high, size)
    tri = np.random.triangular(low, mid, high, size)
    if size == 1:
        return ((uni + tri) / 2.0)[0]
    else:
        return ((uni + tri) / 2.0)

# initialize interactive plotting
#plt.ion()

##
##  Import list of symbols to process.
##

# read list of symbols from disk.
symbols_path = "/Users/donaldpg/pyTAAA_data/SP500"
filename = os.path.join(symbols_path, 'symbols', 'SP500_Symbols.txt')                   # plotmax = 1.e10, runnum = 902

json_fn = '/Users/donaldpg/pyTAAA_data/sp500_pine/pytaaa_sp500_pine.json'
params = get_json_params(json_fn,verbose=True)

##
## Get quotes for each symbol in list
## process dates.
## Clean up quotes.
## Make a plot showing all symbols in list
##

randomtrials = 250
# randomtrials = 3
# randomtrials = 51

firstTradePrintDate = (2005,1,1)

#firstdate=(1991,1,1)
#firstdate=(2003,1,1)
#lastdate=(2012,8,31)
#lastdate=(2012,9,15)
#adjClose, symbols, datearray = arrayFromQuotesForList(filename, firstdate, lastdate)

## update quotes from list of symbols
# (symbols_directory, symbols_file) = os.path.split( filename )
# basename, extension = os.path.splitext( symbols_file )

symbols_file = params["symbols_file"]
symbols_directory = os.path.split(symbols_file)[0]

print(" symbols_directory = ", symbols_directory)
print(" symbols_file = ", symbols_file)
# print("symbols_directory, symbols.file = ", symbols_directory, symbols_file)
#UpdateHDF5( symbols_directory, symbols_file )
###############################################################################################
###  UpdateHDF5( symbols_directory, symbols_file )  ### assume hdf is already up to date
adjClose, symbols, datearray, _, _ = loadQuotes_fromHDF(symbols_file, json_fn)
firstdate = datearray[0]


# Clean up missing values in input quotes
#  - infill interior NaN values using nearest good values to linearly interpolate
#  - copy first valid quote from valid date to all earlier positions
#  - copy last valid quote from valid date to all later positions
for ii in range(adjClose.shape[0]):
    adjClose[ii,:] = interpolate(adjClose[ii,:])
    adjClose[ii,:] = cleantobeginning(adjClose[ii,:])
    adjClose[ii,:] = cleantoend(adjClose[ii,:])


# find index of firstTradePrintDate
firstTradePrintDateFound = False
firstTradePrintDateTest = datetime.date(firstTradePrintDate[0], firstTradePrintDate[1], firstTradePrintDate[2])
for ii in range(len(datearray)):
    if datearray[ii] >= firstTradePrintDateTest:
        firstTradePrintDateFound = True
        firstTradePrintDateIndex = ii
        break
print("index of first date for printing trades = ", firstTradePrintDateIndex, firstTradePrintDate, datearray[firstTradePrintDateIndex])

import os
basename = os.path.split( filename )[-1]
print("basename = ", basename)

# set up to write monte carlo results to disk.
if basename == "symbols.txt" :
    runnum = 'run2501a'
    plotmax = 1.e5     # maximum value for plot (figure 3)
    holdMonths = [1,2,3,4,6,12]
elif basename == "Naz100_Symbols.txt" :
    runnum = 'run250a'
    runnum = 'run250b'
    plotmax = 1.e10     # maximum value for plot (figure 3)
    holdMonths = [1,1,1,1,1,1,1,1,2,2,3,4,6,12]
elif basename == "biglist.txt" :
    runnum = 'run2503'
    plotmax = 1.e9     # maximum value for plot (figure 3)
    holdMonths = [1,2,3,4,6,12]
elif basename == "ProvidentFundSymbols.txt" :
    runnum = 'run2504'
    plotmax = 1.e7     # maximum value for plot (figure 3)
    holdMonths = [4,6,12]
elif basename == "sp500_symbols.txt" :
    runnum = 'run2505'
    plotmax = 1.e8     # maximum value for plot (figure 3)
    holdMonths = [1,2,3,4,6,12]
elif basename == "cmg_symbols.txt" :
    runnum = 'run2507'
    plotmax = 1.e7     # maximum value for plot (figure 3)
    holdMonths = [3,4,6,12]
else :
    runnum = 'run2506'
    plotmax = 1.e9     # maximum value for plot (figure 3)
    holdMonths = [1,2,3,4,6,12]

if firstdate == (2003,1,1):
    runnum=runnum+"short"
    plotmax /= 100
    plotmax = max(plotmax,100000)
elif firstdate == (2007,1,1):
    runnum=runnum+"vshort"
    plotmax /= 250
    plotmax = max(plotmax,100000)

"""
plt.figure(1)
plt.grid()
plt.title('fund closing prices')
for ii in range(adjClose.shape[0]):
    plt.plot(datearray,adjClose[ii,:])
"""

print(" security values check: ",adjClose[np.isnan(adjClose)].shape)

# ########################################################################
# # take inverse of quotes for declines
# ########################################################################
# for iCompany in range( adjClose.shape[0] ):
#     tempQuotes = adjClose[iCompany,:]
#     tempQuotes[ np.isnan(tempQuotes) ] = 1.0
#     index = np.argmax(np.clip(np.abs(tempQuotes[::-1]-1),0,1e-8)) - 1
#     if index == -1:
#         lastquote = adjClose[iCompany,-1]
#         lastquote = lastquote ** 2
#         ##adjClose[iCompany,:] = lastquote / adjClose[iCompany,:]
#     else:
#         lastQuoteIndex = -index-1
#         lastquote = adjClose[iCompany,lastQuoteIndex]
#         print("\nlast quote index and quote for ", symbols[iCompany],lastQuoteIndex,adjClose[iCompany,lastQuoteIndex])
#         lastquote = lastquote ** 2
#         ##adjClose[iCompany,:] = lastquote / adjClose[iCompany,:]
#         adjClose[iCompany,lastQuoteIndex:] = adjClose[iCompany,lastQuoteIndex-1]
#         print(adjClose[iCompany,lastQuoteIndex-3:])

# for iCompany in range( adjClose.shape[0] ):
#     lastquote = adjClose[iCompany,-1]
#     if ~np.isnan(adjClose[iCompany,-1]):
#         lastquote = lastquote ** 2
#         adjClose[iCompany,:] = lastquote / adjClose[iCompany,:]
#     else:
#         adjClose[iCompany,:] *= 0.
#         #adjClose[iCompany,:] += 10.

########################################################################


gainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
activeCount = np.zeros(adjClose.shape[1],dtype=float)

numberSharesCalc = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
gainloss[:,1:] = adjClose[:,1:] / adjClose[:,:-1]
gainloss[np.isnan(gainloss)]=1.
value = 10000. * np.cumprod(gainloss,axis=1)   ### used bigger constant for inverse of quotes
BuyHoldFinalValue = np.average(value,axis=0)[-1]

print(" gainloss check: ",gainloss[np.isnan(gainloss)].shape)
print(" value check: ",value[np.isnan(value)].shape)
lastEmptyPriceIndex = np.zeros(adjClose.shape[0],dtype=int)
firstTrailingEmptyPriceIndex = np.zeros(adjClose.shape[0],dtype=int)

for ii in range(adjClose.shape[0]):
    # take care of special case where constant share price is inserted at beginning of series
    index = np.argmax(np.clip(np.abs(gainloss[ii,:]-1),0,1e-8)) - 1
    print("fist valid price and date = ",symbols[ii]," ",index," ",datearray[index])
    lastEmptyPriceIndex[ii] = index
    activeCount[lastEmptyPriceIndex[ii]+1:] += 1

for ii in range(adjClose.shape[0]):
    # take care of special case where no quote exists at end of series
    tempQuotes = adjClose[ii,:]
    tempQuotes[ np.isnan(tempQuotes) ] = 1.0
    index = np.argmax(np.clip(np.abs(tempQuotes[::-1]-1),0,1e-8)) - 1
    if index != -1:
        firstTrailingEmptyPriceIndex[ii] = -index
        print("first trailing invalid price: index and date = ",symbols[ii]," ",firstTrailingEmptyPriceIndex[ii]," ",datearray[index])
        activeCount[firstTrailingEmptyPriceIndex[ii]:] -= 1

"""
plt.figure(29)
plt.grid()
plt.title('fund monthly gains & losses')
"""

dateForFilename = str(datearray[-1].year)+"-"+str(datearray[-1].month)+"-"+str(datearray[-1].day)
# outfilename = os.path.join("pngs","Naz100-tripleHMAs_montecarlo_"+str(dateForFilename)+"_"+str(runnum)+".csv")
outfiledir = "/Users/donaldpg/pyTAAA_data/sp500_pine/data_store/pngs"
outfilename = os.path.join(
    outfiledir,
    "sp500_pine_montecarlo_"+str(dateForFilename)+"_"+str(runnum)+".csv"
)


column_text = "run,trial, \
              Number stocks,\
              monthsToHold,  \
              LongPeriod,  \
              MA1,  \
              MA2,  \
              MA3,  \
              volatility min,volatility max, \
              Portfolio Final Value,\
              stddevThreshold, \
              sma2factor, \
              rank Threshold (%), \
              sma_filt_val, \
              Portfolio std,Portfolio Sharpe,\
              begin date for recent performance,\
              Portfolio Ann Gain - recent,\
              Portfolio Sharpe - recent,\
              B&H Ann Gain - recent,B&H Sharpe - recent,\
              Sharpe 15 Yr,\
              Sharpe 10 Yr,\
              Sharpe 5 Yr,\
              Sharpe 3 Yr,\
              Sharpe 2 Yr,\
              Sharpe 1 Yr,\
              Return 15 Yr,\
              Return 10 Yr,\
              Return 5 Yr,\
              Return 3 Yr,\
              Return 2 Yr,\
              Return 1 Yr,\
              CAGR 15 Yr,\
              CAGR 10 Yr,\
              CAGR 5 Yr,\
              CAGR 3 Yr,\
              CAGR 2 Yr,\
              CAGR 1 Yr,\
              B&H CAGR 15 Yr,\
              B&H CAGR 10 Yr,\
              B&H CAGR 5 Yr,\
              B&H CAGR 3 Yr,\
              B&H CAGR 2 Yr,\
              B&H CAGR 1 Yr,\
              Avg Drawdown 15 Yr, \
              Avg Drawdown 10 Yr, \
              Avg Drawdown 5 Yr, \
              Avg Drawdown 3 Yr, \
              Avg Drawdown 2 Yr, \
              Avg Drawdown 1 Yr, \
              beatBuyHoldTest,\
              beatBuyHoldTest2,\
              \n"
for i in range(50):
    column_text = column_text.replace(", ", ",")

with open(outfilename,"w") as OUTFILE:
    OUTFILE.write(column_text)


FinalTradedPortfolioValue = np.zeros(randomtrials,dtype=float)
PortfolioReturn = np.zeros(randomtrials,dtype=float)
PortfolioSharpe = np.zeros(randomtrials,dtype=float)
MaxPortfolioValue = np.zeros(adjClose.shape[1],dtype=float)
MaxBuyHoldPortfolioValue = np.zeros(adjClose.shape[1],dtype=float)
periodForSignal = np.zeros(randomtrials,dtype=float)
numberStocksUpTrending = np.zeros( (randomtrials,adjClose.shape[1]), dtype=float)
numberStocksUpTrendingNearHigh = np.zeros( adjClose.shape[1], dtype=float)
numberStocksUpTrendingBeatBuyHold = np.zeros( adjClose.shape[1], dtype=float)

LP_montecarlo = np.zeros(randomtrials,dtype=float)
MA1_montecarlo = np.zeros(randomtrials,dtype=float)
MA2_montecarlo = np.zeros(randomtrials,dtype=float)
MA2offset_montecarlo = np.zeros(randomtrials,dtype=float)
numberStocksTraded_montecarlo = np.zeros(randomtrials,dtype=float)
monthsToHold_montecarlo = np.zeros(randomtrials,dtype=float)
riskDownside_min_montecarlo = np.zeros(randomtrials,dtype=float)
riskDownside_max_montecarlo = np.zeros(randomtrials,dtype=float)
sma2factor_montecarlo = np.zeros(randomtrials,dtype=float)
rankThresholdPct_montecarlo = np.zeros(randomtrials,dtype=float)

for iter in range(randomtrials):

    if iter%1==0:
        print("")
        print("")
        print(" random trial:  ",iter)

    #############################################################################
    # Generate Monte Carlo parameters for this realization
    #############################################################################
    
    # Generate random parameters
    LongPeriod_random = int(random.uniform(55,280)+.5) // 2
    stddevThreshold = random.uniform(3.97*0.8, 3.97*1.2)
    # MA1 = int(random.uniform(35,250)+.5) // 2
    # MA2 = int(random.uniform(7,30)+.5) // 2
    # # MA2offset = int(random.uniform(.6,5)+.5)
    # MA2offset = int(
    #     random_triangle(
    #         low=(MA1-MA2)//25,
    #         mid=(MA1-MA2)//20,
    #         high=(MA1-MA2)//15,
    #         size=1
    #     )
    # )
    numberStocksTraded = int(random.uniform(1.9,8.9)+.5) // 2
    monthsToHold = choice(holdMonths)
    
    # Add percentile channel parameters for optimization
    lowPct = random.uniform(10.0, 30.0)   # Lower percentile threshold
    hiPct = random.uniform(70.0, 90.0)    # Upper percentile threshold
    
    print("")
    print("months to hold = ",holdMonths,monthsToHold)
    print("")

    riskDownside_min = random.triangular(.2,.25,.3)
    riskDownside_max = random.triangular(3.5,4.25,5)
    sma2factor = random.triangular(.85,.91,.999)
    rankThresholdPct = int(random.triangular(0,2,25)) / 100.
    sma_filt_val = random.uniform(.01, .025)
    
    runs_fraction = 4
    LongPeriod = LongPeriod_random

    # Handle different parameter sets based on iteration
    if iter >= randomtrials / runs_fraction :
        print("\n\n\n")
        print("*********************************\nUsing pyTAAA parameters .....\n")
        numberStocksTraded = 6
        monthsToHold = 1
        LongPeriod = 412
        stddevThreshold = 8.495
        MA1 = 264
        MA2 = 22
        MA3 = 26
        sma2factor = 3.495
        rankThresholdPct = .3210
        riskDownside_min = 0.855876
        riskDownside_max = 16.9086
        sma_filt_val = .02988
        paramNumberToVary = choice([0,1,2,3,4,5,6,7,8,9,10,11])

        # Parameter variation logic
        if paramNumberToVary == 0 :
            numberStocksTraded += choice([-1,0,1])
        if paramNumberToVary == 1 :
            for kk in range(15):
                temp = choice(holdMonths)
                if temp != monthsToHold:
                    monthsToHold = temp
                    break
        if paramNumberToVary == 2 :
            LongPeriod = int(LongPeriod * np.around(random.uniform(-.01*LongPeriod, .01*LongPeriod)))
        if paramNumberToVary == 3 :
            MA1 = int(MA1 * np.around(random.uniform(-.01*MA1, .01*MA1)))
        if paramNumberToVary == 4 :
            MA2 = int(MA2 * np.around(random.uniform(-.01*MA2, .01*MA2)))
        if paramNumberToVary == 5 :
            MA2offset = choice([1,2,3])
        if paramNumberToVary == 6 :
            sma2factor = sma2factor * np.around(random.uniform(-.01*sma2factor, .01*sma2factor),-3)
        if paramNumberToVary == 7 :
            rankThresholdPct = rankThresholdPct * np.around(random.uniform(-.01*rankThresholdPct, .01*rankThresholdPct),-2)
        if paramNumberToVary == 8 :
            riskDownside_min = riskDownside_min * np.around(random.uniform(-.01*riskDownside_min, .01*riskDownside_min),-3)
        if paramNumberToVary == 9 :
            riskDownside_max = riskDownside_max * np.around(random.uniform(-.01*riskDownside_max, .01*riskDownside_max),-3)
        if paramNumberToVary == 10 :
            stddevThreshold = stddevThreshold * random.uniform(0.8, 1.2)
        if paramNumberToVary == 11 :
            sma_filt_val = sma_filt_val * random.uniform(0.8, 1.2)

    if iter < randomtrials / runs_fraction:
        paramNumberToVary = -999
        print("\n\n\n")
        print("*********************************\nUsing pyTAAA parameters .....\n")

        # Use triangular distributions for better parameter exploration
        lo_factor, hi_factor = 0.65, 1.5
        lo_facto, hi_factor2 = 0.8, 1.25
        numberStocksTraded = choice([5,6,6,7,7,8,8])
        monthsToHold = choice([1,1,1,1,1,1,1,1,1,2])
        LongPeriod = int(random_triangle(low=190, mid=370, high=550, size=1))
        stddevThreshold = random_triangle(low=5.0, mid=7.50, high=10., size=1)
        MA1 = int(random_triangle(low=75, mid=151, high=300, size=1))
        MA2 = int(random_triangle(low=10, mid=20, high=50, size=1))
        # MA2offset = choice([1,1,1,2,2,3,4,5,6,7,8,9,10])
        MA2offset = int(
            random_triangle(
                low=(MA1-MA2)//20,
                mid=(MA1-MA2)//15,
                high=(MA1-MA2)//10,
                size=1
            )
        )
        MA3 = int( 22 * random.uniform(lo_factor, hi_factor) )
        print(" ... initial MA1, MA2, MA3 = " + str(MA1) + ", " + str(MA2) + ", " + str(MA3))
        sma2factor = random_triangle(low=01.65, mid=2.5, high=2.75, size=1)
        rankThresholdPct = random_triangle(low=0.14, mid=0.20, high=.26, size=1)
        riskDownside_min = random_triangle(low=0.50, mid=0.70, high=0.90, size=1)
        riskDownside_max = random_triangle(low=8.0, mid=10.0, high=13.0, size=1)
        sma_filt_val = random_triangle(low=0.010, mid=0.015, high=.0225, size=1)

    if iter == randomtrials-1 :
        print("\n\n\n")
        print("*********************************\nUsing pyTAAApi linux edition parameters .....\n")
        numberStocksTraded = 7
        monthsToHold = 1
        LongPeriod = 455
        stddevThreshold = 6.12
        MA1 = 197
        MA2 = 19
        MA3 = 21
        sma2factor = 1.46
        rankThresholdPct = .132
        riskDownside_min = 0.5
        riskDownside_max = 7.4

    # Ensure valid parameter ranges
    MA2 = max(MA2, 3)
    MA1 = max(MA1, MA2+1)
    MA3 = MA2 + MA2offset

    print(" ... MA1, MA2, MA3 = " + str(MA1) + ", " + str(MA2) + ", " + str(MA3))

    #############################################################################
    # Create realization parameters dictionary
    #############################################################################
    
    realization_params = {
        'monthsToHold': monthsToHold,
        'numberStocksTraded': numberStocksTraded,
        'LongPeriod': LongPeriod,
        'stddevThreshold': stddevThreshold,
        'MA1': MA1,
        'MA2': MA2,
        'MA3': MA3,
        'MA2offset': MA2offset,
        'sma2factor': sma2factor,
        'rankThresholdPct': rankThresholdPct,
        'riskDownside_min': riskDownside_min,
        'riskDownside_max': riskDownside_max,
        'lowPct': lowPct,
        'hiPct': hiPct,
        'uptrendSignalMethod': 'percentileChannels',
        'sma_filt_val': sma_filt_val
    }

    #############################################################################
    # Run single Monte Carlo realization using temporary JSON
    #############################################################################
    
    try:
        print(f" ... Running realization {iter} with temporary JSON parameters")
        
        # Run the single realization using our new function
        results = run_single_monte_carlo_realization(
            json_fn,
            realization_params,
            iter,
            adjClose,
            symbols,
            datearray,
            gainloss,
            value,
            activeCount,
            holdMonths,
            verbose=(iter <= 2)  # Verbose output for first few iterations
        )
        
        # Extract results from the function
        monthvalue = results['monthvalue']
        signal2D = results['signal2D'] 
        numberStocks = results['numberStocks']
        monthgainlossweight = results['monthgainlossweight']
        
        # Create monthgainloss for the rest of the code to use
        monthgainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
        monthgainloss[:,LongPeriod:] = adjClose[:,LongPeriod:] / adjClose[:,:-LongPeriod]
        monthgainloss[np.isnan(monthgainloss)]=1.
        
    except Exception as e:
        print(f"Error in realization {iter}: {str(e)}")
        print("Using fallback values to continue execution")
        
        # Create fallback values to prevent further errors
        monthvalue = value.copy()
        signal2D = np.zeros_like(adjClose)
        numberStocks = np.zeros(adjClose.shape[1])
        monthgainlossweight = np.zeros_like(adjClose)
        
        # Create monthgainloss for the rest of the code to use
        monthgainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
        monthgainloss[:,LongPeriod:] = adjClose[:,LongPeriod:] / adjClose[:,:-LongPeriod]
        monthgainloss[np.isnan(monthgainloss)]=1.
        
        # Continue with next iteration if this one failed
        continue

    #############################################################################
    # Continue with existing analysis and statistics code
    #############################################################################
    
    # ...existing code...

    ########################################################################
    ### gather statistics on number of uptrending stocks
    ########################################################################

    numberStocksUpTrending[iter,:] = numberStocks
    numberStocksUpTrendingMedian = np.median(numberStocksUpTrending[:iter,:],axis=0)
    numberStocksUpTrendingMean   = np.mean(numberStocksUpTrending[:iter,:],axis=0)

    index = 3780
    if monthvalue.shape[1] < 3780: index = monthvalue.shape[1]

    PortfolioValue = np.average(monthvalue,axis=0)
    PortfolioDailyGains = PortfolioValue[1:] / PortfolioValue[:-1]
    Sharpe15Yr = ( gmean(PortfolioDailyGains[-index:])**252 -1. ) / ( np.std(PortfolioDailyGains[-index:])*np.sqrt(252) )
    Sharpe10Yr = ( gmean(PortfolioDailyGains[-2520:])**252 -1. ) / ( np.std(PortfolioDailyGains[-2520:])*np.sqrt(252) )
    Sharpe5Yr = ( gmean(PortfolioDailyGains[-1260:])**252 -1. ) / ( np.std(PortfolioDailyGains[-1260:])*np.sqrt(252) )
    Sharpe3Yr = ( gmean(PortfolioDailyGains[-756:])**252 -1. ) / ( np.std(PortfolioDailyGains[-756:])*np.sqrt(252) )
    Sharpe2Yr = ( gmean(PortfolioDailyGains[-504:])**252 -1. ) / ( np.std(PortfolioDailyGains[-504:])*np.sqrt(252) )
    Sharpe1Yr = ( gmean(PortfolioDailyGains[-252:])**252 -1. ) / ( np.std(PortfolioDailyGains[-252:])*np.sqrt(252) )
    PortfolioSharpe[iter] = ( gmean(PortfolioDailyGains)**252 -1. ) / ( np.std(PortfolioDailyGains)*np.sqrt(252) )

    print("15 year : ",index,PortfolioValue[-1], PortfolioValue[-index],datearray[-index])

    Return15Yr = (PortfolioValue[-1] / PortfolioValue[-index])**(252./index)
    Return10Yr = (PortfolioValue[-1] / PortfolioValue[-2520])**(1/10.)
    Return5Yr = (PortfolioValue[-1] / PortfolioValue[-1260])**(1/5.)
    Return3Yr = (PortfolioValue[-1] / PortfolioValue[-756])**(1/3.)
    Return2Yr = (PortfolioValue[-1] / PortfolioValue[-504])**(1/2.)
    Return1Yr = (PortfolioValue[-1] / PortfolioValue[-252])
    PortfolioReturn[iter] = gmean(PortfolioDailyGains)**252 -1.

    #############################################################################
    # Calculate CAGR for Trading System Portfolio
    #############################################################################
    print(" ... Calculating CAGR for trading system portfolio")
    
    CAGR15Yr = calculate_cagr(PortfolioValue[-1], PortfolioValue[-index], index)
    CAGR10Yr = calculate_cagr(PortfolioValue[-1], PortfolioValue[-2520], 2520)
    CAGR5Yr = calculate_cagr(PortfolioValue[-1], PortfolioValue[-1260], 1260)
    CAGR3Yr = calculate_cagr(PortfolioValue[-1], PortfolioValue[-756], 756)
    CAGR2Yr = calculate_cagr(PortfolioValue[-1], PortfolioValue[-504], 504)
    CAGR1Yr = calculate_cagr(PortfolioValue[-1], PortfolioValue[-252], 252)
    
    print(f" ... Trading System CAGR: 15Y={CAGR15Yr:.1%}, 10Y={CAGR10Yr:.1%}, 5Y={CAGR5Yr:.1%}")
    print(f" ... Trading System CAGR: 3Y={CAGR3Yr:.1%}, 2Y={CAGR2Yr:.1%}, 1Y={CAGR1Yr:.1%}")

    MaxPortfolioValue *= 0.
    for jj in range(PortfolioValue.shape[0]):
        MaxPortfolioValue[jj] = max(MaxPortfolioValue[jj-1],PortfolioValue[jj])
    PortfolioDrawdown = PortfolioValue / MaxPortfolioValue - 1.
    Drawdown15Yr = np.mean(PortfolioDrawdown[-index:])
    Drawdown10Yr = np.mean(PortfolioDrawdown[-2520:])
    Drawdown5Yr = np.mean(PortfolioDrawdown[-1260:])
    Drawdown3Yr = np.mean(PortfolioDrawdown[-756:])
    Drawdown2Yr = np.mean(PortfolioDrawdown[-504:])
    Drawdown1Yr = np.mean(PortfolioDrawdown[-252:])

    if iter == 0:
        BuyHoldPortfolioValue = np.mean(value,axis=0)
        BuyHoldDailyGains = BuyHoldPortfolioValue[1:] / BuyHoldPortfolioValue[:-1]
        BuyHoldSharpe15Yr = ( gmean(BuyHoldDailyGains[-index:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-index:])*np.sqrt(252) )
        BuyHoldSharpe10Yr = ( gmean(BuyHoldDailyGains[-2520:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-2520:])*np.sqrt(252) )
        BuyHoldSharpe5Yr  = ( gmean(BuyHoldDailyGains[-1126:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-1260:])*np.sqrt(252) )
        BuyHoldSharpe3Yr  = ( gmean(BuyHoldDailyGains[-756:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-756:])*np.sqrt(252) )
        BuyHoldSharpe2Yr  = ( gmean(BuyHoldDailyGains[-504:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-504:])*np.sqrt(252) )
        BuyHoldSharpe1Yr  = ( gmean(BuyHoldDailyGains[-252:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-252:])*np.sqrt(252) )
        BuyHoldReturn15Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-index])**(252./index)
        BuyHoldReturn10Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-2520])**(1/10.)
        BuyHoldReturn5Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-1260])**(1/5.)
        BuyHoldReturn3Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-756])**(1/3.)
        BuyHoldReturn2Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-504])**(1/2.)
        BuyHoldReturn1Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-252])
        
        #############################################################################
        # Calculate CAGR for Buy & Hold Portfolio (once at iter==0)
        #############################################################################
        print(" ... Calculating CAGR for Buy & Hold portfolio")
        
        BuyHoldCAGR15Yr = calculate_cagr(BuyHoldPortfolioValue[-1], BuyHoldPortfolioValue[-index], index)
        BuyHoldCAGR10Yr = calculate_cagr(BuyHoldPortfolioValue[-1], BuyHoldPortfolioValue[-2520], 2520)
        BuyHoldCAGR5Yr = calculate_cagr(BuyHoldPortfolioValue[-1], BuyHoldPortfolioValue[-1260], 1260)
        BuyHoldCAGR3Yr = calculate_cagr(BuyHoldPortfolioValue[-1], BuyHoldPortfolioValue[-756], 756)
        BuyHoldCAGR2Yr = calculate_cagr(BuyHoldPortfolioValue[-1], BuyHoldPortfolioValue[-504], 504)
        BuyHoldCAGR1Yr = calculate_cagr(BuyHoldPortfolioValue[-1], BuyHoldPortfolioValue[-252], 252)
        
        print(f" ... Buy & Hold CAGR: 15Y={BuyHoldCAGR15Yr:.1%}, 10Y={BuyHoldCAGR10Yr:.1%}, 5Y={BuyHoldCAGR5Yr:.1%}")
        print(f" ... Buy & Hold CAGR: 3Y={BuyHoldCAGR3Yr:.1%}, 2Y={BuyHoldCAGR2Yr:.1%}, 1Y={BuyHoldCAGR1Yr:.1%}")
        for jj in range(BuyHoldPortfolioValue.shape[0]):
            MaxBuyHoldPortfolioValue[jj] = max(MaxBuyHoldPortfolioValue[jj-1],BuyHoldPortfolioValue[jj])

        BuyHoldPortfolioDrawdown = BuyHoldPortfolioValue / MaxBuyHoldPortfolioValue - 1.
        BuyHoldDrawdown15Yr = np.mean(BuyHoldPortfolioDrawdown[-index:])
        BuyHoldDrawdown10Yr = np.mean(BuyHoldPortfolioDrawdown[-2520:])
        BuyHoldDrawdown5Yr = np.mean(BuyHoldPortfolioDrawdown[-1260:])
        BuyHoldDrawdown3Yr = np.mean(BuyHoldPortfolioDrawdown[-756:])
        BuyHoldDrawdown2Yr = np.mean(BuyHoldPortfolioDrawdown[-504:])
        BuyHoldDrawdown1Yr = np.mean(BuyHoldPortfolioDrawdown[-252:])


    print("")
    print("")
    print("Sharpe15Yr, BuyHoldSharpe15Yr = ", Sharpe15Yr, BuyHoldSharpe15Yr)
    print("Sharpe10Yr, BuyHoldSharpe10Yr = ", Sharpe10Yr, BuyHoldSharpe10Yr)
    print("Sharpe5Yr, BuyHoldSharpe5Yr =   ", Sharpe5Yr, BuyHoldSharpe5Yr)
    print("Sharpe3Yr, BuyHoldSharpe3Yr =   ", Sharpe3Yr, BuyHoldSharpe3Yr)
    print("Sharpe2Yr, BuyHoldSharpe2Yr =   ", Sharpe2Yr, BuyHoldSharpe2Yr)
    print("Sharpe1Yr, BuyHoldSharpe1Yr =   ", Sharpe1Yr, BuyHoldSharpe1Yr)
    print("Return15Yr, BuyHoldReturn15Yr = ", Return15Yr, BuyHoldReturn15Yr)
    print("Return10Yr, BuyHoldReturn10Yr = ", Return10Yr, BuyHoldReturn10Yr)
    print("Return5Yr, BuyHoldReturn5Yr =   ", Return5Yr, BuyHoldReturn5Yr)
    print("Return3Yr, BuyHoldReturn3Yr =   ", Return3Yr, BuyHoldReturn3Yr)
    print("Return2Yr, BuyHoldReturn2Yr =   ", Return2Yr, BuyHoldReturn2Yr)
    print("Return1Yr, BuyHoldReturn1Yr =   ", Return1Yr, BuyHoldReturn1Yr)
    print("Drawdown15Yr, BuyHoldDrawdown15Yr = ", Drawdown15Yr, BuyHoldDrawdown15Yr)
    print("Drawdown10Yr, BuyHoldDrawdown10Yr = ", Drawdown10Yr, BuyHoldDrawdown10Yr)
    print("Drawdown5Yr, BuyHoldDrawdown5Yr =   ", Drawdown5Yr, BuyHoldDrawdown5Yr)
    print("Drawdown3Yr, BuyHoldDrawdown3Yr =   ", Drawdown3Yr, BuyHoldDrawdown3Yr)
    print("Drawdown2Yr, BuyHoldDrawdown2Yr =   ", Drawdown2Yr, BuyHoldDrawdown2Yr)
    print("Drawdown1Yr, BuyHoldDrawdown1Yr =   ", Drawdown1Yr, BuyHoldDrawdown1Yr)

    if iter == 0:
        beatBuyHoldCount = 0
        beatBuyHold2Count = 0
    beatBuyHoldTest = ( (Sharpe15Yr-BuyHoldSharpe15Yr)/15. + \
                        (Sharpe10Yr-BuyHoldSharpe10Yr)/10. + \
                        (Sharpe5Yr-BuyHoldSharpe5Yr)/5. + \
                        (Sharpe3Yr-BuyHoldSharpe3Yr)/3. + \
                        (Sharpe2Yr-BuyHoldSharpe2Yr)/2. + \
                        (Sharpe1Yr-BuyHoldSharpe1Yr)/1. ) / (1/15. + 1/10.+1/5.+1/3.+1/2.+1)
    if beatBuyHoldTest > 0. :
        #print "found monte carlo trial that beats BuyHold..."
        #print "shape of numberStocksUpTrendingBeatBuyHold = ",numberStocksUpTrendingBeatBuyHold.shape
        #print "mean of numberStocksUpTrendingBeatBuyHold values = ",np.mean(numberStocksUpTrendingBeatBuyHold)
        beatBuyHoldCount += 1
        #numberStocksUpTrendingBeatBuyHold = (numberStocksUpTrendingBeatBuyHold * (beatBuyHoldCount -1) + numberStocks) / beatBuyHoldCount

    beatBuyHoldTest2 = 0
    if Return15Yr > BuyHoldReturn15Yr: beatBuyHoldTest2 += 1
    if Return10Yr > BuyHoldReturn10Yr: beatBuyHoldTest2 += 1
    if Return5Yr  > BuyHoldReturn5Yr:  beatBuyHoldTest2 += 1
    if Return3Yr  > BuyHoldReturn3Yr:  beatBuyHoldTest2 += 1.5
    if Return2Yr  > BuyHoldReturn2Yr:  beatBuyHoldTest2 += 2
    if Return1Yr  > BuyHoldReturn1Yr:  beatBuyHoldTest2 += 2.5
    if Return15Yr > 0: beatBuyHoldTest2 += 1
    if Return10Yr > 0: beatBuyHoldTest2 += 1
    if Return5Yr  > 0: beatBuyHoldTest2 += 1
    if Return3Yr  > 0: beatBuyHoldTest2 += 1.5
    if Return2Yr  > 0: beatBuyHoldTest2 += 2
    if Return1Yr  > 0: beatBuyHoldTest2 += 2.5
    if Drawdown15Yr > BuyHoldDrawdown15Yr: beatBuyHoldTest2 += 1
    if Drawdown10Yr > BuyHoldDrawdown10Yr: beatBuyHoldTest2 += 1
    if Drawdown5Yr  > BuyHoldDrawdown5Yr:  beatBuyHoldTest2 += 1
    if Drawdown3Yr  > BuyHoldDrawdown3Yr:  beatBuyHoldTest2 += 1.5
    if Drawdown2Yr  > BuyHoldDrawdown2Yr:  beatBuyHoldTest2 += 2
    if Drawdown1Yr  > BuyHoldDrawdown1Yr:  beatBuyHoldTest2 += 2.5
    # make it a ratio ranging from 0 to 1
    beatBuyHoldTest2 /= 27

    if beatBuyHoldTest2 > .60 :
        print("found monte carlo trial that beats BuyHold (test2)...")
        print("shape of numberStocksUpTrendingBeatBuyHold = ",numberStocksUpTrendingBeatBuyHold.shape)
        print("mean of numberStocksUpTrendingBeatBuyHold values = ",np.mean(numberStocksUpTrendingBeatBuyHold))
        beatBuyHold2Count += 1
        numberStocksUpTrendingBeatBuyHold = (numberStocksUpTrendingBeatBuyHold * (beatBuyHold2Count -1) + numberStocks) / beatBuyHold2Count

    '''
    ####################################################################
    ###
    ### calculate running mean for number of up-trending (inverse) stocks
    ###
    uptrendOffset = random.triangular(0.,0.,.25)
    uptrendDaysMedian = int(random.triangular(100,700,1000)+.5)
    #numberStocksUpTrendingThreshold = MoveMedian( numberStocksUpTrendingBeatBuyHold, uptrendDaysMedian ) + uptrendOffset*activeCount
    numberStocksUpTrendingThreshold = SMA( numberStocksUpTrendingBeatBuyHold, uptrendDaysMedian ) + uptrendOffset*activeCount

    minDays = int(random.uniform(75,252)+.5)
    maxDays = int(random.uniform(3*252,6*252)+.5)
    incDays = (maxDays-minDays)/6.-1
    numberStocksUpTrendingThreshold, _ = dpgchannel( numberStocksUpTrendingBeatBuyHold, minDays, maxDays, incDays ) + uptrendOffset*activeCount
    ####################################################################
    '''

    print("beatBuyHoldTest = ", beatBuyHoldTest, beatBuyHoldTest2)
    print("countof trials that BeatBuyHold  = ", beatBuyHoldCount)
    print("countof trials that BeatBuyHold2 = ", beatBuyHold2Count)
    print("")
    print("")

    from scipy.stats import scoreatpercentile
    if iter > 1:
        for jj in range(adjClose.shape[1]):
            numberStocksUpTrendingNearHigh[jj]   = scoreatpercentile(numberStocksUpTrending[:iter,jj], 90)

    if iter == 0:
        from time import sleep
        for i in range(len(symbols)):
            plt.clf()
            plt.grid()
            ##plot(datearray,signal2D[i,:]*np.mean(adjClose[i,:])*numberStocksTraded/2)
            plot_vals = adjClose[i,:] * 10000. / adjClose[i,0]
            plt.plot(datearray, plot_vals)
            aaa = signal2D[i,:]
            NaNcount = aaa[np.isnan(aaa)].shape[0]
            plt.title("signal2D before figure3 ... "+symbols[i]+"   "+str(NaNcount))
            plt.draw()
            #time.sleep(.2)

    print(" ")
    print("The B&H portfolio final value is: ","{:,}".format(int(BuyHoldFinalValue)))
    print(" ")
    print("Monthly re-balance based on ",LongPeriod, "days of recent performance.")
    print("The portfolio final value is: ","{:,}".format(int(np.average(monthvalue,axis=0)[-1])))
    print(" ")
    print("Today's top ranking choices are: ")
    last_symbols_text = []
    for ii in range(len(symbols)):
        if monthgainlossweight[ii,-1] > 0:
            # print symbols[ii]
            print(datearray[-1], symbols[ii],monthgainlossweight[ii,-1])
            last_symbols_text.append( symbols[ii] )


    ########################################################################
    ### compute traded value of stock for each month (using varying percent invested)
    ########################################################################

    ###
    ### gather sum of all quotes minus SMA
    ###
    QminusSMADays = int(random.uniform(252,5*252)+.5)
    QminusSMAFactor = random.triangular(.88,.91,.999)

    # re-calc constant monthPctInvested
    uptrendConst = random.uniform(0.45,0.75)
    PctInvestSlope = random.triangular(2.,5.,7.)
    PctInvestIntercept = -random.triangular(-.05,0.0,.07)
    maxPctInvested = choice([1.0,1.0,1.0,1.2,1.33,1.5])

    if iter == randomtrials-1 :
        print("\n\n\n")
        print("*********************************\nUsing pyTAAA parameters .....\n")
        QminusSMADays = 355
        QminusSMAFactor = .90
        PctInvestSlope = 5.45
        PctInvestIntercept = -.01
        maxPctInvested = 1.25

    # adjCloseSMA = QminusSMAFactor * SMA_2D( adjClose, QminusSMADays )  # MA1 is longest
    adjCloseSMA = QminusSMAFactor * SMA_filtered_2D( adjClose, QminusSMADays )  # MA1 is longest

    QminusSMA = np.zeros( adjClose.shape[1], 'float' )
    for ii in range( 1,adjClose.shape[1] ):
        ajdClose_date = adjClose[:,ii]
        ajdClose_prevdate = adjClose[:,ii-1]
        adjCloseSMA_date = adjCloseSMA[:,ii]
        ajdClose_date_edit = ajdClose_date[ajdClose_date != ajdClose_prevdate]
        adjCloseSMA_date_edit = adjCloseSMA_date[ajdClose_date != ajdClose_prevdate]
        QminusSMA[ii] = np.sum( ajdClose_date_edit - adjCloseSMA_date_edit  ) / np.sum( adjCloseSMA_date_edit )
    #QminusSMA = np.sum( adjClose - sma2, axis = 0  ) / np.sum( sma2, axis = 0 )


    '''
    numberStocksUpTrendingThreshold *= 0.
    numberStocksUpTrendingThreshold += uptrendConst*activeCount
    '''

    # Calculate percent to invest in inverse strategy. Set first 2 years to zero to build history
    #aa = ( QminusSMA + PctInvestIntercept ) * PctInvestSlope
    #monthPctInvested = QminusSMA.copy()
    #monthPctInvested = np.clip( aa, 0., 1. )

    ###
    ### do MACD on monthPctInvested
    ###
    monthPctInvestedDaysMAshort = int(random.uniform(5,35)+.5)

    monthPctInvestedSMAshort = SMA( QminusSMA, monthPctInvestedDaysMAshort )
    monthPctInvestedDaysMAlong = int(random.uniform(3,100)+.5) + monthPctInvestedDaysMAshort
    monthPctInvestedSMAlong = SMA( QminusSMA, monthPctInvestedDaysMAlong )
    #monthPctInvestedMACD = monthPctInvested - monthPctInvestedSMA
    monthPctInvestedMACD = monthPctInvestedSMAshort - monthPctInvestedSMAlong

    #aa = ( monthPctInvestedMACD + PctInvestIntercept ) * PctInvestSlope
    aa = ( QminusSMA + PctInvestIntercept ) * PctInvestSlope
    monthPctInvested = np.clip( aa, 0., maxPctInvested )


    print(" NaNs in value = ", (value[np.isnan(value)]).shape)

    monthvalueVariablePctInvest = value.copy()
    print(" 1 - monthvalueVariablePctInvest check: ",monthvalueVariablePctInvest[np.isnan(monthvalueVariablePctInvest)].shape)
    for ii in np.arange(1,monthgainloss.shape[1]):
        if (datearray[ii].month != datearray[ii-1].month) and ( (datearray[ii].month - 1)%monthsToHold == 0):
            valuesum=np.sum(monthvalueVariablePctInvest[:,ii-1])
            for jj in range(value.shape[0]):
                monthvalueVariablePctInvest[jj,ii] = monthgainlossweight[jj,ii]*valuesum*(1.0+(gainloss[jj,ii]-1.0)*monthPctInvested[ii])   # re-balance using weights (that sum to 1.0)
        else:
            monthPctInvested[ii] = monthPctInvested[ii-1]
            for jj in range(value.shape[0]):
                monthvalueVariablePctInvest[jj,ii] = monthvalueVariablePctInvest[jj,ii-1]*(1.0+(gainloss[jj,ii]-1.0)*monthPctInvested[ii])

    ########################################################################
    ### gather statistics on number of uptrending stocks (using varying percent invested)
    ########################################################################

    index = 3780
    if monthvalueVariablePctInvest.shape[1] < 3780: index = monthvalueVariablePctInvest.shape[1]

    PortfolioValue = np.average(monthvalueVariablePctInvest,axis=0)
    PortfolioDailyGains = PortfolioValue[1:] / PortfolioValue[:-1]
    VarPctSharpe15Yr = ( gmean(PortfolioDailyGains[-index:])**252 -1. ) / ( np.std(PortfolioDailyGains[-index:])*np.sqrt(252) )
    VarPctSharpe10Yr = ( gmean(PortfolioDailyGains[-2520:])**252 -1. ) / ( np.std(PortfolioDailyGains[-2520:])*np.sqrt(252) )
    VarPctSharpe5Yr = ( gmean(PortfolioDailyGains[-1260:])**252 -1. ) / ( np.std(PortfolioDailyGains[-1260:])*np.sqrt(252) )
    VarPctSharpe3Yr = ( gmean(PortfolioDailyGains[-756:])**252 -1. ) / ( np.std(PortfolioDailyGains[-756:])*np.sqrt(252) )
    VarPctSharpe2Yr = ( gmean(PortfolioDailyGains[-504:])**252 -1. ) / ( np.std(PortfolioDailyGains[-504:])*np.sqrt(252) )
    VarPctSharpe1Yr = ( gmean(PortfolioDailyGains[-252:])**252 -1. ) / ( np.std(PortfolioDailyGains[-252:])*np.sqrt(252) )
    #VarPctPortfolioSharpe[iter] = ( gmean(PortfolioDailyGains)**252 -1. ) / ( np.std(PortfolioDailyGains)*np.sqrt(252) )

    print("15 year : ",index,PortfolioValue[-1], PortfolioValue[-index],datearray[-index])

    VarPctReturn15Yr = (PortfolioValue[-1] / PortfolioValue[-index])**(252./index)
    VarPctReturn10Yr = (PortfolioValue[-1] / PortfolioValue[-2520])**(1/10.)
    VarPctReturn5Yr = (PortfolioValue[-1] / PortfolioValue[-1260])**(1/5.)
    VarPctReturn3Yr = (PortfolioValue[-1] / PortfolioValue[-756])**(1/3.)
    VarPctReturn2Yr = (PortfolioValue[-1] / PortfolioValue[-504])**(1/2.)
    VarPctReturn1Yr = (PortfolioValue[-1] / PortfolioValue[-252])
    #VarPctPortfolioReturn[iter] = gmean(PortfolioDailyGains)**252 -1.

    MaxPortfolioValue *= 0.
    for jj in range(PortfolioValue.shape[0]):
        MaxPortfolioValue[jj] = max(MaxPortfolioValue[jj-1],PortfolioValue[jj])
    PortfolioDrawdown = PortfolioValue / MaxPortfolioValue - 1.
    VarPctDrawdown15Yr = np.mean(PortfolioDrawdown[-index:])
    VarPctDrawdown10Yr = np.mean(PortfolioDrawdown[-2520:])
    VarPctDrawdown5Yr = np.mean(PortfolioDrawdown[-1260:])
    VarPctDrawdown3Yr = np.mean(PortfolioDrawdown[-756:])
    VarPctDrawdown2Yr = np.mean(PortfolioDrawdown[-504:])
    VarPctDrawdown1Yr = np.mean(PortfolioDrawdown[-252:])


    beatBuyHoldTestVarPct = ( (VarPctSharpe15Yr-BuyHoldSharpe15Yr)/15. + \
                            (VarPctSharpe10Yr-BuyHoldSharpe10Yr)/10. + \
                            (VarPctSharpe5Yr-BuyHoldSharpe5Yr)/5. + \
                            (VarPctSharpe3Yr-BuyHoldSharpe3Yr)/3. + \
                            (VarPctSharpe2Yr-BuyHoldSharpe2Yr)/2. + \
                            (VarPctSharpe1Yr-BuyHoldSharpe1Yr)/1. ) / (1/15. + 1/10.+1/5.+1/3.+1/2.+1)


    beatBuyHoldTest2VarPct = 0
    if VarPctReturn15Yr > BuyHoldReturn15Yr: beatBuyHoldTest2VarPct += 1
    if VarPctReturn10Yr > BuyHoldReturn10Yr: beatBuyHoldTest2VarPct += 1
    if VarPctReturn5Yr  > BuyHoldReturn5Yr:  beatBuyHoldTest2VarPct += 1
    if VarPctReturn3Yr  > BuyHoldReturn3Yr:  beatBuyHoldTest2VarPct += 1.5
    if VarPctReturn2Yr  > BuyHoldReturn2Yr:  beatBuyHoldTest2VarPct += 2
    if VarPctReturn1Yr  > BuyHoldReturn1Yr:  beatBuyHoldTest2VarPct += 2.5
    if VarPctReturn15Yr > 0: beatBuyHoldTest2VarPct += 1
    if VarPctReturn10Yr > 0: beatBuyHoldTest2VarPct += 1
    if VarPctReturn5Yr  > 0: beatBuyHoldTest2VarPct += 1
    if VarPctReturn3Yr  > 0: beatBuyHoldTest2VarPct += 1.5
    if VarPctReturn2Yr  > 0: beatBuyHoldTest2VarPct += 2
    if VarPctReturn1Yr  > 0: beatBuyHoldTest2VarPct += 2.5
    if VarPctDrawdown15Yr > BuyHoldDrawdown15Yr: beatBuyHoldTest2VarPct += 1
    if VarPctDrawdown10Yr > BuyHoldDrawdown10Yr: beatBuyHoldTest2VarPct += 1
    if VarPctDrawdown5Yr  > BuyHoldDrawdown5Yr:  beatBuyHoldTest2VarPct += 1
    if VarPctDrawdown3Yr  > BuyHoldDrawdown3Yr:  beatBuyHoldTest2VarPct += 1.5
    if VarPctDrawdown2Yr  > BuyHoldDrawdown2Yr:  beatBuyHoldTest2VarPct += 2
    if VarPctDrawdown1Yr  > BuyHoldDrawdown1Yr:  beatBuyHoldTest2VarPct += 2.5
    # make it a ratio ranging from 0 to 1
    beatBuyHoldTest2VarPct /= 27


    '''
    ########################################################################
    ### plot results
    ########################################################################


    matplotlib.rcParams['figure.edgecolor'] = 'grey'
    rc('savefig',edgecolor = 'grey')
    fig = figure(1)
    clf()
    #fig.set_edgecolor((.8,.8,.8))
    subplotsize = gridspec.GridSpec(3,1,height_ratios=[4,1,1])
    subplot(subplotsize[0])
    grid()
    ##
    ## make plot of all stocks' individual prices
    ##
    if iter == 0:
        yscale('log')
        ylim([1000,max(10000,plotmax)])
        ymin, ymax = emath.log10(1e3), emath.log10(max(10000,plotmax))
        bin_width = (ymax - ymin) / 50
        y_bins = np.arange(ymin, ymax+.0000001, bin_width)
        AllStocksHistogram = np.ones((len(y_bins)-1, len(datearray),3))
        HH = np.zeros((len(y_bins)-1, len(datearray)))
        mm = np.zeros(len(datearray))
        xlocs = []
        xlabels = []
        for i in range(1,len(datearray)):
            ValueOnDate = value[:,i]
            if ValueOnDate[ValueOnDate == 10000].shape[0] > 1:
                ValueOnDate[ValueOnDate == 10000] = 0.
                ValueOnDate[np.argmin(ValueOnDate)] = 10000.
            h, _ = np.histogram(np.log10(ValueOnDate), bins=y_bins, density=True)
            # reverse so big numbers become small(and print out black)
            h = 1. - h
            # set range to [.5,1.]
            h /= 2.
            h += .5
            HH[:,i] = h
            mm[i] = np.median(value[-1,:])
            if datearray[i].year != datearray[i-1].year:
                print(" inside histogram evaluation for date = ", datearray[i])
                xlocs.append(i)
                xlabels.append(str(datearray[i].year))
        #AllStocksHistogram[:,:,2] = ndimage.gaussian_filter(HH, sigma=1)
        AllStocksHistogram[:,:,2] = HH
        AllStocksHistogram[:,:,1] = AllStocksHistogram[:,:,2]
        #AllStocksHistogram = ndimage.gaussian_filter(HH, sigma=1)
        #AllStocksHistogram = np.clip(AllStocksHistogram,AllStocksHistogram.max()*.0,1)
        #AllStocksHistogram = AllStocksHistogram ** 1.2
        AllStocksHistogram = np.clip(AllStocksHistogram,AllStocksHistogram.max()*.05,1)
        AllStocksHistogram /= AllStocksHistogram.max()

    plt.imshow(AllStocksHistogram, cmap='Reds_r', aspect='auto', origin='lower', extent=(0, len(datearray), 10**ymin, 10**ymax))
    plt.grid()
    ##
    ## cumulate final values for grayscale histogram overlay
    ##
    if iter == 0:
        MonteCarloPortfolioValues = np.zeros( (randomtrials, len(datearray) ), float )
    MonteCarloPortfolioValues[iter,:] = np.average(monthvalue,axis=0)

    ##
    ## cumulate final values for grayscale histogram overlay
    ##
    if iter == 0:
        MonteCarloPortfolioValues = np.zeros( (randomtrials, len(datearray) ), float )
    MonteCarloPortfolioValues[iter,:] = np.average(monthvalue,axis=0)


    if iter > 9 and iter%10 == 0:
        yscale('log')
        ylim([1000,max(10000,plotmax)])
        ymin, ymax = emath.log10(1e3), emath.log10(max(10000,plotmax))
        bin_width = (ymax - ymin) / 50
        y_bins = np.arange(ymin, ymax+.0000001, bin_width)
        H = np.zeros((len(y_bins)-1, len(datearray)))
        m = np.zeros(len(datearray))
        #hb = np.zeros((len(y_bins)-1, len(datearray),4))
        hb = np.zeros((len(y_bins)-1, len(datearray),3))
        for i in range(1,len(datearray)):
            h, _ = np.histogram(np.log10(MonteCarloPortfolioValues[:iter,i]), bins=y_bins, density=True)
            # reverse so big numbers become small(and print out black)
            h = 1. - h
            # set range to [.5,1.]
            h = np.clip( h, .05, 1. )
            h /= 2.
            h += .5
            H[:,i] = h
            m[i] = np.median(value[-1,:])
            if datearray[i].year != datearray[i-1].year:
                print(" inside histogram evaluation for date = ", datearray[i])
        #hb[:,:,2] = ndimage.gaussian_filter(H, sigma=1)
        hb[:,:,0] = H
        hb[:,:,1] = H
        hb[:,:,2] = H
        #hb[:,:,3] *= 0.
        #hb[:,:,3] += 0.5
        hb = .5 * AllStocksHistogram + .5 * hb

    if iter > 10  :
        yscale('log')
        plt.imshow(hb, cmap='gray', aspect='auto', origin='lower', extent=(0, len(datearray), 10**ymin, 10**ymax))

    yscale('log')
    #ymin, ymax = emath.log10(1e3), emath.log10(max(10000,plotmax))
    #ylim([ymin,max(10000,plotmax)])
    #plot( (np.log10(np.average(monthvalue,axis=0))-(ymin))/(ymax-ymin)*(10**ymax), lw=3, c='k' )
    # plot( np.average(monthvalue,axis=0), lw=3, c='k' )
    plot(datearray, np.average(monthvalue,axis=0), lw=3, c='k' )

    grid()
    draw()
    '''
    ########################################################################
    ### plot recent results
    ########################################################################

    ##
    ## cumulate final values for grayscale histogram overlay
    ##
    if iter == 0:
        MonteCarloPortfolioValues = np.zeros( (randomtrials, len(datearray) ), float )
    MonteCarloPortfolioValues[iter,:] = np.average(monthvalue,axis=0)
    scale_factor = 10000.0 / MonteCarloPortfolioValues[iter,0]
    MonteCarloPortfolioValues[iter,:] *= scale_factor

    plt.rcParams['figure.edgecolor'] = 'grey'
    plt.rc('savefig',edgecolor = 'grey')
    plt.close(1)
    fig = plt.figure(1, figsize=(10, 10*1080/1920))
    plt.clf()
    subplotsize = gridspec.GridSpec(2,1,height_ratios=[5,1])
    plt.subplot(subplotsize[0])
    plt.grid()
    ##
    ## make plot of all stocks' individual prices
    ##
    #yscale('log')
    #ylim([1000,max(100000,plotmax)])
    plt.ylim([7000,plotmax])
    #ymin, ymax = emath.log10(1e3), emath.log10(max(10000,plotmax))
    ymin, ymax = 7000,plotmax
    #bin_width = (np.log10(ymax) - np.log10(ymin)) / 50
    #y_bins = np.arange(np.log10(ymin), np.log10(ymax)+.0000001, bin_width)
    n_bins = 150
    #bin_width = (ymax - ymin) / n_bins
    #y_bins = np.arange(ymin, ymax+.0000001, bin_width)
    y_bins = np.linspace(ymin, ymax, n_bins)

    AllStocksHistogram = np.ones((len(y_bins)-1, len(datearray),3))
    HH = np.zeros((len(y_bins)-1, len(datearray)))
    mm = np.zeros(len(datearray))
    xlocs = []
    xlabels = []
    for i in range(1,len(datearray)):
        #ValueOnDate = np.log10(value[:,i])
        ValueOnDate = value[:,i]
        '''
        if ValueOnDate[ValueOnDate == 10000].shape[0] > 1:
            ValueOnDate[ValueOnDate == 10000] = 0.
            ValueOnDate[np.argmin(ValueOnDate)] = 10000.
        '''
        #h, _ = np.histogram(np.log10(ValueOnDate), bins=y_bins, density=True)
        h, _ = np.histogram(ValueOnDate, bins=y_bins, density=True)
        h /= h.sum()
        # reverse so big numbers become small(and print out black)
        h = 1. - h
        # set range to [.5,1.]
        h /= 2.
        h += .5
        #print "idatearray[i],h min,mean,max = ", h.min(),h.mean(),h.max()
        HH[:,i] = h
        mm[i] = np.median(value[-1,:])
        if datearray[i].year != datearray[i-1].year:
            print(" inside histogram evaluation for date = ", datearray[i])
            xlocs.append(i)
            xlabels.append(str(datearray[i].year))
    HH -= np.percentile(HH.flatten(),2)
    HH /= HH.max()
    HH = np.clip( HH, 0., 1. )
    #print "HH min,mean,max = ", HH.min(),HH.mean(),HH.max()
    AllStocksHistogram[:,:,2] = HH
    AllStocksHistogram[:,:,1] = AllStocksHistogram[:,:,2]
    AllStocksHistogram = np.clip(AllStocksHistogram,AllStocksHistogram.max()*.05,1)
    AllStocksHistogram /= AllStocksHistogram.max()

    #plt.imshow(AllStocksHistogram, cmap='Reds_r', aspect='auto', origin='lower', extent=(0, len(datearray), 10**ymin, 10**ymax))
    ##plt.imshow(AllStocksHistogram, cmap='Reds_r', aspect='auto', origin='lower', extent=(0, len(datearray), ymin, ymax))
    plt.grid()
    ##
    ## cumulate final values for grayscale histogram overlay
    ##
    '''
    if iter == 0:
        MonteCarloPortfolioValues = np.zeros( (randomtrials, len(datearray) ), float )
    MonteCarloPortfolioValues[iter,:] = np.average(monthvalue,axis=0)
    '''

    ##
    ## cumulate final values for grayscale histogram overlay
    ##

    '''
    #yscale('log')
    #ylim([1000,max(10000,plotmax)])
    plt.ylim([7000,plotmax])
    #ymin, ymax = emath.log10(1e3), emath.log10(max(10000,plotmax))
    ymin, ymax = 7000,plotmax
    #bin_width = (np.log10(ymax) - np.log10(ymin)) / 50
    #y_bins = np.arange(np.log10(ymin), np.log10(ymax)+.0000001, bin_width)
    n_bins = 150
    #bin_width = (np.log10(ymax) - np.log10(ymin)) / n_bins
    #y_bins = np.arange(np.log10(ymin), np.log10(ymax)+.0000001, bin_width)
    y_bins = np.linspace(np.log10(ymin), np.log10(ymax), n_bins)

    H = np.zeros((len(y_bins)-1, len(datearray)))
    m = np.zeros(len(datearray))
    hb = np.zeros((len(y_bins)-1, len(datearray),3))

    ## TODO: remove the following lines after QC and de-bugging
    print("\n\n ... inside dailyBacktest_pctLong.py near line 338")
    print(" ... MonteCarloPortfolioValues[:,:].min() = "+str(MonteCarloPortfolioValues[:,:].min()))
    print(" ... MonteCarloPortfolioValues[:,:].max() = "+str(MonteCarloPortfolioValues[:,:].max()))
    print(" ... MonteCarloPortfolioValues[0,:] = "+str(MonteCarloPortfolioValues[0,:]))
    print(" ... y_bins[0], y_bins[-1] = "+str((y_bins[0], y_bins[-1])))

    for i in range(1,len(datearray)):
        #h, _ = np.histogram(np.log10(MonteCarloPortfolioValues[:,i]), bins=y_bins, density=True)
        h, _ = np.histogram(np.log10(MonteCarloPortfolioValues[:,i]), bins=y_bins, density=True)
        h /= h.sum()
        # reverse so big numbers become small(and print out black)
        h = 1. - h
        # set range to [.5,1.]
        h = np.clip( h, .05, 1. )
        h /= 2.
        h += .5
        H[:,i] = h
        m[i] = np.median(value[-1,:])
        if datearray[i].year != datearray[i-1].year:
            print(" inside histogram evaluation for date = ", datearray[i])
    H -= np.percentile(H.flatten(),2)
    H /= H.max()
    H = np.clip( H, 0., 1. )
    hb[:,:,0] = H
    hb[:,:,1] = H
    hb[:,:,2] = H
    hb = .5 * AllStocksHistogram + .5 * hb
    '''

    plt.ylim([7000,plotmax])
    #ymin, ymax = emath.log10(1e3), emath.log10(max(10000,plotmax))
    ymin, ymax = 7000,plotmax
    #bin_width = (np.log10(ymax) - np.log10(ymin)) / 50
    #y_bins = np.arange(np.log10(ymin), np.log10(ymax)+.0000001, bin_width)
    n_bins = 150
    #bin_width = (np.log10(ymax) - np.log10(ymin)) / n_bins
    #y_bins = np.arange(np.log10(ymin), np.log10(ymax)+.0000001, bin_width)
    #y_bins = np.linspace(np.log10(ymin), np.log10(ymax), n_bins)
    #bin_width = (ymax - ymin) / n_bins
    #y_bins = np.arange(ymin, ymax+.0000001, bin_width)
    y_bins = np.linspace(ymin, ymax, n_bins)

    H = np.zeros((len(y_bins)-1, len(datearray)))
    m = np.zeros(len(datearray))
    hb = np.zeros((len(y_bins)-1, len(datearray),3))

    ## TODO: remove the following lines after QC and de-bugging
    print("\n\n ... inside dailyBacktest_pctLong.py near line 338")
    print(" ... MonteCarloPortfolioValues[:,:].min() = "+str(MonteCarloPortfolioValues[:,:].min()))
    print(" ... MonteCarloPortfolioValues[:,:].max() = "+str(MonteCarloPortfolioValues[:,:].max()))
    print(" ... MonteCarloPortfolioValues[0,:] = "+str(MonteCarloPortfolioValues[0,:]))
    print(" ... y_bins[0], y_bins[-1] = "+str((y_bins[0], y_bins[-1])))

    for i in range(1,len(datearray)):
        ValueOnDate = MonteCarloPortfolioValues[:,i]
        #h, _ = np.histogram(np.log10(MonteCarloPortfolioValues[:,i]), bins=y_bins, density=True)
        #h, _ = np.histogram(np.log10(MonteCarloPortfolioValues[:,i]), bins=y_bins, density=True)
        h, _ = np.histogram(ValueOnDate, bins=y_bins, density=True)
        h /= h.sum()
        # reverse so big numbers become small(and print out black)
        h = 1. - h
        # set range to [.5,1.]
        h = np.clip( h, .05, 1. )
        h /= 2.
        h += .5
        H[:,i] = h
        m[i] = np.median(value[-1,:])
        if datearray[i].year != datearray[i-1].year:
            print(" inside histogram evaluation for date = ", datearray[i])
    H -= np.percentile(H.flatten(),2)
    H /= H.max()
    H = np.clip( H, 0., 1. )
    hb[:,:,0] = H
    hb[:,:,1] = H
    hb[:,:,2] = H
    hb = .5 * AllStocksHistogram + .5 * hb

    #yscale('log')
    #plt.imshow(hb, cmap='gray', aspect='auto', origin='lower', extent=(0, len(datearray), 10**ymin, 10**ymax))
    ##plt.imshow(hb, cmap='gray', aspect='auto', origin='lower', extent=(0, len(datearray), ymin, ymax))

    #yscale('log')
    plt.yscale('log')   ## TODO: check this
    scale_factor = 10000.0 / np.average(monthvalue,axis=0)
    scale_factor = 1.0
    plt.plot( np.average(monthvalue,axis=0) * scale_factor, lw=3, c='k' )
    plt.grid()
    plt.draw()

    ##
    ## continue
    ##
    FinalTradedPortfolioValue[iter] = np.average(monthvalue[:,-1])
    fFinalTradedPortfolioValue = "{:,}".format(int(FinalTradedPortfolioValue[iter]))
    PortfolioDailyGains = np.average(monthvalue,axis=0)[1:] / np.average(monthvalue,axis=0)[:-1]
    PortfolioSharpe[iter] = ( gmean(PortfolioDailyGains)**252 -1. ) / ( np.std(PortfolioDailyGains)*np.sqrt(252) )
    fPortfolioSharpe = format(PortfolioSharpe[iter],'5.2f')

    FinalTradedVarPctPortfolioValue = np.average(monthvalueVariablePctInvest[:,-1])
    fVFinalTradedPortfolioValue = "{:,}".format(int(FinalTradedVarPctPortfolioValue))
    PortfolioDailyGains = np.average(monthvalueVariablePctInvest,axis=0)[1:] / np.average(monthvalueVariablePctInvest,axis=0)[:-1]
    PortVarPctfolioSharpe = ( gmean(PortfolioDailyGains)**252 -1. ) / ( np.std(PortfolioDailyGains)*np.sqrt(252) )
    fVPortfolioSharpe = format(PortVarPctfolioSharpe,'5.2f')

    print("")
    print(" value 2 yrs ago, 1 yr ago, last = ",np.average(monthvalue[:,-504]),np.average(monthvalue[:,-252]),np.average(monthvalue[:,-1]))
    print(" one year gain, daily geom mean, stdev = ",np.average(monthvalue,axis=0)[-1] / np.average(monthvalue,axis=0)[-252],gmean(PortfolioDailyGains[-252:])**252 -1.,np.std(PortfolioDailyGains[-252:])*np.sqrt(252))
    print(" two year gain, daily geom mean, stdev = ",np.average(monthvalue,axis=0)[-1] / np.average(monthvalue,axis=0)[-504],gmean(PortfolioDailyGains[-504:])**252 -1.,np.std(PortfolioDailyGains[-504:])*np.sqrt(252))

    title_text = str(iter)+":  "+ \
                  str(int(numberStocksTraded))+"__"+   \
                  str(int(monthsToHold))+"__"+   \
                  str(int(LongPeriod))+"-"+   \
                  str(int(MA1))+"-"+   \
                  str(int(MA2))+"-"+   \
                  str(int(MA2+MA2offset))+"-"+   \
                  format(sma2factor,'5.3f')+"_"+   \
                  format(rankThresholdPct,'.1%')+"_"+   \
                  format(sma_filt_val,'6.5f')+"__"+   \
                  format(riskDownside_min,'6.3f')+"-"+  \
                  format(riskDownside_max,'6.3f')+"__"+   \
                  fFinalTradedPortfolioValue+'__'+   \
                  fPortfolioSharpe+'\n{'+   \
                  str(QminusSMADays)+"-"+   \
                  format(QminusSMAFactor,'6.2f')+"_-"+   \
                  format(PctInvestSlope,'6.2f')+"_"+   \
                  format(PctInvestIntercept,'6.2f')+"_"+   \
                  format(maxPctInvested,'4.2f')+"}__"+   \
                  fVFinalTradedPortfolioValue+'__'+   \
                  fVPortfolioSharpe

    plt.title( title_text, fontsize = 9 )
    fSharpe15Yr = format(Sharpe15Yr,'5.2f')
    fSharpe10Yr = format(Sharpe10Yr,'5.2f')
    fSharpe5Yr = format(Sharpe5Yr,'5.2f')
    fSharpe3Yr = format(Sharpe3Yr,'5.2f')
    fSharpe2Yr = format(Sharpe2Yr,'5.2f')
    fSharpe1Yr = format(Sharpe1Yr,'5.2f')
    fReturn15Yr = format(Return15Yr,'5.2f')
    fReturn10Yr = format(Return10Yr,'5.2f')
    fReturn5Yr = format(Return5Yr,'5.2f')
    fReturn3Yr = format(Return3Yr,'5.2f')
    fReturn2Yr = format(Return2Yr,'5.2f')
    fReturn1Yr = format(Return1Yr,'5.2f')
    fDrawdown15Yr = format(Drawdown15Yr,'.1%')
    fDrawdown10Yr = format(Drawdown10Yr,'.1%')
    fDrawdown5Yr = format(Drawdown5Yr,'.1%')
    fDrawdown3Yr = format(Drawdown3Yr,'.1%')
    fDrawdown2Yr = format(Drawdown2Yr,'.1%')
    fDrawdown1Yr = format(Drawdown1Yr,'.1%')
    
    #############################################################################
    # Format CAGR values for plot display with conditional toggle
    #############################################################################
    fCAGR15Yr = format(CAGR15Yr, '.1%')
    fCAGR10Yr = format(CAGR10Yr, '.1%')
    fCAGR5Yr = format(CAGR5Yr, '.1%')
    fCAGR3Yr = format(CAGR3Yr, '.1%')
    fCAGR2Yr = format(CAGR2Yr, '.1%')
    fCAGR1Yr = format(CAGR1Yr, '.1%')
    
    fBuyHoldCAGR15Yr = format(BuyHoldCAGR15Yr, '.1%')
    fBuyHoldCAGR10Yr = format(BuyHoldCAGR10Yr, '.1%')
    fBuyHoldCAGR5Yr = format(BuyHoldCAGR5Yr, '.1%')
    fBuyHoldCAGR3Yr = format(BuyHoldCAGR3Yr, '.1%')
    fBuyHoldCAGR2Yr = format(BuyHoldCAGR2Yr, '.1%')
    fBuyHoldCAGR1Yr = format(BuyHoldCAGR1Yr, '.1%')

    fVSharpe15Yr = format(VarPctSharpe15Yr,'5.2f')
    fVSharpe10Yr = format(VarPctSharpe10Yr,'5.2f')
    fVSharpe5Yr = format(VarPctSharpe5Yr,'5.2f')
    fVSharpe3Yr = format(VarPctSharpe3Yr,'5.2f')
    fVSharpe2Yr = format(VarPctSharpe2Yr,'5.2f')
    fVSharpe1Yr = format(VarPctSharpe1Yr,'5.2f')
    fVReturn15Yr = format(VarPctReturn15Yr,'5.2f')
    fVReturn10Yr = format(VarPctReturn10Yr,'5.2f')
    fVReturn5Yr = format(VarPctReturn5Yr,'5.2f')
    fVReturn3Yr = format(VarPctReturn3Yr,'5.2f')
    fVReturn2Yr = format(VarPctReturn2Yr,'5.2f')
    fVReturn1Yr = format(VarPctReturn1Yr,'5.2f')
    fVDrawdown15Yr = format(VarPctDrawdown15Yr,'.1%')
    fVDrawdown10Yr = format(VarPctDrawdown10Yr,'.1%')
    fVDrawdown5Yr = format(VarPctDrawdown5Yr,'.1%')
    fVDrawdown3Yr = format(VarPctDrawdown3Yr,'.1%')
    fVDrawdown2Yr = format(VarPctDrawdown2Yr,'.1%')
    fVDrawdown1Yr = format(VarPctDrawdown1Yr,'.1%')

    print(" one year sharpe = ",fSharpe1Yr)
    print("")
    # plotrange = np.log10(plotmax / 1000.)
    plotrange = np.log10(plotmax) - np.log10(7000.)
    plt.text( 50,10.**(np.log10(7000)+(.47*plotrange)), symbols_file, fontsize=8 )
    plt.text( 50,10.**(np.log10(7000)+(.05*plotrange)), "Backtested on "+datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"), fontsize=7.5 )

    #############################################################################
    # Conditional plot display logic for CAGR vs AvgProfit toggle
    #############################################################################
    if show_cagr_in_plot:
        # CAGR mode - display CAGR percentages
        header_text = 'Period Sharpe CAGR      Avg DD'
        display_15yr = fCAGR15Yr
        display_10yr = fCAGR10Yr 
        display_5yr = fCAGR5Yr
        display_3yr = fCAGR3Yr
        display_2yr = fCAGR2Yr
        display_1yr = fCAGR1Yr
        
        # Variable percentage display values (also CAGR mode)
        vdisplay_15yr = format(calculate_cagr(PortfolioValue[-1], PortfolioValue[-index], index), '.1%')
        vdisplay_10yr = format(calculate_cagr(PortfolioValue[-1], PortfolioValue[-2520], 2520), '.1%')
        vdisplay_5yr = format(calculate_cagr(PortfolioValue[-1], PortfolioValue[-1260], 1260), '.1%')
        vdisplay_3yr = format(calculate_cagr(PortfolioValue[-1], PortfolioValue[-756], 756), '.1%')
        vdisplay_2yr = format(calculate_cagr(PortfolioValue[-1], PortfolioValue[-504], 504), '.1%')
        vdisplay_1yr = format(calculate_cagr(PortfolioValue[-1], PortfolioValue[-252], 252), '.1%')
    else:
        # AvgProfit mode - display existing Return values as decimals
        header_text = 'Period Sharpe AvgProfit  Avg DD'
        display_15yr = fReturn15Yr
        display_10yr = fReturn10Yr
        display_5yr = fReturn5Yr
        display_3yr = fReturn3Yr
        display_2yr = fReturn2Yr
        display_1yr = fReturn1Yr
        
        # Variable percentage display values (AvgProfit mode)
        vdisplay_15yr = fVReturn15Yr
        vdisplay_10yr = fVReturn10Yr
        vdisplay_5yr = fVReturn5Yr
        vdisplay_3yr = fVReturn3Yr
        vdisplay_2yr = fVReturn2Yr
        vdisplay_1yr = fVReturn1Yr
    
    # Apply conditional display to plot tables
    plt.text(50,10.**(np.log10(7000)+(.95*plotrange)),header_text,fontsize=7.5)
    plt.text(50,10.**(np.log10(7000)+(.91*plotrange)),'15 Yr '+fSharpe15Yr+'  '+display_15yr+'  '+fDrawdown15Yr,fontsize=8)
    plt.text(50,10.**(np.log10(7000)+(.87*plotrange)),'10 Yr '+fSharpe10Yr+'  '+display_10yr+'  '+fDrawdown10Yr,fontsize=8)
    plt.text(50,10.**(np.log10(7000)+(.83*plotrange)),' 5 Yr  '+fSharpe5Yr+'  '+display_5yr+'  '+fDrawdown5Yr,fontsize=8)
    plt.text(50,10.**(np.log10(7000)+(.79*plotrange)),' 3 Yr  '+fSharpe3Yr+'  '+display_3yr+'  '+fDrawdown3Yr,fontsize=8)
    plt.text(50,10.**(np.log10(7000)+(.75*plotrange)),' 2 Yr  '+fSharpe2Yr+'  '+display_2yr+'  '+fDrawdown2Yr,fontsize=8)
    plt.text(50,10.**(np.log10(7000)+(.71*plotrange)),' 1 Yr  '+fSharpe1Yr+'  '+display_1yr+'  '+fDrawdown1Yr,fontsize=8)

    # Variable percentage table (blue table) with conditional display  
    plt.text(2250,10.**(np.log10(7000)+(.95*plotrange)),header_text,fontsize=7.5,color='b')
    plt.text(2250,10.**(np.log10(7000)+(.91*plotrange)),'15 Yr '+fVSharpe15Yr+'  '+vdisplay_15yr+'  '+fVDrawdown15Yr,fontsize=8,color='b')
    plt.text(2250,10.**(np.log10(7000)+(.87*plotrange)),'10 Yr '+fVSharpe10Yr+'  '+vdisplay_10yr+'  '+fVDrawdown10Yr,fontsize=8,color='b')
    plt.text(2250,10.**(np.log10(7000)+(.83*plotrange)),' 5 Yr  '+fVSharpe5Yr+'  '+vdisplay_5yr+'  '+fVDrawdown5Yr,fontsize=8,color='b')
    plt.text(2250,10.**(np.log10(7000)+(.79*plotrange)),' 3 Yr  '+fVSharpe3Yr+'  '+vdisplay_3yr+'  '+fVDrawdown3Yr,fontsize=8,color='b')
    plt.text(2250,10.**(np.log10(7000)+(.75*plotrange)),' 2 Yr  '+fVSharpe2Yr+'  '+vdisplay_2yr+'  '+fVDrawdown2Yr,fontsize=8,color='b')
    plt.text(2250,10.**(np.log10(7000)+(.71*plotrange)),' 1 Yr  '+fVSharpe1Yr+'  '+vdisplay_1yr+'  '+fVDrawdown1Yr,fontsize=8,color='b')

    if beatBuyHoldTest > 0. :
        plt.text(50,10.**(np.log10(7000)+(.65*plotrange)),format(beatBuyHoldTest2,'.2%')+'  beats BuyHold...')
    else:
        plt.text(50,10.**(np.log10(7000)+(.65*plotrange)),format(beatBuyHoldTest2,'.2%'))

    if beatBuyHoldTestVarPct > 0. :
        plt.text(50,10.**(np.log10(7000)+(.59*plotrange)),format(beatBuyHoldTest2VarPct,'.2%')+'  beats BuyHold...',color='b')
    else:
        plt.text(50,10.**(np.log10(7000)+(.59*plotrange)),format(beatBuyHoldTest2VarPct,'.2%'),color='b'
        )

    plt.text(50,10.**(np.log10(7000)+(.54*plotrange)),last_symbols_text,fontsize=8)

    #plot(datearray,BuyHoldPortfolioValue,lw=5,c='r')
    #plot(datearray,np.average(monthvalue,axis=0),lw=7,c='k')
    #plot(datearray[0],plotmax)
    # plot(BuyHoldPortfolioValue,lw=3,c='r')
    # plot(np.average(monthvalue,axis=0),lw=4,c='k')
    # plot(np.average(monthvalueVariablePctInvest,axis=0),lw=2,c='b')
    plt.plot(BuyHoldPortfolioValue,lw=3,c='r')
    plt.plot(np.average(monthvalue,axis=0),lw=4,c='k')
    plt.plot(np.average(monthvalueVariablePctInvest,axis=0),lw=2,c='b')
    # scale_factor1 = 10000.0 / BuyHoldPortfolioValue[0]
    # scale_factor2 = 10000.0 / np.average(monthvalue,axis=0)[0]
    # scale_factor3 = 10000.0 / np.average(monthvalueVariablePctInvest, axis=0)[0]
    # scale_factor1 = 1.0
    # scale_factor2 = 1.0
    # scale_factor3 = 1.0
    # plt.plot(datearray, BuyHoldPortfolioValue * scale_factor1, lw=3,c='r')
    # plt.plot(datearray, np.average(monthvalue,axis=0) * scale_factor2 ,lw=4,c='k')
    # plt.plot(datearray, np.average(monthvalueVariablePctInvest, axis=0) * scale_factor3,lw=2,c='b')

    ###plot(plotmax)
    # set up to use dates for labels
    plt.xlocs = []
    plt.xlabels = []
    for i in range(1,len(datearray)):
        if datearray[i].year != datearray[i-1].year:
            plt.xlocs.append(i)
            plt.xlabels.append(str(datearray[i].year))
    print("xlocs,xlabels = ", xlocs, xlabels)
    if len(xlocs) < 12 :
        plt.xticks(xlocs, xlabels)
    else:
        plt.xticks(xlocs[::2], xlabels[::2])
    # plt.xlim(0,len(datearray))
    # plt.subplot(subplotsize[1])
    # plt.grid()
    # ##ylim(0, value.shape[0])
    # plt.ylim(0, 1.2)
    # plt.plot(datearray,numberStocksUpTrendingMedian / activeCount,'g-',lw=1)
    # plt.plot(datearray,numberStocksUpTrendingNearHigh / activeCount,'b-',lw=1)
    # plt.plot(datearray,numberStocksUpTrendingBeatBuyHold / activeCount,'k-',lw=2)
    # plt.plot(datearray,numberStocks  / activeCount,'r-')

    # plt.subplot(subplotsize[2])
    plt.subplot(subplotsize[1])
    plt.grid()
    # plt.xlim(0,len(datearray))
    # set up to use dates for labels
    plt.xlocs = []
    plt.xlabels = []
    for i in range(1,len(datearray)):
        if datearray[i].year != datearray[i-1].year:
            plt.xlocs.append(i)
            plt.xlabels.append(str(datearray[i].year))
    print("xlocs,xlabels = ", xlocs, xlabels)
    if len(xlocs) < 12 :
        plt.xticks(xlocs, xlabels)
    else:
        plt.xticks(xlocs[::2], xlabels[::2])

    ##ylim(0, value.shape[0])
    #ylim(0, 1.2)
    #plot(datearray,monthPctInvestedMACD,'k-',lw=.8)
    # plt.plot(datearray, QminusSMA,'m-',lw=.8)
    # plt.plot(datearray, monthPctInvested,'r-',lw=.8)
    plt.plot(QminusSMA,'m-',lw=.8)
    plt.plot(monthPctInvested,'r-',lw=.8)
    #######text(datearray[50],5,last_symbols_text)
    plt.draw()
    # save figure to disk, but only if trades produce good results
    ###if beatBuyHoldTest2 > .50 and Return1Yr+Return2Yr > 2. and mean(Drawdown1Yr,Drawdown2Yr,Drawdown3Yr) > -0.12 :
    if 2>1:
        plot_fn = os.path.join(
            outfiledir,
            "Naz100-fSMAs_montecarlo_" + \
            str(dateForFilename) + "__" + \
            str(runnum) + "__" + \
            format(iter,'03d') + ".png"
        )
        # plot_fn = os.path.join(
        #     os.getcwd(),
        #     "pngs",
        #     "Naz100-tripleHMAs_montecarlo_"+str(dateForFilename)+"_"+format(iter,'03d')+".png"
        # )
        plt.savefig(plot_fn, format='png', edgecolor='gray' )
        #savefig("pngs\Naz100-tripleMAs_montecarlo_"+str(dateForFilename)+runnum+"_"+str(iter)+".png", format='png' )
    #plt.show()
    #time.sleep(1)

    ###
    ### save backtest portfolio values ( B&H and system )
    ###
    try:
        # Use plot_fn instead of undefined filepath
        filepath = os.path.join(
            outfiledir,
            "pyTAAAweb_fSMAbacktestPortfolioValue.params"
        )
        textmessage = ""
        for idate in range(len(BuyHoldPortfolioValue)):
            textmessage = textmessage + str(datearray[idate])+"  "+str(BuyHoldPortfolioValue[idate])+"  "+str(np.average(monthvalue[:,idate]))+"\n"
        with open( filepath, "w" ) as f:
            f.write(textmessage)
    except:
        pass


    ########################################################################
    ### compute some portfolio performance statistics and print summary
    ########################################################################

    print("final value for portfolio ", "{:,}".format(np.average(monthvalue[:,-1])))


    print("portfolio annualized gains : ", ( gmean(PortfolioDailyGains)**252 ))
    print("portfolio annualized StdDev : ", ( np.std(PortfolioDailyGains)*np.sqrt(252) ))
    print("portfolio sharpe ratio : ",PortfolioSharpe[iter])

    # Compute trading days back to target start date
    targetdate = datetime.date(2008,1,1)
    lag = int((datearray[-1] - targetdate).days*252/365.25)

    # Print some stats for B&H and trading from target date to end_date
    print("")
    print("")
    BHValue = np.average(value,axis=0)
    BHdailygains = np.concatenate( (np.array([0.]), BHValue[1:]/BHValue[:-1]), axis = 0 )
    BHsharpefromtargetdate = ( gmean(BHdailygains[-lag:])**252 -1. ) / ( np.std(BHdailygains[-lag:])*np.sqrt(252) )
    BHannualgainfromtargetdate = ( gmean(BHdailygains[-lag:])**252 )
    print("start date for recent performance measures: ",targetdate)
    print("BuyHold annualized gains & sharpe from target date:   ", BHannualgainfromtargetdate,BHsharpefromtargetdate)

    Portfoliosharpefromtargetdate = ( gmean(PortfolioDailyGains[-lag:])**252 -1. ) / ( np.std(PortfolioDailyGains[-lag:])*np.sqrt(252) )
    Portfolioannualgainfromtargetdate = ( gmean(PortfolioDailyGains[-lag:])**252 )
    print("portfolio annualized gains & sharpe from target date: ", Portfolioannualgainfromtargetdate,Portfoliosharpefromtargetdate)

    csv_text = runnum+","+str(iter)+","+    \
                  str(numberStocksTraded)+","+   \
                  str(monthsToHold)+","+  \
                  str(LongPeriod)+","+   \
                  str(MA1)+","+   \
                  str(MA2)+","+   \
                  str(MA2+MA2offset)+","+   \
                  str(riskDownside_min)+","+str(riskDownside_max)+","+   \
                  str(FinalTradedPortfolioValue[iter])+','+   \
                  format(stddevThreshold,'5.3f')+","+  \
                  format(sma2factor,'5.3f')+","+  \
                  format(rankThresholdPct,'.1%')+","+  \
                  format(sma_filt_val, '5.5f')+","+  \
                  str(np.std(PortfolioDailyGains)*np.sqrt(252))+','+   \
                  str(PortfolioSharpe[iter])+','+   \
                  str(targetdate)+','+   \
                  str(Portfolioannualgainfromtargetdate)+','+   \
                  str(Portfoliosharpefromtargetdate)+','+   \
                  str(BHannualgainfromtargetdate)+','+   \
                  str(BHsharpefromtargetdate)+","+   \
                  fSharpe15Yr+","+   \
                  fSharpe10Yr+","+   \
                  fSharpe5Yr+","+   \
                  fSharpe3Yr+","+   \
                  fSharpe2Yr+","+   \
                  fSharpe1Yr+","+   \
                  fReturn15Yr+","+   \
                  fReturn10Yr+","+   \
                  fReturn5Yr+","+   \
                  fReturn3Yr+","+   \
                  fReturn2Yr+","+   \
                  fReturn1Yr+","+   \
                  format(CAGR15Yr, '.4f')+","+   \
                  format(CAGR10Yr, '.4f')+","+   \
                  format(CAGR5Yr, '.4f')+","+   \
                  format(CAGR3Yr, '.4f')+","+   \
                  format(CAGR2Yr, '.4f')+","+   \
                  format(CAGR1Yr, '.4f')+","+   \
                  format(BuyHoldCAGR15Yr, '.4f')+","+   \
                  format(BuyHoldCAGR10Yr, '.4f')+","+   \
                  format(BuyHoldCAGR5Yr, '.4f')+","+   \
                  format(BuyHoldCAGR3Yr, '.4f')+","+   \
                  format(BuyHoldCAGR2Yr, '.4f')+","+   \
                  format(BuyHoldCAGR1Yr, '.4f')+","+   \
                  fDrawdown15Yr+","+   \
                  fDrawdown10Yr+","+   \
                  fDrawdown5Yr+","+   \
                  fDrawdown3Yr+","+   \
                  fDrawdown2Yr+","+   \
                  fDrawdown1Yr+","+   \
                  format(beatBuyHoldTest,'5.3f')+","+\
                  format(beatBuyHoldTest2,'.2%')+","+\
                  str(paramNumberToVary)+\
                  " \n"

    with open(outfilename,"a") as OUTFILE:
        OUTFILE.write(csv_text)

    periodForSignal[iter] = LongPeriod


    # create and update counter for holding period
    # with number of random trials choosing this symbol on last date times Sharpe ratio for trial in last year
    print("")
    print("")
    print("cumulative tally of holding periods for last date")
    if iter == 0:
        print("initializing cumulative talley of holding periods chosen for last date...")
        holdmonthscount = np.zeros(len(holdMonths),dtype=float)
    if beatBuyHoldTest > 0 :
        numdays1Yr = 252
        Sharpe1Yr = ( gmean(PortfolioDailyGains[-numdays1Yr:])**252 -1. ) / ( np.std(PortfolioDailyGains[-numdays1Yr:])*np.sqrt(252) )
        Sharpe2Yr = ( gmean(PortfolioDailyGains[-2*numdays1Yr:])**252 -1. ) / ( np.std(PortfolioDailyGains[-2*numdays1Yr:])*np.sqrt(252) )
        Sharpe3Yr = ( gmean(PortfolioDailyGains[-3*numdays1Yr:])**252 -1. ) / ( np.std(PortfolioDailyGains[-3*numdays1Yr:])*np.sqrt(252) )
        Sharpe5Yr = ( gmean(PortfolioDailyGains[-5*numdays1Yr:])**252 -1. ) / ( np.std(PortfolioDailyGains[-5*numdays1Yr:])*np.sqrt(252) )
        Sharpe10Yr = ( gmean(PortfolioDailyGains[-10*numdays1Yr:])**252 -1. ) / ( np.std(PortfolioDailyGains[-10*numdays1Yr:])*np.sqrt(252) )
        for ii in range(len(holdMonths)):
            if monthsToHold == holdMonths[ii]:
                #print symbols[ii],"  weight = ",max( 0., ( .666*Sharpe1Yr + .333*Sharpe2Yr ) * monthgainlossweight[ii,-1] )
                #symbolscount[ii] += max( 0., ( .666*Sharpe1Yr + .333*Sharpe2Yr ) * monthgainlossweight[ii,-1] )
                holdmonthscount[ii] += ( 1.0*Sharpe1Yr + 1./2*Sharpe2Yr + 1./3.*Sharpe3Yr + 1./5.*Sharpe5Yr + 1./10.*Sharpe10Yr ) * (1+2+3+5+10)
        bestchoicethreshold = 3. * np.median(holdmonthscount[holdmonthscount > 0.])
        holdmonthscountnorm = holdmonthscount*1.
        if holdmonthscountnorm[holdmonthscountnorm > 0].shape[0] > 0:
            holdmonthscountnorm -= holdmonthscountnorm[holdmonthscountnorm > 0].min()
            holdmonthscountnorm /= holdmonthscountnorm.max()
        holdmonthscountint = np.round(holdmonthscountnorm*40)
        holdmonthscountint[np.isnan(holdmonthscountint)] =0
        print("   . holdmonthscountint = " + str(holdmonthscountint))
        print("   . holdmonths = " + str(holdMonths))
        try:
            for ii in range(len(holdMonths)):
                if holdmonthscountint[ii] > 0:
                    tagnorm = "*"* int(holdmonthscountint[ii])
                    print(format(str(holdMonths[ii]),'7s')+   \
                          str(datearray[-1])+         \
                          format(holdmonthscount[ii],'7.2f'), tagnorm)
        except:
            pass


