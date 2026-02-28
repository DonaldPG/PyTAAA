"""
Core backtest execution functions extracted from PyTAAA_backtest_sp500_pine_refactored.py

This module contains the main backtest execution logic for Monte Carlo simulations,
providing clean separation from the legacy implementation.
"""

import os
import json
import numpy as np
import datetime
from typing import Dict, Tuple
from scipy.stats import gmean

# Local imports
from functions.ta.data_cleaning import interpolate, cleantobeginning, cleantoend
from functions.ta.moving_averages import SMA, hma, SMA_filtered_2D
from functions.ta.channels import percentileChannel_2D
from functions.ta.signal_generation import computeSignal2D
from functions.TAfunctions import sharpeWeightedRank_2D
from functions.GetParams import get_json_params, get_performance_store, get_webpage_store


class TradingConstants:
    """
    Trading day constants used for performance calculations.

    These represent standard trading day counts for various time periods,
    based on approximately 252 trading days per year.
    """

    TRADING_DAYS_PER_YEAR: int = 252
    TRADING_DAYS_1_YEAR: int = 252
    TRADING_DAYS_2_YEARS: int = 504
    TRADING_DAYS_3_YEARS: int = 756
    TRADING_DAYS_5_YEARS: int = 1260


def random_triangle(low=0.0, mid=0.5, high=1.0, size=1):
    """
    Generate random values using a triangular distribution mixed with uniform.
    
    Args:
        low: Lower bound
        mid: Mode (peak) of the distribution
        high: Upper bound
        size: Number of values to generate
        
    Returns:
        Single value if size=1, otherwise array of values
    """
    uni = np.random.uniform(low, high, size)
    tri = np.random.triangular(low, mid, high, size)
    if size == 1:
        return ((uni + tri) / 2.0)[0]
    else:
        return ((uni + tri) / 2.0)


def validate_backtest_parameters(params: Dict) -> Dict:
    """
    Validate and set defaults for backtest parameters.
    
    Args:
        params: Dictionary of parameters from JSON config
        
    Returns:
        Validated parameters dictionary with defaults applied
        
    Raises:
        ValueError: If critical parameters are missing or invalid
    """
    validated = params.copy()
    
    # Required parameters with validation
    required_params = {
        'monthsToHold': (int, 1, 12),
        'numberStocksTraded': (int, 1, 20),
        'LongPeriod': (int, 50, 1000),
        'stddevThreshold': (float, 0.1, 20.0),
        'MA1': (int, 10, 500),
        'MA2': (int, 5, 100),
        'rankThresholdPct': (float, 0.0, 1.0),
        'riskDownside_min': (float, 0.0, 1.0),
        'riskDownside_max': (float, 0.1, 20.0),
        'lowPct': (float, 0.0, 50.0),
        'hiPct': (float, 50.0, 100.0),
    }
    
    for param_name, (param_type, min_val, max_val) in required_params.items():
        value = params.get(param_name)
        if value is None:
            raise ValueError(f"Required parameter '{param_name}' is missing")
        
        try:
            # Convert to correct type
            if param_type == int:
                value = int(value)
            elif param_type == float:
                value = float(value)
            
            # Validate range
            if not (min_val <= value <= max_val):
                raise ValueError(f"Parameter '{param_name}' value {value} is outside valid range [{min_val}, {max_val}]")
            
            validated[param_name] = value
            
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid value for parameter '{param_name}': {e}")
    
    # Optional parameters with defaults
    defaults = {
        'uptrendSignalMethod': 'percentileChannels',
        'sma_filt_val': 0.02,
        'MA2factor': 0.91,
        'trade_cost': 0.0,
        'max_weight_factor': 3.0,
        'min_weight_factor': 0.3,
        'absolute_max_weight': 0.9,
        'apply_constraints': True,
        'stockList': 'SP500',  # Default to SP500 for this backtest script
        'enable_rolling_filter': True,  # Enable by default to catch interpolated data
        'window_size': 50,
    }
    
    for param_name, default_value in defaults.items():
        if param_name not in validated or validated[param_name] is None:
            validated[param_name] = default_value
    
    # Derived parameters
    validated['MA2offset'] = validated.get('MA2offset', validated.get('MA3', validated['MA2'] + 2) - validated['MA2'])
    validated['sma2factor'] = validated.get('sma2factor', validated.get('MA2factor', 0.91))
    
    # Validate derived parameters
    if validated['MA2offset'] < 0:
        raise ValueError("MA2offset cannot be negative")
    
    return validated


def create_temporary_json(base_json_fn, realization_params, iter_num, runnum=None):
    """
    Create a temporary JSON file for a single Monte Carlo realization.
    
    Uses standardized naming: {model_id}_backtest_montecarlo_{date}_{runnum}_{trial:03d}.json
    
    Args:
        base_json_fn: Path to base JSON configuration file
        realization_params: Dictionary with parameters for this realization
        iter_num: Iteration number for unique temp file naming (zero-padded to 3 digits)
        runnum: Run identifier string (if None, auto-generated from symbols file)
        
    Returns:
        Path to temporary JSON file in {model_base}/pytaaa_backtest/
    """
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
    
    # Create temporary file in pytaaa_backtest subdirectory
    # Extract performance_store to determine output location
    perf_store = get_performance_store(base_json_fn)
    model_base = os.path.dirname(perf_store)  # Go up one level from data_store
    temp_dir = os.path.join(model_base, "pytaaa_backtest")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate standardized filename: {model_name}_backtest_montecarlo_{datestamp}_{runnum}_{trial}.json
    webpage_path = get_webpage_store(base_json_fn)
    model_id = webpage_path.rstrip("/").split("/")[-2]  # Extract model identifier
    
    # Get date string
    today = datetime.date.today()
    date_str = f"{today.year}-{today.month}-{today.day}"
    
    # Determine runnum from parameter or symbols_file
    if runnum is None:
        symbols_file = updated_params.get("Valuation", {}).get("symbols_file", "")
        basename = os.path.basename(symbols_file)
        runnum_map = {
            "symbols.txt": "2501a",
            "Naz100_Symbols.txt": "250b",
            "biglist.txt": "2503",
            "ProvidentFundSymbols.txt": "2504",
            "sp500_symbols.txt": "2505",
            "cmg_symbols.txt": "2507",
            "SP500_Symbols.txt": "2506",
        }
        runnum = runnum_map.get(basename, "2508d")
    
    temp_json_fn = os.path.join(
        temp_dir, 
        f"{model_id}_backtest_montecarlo_{date_str}_{runnum}_{iter_num:03d}.json"
    )

    # Record the trial identifier in the Valuation section so the JSON
    # is self-documenting and easy to match against CSV rows.
    updated_params["Valuation"]["trial"] = f"{runnum}_{iter_num:03d}"

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


def print_symbols_with_sharpe_on_date(
    datearray: list,
    symbols: list,
    adjClose: np.ndarray,
    target_date: datetime.date,
    lookback_period: int = 252,
    signal2D: np.ndarray = None,
    weights: np.ndarray = None,
    numberStocksTraded: int = 5
) -> None:
    """
    Print all symbols with their adjusted close and Sharpe ratios on a
    specific date, sorted by Sharpe ratio from minimum to maximum.

    Args:
        datearray: Array of dates corresponding to each column.
        symbols: List of stock symbols.
        adjClose: 2D array of adjusted close prices [n_stocks, n_days].
        target_date: The target date to display data for.
        lookback_period: Number of days for Sharpe ratio calculation.
        signal2D: 2D array of signals [n_stocks, n_days] (optional).
        weights: 2D array of portfolio weights [n_stocks, n_days] (optional).
        numberStocksTraded: Number of stocks expected to be traded (default 5).
    """
    from math import sqrt

    # Find the index for the target date.
    target_index = None
    for i, d in enumerate(datearray):
        if d >= target_date:
            target_index = i
            break

    if target_index is None:
        print(f"Date {target_date} not found in datearray.")
        return

    actual_date = datearray[target_index]
    print("")
    print("=" * 60)
    print(f"SYMBOLS WITH ADJUSTED CLOSE AND SHARPE RATIOS")
    print(f"Target Date: {target_date} (Actual: {actual_date})")
    print(f"Lookback Period: {lookback_period} days")
    print("=" * 60)

    n_stocks = adjClose.shape[0]

    # Compute daily gain/loss for Sharpe calculation.
    dailygainloss = np.ones_like(adjClose, dtype=float)
    dailygainloss[:, 1:] = adjClose[:, 1:] / adjClose[:, :-1]
    dailygainloss[np.isnan(dailygainloss)] = 1.0
    dailygainloss[np.isinf(dailygainloss)] = 1.0

    # Arrays to store results.
    symbol_arr = []
    adj_close_arr = []
    sharpe_arr = []
    signal_arr = []
    weight_arr = []

    # Compute Sharpe ratio for each stock.
    start_idx = max(0, target_index - lookback_period)

    for i in range(n_stocks):
        adj_close_val = adjClose[i, target_index]

        # CASH should always have Sharpe = 0.0
        if symbols[i] == 'CASH':
            sharpe_val = 0.0
        else:
            # Get daily returns for this stock in the lookback window.
            returns_window = dailygainloss[i, start_idx:target_index + 1]
            valid_returns = returns_window[~np.isnan(returns_window)]

            if len(valid_returns) < lookback_period // 2:
                sharpe_val = 0.0
            else:
                # Compute Sharpe ratio.
                mean_ret = np.mean(valid_returns) - 1.0
                std_ret = np.std(valid_returns - 1.0)
                if std_ret > 1e-8:
                    sharpe_val = (mean_ret / std_ret) * sqrt(252)
                else:
                    sharpe_val = 0.0

        symbol_arr.append(symbols[i])
        adj_close_arr.append(adj_close_val)
        sharpe_arr.append(sharpe_val)
        
        # Extract signal2D and weights values if provided
        signal_val = signal2D[i, target_index] if signal2D is not None else 0.0
        weight_val = weights[i, target_index] if weights is not None else 0.0
        
        signal_arr.append(signal_val)
        weight_arr.append(weight_val)

    # Convert to numpy arrays.
    symbol_arr = np.array(symbol_arr)
    adj_close_arr = np.array(adj_close_arr)
    sharpe_arr = np.array(sharpe_arr)
    signal_arr = np.array(signal_arr)
    weight_arr = np.array(weight_arr)

    # Sort by Sharpe ratio from minimum to maximum.
    sort_indices = np.argsort(sharpe_arr)
    symbol_arr = symbol_arr[sort_indices]
    adj_close_arr = adj_close_arr[sort_indices]
    sharpe_arr = sharpe_arr[sort_indices]
    signal_arr = signal_arr[sort_indices]
    weight_arr = weight_arr[sort_indices]

    # Print header.
    print(f"{'Symbol':<6} {'AdjClose':>7} {'Sharpe':>7} {'Signal':>7} {'Weight':>7}")
    print("-" * 36)

    # Print one line per symbol.
    for i in range(len(symbol_arr)):
        print(f"{symbol_arr[i]:<6} {adj_close_arr[i]:7.2f} {sharpe_arr[i]:7.2f} {signal_arr[i]:7.2f} {weight_arr[i]:7.2f}")

    # Calculate weight statistics.
    min_weight = np.min(weight_arr)
    max_weight = np.max(weight_arr)
    non_zero_weights = np.sum(weight_arr > 0)
    sum_weights = np.sum(weight_arr)

    print("=" * 60)
    print(f"Total symbols: {len(symbol_arr)}")


def execute_single_backtest(
    json_fn,
    adjClose, symbols, datearray, gainloss, value, activeCount,
    monthsToHold, numberStocksTraded, LongPeriod, stddevThreshold,
    MA1, MA2, MA2offset, sma2factor, rankThresholdPct,
    riskDownside_min, riskDownside_max, lowPct, hiPct,
    uptrendSignalMethod, sma_filt_val, iter_num,
    validated_params,
    max_weight_factor=3.0, min_weight_factor=0.3,
    absolute_max_weight=0.9, apply_constraints=True,
    verbose=False,
    generate_plot=False,
    runnum=None
):
    """
    Execute the core backtest logic for a single realization.
    
    Args:
        generate_plot: If True, generate PNG plot of results
        runnum: Run identifier string (passed to plot filename)
        
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
    # Generate signals using computeSignal2D for compatibility with pytaaa_main.py
    #############################################################################
    
    print(f" ... Calling computeSignal2D with MA1={MA1}, MA2={MA2}, MA2offset={MA2offset}")
    
    try:
        # Prepare parameters for computeSignal2D.
        # Use uptrendSignalMethod from the validated JSON params (passed
        # as an argument to this function) rather than hardcoding a value,
        # so that the method printed in the log matches what is actually used.
        signal_params = {
            'MA1': MA1,
            'MA2': MA2,
            'MA2offset': MA2offset,
            'MA2factor': sma2factor,
            'uptrendSignalMethod': uptrendSignalMethod,
            'narrowDays': validated_params.get(
                'narrowDays', [6.0, 40.2]
            ),
            'mediumDays': validated_params.get(
                'mediumDays', [25.2, 38.3]
            ),
            'wideDays': validated_params.get(
                'wideDays', [75.2, 512.3]
            ),
            'lowPct': lowPct,
            'hiPct': hiPct,
            'minperiod': validated_params.get('minperiod', 4),
            'maxperiod': validated_params.get('maxperiod', 12),
            'incperiod': validated_params.get('incperiod', 3)
        }
        
        # Call computeSignal2D for signal generation
        signal_result = computeSignal2D(adjClose, gainloss, signal_params)
        
        # For percentileChannels, computeSignal2D returns (signal2D, lowChannel, hiChannel)
        if isinstance(signal_result, tuple):
            signal2D, lowChannel, hiChannel = signal_result
            print(f" ... Generated signal2D with shape: {signal2D.shape}")
            print(f" ... Generated channels with shapes: low={lowChannel.shape}, hi={hiChannel.shape}")
        else:
            # Fallback if not tuple
            signal2D = signal_result
            lowChannel = np.zeros_like(adjClose)
            hiChannel = np.zeros_like(adjClose)
            print(f" ... Generated signal2D only (fallback), shape: {signal2D.shape}")
        
        print(f" ... Signal2D stats: min={signal2D.min():.3f}, max={signal2D.max():.3f}, mean={signal2D.mean():.3f}")
        
    except Exception as e:
        print(f" ... Error in computeSignal2D: {e}")
        print(" ... Using fallback signal generation")
        
        # Fallback to simple moving average signals
        try:
            # Use filtered SMAs as fallback
            sma_short = SMA_filtered_2D(adjClose, MA2, sma_filt_val)
            sma_long = SMA_filtered_2D(adjClose, MA1, sma_filt_val)
            
            signal2D = np.zeros_like(adjClose, dtype=float)
            signal2D[sma_short > sma_long] = 1.0
            
            lowChannel = np.zeros_like(adjClose)
            hiChannel = np.zeros_like(adjClose)
            
            print(" ... Generated fallback SMA-based signals")
            
        except Exception as e2:
            print(f" ... Error with SMA fallback: {e2}")
            # Final fallback - simple trend-following signals
            signal2D = np.ones_like(adjClose) * 0.5  # Neutral signal
            print(" ... Using neutral signals as final fallback")
    
    # Apply rolling window data quality filter if enabled
    print(f"DEBUG: enable_rolling_filter = {validated_params.get('enable_rolling_filter', True)}")
    if validated_params.get('enable_rolling_filter', True):  # Default enabled to catch interpolated data
        from functions.rolling_window_filter import apply_rolling_window_filter
        print(" ... Applying rolling window data quality filter to detect interpolated data...")
        original_signal_count = np.sum(signal2D > 0)
        # FIXED: Changed verbose=True to verbose=False to suppress messages
        signal2D = apply_rolling_window_filter(
            adjClose, signal2D, validated_params.get('window_size', 50),
            symbols=symbols, datearray=datearray, verbose=False
        )
        filtered_signal_count = np.sum(signal2D > 0)
        print(f" ... Rolling window filter complete: {original_signal_count} -> {filtered_signal_count} signals (window_size={validated_params.get('window_size', 50)})")
    else:
        print("DEBUG: Rolling filter SKIPPED because enable_rolling_filter is False")

    # Create signal2D_daily for daily signals (before monthly hold logic)
    signal2D_daily = signal2D.copy()
        
    # Hold signal constant for each month based on monthsToHold parameter
    for jj in np.arange(1, adjClose.shape[1]):
        if not ((datearray[jj].month != datearray[jj-1].month) and 
                (datearray[jj].month - 1) % monthsToHold == 0):
            signal2D[:, jj] = signal2D[:, jj-1]

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
            print("\nBACKTEST_OVERWRITE ASSERT: mismatches at rebalance dates")
            for m in mismatches[:10]:
                print(m)
            raise AssertionError("Monthly signals differ from filtered daily signals after forward-fill in core_backtest.py")
    except AssertionError:
        raise
    except Exception:
        pass

    # Apply SP500 pre-2002 constraint to signals: no signals for dates before 2002-01-01
    if validated_params.get('stockList') == 'SP500':
        cutoff_date = np.datetime64('2002-01-01')
        for j in range(len(datearray)):
            if datearray[j] < cutoff_date:
                signal2D[:, j] = 0.0  # No signals = no stock selection
        print(f" ... Applied SP500 pre-2002 signal constraint: zero signals for dates before {cutoff_date}")

    numberStocks = np.sum(signal2D, axis=0)
    print(f" ... Number of stocks with signals: min={numberStocks.min():.1f}, max={numberStocks.max():.1f}, mean={numberStocks.mean():.1f}")

    #############################################################################
    # Compute portfolio weights using sharpeWeightedRank_2D
    #############################################################################
    
    # Use weight constraint parameters passed to this function (no hardcoding)
    print(f" ... Weight constraint parameters for realization {iter_num}:")
    print(f" ...   max_weight_factor = {max_weight_factor}")
    print(f" ...   min_weight_factor = {min_weight_factor}")
    print(f" ...   absolute_max_weight = {absolute_max_weight}")
    print(f" ...   apply_constraints = {apply_constraints}")
    print(f" ...   numberStocksTraded = {numberStocksTraded}")
    if numberStocksTraded > 0:
        equal_weight = 1.0 / numberStocksTraded
        print(f" ...   equal_weight (1/numberStocksTraded) = {equal_weight:.4f}")
        print(f" ...   max_weight = {min(equal_weight * max_weight_factor, absolute_max_weight):.4f}")
        print(f" ...   min_weight = {equal_weight * min_weight_factor:.4f}")
    
    try:
        monthgainlossweight = sharpeWeightedRank_2D(
            json_fn, datearray, symbols, adjClose, signal2D, signal2D_daily,
            LongPeriod, numberStocksTraded,
            riskDownside_min, riskDownside_max, rankThresholdPct,
            stddevThreshold=stddevThreshold,
            makeQCPlots=False,
            max_weight_factor=max_weight_factor,
            min_weight_factor=min_weight_factor,
            absolute_max_weight=absolute_max_weight,
            apply_constraints=apply_constraints,
            is_backtest=True,
            stockList=validated_params.get('stockList', 'SP500')  # Pass stockList for early period logic
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
    # Calculate variable percent investment portfolio values
    #############################################################################
    # Get variable percent invest parameters (use defaults from original if not in JSON)
    QminusSMADays = validated_params.get('QminusSMADays', 355)
    QminusSMAFactor = validated_params.get('QminusSMAFactor', 0.90)
    PctInvestSlope = validated_params.get('PctInvestSlope', 5.45)
    PctInvestIntercept = validated_params.get('PctInvestIntercept', -0.01)
    maxPctInvested = validated_params.get('maxPctInvested', 1.25)
    
    # Calculate smooth SMA of adj close
    adjCloseSMA = QminusSMAFactor * SMA_filtered_2D(adjClose, QminusSMADays)
    
    # Calculate QminusSMA metric
    QminusSMA = np.zeros(adjClose.shape[1], dtype=float)
    for ii in range(1, adjClose.shape[1]):
        adjClose_date = adjClose[:, ii]
        adjClose_prevdate = adjClose[:, ii-1]
        adjCloseSMA_date = adjCloseSMA[:, ii]
        # Only include stocks that changed price
        adjClose_date_edit = adjClose_date[adjClose_date != adjClose_prevdate]
        adjCloseSMA_date_edit = adjCloseSMA_date[adjClose_date != adjClose_prevdate]
        if np.sum(adjCloseSMA_date_edit) > 0:
            QminusSMA[ii] = (np.sum(adjClose_date_edit - adjCloseSMA_date_edit) / 
                            np.sum(adjCloseSMA_date_edit))
        else:
            QminusSMA[ii] = 0.0
    
    # Calculate percent invested signal from QminusSMA
    aa = (QminusSMA + PctInvestIntercept) * PctInvestSlope
    monthPctInvested = np.clip(aa, 0., maxPctInvested)
    
    # Calculate monthvalue with variable percent invested
    monthvalueVariablePctInvest = value.copy()
    for ii in np.arange(1, monthgainloss.shape[1]):
        if ((datearray[ii].month != datearray[ii-1].month) and 
            ((datearray[ii].month - 1) % monthsToHold == 0)):
            # Rebalancing date
            valuesum = np.sum(monthvalueVariablePctInvest[:, ii-1])
            for jj in range(value.shape[0]):
                monthvalueVariablePctInvest[jj, ii] = (
                    monthgainlossweight[jj, ii] * valuesum * 
                    (1.0 + (gainloss[jj, ii] - 1.0) * monthPctInvested[ii])
                )
        else:
            # Non-rebalancing date
            monthPctInvested[ii] = monthPctInvested[ii-1]
            for jj in range(value.shape[0]):
                monthvalueVariablePctInvest[jj, ii] = (
                    monthvalueVariablePctInvest[jj, ii-1] * 
                    (1.0 + (gainloss[jj, ii] - 1.0) * monthPctInvested[ii])
                )
    
    #############################################################################
    # Validate monthvalue before calculating PortfolioValue
    #############################################################################
    if monthvalue.size == 0 or np.isnan(monthvalue).any() or np.isinf(monthvalue).any():
        print("Error: monthvalue contains invalid data (empty, NaN, or Inf).")
        PortfolioValue = np.zeros(monthvalue.shape[1])
        PortfolioDailyGains = np.zeros(monthvalue.shape[1] - 1)
        FinalValue = 0.0
    else:
        PortfolioValue = np.average(monthvalue, axis=0)
        PortfolioDailyGains = np.divide(
            PortfolioValue[1:], PortfolioValue[:-1],
            out=np.zeros_like(PortfolioValue[1:]), where=PortfolioValue[:-1] != 0
        )
        FinalValue = np.average(monthvalue[:, -1])

    #############################################################################
    # Debugging: Log PortfolioValue and PortfolioDailyGains
    #############################################################################
    print(f"PortfolioValue: {PortfolioValue}")
    print(f"PortfolioDailyGains: {PortfolioDailyGains}")

    # Calculate Sharpe ratio with error handling
    try:
        daily_returns = PortfolioDailyGains
        annual_return = gmean(daily_returns) ** 252 - 1.0
        annual_volatility = np.std(daily_returns) * np.sqrt(252)
        PortfolioSharpe = annual_return / annual_volatility if annual_volatility > 0 else 0.0
    except Exception as e:
        print(f"Error calculating Sharpe ratio: {e}")
        PortfolioSharpe = 0.0

    print(f" ... Realization {iter_num} complete: Final Value = {FinalValue:,.0f}, Sharpe = {PortfolioSharpe:.3f}")

    # Print diagnostic for first realization
    if iter_num == 0:
        print_symbols_with_sharpe_on_date(
            datearray,
            symbols,
            adjClose,
            target_date=datetime.date(2025, 12, 1),
            lookback_period=252,
            signal2D=signal2D,
            weights=monthgainlossweight,
            numberStocksTraded=validated_params['numberStocksTraded']
        )

    # Generate plot if requested
    if generate_plot:
        try:
            from functions.backtesting.plot_generation import generate_backtest_plot
            
            # Calculate Buy & Hold portfolio for comparison (average of all stocks)
            BuyHoldPortfolioValue = np.mean(value, axis=0)
            
            # Build list of symbols held at final date 
            last_symbols_text = []
            for ii in range(len(symbols)):
                if monthgainlossweight[ii, -1] > 0:
                    last_symbols_text.append(symbols[ii])
            last_symbols_str = ", ".join(last_symbols_text)
            
            # Build results dict for plotting
            plot_results = {
                'finalValue': FinalValue,
                'sharpeRatio': PortfolioSharpe,
                'numberStocks': numberStocks,
                'QminusSMADays': QminusSMADays,
                'QminusSMAFactor': QminusSMAFactor,
                'PctInvestSlope': PctInvestSlope,
                'PctInvestIntercept': PctInvestIntercept,
                'maxPctInvested': maxPctInvested,
                'parameters': {
                    'monthsToHold': monthsToHold,
                    'numberStocksTraded': numberStocksTraded,
                    'LongPeriod': LongPeriod,
                    'MA1': MA1,
                    'MA2': MA2,
                    'MA2offset': MA2offset,
                    'lowPct': lowPct,
                    'hiPct': hiPct,
                    'sma2factor': validated_params.get('sma2factor', 0.91),
                    'rankThresholdPct': validated_params.get('rankThresholdPct', 0.5),
                    'sma_filt_val': validated_params.get('sma_filt_val', 0.02),
                    'riskDownside_min': validated_params.get('riskDownside_min', 0.5),
                    'riskDownside_max': validated_params.get('riskDownside_max', 5.0),
                }
            }
            
            symbols_file = validated_params.get('symbols_file', 'unknown_symbols.txt')
            plot_path = generate_backtest_plot(
                datearray,
                monthvalue,
                monthvalueVariablePctInvest,
                BuyHoldPortfolioValue,
                symbols_file,
                plot_results,
                json_fn,
                iter_num,
                QminusSMA,
                monthPctInvested,
                last_symbols_str,
                output_dir=None,
                runnum=runnum
            )
        except Exception as e:
            print(f" ... Warning: Could not generate plot: {e}")
            import traceback
            traceback.print_exc()

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
    verbose=False,
    generate_plot=False,
    runnum=None
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
        generate_plot: If True, generate PNG plot of results
        runnum: Run identifier string (passed to temp JSON filename)
        
    Returns:
        Dictionary with backtest results for this realization
    """
    temp_json_fn = None
    
    try:
        print(f" ... Creating temporary JSON for realization {iter_num}")
        
        # Create temporary JSON file
        temp_json_fn = create_temporary_json(base_json_fn, realization_params, iter_num, runnum)
        
        print(f" ... Created temp file: {temp_json_fn}")
        
        # Load parameters from temporary JSON
        params = get_json_params(temp_json_fn)
        
        if params is None:
            raise ValueError("Failed to load parameters from temporary JSON file")
        
        print(f" ... Loaded parameters successfully")
        
        # Validate parameters
        try:
            validated_params = validate_backtest_parameters(params)
            print(f" ... Parameters validated successfully")
        except ValueError as e:
            print(f" ... Parameter validation failed: {e}")
            raise
        
        # Extract validated parameters
        monthsToHold = validated_params['monthsToHold']
        numberStocksTraded = validated_params['numberStocksTraded']
        LongPeriod = validated_params['LongPeriod']
        stddevThreshold = validated_params['stddevThreshold']
        MA1 = validated_params['MA1']
        MA2 = validated_params['MA2']
        MA2offset = validated_params['MA2offset']
        sma2factor = validated_params['sma2factor']
        rankThresholdPct = validated_params['rankThresholdPct']
        riskDownside_min = validated_params['riskDownside_min']
        riskDownside_max = validated_params['riskDownside_max']
        lowPct = validated_params['lowPct']
        hiPct = validated_params['hiPct']
        uptrendSignalMethod = validated_params['uptrendSignalMethod']
        sma_filt_val = validated_params['sma_filt_val']
        
        # Extract weight constraint parameters directly from realization_params
        # to ensure we use the randomly generated values, not defaults
        max_weight_factor = realization_params.get('max_weight_factor', validated_params.get('max_weight_factor', 3.0))
        min_weight_factor = realization_params.get('min_weight_factor', validated_params.get('min_weight_factor', 0.3))
        absolute_max_weight = realization_params.get('absolute_max_weight', validated_params.get('absolute_max_weight', 0.9))
        apply_constraints = realization_params.get('apply_constraints', validated_params.get('apply_constraints', True))
        
        print(f" ... Weight params: max={max_weight_factor:.3f}, min={min_weight_factor:.3f}, abs_max={absolute_max_weight:.3f}")
        
        # Additional validation for array inputs
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
            uptrendSignalMethod, sma_filt_val, iter_num,
            validated_params,
            max_weight_factor=max_weight_factor,
            min_weight_factor=min_weight_factor,
            absolute_max_weight=absolute_max_weight,
            apply_constraints=apply_constraints,
            verbose=verbose,
            generate_plot=generate_plot,
            runnum=runnum
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
        # Keep temporary files for debugging/analysis (not cleaning up)
        if temp_json_fn:
            print(f" ... Temporary file retained: {temp_json_fn}")
            # cleanup_temporary_json(temp_json_fn)  # Commented out to preserve temp files
