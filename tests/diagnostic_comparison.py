"""
Step-by-step diagnostic comparison between original and refactored backtest.

This script isolates each computation stage to identify where results diverge.
"""

import os
import sys
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

# Add project root to path.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"STAGE: {title}")
    print("=" * 70)


def compare_arrays(
    name: str,
    arr1: np.ndarray,
    arr2: np.ndarray,
    tolerance: float = 1e-10
) -> bool:
    """
    Compare two arrays and report differences.
    
    Returns True if arrays match within tolerance.
    """
    if arr1.shape != arr2.shape:
        print(f"  {name}: ✗ SHAPE MISMATCH - {arr1.shape} vs {arr2.shape}")
        return False
    
    # Handle NaN values.
    nan1 = np.isnan(arr1)
    nan2 = np.isnan(arr2)
    
    if not np.array_equal(nan1, nan2):
        nan_diff = np.sum(nan1 != nan2)
        print(f"  {name}: ✗ NaN MISMATCH - {nan_diff} positions differ")
        return False
    
    # Compare non-NaN values.
    mask = ~nan1
    if not np.any(mask):
        print(f"  {name}: ✓ MATCH (all NaN)")
        return True
    
    diff = np.abs(arr1[mask] - arr2[mask])
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # Relative difference.
    denom = np.maximum(np.abs(arr1[mask]), 1e-10)
    rel_diff = diff / denom
    max_rel_diff = np.max(rel_diff)
    
    match = max_diff < tolerance
    status = "✓ MATCH" if match else "✗ DIFFER"
    
    print(f"  {name}: {status}")
    print(f"    Shape: {arr1.shape}")
    print(f"    Max Abs Diff: {max_diff:.2e}")
    print(f"    Mean Abs Diff: {mean_diff:.2e}")
    print(f"    Max Rel Diff: {max_rel_diff:.2e}")
    
    if not match:
        # Find where differences occur.
        diff_full = np.zeros_like(arr1)
        diff_full[mask] = diff
        max_idx = np.unravel_index(np.argmax(diff_full), arr1.shape)
        print(f"    Max diff at index: {max_idx}")
        print(f"    Value 1: {arr1[max_idx]}")
        print(f"    Value 2: {arr2[max_idx]}")
    
    return match


def compare_scalars(
    name: str,
    val1: float,
    val2: float,
    tolerance: float = 1e-10
) -> bool:
    """Compare two scalar values."""
    diff = abs(val1 - val2)
    rel_diff = diff / max(abs(val1), 1e-10)
    match = diff < tolerance
    
    status = "✓ MATCH" if match else "✗ DIFFER"
    print(f"  {name}: {status}")
    print(f"    Value 1: {val1}")
    print(f"    Value 2: {val2}")
    print(f"    Abs Diff: {diff:.2e}, Rel Diff: {rel_diff:.2e}")
    
    return match


def stage_1_data_loading() -> Tuple[Any, Any, Any, str]:
    """
    Stage 1: Load data from HDF5.
    
    Both versions should use identical data loading.
    """
    print_header("1. DATA LOADING")
    
    from functions.UpdateSymbols_inHDF5 import loadQuotes_fromHDF
    from functions.GetParams import get_json_params
    
    json_fn = "/Users/donaldpg/pyTAAA_data/sp500_pine/pytaaa_sp500_pine.json"
    params = get_json_params(json_fn, verbose=False)
    symbols_file = params["symbols_file"]
    
    print(f"  JSON file: {json_fn}")
    print(f"  Symbols file: {symbols_file}")
    
    adjClose, symbols, datearray, _, _ = loadQuotes_fromHDF(
        symbols_file, json_fn
    )
    
    print(f"  Loaded adjClose shape: {adjClose.shape}")
    print(f"  Number of symbols: {len(symbols)}")
    print(f"  Date range: {datearray[0]} to {datearray[-1]}")
    print(f"  adjClose sample [0,0:5]: {adjClose[0, 0:5]}")
    
    return adjClose, symbols, datearray, json_fn


def stage_2_data_cleaning(
    adjClose: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stage 2: Clean the data (interpolate, clean to beginning/end).
    
    Returns cleaned data for both original and refactored (should be same).
    """
    print_header("2. DATA CLEANING")
    
    from functions.quotes_for_list_adjClose import (
        interpolate, cleantobeginning, cleantoend
    )
    
    # Clean using original method.
    adjClose_orig = adjClose.copy()
    for ii in range(adjClose_orig.shape[0]):
        adjClose_orig[ii, :] = interpolate(adjClose_orig[ii, :])
        adjClose_orig[ii, :] = cleantobeginning(adjClose_orig[ii, :])
        adjClose_orig[ii, :] = cleantoend(adjClose_orig[ii, :])
    
    # Clean using same method (should be identical).
    adjClose_ref = adjClose.copy()
    for ii in range(adjClose_ref.shape[0]):
        adjClose_ref[ii, :] = interpolate(adjClose_ref[ii, :])
        adjClose_ref[ii, :] = cleantobeginning(adjClose_ref[ii, :])
        adjClose_ref[ii, :] = cleantoend(adjClose_ref[ii, :])
    
    compare_arrays("Cleaned adjClose", adjClose_orig, adjClose_ref)
    
    print(f"  Cleaned adjClose sample [0,0:5]: {adjClose_orig[0, 0:5]}")
    print(f"  NaN count after cleaning: {np.sum(np.isnan(adjClose_orig))}")
    
    return adjClose_orig, adjClose_ref


def stage_3_gainloss_calculation(
    adjClose: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stage 3: Calculate gain/loss and cumulative value.
    """
    print_header("3. GAIN/LOSS CALCULATION")
    
    # Original method.
    gainloss_orig = np.ones(
        (adjClose.shape[0], adjClose.shape[1]), dtype=float
    )
    gainloss_orig[:, 1:] = adjClose[:, 1:] / adjClose[:, :-1]
    gainloss_orig[np.isnan(gainloss_orig)] = 1.0
    value_orig = 10000.0 * np.cumprod(gainloss_orig, axis=1)
    
    # Refactored method (using TradingConstants).
    from src.backtest.config import TradingConstants
    
    gainloss_ref = np.ones(
        (adjClose.shape[0], adjClose.shape[1]), dtype=float
    )
    gainloss_ref[:, 1:] = adjClose[:, 1:] / adjClose[:, :-1]
    gainloss_ref[np.isnan(gainloss_ref)] = 1.0
    value_ref = (
        TradingConstants.INITIAL_PORTFOLIO_VALUE *
        np.cumprod(gainloss_ref, axis=1)
    )
    
    compare_arrays("gainloss", gainloss_orig, gainloss_ref)
    compare_arrays("value", value_orig, value_ref)
    
    print(f"  gainloss sample [0,0:5]: {gainloss_orig[0, 0:5]}")
    print(f"  value sample [0,0:5]: {value_orig[0, 0:5]}")
    
    return gainloss_orig, gainloss_ref, value_orig, value_ref


def stage_4_parameter_generation(seed: int = 42) -> Tuple[Dict, Dict]:
    """
    Stage 4: Compare random parameter generation.
    """
    print_header("4. PARAMETER GENERATION")
    
    from random import choice, seed as set_seed
    from src.backtest.montecarlo import random_triangle as random_triangle_ref
    
    # Original random_triangle.
    def random_triangle_orig(low=0.0, mid=0.5, high=1.0, size=1):
        uni = np.random.uniform(low, high, size)
        tri = np.random.triangular(low, mid, high, size)
        if size == 1:
            return ((uni + tri) / 2.0)[0]
        else:
            return (uni + tri) / 2.0
    
    # Generate with original.
    np.random.seed(seed)
    set_seed(seed)
    
    params_orig = {
        "numberStocksTraded": choice([5, 6, 6, 7, 7, 8, 8]),
        "monthsToHold": choice([1, 1, 1, 1, 1, 1, 1, 1, 1, 2]),
        "LongPeriod": int(random_triangle_orig(low=190, mid=370, high=550)),
        "stddevThreshold": random_triangle_orig(low=5.0, mid=7.50, high=10.0),
        "MA1": int(random_triangle_orig(low=75, mid=151, high=300)),
        "MA2": int(random_triangle_orig(low=10, mid=20, high=50)),
        "lowPct": np.random.uniform(10.0, 30.0),
        "hiPct": np.random.uniform(70.0, 90.0),
    }
    params_orig["MA2"] = max(params_orig["MA2"], 3)
    params_orig["MA1"] = max(params_orig["MA1"], params_orig["MA2"] + 1)
    params_orig["MA2offset"] = int(
        random_triangle_orig(
            low=(params_orig["MA1"] - params_orig["MA2"]) // 20,
            mid=(params_orig["MA1"] - params_orig["MA2"]) // 15,
            high=(params_orig["MA1"] - params_orig["MA2"]) // 10
        )
    )
    
    # Generate with refactored.
    np.random.seed(seed)
    set_seed(seed)
    
    params_ref = {
        "numberStocksTraded": choice([5, 6, 6, 7, 7, 8, 8]),
        "monthsToHold": choice([1, 1, 1, 1, 1, 1, 1, 1, 1, 2]),
        "LongPeriod": int(random_triangle_ref(low=190, mid=370, high=550)),
        "stddevThreshold": random_triangle_ref(low=5.0, mid=7.50, high=10.0),
        "MA1": int(random_triangle_ref(low=75, mid=151, high=300)),
        "MA2": int(random_triangle_ref(low=10, mid=20, high=50)),
        "lowPct": np.random.uniform(10.0, 30.0),
        "hiPct": np.random.uniform(70.0, 90.0),
    }
    params_ref["MA2"] = max(params_ref["MA2"], 3)
    params_ref["MA1"] = max(params_ref["MA1"], params_ref["MA2"] + 1)
    params_ref["MA2offset"] = int(
        random_triangle_ref(
            low=(params_ref["MA1"] - params_ref["MA2"]) // 20,
            mid=(params_ref["MA1"] - params_ref["MA2"]) // 15,
            high=(params_ref["MA1"] - params_ref["MA2"]) // 10
        )
    )
    
    # Compare each parameter.
    print("  Parameter comparison:")
    all_match = True
    for key in params_orig:
        orig_val = params_orig[key]
        ref_val = params_ref[key]
        
        if isinstance(orig_val, float):
            match = abs(orig_val - ref_val) < 1e-10
        else:
            match = orig_val == ref_val
        
        status = "✓" if match else "✗"
        print(f"    {key}: {status} orig={orig_val}, ref={ref_val}")
        
        if not match:
            all_match = False
    
    if all_match:
        print("  ✓ All parameters MATCH")
    else:
        print("  ✗ Parameters DIFFER")
    
    return params_orig, params_ref


def stage_5_signal_generation(
    adjClose: np.ndarray,
    params: Dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stage 5: Generate signals using percentileChannel_2D.
    """
    print_header("5. SIGNAL GENERATION (percentileChannel_2D)")
    
    from functions.TAfunctions import percentileChannel_2D
    
    MA1 = params["MA1"]
    MA2 = params["MA2"]
    MA2offset = params["MA2offset"]
    lowPct = params["lowPct"]
    hiPct = params["hiPct"]
    
    print(f"  Parameters: MA1={MA1}, MA2={MA2}, MA2offset={MA2offset}")
    print(f"  lowPct={lowPct:.2f}, hiPct={hiPct:.2f}")
    
    # Generate channels.
    lowChannel, hiChannel = percentileChannel_2D(
        adjClose, MA2, MA1 + 0.01, MA2offset, lowPct, hiPct
    )
    
    print(f"  lowChannel shape: {lowChannel.shape}")
    print(f"  hiChannel shape: {hiChannel.shape}")
    print(f"  lowChannel sample [0,100:105]: {lowChannel[0, 100:105]}")
    print(f"  hiChannel sample [0,100:105]: {hiChannel[0, 100:105]}")
    
    # Generate signals - Original method.
    signal2D_orig = np.zeros_like(adjClose, dtype=float)
    for i in range(adjClose.shape[0]):
        for j in range(1, adjClose.shape[1]):
            price = adjClose[i, j]
            low_thresh = lowChannel[i, j]
            hi_thresh = hiChannel[i, j]
            
            if not (np.isnan(price) or np.isnan(low_thresh) or
                    np.isnan(hi_thresh)):
                if price > hi_thresh:
                    signal2D_orig[i, j] = 1.0
                elif price > low_thresh:
                    signal2D_orig[i, j] = signal2D_orig[i, j - 1]
                else:
                    signal2D_orig[i, j] = 0.0
            else:
                signal2D_orig[i, j] = (
                    signal2D_orig[i, j - 1] if j > 0 else 0.0
                )
    
    # Generate signals - Refactored method (same logic).
    signal2D_ref = np.zeros_like(adjClose, dtype=float)
    for i in range(adjClose.shape[0]):
        for j in range(1, adjClose.shape[1]):
            price = adjClose[i, j]
            low_thresh = lowChannel[i, j]
            hi_thresh = hiChannel[i, j]
            
            if not (np.isnan(price) or np.isnan(low_thresh) or
                    np.isnan(hi_thresh)):
                if price > hi_thresh:
                    signal2D_ref[i, j] = 1.0
                elif price > low_thresh:
                    signal2D_ref[i, j] = signal2D_ref[i, j - 1]
                else:
                    signal2D_ref[i, j] = 0.0
            else:
                signal2D_ref[i, j] = (
                    signal2D_ref[i, j - 1] if j > 0 else 0.0
                )
    
    compare_arrays("signal2D", signal2D_orig, signal2D_ref)
    
    print(f"  signal2D sample [0,100:105]: {signal2D_orig[0, 100:105]}")
    print(f"  Total buy signals: {np.sum(signal2D_orig > 0.5)}")
    
    return signal2D_orig, signal2D_ref, lowChannel, hiChannel


def stage_6_monthly_hold(
    signal2D: np.ndarray,
    datearray: list,
    monthsToHold: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stage 6: Apply monthly hold logic.
    """
    print_header("6. MONTHLY HOLD LOGIC")
    
    print(f"  monthsToHold: {monthsToHold}")
    
    # Original method.
    signal2D_orig = signal2D.copy()
    signal2D_daily_orig = signal2D.copy()
    
    for jj in range(1, signal2D_orig.shape[1]):
        if not ((datearray[jj].month != datearray[jj - 1].month) and
                (datearray[jj].month - 1) % monthsToHold == 0):
            signal2D_orig[:, jj] = signal2D_orig[:, jj - 1]
    
    # Refactored method (same logic).
    signal2D_ref = signal2D.copy()
    signal2D_daily_ref = signal2D.copy()
    
    for jj in range(1, signal2D_ref.shape[1]):
        if not ((datearray[jj].month != datearray[jj - 1].month) and
                (datearray[jj].month - 1) % monthsToHold == 0):
            signal2D_ref[:, jj] = signal2D_ref[:, jj - 1]
    
    compare_arrays("signal2D after hold logic", signal2D_orig, signal2D_ref)
    
    # Count rebalance dates.
    rebalance_count = 0
    for jj in range(1, len(datearray)):
        if ((datearray[jj].month != datearray[jj - 1].month) and
                (datearray[jj].month - 1) % monthsToHold == 0):
            rebalance_count += 1
    
    print(f"  Rebalance dates: {rebalance_count}")
    
    return signal2D_orig, signal2D_daily_orig


def stage_7_weight_calculation(
    json_fn: str,
    datearray: list,
    symbols: list,
    adjClose: np.ndarray,
    signal2D: np.ndarray,
    signal2D_daily: np.ndarray,
    params: Dict
) -> np.ndarray:
    """
    Stage 7: Calculate portfolio weights using sharpeWeightedRank_2D.
    """
    print_header("7. WEIGHT CALCULATION (sharpeWeightedRank_2D)")
    
    from functions.TAfunctions import sharpeWeightedRank_2D
    
    LongPeriod = params["LongPeriod"]
    numberStocksTraded = params["numberStocksTraded"]
    riskDownside_min = params.get("riskDownside_min", 0.7)
    riskDownside_max = params.get("riskDownside_max", 10.0)
    rankThresholdPct = params.get("rankThresholdPct", 0.2)
    stddevThreshold = params.get("stddevThreshold", 7.5)
    
    print(f"  LongPeriod: {LongPeriod}")
    print(f"  numberStocksTraded: {numberStocksTraded}")
    print(f"  riskDownside: [{riskDownside_min}, {riskDownside_max}]")
    print(f"  rankThresholdPct: {rankThresholdPct}")
    print(f"  stddevThreshold: {stddevThreshold}")
    
    try:
        monthgainlossweight = sharpeWeightedRank_2D(
            json_fn, datearray, symbols, adjClose, signal2D,
            signal2D_daily, LongPeriod, numberStocksTraded,
            riskDownside_min, riskDownside_max, rankThresholdPct,
            stddevThreshold=stddevThreshold, makeQCPlots=False
        )
        
        print(f"  Weight matrix shape: {monthgainlossweight.shape}")
        print(f"  Non-zero weights: {np.sum(monthgainlossweight > 0)}")
        print(f"  Weight sum at end: {np.sum(monthgainlossweight[:, -1]):.4f}")
        print(f"  Weights sample [-1]: {monthgainlossweight[:5, -1]}")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        monthgainlossweight = np.zeros_like(adjClose)
    
    return monthgainlossweight


def stage_8_portfolio_calculation(
    value: np.ndarray,
    gainloss: np.ndarray,
    monthgainlossweight: np.ndarray,
    datearray: list,
    monthsToHold: int
) -> np.ndarray:
    """
    Stage 8: Calculate portfolio values over time.
    """
    print_header("8. PORTFOLIO VALUE CALCULATION")
    
    monthvalue = value.copy()
    
    rebalance_count = 0
    for ii in range(1, gainloss.shape[1]):
        if ((datearray[ii].month != datearray[ii - 1].month) and
                ((datearray[ii].month - 1) % monthsToHold == 0)):
            rebalance_count += 1
            valuesum = np.sum(monthvalue[:, ii - 1])
            for jj in range(value.shape[0]):
                monthvalue[jj, ii] = (
                    monthgainlossweight[jj, ii] * valuesum *
                    gainloss[jj, ii]
                )
        else:
            for jj in range(value.shape[0]):
                monthvalue[jj, ii] = (
                    monthvalue[jj, ii - 1] * gainloss[jj, ii]
                )
    
    PortfolioValue = np.average(monthvalue, axis=0)
    
    print(f"  Rebalance events: {rebalance_count}")
    print(f"  Portfolio value shape: {PortfolioValue.shape}")
    print(f"  Initial value: {PortfolioValue[0]:,.2f}")
    print(f"  Final value: {PortfolioValue[-1]:,.2f}")
    print(f"  Min value: {PortfolioValue.min():,.2f}")
    print(f"  Max value: {PortfolioValue.max():,.2f}")
    
    return monthvalue


def stage_9_metrics_calculation(
    monthvalue: np.ndarray
) -> Tuple[float, float]:
    """
    Stage 9: Calculate final metrics (Sharpe ratio, etc.).
    """
    print_header("9. METRICS CALCULATION")
    
    from scipy.stats import gmean
    from src.backtest.config import TradingConstants
    
    PortfolioValue = np.average(monthvalue, axis=0)
    FinalValue = PortfolioValue[-1]
    
    PortfolioDailyGains = PortfolioValue[1:] / PortfolioValue[:-1]
    
    # Original calculation (hardcoded 252).
    try:
        annual_return_orig = gmean(PortfolioDailyGains) ** 252 - 1.0
        annual_vol_orig = np.std(PortfolioDailyGains) * np.sqrt(252)
        sharpe_orig = (
            annual_return_orig / annual_vol_orig
            if annual_vol_orig > 0 else 0.0
        )
    except Exception:
        sharpe_orig = 0.0
    
    # Refactored calculation (using TradingConstants).
    try:
        trading_days = TradingConstants.TRADING_DAYS_PER_YEAR
        annual_return_ref = gmean(PortfolioDailyGains) ** trading_days - 1.0
        annual_vol_ref = np.std(PortfolioDailyGains) * np.sqrt(trading_days)
        sharpe_ref = (
            annual_return_ref / annual_vol_ref
            if annual_vol_ref > 0 else 0.0
        )
    except Exception:
        sharpe_ref = 0.0
    
    print(f"  Final Value: {FinalValue:,.2f}")
    print(f"  Annual Return (orig): {annual_return_orig:.4f}")
    print(f"  Annual Return (ref): {annual_return_ref:.4f}")
    print(f"  Annual Volatility (orig): {annual_vol_orig:.4f}")
    print(f"  Annual Volatility (ref): {annual_vol_ref:.4f}")
    
    compare_scalars("Sharpe Ratio", sharpe_orig, sharpe_ref)
    
    return FinalValue, sharpe_orig


def run_full_diagnostic(seed: int = 42):
    """Run the full diagnostic comparison."""
    print("\n" + "#" * 70)
    print("# STEP-BY-STEP DIAGNOSTIC COMPARISON")
    print(f"# Random Seed: {seed}")
    print(f"# Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70)
    
    # Stage 1: Data Loading.
    adjClose_raw, symbols, datearray, json_fn = stage_1_data_loading()
    
    # Stage 2: Data Cleaning.
    adjClose_orig, adjClose_ref = stage_2_data_cleaning(adjClose_raw)
    
    # Stage 3: Gain/Loss Calculation.
    gainloss_orig, gainloss_ref, value_orig, value_ref = \
        stage_3_gainloss_calculation(adjClose_orig)
    
    # Stage 4: Parameter Generation.
    params_orig, params_ref = stage_4_parameter_generation(seed)
    
    # Use original params for remaining stages.
    params = params_orig
    
    # Add missing parameters with defaults.
    params.setdefault("riskDownside_min", 0.7)
    params.setdefault("riskDownside_max", 10.0)
    params.setdefault("rankThresholdPct", 0.2)
    params.setdefault("numberStocksTraded", 7)
    
    # Stage 5: Signal Generation.
    signal2D_orig, signal2D_ref, lowChannel, hiChannel = \
        stage_5_signal_generation(adjClose_orig, params)
    
    # Stage 6: Monthly Hold Logic.
    signal2D_hold, signal2D_daily = stage_6_monthly_hold(
        signal2D_orig, datearray, params["monthsToHold"]
    )
    
    # Stage 7: Weight Calculation.
    monthgainlossweight = stage_7_weight_calculation(
        json_fn, datearray, symbols, adjClose_orig,
        signal2D_hold, signal2D_daily, params
    )
    
    # Stage 8: Portfolio Calculation.
    monthvalue = stage_8_portfolio_calculation(
        value_orig, gainloss_orig, monthgainlossweight,
        datearray, params["monthsToHold"]
    )
    
    # Stage 9: Metrics Calculation.
    final_value, sharpe = stage_9_metrics_calculation(monthvalue)
    
    # Summary.
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"  Final Portfolio Value: {final_value:,.2f}")
    print(f"  Sharpe Ratio: {sharpe:.4f}")
    print(f"  Parameters used: {params}")


if __name__ == "__main__":
    run_full_diagnostic(seed=42)
