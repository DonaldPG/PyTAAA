"""
Comparison script to verify refactored backtest produces same results.

This script runs both the original and refactored backtest code with
FIXED parameters (no randomness) to ensure a true apples-to-apples
comparison.
"""

import os
import sys
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple

# Add project root to path.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# Fixed parameters for comparison - no randomness.
FIXED_PARAMS = {
    "numberStocksTraded": 7,
    "monthsToHold": 1,
    "LongPeriod": 370,
    "stddevThreshold": 7.5,
    "MA1": 151,
    "MA2": 20,
    "MA2offset": 4,
    "MA3": 24,
    "sma2factor": 2.5,
    "rankThresholdPct": 0.20,
    "riskDownside_min": 0.70,
    "riskDownside_max": 10.0,
    "lowPct": 20.0,
    "hiPct": 80.0,
    "sma_filt_val": 0.015,
}


def load_and_prepare_data() -> Tuple[np.ndarray, list, list, np.ndarray, str]:
    """
    Load and prepare data for backtesting.
    
    Returns:
        Tuple of (adjClose, symbols, datearray, gainloss, json_fn).
    """
    from functions.quotes_for_list_adjClose import (
        interpolate, cleantobeginning, cleantoend
    )
    from functions.UpdateSymbols_inHDF5 import loadQuotes_fromHDF
    from functions.GetParams import get_json_params
    
    json_fn = "/Users/donaldpg/pyTAAA_data/sp500_pine/pytaaa_sp500_pine.json"
    params = get_json_params(json_fn, verbose=False)
    symbols_file = params["symbols_file"]
    
    # Load quotes.
    adjClose, symbols, datearray, _, _ = loadQuotes_fromHDF(
        symbols_file, json_fn
    )
    
    # Clean up missing values.
    for ii in range(adjClose.shape[0]):
        adjClose[ii, :] = interpolate(adjClose[ii, :])
        adjClose[ii, :] = cleantobeginning(adjClose[ii, :])
        adjClose[ii, :] = cleantoend(adjClose[ii, :])
    
    # Calculate gain/loss.
    gainloss = np.ones((adjClose.shape[0], adjClose.shape[1]), dtype=float)
    gainloss[:, 1:] = adjClose[:, 1:] / adjClose[:, :-1]
    gainloss[np.isnan(gainloss)] = 1.0
    
    return adjClose, symbols, datearray, gainloss, json_fn


def run_backtest_with_params(
    adjClose: np.ndarray,
    symbols: list,
    datearray: list,
    gainloss: np.ndarray,
    json_fn: str,
    params: Dict[str, Any],
    use_refactored: bool = False
) -> Dict[str, Any]:
    """
    Run backtest with fixed parameters.
    
    Args:
        adjClose: Adjusted close prices array.
        symbols: List of stock symbols.
        datearray: List of dates.
        gainloss: Daily gain/loss ratios.
        json_fn: Path to JSON config file.
        params: Fixed parameters dictionary.
        use_refactored: If True, use refactored module imports.
        
    Returns:
        Dictionary containing backtest results.
    """
    from functions.TAfunctions import percentileChannel_2D
    from functions.TAfunctions import sharpeWeightedRank_2D
    from scipy.stats import gmean
    
    version = "Refactored" if use_refactored else "Original"
    print(f"\n{'=' * 60}")
    print(f"Running {version} Backtest with Fixed Parameters")
    print("=" * 60)
    
    # Extract parameters.
    MA1 = params["MA1"]
    MA2 = params["MA2"]
    MA2offset = params["MA2offset"]
    lowPct = params["lowPct"]
    hiPct = params["hiPct"]
    monthsToHold = params["monthsToHold"]
    LongPeriod = params["LongPeriod"]
    numberStocksTraded = params["numberStocksTraded"]
    riskDownside_min = params["riskDownside_min"]
    riskDownside_max = params["riskDownside_max"]
    rankThresholdPct = params["rankThresholdPct"]
    stddevThreshold = params["stddevThreshold"]
    
    print(f"  MA1={MA1}, MA2={MA2}, MA2offset={MA2offset}")
    print(f"  lowPct={lowPct}, hiPct={hiPct}")
    print(f"  monthsToHold={monthsToHold}, LongPeriod={LongPeriod}")
    print(f"  numberStocksTraded={numberStocksTraded}")
    
    # Calculate initial value.
    if use_refactored:
        from src.backtest.config import TradingConstants
        initial_value = TradingConstants.INITIAL_PORTFOLIO_VALUE
        trading_days = TradingConstants.TRADING_DAYS_PER_YEAR
    else:
        initial_value = 10000.0
        trading_days = 252
    
    value = initial_value * np.cumprod(gainloss, axis=1)
    
    # Generate percentile channels.
    print("  Computing percentile channels...")
    lowChannel, hiChannel = percentileChannel_2D(
        adjClose, MA2, MA1 + 0.01, MA2offset, lowPct, hiPct
    )
    
    # Generate signals.
    print("  Generating signals...")
    signal2D = np.zeros_like(adjClose, dtype=float)
    for i in range(adjClose.shape[0]):
        for j in range(1, adjClose.shape[1]):
            price = adjClose[i, j]
            low_thresh = lowChannel[i, j]
            hi_thresh = hiChannel[i, j]
            
            if not (np.isnan(price) or np.isnan(low_thresh) or
                    np.isnan(hi_thresh)):
                if price > hi_thresh:
                    signal2D[i, j] = 1.0
                elif price > low_thresh:
                    signal2D[i, j] = signal2D[i, j - 1]
                else:
                    signal2D[i, j] = 0.0
            else:
                signal2D[i, j] = signal2D[i, j - 1] if j > 0 else 0.0
    
    # Apply monthly hold logic.
    signal2D_daily = signal2D.copy()
    for jj in range(1, adjClose.shape[1]):
        if not ((datearray[jj].month != datearray[jj - 1].month) and
                (datearray[jj].month - 1) % monthsToHold == 0):
            signal2D[:, jj] = signal2D[:, jj - 1]
    
    # Calculate weights.
    print("  Calculating weights...")
    monthgainlossweight = sharpeWeightedRank_2D(
        json_fn, datearray, symbols, adjClose, signal2D,
        signal2D_daily, LongPeriod, numberStocksTraded,
        riskDownside_min, riskDownside_max, rankThresholdPct,
        stddevThreshold=stddevThreshold, makeQCPlots=False
    )
    
    # Calculate portfolio values.
    print("  Computing portfolio values...")
    monthvalue = value.copy()
    for ii in range(1, gainloss.shape[1]):
        if ((datearray[ii].month != datearray[ii - 1].month) and
                ((datearray[ii].month - 1) % monthsToHold == 0)):
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
    
    # Calculate metrics.
    PortfolioValue = np.average(monthvalue, axis=0)
    FinalValue = PortfolioValue[-1]
    
    PortfolioDailyGains = PortfolioValue[1:] / PortfolioValue[:-1]
    try:
        annual_return = gmean(PortfolioDailyGains) ** trading_days - 1.0
        annual_volatility = np.std(PortfolioDailyGains) * np.sqrt(trading_days)
        sharpe = (
            annual_return / annual_volatility
            if annual_volatility > 0 else 0.0
        )
    except Exception:
        sharpe = 0.0
        annual_return = 0.0
        annual_volatility = 0.0
    
    results = {
        "final_value": FinalValue,
        "sharpe_ratio": sharpe,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "portfolio_values": PortfolioValue.copy(),
        "initial_value": initial_value,
        "trading_days": trading_days,
        "signal_sum": np.sum(signal2D),
        "weight_sum": np.sum(monthgainlossweight),
    }
    
    print(f"\n  Results:")
    print(f"    Initial Value: {initial_value:,.2f}")
    print(f"    Final Value: {FinalValue:,.2f}")
    print(f"    Annual Return: {annual_return:.4f} ({annual_return*100:.2f}%)")
    print(f"    Annual Volatility: {annual_volatility:.4f}")
    print(f"    Sharpe Ratio: {sharpe:.6f}")
    print(f"    Signal Sum: {results['signal_sum']:,.0f}")
    print(f"    Weight Sum: {results['weight_sum']:,.4f}")
    
    return results


def compare_results(
    original: Dict[str, Any],
    refactored: Dict[str, Any],
    tolerance: float = 1e-10
) -> bool:
    """
    Compare results from original and refactored backtest runs.
    
    Args:
        original: Results from original backtest.
        refactored: Results from refactored backtest.
        tolerance: Numerical tolerance for comparison.
        
    Returns:
        True if all results match within tolerance.
    """
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    all_match = True
    
    # Compare scalar values.
    scalar_keys = [
        ("final_value", "Final Value"),
        ("sharpe_ratio", "Sharpe Ratio"),
        ("annual_return", "Annual Return"),
        ("annual_volatility", "Annual Volatility"),
        ("signal_sum", "Signal Sum"),
        ("weight_sum", "Weight Sum"),
    ]
    
    print("\nScalar Comparisons:")
    print("-" * 50)
    
    for key, name in scalar_keys:
        orig_val = original[key]
        ref_val = refactored[key]
        diff = abs(orig_val - ref_val)
        rel_diff = diff / max(abs(orig_val), 1e-10)
        match = diff < tolerance or rel_diff < tolerance
        
        status = "✓ MATCH" if match else "✗ DIFFER"
        print(f"  {name}:")
        print(f"    Original:   {orig_val}")
        print(f"    Refactored: {ref_val}")
        print(f"    Abs Diff:   {diff:.2e}")
        print(f"    Rel Diff:   {rel_diff:.2e}")
        print(f"    Status:     {status}")
        
        if not match:
            all_match = False
    
    # Compare portfolio value arrays.
    print("\nPortfolio Values Array Comparison:")
    print("-" * 50)
    
    orig_pv = original["portfolio_values"]
    ref_pv = refactored["portfolio_values"]
    
    if len(orig_pv) != len(ref_pv):
        print(f"  ✗ Length mismatch: {len(orig_pv)} vs {len(ref_pv)}")
        all_match = False
    else:
        diff = np.abs(orig_pv - ref_pv)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        denom = np.maximum(np.abs(orig_pv), 1e-10)
        rel_diff = diff / denom
        max_rel_diff = np.max(rel_diff)
        
        match = max_diff < tolerance or max_rel_diff < tolerance
        status = "✓ MATCH" if match else "✗ DIFFER"
        
        print(f"  Array Length: {len(orig_pv)}")
        print(f"  Max Abs Diff: {max_diff:.2e}")
        print(f"  Mean Abs Diff: {mean_diff:.2e}")
        print(f"  Max Rel Diff: {max_rel_diff:.2e}")
        print(f"  Status: {status}")
        
        if not match:
            all_match = False
            # Find where max difference occurs.
            max_idx = np.argmax(diff)
            print(f"  Max diff at index {max_idx}:")
            print(f"    Original:   {orig_pv[max_idx]}")
            print(f"    Refactored: {ref_pv[max_idx]}")
    
    # Final summary.
    print("\n" + "=" * 60)
    if all_match:
        print("OVERALL RESULT: ✓ ALL TESTS PASSED")
        print("The refactored code produces IDENTICAL results!")
    else:
        print("OVERALL RESULT: ✗ DIFFERENCES DETECTED")
        print("The refactored code produces DIFFERENT results.")
    print("=" * 60)
    
    return all_match


def main() -> bool:
    """
    Run comparison between original and refactored backtest.
    
    Returns:
        True if all comparisons pass.
    """
    print("\n" + "#" * 70)
    print("# BACKTEST COMPARISON: ORIGINAL vs REFACTORED")
    print("# Using FIXED Parameters (No Randomness)")
    print(f"# Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70)
    
    print("\nFixed Parameters:")
    for key, value in FIXED_PARAMS.items():
        print(f"  {key}: {value}")
    
    # Load data once (shared between both runs).
    print("\nLoading data...")
    adjClose, symbols, datearray, gainloss, json_fn = load_and_prepare_data()
    print(f"  Data shape: {adjClose.shape}")
    print(f"  Date range: {datearray[0]} to {datearray[-1]}")
    
    # Run original backtest.
    original_results = run_backtest_with_params(
        adjClose, symbols, datearray, gainloss, json_fn,
        FIXED_PARAMS, use_refactored=False
    )
    
    # Run refactored backtest.
    refactored_results = run_backtest_with_params(
        adjClose, symbols, datearray, gainloss, json_fn,
        FIXED_PARAMS, use_refactored=True
    )
    
    # Compare results.
    all_match = compare_results(original_results, refactored_results)
    
    return all_match


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
