"""
Diagnostic script to examine JEF stock data and identify issues with
artificially inflated Sharpe ratios due to missing/infilled quotes.
"""
import os
import sys
import datetime
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions.UpdateSymbols_inHDF5 import loadQuotes_fromHDF
from functions.TAfunctions import interpolate, cleantobeginning, cleantoend
from functions.GetParams import get_json_params


def main():
    """Main diagnostic function."""
    print("=" * 80)
    print("JEF STOCK DATA DIAGNOSTIC")
    print("=" * 80)
    
    # Load data
    json_fn = "/Users/donaldpg/pyTAAA_data/sp500_pine/pytaaa_sp500_pine.json"
    params = get_json_params(json_fn, verbose=False)
    symbols_file = params["symbols_file"]
    
    print(f"\nLoading quotes from: {symbols_file}")
    adjClose_raw, symbols, datearray, _, _ = loadQuotes_fromHDF(symbols_file, json_fn)
    
    print(f"Loaded {len(symbols)} symbols, {adjClose_raw.shape[1]} days")
    print(f"Date range: {datearray[0]} to {datearray[-1]}")
    
    # Check if JEF exists
    if "JEF" not in symbols:
        print("\nERROR: JEF not found in symbols list!")
        return
    
    jef_idx = symbols.index("JEF")
    print(f"\nJEF index: {jef_idx}")
    
    # Show RAW data before cleaning
    print("\n" + "=" * 80)
    print("RAW DATA (before interpolate/cleantobeginning)")
    print("=" * 80)
    
    jef_raw = adjClose_raw[jef_idx, :].copy()
    nan_count_raw = np.sum(np.isnan(jef_raw))
    print(f"NaN count in raw data: {nan_count_raw}")
    
    # Find first non-NaN value
    if nan_count_raw > 0:
        first_non_nan = np.argmax(~np.isnan(jef_raw))
    else:
        # Find first price that differs from initial price
        first_non_nan = 0
    
    print(f"First non-NaN index: {first_non_nan}, date: {datearray[first_non_nan]}")
    print(f"First price: {jef_raw[first_non_nan]:.4f}")
    
    # Now apply cleaning (as done in backtest)
    print("\n" + "=" * 80)
    print("AFTER CLEANING (interpolate + cleantobeginning + cleantoend)")
    print("=" * 80)
    
    adjClose = adjClose_raw.copy()
    for ii in range(adjClose.shape[0]):
        adjClose[ii, :] = interpolate(adjClose[ii, :])
        adjClose[ii, :] = cleantobeginning(adjClose[ii, :])
        adjClose[ii, :] = cleantoend(adjClose[ii, :])
    
    # Calculate gainloss
    gainloss = np.ones_like(adjClose)
    gainloss[:, 1:] = adjClose[:, 1:] / adjClose[:, :-1]
    gainloss[np.isnan(gainloss)] = 1.0
    
    jef_prices = adjClose[jef_idx, :]
    jef_gains = gainloss[jef_idx, :]
    
    # Find first valid price (where gainloss != 1.0)
    first_valid_idx = np.argmax(np.abs(jef_gains - 1.0) > 1e-8)
    print(f"\nFirst valid trading index: {first_valid_idx}")
    print(f"First valid trading date: {datearray[first_valid_idx]}")
    print(f"Price at first valid date: {jef_prices[first_valid_idx]:.4f}")
    
    # Count days with gainloss == 1.0
    no_change_count = np.sum(np.abs(jef_gains - 1.0) < 1e-8)
    print(f"\nDays with gainloss = 1.0 (no change): {no_change_count}")
    print(f"Total days: {len(jef_gains)}")
    print(f"Percentage with no change: {100*no_change_count/len(jef_gains):.1f}%")
    
    # Show prices around key dates
    print("\n" + "=" * 80)
    print("PRICES AROUND KEY DATES (2016, 2018)")
    print("=" * 80)
    
    key_dates = [
        datetime.date(2015, 12, 15),
        datetime.date(2016, 1, 4),
        datetime.date(2016, 1, 15),
        datetime.date(2017, 12, 15),
        datetime.date(2018, 1, 2),
        datetime.date(2018, 1, 15),
        datetime.date(2018, 6, 1),
    ]
    
    print(f"\n{'Date':<12} {'Price':>12} {'GainLoss':>12} {'Notes'}")
    print("-" * 50)
    
    for target in key_dates:
        # Find closest date
        for i, d in enumerate(datearray):
            if d >= target:
                price = jef_prices[i]
                gl = jef_gains[i]
                notes = ""
                if abs(gl - 1.0) < 1e-8:
                    notes = "GL=1.0 (CONSTANT)"
                elif i == first_valid_idx:
                    notes = "<<< FIRST VALID"
                print(f"{d!s:<12} {price:>12.4f} {gl:>12.6f} {notes}")
                break
    
    # Show range of constant prices
    print("\n" + "=" * 80)
    print("CONSTANT PRICE PERIOD ANALYSIS")
    print("=" * 80)
    
    # Find all constant price periods
    constant_start = None
    for i in range(len(jef_gains)):
        is_constant = abs(jef_gains[i] - 1.0) < 1e-8
        if is_constant and constant_start is None:
            constant_start = i
        elif not is_constant and constant_start is not None:
            duration = i - constant_start
            if duration > 10:  # Only show periods > 10 days
                print(f"Constant period: {datearray[constant_start]} to {datearray[i-1]} "
                      f"({duration} days), price={jef_prices[constant_start]:.4f}")
            constant_start = None
    
    # Sharpe ratio analysis
    print("\n" + "=" * 80)
    print("SHARPE RATIO ANALYSIS - WHY JEF GETS SELECTED")
    print("=" * 80)
    
    from scipy.stats import gmean
    
    # Calculate Sharpe for period BEFORE first valid date
    if first_valid_idx > 252:
        period_gains = jef_gains[first_valid_idx - 252:first_valid_idx]
        ann_return = gmean(period_gains) ** 252 - 1.0
        ann_vol = np.std(period_gains) * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 1e-10 else float('inf')
        print(f"\n252 days BEFORE first valid date (infilled period):")
        print(f"  Annual return: {ann_return:.4%}")
        print(f"  Annual volatility: {ann_vol:.6f}")
        print(f"  Sharpe ratio: {sharpe:.2f}")
        print(f"  >>> This artificially high Sharpe causes selection!")
    
    # Calculate Sharpe for period AFTER first valid date  
    if first_valid_idx + 252 < len(jef_gains):
        period_gains = jef_gains[first_valid_idx:first_valid_idx + 252]
        ann_return = gmean(period_gains) ** 252 - 1.0
        ann_vol = np.std(period_gains) * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 1e-10 else 0
        print(f"\n252 days AFTER first valid date (real trading):")
        print(f"  Annual return: {ann_return:.4%}")
        print(f"  Annual volatility: {ann_vol:.4f}")
        print(f"  Sharpe ratio: {sharpe:.2f}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print("""
PROBLEM: cleantobeginning() copies the first valid price to all earlier dates.
This creates constant prices with gainloss = 1.0, resulting in:
  1. Zero volatility for infilled periods
  2. Artificially high/infinite Sharpe ratios
  3. Stock incorrectly selected as "safe" investment

SOLUTION: Modify gainloss calculation to mark infilled periods as invalid.
Set gainloss = NaN (or 0.0) for dates before first_valid_idx.
Then ensure sharpeWeightedRank_2D excludes these stocks from selection.
""")


if __name__ == "__main__":
    main()
