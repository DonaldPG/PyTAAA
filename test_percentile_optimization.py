#!/usr/bin/env python
"""
Test script to verify the optimized percentileChannel_2D function produces
identical results to the original implementation.
"""

import numpy as np
import sys
import os

# Add src/backtest/functions to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'backtest'))

from functions.TAfunctions import percentileChannel_2D, percentileChannel_2D_optimized

def test_percentile_channel_correctness():
    """
    Test that optimized version produces identical results to original.
    """
    print("\n" + "="*70)
    print("TESTING CORRECTNESS OF OPTIMIZED percentileChannel_2D")
    print("="*70 + "\n")
    
    # Test parameters (typical SP500 pine values)
    test_cases = [
        # (num_stocks, num_days, minperiod, maxperiod, incperiod, lowPct, hiPct, description)
        (10, 100, 13, 21, 2, 17, 84, "Small dataset (quick test)"),
        (50, 500, 13, 21, 2, 17, 84, "Medium dataset (typical NAZ100 subset)"),
        (100, 1000, 13, 21, 2, 17, 84, "Large dataset (simulating full NAZ100)"),
    ]
    
    all_passed = True
    
    for num_stocks, num_days, minperiod, maxperiod, incperiod, lowPct, hiPct, description in test_cases:
        print(f"\nTest: {description}")
        print(f"  Stocks: {num_stocks}, Days: {num_days}")
        print(f"  Periods: {minperiod} to {maxperiod} by {incperiod}")
        print(f"  Percentiles: {lowPct}th and {hiPct}th")
        
        # Generate synthetic price data (realistic stock-like behavior)
        np.random.seed(42)  # For reproducibility
        x = np.zeros((num_stocks, num_days))
        for i in range(num_stocks):
            # Start at $100, add random walk with drift
            x[i, 0] = 100.0
            for j in range(1, num_days):
                daily_return = np.random.normal(0.0005, 0.02)  # ~0.05% drift, 2% volatility
                x[i, j] = x[i, j-1] * (1 + daily_return)
        
        # Run both versions
        print("  Running original version...")
        min_orig, max_orig = percentileChannel_2D(x, minperiod, maxperiod, incperiod, lowPct, hiPct)
        
        print("  Running optimized version...")
        min_opt, max_opt = percentileChannel_2D_optimized(x, minperiod, maxperiod, incperiod, lowPct, hiPct, verbose=False)
        
        # Compare results
        min_diff = np.abs(min_orig - min_opt)
        max_diff = np.abs(max_orig - max_opt)
        
        min_max_diff = min_diff.max()
        max_max_diff = max_diff.max()
        min_mean_diff = min_diff.mean()
        max_mean_diff = max_diff.mean()
        
        # Check if differences are within floating point precision
        tolerance = 1e-10
        min_passed = min_max_diff < tolerance
        max_passed = max_max_diff < tolerance
        
        print(f"\n  Results comparison:")
        print(f"    Min channel - Max diff: {min_max_diff:.2e}, Mean diff: {min_mean_diff:.2e}")
        print(f"    Max channel - Max diff: {max_max_diff:.2e}, Mean diff: {max_mean_diff:.2e}")
        print(f"    Tolerance: {tolerance:.2e}")
        
        if min_passed and max_passed:
            print(f"  ✓ PASSED - Results are identical within tolerance")
        else:
            print(f"  ✗ FAILED - Results differ beyond tolerance!")
            all_passed = False
            
            # Show some sample differences for debugging
            if not min_passed:
                worst_stock, worst_day = np.unravel_index(min_diff.argmax(), min_diff.shape)
                print(f"    Worst min diff at stock {worst_stock}, day {worst_day}:")
                print(f"      Original: {min_orig[worst_stock, worst_day]:.10f}")
                print(f"      Optimized: {min_opt[worst_stock, worst_day]:.10f}")
            if not max_passed:
                worst_stock, worst_day = np.unravel_index(max_diff.argmax(), max_diff.shape)
                print(f"    Worst max diff at stock {worst_stock}, day {worst_day}:")
                print(f"      Original: {max_orig[worst_stock, worst_day]:.10f}")
                print(f"      Optimized: {max_opt[worst_stock, worst_day]:.10f}")
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Optimized version is correct!")
        print("="*70 + "\n")
        return True
    else:
        print("✗ SOME TESTS FAILED - Check implementation!")
        print("="*70 + "\n")
        return False


if __name__ == "__main__":
    success = test_percentile_channel_correctness()
    sys.exit(0 if success else 1)
