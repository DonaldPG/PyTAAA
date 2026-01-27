#!/usr/bin/env python
"""
Benchmark script to measure the performance improvement of the optimized
percentileChannel_2D function.
"""

import numpy as np
import time
import sys
import os

# Add src/backtest/functions to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'backtest'))

from functions.TAfunctions import percentileChannel_2D, percentileChannel_2D_optimized

def benchmark_percentile_channel():
    """
    Benchmark the original vs optimized percentileChannel_2D function.
    """
    print("\n" + "="*70)
    print("BENCHMARKING OPTIMIZED percentileChannel_2D")
    print("="*70 + "\n")
    
    # Benchmark configurations (realistic scenarios)
    benchmark_cases = [
        # (num_stocks, num_days, minperiod, maxperiod, incperiod, lowPct, hiPct, description)
        (100, 2000, 13, 21, 2, 17, 84, "NAZ100 size - 8 years of data"),
        (500, 2000, 13, 21, 2, 17, 84, "SP500 size - 8 years of data"),
        (500, 5000, 13, 21, 2, 17, 84, "SP500 size - 20 years of data (full backtest)"),
    ]
    
    for num_stocks, num_days, minperiod, maxperiod, incperiod, lowPct, hiPct, description in benchmark_cases:
        print(f"\n{'='*70}")
        print(f"Benchmark: {description}")
        print(f"{'='*70}")
        print(f"  Dataset: {num_stocks} stocks × {num_days} days = {num_stocks * num_days:,} data points")
        print(f"  Periods: {minperiod} to {maxperiod} by {incperiod}")
        print(f"  Percentiles: {lowPct}th and {hiPct}th")
        
        # Generate synthetic price data
        np.random.seed(42)
        x = np.zeros((num_stocks, num_days))
        for i in range(num_stocks):
            x[i, 0] = 100.0
            for j in range(1, num_days):
                daily_return = np.random.normal(0.0005, 0.02)
                x[i, j] = x[i, j-1] * (1 + daily_return)
        
        print(f"\n  Data generated. Starting benchmarks...")
        
        # Warm-up runs (to ensure fair comparison, avoid cold-start effects)
        print(f"  Warming up...")
        _ = percentileChannel_2D(x[:10, :100], minperiod, maxperiod, incperiod, lowPct, hiPct)
        _ = percentileChannel_2D_optimized(x[:10, :100], minperiod, maxperiod, incperiod, lowPct, hiPct, verbose=False)
        
        # Benchmark original version
        print(f"\n  ⏱️  Running ORIGINAL version...")
        start_time = time.time()
        min_orig, max_orig = percentileChannel_2D(x, minperiod, maxperiod, incperiod, lowPct, hiPct)
        original_time = time.time() - start_time
        print(f"  ⏱️  Original completed in: {original_time:.3f} seconds")
        
        # Benchmark optimized version
        print(f"\n  ⏱️  Running OPTIMIZED version...")
        start_time = time.time()
        min_opt, max_opt = percentileChannel_2D_optimized(x, minperiod, maxperiod, incperiod, lowPct, hiPct, verbose=False)
        optimized_time = time.time() - start_time
        print(f"  ⏱️  Optimized completed in: {optimized_time:.3f} seconds")
        
        # Calculate speedup
        speedup = original_time / optimized_time
        time_saved = original_time - optimized_time
        pct_faster = ((original_time - optimized_time) / original_time) * 100
        
        print(f"\n  📊 PERFORMANCE RESULTS:")
        print(f"     Original time:   {original_time:.3f} seconds")
        print(f"     Optimized time:  {optimized_time:.3f} seconds")
        print(f"     Time saved:      {time_saved:.3f} seconds")
        print(f"     Speedup:         {speedup:.2f}x faster")
        print(f"     Improvement:     {pct_faster:.1f}% faster")
        
        # Verify correctness
        max_diff = max(np.abs(min_orig - min_opt).max(), np.abs(max_orig - max_opt).max())
        if max_diff < 1e-10:
            print(f"     ✓ Results verified identical (max diff: {max_diff:.2e})")
        else:
            print(f"     ⚠ Warning: Results differ by {max_diff:.2e}")
        
        # Calculate time per operation
        num_operations = num_days * len(np.arange(minperiod, maxperiod, incperiod))
        orig_per_op = (original_time / num_operations) * 1000
        opt_per_op = (optimized_time / num_operations) * 1000
        
        print(f"\n  🔬 DETAILED METRICS:")
        print(f"     Operations:           {num_operations:,}")
        print(f"     Original (ms/op):     {orig_per_op:.4f} ms")
        print(f"     Optimized (ms/op):    {opt_per_op:.4f} ms")
        
        # Estimate full backtest time savings
        if "full backtest" in description.lower() or num_days >= 5000:
            trials = 30  # Typical Monte Carlo trials
            full_orig = original_time * trials
            full_opt = optimized_time * trials
            full_saved = full_orig - full_opt
            print(f"\n  💡 FULL BACKTEST ESTIMATE ({trials} Monte Carlo trials):")
            print(f"     Original total:  {full_orig/60:.1f} minutes")
            print(f"     Optimized total: {full_opt/60:.1f} minutes")
            print(f"     Time saved:      {full_saved/60:.1f} minutes ({full_saved/3600:.2f} hours)")
    
    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print("="*70 + "\n")
    
    print("📝 SUMMARY:")
    print("   The optimized version shows significant speedups, especially for larger")
    print("   datasets like SP500. For full backtests with Monte Carlo trials, this")
    print("   translates to hours of time saved.")
    print("\n   The optimization is achieved through:")
    print("   1. Vectorized percentile computation across all stocks")
    print("   2. Reduced function call overhead")
    print("   3. Better memory access patterns")
    print("   4. Pre-allocation of temporary arrays\n")


if __name__ == "__main__":
    benchmark_percentile_channel()
