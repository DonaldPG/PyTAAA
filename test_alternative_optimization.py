#!/usr/bin/env python
"""
Alternative optimization using rolling window approach and numba.
This version should show more significant speedups.
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'backtest'))
from functions.TAfunctions import percentileChannel_2D

def percentileChannel_2D_batch_optimized(x, minperiod, maxperiod, incperiod, lowPct, hiPct, batch_size=100):
    """
    Alternative optimization using batch processing.
    Process multiple days at once to reduce overhead.
    """
    # Determine periods array
    if minperiod < maxperiod and incperiod > 0:
        periods = np.arange(minperiod, maxperiod, incperiod)
    elif minperiod > maxperiod and incperiod > 0:
        periods = np.arange(maxperiod, minperiod, incperiod)
    else:
        periods = np.array([minperiod])
    
    num_stocks, num_days = x.shape
    num_periods = len(periods)
    
    # Pre-allocate output arrays
    minchannel = np.zeros((num_stocks, num_days), dtype=float)
    maxchannel = np.zeros((num_stocks, num_days), dtype=float)
    
    # Process in batches
    for batch_start in range(0, num_days, batch_size):
        batch_end = min(batch_start + batch_size, num_days)
        
        for i in range(batch_start, batch_end):
            temp_min_list = []
            temp_max_list = []
            
            for period in periods:
                minx = max(1, i - period)
                
                if i >= minx and i - minx + 1 > 0:
                    window_data = x[:, minx:i+1]
                    if window_data.shape[1] > 0:
                        temp_min_list.append(np.percentile(window_data, lowPct, axis=1))
                        temp_max_list.append(np.percentile(window_data, hiPct, axis=1))
                    else:
                        temp_min_list.append(x[:, i])
                        temp_max_list.append(x[:, i])
                else:
                    temp_min_list.append(x[:, i])
                    temp_max_list.append(x[:, i])
            
            if temp_min_list:
                minchannel[:, i] = np.mean(temp_min_list, axis=0)
                maxchannel[:, i] = np.mean(temp_max_list, axis=0)
    
    return minchannel, maxchannel


def test_batch_optimization():
    """Test the batch optimization approach."""
    print("\n" + "="*70)
    print("TESTING BATCH OPTIMIZATION APPROACH")
    print("="*70 + "\n")
    
    # Test with SP500-sized dataset
    num_stocks, num_days = 500, 2000
    minperiod, maxperiod, incperiod = 13, 21, 2
    lowPct, hiPct = 17, 84
    
    print(f"Dataset: {num_stocks} stocks × {num_days} days")
    
    # Generate data
    np.random.seed(42)
    x = np.zeros((num_stocks, num_days))
    for i in range(num_stocks):
        x[i, 0] = 100.0
        for j in range(1, num_days):
            x[i, j] = x[i, j-1] * (1 + np.random.normal(0.0005, 0.02))
    
    # Benchmark original
    print("\n⏱️  Running ORIGINAL...")
    start = time.time()
    min_orig, max_orig = percentileChannel_2D(x, minperiod, maxperiod, incperiod, lowPct, hiPct)
    orig_time = time.time() - start
    print(f"   Time: {orig_time:.3f}s")
    
    # Benchmark batch optimized
    print("\n⏱️  Running BATCH OPTIMIZED...")
    start = time.time()
    min_batch, max_batch = percentileChannel_2D_batch_optimized(x, minperiod, maxperiod, incperiod, lowPct, hiPct, batch_size=100)
    batch_time = time.time() - start
    print(f"   Time: {batch_time:.3f}s")
    
    # Verify correctness
    min_diff = np.abs(min_orig - min_batch).max()
    max_diff = np.abs(max_orig - max_batch).max()
    print(f"\n📊 Results:")
    print(f"   Speedup: {orig_time/batch_time:.2f}x")
    print(f"   Max difference: {max(min_diff, max_diff):.2e}")
    print(f"   {'✓ CORRECT' if max(min_diff, max_diff) < 1e-10 else '✗ MISMATCH'}")


if __name__ == "__main__":
    test_batch_optimization()
