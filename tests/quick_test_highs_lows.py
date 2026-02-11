"""
Quick script to compare newHighsAndLows results between codebases.
"""

import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from functions.CountNewHighsLows import newHighsAndLows

def test_sp500():
    """Test SP500 results."""
    json_fn = "/Users/donaldpg/pyTAAA_data_static/sp500_pine/pytaaa_sp500_pine.json"
    
    if not os.path.exists(json_fn):
        print(f"SP500 JSON not found: {json_fn}")
        return None
    
    print("\n" + "="*70)
    print("Testing SP500 Pine")
    print("="*70)
    
    sumNewHighs, sumNewLows, meanTradedValue = newHighsAndLows(
        json_fn=json_fn,
        num_days_highlow=(73, 293),
        num_days_cumu=(50, 159),
        HighLowRatio=(1.654, 2.019),
        HighPctile=(8.499, 8.952),
        HGamma=(1., 1.),
        LGamma=(1.176, 1.223),
        makeQCPlots=False,
        outputStats=False
    )
    
    print(f"\nResults:")
    print(f"  sumNewHighs[500:510] = {sumNewHighs[500:510]}")
    print(f"  sumNewLows[500:510] = {sumNewLows[500:510]}")
    print(f"  meanTradedValue = {meanTradedValue}")
    print(f"  sumNewHighs shape = {sumNewHighs.shape}")
    
    return {
        'sumNewHighs': sumNewHighs,
        'sumNewLows': sumNewLows,
        'meanTradedValue': meanTradedValue
    }

def test_naz100():
    """Test NAZ100 results."""
    json_fn = "/Users/donaldpg/pyTAAA_data/naz100_pine/pytaaa_naz100_pine.json"
    
    if not os.path.exists(json_fn):
        print(f"NAZ100 JSON not found: {json_fn}")
        return None
    
    print("\n" + "="*70)
    print("Testing NAZ100 Pine")
    print("="*70)
    
    sumNewHighs, sumNewLows, meanTradedValue = newHighsAndLows(
        json_fn=json_fn,
        num_days_highlow=(73, 293),
        num_days_cumu=(50, 159),
        HighLowRatio=(1.654, 2.019),
        HighPctile=(8.499, 8.952),
        HGamma=(1., 1.),
        LGamma=(1.176, 1.223),
        makeQCPlots=False,
        outputStats=False
    )
    
    print(f"\nResults:")
    print(f"  sumNewHighs[500:510] = {sumNewHighs[500:510]}")
    print(f"  sumNewLows[500:510] = {sumNewLows[500:510]}")
    print(f"  meanTradedValue = {meanTradedValue}")
    print(f"  sumNewHighs shape = {sumNewHighs.shape}")
    
    return {
        'sumNewHighs': sumNewHighs,
        'sumNewLows': sumNewLows,
        'meanTradedValue': meanTradedValue
    }

if __name__ == "__main__":
    sp500_results = test_sp500()
    naz100_results = test_naz100()
    
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    if sp500_results:
        print("✓ SP500 test completed")
    else:
        print("✗ SP500 test skipped")
    
    if naz100_results:
        print("✓ NAZ100 test completed")
    else:
        print("✗ NAZ100 test skipped")
