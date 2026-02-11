#!/usr/bin/env python
"""
Verify that percentile subtraction is working correctly in newHighsAndLows.

This script tests the critical line that subtracts the percentile from sumNewHighs.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Temporarily patch the function to add debug output
import functions.CountNewHighsLows as chl

# Save original function
original_newHighsAndLows = chl.newHighsAndLows

def patched_newHighsAndLows(*args, **kwargs):
    """Wrapper that adds debug output."""
    import numpy as np
    from functions.TAfunctions import SMS
    
    # Call original with debug flag
    json_fn = args[0] if len(args) > 0 else kwargs['json_fn']
    num_days_highlow = args[1] if len(args) > 1 else kwargs.get('num_days_highlow', 252)
    num_days_cumu = args[2] if len(args) > 2 else kwargs.get('num_days_cumu', 21)
    HighPctile = kwargs.get('HighPctile', 1.)
    
    print(f"\nDEBUG: Calling newHighsAndLows")
    print(f"  num_days_highlow type: {type(num_days_highlow)}")
    print(f"  num_days_highlow value: {num_days_highlow}")
    print(f"  HighPctile type: {type(HighPctile)}")
    print(f"  HighPctile value: {HighPctile}")
    
    # Call original
    result = original_newHighsAndLows(*args, **kwargs)
    
    sumNewHighs = result[0]
    print(f"\nDEBUG: After newHighsAndLows")
    print(f"  sumNewHighs[500:510]: {sumNewHighs[500:510]}")
    print(f"  sumNewHighs min: {sumNewHighs.min()}")
    print(f"  sumNewHighs max: {sumNewHighs.max()}")
    print(f"  sumNewHighs mean: {sumNewHighs.mean()}")
    
    # Check if values look like percentile was subtracted (should have negative values)
    has_negative = np.any(sumNewHighs < 0)
    print(f"  Has negative values: {has_negative}")
    if has_negative:
        print(f"  ✓ Percentile subtraction appears to have been applied")
    else:
        print(f"  ✗ WARNING: No negative values - percentile subtraction may not have been applied!")
    
    return result

# Monkey-patch
chl.newHighsAndLows = patched_newHighsAndLows

if __name__ == "__main__":
    from functions.CountNewHighsLows import newHighsAndLows
    
    print("="*70)
    print("Testing SP500 with actual dailyBacktest parameters")
    print("="*70)
    
    json_fn = "/Users/donaldpg/pyTAAA_data_static/sp500_pine/pytaaa_sp500_pine.json"
    
    if not os.path.exists(json_fn):
        print(f"ERROR: JSON file not found: {json_fn}")
        sys.exit(1)
    
    sumHighs, sumLows, val = newHighsAndLows(
        json_fn,
        num_days_highlow=(73, 293),
        num_days_cumu=(50, 159),
        HighLowRatio=(1.654, 2.019),
        HighPctile=(8.499, 8.952),
        HGamma=(1., 1.),
        LGamma=(1.176, 1.223),
        makeQCPlots=False,
        outputStats=False
    )
    
    print(f"\n" + "="*70)
    print(f"Final Results:")
    print(f"="*70)
    print(f"sumNewHighs shape: {sumHighs.shape}")
    print(f"sumNewHighs[5000]: {sumHighs[5000]}")
    print(f"sumNewLows[5000]: {sumLows[5000]}")
    print(f"meanTradedValue: {val}")
