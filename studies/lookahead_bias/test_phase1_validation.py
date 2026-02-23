"""
Quick validation of Phase 1 infrastructure.

Tests that all modules load without errors and basic functions work.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("PHASE 1 VALIDATION TEST")
print("=" * 70)

# Test 1: Import hdf5_utils
print("\n[Test 1] Importing hdf5_utils...")
try:
    from studies.lookahead_bias.hdf5_utils import copy_hdf5, patch_hdf5_prices
    print("  ✓ hdf5_utils imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import hdf5_utils: {e}")
    sys.exit(1)

# Test 2: Import patch_strategies
print("\n[Test 2] Importing patch_strategies...")
try:
    from studies.lookahead_bias.patch_strategies import (
        step_down, step_up, linear_down, linear_up
    )
    print("  ✓ patch_strategies imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import patch_strategies: {e}")
    sys.exit(1)

# Test 3: Import selection_runner
print("\n[Test 3] Importing selection_runner...")
try:
    from studies.lookahead_bias.selection_runner import get_ranked_stocks_for_date
    print("  ✓ selection_runner imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import selection_runner: {e}")
    sys.exit(1)

# Test 4: Test patch_strategies callables
print("\n[Test 4] Testing patch_strategies callables...")
import pandas as pd
import numpy as np

test_series = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])

try:
    # Test step_down
    fn_down = step_down(0.30)
    result_down = fn_down(test_series.copy())
    expected_down = test_series * 0.70
    assert np.allclose(result_down, expected_down), "step_down failed"
    print("  ✓ step_down(0.30) works correctly")
    
    # Test step_up
    fn_up = step_up(0.30)
    result_up = fn_up(test_series.copy())
    expected_up = test_series * 1.30
    assert np.allclose(result_up, expected_up), "step_up failed"
    print("  ✓ step_up(0.30) works correctly")
    
    # Test linear_down
    fn_lin_down = linear_down(0.01)  # Larger slope for test sensitivity
    result_lin_down = fn_lin_down(test_series.copy())
    assert len(result_lin_down) == len(test_series), "linear_down length mismatch"
    # Check that the growth factor is decreasing (even if absolute values grow)
    growth_factor = result_lin_down.values / test_series.values
    assert growth_factor[0] > growth_factor[-1], "linear_down growth factor not decreasing"
    print("  ✓ linear_down(0.01) works correctly")
    
    # Test linear_up
    fn_lin_up = linear_up(0.01)  # Larger slope for test sensitivity
    result_lin_up = fn_lin_up(test_series.copy())
    assert len(result_lin_up) == len(test_series), "linear_up length mismatch"
    # Check that the growth factor is increasing
    growth_factor = result_lin_up.values / test_series.values
    assert growth_factor[-1] > growth_factor[0], "linear_up growth factor not increasing"
    print("  ✓ linear_up(0.01) works correctly")
    
except Exception as e:
    print(f"  ✗ Patch strategies test failed: {e}")
    sys.exit(1)

# Test 5: Check directory structure
print("\n[Test 5] Checking directory structure...")
try:
    required_dirs = [
        Path(project_root) / "studies" / "lookahead_bias",
        Path(project_root) / "studies" / "lookahead_bias" / "params",
        Path(project_root) / "studies" / "lookahead_bias" / "plots",
        Path(project_root) / "studies" / "synthetic_cagr",
        Path(project_root) / "studies" / "synthetic_cagr" / "params",
        Path(project_root) / "studies" / "synthetic_cagr" / "data",
        Path(project_root) / "studies" / "synthetic_cagr" / "plots",
    ]
    
    for d in required_dirs:
        assert d.exists(), f"Directory {d} does not exist"
        print(f"  ✓ {d.relative_to(project_root)} exists")
    
except Exception as e:
    print(f"  ✗ Directory structure check failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL PHASE 1 VALIDATION TESTS PASSED ✓")
print("=" * 70)
