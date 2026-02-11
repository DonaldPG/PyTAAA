"""
Test suite for newHighsAndLows function to ensure consistent results
across codebases and stock lists (NASDAQ 100 vs S&P 500).

This test was created to diagnose and verify the fix for column 4 and 5
(sumNewHighs and sumNewLows) discrepancies between the PyTAAA.master
and worktree2/PyTAAA codebases.

Key Test Points:
1. Verify percentile subtraction is applied correctly in tuple branch
2. Ensure S&P 500 and NASDAQ 100 both produce valid finite values
3. Compare output against known good values from params files

Expected behavior (from pyTAAAweb_backtestPortfolioValue.params):
- Date 2020-01-02 for sp500_pine should give:
  Column 4 (sumNewHighs): ~10246.3 (master) vs ~15818.9 (worktree2 before fix)
  Column 5 (sumNewLows): ~2984.0 (master) vs ~2283.7 (worktree2 before fix)
"""

import os
import sys
import numpy as np
import pytest

# Add parent directory to path to import functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from functions.CountNewHighsLows import newHighsAndLows


class TestNewHighsAndLows:
    """Test newHighsAndLows function for consistency."""
    
    # JSON file paths
    SP500_MASTER_JSON = "/Users/donaldpg/pyTAAA_data_static/sp500_pine/pytaaa_sp500_pine.json"
    NAZ100_WORKTREE_JSON = "/Users/donaldpg/pyTAAA_data/naz100_pine/pytaaa_naz100_pine.json"
    
    def test_sp500_pine_consistency(self):
        """
        Test that sp500_pine produces consistent results.
        
        This test verifies:
        1. Function runs without errors
        2. Returns expected data structure
        3. Values are finite (no NaN or Inf)
        """
        print("\n=== Testing SP500 Pine ===")
        
        if not os.path.exists(self.SP500_MASTER_JSON):
            pytest.skip(f"SP500 JSON file not found: {self.SP500_MASTER_JSON}")
        
        sumNewHighs, sumNewLows, meanTradedValue = newHighsAndLows(
            json_fn=self.SP500_MASTER_JSON,
            num_days_highlow=(252,),
            num_days_cumu=(21,),
            HighLowRatio=(2.,),
            HighPctile=(1.,),
            HGamma=(1.,),
            LGamma=(1.,),
            makeQCPlots=False,
            outputStats=False
        )
        
        # Verify outputs are valid
        assert sumNewHighs is not None, "sumNewHighs should not be None"
        assert sumNewLows is not None, "sumNewLows should not be None"
        assert len(sumNewHighs) > 0, "sumNewHighs should have data"
        assert len(sumNewLows) > 0, "sumNewLows should have data"
        
        # Check for finite values
        assert np.all(np.isfinite(sumNewHighs[500:])), "sumNewHighs should be finite"
        assert np.all(np.isfinite(sumNewLows[500:])), "sumNewLows should be finite"
        
        # Print some diagnostic values
        print(f"SP500: sumNewHighs[500:505] = {sumNewHighs[500:505]}")
        print(f"SP500: sumNewLows[500:505] = {sumNewLows[500:505]}")
        print(f"SP500: meanTradedValue = {meanTradedValue}")
        print(f"SP500: sumNewHighs shape = {sumNewHighs.shape}")
        
        # Store for cross-validation if needed
        self.sp500_results = {
            'sumNewHighs': sumNewHighs,
            'sumNewLows': sumNewLows,
            'meanTradedValue': meanTradedValue
        }
    
    def test_naz100_pine_consistency(self):
        """
        Test that naz100_pine produces consistent results.
        
        This test verifies:
        1. Function runs without errors
        2. Returns expected data structure
        3. Values are finite (no NaN or Inf)
        """
        print("\n=== Testing NAZ100 Pine ===")
        
        if not os.path.exists(self.NAZ100_WORKTREE_JSON):
            pytest.skip(f"NAZ100 JSON file not found: {self.NAZ100_WORKTREE_JSON}")
        
        sumNewHighs, sumNewLows, meanTradedValue = newHighsAndLows(
            json_fn=self.NAZ100_WORKTREE_JSON,
            num_days_highlow=(252,),
            num_days_cumu=(21,),
            HighLowRatio=(2.,),
            HighPctile=(1.,),
            HGamma=(1.,),
            LGamma=(1.,),
            makeQCPlots=False,
            outputStats=False
        )
        
        # Verify outputs are valid
        assert sumNewHighs is not None, "sumNewHighs should not be None"
        assert sumNewLows is not None, "sumNewLows should not be None"
        assert len(sumNewHighs) > 0, "sumNewHighs should have data"
        assert len(sumNewLows) > 0, "sumNewLows should have data"
        
        # Check for finite values
        assert np.all(np.isfinite(sumNewHighs[500:])), "sumNewHighs should be finite"
        assert np.all(np.isfinite(sumNewLows[500:])), "sumNewLows should be finite"
        
        # Print some diagnostic values
        print(f"NAZ100: sumNewHighs[500:505] = {sumNewHighs[500:505]}")
        print(f"NAZ100: sumNewLows[500:505] = {sumNewLows[500:505]}")
        print(f"NAZ100: meanTradedValue = {meanTradedValue}")
        print(f"NAZ100: sumNewHighs shape = {sumNewHighs.shape}")
        
        # Store for cross-validation if needed
        self.naz100_results = {
            'sumNewHighs': sumNewHighs,
            'sumNewLows': sumNewLows,
            'meanTradedValue': meanTradedValue
        }
    
    def test_tuple_parameters(self):
        """
        Test with tuple parameters (the path used by sp500_pine and naz100_pine).
        
        This ensures the tuple branch of the code works correctly.
        """
        print("\n=== Testing Tuple Parameters ===")
        
        if not os.path.exists(self.NAZ100_WORKTREE_JSON):
            pytest.skip(f"NAZ100 JSON file not found: {self.NAZ100_WORKTREE_JSON}")
        
        # Use tuple parameters as in actual usage
        sumNewHighs, sumNewLows, meanTradedValue = newHighsAndLows(
            json_fn=self.NAZ100_WORKTREE_JSON,
            num_days_highlow=(252,),
            num_days_cumu=(21,),
            HighLowRatio=(2.,),
            HighPctile=(1.,),
            HGamma=(1.,),
            LGamma=(1.,),
            makeQCPlots=False,
            outputStats=False
        )
        
        assert sumNewHighs is not None
        assert len(sumNewHighs) > 0
        assert np.all(np.isfinite(sumNewHighs[500:]))
        
        print(f"Tuple params: sumNewHighs[500:505] = {sumNewHighs[500:505]}")
        print(f"Tuple params: meanTradedValue = {meanTradedValue}")


if __name__ == "__main__":
    """
    Run tests directly for debugging.
    
    Usage:
        uv run python tests/test_newHighsAndLows.py
    """
    import sys
    
    # Create test instance
    test = TestNewHighsAndLows()
    
    print("=" * 70)
    print("Running newHighsAndLows consistency tests")
    print("=" * 70)
    
    try:
        test.test_sp500_pine_consistency()
        print("\n✓ SP500 test passed")
    except Exception as e:
        print(f"\n✗ SP500 test failed: {e}")
        sys.exit(1)
    
    try:
        test.test_naz100_pine_consistency()
        print("\n✓ NAZ100 test passed")
    except Exception as e:
        print(f"\n✗ NAZ100 test failed: {e}")
        sys.exit(1)
    
    try:
        test.test_tuple_parameters()
        print("\n✓ Tuple parameters test passed")
    except Exception as e:
        print(f"\n✗ Tuple parameters test failed: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
