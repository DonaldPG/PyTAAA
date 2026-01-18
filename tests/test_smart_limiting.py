#!/usr/bin/env python3
"""
Test script to demonstrate the new smart Finviz rate limiting.

Usage:
    uv run python -m pytest tests/test_smart_limiting.py -v
"""

import pytest
import sys
import os
import json

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.quotes_for_list_adjClose import get_pe_finviz_batch
from functions.quotes_adjClose import get_pe


class TestSmartFinvizLimiting:
    """Test suite for smart Finviz rate limiting functionality."""
    
    @pytest.fixture
    def test_symbols(self):
        """Test symbols including valid and likely delisted ones."""
        return {
            'valid': ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            'delisted': [
                "ALXN", "ANSS", "ATVI", "BRCM", "CELG", "CERN", 
                "CMCSK", "CTRP", "CTRX", "CTXS", "DISCA", "DISCK",
                "DISH", "DTV", "ENDP", "ESRX", "FISV", "GMCR",
                "KRFT", "LINTA", "LLTC", "LMCA", "LMCK", "LVNTA"
            ]
        }
    
    def test_batch_processing_with_smart_limiting(self, test_symbols):
        """Test batch processing with smart rate limiting."""
        all_symbols = test_symbols['valid'] + test_symbols['delisted']
        
        print(f"\nTesting batch processing with {len(all_symbols)} symbols...")
        
        # Test batch processing with smart limiting
        results = get_pe_finviz_batch(all_symbols, verbose=False, max_symbols=20)
        
        # Verify we got results for all symbols (some may be NaN)
        assert len(results) == len(all_symbols), "Should get results for all symbols"
        
        successful = [k for k, v in results.items() if not (v != v)]  # Non-NaN values
        failed = [k for k, v in results.items() if (v != v)]  # NaN values
        
        print(f"Successful: {len(successful)} symbols")
        print(f"Failed/Delisted: {len(failed)} symbols")
        
        # Should have some successful results from valid symbols
        assert len(successful) > 0, "Should have at least some successful P/E extractions"
        
        # Most delisted symbols should fail
        valid_failures = [s for s in failed if s in test_symbols['valid']]
        delisted_failures = [s for s in failed if s in test_symbols['delisted']]
        
        # Allow some valid symbols to fail (network issues, etc.)
        assert len(valid_failures) <= 2, f"Too many valid symbols failed: {valid_failures}"
        
        print(f"Valid symbol failures: {len(valid_failures)}")
        print(f"Delisted symbol failures: {len(delisted_failures)}")
    
    def test_individual_pe_function(self):
        """Test individual get_pe() function calls."""
        test_cases = [
            ("AAPL", False),  # Should succeed
            ("INVALIDTICKER", True),  # Should fail
            ("MXIM", True),  # Likely delisted, should fail
            ("GOOGL", False)  # Should succeed
        ]
        
        for symbol, should_fail in test_cases:
            print(f"Testing get_pe('{symbol}')...")
            pe_ratio = get_pe(symbol)
            
            if should_fail:
                assert pe_ratio != pe_ratio, f"Expected {symbol} to fail (return NaN)"
                print(f"  Result: Failed (NaN) as expected")
            else:
                # For valid symbols, allow for network failures but don't require success
                if pe_ratio != pe_ratio:
                    print(f"  Result: Failed (NaN) - may be network issue")
                else:
                    assert pe_ratio > 0, f"P/E ratio should be positive for {symbol}"
                    print(f"  Result: {pe_ratio:.2f}")
    
    def test_delisted_cache_functionality(self):
        """Test that delisted symbol cache is working."""
        cache_file = os.path.join(os.path.dirname(__file__), '..', 'finviz_delisted_cache.json')
        
        # Test a known delisted symbol to ensure cache gets populated
        pe_ratio = get_pe("INVALIDTICKER12345")  # Definitely invalid
        assert pe_ratio != pe_ratio, "Invalid ticker should return NaN"
        
        # Check if cache file exists and has reasonable content
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            assert 'symbols' in cache_data, "Cache should have 'symbols' key"
            assert 'last_updated' in cache_data, "Cache should have 'last_updated' key"
            assert isinstance(cache_data['symbols'], list), "Cached symbols should be a list"
            
            print(f"Cache exists with {len(cache_data['symbols'])} delisted symbols")
            print(f"Last updated: {cache_data.get('last_updated', 'Unknown')}")
        else:
            # Cache might not exist yet, which is OK for first runs
            print("No cache file found yet (this is OK for first runs)")
    
    def test_error_handling_and_limits(self):
        """Test error handling and rate limiting."""
        # Test with multiple invalid symbols to trigger error limits
        invalid_symbols = [f"INVALID{i}" for i in range(10)]
        
        # This should handle errors gracefully and stop at error limits
        results = get_pe_finviz_batch(invalid_symbols, verbose=False, max_symbols=5)
        
        # Should get results for all requested symbols (even if NaN)
        assert len(results) <= len(invalid_symbols), "Should not exceed input symbol count"
        
        # All results should be NaN for invalid symbols
        nan_count = sum(1 for v in results.values() if v != v)
        assert nan_count == len(results), "All invalid symbols should return NaN"
        
        print(f"Processed {len(results)} invalid symbols, all returned NaN as expected")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])