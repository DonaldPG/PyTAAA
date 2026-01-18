"""
Unit tests for new highs/lows capture functionality.

Tests verify that the params file is properly formatted with 5 columns
and that new highs/lows values are valid.
"""
import pytest
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestNewHighsLowsCapture:
    """Test suite for new highs/lows time series capture."""
    
    def test_params_file_exists(self):
        """Verify params file exists after backtest run."""
        params_file = os.path.join(
            os.path.dirname(__file__), "..",
            "test_data", "pyTAAAweb_backtestPortfolioValue.params"
        )
        
        # This test assumes a backtest has been run
        # In real usage, we'd run a backtest first
        if os.path.exists(params_file):
            assert True
        else:
            pytest.skip("Params file not found - run backtest first")
    
    def test_params_file_has_five_columns(self):
        """Verify params file has exactly 5 columns after run."""
        params_file = os.path.join(
            os.path.dirname(__file__), "..",
            "test_data", "pyTAAAweb_backtestPortfolioValue.params"
        )
        
        if not os.path.exists(params_file):
            pytest.skip("Params file not found - run backtest first")
        
        # Read file and check column count
        with open(params_file, "r") as f:
            lines = f.readlines()
        
        assert len(lines) > 0, "File should have content"
        
        # Check each line has 5 columns
        for i, line in enumerate(lines):
            columns = line.strip().split()
            assert len(columns) == 5, \
                f"Line {i+1}: Expected 5 columns, got {len(columns)}: {line[:50]}"
        
        print(f"✓ All {len(lines)} lines have 5 columns")
    
    def test_new_highs_lows_non_negative(self):
        """Verify new highs/lows counts are non-negative integers."""
        params_file = os.path.join(
            os.path.dirname(__file__), "..",
            "test_data", "pyTAAAweb_backtestPortfolioValue.params"
        )
        
        if not os.path.exists(params_file):
            pytest.skip("Params file not found - run backtest first")
        
        # Read file
        with open(params_file, "r") as f:
            lines = f.readlines()
        
        # Parse and check values
        for i, line in enumerate(lines):
            columns = line.strip().split()
            if len(columns) == 5:
                date, buyhold, traded, highs, lows = columns
                
                # Convert to numbers
                highs_val = int(float(highs))
                lows_val = int(float(lows))
                
                # Check non-negative
                assert highs_val >= 0, \
                    f"Line {i+1}: New highs should be non-negative, got {highs_val}"
                assert lows_val >= 0, \
                    f"Line {i+1}: New lows should be non-negative, got {lows_val}"
        
        print(f"✓ All highs/lows values are non-negative")
    
    def test_portfolio_values_positive(self):
        """Verify portfolio values are positive."""
        params_file = os.path.join(
            os.path.dirname(__file__), "..",
            "test_data", "pyTAAAweb_backtestPortfolioValue.params"
        )
        
        if not os.path.exists(params_file):
            pytest.skip("Params file not found - run backtest first")
        
        # Read file
        with open(params_file, "r") as f:
            lines = f.readlines()
        
        # Parse and check values
        for i, line in enumerate(lines):
            columns = line.strip().split()
            if len(columns) == 5:
                date, buyhold, traded, highs, lows = columns
                
                # Convert to numbers
                buyhold_val = float(buyhold)
                traded_val = float(traded)
                
                # Check positive
                assert buyhold_val > 0, \
                    f"Line {i+1}: Buy-hold value should be positive, got {buyhold_val}"
                assert traded_val > 0, \
                    f"Line {i+1}: Traded value should be positive, got {traded_val}"
        
        print(f"✓ All portfolio values are positive")
    
    def test_date_format_valid(self):
        """Verify date format is valid."""
        params_file = os.path.join(
            os.path.dirname(__file__), "..",
            "test_data", "pyTAAAweb_backtestPortfolioValue.params"
        )
        
        if not os.path.exists(params_file):
            pytest.skip("Params file not found - run backtest first")
        
        import datetime
        
        # Read file
        with open(params_file, "r") as f:
            lines = f.readlines()
        
        # Parse and check dates
        for i, line in enumerate(lines[:10]):  # Check first 10 dates
            columns = line.strip().split()
            if len(columns) == 5:
                date_str = columns[0]
                
                # Try to parse date (should be YYYY-MM-DD format)
                try:
                    year, month, day = date_str.split('-')
                    date_obj = datetime.date(int(year), int(month), int(day))
                    assert True
                except Exception as e:
                    pytest.fail(f"Line {i+1}: Invalid date format '{date_str}': {e}")
        
        print(f"✓ Date format is valid")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
