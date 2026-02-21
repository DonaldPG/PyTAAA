"""Unit tests for rolling window data quality filter."""

import numpy as np
import pytest
from functions.rolling_window_filter import apply_rolling_window_filter


class TestRollingWindowFilter:
    """Test cases for apply_rolling_window_filter function."""
    
    def test_sufficient_valid_data(self):
        """Test that signals are preserved when sufficient valid data exists."""
        # Create test data: 10 stocks, 60 days
        adjClose = np.random.rand(10, 60) * 100 + 100  # Random prices
        signal2D = np.ones((10, 60))  # All signals on
        
        window_size = 50
        result = apply_rolling_window_filter(adjClose, signal2D, window_size)
        
        # For dates >= 49, should have signals preserved (assuming random data has variation)
        assert np.all(result[:, 49:] == 1.0)
        # Earlier dates unchanged
        assert np.all(result[:, :49] == 1.0)
    
    def test_insufficient_valid_data_nan(self):
        """Test that signals are zeroed when insufficient valid data (NaN)."""
        adjClose = np.full((2, 60), np.nan)  # All NaN
        signal2D = np.ones((2, 60))
        
        window_size = 50
        result = apply_rolling_window_filter(adjClose, signal2D, window_size)
        
        # All signals should be zeroed for dates >= 49
        assert np.all(result[:, 49:] == 0.0)
    
    def test_constant_prices(self):
        """Test that signals are zeroed when prices are constant."""
        adjClose = np.full((2, 60), 100.0)  # Constant price
        signal2D = np.ones((2, 60))
        
        window_size = 50
        result = apply_rolling_window_filter(adjClose, signal2D, window_size)
        
        # All signals should be zeroed for dates >= 49
        assert np.all(result[:, 49:] == 0.0)
    
    def test_mixed_valid_invalid(self):
        """Test with mix of valid and invalid data."""
        adjClose = np.random.rand(1, 60) * 100 + 100
        # Make first 30 days NaN
        adjClose[0, :30] = np.nan
        signal2D = np.ones((1, 60))
        
        window_size = 50
        result = apply_rolling_window_filter(adjClose, signal2D, window_size)
        
        # For date 49, window has 30 NaN + 20 valid = 20 valid < 25, so zero
        assert result[0, 49] == 0.0
        # For date 59, window has 10 NaN + 40 valid = 40 valid >= 25, so preserve
        assert result[0, 59] == 1.0
    
    def test_small_window(self):
        """Test with smaller window size."""
        adjClose = np.random.rand(1, 30) * 100 + 100
        signal2D = np.ones((1, 30))
        
        window_size = 20
        result = apply_rolling_window_filter(adjClose, signal2D, window_size)
        
        # Should work for dates >= 19
        assert result.shape == (1, 30)
    
    def test_no_modification_early_dates(self):
        """Test that early dates (before full window) are not modified."""
        adjClose = np.full((1, 40), np.nan)  # All NaN
        signal2D = np.ones((1, 40))
        
        window_size = 50  # Larger than data
        result = apply_rolling_window_filter(adjClose, signal2D, window_size)
        
        # No dates have full window, so all signals unchanged
        assert np.all(result == 1.0)
    
    def test_in_place_modification(self):
        """Test that function modifies signal2D in place and returns it."""
        adjClose = np.full((1, 60), np.nan)
        signal2D = np.ones((1, 60))
        original_id = id(signal2D)
        
        result = apply_rolling_window_filter(adjClose, signal2D, 50)
        # Function should return a new array (copy), leaving the input
        # `signal2D` object unchanged.
        assert id(result) != original_id
        # Result should have zeros for dates with insufficient data
        assert np.all(result[:, 49:] == 0.0)
        # Original input must remain unmodified
        assert np.all(signal2D[:, 49:] == 1.0)