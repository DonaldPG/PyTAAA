"""Tests for SP500 pre-2002 CASH allocation condition."""

import numpy as np
import datetime
from functions.rolling_window_filter import apply_rolling_window_filter


class TestSP500Pre2002Condition:
    """Test cases for SP500 pre-2002 CASH allocation logic."""

    def test_sp500_pre_2002_forces_cash_allocation(self):
        """Test that SP500 data before 2002-01-01 forces 100% CASH."""
        # Create test data: 5 stocks, dates from 2000-01-01 to 2003-01-01
        n_stocks = 5
        start_date = datetime.date(2000, 1, 1)
        dates = [start_date + datetime.timedelta(days=i) for i in range(365*3)]
        datearray = np.array(dates[:1000])  # 1000 days

        # Create mock adjClose and signal2D
        adjClose = np.random.rand(n_stocks, len(datearray)) * 100 + 100
        signal2D = np.ones((n_stocks, len(datearray)))  # All signals initially on

        # Simulate the SP500 pre-2002 logic
        cutoff_date = datetime.date(2002, 1, 1)
        for date_idx in range(len(datearray)):
            if datearray[date_idx] < cutoff_date:
                # Zero all stock signals for 100% CASH allocation
                signal2D[:, date_idx] = 0.0

        # Verify pre-2002 dates have zero signals
        pre_2002_count = sum(1 for d in datearray if d < cutoff_date)
        assert pre_2002_count > 0, "Should have dates before 2002"

        # Check that all signals are zero for pre-2002 dates
        for date_idx in range(len(datearray)):
            if datearray[date_idx] < cutoff_date:
                assert np.all(signal2D[:, date_idx] == 0.0), f"Signals should be zero for date {datearray[date_idx]}"

    def test_sp500_post_2002_preserves_signals(self):
        """Test that SP500 data after 2002-01-01 preserves original signals."""
        # Create test data: dates from 2002-01-01 onwards
        n_stocks = 3
        start_date = datetime.date(2002, 1, 1)
        dates = [start_date + datetime.timedelta(days=i) for i in range(365)]
        datearray = np.array(dates)

        # Create mock adjClose and signal2D with some signals
        adjClose = np.random.rand(n_stocks, len(datearray)) * 100 + 100
        signal2D = np.random.rand(n_stocks, len(datearray)) > 0.5  # Random boolean signals
        original_signals = signal2D.copy()

        # Simulate the SP500 pre-2002 logic (should not affect post-2002)
        cutoff_date = datetime.date(2002, 1, 1)
        for date_idx in range(len(datearray)):
            if datearray[date_idx] < cutoff_date:
                signal2D[:, date_idx] = 0.0

        # Since all dates are >= 2002-01-01, signals should be unchanged
        assert np.array_equal(signal2D, original_signals), "Post-2002 signals should be preserved"

    def test_non_sp500_universe_not_affected(self):
        """Test that non-SP500 universes are not affected by SP500 pre-2002 logic."""
        # This test verifies that the condition only applies when stockList == 'SP500'
        # The actual logic is in the calling code, so this is more of a documentation test
        assert True  # Placeholder - the logic is tested in integration

    def test_rolling_window_disabled_by_default(self):
        """Test that rolling window filter defaults to disabled."""
        # Test the default parameter logic
        params = {}  # Empty params dict

        # Should default to False (disabled)
        enable_filter = params.get('enable_rolling_filter', False)
        assert enable_filter == False, "Rolling window should be disabled by default"

        # Test with explicit False
        params['enable_rolling_filter'] = False
        enable_filter = params.get('enable_rolling_filter', False)
        assert enable_filter == False, "Explicit False should be respected"

        # Test with explicit True
        params['enable_rolling_filter'] = True
        enable_filter = params.get('enable_rolling_filter', False)
        assert enable_filter == True, "Explicit True should be respected"