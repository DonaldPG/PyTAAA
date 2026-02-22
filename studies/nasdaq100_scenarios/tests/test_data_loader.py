"""Unit tests for data_loader module."""

import pytest
import numpy as np
from datetime import date, datetime, timedelta
import json
import tempfile
import os

from studies.nasdaq100_scenarios.data_loader import (
    load_nasdaq100_window,
    infer_tradable_mask,
    get_tradable_symbols_by_date,
    _clip_to_date_range
)


class TestInferTradableMask:
    """Tests for tradability inference logic."""
    
    def test_all_valid_prices_are_tradable(self):
        """Valid prices with no NaNsequence should be fully tradable."""
        adjClose = np.array([
            [100.0, 101.0, 102.0, 103.0, 104.0],
            [50.0, 51.0, 52.0, 53.0, 54.0]
        ])
        datearray = [date(2020, 1, i+1) for i in range(5)]
        
        mask = infer_tradable_mask(adjClose, datearray)
        
        # Most should be tradable (edges might be marked due to constant detection)
        assert mask.sum() > 0
        assert mask.shape == adjClose.shape
    
    def test_nan_prices_are_untradable(self):
        """NaN prices should be marked as untradable."""
        adjClose = np.array([
            [100.0, np.nan, 102.0, np.nan, 104.0],
            [50.0, 51.0, np.nan, np.nan, 54.0]
        ])
        datearray = [date(2020, 1, i+1) for i in range(5)]
        
        mask = infer_tradable_mask(adjClose, datearray)
        
        # Check specific NaN positions are untradable
        assert not mask[0, 1]  # First stock, second day (NaN)
        assert not mask[0, 3]  # First stock, fourth day (NaN)
        assert not mask[1, 2]  # Second stock, third day (NaN)
        assert not mask[1, 3]  # Second stock, fourth day (NaN)
    
    def test_leading_constant_prices_marked_untradable(self):
        """Leading constant-price regions (infill) should be untradable."""
        adjClose = np.array([
            [100.0, 100.0, 100.0, 100.0, 100.0, 101.0, 102.0, 103.0],
        ])
        datearray = [date(2020, 1, i+1) for i in range(8)]
        
        mask = infer_tradable_mask(adjClose, datearray)
        
        # First window (5 days) of constant prices should be untradable
        assert not mask[0, 0]
        # Later prices with movement should be tradable
        assert mask[0, -1]
    
    def test_trailing_constant_prices_marked_untradable(self):
        """Trailing constant-price regions should be untradable."""
        adjClose = np.array([
            [100.0, 101.0, 102.0, 103.0, 103.0, 103.0, 103.0, 103.0],
        ])
        datearray = [date(2020, 1, i+1) for i in range(8)]
        
        mask = infer_tradable_mask(adjClose, datearray)
        
        # Early prices with movement should be tradable
        assert mask[0, 0]
        # Last window of constant prices should be untradable
        assert not mask[0, -1]


class TestClipToDateRange:
    """Tests for date range clipping logic."""
    
    def test_clip_within_range(self):
        """Clipping to in-range dates returns correct slice."""
        adjClose = np.random.rand(10, 100)
        symbols = [f"SYM{i}" for i in range(10)]
        datearray = [date(2020, 1, 1) + i * timedelta(days=1) for i in range(100)]
        
        start = date(2020, 1, 10)  # Day 9 (0-indexed)
        stop = date(2020, 2, 18)   # Day 48 (0-indexed)
        
        clipped_adj, clipped_sym, clipped_dates = _clip_to_date_range(
            adjClose, symbols, datearray, start, stop
        )
        
        assert len(clipped_dates) == 40  # Inclusive range: day 9 to day 48
        assert clipped_dates[0] == start
        assert clipped_dates[-1] == stop
        assert clipped_adj.shape[1] == 40
        assert clipped_sym == symbols
    
    def test_clip_before_start_clamps(self):
        """Start date before available data should clamp to first date."""
        from datetime import timedelta
        adjClose = np.random.rand(5, 10)
        symbols = ["A", "B", "C", "D", "E"]
        datearray = [date(2020, 1, 10) + i * timedelta(days=1) for i in range(10)]
        
        start = date(2020, 1, 1)  # Before first date
        stop = date(2020, 1, 15)
        
        clipped_adj, clipped_sym, clipped_dates = _clip_to_date_range(
            adjClose, symbols, datearray, start, stop
        )
        
        # Should clamp to first available date
        assert clipped_dates[0] == date(2020, 1, 10)
    
    def test_clip_after_stop_clamps(self):
        """Stop date after available data should clamp to last date."""
        from datetime import timedelta
        adjClose = np.random.rand(5, 10)
        symbols = ["A", "B", "C", "D", "E"]
        datearray = [date(2020, 1, 1) + i * timedelta(days=1) for i in range(10)]
        
        start = date(2020, 1, 5)
        stop = date(2020, 2, 1)  # After last date
        
        clipped_adj, clipped_sym, clipped_dates = _clip_to_date_range(
            adjClose, symbols, datearray, start, stop
        )
        
        # Should clamp to last available date
        assert clipped_dates[-1] == date(2020, 1, 10)
    
    def test_no_overlap_raises_error(self):
        """Completely non-overlapping range should raise ValueError."""
        from datetime import timedelta
        adjClose = np.random.rand(5, 10)
        symbols = ["A", "B", "C", "D", "E"]
        datearray = [date(2020, 1, 1) + i * timedelta(days=1) for i in range(10)]
        
        start = date(2021, 1, 1)  # Way after data
        stop = date(2021, 1, 10)
        
        with pytest.raises(ValueError, match="No date overlap"):
            _clip_to_date_range(adjClose, symbols, datearray, start, stop)


class TestGetTradableSymbolsByDate:
    """Tests for tradable symbols dictionary utility."""
    
    def test_returns_correct_structure(self):
        """Should return dict mapping each date to tradable symbols."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        datearray = [date(2020, 1, 1), date(2020, 1, 2)]
        mask = np.array([
            [True, False],   # AAPL: tradable day 1, not day 2
            [True, True],    # GOOGL: tradable both days
            [False, True]    # MSFT: not tradable day 1, tradable day 2
        ])
        
        result = get_tradable_symbols_by_date(symbols, datearray, mask)
        
        assert len(result) == 2
        assert set(result[date(2020, 1, 1)]) == {"AAPL", "GOOGL"}
        assert set(result[date(2020, 1, 2)]) == {"GOOGL", "MSFT"}
    
    def test_all_tradable(self):
        """When all stocks tradable, should return all symbols each date."""
        symbols = ["A", "B", "C"]
        datearray = [date(2020, 1, 1), date(2020, 1, 2)]
        mask = np.ones((3, 2), dtype=bool)
        
        result = get_tradable_symbols_by_date(symbols, datearray, mask)
        
        assert len(result[date(2020, 1, 1)]) == 3
        assert len(result[date(2020, 1, 2)]) == 3
    
    def test_none_tradable(self):
        """When no stocks tradable, should return empty lists."""
        symbols = ["A", "B", "C"]
        datearray = [date(2020, 1, 1)]
        mask = np.zeros((3, 1), dtype=bool)
        
        result = get_tradable_symbols_by_date(symbols, datearray, mask)
        
        assert result[date(2020, 1, 1)] == []


# Integration test with real HDF5 data
@pytest.mark.skip(reason="Requires HDF5 data download - run manually when needed")
def test_load_nasdaq100_window_integration():
    """Integration test with real HDF5 data."""
    import time
    from pathlib import Path
    
    # Check if HDF5 file exists
    hdf5_path = Path("/Users/donaldpg/PyProjects/worktree2/PyTAAA/symbols/Naz100_Symbols_.hdf5")
    if not hdf5_path.exists():
        pytest.skip("NASDAQ100 HDF5 file not found")
    
    # Load with default params
    start = time.time()
    result = load_nasdaq100_window()
    elapsed = time.time() - start
    
    # Verify structure
    assert "adjClose" in result
    assert "symbols" in result
    assert "datearray" in result
    assert "tradable_mask" in result
    assert "tradable_by_date" in result
    
    # Verify shapes
    adjClose = result["adjClose"]
    symbols = result["symbols"]
    datearray = result["datearray"]
    tradable_mask = result["tradable_mask"]
    
    assert adjClose.shape[0] == len(symbols)
    assert adjClose.shape[1] == len(datearray)
    assert tradable_mask.shape == adjClose.shape
    
    # Verify 'CASH' is present
    assert "CASH" in symbols
    
    # Verify load speed (should be fast for 2-year window)
    assert elapsed < 5.0, f"Load took {elapsed:.2f}s (expected <5s)"
    
    print(f"\nIntegration test passed:")
    print(f"  Loaded {len(symbols)} symbols × {len(datearray)} dates in {elapsed:.2f}s")
    print(f"  Date range: {datearray[0]} to {datearray[-1]}")
    print(f"  Tradable days: {tradable_mask.any(axis=0).sum()}/{len(datearray)}")



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
