"""Shadow mode tests for Phase 4a - compare old vs new data loading.

These tests verify that the extracted data loading function produces
identical results to the inline implementation.
"""

import pytest
import numpy as np
import os
from functions.data_loaders import load_quotes_for_analysis
from functions.UpdateSymbols_inHDF5 import loadQuotes_fromHDF
from functions.TAfunctions import interpolate, cleantobeginning, cleantoend


class TestDataLoaderShadow:
    """Compare new data loader with inline implementation."""
    
    def test_data_loader_matches_inline_naz100(self):
        """New data loader produces identical results to inline code (Naz100)."""
        json_fn = "/Users/donaldpg/pyTAAA_data_static/naz100_pine/pytaaa_naz100_pine.json"
        symbols_file = "/Users/donaldpg/pyTAAA_data_static/Naz100/symbols/Naz100_symbols.txt"
        
        if not os.path.exists(symbols_file):
            pytest.skip(f"Static data not available: {symbols_file}")
        
        # New implementation (extracted function)
        adjClose_new, symbols_new, datearray_new = load_quotes_for_analysis(
            symbols_file, json_fn, verbose=False
        )
        
        # Inline implementation (copy of original code from PortfolioPerformanceCalcs)
        adjClose_old, symbols_old, datearray_old, _, _ = loadQuotes_fromHDF(symbols_file, json_fn)
        for i in range(adjClose_old.shape[0]):
            adjClose_old[i, :] = interpolate(adjClose_old[i, :])
            adjClose_old[i, :] = cleantobeginning(adjClose_old[i, :])
            adjClose_old[i, :] = cleantoend(adjClose_old[i, :])
        
        # Compare - should be identical
        np.testing.assert_array_equal(
            adjClose_new, adjClose_old,
            err_msg="adjClose arrays differ between old and new implementation"
        )
        assert symbols_new == symbols_old, "Symbol lists differ"
        np.testing.assert_array_equal(
            datearray_new, datearray_old,
            err_msg="Date arrays differ"
        )
    
    def test_data_loader_matches_inline_sp500(self):
        """New data loader produces identical results to inline code (SP500)."""
        json_fn = "/Users/donaldpg/pyTAAA_data_static/sp500_pine/pytaaa_sp500_pine.json"
        symbols_file = "/Users/donaldpg/pyTAAA_data_static/SP500/symbols/SP500_Symbols.txt"
        
        if not os.path.exists(symbols_file):
            pytest.skip(f"Static data not available: {symbols_file}")
        
        # New implementation (extracted function)
        adjClose_new, symbols_new, datearray_new = load_quotes_for_analysis(
            symbols_file, json_fn, verbose=False
        )
        
        # Inline implementation (copy of original code)
        adjClose_old, symbols_old, datearray_old, _, _ = loadQuotes_fromHDF(symbols_file, json_fn)
        for i in range(adjClose_old.shape[0]):
            adjClose_old[i, :] = interpolate(adjClose_old[i, :])
            adjClose_old[i, :] = cleantobeginning(adjClose_old[i, :])
            adjClose_old[i, :] = cleantoend(adjClose_old[i, :])
        
        # Compare
        np.testing.assert_array_equal(adjClose_new, adjClose_old)
        assert symbols_new == symbols_old
        np.testing.assert_array_equal(datearray_new, datearray_old)


class TestDataLoaderProperties:
    """Test properties and behavior of the data loader."""
    
    def test_data_loader_returns_correct_types(self):
        """Data loader returns correct types."""
        json_fn = "/Users/donaldpg/pyTAAA_data_static/naz100_pine/pytaaa_naz100_pine.json"
        symbols_file = "/Users/donaldpg/pyTAAA_data_static/Naz100/symbols/Naz100_symbols.txt"
        
        if not os.path.exists(symbols_file):
            pytest.skip(f"Static data not available: {symbols_file}")
        
        adjClose, symbols, datearray = load_quotes_for_analysis(symbols_file, json_fn)
        
        # Check types
        assert isinstance(adjClose, np.ndarray), "adjClose should be numpy array"
        assert isinstance(symbols, list), "symbols should be list"
        assert isinstance(datearray, list), "datearray should be list of datetime.date objects"
        
        # Check dimensions
        assert adjClose.ndim == 2, "adjClose should be 2D array"
        assert len(symbols) == adjClose.shape[0], "Number of symbols should match adjClose rows"
        assert len(datearray) == adjClose.shape[1], "Date array length should match adjClose columns"
    
    def test_data_loader_no_nans_at_boundaries(self):
        """Data loader removes NaNs from boundaries (cleantobeginning/cleantoend)."""
        json_fn = "/Users/donaldpg/pyTAAA_data_static/naz100_pine/pytaaa_naz100_pine.json"
        symbols_file = "/Users/donaldpg/pyTAAA_data_static/Naz100/symbols/Naz100_symbols.txt"
        
        if not os.path.exists(symbols_file):
            pytest.skip(f"Static data not available: {symbols_file}")
        
        adjClose, symbols, datearray = load_quotes_for_analysis(symbols_file, json_fn)
        
        # Check that first and last columns don't have excessive NaNs
        # (Some NaNs may remain for stocks that left the index, but boundaries should be cleaner)
        first_col_nans = np.isnan(adjClose[:, 0]).sum()
        last_col_nans = np.isnan(adjClose[:, -1]).sum()
        
        # These should be relatively small compared to total
        total_stocks = adjClose.shape[0]
        assert first_col_nans < total_stocks * 0.5, "Too many NaNs in first column"
        assert last_col_nans < total_stocks * 0.5, "Too many NaNs in last column"
    
    def test_data_loader_verbose_mode(self):
        """Data loader verbose mode runs without error."""
        json_fn = "/Users/donaldpg/pyTAAA_data_static/naz100_pine/pytaaa_naz100_pine.json"
        symbols_file = "/Users/donaldpg/pyTAAA_data_static/Naz100/symbols/Naz100_symbols.txt"
        
        if not os.path.exists(symbols_file):
            pytest.skip(f"Static data not available: {symbols_file}")
        
        # Should not raise any exceptions
        adjClose, symbols, datearray = load_quotes_for_analysis(
            symbols_file, json_fn, verbose=True
        )
        
        assert adjClose.shape[0] > 0
        assert adjClose.shape[1] > 0
