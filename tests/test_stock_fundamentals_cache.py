"""Tests for stock fundamentals cache."""

import pytest
import tempfile
import os
from functions.stock_fundamentals_cache import StockFundamentalsCache


class TestStockFundamentalsCache:
    """Test cases for StockFundamentalsCache."""
    
    def test_update_with_valid_symbols_filter(self):
        """Test that update_for_current_symbols filters by valid_symbols list."""
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
            cache_file = f.name
        
        try:
            cache = StockFundamentalsCache(cache_file)
            
            # Test with valid_symbols filter
            all_symbols = ['AAPL', 'INVALID', 'MSFT', 'DELISTED']
            valid_symbols = ['AAPL', 'MSFT']  # Only these are valid
            
            # Mock the actual update methods to avoid web calls
            original_get_pe = cache.get_pe_ratio
            original_get_sector = cache.get_sector_industry
            
            updated_symbols = []
            def mock_get_pe(symbol, force_refresh=False):
                updated_symbols.append(f"PE_{symbol}")
                return 15.0
            
            def mock_get_sector(symbol, force_refresh=False):
                updated_symbols.append(f"SECTOR_{symbol}")
                return "Technology", "Software"
            
            cache.get_pe_ratio = mock_get_pe
            cache.get_sector_industry = mock_get_sector
            
            # Update with valid_symbols filter
            cache.update_for_current_symbols(all_symbols, valid_symbols=valid_symbols)
            
            # Should only have updated AAPL and MSFT
            expected_updates = ['PE_AAPL', 'SECTOR_AAPL', 'PE_MSFT', 'SECTOR_MSFT']
            assert updated_symbols == expected_updates
            
            # Restore original methods
            cache.get_pe_ratio = original_get_pe
            cache.get_sector_industry = original_get_sector
            
        finally:
            if os.path.exists(cache_file):
                os.unlink(cache_file)
            if os.path.exists(cache_file + '.lock'):
                os.unlink(cache_file + '.lock')
    
    def test_update_without_valid_symbols_filter(self):
        """Test that update_for_current_symbols works without valid_symbols filter."""
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
            cache_file = f.name
        
        try:
            cache = StockFundamentalsCache(cache_file)
            
            # Test without valid_symbols filter
            all_symbols = ['AAPL', 'MSFT']
            
            # Mock the actual update methods
            updated_symbols = []
            def mock_get_pe(symbol, force_refresh=False):
                updated_symbols.append(f"PE_{symbol}")
                return 15.0
            
            def mock_get_sector(symbol, force_refresh=False):
                updated_symbols.append(f"SECTOR_{symbol}")
                return "Technology", "Software"
            
            cache.get_pe_ratio = mock_get_pe
            cache.get_sector_industry = mock_get_sector
            
            # Update without filter
            cache.update_for_current_symbols(all_symbols)
            
            # Should have updated all symbols
            expected_updates = ['PE_AAPL', 'SECTOR_AAPL', 'PE_MSFT', 'SECTOR_MSFT']
            assert updated_symbols == expected_updates
            
        finally:
            if os.path.exists(cache_file):
                os.unlink(cache_file)
            if os.path.exists(cache_file + '.lock'):
                os.unlink(cache_file + '.lock')
    
    def test_skip_cash_symbol(self):
        """Test that CASH symbol is skipped."""
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
            cache_file = f.name
        
        try:
            cache = StockFundamentalsCache(cache_file)
            
            symbols = ['AAPL', 'CASH', 'MSFT']
            
            updated_symbols = []
            def mock_get_pe(symbol, force_refresh=False):
                updated_symbols.append(f"PE_{symbol}")
                return 15.0
            
            def mock_get_sector(symbol, force_refresh=False):
                updated_symbols.append(f"SECTOR_{symbol}")
                return "Technology", "Software"
            
            cache.get_pe_ratio = mock_get_pe
            cache.get_sector_industry = mock_get_sector
            
            cache.update_for_current_symbols(symbols)
            
            # Should not have updated CASH
            assert 'PE_CASH' not in updated_symbols
            assert 'SECTOR_CASH' not in updated_symbols
            assert 'PE_AAPL' in updated_symbols
            assert 'PE_MSFT' in updated_symbols
            
        finally:
            if os.path.exists(cache_file):
                os.unlink(cache_file)
            if os.path.exists(cache_file + '.lock'):
                os.unlink(cache_file + '.lock')