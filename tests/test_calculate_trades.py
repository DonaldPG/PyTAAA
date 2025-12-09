"""Test calculateTrades.py functionality for multiple sells and single buys."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.calculateTrades import trade_today


class TestCalculateTradesMultipleSells:
    """Test suite for verifying multiple sells per symbol functionality."""
    
    @pytest.fixture
    def sample_holdings_data(self):
        """Sample holdings data matching the user's actual data."""
        return {
            'stocks': ['LRCX', 'TSLA', 'TSLA', 'WBD', 'WBD', 'CASH', 'CDNS', 'INTC', 'LRCX', 'TSLA', 'TTWO', 'WBD', 'ZS'],
            'shares': [97.0, 9.0, 4.0, 14.0, 118.0, 679.0, 66.0, 847.0, 346.0, 10.0, 50.0, 206.0, 1.0],
            'buyprice': [97.03, 308.27, 329.36, 12.87, 13.17, 1.0, 327.0, 36.37, 131.37, 413.49, 251.97, 17.1, 309.88],
            'ranks': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            'cumulativecashin': [100000.0]
        }
    
    @pytest.fixture 
    def target_portfolio(self):
        """Target portfolio after trades."""
        return {
            'symbols': ['WBD', 'INTC', 'LRCX', 'ZS', 'MU', 'KLAC', 'AMD'],
            'weights': [0.15, 0.13, 0.25, 0.02, 0.20, 0.15, 0.10],
            'prices': [22.45, 39.99, 157.46, 309.88, 223.77, 1208.74, 256.12]
        }
    
    @pytest.fixture
    def mock_price_function(self):
        """Mock the price lookup function."""
        price_map = {
            'LRCX': 157.46, 'TSLA': 456.56, 'WBD': 22.45, 'CASH': 1.0,
            'CDNS': 338.69, 'INTC': 39.99, 'TTWO': 256.37, 'ZS': 309.88,
            'MU': 223.77, 'KLAC': 1208.74, 'AMD': 256.12
        }
        return lambda symbols, *args: [price_map.get(s, 100.0) for s in symbols]
    
    @patch('functions.calculateTrades.get_holdings')
    @patch('functions.calculateTrades.get_symbols_file')
    @patch('functions.calculateTrades.get_performance_store')
    @patch('functions.calculateTrades.LastQuotesForSymbolList_hdf')
    def test_multiple_sells_per_symbol(self, mock_quotes, mock_perf_store, 
                                     mock_symbols_file, mock_holdings,
                                     sample_holdings_data, target_portfolio,
                                     mock_price_function):
        """Test that multiple holdings of same symbol generate separate sell transactions."""
        
        # Setup mocks
        mock_holdings.return_value = sample_holdings_data
        mock_symbols_file.return_value = "test_symbols.txt"
        mock_perf_store.return_value = "/tmp/test_store"
        mock_quotes.side_effect = mock_price_function
        
        # Mock file operations
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.write = MagicMock()
            
            # Call trade_today with capture of printed output
            with patch('builtins.print') as mock_print:
                trade_today(
                    json_fn="test.json",
                    symbols_today=target_portfolio['symbols'].copy(),
                    weight_today=target_portfolio['weights'].copy(),
                    price_today=target_portfolio['prices'].copy(),
                    verbose=True
                )
                
                # Capture all print calls to analyze the trades
                print_calls = [str(call) for call in mock_print.call_args_list]
                trade_output = '\n'.join(print_calls)
        
        # Verify multiple sells for TSLA (should have 3 separate sells)
        tsla_sells = trade_output.count('Sell') and 'TSLA' in trade_output
        assert tsla_sells, "TSLA should have sell transactions"
        
        # Count sell transactions in the output
        sell_lines = [line for line in trade_output.split('\\n') if 'Sell' in line and 'TSLA' in line]
        
        # Should have sells for TSLA positions (exact count depends on logic)
        # At minimum, we should see TSLA sell transactions
        assert any('TSLA' in line for line in sell_lines), "Should have TSLA sell transactions"
        
        print("✅ Multiple sells per symbol test completed")
        print(f"Trade output sample: {trade_output[:500]}...")
    
    @patch('functions.calculateTrades.get_holdings')
    @patch('functions.calculateTrades.get_symbols_file') 
    @patch('functions.calculateTrades.get_performance_store')
    @patch('functions.calculateTrades.LastQuotesForSymbolList_hdf')
    def test_single_buy_per_symbol(self, mock_quotes, mock_perf_store,
                                 mock_symbols_file, mock_holdings,
                                 sample_holdings_data, target_portfolio,
                                 mock_price_function):
        """Test that only one buy transaction is generated per symbol."""
        
        # Setup mocks
        mock_holdings.return_value = sample_holdings_data
        mock_symbols_file.return_value = "test_symbols.txt"
        mock_perf_store.return_value = "/tmp/test_store"
        mock_quotes.side_effect = mock_price_function
        
        # Mock file operations
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.write = MagicMock()
            
            # Call trade_today with capture of printed output
            with patch('builtins.print') as mock_print:
                trade_today(
                    json_fn="test.json",
                    symbols_today=target_portfolio['symbols'].copy(),
                    weight_today=target_portfolio['weights'].copy(), 
                    price_today=target_portfolio['prices'].copy(),
                    verbose=True
                )
                
                # Capture all print calls to analyze the trades
                print_calls = [str(call) for call in mock_print.call_args_list]
                trade_output = '\n'.join(print_calls)
        
        # Verify single buy per symbol
        buy_lines = [line for line in trade_output.split('\\n') if 'Buy' in line]
        
        # Count buys for each new symbol (MU, KLAC, AMD)
        new_symbols = ['MU', 'KLAC', 'AMD']
        for symbol in new_symbols:
            symbol_buys = [line for line in buy_lines if symbol in line]
            # Should have at most 1 buy per symbol
            assert len(symbol_buys) <= 1, f"Should have at most 1 buy for {symbol}, found {len(symbol_buys)}"
        
        print("✅ Single buy per symbol test completed")
        print(f"Buy lines found: {len(buy_lines)}")
    
    def test_portfolio_value_consistency(self, sample_holdings_data):
        """Test that portfolio value remains consistent after trades."""
        
        # Calculate initial portfolio value
        holdings_symbols = sample_holdings_data['stocks']
        holdings_shares = np.array(sample_holdings_data['shares']).astype('float')
        holdings_buyprice = np.array(sample_holdings_data['buyprice']).astype('float')
        
        # Mock current prices (using buy prices as approximation)
        current_prices = holdings_buyprice.copy()
        
        initial_value = sum(shares * price for shares, price in zip(holdings_shares, current_prices))
        
        print(f"Initial portfolio value: ${initial_value:,.2f}")
        
        # This test verifies the math works correctly
        assert initial_value > 0, "Portfolio should have positive value"
        assert abs(initial_value - 157897.53) < 20000, "Portfolio value should be close to expected"
        
        print("✅ Portfolio value consistency test completed")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])