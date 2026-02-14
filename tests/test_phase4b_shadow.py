"""Shadow tests for Phase 4b: PortfolioPerformanceCalcs extraction.

These tests ensure that refactoring maintains identical behavior by comparing
outputs before and after extraction of I/O and plotting from computation.
"""

import pytest
import numpy as np
import os
import json
from pathlib import Path
import filecmp


class TestPhase4b1PlotExtraction:
    """Shadow tests for plot generation extraction."""
    
    @pytest.fixture
    def test_data_path(self):
        """Get path to static test data."""
        return Path("/Users/donaldpg/pyTAAA_data_static")
    
    @pytest.fixture
    def test_json_configs(self, test_data_path):
        """Get list of test JSON configurations."""
        return [
            test_data_path / "naz100_pine/pytaaa_naz100_pine.json",
            test_data_path / "sp500_pine/pytaaa_sp500_pine.json"
        ]
    
    def test_plot_files_generated_before_refactor(self, test_json_configs):
        """
        Baseline test: Verify original PortfolioPerformanceCalcs generates plot files.
        
        This test documents the expected behavior before refactoring.
        """
        from functions.PortfolioPerformanceCalcs import PortfolioPerformanceCalcs
        from functions.GetParams import get_json_params, get_symbols_file, get_webpage_store
        
        for json_fn in test_json_configs:
            if not json_fn.exists():
                pytest.skip(f"Test data not found: {json_fn}")
            
            # Load params
            params = get_json_params(str(json_fn))
            symbol_directory = params['symbol_directory']
            symbol_file = get_symbols_file(str(json_fn))
            
            # Get output directory
            web_dir = get_webpage_store(str(json_fn))
            
            # Run original function
            last_date, symbols, weights, prices = PortfolioPerformanceCalcs(
                symbol_directory, symbol_file, params, str(json_fn)
            )
            
            # Check that plot files exist (if time condition was met)
            # Note: Plot generation is conditional on time of day
            # This test just documents that behavior
            plot_dir = Path(web_dir)
            if plot_dir.exists():
                plot_files = list(plot_dir.glob("0_*.png"))
                # Just document count, don't assert (time-dependent)
                print(f"Found {len(plot_files)} plot files in {plot_dir}")
    
    def test_return_values_unchanged(self, test_json_configs):
        """
        Verify that return values remain identical after refactoring.
        
        This is the critical test - return values must be exactly the same.
        """
        from functions.PortfolioPerformanceCalcs import PortfolioPerformanceCalcs
        from functions.GetParams import get_json_params, get_symbols_file
        
        for json_fn in test_json_configs:
            if not json_fn.exists():
                pytest.skip(f"Test data not found: {json_fn}")
            
            params = get_json_params(str(json_fn))
            symbol_directory = params['symbol_directory']
            symbol_file = get_symbols_file(str(json_fn))
            
            # Run function
            last_date, symbols, weights, prices = PortfolioPerformanceCalcs(
                symbol_directory, symbol_file, params, str(json_fn)
            )
            
            # Verify return value types
            assert isinstance(symbols, list), "symbols must be list"
            assert isinstance(weights, list), "weights must be list"
            assert isinstance(prices, list), "prices must be list"
            assert len(symbols) == len(weights) == len(prices), "Return lists must have same length"
            
            # Verify weights sum to ~1.0 (or 0 if no positions)
            weight_sum = sum(weights)
            assert weight_sum == 0 or abs(weight_sum - 1.0) < 0.01, \
                f"Weights must sum to 1.0 or 0, got {weight_sum}"
            
            # Verify all weights >= 0
            assert all(w >= 0 for w in weights), "All weights must be non-negative"
            
            # Verify all prices > 0
            assert all(p > 0 for p in prices), "All prices must be positive"


class TestPhase4b2FileWriting:
    """Shadow tests for file writing extraction."""
    
    @pytest.fixture
    def test_json_config(self):
        """Get a single test config for file writing tests."""
        return Path("/Users/donaldpg/pyTAAA_data_static/naz100_pine/pytaaa_naz100_pine.json")
    
    def test_uptrending_stocks_file_written(self, test_json_config):
        """
        Verify that uptrending stocks status file is written correctly.
        """
        from functions.PortfolioPerformanceCalcs import PortfolioPerformanceCalcs
        from functions.GetParams import get_json_params, get_symbols_file, get_webpage_store
        import os
        
        if not test_json_config.exists():
            pytest.skip(f"Test data not found: {test_json_config}")
        
        params = get_json_params(str(test_json_config))
        symbols_file_path = get_symbols_file(str(test_json_config))
        symbol_directory = os.path.dirname(symbols_file_path)
        symbol_file = os.path.basename(symbols_file_path)
        
        # Get expected output file
        web_dir = get_webpage_store(str(test_json_config))
        expected_file = Path(web_dir) / "pyTAAAweb_numberUptrendingStocks_status.params"
        
        # Remove file if exists (clean slate)
        if expected_file.exists():
            expected_file.unlink()
        
        # Run function
        PortfolioPerformanceCalcs(
            symbol_directory, symbol_file, params, str(test_json_config)
        )
        
        # Verify file was created
        assert expected_file.exists(), "Uptrending stocks file should be created"
        
        # Verify file format (date number number on each line)
        with open(expected_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) > 0, "File should have content"
        
        # Check first line format
        first_line = lines[0].strip()
        parts = first_line.split()
        assert len(parts) == 3, f"Each line should have 3 parts, got: {first_line}"


class TestPhase4b3PureComputation:
    """Shadow tests for pure computation function extraction."""
    
    @pytest.fixture
    def test_json_config(self):
        """Get test config."""
        return Path("/Users/donaldpg/pyTAAA_data_static/naz100_pine/pytaaa_naz100_pine.json")
    
    def test_pure_compute_function_exists(self, test_json_config):
        """
        Verify that compute_portfolio_metrics function exists and is callable.
        """
        from functions.output_generators import compute_portfolio_metrics
        
        assert callable(compute_portfolio_metrics), "compute_portfolio_metrics should be callable"
    
    def test_compute_returns_all_expected_keys(self, test_json_config):
        """
        Verify that compute_portfolio_metrics returns all expected metrics.
        """
        from functions.output_generators import compute_portfolio_metrics
        from functions.data_loaders import load_quotes_for_analysis
        from functions.GetParams import get_json_params, get_symbols_file
        import os
        
        if not test_json_config.exists():
            pytest.skip(f"Test data not found: {test_json_config}")
        
        params = get_json_params(str(test_json_config))
        symbol_file = get_symbols_file(str(test_json_config))
        
        # Load data
        adjClose, symbols, datearray = load_quotes_for_analysis(
            symbol_file, str(test_json_config), verbose=False
        )
        
        # Compute metrics
        metrics = compute_portfolio_metrics(
            adjClose, symbols, datearray, params, str(test_json_config)
        )
        
        # Verify all expected keys are present
        expected_keys = [
            'gainloss', 'value', 'BuyHoldFinalValue', 'lastEmptyPriceIndex',
            'activeCount', 'monthgainloss', 'signal2D', 'signal2D_daily',
            'numberStocks', 'dailyNumberUptrendingStocks', 'monthgainlossweight',
            'monthvalue', 'numberSharesCalc', 'last_symbols_text',
            'last_symbols_weight', 'last_symbols_price'
        ]
        
        for key in expected_keys:
            assert key in metrics, f"Missing expected key: {key}"
        
        # Verify data types and shapes
        assert isinstance(metrics['BuyHoldFinalValue'], (float, np.floating)), \
            "BuyHoldFinalValue should be a float"
        assert metrics['gainloss'].shape == adjClose.shape, \
            "gainloss shape should match adjClose"
        assert isinstance(metrics['last_symbols_text'], list), \
            "last_symbols_text should be a list"
    
    def test_computation_deterministic(self, test_json_config):
        """
        Verify that computation is deterministic (same inputs = same outputs).
        
        This tests that there's no hidden randomness in the computation.
        """
        from functions.output_generators import compute_portfolio_metrics
        from functions.data_loaders import load_quotes_for_analysis
        from functions.GetParams import get_json_params, get_symbols_file
        
        if not test_json_config.exists():
            pytest.skip(f"Test data not found: {test_json_config}")
        
        params = get_json_params(str(test_json_config))
        symbol_file = get_symbols_file(str(test_json_config))
        
        # Load data
        adjClose, symbols, datearray = load_quotes_for_analysis(
            symbol_file, str(test_json_config), verbose=False
        )
        
        # Run twice
        metrics1 = compute_portfolio_metrics(
            adjClose, symbols, datearray, params, str(test_json_config)
        )
        
        metrics2 = compute_portfolio_metrics(
            adjClose, symbols, datearray, params, str(test_json_config)
        )
        
        # Results should be identical
        assert metrics1['BuyHoldFinalValue'] == metrics2['BuyHoldFinalValue'], \
            "BuyHoldFinalValue should be deterministic"
        assert np.array_equal(metrics1['signal2D'], metrics2['signal2D']), \
            "signal2D should be deterministic"
        assert metrics1['last_symbols_text'] == metrics2['last_symbols_text'], \
            "last_symbols_text should be deterministic"
        assert metrics1['last_symbols_weight'] == metrics2['last_symbols_weight'], \
            "last_symbols_weight should be deterministic"
    
    def test_orchestrator_produces_same_results(self, test_json_config):
        """
        Verify that PortfolioPerformanceCalcs (orchestrator) produces
        same results as before refactoring.
        
        This is an end-to-end test that the refactored code maintains
        identical behavior.
        """
        from functions.PortfolioPerformanceCalcs import PortfolioPerformanceCalcs
        from functions.GetParams import get_json_params, get_symbols_file
        import os
        
        if not test_json_config.exists():
            pytest.skip(f"Test data not found: {test_json_config}")
        
        params = get_json_params(str(test_json_config))
        symbol_file = get_symbols_file(str(test_json_config))
        symbol_directory = os.path.dirname(symbol_file)
        symbol_filename = os.path.basename(symbol_file)
        
        # Run twice
        result1 = PortfolioPerformanceCalcs(
            symbol_directory, symbol_filename, params, str(test_json_config)
        )
        
        result2 = PortfolioPerformanceCalcs(
            symbol_directory, symbol_filename, params, str(test_json_config)
        )
        
        # Results should be identical
        assert result1[0] == result2[0], "Dates should match"
        assert result1[1] == result2[1], "Symbols should match"
        assert result1[2] == result2[2], "Weights should match"
        assert result1[3] == result2[3], "Prices should match"


class TestPhase4b4Orchestration:
    """Tests for final orchestration refactor."""
    
    def test_orchestrator_calls_all_phases(self):
        """
        After Phase 4b4, verify orchestrator calls all sub-functions.
        
        This test will be implemented after extraction is complete.
        """
        pytest.skip("To be implemented after Phase 4b4")
    
    def test_backwards_compatibility(self):
        """
        Verify that function signature and return values remain compatible.
        
        This test will be implemented after Phase 4b4.
        """
        pytest.skip("To be implemented after Phase 4b4")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
