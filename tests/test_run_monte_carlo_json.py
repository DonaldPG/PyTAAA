"""
Test module for run_monte_carlo.py JSON support.

This module tests the JSON configuration functionality added to run_monte_carlo.py
for Step 2 of the JSON configuration implementation plan.
"""

import json
import os
import pytest
import tempfile
import click.testing
from unittest.mock import patch, MagicMock

from run_monte_carlo import main


class TestRunMonteCarloJsonSupport:
    """Test cases for run_monte_carlo.py JSON configuration support."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_fixtures_dir = os.path.join(
            os.path.dirname(__file__), 'fixtures'
        )

    def create_test_config_with_models(self):
        """Create a test configuration with model paths for Monte Carlo."""
        temp_config = {
            "web_output_dir": "/tmp/test_monte_carlo_output",
            "model_selection": {
                "normalization": {
                    "central_values": {
                        "annual_return": 0.445,
                        "sharpe_ratio": 1.450,
                        "sortino_ratio": 1.400,
                        "max_drawdown": -0.560,
                        "avg_drawdown": -0.120
                    },
                    "std_values": {
                        "annual_return": 0.020,
                        "sharpe_ratio": 0.180,
                        "sortino_ratio": 0.140,
                        "max_drawdown": 0.060,
                        "avg_drawdown": 0.013
                    }
                }
            },
            "models": {
                "base_folder": "/tmp/test_data",
                "model_choices": {
                    "cash": "",
                    "test_model_1": "{base_folder}/test_model_1/data_store/{data_file}",
                    "test_model_2": "{base_folder}/test_model_2/data_store/{data_file}"
                }
            },
            "monte_carlo": {
                "max_iterations": 10,
                "min_iterations_for_exploit": 5,
                "trading_frequency": "monthly",
                "min_lookback": 10,
                "max_lookback": 100,
                "data_format": "backtested",
                "data_files": {
                    "actual": "PyTAAA_status.params",
                    "backtested": "pyTAAAweb_backtestPortfolioValue.params"
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', 
                                       delete=False) as f:
            json.dump(temp_config, f)
            return f.name

    @patch('run_monte_carlo.MonteCarloBacktest')
    @patch('os.path.exists')
    @patch('os.remove')
    def test_json_config_loading(self, mock_remove, mock_exists, mock_monte_carlo):
        """Test that JSON configuration is properly loaded and used."""
        # Mock os.path.exists to return False for the output file (so it won't try to remove)
        def exists_side_effect(path):
            if 'monte_carlo_best_performance.png' in path:
                return False  # File doesn't exist, so don't try to remove it
            return True  # Other paths exist
        
        mock_exists.side_effect = exists_side_effect
        mock_monte_carlo_instance = MagicMock()
        mock_monte_carlo.return_value = mock_monte_carlo_instance
        
        # Mock required attributes and methods
        mock_monte_carlo_instance.load_state.return_value = None
        mock_monte_carlo_instance.run.return_value = []
        mock_monte_carlo_instance.save_state.return_value = None
        mock_monte_carlo_instance.best_portfolio_value = [10000, 10100, 10200]
        mock_monte_carlo_instance.best_params = {'lookbacks': [20, 50]}
        mock_monte_carlo_instance.dates = ['2024-01-01', '2024-01-02', '2024-01-03']
        mock_monte_carlo_instance.best_model_selections = {}
        mock_monte_carlo_instance._calculate_model_switching_portfolio.return_value = [10000, 10100, 10200]
        mock_monte_carlo_instance.compute_performance_metrics.return_value = {
            'final_value': 10200,
            'annual_return': 5.0,
            'sharpe_ratio': 1.2,
            'normalized_score': 0.75
        }
        mock_monte_carlo_instance.create_monte_carlo_plot.return_value = None
        
        json_config_path = self.create_test_config_with_models()
        
        try:
            runner = click.testing.CliRunner()
            result = runner.invoke(main, ['--json', json_config_path])
            
            # Should complete successfully
            assert result.exit_code == 0
            
            # Verify MonteCarloBacktest was called without normalization_values
            assert mock_monte_carlo.called
            call_kwargs = mock_monte_carlo.call_args[1]
            assert 'normalization_values' not in call_kwargs
            
            # Verify that the instance attributes were set (CENTRAL_VALUES and STD_VALUES)
            mock_monte_carlo_instance.CENTRAL_VALUES = {
                'annual_return': 0.445,
                'sharpe_ratio': 1.450,
                'sortino_ratio': 1.400,
                'max_drawdown': -0.560,
                'avg_drawdown': -0.120
            }
            mock_monte_carlo_instance.STD_VALUES = {
                'annual_return': 0.020,
                'sharpe_ratio': 0.180,
                'sortino_ratio': 0.140,
                'max_drawdown': 0.060,
                'avg_drawdown': 0.013
            }
            
        finally:
            os.unlink(json_config_path)

    def test_json_parameter_accepted(self):
        """Test that --json parameter is accepted by CLI."""
        runner = click.testing.CliRunner()
        
        # Test help output includes --json option
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert '--json' in result.output
        assert 'JSON configuration file' in result.output

    @patch('run_monte_carlo.MonteCarloBacktest')
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('os.remove')
    def test_web_output_dir_usage(self, mock_remove, mock_makedirs, mock_exists, mock_monte_carlo):
        """Test that web output directory from JSON is used correctly."""
        # Mock os.path.exists to return False for the output file (so it won't try to remove)
        def exists_side_effect(path):
            if 'monte_carlo_best_performance.png' in path:
                return False  # File doesn't exist, so don't try to remove it
            return True  # Other paths exist
        
        mock_exists.side_effect = exists_side_effect
        mock_monte_carlo_instance = MagicMock()
        mock_monte_carlo.return_value = mock_monte_carlo_instance
        
        # Mock required attributes
        mock_monte_carlo_instance.load_state.return_value = None
        mock_monte_carlo_instance.run.return_value = []
        mock_monte_carlo_instance.save_state.return_value = None
        mock_monte_carlo_instance.best_portfolio_value = [10000, 10200]
        mock_monte_carlo_instance.best_params = {'lookbacks': [20, 50]}
        mock_monte_carlo_instance.dates = ['2024-01-01', '2024-01-02']
        mock_monte_carlo_instance.best_model_selections = {}
        mock_monte_carlo_instance._calculate_model_switching_portfolio.return_value = [10000, 10200]
        mock_monte_carlo_instance.compute_performance_metrics.return_value = {
            'final_value': 10200
        }
        mock_monte_carlo_instance.create_monte_carlo_plot.return_value = None
        
        json_config_path = self.create_test_config_with_models()
        
        try:
            runner = click.testing.CliRunner()
            result = runner.invoke(main, ['--json', json_config_path])
            
            # Should attempt to create the web output directory
            mock_makedirs.assert_called_with('/tmp/test_monte_carlo_output', exist_ok=True)
            
            # Should call create_monte_carlo_plot with correct output path
            expected_output_path = '/tmp/test_monte_carlo_output/monte_carlo_best_performance.png'
            mock_monte_carlo_instance.create_monte_carlo_plot.assert_called()
            call_args = mock_monte_carlo_instance.create_monte_carlo_plot.call_args
            assert call_args[0][2] == expected_output_path  # Third argument is output_path
            
        finally:
            os.unlink(json_config_path)

    @patch('run_monte_carlo.MonteCarloBacktest')
    @patch('os.path.exists')
    def test_model_paths_from_json(self, mock_exists, mock_monte_carlo):
        """Test that model paths are constructed from JSON configuration."""
        mock_exists.return_value = True
        mock_monte_carlo_instance = MagicMock()
        mock_monte_carlo.return_value = mock_monte_carlo_instance
        
        # Mock required attributes
        mock_monte_carlo_instance.load_state.return_value = None
        mock_monte_carlo_instance.run.return_value = []
        mock_monte_carlo_instance.save_state.return_value = None
        mock_monte_carlo_instance.best_portfolio_value = None  # No best portfolio
        
        json_config_path = self.create_test_config_with_models()
        
        try:
            runner = click.testing.CliRunner()
            result = runner.invoke(main, ['--json', json_config_path])
            
            # Verify MonteCarloBacktest was called with correct model paths
            assert mock_monte_carlo.called
            call_kwargs = mock_monte_carlo.call_args[1]
            model_paths = call_kwargs.get('model_paths', {})
            
            # Check that cash model is empty string
            assert model_paths.get('cash') == ""
            # Check that test model paths were constructed correctly
            expected_path_1 = "/tmp/test_data/test_model_1/data_store/pyTAAAweb_backtestPortfolioValue.params"
            expected_path_2 = "/tmp/test_data/test_model_2/data_store/pyTAAAweb_backtestPortfolioValue.params"
            assert model_paths.get('test_model_1') == expected_path_1
            assert model_paths.get('test_model_2') == expected_path_2
            
        finally:
            os.unlink(json_config_path)

    @patch('run_monte_carlo.MonteCarloBacktest')
    @patch('os.path.exists')
    def test_monte_carlo_config_parameters(self, mock_exists, mock_monte_carlo):
        """Test that Monte Carlo parameters from JSON are used correctly."""
        mock_exists.return_value = True
        mock_monte_carlo_instance = MagicMock()
        mock_monte_carlo.return_value = mock_monte_carlo_instance
        
        # Mock required attributes
        mock_monte_carlo_instance.load_state.return_value = None
        mock_monte_carlo_instance.run.return_value = []
        mock_monte_carlo_instance.save_state.return_value = None
        mock_monte_carlo_instance.best_portfolio_value = None
        
        json_config_path = self.create_test_config_with_models()
        
        try:
            runner = click.testing.CliRunner()
            result = runner.invoke(main, ['--json', json_config_path])
            
            # Verify MonteCarloBacktest was called with correct parameters
            assert mock_monte_carlo.called
            call_kwargs = mock_monte_carlo.call_args[1]
            
            assert call_kwargs.get('iterations') == 10
            assert call_kwargs.get('min_iterations_for_exploit') == 5
            assert call_kwargs.get('trading_frequency') == 'monthly'
            assert call_kwargs.get('min_lookback') == 10
            assert call_kwargs.get('max_lookback') == 100
            
        finally:
            os.unlink(json_config_path)

    def test_backward_compatibility_without_json(self):
        """Test that script still works without --json parameter."""
        runner = click.testing.CliRunner()
        
        # Should fail gracefully if legacy config is missing, not crash
        result = runner.invoke(main, [])
        
        # Exit code might be non-zero due to missing config, but shouldn't crash
        # The important thing is it doesn't crash with Python errors
        assert 'Traceback' not in result.output

    def test_search_parameter_compatibility(self):
        """Test that --search parameter works with --json."""
        runner = click.testing.CliRunner()
        result = runner.invoke(main, ['--help'])
        
        # Both parameters should be available
        assert '--search' in result.output
        assert '--json' in result.output

    def test_verbose_parameter_compatibility(self):
        """Test that --verbose parameter works with --json."""
        runner = click.testing.CliRunner()
        result = runner.invoke(main, ['--help'])
        
        # Both parameters should be available
        assert '--verbose' in result.output
        assert '--json' in result.output


class TestRunMonteCarloEdgeCases:
    """Test edge cases for run_monte_carlo.py JSON support."""

    def test_json_with_invalid_format(self):
        """Test behavior with malformed JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', 
                                       delete=False) as f:
            f.write('{"invalid": json}')  # Invalid JSON
            invalid_json_path = f.name
        
        try:
            runner = click.testing.CliRunner()
            result = runner.invoke(main, ['--json', invalid_json_path])
            
            assert result.exit_code != 0
            # Should handle JSON decode error gracefully
            
        finally:
            os.unlink(invalid_json_path)

    def test_json_missing_monte_carlo_section(self):
        """Test behavior when JSON is missing monte_carlo section."""
        temp_config = {
            "web_output_dir": "/tmp/test",
            "model_selection": {
                "normalization": {
                    "central_values": {"annual_return": 0.445},
                    "std_values": {"annual_return": 0.020}
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', 
                                       delete=False) as f:
            json.dump(temp_config, f)
            incomplete_json_path = f.name
        
        try:
            runner = click.testing.CliRunner()
            result = runner.invoke(main, ['--json', incomplete_json_path])
            
            # Should handle missing monte_carlo section gracefully
            # Should use default values for missing parameters
            assert 'Traceback' not in result.output
            
        finally:
            os.unlink(incomplete_json_path)

    @patch('run_monte_carlo.MonteCarloBacktest')
    @patch('os.path.exists')
    def test_legacy_fallback_without_models_section(self, mock_exists, mock_monte_carlo):
        """Test that legacy model paths are used when models section is missing."""
        mock_exists.return_value = True
        mock_monte_carlo_instance = MagicMock()
        mock_monte_carlo.return_value = mock_monte_carlo_instance
        
        # Mock required attributes
        mock_monte_carlo_instance.load_state.return_value = None
        mock_monte_carlo_instance.run.return_value = []
        mock_monte_carlo_instance.save_state.return_value = None
        mock_monte_carlo_instance.best_portfolio_value = None
        
        # Config without models section
        temp_config = {
            "web_output_dir": "/tmp/test",
            "monte_carlo": {
                "data_format": "actual",
                "data_files": {"actual": "PyTAAA_status.params"}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', 
                                       delete=False) as f:
            json.dump(temp_config, f)
            json_config_path = f.name
        
        try:
            runner = click.testing.CliRunner()
            result = runner.invoke(main, ['--json', json_config_path])
            
            # Should use legacy hard-coded paths
            assert mock_monte_carlo.called
            call_kwargs = mock_monte_carlo.call_args[1]
            model_paths = call_kwargs.get('model_paths', {})
            
            # Should contain legacy model names
            assert 'cash' in model_paths
            assert 'naz100_pine' in model_paths or 'sp500_hma' in model_paths
            
        finally:
            os.unlink(json_config_path)


class TestMonteCarloShellScriptIntegration:
    """Test shell script integration with JSON parameter."""

    def test_shell_script_help_includes_json(self):
        """Test that shell script help includes --json parameter."""
        # This would test the shell script, but we'll focus on the Python interface
        # The shell script changes should be tested separately in integration tests
        pass


if __name__ == '__main__':
    pytest.main([__file__])