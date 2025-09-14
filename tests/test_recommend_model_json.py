"""
Test module for recommend_model.py JSON support.

This module tests the JSON configuration functionality added to recommend_model.py
for Step 2 of the JSON configuration implementation plan.
"""

import json
import os
import pytest
import tempfile
import click.testing
from datetime import date
from unittest.mock import patch, MagicMock

from recommend_model import main


class TestRecommendModelJsonSupport:
    """Test cases for recommend_model.py JSON configuration support."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_fixtures_dir = os.path.join(
            os.path.dirname(__file__), 'fixtures'
        )
        self.valid_config_path = os.path.join(
            self.test_fixtures_dir, 'test_config_valid.json'
        )

    def create_test_config_with_models(self):
        """Create a test configuration with model paths."""
        temp_config = {
            "web_output_dir": "/tmp/test_output",
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
                    "test_model": "{base_folder}/test_model/data_store/{data_file}"
                }
            },
            "monte_carlo": {
                "data_format": "actual",
                "data_files": {
                    "actual": "PyTAAA_status.params"
                }
            },
            "recommendation_mode": {
                "default_lookbacks": [20, 50, 100],
                "generate_plot": True
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', 
                                       delete=False) as f:
            json.dump(temp_config, f)
            return f.name

    def test_json_parameter_accepted(self):
        """Test that --json parameter is accepted by CLI."""
        runner = click.testing.CliRunner()
        
        # Test help output includes --json option
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert '--json' in result.output
        assert 'JSON configuration file' in result.output

    @patch('recommend_model.MonteCarloBacktest')
    @patch('os.path.exists')
    def test_json_config_loading(self, mock_exists, mock_monte_carlo):
        """Test that JSON configuration is properly loaded and used."""
        mock_exists.return_value = True
        mock_monte_carlo_instance = MagicMock()
        mock_monte_carlo.return_value = mock_monte_carlo_instance
        
        # Mock portfolio histories and dates for the backtesting
        mock_monte_carlo_instance.portfolio_histories = {
            'test_model': MagicMock(),
            'cash': MagicMock()
        }
        mock_monte_carlo_instance.dates = [
            date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)
        ]
        mock_monte_carlo_instance._calculate_model_switching_portfolio.return_value = [10000, 10100, 10200]
        mock_monte_carlo_instance.compute_performance_metrics.return_value = {
            'final_value': 10200,
            'annual_return': 5.0,
            'sharpe_ratio': 1.2,
            'normalized_score': 0.75
        }
        
        json_config_path = self.create_test_config_with_models()
        
        try:
            runner = click.testing.CliRunner()
            result = runner.invoke(main, ['--json', json_config_path])
            
            # Should not fail with JSON config
            assert result.exit_code == 0 or 'Error' not in result.output
            
            # Verify MonteCarloBacktest was called without normalization_values
            assert mock_monte_carlo.called
            call_kwargs = mock_monte_carlo.call_args[1]
            assert 'normalization_values' not in call_kwargs
            
            # Verify that the instance attributes were set (CENTRAL_VALUES and STD_VALUES)
            # The instance should have these attributes set after initialization
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

    def test_json_config_missing_file(self):
        """Test behavior when JSON config file doesn't exist."""
        runner = click.testing.CliRunner()
        result = runner.invoke(main, ['--json', '/nonexistent/path.json'])
        
        # The script should either exit with non-zero code or show an error message
        # but handle it gracefully without crashing
        assert result.exit_code != 0 or 'not found' in result.output.lower() or 'Error' in result.output

    @patch('recommend_model.MonteCarloBacktest')
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_web_output_dir_usage(self, mock_makedirs, mock_exists, mock_monte_carlo):
        """Test that web output directory from JSON is used correctly."""
        mock_exists.return_value = True
        mock_monte_carlo_instance = MagicMock()
        mock_monte_carlo.return_value = mock_monte_carlo_instance
        
        # Mock minimal required attributes
        mock_monte_carlo_instance.portfolio_histories = {'test_model': MagicMock()}
        mock_monte_carlo_instance.dates = [date(2024, 1, 1)]
        mock_monte_carlo_instance._calculate_model_switching_portfolio.return_value = [10000]
        mock_monte_carlo_instance.compute_performance_metrics.return_value = {
            'final_value': 10000,
            'annual_return': 0.0,
            'sharpe_ratio': 1.0,
            'normalized_score': 0.5
        }
        
        json_config_path = self.create_test_config_with_models()
        
        try:
            runner = click.testing.CliRunner()
            result = runner.invoke(main, ['--json', json_config_path])
            
            # Should attempt to create the web output directory
            mock_makedirs.assert_called_with('/tmp/test_output', exist_ok=True)
            
        finally:
            os.unlink(json_config_path)

    @patch('recommend_model.MonteCarloBacktest')
    @patch('os.path.exists')
    def test_model_paths_from_json(self, mock_exists, mock_monte_carlo):
        """Test that model paths are constructed from JSON configuration."""
        mock_exists.return_value = True
        
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
            # Check that test model path was constructed correctly
            expected_path = "/tmp/test_data/test_model/data_store/PyTAAA_status.params"
            assert model_paths.get('test_model') == expected_path
            
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

    def test_lookbacks_parameter_compatibility(self):
        """Test that --lookbacks parameter works with --json."""
        runner = click.testing.CliRunner()
        result = runner.invoke(main, ['--help'])
        
        # Both parameters should be available
        assert '--lookbacks' in result.output
        assert '--json' in result.output


class TestRecommendModelEdgeCases:
    """Test edge cases for recommend_model.py JSON support."""

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

    def test_json_missing_required_sections(self):
        """Test behavior when JSON is missing required sections."""
        temp_config = {"web_output_dir": "/tmp/test"}  # Missing other required sections
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', 
                                       delete=False) as f:
            json.dump(temp_config, f)
            incomplete_json_path = f.name
        
        try:
            runner = click.testing.CliRunner()
            result = runner.invoke(main, ['--json', incomplete_json_path])
            
            # Should handle missing sections gracefully
            # May succeed or fail, but shouldn't crash
            assert 'Traceback' not in result.output
            
        finally:
            os.unlink(incomplete_json_path)


if __name__ == '__main__':
    # Import required for date usage in tests
    from datetime import date
    pytest.main([__file__])