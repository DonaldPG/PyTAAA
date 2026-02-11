"""
Test module for GetParams JSON extensions.

This module tests the new JSON configuration functions added to GetParams.py
for Step 1 of the JSON configuration implementation plan.
"""

import json
import os
import pytest
import tempfile
from unittest.mock import patch

from functions.GetParams import get_web_output_dir, get_central_std_values


class TestGetWebOutputDir:
    """Test cases for get_web_output_dir function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_fixtures_dir = os.path.join(
            os.path.dirname(__file__), 'fixtures'
        )
        self.valid_config_path = os.path.join(
            self.test_fixtures_dir, 'test_config_valid.json'
        )
        self.missing_fields_path = os.path.join(
            self.test_fixtures_dir, 'test_config_missing_fields.json'
        )
        self.malformed_path = os.path.join(
            self.test_fixtures_dir, 'test_config_malformed.json'
        )

    def test_get_web_output_dir_success(self):
        """Test successful retrieval of web output directory."""
        result = get_web_output_dir(self.valid_config_path)
        expected = "/Users/donaldpg/pyTAAA_data/test_output"
        assert result == expected

    def test_get_web_output_dir_missing_file(self):
        """Test behavior when JSON file doesn't exist."""
        non_existent_path = "/path/that/does/not/exist.json"
        with pytest.raises(FileNotFoundError):
            get_web_output_dir(non_existent_path)

    def test_get_web_output_dir_missing_key(self):
        """Test behavior when web_output_dir key is missing."""
        with pytest.raises(KeyError, match="'web_output_dir' key not found"):
            get_web_output_dir(self.missing_fields_path)

    def test_get_web_output_dir_malformed_json(self):
        """Test behavior with malformed JSON."""
        with pytest.raises(json.JSONDecodeError):
            get_web_output_dir(self.malformed_path)

    def test_get_web_output_dir_with_real_config(self):
        """Test with the actual project configuration file."""
        real_config_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'abacus_combined_PyTAAA_status.params.json'
        )
        
        if os.path.exists(real_config_path):
            result = get_web_output_dir(real_config_path)
            # Verify it returns a valid path string
            assert isinstance(result, str)
            assert len(result) > 0
            assert 'pyTAAA_web' in result


class TestGetCentralStdValues:
    """Test cases for get_central_std_values function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_fixtures_dir = os.path.join(
            os.path.dirname(__file__), 'fixtures'
        )
        self.valid_config_path = os.path.join(
            self.test_fixtures_dir, 'test_config_valid.json'
        )
        self.missing_fields_path = os.path.join(
            self.test_fixtures_dir, 'test_config_missing_fields.json'
        )
        self.malformed_path = os.path.join(
            self.test_fixtures_dir, 'test_config_malformed.json'
        )

    def test_get_central_std_values_success(self):
        """Test successful retrieval of normalization values."""
        result = get_central_std_values(self.valid_config_path)
        
        # Verify structure
        assert 'central_values' in result
        assert 'std_values' in result
        
        # Verify central_values content
        central_values = result['central_values']
        assert central_values['annual_return'] == 0.445
        assert central_values['sharpe_ratio'] == 1.450
        assert central_values['sortino_ratio'] == 1.400
        assert central_values['max_drawdown'] == -0.560
        assert central_values['avg_drawdown'] == -0.120
        
        # Verify std_values content
        std_values = result['std_values']
        assert std_values['annual_return'] == 0.020
        assert std_values['sharpe_ratio'] == 0.180
        assert std_values['sortino_ratio'] == 0.140
        assert std_values['max_drawdown'] == 0.060
        assert std_values['avg_drawdown'] == 0.013

    def test_get_central_std_values_missing_file(self):
        """Test behavior when JSON file doesn't exist."""
        non_existent_path = "/path/that/does/not/exist.json"
        with pytest.raises(FileNotFoundError):
            get_central_std_values(non_existent_path)

    def test_get_central_std_values_missing_model_selection(self):
        """Test behavior when model_selection section is missing."""
        # Create temporary config without model_selection
        temp_config = {"web_output_dir": "/test/path"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', 
                                       delete=False) as f:
            json.dump(temp_config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(KeyError, 
                             match="'model_selection' section not found"):
                get_central_std_values(temp_path)
        finally:
            os.unlink(temp_path)

    def test_get_central_std_values_missing_normalization(self):
        """Test behavior when normalization section is missing."""
        # Create temporary config without normalization
        temp_config = {
            "model_selection": {
                "n_lookbacks": 3
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', 
                                       delete=False) as f:
            json.dump(temp_config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(KeyError, 
                             match="'normalization' section not found"):
                get_central_std_values(temp_path)
        finally:
            os.unlink(temp_path)

    def test_get_central_std_values_missing_central_values(self):
        """Test behavior when central_values are missing."""
        # Create temporary config without central_values
        temp_config = {
            "model_selection": {
                "normalization": {
                    "std_values": {
                        "annual_return": 0.020
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', 
                                       delete=False) as f:
            json.dump(temp_config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(KeyError, 
                             match="'central_values' not found"):
                get_central_std_values(temp_path)
        finally:
            os.unlink(temp_path)

    def test_get_central_std_values_missing_std_values(self):
        """Test behavior when std_values are missing."""
        # Create temporary config without std_values
        temp_config = {
            "model_selection": {
                "normalization": {
                    "central_values": {
                        "annual_return": 0.445
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', 
                                       delete=False) as f:
            json.dump(temp_config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(KeyError, 
                             match="'std_values' not found"):
                get_central_std_values(temp_path)
        finally:
            os.unlink(temp_path)

    def test_get_central_std_values_malformed_json(self):
        """Test behavior with malformed JSON."""
        with pytest.raises(json.JSONDecodeError):
            get_central_std_values(self.malformed_path)

    def test_get_central_std_values_with_real_config(self):
        """Test with the actual project configuration file."""
        real_config_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'abacus_combined_PyTAAA_status.params.json'
        )
        
        if os.path.exists(real_config_path):
            result = get_central_std_values(real_config_path)
            
            # Verify structure
            assert 'central_values' in result
            assert 'std_values' in result
            
            # Verify both are dictionaries with numeric values
            central_values = result['central_values']
            std_values = result['std_values']
            
            assert isinstance(central_values, dict)
            assert isinstance(std_values, dict)
            
            # Check that we have the expected keys
            expected_keys = [
                'annual_return', 'sharpe_ratio', 'sortino_ratio',
                'max_drawdown', 'avg_drawdown'
            ]
            
            for key in expected_keys:
                assert key in central_values
                assert key in std_values
                assert isinstance(central_values[key], (int, float))
                assert isinstance(std_values[key], (int, float))


class TestBackwardCompatibility:
    """Test cases to ensure new functions don't break existing functionality."""

    def test_existing_functions_still_work(self):
        """Test that existing GetParams functions are not affected."""
        from functions.GetParams import (
            get_json_params, get_performance_store, get_webpage_store
        )
        
        real_config_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'abacus_combined_PyTAAA_status.params.json'
        )
        
        if os.path.exists(real_config_path):
            # Test that existing functions still work
            try:
                params = get_json_params(real_config_path)
                assert isinstance(params, dict)
                
                perf_store = get_performance_store(real_config_path)
                assert isinstance(perf_store, str)
                
                webpage_store = get_webpage_store(real_config_path)
                assert isinstance(webpage_store, str)
                
            except Exception as e:
                pytest.fail(f"Existing functions broken: {e}")

    def test_import_paths_unchanged(self):
        """Test that import paths for existing functions haven't changed."""
        # This should not raise ImportError
        from functions.GetParams import (
            get_json_params, get_performance_store, get_webpage_store,
            get_json_ftp_params, get_holdings, get_json_status
        )
        
        # New functions should also be importable
        from functions.GetParams import (
            get_web_output_dir, get_central_std_values
        )


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_empty_json_file(self):
        """Test behavior with empty JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', 
                                       delete=False) as f:
            f.write("{}")
            temp_path = f.name
        
        try:
            # Should raise KeyError for missing keys
            with pytest.raises(KeyError):
                get_web_output_dir(temp_path)
            
            with pytest.raises(KeyError):
                get_central_std_values(temp_path)
        finally:
            os.unlink(temp_path)

    def test_null_values_in_json(self):
        """Test behavior when JSON contains null values."""
        temp_config = {
            "web_output_dir": None,
            "model_selection": {
                "normalization": {
                    "central_values": None,
                    "std_values": None
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', 
                                       delete=False) as f:
            json.dump(temp_config, f)
            temp_path = f.name
        
        try:
            # Should handle null values appropriately
            result = get_web_output_dir(temp_path)
            assert result is None
            
            with pytest.raises(KeyError):
                get_central_std_values(temp_path)
        finally:
            os.unlink(temp_path)

    def test_unicode_paths(self):
        """Test behavior with unicode characters in paths."""
        temp_config = {
            "web_output_dir": "/Users/测试/pyTAAA_data/test_output",
            "model_selection": {
                "normalization": {
                    "central_values": {"annual_return": 0.445},
                    "std_values": {"annual_return": 0.020}
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', 
                                       delete=False, encoding='utf-8') as f:
            json.dump(temp_config, f, ensure_ascii=False)
            temp_path = f.name
        
        try:
            result = get_web_output_dir(temp_path)
            assert result == "/Users/测试/pyTAAA_data/test_output"
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__])