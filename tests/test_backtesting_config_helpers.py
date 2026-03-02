"""Tests for functions.backtesting.config_helpers module."""

import os
import tempfile
import json
import pytest

from functions.backtesting.config_helpers import (
    extract_model_identifier,
    generate_output_filename,
    validate_configuration,
)


class TestExtractModelIdentifier:
    """Tests for extract_model_identifier function."""

    def test_extract_model_identifier_valid_path(self):
        """Test valid path extraction returns correct model ID."""
        result = extract_model_identifier("/path/to/sp500_pine/webpage")
        assert result == "sp500_pine"

    def test_extract_model_identifier_multiple_models(self):
        """Test different model name patterns are extracted correctly."""
        assert extract_model_identifier(
            "/data/sp500_pine/webpage"
        ) == "sp500_pine"
        assert extract_model_identifier(
            "/data/naz100_hma/webpage"
        ) == "naz100_hma"
        assert extract_model_identifier(
            "/data/sp500_hma/webpage"
        ) == "sp500_hma"

    def test_extract_model_identifier_short_path_raises(self):
        """Test ValueError is raised for a path with fewer than 2 components."""
        with pytest.raises(ValueError):
            extract_model_identifier("webpage")

    def test_extract_model_identifier_no_webpage_suffix_raises(self):
        """Test ValueError when last path component is not 'webpage'."""
        with pytest.raises(ValueError):
            extract_model_identifier("/path/to/sp500_pine/data")


class TestGenerateOutputFilename:
    """Tests for generate_output_filename function."""

    def test_generate_output_filename_csv(self):
        """Test basic CSV filename generation without suffix."""
        result = generate_output_filename("sp500_pine", "montecarlo", "2025-6-1")
        assert result == "sp500_pine_montecarlo_2025-6-1"

    def test_generate_output_filename_with_suffix(self):
        """Test filename generation with an appended suffix."""
        result = generate_output_filename(
            "sp500_pine", "montecarlo", "2025-6-1", "run2501a"
        )
        assert result == "sp500_pine_montecarlo_2025-6-1_run2501a"


class TestValidateConfiguration:
    """Tests for validate_configuration function."""

    def _base_params(self) -> dict:
        """Return a minimal valid params dict."""
        return {
            "symbols_file": "/tmp/symbols.txt",
            "performance_store": "/tmp/store",
            "webpage": "/tmp/sp500_pine/webpage",
        }

    def test_validate_configuration_returns_defaults(self):
        """Test that validate_configuration returns the same dict."""
        params = self._base_params()
        result = validate_configuration(params)
        assert result is params

    def test_validate_configuration_sets_trials_default(self):
        """Test backtest_monte_carlo_trials defaults to 250 when absent."""
        params = self._base_params()
        result = validate_configuration(params)
        assert result["backtest_monte_carlo_trials"] == 250

    def test_validate_configuration_preserves_existing_trials(self):
        """Test that an existing trials value is not overwritten."""
        params = self._base_params()
        params["backtest_monte_carlo_trials"] = 100
        result = validate_configuration(params)
        assert result["backtest_monte_carlo_trials"] == 100

    def test_validate_configuration_raises_for_missing_required_key(self):
        """Test KeyError raised when a required key is missing."""
        params = {
            "symbols_file": "/tmp/symbols.txt",
            "performance_store": "/tmp/store",
            # missing 'webpage'
        }
        with pytest.raises(KeyError):
            validate_configuration(params)
