"""Tests for pytaaa_backtest_montecarlo CLI entry point."""

import os
import pytest
from click.testing import CliRunner

from pytaaa_backtest_montecarlo import main
from functions.backtesting.config_helpers import extract_model_identifier


class TestCliHelpText:
    """Tests for CLI help output."""

    def test_cli_help_text(self):
        """Test that --help exits cleanly and shows expected text."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--json" in result.output


class TestCliMissingArgs:
    """Tests for CLI missing argument handling."""

    def test_cli_missing_json_exits(self):
        """Test that running without --json raises a usage error."""
        runner = CliRunner()
        result = runner.invoke(main, [])
        assert result.exit_code != 0

    def test_cli_json_must_exist(self):
        """Test that a nonexistent JSON file causes an error."""
        runner = CliRunner()
        result = runner.invoke(main, ["--json", "/nonexistent/path.json"])
        assert result.exit_code != 0


class TestExtractModelIdentifierFromCliModule:
    """Integration test: import and test helper used by CLI."""

    def test_extract_model_identifier_from_cli_module(self):
        """Verify the config helper imported by the CLI module works."""
        result = extract_model_identifier(
            "/Users/user/pyTAAA_data/sp500_pine/webpage"
        )
        assert result == "sp500_pine"
