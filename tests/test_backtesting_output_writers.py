"""Tests for functions.backtesting.output_writers module."""

import json
import os
import tempfile

import pytest

from functions.backtesting.output_writers import (
    get_csv_header,
    write_csv_header,
    append_csv_row,
    export_optimized_parameters,
    format_csv_row,
)


class TestGetCsvHeader:
    """Tests for get_csv_header function."""

    def test_get_csv_header_contains_key_columns(self):
        """Check that the header string contains important column names."""
        header = get_csv_header()
        for col in [
            "Portfolio Sharpe",
            "CAGR 15 Yr",
            "Sharpe 15 Yr",
            "beatBuyHoldTest",
            "param varied",
        ]:
            assert col in header, f"Expected column '{col}' in header"

    def test_get_csv_header_ends_with_newline(self):
        """Header string should end with a newline."""
        assert get_csv_header().endswith("\n")


class TestWriteCsvHeader:
    """Tests for write_csv_header function."""

    def test_write_csv_header_creates_file(self):
        """Test that write_csv_header creates the output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "results.csv")
            write_csv_header(filepath)
            assert os.path.exists(filepath)
            with open(filepath) as fid:
                content = fid.read()
            assert "Portfolio Sharpe" in content


class TestAppendCsvRow:
    """Tests for append_csv_row function."""

    def test_append_csv_row_adds_line(self):
        """Test that appending a row adds a line to the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "results.csv")
            write_csv_header(filepath)
            row = "run2501a,0,6,1,412,264,22,26,0.5,7.0,3.0,0.3,0.9,"
            row += "12345.0,8.5,3.5,32.10%,0.02988,0.15,1.2,"
            row += "2020-01-01,0.15,1.2,0.10,0.9,"
            row += "1.5,1.4,1.3,1.2,1.1,1.0,"
            row += "1.1,1.05,1.04,1.03,1.02,1.01,"
            row += "0.15,0.14,0.13,0.12,0.11,0.10,"
            row += "0.08,0.07,0.06,0.05,0.04,0.03,"
            row += "-0.05,-0.04,-0.03,-0.02,-0.01,-0.005,"
            row += "0.500,50.00%,3\n"
            append_csv_row(filepath, row)
            with open(filepath) as fid:
                lines = fid.readlines()
            # Header + 1 data row
            assert len(lines) == 2


class TestExportOptimizedParameters:
    """Tests for export_optimized_parameters function."""

    def _make_base_json(self, tmpdir: str) -> str:
        """Create a minimal base JSON config file."""
        config = {
            "Valuation": {
                "symbols_file": "/tmp/symbols.txt",
                "performance_store": tmpdir,
                "webpage": "/tmp/sp500_pine/webpage",
                "LongPeriod": 100,
            }
        }
        path = os.path.join(tmpdir, "base.json")
        with open(path, "w") as fid:
            json.dump(config, fid)
        return path

    def test_export_optimized_parameters_creates_json(self):
        """Test that the function creates a JSON file in output_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_json = self._make_base_json(tmpdir)
            optimized = {"LongPeriod": 412, "MA1": 264}
            output_path = export_optimized_parameters(
                base_json, optimized, tmpdir, "sp500_pine", "2025-6-1"
            )
            assert os.path.exists(output_path)
            assert output_path.endswith(".json")
            with open(output_path) as fid:
                data = json.load(fid)
            assert data["Valuation"]["LongPeriod"] == 412
            assert data["Valuation"]["MA1"] == 264
