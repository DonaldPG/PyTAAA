"""Tests for the GetParams.py split into config_loader, config_validators,
and config_accessors.

Verifies:
- config_loader: file I/O functions work correctly with temp files
- config_validators: path-existence checks
- config_accessors: typed getters read correctly from a mocked cache
- GetParams shim: all names re-exported; no function definitions (AST)
"""

import ast
import configparser
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# config_loader tests
# ---------------------------------------------------------------------------

class TestConfigLoader:

    def test_from_config_file_returns_configparser(self, tmp_path):
        """from_config_file() should return a populated ConfigParser."""
        from functions.config_loader import from_config_file

        ini_file = tmp_path / "test.cfg"
        ini_file.write_text("[Section]\nkey = value\n")

        result = from_config_file(str(ini_file))

        assert isinstance(result, configparser.ConfigParser)
        assert result.get("Section", "key") == "value"

    def test_parse_pytaaa_status_extracts_values(self, tmp_path):
        """parse_pytaaa_status() should return parallel date and value lists."""
        from functions.config_loader import parse_pytaaa_status

        status_file = tmp_path / "PyTAAA_status.params"
        status_file.write_text(
            "cumu_value: 2026-01-01 10:00:00 12345.67 1 12400.00\n"
            "cumu_value: 2026-01-02 10:00:00 12500.00 1 12600.00\n"
        )

        dates, values = parse_pytaaa_status(str(status_file))

        assert dates == ["2026-01-01", "2026-01-02"]
        assert values == pytest.approx([12345.67, 12500.00], rel=1e-5)

    def test_parse_pytaaa_status_skips_short_lines(self, tmp_path):
        """parse_pytaaa_status() should skip malformed/short lines."""
        from functions.config_loader import parse_pytaaa_status

        status_file = tmp_path / "PyTAAA_status.params"
        status_file.write_text(
            "cumu_value: 2026-01-01 10:00:00 9999.00 1 10000.00\n"
            "\n"
            "bad line\n"
        )

        dates, values = parse_pytaaa_status(str(status_file))

        assert len(dates) == 1
        assert values[0] == pytest.approx(9999.00)

    def test_parse_pytaaa_status_raises_for_missing_file(self):
        """parse_pytaaa_status() should raise FileNotFoundError."""
        from functions.config_loader import parse_pytaaa_status

        with pytest.raises(FileNotFoundError):
            parse_pytaaa_status("/nonexistent/path/PyTAAA_status.params")

    def test_write_status_line_appends_correctly(self, tmp_path):
        """_write_status_line() should append a parseable cumu_value line."""
        from functions.config_loader import _write_status_line

        status_file = tmp_path / "PyTAAA_status.params"
        status_file.write_text("")

        _write_status_line(str(status_file), 12345.67, 1, 12400.00)

        content = status_file.read_text()
        assert "cumu_value:" in content
        assert "12345.67" in content
        assert " 1 " in content
        assert "12400.0" in content


# ---------------------------------------------------------------------------
# config_validators tests
# ---------------------------------------------------------------------------

class TestConfigValidators:

    def test_validate_model_choices_existing_path(self, tmp_path):
        """validate_model_choices() should return True for existing paths."""
        from functions.config_validators import validate_model_choices

        json_file = tmp_path / "model.json"
        json_file.write_text("{}")

        result = validate_model_choices({"mymodel": str(json_file)})

        assert result["mymodel"] is True

    def test_validate_model_choices_missing_path(self):
        """validate_model_choices() should return False for missing paths."""
        from functions.config_validators import validate_model_choices

        result = validate_model_choices({
            "mymodel": "/nonexistent/path/config.json"
        })

        assert result["mymodel"] is False

    def test_validate_model_choices_cash_model_empty_path(self):
        """Cash model (empty path) should always validate as True."""
        from functions.config_validators import validate_model_choices

        result = validate_model_choices({"cash": ""})

        assert result["cash"] is True

    def test_validate_required_keys_passes_when_all_present(self):
        """validate_required_keys() should not raise when all keys exist."""
        from functions.config_validators import validate_required_keys

        validate_required_keys(
            {"MA1": 10, "MA2": 20, "MA3": 30},
            ["MA1", "MA2", "MA3"],
        )

    def test_validate_required_keys_raises_on_missing_key(self):
        """validate_required_keys() should raise KeyError listing missing keys."""
        from functions.config_validators import validate_required_keys

        with pytest.raises(KeyError, match="MA3"):
            validate_required_keys(
                {"MA1": 10, "MA2": 20},
                ["MA1", "MA2", "MA3"],
                context="Valuation",
            )


# ---------------------------------------------------------------------------
# config_accessors tests (use mocked config_cache)
# ---------------------------------------------------------------------------

class TestConfigAccessors:

    def _make_config(self, overrides: dict = None) -> dict:
        """Build a minimal config dict that satisfies get_json_params."""
        base = {
            "Email": {"To": "a@b.com", "From": "b@c.com", "PW": "secret"},
            "Text_from_email": {
                "phoneEmail": "1234567890@txt.net",
                "send_texts": "false",
            },
            "Setup": {"runtime": "1 days", "pausetime": "1 hours"},
            "stock_server": {"quote_download_server": "yahoo"},
            "Valuation": {
                "numberStocksTraded": "5",
                "trade_cost": "0.0",
                "monthsToHold": "1",
                "LongPeriod": "252",
                "stddevThreshold": "1.0",
                "MA1": "20",
                "MA2": "50",
                "MA3": "100",
                "sma2factor": "1.0",
                "rankThresholdPct": "0.5",
                "riskDownside_min": "0.0",
                "riskDownside_max": "1.0",
                "narrowDays_min": "5",
                "narrowDays_max": "10",
                "mediumDays_min": "10",
                "mediumDays_max": "20",
                "wideDays_min": "20",
                "wideDays_max": "40",
                "uptrendSignalMethod": "HMA",
                "lowPct": "0.02",
                "hiPct": "0.98",
                "stockList": "Naz100",
                "symbols_file": "/data/symbols/Naz100_Symbols.txt",
                "performance_store": "/data/perf_store",
                "webpage": "/data/webpage",
            },
            "FTP": {
                "hostname": "ftp.example.com",
                "username": "user",
                "password": "pw",
                "remotepath": "/remote",
                "remoteIP": "1.2.3.4",
            },
        }
        if overrides:
            base.update(overrides)
        return base

    def test_get_performance_store_returns_string(self):
        """get_performance_store() should return the performance_store value."""
        from functions.config_accessors import get_performance_store

        config = self._make_config()
        with patch(
            "functions.config_accessors.config_cache"
        ) as mock_cache:
            mock_cache.get.return_value = config
            result = get_performance_store("fake.json")

        assert result == "/data/perf_store"

    def test_get_symbols_file_uses_symbols_file_key(self):
        """get_symbols_file() should prefer 'symbols_file' if set in params."""
        from functions.config_accessors import get_symbols_file

        config = self._make_config()
        with patch(
            "functions.config_accessors.config_cache"
        ) as mock_cache:
            mock_cache.get.return_value = config
            result = get_symbols_file("fake.json")

        assert result == "/data/symbols/Naz100_Symbols.txt"

    def test_get_json_params_returns_typed_dict(self):
        """get_json_params() should return a dict with typed values."""
        from functions.config_accessors import get_json_params

        config = self._make_config()
        with patch(
            "functions.config_accessors.config_cache"
        ) as mock_cache:
            mock_cache.get.return_value = config
            params = get_json_params("fake.json")

        assert isinstance(params["numberStocksTraded"], int)
        assert params["numberStocksTraded"] == 5
        assert isinstance(params["MA1"], int)
        assert params["MA1"] == 20
        assert params["MA2offset"] == 50  # MA3 - MA2 = 100 - 50


# ---------------------------------------------------------------------------
# GetParams shim tests
# ---------------------------------------------------------------------------

class TestGetParamsShim:

    _EXPECTED_NAMES = [
        "from_config_file",
        "parse_pytaaa_status",
        "validate_model_choices",
        "get_json_params",
        "get_json_ftp_params",
        "get_symbols_file",
        "get_performance_store",
        "get_webpage_store",
        "get_web_output_dir",
        "get_central_std_values",
        "get_holdings",
        "get_json_status",
        "get_status",
        "compute_long_hold_signal",
        "put_status",
        "GetIP",
        "GetEdition",
    ]

    def test_shim_re_exports_all_expected_names(self):
        """All expected names should be importable from functions.GetParams."""
        import functions.GetParams as gp

        missing = [n for n in self._EXPECTED_NAMES if not hasattr(gp, n)]
        assert missing == [], f"Missing re-exports: {missing}"

    def test_shim_has_no_function_definitions(self):
        """GetParams.py must contain zero FunctionDef or ClassDef nodes (AST guard)."""
        src_path = Path(__file__).parent.parent / "functions" / "GetParams.py"
        tree = ast.parse(src_path.read_text())

        defs = [
            node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef))
        ]
        assert defs == [], (
            f"GetParams.py should be a pure shim but contains "
            f"{len(defs)} definition(s): "
            f"{[ast.dump(d)[:60] for d in defs]}"
        )
