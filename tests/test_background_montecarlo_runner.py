"""Unit tests for functions/background_montecarlo_runner.py.

All tests are marked with @pytest.mark.agent_runnable and rely on
mocks only -- no real quote files or HDF5 data are required.
"""

import argparse
import datetime
import importlib
import sys
import os
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


##############################################################################
# Tests: CLI argument parsing
##############################################################################

class TestParseArgs:
    """Tests for _parse_args() in background_montecarlo_runner."""

    @pytest.mark.agent_runnable
    def test_json_file_required(self):
        """_parse_args raises SystemExit when --json-file is missing."""
        from functions.background_montecarlo_runner import _parse_args

        with pytest.raises(SystemExit):
            _parse_args([])

    @pytest.mark.agent_runnable
    def test_json_file_parsed(self):
        """_parse_args correctly reads the --json-file argument."""
        from functions.background_montecarlo_runner import _parse_args

        args = _parse_args(["--json-file", "my_config.json"])
        assert args.json_file == "my_config.json"

    @pytest.mark.agent_runnable
    def test_json_file_with_path(self):
        """_parse_args accepts an absolute path for --json-file."""
        from functions.background_montecarlo_runner import _parse_args

        args = _parse_args(["--json-file", "/tmp/test/config.json"])
        assert args.json_file == "/tmp/test/config.json"


##############################################################################
# Tests: main() function
##############################################################################

class TestMain:
    """Tests for main() in background_montecarlo_runner."""

    @pytest.mark.agent_runnable
    def test_main_calls_dailyBacktest(self):
        """main() calls dailyBacktest_pctLong with the json_file argument."""
        from functions import background_montecarlo_runner

        with patch(
            "functions.dailyBacktest_pctLong.dailyBacktest_pctLong"
        ) as mock_backtest:
            background_montecarlo_runner.main("test_config.json")
            mock_backtest.assert_called_once_with("test_config.json", verbose=True)

    @pytest.mark.agent_runnable
    def test_main_prints_start_and_finish(self, capsys):
        """main() prints timestamped start and finish messages."""
        from functions import background_montecarlo_runner

        with patch(
            "functions.dailyBacktest_pctLong.dailyBacktest_pctLong"
        ):
            background_montecarlo_runner.main("cfg.json")

        out = capsys.readouterr().out
        assert "Starting Monte Carlo backtest" in out
        assert "Monte Carlo backtest complete" in out

    @pytest.mark.agent_runnable
    def test_main_handles_exception_and_exits(self):
        """main() catches exceptions from dailyBacktest_pctLong and exits."""
        from functions import background_montecarlo_runner

        with patch(
            "functions.dailyBacktest_pctLong.dailyBacktest_pctLong",
            side_effect=RuntimeError("boom"),
        ), pytest.raises(SystemExit) as exc_info:
            background_montecarlo_runner.main("cfg.json")

        assert exc_info.value.code == 1

    @pytest.mark.agent_runnable
    def test_main_prints_error_on_exception(self, capsys):
        """main() prints an error message when dailyBacktest_pctLong raises."""
        from functions import background_montecarlo_runner

        with patch(
            "functions.dailyBacktest_pctLong.dailyBacktest_pctLong",
            side_effect=ValueError("bad param"),
        ), pytest.raises(SystemExit):
            background_montecarlo_runner.main("cfg.json")

        out = capsys.readouterr().out
        assert "ERROR" in out


##############################################################################
# Tests: module-level attributes
##############################################################################

class TestModuleAttributes:
    """Tests for module-level correctness."""

    @pytest.mark.agent_runnable
    def test_module_importable(self):
        """background_montecarlo_runner is importable without side effects."""
        import functions.background_montecarlo_runner  # noqa: F401

    @pytest.mark.agent_runnable
    def test_matplotlib_backend_is_agg(self):
        """background_montecarlo_runner forces Matplotlib Agg backend."""
        import matplotlib
        # Importing the module should set/confirm Agg
        import functions.background_montecarlo_runner  # noqa: F401
        # The backend may have been set before this test runs;
        # we just verify the module's top-level use of matplotlib.use("Agg")
        # by checking that the module source contains the call.
        import inspect
        src = inspect.getsource(
            sys.modules["functions.background_montecarlo_runner"]
        )
        assert 'matplotlib.use("Agg")' in src

    @pytest.mark.agent_runnable
    def test_parse_args_is_callable(self):
        """_parse_args is a callable exported from the module."""
        from functions.background_montecarlo_runner import _parse_args
        assert callable(_parse_args)

    @pytest.mark.agent_runnable
    def test_main_is_callable(self):
        """main is a callable exported from the module."""
        from functions.background_montecarlo_runner import main
        assert callable(main)
