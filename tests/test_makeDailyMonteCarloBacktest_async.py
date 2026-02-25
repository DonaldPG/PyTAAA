"""Unit tests for the async mode additions to makeDailyMonteCarloBacktest().

All tests are marked with @pytest.mark.agent_runnable and rely on mocks
only - no real HDF5 data or quote files required.
"""

import datetime
import inspect
import os
import sys
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


##############################################################################
# Tests: function signature
##############################################################################

class TestMakeDailyMonteCarloBacktestSignature:
    """Verify the updated function signature."""

    @pytest.mark.agent_runnable
    def test_async_mode_parameter_exists(self):
        """makeDailyMonteCarloBacktest has an async_mode parameter."""
        from functions.MakeValuePlot import makeDailyMonteCarloBacktest
        sig = inspect.signature(makeDailyMonteCarloBacktest)
        assert "async_mode" in sig.parameters

    @pytest.mark.agent_runnable
    def test_async_mode_default_is_false(self):
        """async_mode defaults to False for backward compatibility."""
        from functions.MakeValuePlot import makeDailyMonteCarloBacktest
        sig = inspect.signature(makeDailyMonteCarloBacktest)
        assert sig.parameters["async_mode"].default is False

    @pytest.mark.agent_runnable
    def test_json_fn_is_first_parameter(self):
        """json_fn is the first positional parameter."""
        from functions.MakeValuePlot import makeDailyMonteCarloBacktest
        sig = inspect.signature(makeDailyMonteCarloBacktest)
        params = list(sig.parameters.keys())
        assert params[0] == "json_fn"


##############################################################################
# Tests: _spawn_background_montecarlo
##############################################################################

class TestSpawnBackgroundMontecarlo:
    """Tests for the subprocess-spawning helper."""

    @pytest.mark.agent_runnable
    def test_spawn_calls_popen(self, tmp_path):
        """_spawn_background_montecarlo calls subprocess.Popen."""
        from functions.MakeValuePlot import _spawn_background_montecarlo

        with patch("functions.MakeValuePlot.subprocess.Popen") as mock_popen:
            _spawn_background_montecarlo(
                "config.json", str(tmp_path)
            )
            assert mock_popen.called

    @pytest.mark.agent_runnable
    def test_spawn_uses_start_new_session(self, tmp_path):
        """_spawn_background_montecarlo passes start_new_session=True."""
        from functions.MakeValuePlot import _spawn_background_montecarlo

        with patch("functions.MakeValuePlot.subprocess.Popen") as mock_popen:
            _spawn_background_montecarlo("config.json", str(tmp_path))
            _, kwargs = mock_popen.call_args
            assert kwargs.get("start_new_session") is True

    @pytest.mark.agent_runnable
    def test_spawn_creates_log_file(self, tmp_path):
        """_spawn_background_montecarlo creates montecarlo_backtest.log."""
        from functions.MakeValuePlot import _spawn_background_montecarlo

        with patch("functions.MakeValuePlot.subprocess.Popen"):
            _spawn_background_montecarlo("config.json", str(tmp_path))

        log_file = tmp_path / "montecarlo_backtest.log"
        assert log_file.exists()

    @pytest.mark.agent_runnable
    def test_spawn_passes_json_file_in_cmd(self, tmp_path):
        """_spawn_background_montecarlo includes --json-file in the command."""
        from functions.MakeValuePlot import _spawn_background_montecarlo

        captured_cmd = []

        def fake_popen(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return MagicMock()

        with patch(
            "functions.MakeValuePlot.subprocess.Popen",
            side_effect=fake_popen,
        ):
            _spawn_background_montecarlo(
                "/path/to/config.json", str(tmp_path)
            )

        assert "--json-file" in captured_cmd
        idx = captured_cmd.index("--json-file")
        assert captured_cmd[idx + 1] == "/path/to/config.json"

    @pytest.mark.agent_runnable
    def test_spawn_sets_pythonpath_in_env(self, tmp_path):
        """_spawn_background_montecarlo sets PYTHONPATH in subprocess env."""
        from functions.MakeValuePlot import _spawn_background_montecarlo

        with patch("functions.MakeValuePlot.subprocess.Popen") as mock_popen:
            _spawn_background_montecarlo("config.json", str(tmp_path))
            _, kwargs = mock_popen.call_args
            assert "env" in kwargs
            assert "PYTHONPATH" in kwargs["env"]

    @pytest.mark.agent_runnable
    def test_spawn_uses_module_mode(self, tmp_path):
        """_spawn_background_montecarlo uses -m flag for module invocation."""
        from functions.MakeValuePlot import _spawn_background_montecarlo

        captured_cmd = []

        def fake_popen(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return MagicMock()

        with patch(
            "functions.MakeValuePlot.subprocess.Popen",
            side_effect=fake_popen,
        ):
            _spawn_background_montecarlo("config.json", str(tmp_path))

        assert "-m" in captured_cmd
        idx = captured_cmd.index("-m")
        assert "background_montecarlo_runner" in captured_cmd[idx + 1]


##############################################################################
# Tests: async vs sync dispatch
##############################################################################

class TestAsyncVsSyncDispatch:
    """Verify makeDailyMonteCarloBacktest dispatches correctly."""

    def _make_mock_datearray(self):
        return [datetime.datetime(2024, 1, 1)]

    @pytest.mark.agent_runnable
    def test_sync_mode_calls_dailyBacktest_when_stale(self, tmp_path):
        """In sync mode, dailyBacktest_pctLong is called when plot is stale."""
        from functions.MakeValuePlot import makeDailyMonteCarloBacktest

        datearray = self._make_mock_datearray()

        with patch(
            "functions.MakeValuePlot.get_webpage_store",
            return_value=str(tmp_path),
        ), patch(
            "functions.MakeValuePlot.get_symbols_file",
            return_value="symbols.txt",
        ), patch(
            "functions.UpdateSymbols_inHDF5.loadQuotes_fromHDF",
            return_value=(None, None, datearray, None, None),
        ), patch(
            "functions.dailyBacktest_pctLong.dailyBacktest_pctLong"
        ) as mock_backtest, patch(
            "functions.MakeValuePlot._spawn_background_montecarlo"
        ) as mock_spawn:
            # No PNG file exists -> mtime=0 -> modified_hours >> 20
            makeDailyMonteCarloBacktest("config.json", async_mode=False)

        mock_backtest.assert_called_once_with("config.json")
        mock_spawn.assert_not_called()

    @pytest.mark.agent_runnable
    def test_async_mode_calls_spawn_when_stale(self, tmp_path):
        """In async mode, _spawn_background_montecarlo is called when stale."""
        from functions.MakeValuePlot import makeDailyMonteCarloBacktest

        datearray = self._make_mock_datearray()

        with patch(
            "functions.MakeValuePlot.get_webpage_store",
            return_value=str(tmp_path),
        ), patch(
            "functions.MakeValuePlot.get_symbols_file",
            return_value="symbols.txt",
        ), patch(
            "functions.UpdateSymbols_inHDF5.loadQuotes_fromHDF",
            return_value=(None, None, datearray, None, None),
        ), patch(
            "functions.dailyBacktest_pctLong.dailyBacktest_pctLong"
        ) as mock_backtest, patch(
            "functions.MakeValuePlot._spawn_background_montecarlo"
        ) as mock_spawn:
            makeDailyMonteCarloBacktest("config.json", async_mode=True)

        mock_spawn.assert_called_once()
        mock_backtest.assert_not_called()

    @pytest.mark.agent_runnable
    def test_fresh_plot_skips_both(self, tmp_path):
        """When plot is fresh (<20 hours), neither mode runs computation."""
        from functions.MakeValuePlot import makeDailyMonteCarloBacktest

        datearray = self._make_mock_datearray()
        # Create a fresh PNG file (mtime = now)
        png_path = tmp_path / "PyTAAA_monteCarloBacktest.png"
        png_path.write_text("fake png")

        with patch(
            "functions.MakeValuePlot.get_webpage_store",
            return_value=str(tmp_path),
        ), patch(
            "functions.MakeValuePlot.get_symbols_file",
            return_value="symbols.txt",
        ), patch(
            "functions.UpdateSymbols_inHDF5.loadQuotes_fromHDF",
            return_value=(None, None, datearray, None, None),
        ), patch(
            "functions.dailyBacktest_pctLong.dailyBacktest_pctLong"
        ) as mock_backtest, patch(
            "functions.MakeValuePlot._spawn_background_montecarlo"
        ) as mock_spawn:
            makeDailyMonteCarloBacktest("config.json", async_mode=True)

        mock_backtest.assert_not_called()
        mock_spawn.assert_not_called()

    @pytest.mark.agent_runnable
    def test_returns_html_string(self, tmp_path):
        """makeDailyMonteCarloBacktest returns a non-empty HTML string."""
        from functions.MakeValuePlot import makeDailyMonteCarloBacktest

        datearray = self._make_mock_datearray()

        with patch(
            "functions.MakeValuePlot.get_webpage_store",
            return_value=str(tmp_path),
        ), patch(
            "functions.MakeValuePlot.get_symbols_file",
            return_value="symbols.txt",
        ), patch(
            "functions.UpdateSymbols_inHDF5.loadQuotes_fromHDF",
            return_value=(None, None, datearray, None, None),
        ), patch(
            "functions.dailyBacktest_pctLong.dailyBacktest_pctLong"
        ), patch(
            "functions.MakeValuePlot._spawn_background_montecarlo"
        ):
            result = makeDailyMonteCarloBacktest("config.json")

        assert isinstance(result, str)
        assert "PyTAAA_monteCarloBacktest.png" in result
        assert "PyTAAA_monteCarloBacktestRecent.png" in result
