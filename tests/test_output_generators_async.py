"""Tests for the async mode additions in functions/output_generators.py.

All tests are marked with @pytest.mark.agent_runnable and rely on
synthetic data only.
"""

import datetime
import os
import pickle
import sys
import tempfile
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


##############################################################################
# Helpers
##############################################################################

def _make_datearray(n: int = 60) -> list:
    base = datetime.date(2010, 1, 4)
    return [base + datetime.timedelta(days=d) for d in range(n)]


def _make_adjclose(n_symbols: int = 2, n_days: int = 60) -> np.ndarray:
    np.random.seed(42)
    return np.cumsum(
        np.random.randn(n_symbols, n_days) * 0.5 + 0.1, axis=1
    ) + 100


def _make_signal2D(n_symbols: int = 2, n_days: int = 60) -> np.ndarray:
    return np.ones((n_symbols, n_days))


def _minimal_params() -> dict:
    return {
        "LongPeriod": 10,
        "stddevThreshold": 8.0,
        "uptrendSignalMethod": "HMAs",
        "minperiod": 4,
        "maxperiod": 8,
        "incperiod": 2,
        "numdaysinfit": 20,
        "numdaysinfit2": 10,
        "offset": 3,
    }


##############################################################################
# Tests: generate_portfolio_plots – backward compatibility
##############################################################################

class TestGeneratePortfolioPlotsBackwardCompatibility:
    """Verify that synchronous mode (async_mode=False) still works."""

    @pytest.mark.agent_runnable
    def test_sync_mode_is_default(self):
        """generate_portfolio_plots defaults to synchronous mode."""
        import inspect
        from functions.output_generators import generate_portfolio_plots
        sig = inspect.signature(generate_portfolio_plots)
        assert sig.parameters["async_mode"].default is False

    @pytest.mark.agent_runnable
    def test_max_workers_default_is_2(self):
        """generate_portfolio_plots defaults max_workers to 2."""
        import inspect
        from functions.output_generators import generate_portfolio_plots
        sig = inspect.signature(generate_portfolio_plots)
        assert sig.parameters["max_workers"].default == 2

    @pytest.mark.agent_runnable
    def test_sync_mode_does_not_call_spawn(self, tmp_path):
        """In sync mode, _spawn_background_plot_generation is not called."""
        from functions.output_generators import generate_portfolio_plots

        n_days, n_symbols = 60, 2
        adjClose = _make_adjclose(n_symbols, n_days)
        datearray = _make_datearray(n_days)
        signal2D = _make_signal2D(n_symbols, n_days)
        params = _minimal_params()

        with patch(
            "functions.output_generators._spawn_background_plot_generation"
        ) as mock_spawn, patch(
            "functions.output_generators._generate_full_history_plots"
        ) as mock_full, patch(
            "functions.output_generators._generate_recent_plots"
        ) as mock_recent:
            generate_portfolio_plots(
                adjClose, ["AAPL", "MSFT"], datearray,
                signal2D, signal2D, params, str(tmp_path),
                async_mode=False,
            )
            mock_spawn.assert_not_called()
            mock_full.assert_called_once()
            mock_recent.assert_called_once()

    @pytest.mark.agent_runnable
    def test_async_mode_calls_spawn_not_helpers(self, tmp_path):
        """In async mode, only _spawn_background_plot_generation is called."""
        from functions.output_generators import generate_portfolio_plots

        n_days, n_symbols = 60, 2
        adjClose = _make_adjclose(n_symbols, n_days)
        datearray = _make_datearray(n_days)
        signal2D = _make_signal2D(n_symbols, n_days)
        params = _minimal_params()

        with patch(
            "functions.output_generators._spawn_background_plot_generation"
        ) as mock_spawn, patch(
            "functions.output_generators._generate_full_history_plots"
        ) as mock_full, patch(
            "functions.output_generators._generate_recent_plots"
        ) as mock_recent:
            generate_portfolio_plots(
                adjClose, ["AAPL", "MSFT"], datearray,
                signal2D, signal2D, params, str(tmp_path),
                async_mode=True, max_workers=3,
            )
            mock_spawn.assert_called_once()
            mock_full.assert_not_called()
            mock_recent.assert_not_called()
            # Verify max_workers is forwarded
            _, kwargs = mock_spawn.call_args
            assert kwargs["max_workers"] == 3

    @pytest.mark.agent_runnable
    def test_early_return_outside_hours(self, tmp_path):
        """generate_portfolio_plots returns early when outside allowed hours.

        The allowed hours check is: ``hourOfDay >= 1 or 11 < hourOfDay < 13``.
        Hour 0 (midnight) is the only hour that fails the check and causes
        an early return.
        """
        from functions.output_generators import generate_portfolio_plots

        n_days, n_symbols = 60, 2
        adjClose = _make_adjclose(n_symbols, n_days)
        datearray = _make_datearray(n_days)
        signal2D = _make_signal2D(n_symbols, n_days)
        params = _minimal_params()

        # Hour=0 fails both conditions, triggering the early return.
        fake_now = datetime.datetime(2024, 1, 2, 0, 0, 0)
        with patch(
            "functions.output_generators.datetime"
        ) as mock_dt, patch(
            "functions.output_generators._spawn_background_plot_generation"
        ) as mock_spawn, patch(
            "functions.output_generators._generate_full_history_plots"
        ) as mock_full:
            mock_dt.datetime.now.return_value = fake_now
            mock_dt.timedelta = datetime.timedelta
            generate_portfolio_plots(
                adjClose, ["AAPL", "MSFT"], datearray,
                signal2D, signal2D, params, str(tmp_path),
                async_mode=True,
            )
            # Should return before spawning
            mock_spawn.assert_not_called()
            mock_full.assert_not_called()


##############################################################################
# Tests: _spawn_background_plot_generation
##############################################################################

class TestSpawnBackgroundPlotGeneration:
    """Tests for the subprocess-spawning helper."""

    @pytest.mark.agent_runnable
    def test_creates_pickle_and_spawns_process(self, tmp_path):
        """_spawn_background_plot_generation serializes data and calls Popen."""
        from functions.output_generators import _spawn_background_plot_generation

        n_days, n_symbols = 60, 2
        adjClose = _make_adjclose(n_symbols, n_days)
        datearray = _make_datearray(n_days)
        signal2D = _make_signal2D(n_symbols, n_days)
        params = _minimal_params()

        with patch("functions.output_generators.subprocess.Popen") as mock_popen:
            _spawn_background_plot_generation(
                adjClose, ["A", "B"], datearray,
                signal2D, signal2D, params, str(tmp_path),
                max_workers=2,
            )
            assert mock_popen.called

    @pytest.mark.agent_runnable
    def test_passes_max_workers_to_command(self, tmp_path):
        """_spawn_background_plot_generation passes --max-workers to subprocess."""
        from functions.output_generators import _spawn_background_plot_generation

        n_days, n_symbols = 60, 2
        adjClose = _make_adjclose(n_symbols, n_days)
        datearray = _make_datearray(n_days)
        signal2D = _make_signal2D(n_symbols, n_days)
        params = _minimal_params()

        captured_cmd = []

        def fake_popen(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return MagicMock()

        with patch(
            "functions.output_generators.subprocess.Popen", side_effect=fake_popen
        ):
            _spawn_background_plot_generation(
                adjClose, ["A", "B"], datearray,
                signal2D, signal2D, params, str(tmp_path),
                max_workers=4,
            )

        assert "--max-workers" in captured_cmd
        idx = captured_cmd.index("--max-workers")
        assert captured_cmd[idx + 1] == "4"

    @pytest.mark.agent_runnable
    def test_spawns_with_start_new_session(self, tmp_path):
        """_spawn_background_plot_generation uses start_new_session=True."""
        from functions.output_generators import _spawn_background_plot_generation

        n_days, n_symbols = 60, 2
        adjClose = _make_adjclose(n_symbols, n_days)
        datearray = _make_datearray(n_days)
        signal2D = _make_signal2D(n_symbols, n_days)
        params = _minimal_params()

        with patch("functions.output_generators.subprocess.Popen") as mock_popen:
            _spawn_background_plot_generation(
                adjClose, ["A", "B"], datearray,
                signal2D, signal2D, params, str(tmp_path),
            )
            _, kwargs = mock_popen.call_args
            assert kwargs.get("start_new_session") is True

    @pytest.mark.agent_runnable
    def test_includes_lowChannel_in_pickle_when_provided(self, tmp_path):
        """_spawn_background_plot_generation includes channel arrays in pickle."""
        from functions.output_generators import _spawn_background_plot_generation

        n_days, n_symbols = 60, 2
        adjClose = _make_adjclose(n_symbols, n_days)
        datearray = _make_datearray(n_days)
        signal2D = _make_signal2D(n_symbols, n_days)
        low_ch = np.ones((n_symbols, n_days)) * 90
        hi_ch = np.ones((n_symbols, n_days)) * 110
        params = _minimal_params()
        params["uptrendSignalMethod"] = "percentileChannels"

        written_data = {}

        original_dump = pickle.dump

        def capturing_dump(obj, fh, **kwargs):
            written_data.update(obj)
            original_dump(obj, fh, **kwargs)

        with patch("functions.output_generators.subprocess.Popen"), patch(
            "functions.output_generators.pickle.dump", side_effect=capturing_dump
        ):
            _spawn_background_plot_generation(
                adjClose, ["A", "B"], datearray,
                signal2D, signal2D, params, str(tmp_path),
                lowChannel=low_ch, hiChannel=hi_ch,
            )

        assert "lowChannel" in written_data
        assert "hiChannel" in written_data

    @pytest.mark.agent_runnable
    def test_creates_log_file_in_output_dir(self, tmp_path):
        """_spawn_background_plot_generation creates plot_generation.log."""
        from functions.output_generators import _spawn_background_plot_generation

        n_days, n_symbols = 60, 2
        adjClose = _make_adjclose(n_symbols, n_days)
        datearray = _make_datearray(n_days)
        signal2D = _make_signal2D(n_symbols, n_days)
        params = _minimal_params()

        with patch("functions.output_generators.subprocess.Popen"):
            _spawn_background_plot_generation(
                adjClose, ["A", "B"], datearray,
                signal2D, signal2D, params, str(tmp_path),
            )

        log_file = tmp_path / "plot_generation.log"
        assert log_file.exists()


##############################################################################
# Tests: _generate_full_history_plots helper
##############################################################################

class TestGenerateFullHistoryPlotsHelper:
    """Tests for the extracted _generate_full_history_plots helper."""

    @pytest.mark.agent_runnable
    def test_is_callable(self):
        """_generate_full_history_plots is importable and callable."""
        from functions.output_generators import _generate_full_history_plots
        assert callable(_generate_full_history_plots)

    @pytest.mark.agent_runnable
    def test_signature_has_required_params(self):
        """_generate_full_history_plots has the expected signature."""
        import inspect
        from functions.output_generators import _generate_full_history_plots
        sig = inspect.signature(_generate_full_history_plots)
        required = {"adjClose", "symbols", "datearray", "signal2D", "params", "output_dir"}
        assert required.issubset(set(sig.parameters.keys()))


##############################################################################
# Tests: _generate_recent_plots helper
##############################################################################

class TestGenerateRecentPlotsHelper:
    """Tests for the extracted _generate_recent_plots helper."""

    @pytest.mark.agent_runnable
    def test_is_callable(self):
        """_generate_recent_plots is importable and callable."""
        from functions.output_generators import _generate_recent_plots
        assert callable(_generate_recent_plots)

    @pytest.mark.agent_runnable
    def test_signature_has_required_params(self):
        """_generate_recent_plots has the expected signature."""
        import inspect
        from functions.output_generators import _generate_recent_plots
        sig = inspect.signature(_generate_recent_plots)
        required = {
            "adjClose", "symbols", "datearray", "signal2D", "signal2D_daily",
            "params", "output_dir", "firstdate_index",
        }
        assert required.issubset(set(sig.parameters.keys()))
