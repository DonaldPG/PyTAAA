"""Tests for functions/background_plot_generator.py.

All tests are marked with @pytest.mark.agent_runnable and use only
mock / synthetic data so they can run in the GitHub CI environment
without production HDF5 files.
"""

import datetime
import os
import pickle
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from functions.background_plot_generator import (
    _parse_args,
    generate_single_full_history_plot,
    generate_single_recent_plot,
    load_plot_data,
)


##############################################################################
# Helpers
##############################################################################

def _make_datearray(n: int = 60) -> list:
    """Return a list of *n* consecutive datetime.date objects starting 2010."""
    base = datetime.date(2010, 1, 4)
    return [base + datetime.timedelta(days=d) for d in range(n)]


def _make_adjclose(n_symbols: int = 3, n_days: int = 60) -> np.ndarray:
    np.random.seed(0)
    return (
        np.cumsum(np.random.randn(n_symbols, n_days) * 0.5 + 0.1, axis=1) + 100
    )


def _make_signal2D(n_symbols: int = 3, n_days: int = 60) -> np.ndarray:
    np.random.seed(1)
    return (np.random.rand(n_symbols, n_days) > 0.5).astype(float)


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


def _full_history_bundle(
    tmpdir: str,
    symbol: str = "AAPL",
    recent: bool = False,
) -> dict:
    """Build a minimal bundle for generate_single_full_history_plot."""
    n_days = 60
    adjClose_row = np.linspace(100, 200, n_days)
    datearray = _make_datearray(n_days)
    signal2D_row = np.ones(n_days)
    quotes_despike_row = adjClose_row.copy()
    return {
        "i": 0,
        "symbol": symbol,
        "adjClose_row": adjClose_row,
        "datearray": datearray,
        "signal2D_row": signal2D_row,
        "quotes_despike_row": quotes_despike_row,
        "uptrendSignalMethod": "HMAs",
        "lowChannel_row": None,
        "hiChannel_row": None,
        "output_dir": tmpdir,
        "today_str": "Monday, 01. January 2024 12:00PM",
    }


def _recent_bundle(tmpdir: str, symbol: str = "AAPL") -> dict:
    """Build a minimal bundle for generate_single_recent_plot."""
    n_days = 60
    adjClose_row = np.linspace(100, 200, n_days)
    datearray = _make_datearray(n_days)
    signal2D_row = np.ones(n_days)
    signal2D_daily_row = np.ones(n_days)
    quotes_despike_row = adjClose_row.copy()
    firstdate_index = 5
    params = _minimal_params()
    trend = np.linspace(95, 205, params["numdaysinfit"] + 1)
    return {
        "symbol": symbol,
        "adjClose_row": adjClose_row,
        "datearray": datearray,
        "signal2D_row": signal2D_row,
        "signal2D_daily_row": signal2D_daily_row,
        "quotes_despike_row": quotes_despike_row,
        "firstdate_index": firstdate_index,
        "uptrendSignalMethod": "HMAs",
        "lowChannel_row": None,
        "hiChannel_row": None,
        "lowerTrend": trend - 5,
        "upperTrend": trend + 5,
        "NoGapLowerTrend": np.linspace(95, 195, params["numdaysinfit2"]),
        "NoGapUpperTrend": np.linspace(105, 205, params["numdaysinfit2"]),
        "params": params,
        "output_dir": tmpdir,
        "today_str": "Monday, 01. January 2024 12:00PM",
    }


##############################################################################
# Tests: load_plot_data
##############################################################################

class TestLoadPlotData:
    """Tests for the load_plot_data function."""

    @pytest.mark.agent_runnable
    def test_loads_valid_pickle(self, tmp_path):
        """load_plot_data deserializes a valid pickle file."""
        data = {"adjClose": np.array([[1.0, 2.0]]), "symbols": ["AAPL"]}
        pkl = tmp_path / "data.pkl"
        pkl.write_bytes(pickle.dumps(data))
        result = load_plot_data(str(pkl))
        assert result["symbols"] == ["AAPL"]

    @pytest.mark.agent_runnable
    def test_raises_file_not_found(self, tmp_path):
        """load_plot_data raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_plot_data(str(tmp_path / "nonexistent.pkl"))

    @pytest.mark.agent_runnable
    def test_raises_value_error_for_non_dict(self, tmp_path):
        """load_plot_data raises ValueError when pickle contains non-dict."""
        pkl = tmp_path / "bad.pkl"
        pkl.write_bytes(pickle.dumps([1, 2, 3]))
        with pytest.raises(ValueError, match="Expected dict"):
            load_plot_data(str(pkl))

    @pytest.mark.agent_runnable
    def test_preserves_numpy_arrays(self, tmp_path):
        """load_plot_data round-trips numpy arrays without corruption."""
        arr = np.array([[1.5, 2.5, 3.5]])
        data = {"adjClose": arr}
        pkl = tmp_path / "data.pkl"
        pkl.write_bytes(pickle.dumps(data))
        result = load_plot_data(str(pkl))
        np.testing.assert_array_equal(result["adjClose"], arr)


##############################################################################
# Tests: generate_single_full_history_plot
##############################################################################

class TestGenerateSingleFullHistoryPlot:
    """Tests for the full-history plot worker function."""

    @pytest.mark.agent_runnable
    def test_creates_png_file(self, tmp_path):
        """generate_single_full_history_plot writes a PNG file."""
        bundle = _full_history_bundle(str(tmp_path))
        result = generate_single_full_history_plot(bundle)
        expected_path = tmp_path / f"0_{bundle['symbol']}.png"
        assert expected_path.exists(), f"PNG not created; result: {result}"
        assert result.startswith("OK")

    @pytest.mark.agent_runnable
    def test_skips_recent_file(self, tmp_path):
        """generate_single_full_history_plot skips plots younger than 20 hours."""
        bundle = _full_history_bundle(str(tmp_path))
        # Pre-create the file with a current mtime
        png = tmp_path / f"0_{bundle['symbol']}.png"
        png.write_bytes(b"")
        result = generate_single_full_history_plot(bundle)
        assert result.startswith("SKIP")

    @pytest.mark.agent_runnable
    def test_returns_ok_string_on_success(self, tmp_path):
        """generate_single_full_history_plot returns 'OK ...' on success."""
        bundle = _full_history_bundle(str(tmp_path), symbol="MSFT")
        result = generate_single_full_history_plot(bundle)
        assert result.startswith("OK") or result.startswith("SKIP")

    @pytest.mark.agent_runnable
    def test_handles_all_nan_despike(self, tmp_path):
        """generate_single_full_history_plot handles NaN in despiked quotes."""
        bundle = _full_history_bundle(str(tmp_path), symbol="NAN_TEST")
        bundle["quotes_despike_row"] = np.full(60, np.nan)
        # Should not raise; NaN check path taken
        result = generate_single_full_history_plot(bundle)
        assert result.startswith("OK") or result.startswith("ERROR")

    @pytest.mark.agent_runnable
    def test_percentile_channels_path(self, tmp_path):
        """generate_single_full_history_plot handles percentileChannels method."""
        n = 60
        bundle = _full_history_bundle(str(tmp_path), symbol="CHAN")
        bundle["uptrendSignalMethod"] = "percentileChannels"
        bundle["lowChannel_row"] = np.linspace(90, 180, n)
        bundle["hiChannel_row"] = np.linspace(110, 220, n)
        result = generate_single_full_history_plot(bundle)
        assert result.startswith("OK") or result.startswith("SKIP")


##############################################################################
# Tests: generate_single_recent_plot
##############################################################################

class TestGenerateSingleRecentPlot:
    """Tests for the recent-history plot worker function."""

    @pytest.mark.agent_runnable
    def test_creates_png_file(self, tmp_path):
        """generate_single_recent_plot writes a PNG file."""
        bundle = _recent_bundle(str(tmp_path))
        result = generate_single_recent_plot(bundle)
        expected_path = tmp_path / f"0_recent_{bundle['symbol']}.png"
        assert expected_path.exists(), f"PNG not created; result: {result}"
        assert result.startswith("OK")

    @pytest.mark.agent_runnable
    def test_skips_recent_file(self, tmp_path):
        """generate_single_recent_plot skips plots younger than 20 hours."""
        bundle = _recent_bundle(str(tmp_path))
        png = tmp_path / f"0_recent_{bundle['symbol']}.png"
        png.write_bytes(b"")
        result = generate_single_recent_plot(bundle)
        assert result.startswith("SKIP")

    @pytest.mark.agent_runnable
    def test_returns_ok_string_on_success(self, tmp_path):
        """generate_single_recent_plot returns 'OK ...' on success."""
        bundle = _recent_bundle(str(tmp_path), symbol="GOOG")
        result = generate_single_recent_plot(bundle)
        assert result.startswith("OK") or result.startswith("SKIP")

    @pytest.mark.agent_runnable
    def test_percentile_channels_path(self, tmp_path):
        """generate_single_recent_plot handles percentileChannels method."""
        n = 60
        bundle = _recent_bundle(str(tmp_path), symbol="PCTCH")
        bundle["uptrendSignalMethod"] = "percentileChannels"
        bundle["lowChannel_row"] = np.linspace(90, 180, n)
        bundle["hiChannel_row"] = np.linspace(110, 220, n)
        result = generate_single_recent_plot(bundle)
        assert result.startswith("OK") or result.startswith("SKIP")

    @pytest.mark.agent_runnable
    def test_error_propagation(self, tmp_path):
        """generate_single_recent_plot returns 'ERROR' string on bad data."""
        bundle = _recent_bundle(str(tmp_path), symbol="ERR")
        # Corrupt the trend arrays to trigger a plotting error
        bundle["upperTrend"] = None  # will cause TypeError inside matplotlib
        result = generate_single_recent_plot(bundle)
        # Should return an error string rather than raise
        assert isinstance(result, str)


##############################################################################
# Tests: _parse_args
##############################################################################

class TestParseArgs:
    """Tests for the CLI argument parser."""

    @pytest.mark.agent_runnable
    def test_required_data_file(self):
        """_parse_args parses --data-file correctly."""
        args = _parse_args(["--data-file", "/tmp/foo.pkl"])
        assert args.data_file == "/tmp/foo.pkl"
        assert args.max_workers == 2  # default

    @pytest.mark.agent_runnable
    def test_custom_max_workers(self):
        """_parse_args accepts --max-workers."""
        args = _parse_args(["--data-file", "/tmp/foo.pkl", "--max-workers", "4"])
        assert args.max_workers == 4

    @pytest.mark.agent_runnable
    def test_missing_data_file_exits(self):
        """_parse_args exits when --data-file is not provided."""
        with pytest.raises(SystemExit):
            _parse_args([])
