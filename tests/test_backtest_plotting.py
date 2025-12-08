"""
Unit tests for src.backtest.plotting module.

Tests plotting utility functions and BacktestPlotter class.
"""

import datetime
import numpy as np
import pytest

from src.backtest.plotting import (
    calculate_plot_range,
    get_y_position,
    format_performance_metrics,
    BacktestPlotter,
    create_monte_carlo_histogram,
)


class TestCalculatePlotRange:
    """Tests for calculate_plot_range function."""

    def test_default_range(self):
        """Test plot range with default ymin."""
        result = calculate_plot_range(plotmax=1e9)
        expected = np.log10(1e9) - np.log10(7000.0)
        assert abs(result - expected) < 1e-10

    def test_custom_ymin(self):
        """Test plot range with custom ymin."""
        result = calculate_plot_range(plotmax=1e6, ymin=1000.0)
        expected = np.log10(1e6) - np.log10(1000.0)
        assert abs(result - expected) < 1e-10

    def test_positive_range(self):
        """Test that range is always positive for valid inputs."""
        result = calculate_plot_range(plotmax=1e9, ymin=7000.0)
        assert result > 0


class TestGetYPosition:
    """Tests for get_y_position function."""

    def test_zero_fraction(self):
        """Test y position at fraction 0."""
        plotrange = 5.0
        result = get_y_position(plotrange, fraction=0.0, ymin=7000.0)
        # At fraction 0, result = 10^(log10(7000) + 0*5) = 7000.
        assert abs(result - 7000.0) < 0.01

    def test_one_fraction(self):
        """Test y position at fraction 1."""
        plotrange = np.log10(1e9) - np.log10(7000.0)
        result = get_y_position(plotrange, fraction=1.0, ymin=7000.0)
        expected = 1e9
        assert abs(result - expected) / expected < 0.01

    def test_mid_fraction(self):
        """Test y position at fraction 0.5."""
        plotrange = 2.0  # Log scale range.
        result = get_y_position(plotrange, fraction=0.5, ymin=100.0)
        # At 0.5, should be 10^(log10(100) + 0.5*2) = 10^3 = 1000.
        expected = 1000.0
        assert abs(result - expected) < 1.0


class TestFormatPerformanceMetrics:
    """Tests for format_performance_metrics function."""

    def test_cagr_mode(self):
        """Test formatting in CAGR mode."""
        f_sharpe, f_metric, f_drawdown = format_performance_metrics(
            sharpe=1.25,
            return_val=1.15,
            cagr=0.12,
            drawdown=-0.08,
            show_cagr=True,
        )
        
        assert "1.25" in f_sharpe
        assert "12" in f_metric  # CAGR as percentage.
        assert "8" in f_drawdown  # Drawdown as percentage.

    def test_return_mode(self):
        """Test formatting in return mode."""
        f_sharpe, f_metric, f_drawdown = format_performance_metrics(
            sharpe=1.25,
            return_val=1.15,
            cagr=0.12,
            drawdown=-0.08,
            show_cagr=False,
        )
        
        assert "1.25" in f_sharpe
        assert "1.15" in f_metric  # Return as decimal.
        assert "8" in f_drawdown

    def test_negative_sharpe(self):
        """Test formatting with negative Sharpe ratio."""
        f_sharpe, _, _ = format_performance_metrics(
            sharpe=-0.5,
            return_val=0.95,
            cagr=-0.05,
            drawdown=-0.15,
            show_cagr=True,
        )
        
        assert "-" in f_sharpe


class TestBacktestPlotter:
    """Tests for BacktestPlotter class."""

    @pytest.fixture
    def plotter(self):
        """Create BacktestPlotter instance for testing."""
        return BacktestPlotter(plotmax=1e6, ymin=7000.0)

    def test_initialization(self, plotter):
        """Test BacktestPlotter initialization."""
        assert plotter.plotmax == 1e6
        assert plotter.ymin == 7000.0
        assert plotter.plotrange > 0

    def test_plotrange_calculation(self, plotter):
        """Test that plotrange is calculated correctly."""
        expected = np.log10(1e6) - np.log10(7000.0)
        assert abs(plotter.plotrange - expected) < 1e-10

    def test_custom_figsize(self):
        """Test custom figure size."""
        custom_size = (12, 8)
        plotter = BacktestPlotter(figsize=custom_size)
        assert plotter.figsize == custom_size


class TestCreateMonteCarloHistogram:
    """Tests for create_monte_carlo_histogram function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample portfolio values and dates."""
        np.random.seed(42)
        n_trials = 50
        n_days = 252
        
        # Generate portfolio values.
        values = np.zeros((n_trials, n_days))
        for i in range(n_trials):
            daily_returns = 1.0 + np.random.normal(0.0003, 0.015, n_days)
            values[i, :] = 10000.0 * np.cumprod(daily_returns)
        
        # Generate dates.
        start_date = datetime.date(2020, 1, 1)
        dates = [
            start_date + datetime.timedelta(days=i)
            for i in range(n_days)
        ]
        
        return values, dates

    def test_output_shape(self, sample_data):
        """Test output array shape."""
        values, dates = sample_data
        result = create_monte_carlo_histogram(
            values, dates, n_bins=50, ymin=7000.0, ymax=20000.0
        )
        
        # Shape should be (n_bins-1, n_days, 3) for RGB.
        assert result.shape[0] == 49  # n_bins - 1.
        assert result.shape[1] == len(dates)
        assert result.shape[2] == 3

    def test_output_range(self, sample_data):
        """Test output values are in valid range."""
        values, dates = sample_data
        result = create_monte_carlo_histogram(
            values, dates, n_bins=50, ymin=7000.0, ymax=20000.0
        )
        
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_rgb_channels_equal(self, sample_data):
        """Test that RGB channels are equal (grayscale)."""
        values, dates = sample_data
        result = create_monte_carlo_histogram(
            values, dates, n_bins=50, ymin=7000.0, ymax=20000.0
        )
        
        assert np.allclose(result[:, :, 0], result[:, :, 1])
        assert np.allclose(result[:, :, 1], result[:, :, 2])
