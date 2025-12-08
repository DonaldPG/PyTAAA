"""
Unit tests for src.backtest.montecarlo module.

Tests Monte Carlo simulation functions including parameter generation,
metrics calculation, and buy-hold comparison tests.
"""

import numpy as np
import pytest

from src.backtest.montecarlo import (
    random_triangle,
    calculate_sharpe_ratio,
    calculate_period_metrics,
    calculate_drawdown_metrics,
    beat_buy_hold_test,
    beat_buy_hold_test2,
    MonteCarloBacktest,
)
from src.backtest.config import TradingConstants


class TestRandomTriangle:
    """Tests for random_triangle function."""

    def test_single_value_return_type(self):
        """Test that size=1 returns a float."""
        result = random_triangle(low=0.0, mid=0.5, high=1.0, size=1)
        assert isinstance(result, (float, np.floating))

    def test_array_return_type(self):
        """Test that size>1 returns numpy array."""
        result = random_triangle(low=0.0, mid=0.5, high=1.0, size=10)
        assert isinstance(result, np.ndarray)
        assert len(result) == 10

    def test_values_within_bounds(self):
        """Test that generated values are within bounds."""
        for _ in range(100):
            result = random_triangle(low=0.0, mid=0.5, high=1.0, size=1)
            assert 0.0 <= result <= 1.0

    def test_array_values_within_bounds(self):
        """Test that array values are within bounds."""
        result = random_triangle(low=10.0, mid=50.0, high=100.0, size=1000)
        assert np.all(result >= 10.0)
        assert np.all(result <= 100.0)

    def test_distribution_center(self):
        """Test that distribution is centered around mid value."""
        result = random_triangle(low=0.0, mid=0.5, high=1.0, size=10000)
        mean_val = np.mean(result)
        # Mean should be close to mid value.
        assert 0.4 < mean_val < 0.6


class TestCalculateSharpeRatio:
    """Tests for calculate_sharpe_ratio function."""

    def test_positive_sharpe(self):
        """Test Sharpe ratio with positive returns."""
        # Simulate daily gains with slight positive bias.
        np.random.seed(42)
        daily_gains = 1.0 + np.random.normal(0.0005, 0.01, 252)
        
        sharpe = calculate_sharpe_ratio(daily_gains)
        assert sharpe > 0

    def test_negative_sharpe(self):
        """Test Sharpe ratio with negative returns."""
        # Simulate daily gains with negative bias.
        np.random.seed(42)
        daily_gains = 1.0 + np.random.normal(-0.001, 0.01, 252)
        
        sharpe = calculate_sharpe_ratio(daily_gains)
        assert sharpe < 0

    def test_zero_volatility(self):
        """Test Sharpe ratio with zero volatility."""
        daily_gains = np.ones(252)
        sharpe = calculate_sharpe_ratio(daily_gains)
        assert sharpe == 0.0

    def test_custom_trading_days(self):
        """Test Sharpe ratio with custom trading days."""
        np.random.seed(42)
        daily_gains = 1.0 + np.random.normal(0.0005, 0.01, 252)
        
        sharpe_252 = calculate_sharpe_ratio(daily_gains, trading_days=252)
        sharpe_365 = calculate_sharpe_ratio(daily_gains, trading_days=365)
        
        # Different annualization should give different results.
        assert sharpe_252 != sharpe_365


class TestCalculatePeriodMetrics:
    """Tests for calculate_period_metrics function."""

    @pytest.fixture
    def sample_portfolio_values(self):
        """Create sample portfolio values for testing."""
        np.random.seed(42)
        # Generate 15 years of daily data.
        n_days = 15 * 252
        daily_returns = 1.0 + np.random.normal(0.0003, 0.015, n_days)
        values = 10000.0 * np.cumprod(daily_returns)
        return values

    def test_all_periods_present(self, sample_portfolio_values):
        """Test that all time periods are in results."""
        metrics = calculate_period_metrics(sample_portfolio_values)
        
        expected_periods = ["15Yr", "10Yr", "5Yr", "3Yr", "2Yr", "1Yr"]
        for period in expected_periods:
            assert period in metrics

    def test_metric_keys_present(self, sample_portfolio_values):
        """Test that each period has required metric keys."""
        metrics = calculate_period_metrics(sample_portfolio_values)
        
        required_keys = ["sharpe", "return", "cagr", "days"]
        for period_metrics in metrics.values():
            for key in required_keys:
                assert key in period_metrics

    def test_sharpe_is_float(self, sample_portfolio_values):
        """Test that Sharpe ratio is a float."""
        metrics = calculate_period_metrics(sample_portfolio_values)
        
        for period_metrics in metrics.values():
            assert isinstance(period_metrics["sharpe"], float)

    def test_short_data_handles_missing_periods(self):
        """Test handling of data too short for some periods."""
        # Only 2 years of data.
        np.random.seed(42)
        n_days = 2 * 252 + 10  # Slightly more than 2 years.
        daily_returns = 1.0 + np.random.normal(0.0003, 0.015, n_days)
        values = 10000.0 * np.cumprod(daily_returns)
        
        metrics = calculate_period_metrics(values)
        
        # Should have 2Yr and 1Yr, but not longer periods.
        assert "1Yr" in metrics
        assert "2Yr" in metrics
        assert "5Yr" not in metrics


class TestCalculateDrawdownMetrics:
    """Tests for calculate_drawdown_metrics function."""

    @pytest.fixture
    def sample_portfolio_values(self):
        """Create sample portfolio values with drawdowns."""
        np.random.seed(42)
        n_days = 5 * 252
        daily_returns = 1.0 + np.random.normal(0.0002, 0.02, n_days)
        values = 10000.0 * np.cumprod(daily_returns)
        return values

    def test_drawdown_is_negative_or_zero(self, sample_portfolio_values):
        """Test that drawdown values are negative or zero."""
        metrics = calculate_drawdown_metrics(sample_portfolio_values)
        
        for period, dd in metrics.items():
            assert dd <= 0, f"Drawdown for {period} should be <= 0"

    def test_all_available_periods_present(self, sample_portfolio_values):
        """Test that available periods are in results."""
        metrics = calculate_drawdown_metrics(sample_portfolio_values)
        
        # With 5 years of data, should have 5Yr and shorter.
        assert "5Yr" in metrics
        assert "3Yr" in metrics
        assert "2Yr" in metrics
        assert "1Yr" in metrics

    def test_no_drawdown_for_monotonic_increase(self):
        """Test drawdown is zero for monotonically increasing values."""
        values = np.linspace(10000, 20000, 252)
        metrics = calculate_drawdown_metrics(values)
        
        assert metrics["1Yr"] == 0.0


class TestBeatBuyHoldTest:
    """Tests for beat_buy_hold_test function."""

    def test_positive_score_when_strategy_wins(self):
        """Test positive score when strategy beats buy & hold."""
        strategy = {
            "15Yr": {"sharpe": 1.5},
            "10Yr": {"sharpe": 1.4},
            "5Yr": {"sharpe": 1.3},
            "3Yr": {"sharpe": 1.2},
            "2Yr": {"sharpe": 1.1},
            "1Yr": {"sharpe": 1.0},
        }
        buyhold = {
            "15Yr": {"sharpe": 0.5},
            "10Yr": {"sharpe": 0.5},
            "5Yr": {"sharpe": 0.5},
            "3Yr": {"sharpe": 0.5},
            "2Yr": {"sharpe": 0.5},
            "1Yr": {"sharpe": 0.5},
        }
        
        score = beat_buy_hold_test(strategy, buyhold)
        assert score > 0

    def test_negative_score_when_buyhold_wins(self):
        """Test negative score when buy & hold beats strategy."""
        strategy = {
            "15Yr": {"sharpe": 0.3},
            "10Yr": {"sharpe": 0.3},
            "5Yr": {"sharpe": 0.3},
            "3Yr": {"sharpe": 0.3},
            "2Yr": {"sharpe": 0.3},
            "1Yr": {"sharpe": 0.3},
        }
        buyhold = {
            "15Yr": {"sharpe": 1.0},
            "10Yr": {"sharpe": 1.0},
            "5Yr": {"sharpe": 1.0},
            "3Yr": {"sharpe": 1.0},
            "2Yr": {"sharpe": 1.0},
            "1Yr": {"sharpe": 1.0},
        }
        
        score = beat_buy_hold_test(strategy, buyhold)
        assert score < 0

    def test_zero_score_when_equal(self):
        """Test zero score when performance is equal."""
        metrics = {
            "15Yr": {"sharpe": 1.0},
            "10Yr": {"sharpe": 1.0},
            "5Yr": {"sharpe": 1.0},
            "3Yr": {"sharpe": 1.0},
            "2Yr": {"sharpe": 1.0},
            "1Yr": {"sharpe": 1.0},
        }
        
        score = beat_buy_hold_test(metrics, metrics)
        assert score == 0.0


class TestBeatBuyHoldTest2:
    """Tests for beat_buy_hold_test2 function."""

    def test_high_score_when_strategy_dominates(self):
        """Test high score when strategy dominates."""
        strategy = {
            "15Yr": {"return": 1.15},
            "10Yr": {"return": 1.12},
            "5Yr": {"return": 1.10},
            "3Yr": {"return": 1.08},
            "2Yr": {"return": 1.06},
            "1Yr": {"return": 1.05},
        }
        buyhold = {
            "15Yr": {"return": 1.05},
            "10Yr": {"return": 1.04},
            "5Yr": {"return": 1.03},
            "3Yr": {"return": 1.02},
            "2Yr": {"return": 1.01},
            "1Yr": {"return": 1.00},
        }
        strategy_dd = {
            "15Yr": -0.05, "10Yr": -0.04, "5Yr": -0.03,
            "3Yr": -0.02, "2Yr": -0.01, "1Yr": -0.01,
        }
        buyhold_dd = {
            "15Yr": -0.15, "10Yr": -0.14, "5Yr": -0.13,
            "3Yr": -0.12, "2Yr": -0.11, "1Yr": -0.10,
        }
        
        score = beat_buy_hold_test2(
            strategy, buyhold, strategy_dd, buyhold_dd
        )
        assert score > 0.5

    def test_score_in_valid_range(self):
        """Test that score is between 0 and 1."""
        strategy = {
            "15Yr": {"return": 1.10},
            "10Yr": {"return": 1.08},
            "5Yr": {"return": 1.06},
            "3Yr": {"return": 1.04},
            "2Yr": {"return": 1.02},
            "1Yr": {"return": 1.00},
        }
        buyhold = {
            "15Yr": {"return": 1.08},
            "10Yr": {"return": 1.07},
            "5Yr": {"return": 1.05},
            "3Yr": {"return": 1.03},
            "2Yr": {"return": 1.01},
            "1Yr": {"return": 0.99},
        }
        drawdowns = {
            "15Yr": -0.10, "10Yr": -0.08, "5Yr": -0.06,
            "3Yr": -0.04, "2Yr": -0.02, "1Yr": -0.01,
        }
        
        score = beat_buy_hold_test2(
            strategy, buyhold, drawdowns, drawdowns
        )
        assert 0.0 <= score <= 1.0


class TestMonteCarloBacktest:
    """Tests for MonteCarloBacktest class."""

    @pytest.fixture
    def mc_backtest(self, tmp_path):
        """Create MonteCarloBacktest instance for testing."""
        json_file = tmp_path / "test_config.json"
        json_file.write_text("{}")
        return MonteCarloBacktest(
            base_json_fn=str(json_file),
            n_trials=10,
            hold_months=[1, 2, 3],
        )

    def test_initialization(self, mc_backtest):
        """Test MonteCarloBacktest initialization."""
        assert mc_backtest.n_trials == 10
        assert mc_backtest.hold_months == [1, 2, 3]
        assert mc_backtest.results == []
        assert mc_backtest.best_params is None
        assert mc_backtest.best_sharpe == -np.inf

    def test_generate_random_params(self, mc_backtest):
        """Test random parameter generation."""
        params = mc_backtest.generate_random_params(iteration=0)
        
        # Check required keys exist.
        required_keys = [
            "numberStocksTraded", "monthsToHold", "LongPeriod",
            "stddevThreshold", "MA1", "MA2", "MA2offset", "MA3",
            "sma2factor", "rankThresholdPct", "riskDownside_min",
            "riskDownside_max", "lowPct", "hiPct",
        ]
        for key in required_keys:
            assert key in params

    def test_generate_random_params_valid_ranges(self, mc_backtest):
        """Test that generated params are in valid ranges."""
        for i in range(20):
            params = mc_backtest.generate_random_params(iteration=i)
            
            assert params["MA2"] >= 3
            assert params["MA1"] > params["MA2"]
            assert params["MA3"] == params["MA2"] + params["MA2offset"]
            assert 0.0 <= params["lowPct"] <= 100.0
            assert 0.0 <= params["hiPct"] <= 100.0

    def test_update_best_result(self, mc_backtest):
        """Test updating best result."""
        params1 = {"test": 1}
        mc_backtest.update_best_result(params1, sharpe=1.0)
        
        assert mc_backtest.best_sharpe == 1.0
        assert mc_backtest.best_params == params1
        
        # Update with better result.
        params2 = {"test": 2}
        mc_backtest.update_best_result(params2, sharpe=1.5)
        
        assert mc_backtest.best_sharpe == 1.5
        assert mc_backtest.best_params == params2
        
        # Try to update with worse result.
        params3 = {"test": 3}
        mc_backtest.update_best_result(params3, sharpe=0.5)
        
        # Best should remain unchanged.
        assert mc_backtest.best_sharpe == 1.5
        assert mc_backtest.best_params == params2
