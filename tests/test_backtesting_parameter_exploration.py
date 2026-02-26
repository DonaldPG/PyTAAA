"""Tests for functions.backtesting.parameter_exploration module."""

import pytest

from functions.backtesting.parameter_exploration import (
    random_triangle,
    generate_random_parameters,
)

_REQUIRED_KEYS = {
    "monthsToHold",
    "numberStocksTraded",
    "LongPeriod",
    "stddevThreshold",
    "MA1",
    "MA2",
    "MA3",
    "MA2offset",
    "sma2factor",
    "rankThresholdPct",
    "riskDownside_min",
    "riskDownside_max",
    "lowPct",
    "hiPct",
    "uptrendSignalMethod",
    "sma_filt_val",
    "max_weight_factor",
    "min_weight_factor",
    "absolute_max_weight",
    "apply_constraints",
    "paramNumberToVary",
}

_HOLD_MONTHS = [1, 1, 1, 2, 3, 4, 6, 12]


class TestRandomTriangle:
    """Tests for random_triangle function."""

    def test_random_triangle_returns_in_range(self):
        """Test that returned value lies within [low, high]."""
        for _ in range(20):
            val = random_triangle(10.0, 50.0, 100.0)
            assert 10.0 <= val <= 100.0

    def test_random_triangle_size_1_returns_scalar(self):
        """Test that size=1 returns a scalar (float-like) value."""
        val = random_triangle(0.0, 0.5, 1.0, size=1)
        # Should not be a list
        assert not isinstance(val, list)
        assert isinstance(val, float)

    def test_random_triangle_size_gt_1_returns_list(self):
        """Test that size>1 returns a list."""
        result = random_triangle(0.0, 0.5, 1.0, size=5)
        assert isinstance(result, list)
        assert len(result) == 5


class TestGenerateRandomParameters:
    """Tests for generate_random_parameters function."""

    def test_generate_random_parameters_exploration_phase(self):
        """Exploration phase (iter_num=0) returns a dict with all keys."""
        params = generate_random_parameters(
            _HOLD_MONTHS, iter_num=0, total_trials=100
        )
        assert isinstance(params, dict)

    def test_generate_random_parameters_returns_required_keys(self):
        """All required keys must be present in the returned dict."""
        for iter_num in [0, 10, 25, 99]:
            params = generate_random_parameters(
                _HOLD_MONTHS, iter_num=iter_num, total_trials=100
            )
            missing = _REQUIRED_KEYS - set(params.keys())
            assert not missing, (
                f"iter_num={iter_num}: missing keys {missing}"
            )

    def test_generate_random_parameters_base_phase(self):
        """Base phase (iter_num >= total/4) starts from default params."""
        # iter_num=25 with total=100 → 25 >= 100/4 → base phase
        params = generate_random_parameters(
            _HOLD_MONTHS, iter_num=25, total_trials=100
        )
        # Base defaults: LongPeriod=412 (may be varied by ±1%)
        assert 350 <= params["LongPeriod"] <= 470

    def test_generate_random_parameters_linux_phase(self):
        """Linux edition phase (iter_num == total-1) uses linux defaults."""
        params = generate_random_parameters(
            _HOLD_MONTHS, iter_num=99, total_trials=100
        )
        # Linux default: LongPeriod=455
        assert params["LongPeriod"] == 455
        assert params["numberStocksTraded"] == 7

    def test_generate_random_parameters_ma2_minimum(self):
        """MA2 should always be at least 3."""
        for _ in range(30):
            params = generate_random_parameters(
                _HOLD_MONTHS, iter_num=0, total_trials=100
            )
            assert params["MA2"] >= 3

    def test_generate_random_parameters_ma1_greater_than_ma2(self):
        """MA1 must always be strictly greater than MA2."""
        for _ in range(30):
            for iter_num in [0, 25, 99]:
                params = generate_random_parameters(
                    _HOLD_MONTHS, iter_num=iter_num, total_trials=100
                )
                assert params["MA1"] > params["MA2"], (
                    f"iter_num={iter_num}: MA1={params['MA1']} "
                    f"not > MA2={params['MA2']}"
                )
