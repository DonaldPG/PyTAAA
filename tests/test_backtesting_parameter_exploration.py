"""Tests for functions.backtesting.parameter_exploration module."""

import pytest

from functions.backtesting.parameter_exploration import (
    _RANGES,
    _derive_scenario,
    _get_ranges,
    generate_random_parameters,
    random_triangle,
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

# Representative JSON config dict used by tests that exercise JSON-based modes.
_SAMPLE_PARAMS = {
    "symbols_file": "Naz100_Symbols.txt",
    "uptrendSignalMethod": "percentileChannels",
    "numberStocksTraded": 6,
    "monthsToHold": 1,
    "LongPeriod": 412,
    "stddevThreshold": 8.495,
    "MA1": 264,
    "MA2": 22,
    "MA2offset": 4,
    "sma2factor": 3.495,
    "rankThresholdPct": 0.3210,
    "riskDownside_min": 0.855876,
    "riskDownside_max": 16.9086,
    "sma_filt_val": 0.02988,
    "lowPct": 20.0,
    "hiPct": 80.0,
}


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


class TestDeriveScenario:
    """Tests for _derive_scenario helper."""

    def test_naz100_pine(self):
        """Naz100 + percentileChannels → naz100_pine."""
        assert _derive_scenario({
            "symbols_file": "Naz100_Symbols.txt",
            "uptrendSignalMethod": "percentileChannels",
        }) == "naz100_pine"

    def test_naz100_hma(self):
        """Naz100 + HMAs → naz100_hma."""
        assert _derive_scenario({
            "symbols_file": "/data/naz100_hma/Naz100_Symbols.txt",
            "uptrendSignalMethod": "HMAs",
        }) == "naz100_hma"

    def test_naz100_pi(self):
        """Naz100 + SMAs → naz100_pi."""
        assert _derive_scenario({
            "symbols_file": "Naz100_Symbols.txt",
            "uptrendSignalMethod": "SMAs",
        }) == "naz100_pi"

    def test_sp500_pine(self):
        """SP500 + percentileChannels → sp500_pine."""
        assert _derive_scenario({
            "symbols_file": "SP500_Symbols.txt",
            "uptrendSignalMethod": "percentileChannels",
        }) == "sp500_pine"

    def test_sp500_hma(self):
        """SP500 + HMAs → sp500_hma."""
        assert _derive_scenario({
            "symbols_file": "sp500_symbols.txt",
            "uptrendSignalMethod": "HMAs",
        }) == "sp500_hma"

    def test_unknown_falls_back_to_naz100_pine(self):
        """Unrecognised symbols_file falls back to naz100_pine."""
        result = _derive_scenario({
            "symbols_file": "unknown_symbols.txt",
            "uptrendSignalMethod": "percentileChannels",
        })
        assert result == "naz100_pine"

    def test_get_ranges_returns_correct_sub_dict(self):
        """_get_ranges returns the right sub-dict for a given params."""
        params = {
            "symbols_file": "SP500_Symbols.txt",
            "uptrendSignalMethod": "HMAs",
        }
        result = _get_ranges(params)
        assert result is _RANGES["sp500_hma"]


class TestGenerateRandomParameters:
    """Tests for generate_random_parameters function."""

    def test_exploration_phase_returns_dict(self):
        """Exploration phase (iter_num=0) returns a dict with all keys."""
        # With total_trials=10, mid=4 → iter_num 0 is exploration.
        p = generate_random_parameters(
            _HOLD_MONTHS, iter_num=0, total_trials=10,
            params=_SAMPLE_PARAMS,
        )
        assert isinstance(p, dict)

    def test_returns_required_keys_all_phases(self):
        """All required keys must be present for every phase."""
        # With total_trials=10: exploration=0..3, json+one=4..8, exact=9.
        for iter_num in [0, 3, 4, 8, 9]:
            p = generate_random_parameters(
                _HOLD_MONTHS, iter_num=iter_num, total_trials=10,
                params=_SAMPLE_PARAMS,
            )
            missing = _REQUIRED_KEYS - set(p.keys())
            assert not missing, (
                f"iter_num={iter_num}: missing keys {missing}"
            )

    def test_json_plus_one_phase_has_param_number_to_vary(self):
        """JSON+one phase sets paramNumberToVary in [0, 12]."""
        # With total_trials=10, iter_num=5 is in JSON+one range.
        p = generate_random_parameters(
            _HOLD_MONTHS, iter_num=5, total_trials=10,
            params=_SAMPLE_PARAMS,
        )
        assert 0 <= p["paramNumberToVary"] <= 12

    def test_json_exact_phase_copies_json_values(self):
        """Last trial uses JSON config values without randomisation."""
        # With total_trials=10, iter_num=9 is the JSON-exact trial.
        p = generate_random_parameters(
            _HOLD_MONTHS, iter_num=9, total_trials=10,
            params=_SAMPLE_PARAMS,
        )
        assert p["LongPeriod"] == _SAMPLE_PARAMS["LongPeriod"]
        assert p["numberStocksTraded"] == _SAMPLE_PARAMS["numberStocksTraded"]
        assert p["MA1"] == _SAMPLE_PARAMS["MA1"]
        assert p["paramNumberToVary"] == -999

    def test_single_trial_uses_exploration_not_json_exact(self):
        """With total_trials=1, the only trial uses exploration mode."""
        p = generate_random_parameters(
            _HOLD_MONTHS, iter_num=0, total_trials=1,
            params=_SAMPLE_PARAMS,
        )
        # Exploration draws LongPeriod from the full range, not pinned to JSON.
        assert isinstance(p, dict)
        # paramNumberToVary should be -999 in exploration mode.
        assert p["paramNumberToVary"] == -999

    def test_ma2_minimum_is_3(self):
        """MA2 should always be at least 3."""
        for _ in range(30):
            p = generate_random_parameters(
                _HOLD_MONTHS, iter_num=0, total_trials=10,
                params=_SAMPLE_PARAMS,
            )
            assert p["MA2"] >= 3

    def test_ma1_greater_than_ma2_all_phases(self):
        """MA1 must always be strictly greater than MA2."""
        # With total_trials=10: exploration=0, json+one=5, exact=9.
        for _ in range(20):
            for iter_num in [0, 5, 9]:
                p = generate_random_parameters(
                    _HOLD_MONTHS, iter_num=iter_num, total_trials=10,
                    params=_SAMPLE_PARAMS,
                )
                assert p["MA1"] > p["MA2"], (
                    f"iter_num={iter_num}: MA1={p['MA1']} "
                    f"not > MA2={p['MA2']}"
                )

    def test_ma3_equals_ma2_plus_ma2offset(self):
        """MA3 must always equal MA2 + MA2offset."""
        for iter_num in [0, 5, 9]:
            p = generate_random_parameters(
                _HOLD_MONTHS, iter_num=iter_num, total_trials=10,
                params=_SAMPLE_PARAMS,
            )
            assert p["MA3"] == p["MA2"] + p["MA2offset"], (
                f"iter_num={iter_num}: MA3={p['MA3']} != "
                f"MA2({p['MA2']}) + MA2offset({p['MA2offset']})"
            )
