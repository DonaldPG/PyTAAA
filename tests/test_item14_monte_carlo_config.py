"""Tests for the MonteCarloConfig dataclass (Item 14).

Covers:
- Default construction
- Custom field values
- __post_init__ validation (all guarded branches)
- from_dict() classmethod
- MonteCarloBacktest.__init__ accepts MonteCarloConfig
"""

import logging

import pytest
from unittest.mock import patch, MagicMock
from functions.MonteCarloBacktest import MonteCarloConfig


# ---------------------------------------------------------------------------
# Minimal valid model_paths used across multiple tests.
# ---------------------------------------------------------------------------
_VALID_PATHS = {"cash": "", "naz100_hma": "/tmp/data.params"}


class TestMonteCarloConfigDefaults:
    """MonteCarloConfig default values."""

    def test_required_field_only(self):
        """Supplying only model_paths uses all defaults."""
        cfg = MonteCarloConfig(model_paths=_VALID_PATHS)
        assert cfg.iterations == 50000
        assert cfg.min_iterations_for_exploit == 50
        assert cfg.max_iterations == 50000
        assert cfg.min_lookback == 20
        assert cfg.max_lookback == 300
        assert cfg.n_lookbacks == 3
        assert cfg.trading_frequency == "monthly"
        assert cfg.search_mode == "explore-exploit"
        assert cfg.verbose is False
        assert cfg.workers == 10
        assert cfg.json_config is None

    def test_model_paths_stored(self):
        """model_paths attribute matches constructor argument."""
        paths = {"cash": "", "sp500": "/tmp/sp500.params"}
        cfg = MonteCarloConfig(model_paths=paths)
        assert cfg.model_paths is paths


class TestMonteCarloConfigCustomValues:
    """MonteCarloConfig with explicitly supplied values."""

    def test_custom_iterations(self):
        cfg = MonteCarloConfig(model_paths=_VALID_PATHS, iterations=100)
        assert cfg.iterations == 100

    def test_custom_search_mode_explore(self):
        cfg = MonteCarloConfig(
            model_paths=_VALID_PATHS, search_mode="explore"
        )
        assert cfg.search_mode == "explore"

    def test_custom_search_mode_exploit(self):
        cfg = MonteCarloConfig(
            model_paths=_VALID_PATHS, search_mode="exploit"
        )
        assert cfg.search_mode == "exploit"

    def test_daily_trading_frequency(self):
        cfg = MonteCarloConfig(
            model_paths=_VALID_PATHS, trading_frequency="daily"
        )
        assert cfg.trading_frequency == "daily"

    def test_json_config_stored(self):
        json_cfg = {"metric_blending": {"enabled": True}}
        cfg = MonteCarloConfig(
            model_paths=_VALID_PATHS, json_config=json_cfg
        )
        assert cfg.json_config is json_cfg

    def test_max_iterations_synced_to_iterations(self):
        """max_iterations (legacy alias) is synced to iterations value."""
        cfg = MonteCarloConfig(
            model_paths=_VALID_PATHS, iterations=1000
        )
        # max_iterations not supplied — __post_init__ syncs it.
        assert cfg.max_iterations == cfg.iterations


class TestMonteCarloConfigValidation:
    """__post_init__ validation raises on bad inputs."""

    def test_empty_model_paths_raises(self):
        with pytest.raises(ValueError, match="model_paths must not be empty"):
            MonteCarloConfig(model_paths={})

    def test_iterations_zero_raises(self):
        with pytest.raises(ValueError, match="iterations"):
            MonteCarloConfig(model_paths=_VALID_PATHS, iterations=0)

    def test_iterations_negative_raises(self):
        with pytest.raises(ValueError, match="iterations"):
            MonteCarloConfig(model_paths=_VALID_PATHS, iterations=-1)

    def test_min_iterations_for_exploit_zero_raises(self):
        with pytest.raises(ValueError, match="min_iterations_for_exploit"):
            MonteCarloConfig(
                model_paths=_VALID_PATHS, min_iterations_for_exploit=0
            )

    def test_min_lookback_zero_raises(self):
        with pytest.raises(ValueError, match="min_lookback"):
            MonteCarloConfig(model_paths=_VALID_PATHS, min_lookback=0)

    def test_max_lookback_less_than_min_raises(self):
        with pytest.raises(ValueError, match="max_lookback"):
            MonteCarloConfig(
                model_paths=_VALID_PATHS, min_lookback=100, max_lookback=50
            )

    def test_max_lookback_equal_to_min_allowed(self):
        """max_lookback == min_lookback is a valid edge-case (single value)."""
        cfg = MonteCarloConfig(
            model_paths=_VALID_PATHS, min_lookback=50, max_lookback=50
        )
        assert cfg.min_lookback == cfg.max_lookback == 50

    def test_n_lookbacks_zero_raises(self):
        with pytest.raises(ValueError, match="n_lookbacks"):
            MonteCarloConfig(model_paths=_VALID_PATHS, n_lookbacks=0)

    def test_workers_zero_raises(self):
        with pytest.raises(ValueError, match="workers"):
            MonteCarloConfig(model_paths=_VALID_PATHS, workers=0)

    def test_invalid_search_mode_raises(self):
        with pytest.raises(ValueError, match="search_mode"):
            MonteCarloConfig(
                model_paths=_VALID_PATHS, search_mode="invalid"
            )

    def test_invalid_trading_frequency_raises(self):
        with pytest.raises(ValueError, match="trading_frequency"):
            MonteCarloConfig(
                model_paths=_VALID_PATHS, trading_frequency="weekly"
            )


class TestMonteCarloConfigFromDict:
    """MonteCarloConfig.from_dict() classmethod."""

    def test_from_dict_minimal(self):
        """from_dict with only model_paths key uses defaults."""
        cfg = MonteCarloConfig.from_dict({"model_paths": _VALID_PATHS})
        assert cfg.model_paths is _VALID_PATHS
        assert cfg.iterations == 50000

    def test_from_dict_full(self):
        """from_dict honours all known keys."""
        d = {
            "model_paths": _VALID_PATHS,
            "iterations": 200,
            "min_lookback": 10,
            "max_lookback": 150,
            "trading_frequency": "daily",
            "search_mode": "explore",
            "workers": 4,
        }
        cfg = MonteCarloConfig.from_dict(d)
        assert cfg.iterations == 200
        assert cfg.min_lookback == 10
        assert cfg.max_lookback == 150
        assert cfg.trading_frequency == "daily"
        assert cfg.search_mode == "explore"
        assert cfg.workers == 4

    def test_from_dict_ignores_unknown_keys(self):
        """Extra keys in the dict are silently ignored."""
        d = {
            "model_paths": _VALID_PATHS,
            "unknown_key": "should_be_ignored",
            "another_extra": 99,
        }
        cfg = MonteCarloConfig.from_dict(d)
        assert cfg.model_paths is _VALID_PATHS

    def test_from_dict_returns_correct_type(self):
        cfg = MonteCarloConfig.from_dict({"model_paths": _VALID_PATHS})
        assert isinstance(cfg, MonteCarloConfig)

    def test_from_dict_validation_propagates(self):
        """Validation errors from __post_init__ propagate through from_dict."""
        with pytest.raises(ValueError):
            MonteCarloConfig.from_dict(
                {"model_paths": _VALID_PATHS, "iterations": -5}
            )


class TestMonteCarloBacktestAcceptsConfig:
    """MonteCarloBacktest.__init__ consumes MonteCarloConfig correctly."""

    @patch(
        "functions.MonteCarloBacktest.MonteCarloBacktest._load_historical_data"
    )
    @patch("logging.FileHandler")
    def test_init_applies_config_attributes(
        self, mock_fh_cls, _mock_load
    ):
        """All MonteCarloConfig fields are forwarded to instance attributes."""
        from functions.MonteCarloBacktest import MonteCarloBacktest

        # Give the mocked handler a real integer level so Python's logging
        # internals (levelno >= hdlr.level) don't raise TypeError.
        mock_handler = MagicMock()
        mock_handler.level = logging.INFO
        mock_fh_cls.return_value = mock_handler

        cfg = MonteCarloConfig(
            model_paths=_VALID_PATHS,
            iterations=42,
            min_iterations_for_exploit=7,
            min_lookback=15,
            max_lookback=200,
            n_lookbacks=5,
            trading_frequency="daily",
            search_mode="exploit",
            verbose=True,
            workers=3,
        )
        mc = MonteCarloBacktest(cfg)

        assert mc.iterations == 42
        assert mc.min_iterations_for_exploit == 7
        assert mc.min_lookback == 15
        assert mc.max_lookback == 200
        assert mc.n_lookbacks == 5
        assert mc.trading_frequency == "daily"
        assert mc.search_mode == "exploit"
        assert mc.verbose is True
        assert mc.workers == 3
        assert mc.model_paths is _VALID_PATHS
        assert mc.json_config == {}  # None coerces to {}.
