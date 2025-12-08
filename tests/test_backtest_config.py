"""
Unit tests for src.backtest.config module.

Tests configuration classes for trading constants, backtest parameters,
and file path configuration.
"""

import pytest
from src.backtest.config import (
    TradingConstants,
    BacktestConfig,
    FilePathConfig,
)


class TestTradingConstants:
    """Tests for TradingConstants class."""

    def test_trading_days_per_year(self):
        """Verify trading days per year constant."""
        assert TradingConstants.TRADING_DAYS_PER_YEAR == 252

    def test_trading_days_1_year(self):
        """Verify 1 year trading days constant."""
        assert TradingConstants.TRADING_DAYS_1_YEAR == 252

    def test_trading_days_2_years(self):
        """Verify 2 year trading days constant."""
        assert TradingConstants.TRADING_DAYS_2_YEARS == 504

    def test_trading_days_3_years(self):
        """Verify 3 year trading days constant."""
        assert TradingConstants.TRADING_DAYS_3_YEARS == 756

    def test_trading_days_5_years(self):
        """Verify 5 year trading days constant."""
        assert TradingConstants.TRADING_DAYS_5_YEARS == 1260

    def test_trading_days_10_years(self):
        """Verify 10 year trading days constant."""
        assert TradingConstants.TRADING_DAYS_10_YEARS == 2520

    def test_trading_days_15_years(self):
        """Verify 15 year trading days constant."""
        assert TradingConstants.TRADING_DAYS_15_YEARS == 3780

    def test_initial_portfolio_value(self):
        """Verify initial portfolio value constant."""
        assert TradingConstants.INITIAL_PORTFOLIO_VALUE == 10000.0

    def test_cagr_bounds(self):
        """Verify CAGR validation bounds."""
        assert TradingConstants.CAGR_MIN_REASONABLE == -0.5
        assert TradingConstants.CAGR_MAX_REASONABLE == 1.0

    def test_beat_buyhold_normalization(self):
        """Verify beat buy-hold normalization factor."""
        assert TradingConstants.BEAT_BUYHOLD_NORMALIZATION == 27


class TestBacktestConfig:
    """Tests for BacktestConfig class."""

    def test_default_random_trials(self):
        """Test default random trials setting."""
        assert BacktestConfig.DEFAULT_RANDOM_TRIALS == 250

    def test_runs_fraction(self):
        """Test runs fraction setting."""
        assert BacktestConfig.RUNS_FRACTION == 4

    def test_plot_settings(self):
        """Test plot configuration settings."""
        assert BacktestConfig.PLOT_Y_MIN == 7000
        assert BacktestConfig.HISTOGRAM_BINS == 150
        assert BacktestConfig.PLOT_TEXT_X_LEFT == 50
        assert BacktestConfig.PLOT_TEXT_X_RIGHT == 2250

    def test_pytaaa_defaults(self):
        """Test pyTAAA base configuration defaults."""
        assert BacktestConfig.PYTAAA_NUMBER_STOCKS_TRADED == 6
        assert BacktestConfig.PYTAAA_MONTHS_TO_HOLD == 1
        assert BacktestConfig.PYTAAA_LONG_PERIOD == 412
        assert BacktestConfig.PYTAAA_MA1 == 264
        assert BacktestConfig.PYTAAA_MA2 == 22
        assert BacktestConfig.PYTAAA_MA3 == 26

    def test_linux_defaults(self):
        """Test Linux edition configuration defaults."""
        assert BacktestConfig.LINUX_NUMBER_STOCKS_TRADED == 7
        assert BacktestConfig.LINUX_MONTHS_TO_HOLD == 1
        assert BacktestConfig.LINUX_LONG_PERIOD == 455
        assert BacktestConfig.LINUX_MA1 == 197
        assert BacktestConfig.LINUX_MA2 == 19
        assert BacktestConfig.LINUX_MA3 == 21

    def test_var_pct_defaults(self):
        """Test variable percent invested defaults."""
        assert BacktestConfig.VAR_PCT_Q_MINUS_SMA_DAYS == 355
        assert BacktestConfig.VAR_PCT_Q_MINUS_SMA_FACTOR == 0.90
        assert BacktestConfig.VAR_PCT_INVEST_SLOPE == 5.45
        assert BacktestConfig.VAR_PCT_MAX_INVESTED == 1.25

    def test_exploration_ranges(self):
        """Test exploration parameter ranges are tuples."""
        assert isinstance(BacktestConfig.EXPLORATION_LONG_PERIOD, tuple)
        assert len(BacktestConfig.EXPLORATION_LONG_PERIOD) == 3
        assert isinstance(BacktestConfig.EXPLORATION_MA1, tuple)
        assert len(BacktestConfig.EXPLORATION_MA1) == 3

    def test_symbol_file_configs(self):
        """Test symbol file configurations dictionary."""
        configs = BacktestConfig.SYMBOL_FILE_CONFIGS
        assert isinstance(configs, dict)
        assert "SP500_Symbols.txt" in configs
        
        # Check structure of config entry.
        sp500_config = configs["SP500_Symbols.txt"]
        assert len(sp500_config) == 3
        assert isinstance(sp500_config[0], str)  # runnum
        assert isinstance(sp500_config[1], float)  # plotmax
        assert isinstance(sp500_config[2], list)  # holdMonths

    def test_default_config(self):
        """Test default configuration tuple."""
        default = BacktestConfig.DEFAULT_CONFIG
        assert len(default) == 3
        assert default[0] == "run2506"
        assert default[1] == 1.e9
        assert isinstance(default[2], list)


class TestFilePathConfig:
    """Tests for FilePathConfig class."""

    def test_sp500_data_path(self):
        """Test SP500 data path constant."""
        assert "SP500" in FilePathConfig.SP500_DATA_PATH

    def test_sp500_pine_data_path(self):
        """Test SP500 pine data path constant."""
        assert "sp500_pine" in FilePathConfig.SP500_PINE_DATA_PATH

    def test_json_config_file(self):
        """Test JSON config file path."""
        assert "pytaaa_sp500_pine.json" in FilePathConfig.JSON_CONFIG_FILE

    def test_output_dir(self):
        """Test output directory path."""
        assert "pngs" in FilePathConfig.OUTPUT_DIR

    def test_filename_prefixes(self):
        """Test filename prefix constants."""
        assert "SP500" in FilePathConfig.PLOT_FILENAME_PREFIX
        assert "percentileChannels" in FilePathConfig.PLOT_FILENAME_PREFIX
        assert "sp500_pine" in FilePathConfig.CSV_FILENAME_PREFIX

    def test_get_symbols_file(self):
        """Test get_symbols_file class method."""
        path = FilePathConfig.get_symbols_file()
        assert "SP500_Symbols.txt" in path
        assert "symbols" in path

    def test_get_output_csv_filename(self):
        """Test get_output_csv_filename class method."""
        path = FilePathConfig.get_output_csv_filename(
            date_str="2025-01-01",
            runnum="run001"
        )
        assert "2025-01-01" in path
        assert "run001" in path
        assert ".csv" in path

    def test_get_plot_filename(self):
        """Test get_plot_filename class method."""
        path = FilePathConfig.get_plot_filename(
            date_str="2025-01-01",
            runnum="run001",
            iter_num=42
        )
        assert "2025-01-01" in path
        assert "run001" in path
        assert "042" in path
        assert ".png" in path

    def test_get_backtest_values_filepath(self):
        """Test get_backtest_values_filepath class method."""
        path = FilePathConfig.get_backtest_values_filepath()
        assert "backtestPortfolioValue" in path
        assert ".params" in path
