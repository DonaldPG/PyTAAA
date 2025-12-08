"""
Configuration classes for PyTAAA backtesting system.

This module centralizes all constants, configuration parameters, and file paths
used throughout the backtesting system. It eliminates magic numbers and
hardcoded values from the main code.

Classes:
    TradingConstants: Trading day counts and portfolio constants.
    BacktestConfig: Monte Carlo simulation and parameter defaults.
    FilePathConfig: Data directories and output file paths.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


class TradingConstants:
    """
    Trading day constants used for performance calculations.

    These represent standard trading day counts for various time periods,
    based on approximately 252 trading days per year.
    """

    TRADING_DAYS_PER_YEAR: int = 252
    TRADING_DAYS_1_YEAR: int = 252
    TRADING_DAYS_2_YEARS: int = 504
    TRADING_DAYS_3_YEARS: int = 756
    TRADING_DAYS_5_YEARS: int = 1260
    TRADING_DAYS_10_YEARS: int = 2520
    TRADING_DAYS_15_YEARS: int = 3780

    # Initial portfolio value for backtesting
    INITIAL_PORTFOLIO_VALUE: float = 10000.0

    # CAGR validation bounds
    CAGR_MIN_REASONABLE: float = -0.5   # -50% annual return
    CAGR_MAX_REASONABLE: float = 1.0    # +100% annual return

    # Beat buy-hold test normalization factor
    BEAT_BUYHOLD_NORMALIZATION: int = 27


class BacktestConfig:
    """
    Backtest configuration parameters and default values.

    Contains Monte Carlo simulation settings, parameter ranges,
    and default values for trading system optimization.
    """

    # Monte Carlo simulation settings
    DEFAULT_RANDOM_TRIALS: int = 250
    RUNS_FRACTION: int = 4  # Fraction of trials using base vs variant params

    # Plot settings
    PLOT_Y_MIN: int = 7000
    HISTOGRAM_BINS: int = 150
    PLOT_TEXT_X_LEFT: int = 50
    PLOT_TEXT_X_RIGHT: int = 2250

    # Parameter defaults for pyTAAA base configuration
    PYTAAA_NUMBER_STOCKS_TRADED: int = 6
    PYTAAA_MONTHS_TO_HOLD: int = 1
    PYTAAA_LONG_PERIOD: int = 412
    PYTAAA_STDDEV_THRESHOLD: float = 8.495
    PYTAAA_MA1: int = 264
    PYTAAA_MA2: int = 22
    PYTAAA_MA3: int = 26
    PYTAAA_SMA2_FACTOR: float = 3.495
    PYTAAA_RANK_THRESHOLD_PCT: float = 0.3210
    PYTAAA_RISK_DOWNSIDE_MIN: float = 0.855876
    PYTAAA_RISK_DOWNSIDE_MAX: float = 16.9086
    PYTAAA_SMA_FILT_VAL: float = 0.02988

    # Parameter defaults for Linux edition
    LINUX_NUMBER_STOCKS_TRADED: int = 7
    LINUX_MONTHS_TO_HOLD: int = 1
    LINUX_LONG_PERIOD: int = 455
    LINUX_STDDEV_THRESHOLD: float = 6.12
    LINUX_MA1: int = 197
    LINUX_MA2: int = 19
    LINUX_MA3: int = 21
    LINUX_SMA2_FACTOR: float = 1.46
    LINUX_RANK_THRESHOLD_PCT: float = 0.132
    LINUX_RISK_DOWNSIDE_MIN: float = 0.5
    LINUX_RISK_DOWNSIDE_MAX: float = 7.4

    # Variable percent invested defaults
    VAR_PCT_Q_MINUS_SMA_DAYS: int = 355
    VAR_PCT_Q_MINUS_SMA_FACTOR: float = 0.90
    VAR_PCT_INVEST_SLOPE: float = 5.45
    VAR_PCT_INVEST_INTERCEPT: float = -0.01
    VAR_PCT_MAX_INVESTED: float = 1.25

    # Parameter variation ranges for random exploration
    LONG_PERIOD_RANGE: Tuple[int, int] = (55, 280)
    STDDEV_THRESHOLD_FACTOR_RANGE: Tuple[float, float] = (0.8, 1.2)
    STDDEV_THRESHOLD_BASE: float = 3.97
    NUMBER_STOCKS_RANGE: Tuple[float, float] = (1.9, 8.9)
    LOW_PCT_RANGE: Tuple[float, float] = (10.0, 30.0)
    HI_PCT_RANGE: Tuple[float, float] = (70.0, 90.0)

    # Triangular distribution parameters for exploration (low, mid, high)
    EXPLORATION_LONG_PERIOD: Tuple[int, int, int] = (190, 370, 550)
    EXPLORATION_STDDEV_THRESHOLD: Tuple[float, float, float] = (5.0, 7.50, 10.0)
    EXPLORATION_MA1: Tuple[int, int, int] = (75, 151, 300)
    EXPLORATION_MA2: Tuple[int, int, int] = (10, 20, 50)
    EXPLORATION_SMA2_FACTOR: Tuple[float, float, float] = (1.65, 2.5, 2.75)
    EXPLORATION_RANK_THRESHOLD_PCT: Tuple[float, float, float] = (0.14, 0.20, 0.26)
    EXPLORATION_RISK_DOWNSIDE_MIN: Tuple[float, float, float] = (0.50, 0.70, 0.90)
    EXPLORATION_RISK_DOWNSIDE_MAX: Tuple[float, float, float] = (8.0, 10.0, 13.0)
    EXPLORATION_SMA_FILT_VAL: Tuple[float, float, float] = (0.010, 0.015, 0.0225)

    # Symbol file configurations (basename -> (runnum, plotmax, holdMonths))
    SYMBOL_FILE_CONFIGS: Dict[str, Tuple[str, float, List[int]]] = {
        "symbols.txt": ("run2501a", 1.e5, [1, 2, 3, 4, 6, 12]),
        "Naz100_Symbols.txt": (
            "run250b", 1.e10, [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4, 6, 12]
        ),
        "biglist.txt": ("run2503", 1.e9, [1, 2, 3, 4, 6, 12]),
        "ProvidentFundSymbols.txt": ("run2504", 1.e7, [4, 6, 12]),
        "sp500_symbols.txt": ("run2505", 1.e8, [1, 2, 3, 4, 6, 12]),
        "cmg_symbols.txt": ("run2507", 1.e7, [3, 4, 6, 12]),
        "SP500_Symbols.txt": ("run2506", 1.e9, [1, 2, 3, 4, 6, 12]),
    }
    DEFAULT_CONFIG: Tuple[str, float, List[int]] = (
        "run2506", 1.e9, [1, 2, 3, 4, 6, 12]
    )


class FilePathConfig:
    """
    File path configuration for data directories and output files.

    Centralizes all hardcoded paths used in the backtesting system.
    """

    # Base data directories
    SP500_DATA_PATH: str = "/Users/donaldpg/pyTAAA_data/SP500"
    SP500_PINE_DATA_PATH: str = "/Users/donaldpg/pyTAAA_data/sp500_pine"

    # JSON configuration file
    JSON_CONFIG_FILE: str = (
        "/Users/donaldpg/pyTAAA_data/sp500_pine/pytaaa_sp500_pine.json"
    )

    # Output directories
    OUTPUT_DIR: str = "/Users/donaldpg/pyTAAA_data/sp500_pine/data_store/pngs"

    # File naming patterns - reflects SP500/percentile channel usage
    PLOT_FILENAME_PREFIX: str = "SP500-percentileChannels_montecarlo_"
    CSV_FILENAME_PREFIX: str = "sp500_pine_montecarlo_"
    BACKTEST_VALUES_FILENAME: str = (
        "pyTAAAweb_SP500percentileChannels_backtestPortfolioValue.params"
    )

    @classmethod
    def get_symbols_file(cls) -> str:
        """Get the full path to the SP500 symbols file."""
        return os.path.join(cls.SP500_DATA_PATH, "symbols", "SP500_Symbols.txt")

    @classmethod
    def get_output_csv_filename(cls, date_str: str, runnum: str) -> str:
        """Generate output CSV filename with date and run number."""
        return os.path.join(
            cls.OUTPUT_DIR,
            f"{cls.CSV_FILENAME_PREFIX}{date_str}_{runnum}.csv"
        )

    @classmethod
    def get_plot_filename(
        cls, date_str: str, runnum: str, iter_num: int
    ) -> str:
        """Generate plot filename with date, run number, and iteration."""
        return os.path.join(
            cls.OUTPUT_DIR,
            f"{cls.PLOT_FILENAME_PREFIX}{date_str}__{runnum}__{iter_num:03d}.png"
        )

    @classmethod
    def get_backtest_values_filepath(cls) -> str:
        """Get filepath for saving backtest portfolio values."""
        return os.path.join(cls.OUTPUT_DIR, cls.BACKTEST_VALUES_FILENAME)
