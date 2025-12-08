"""
Backtest package for PyTAAA trading system.

This package contains modules for backtesting configuration,
Monte Carlo simulations, and visualization.
"""

from src.backtest.config import (
    TradingConstants,
    BacktestConfig,
    FilePathConfig,
)
from src.backtest.plotting import (
    BacktestPlotter,
    calculate_plot_range,
    get_y_position,
    format_performance_metrics,
    create_monte_carlo_histogram,
    plot_signal_diagnostic,
    plot_lower_panel,
)

__all__ = [
    # Configuration
    "TradingConstants",
    "BacktestConfig",
    "FilePathConfig",
    # Plotting
    "BacktestPlotter",
    "calculate_plot_range",
    "get_y_position",
    "format_performance_metrics",
    "create_monte_carlo_histogram",
    "plot_signal_diagnostic",
    "plot_lower_panel",
]
