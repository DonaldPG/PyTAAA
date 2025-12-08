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
from src.backtest.montecarlo import (
    MonteCarloBacktest,
    random_triangle,
    create_temporary_json,
    cleanup_temporary_json,
    calculate_sharpe_ratio,
    calculate_period_metrics,
    calculate_drawdown_metrics,
    beat_buy_hold_test,
    beat_buy_hold_test2,
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
    # Monte Carlo
    "MonteCarloBacktest",
    "random_triangle",
    "create_temporary_json",
    "cleanup_temporary_json",
    "calculate_sharpe_ratio",
    "calculate_period_metrics",
    "calculate_drawdown_metrics",
    "beat_buy_hold_test",
    "beat_buy_hold_test2",
]
