"""
PyTAAA Backtest package.

Contains modules for Monte Carlo backtesting and trading system optimization.

Modules:
    config: Configuration classes for trading constants, backtest parameters,
            and file paths.
    metrics: Performance metrics calculations (CAGR, Sharpe, drawdown).
    monte_carlo: Monte Carlo simulation and parameter generation.
    signals: Signal generation using percentile channels and moving averages.
    portfolio: Portfolio value calculations and rebalancing logic.
    plotting: Visualization and plot generation.
    io: File I/O operations for CSV and JSON handling.
"""

from src.backtest.config import (
    TradingConstants,
    BacktestConfig,
    FilePathConfig,
)

__all__ = [
    "TradingConstants",
    "BacktestConfig", 
    "FilePathConfig",
]
