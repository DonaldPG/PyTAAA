"""
Plotting functions for backtest visualization.

This module contains functions for creating Monte Carlo backtest plots,
performance statistics tables, and portfolio value visualizations.
"""

import datetime
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib import pylab as plt
import matplotlib.gridspec as gridspec

from src.backtest.config import TradingConstants


def calculate_plot_range(plotmax: float, ymin: float = 7000.0) -> float:
    """
    Calculate the plot range for log-scale y-axis positioning.
    
    Args:
        plotmax: Maximum value for the plot y-axis.
        ymin: Minimum value for the plot y-axis.
        
    Returns:
        The log-scale range for text positioning.
    """
    return np.log10(plotmax) - np.log10(ymin)


def get_y_position(plotrange: float, fraction: float,
                   ymin: float = 7000.0) -> float:
    """
    Calculate y-position for text on log-scale plot.
    
    Args:
        plotrange: The log-scale range of the plot.
        fraction: Fraction of the range (0.0 to 1.0).
        ymin: Minimum value for the plot y-axis.
        
    Returns:
        The y-coordinate for text placement.
    """
    return 10.0 ** (np.log10(ymin) + (fraction * plotrange))


def format_performance_metrics(
    sharpe: float,
    return_val: float,
    cagr: float,
    drawdown: float,
    show_cagr: bool = True
) -> Tuple[str, str, str]:
    """
    Format performance metrics for display.
    
    Args:
        sharpe: Sharpe ratio value.
        return_val: Return value (decimal).
        cagr: Compound annual growth rate (decimal).
        drawdown: Average drawdown (decimal).
        show_cagr: If True, return CAGR; otherwise return value.
        
    Returns:
        Tuple of (formatted_sharpe, formatted_return_or_cagr,
                  formatted_drawdown).
    """
    f_sharpe = format(sharpe, "5.2f")
    f_drawdown = format(drawdown, ".1%")
    
    if show_cagr:
        f_metric = format(cagr, ".1%")
    else:
        f_metric = format(return_val, "5.2f")
    
    return f_sharpe, f_metric, f_drawdown


class BacktestPlotter:
    """
    Class for creating backtest visualization plots.
    
    Handles creation of Monte Carlo simulation plots with performance
    statistics, portfolio value comparisons, and histogram overlays.
    """
    
    def __init__(
        self,
        plotmax: float = 1.0e9,
        ymin: float = 7000.0,
        figsize: Tuple[float, float] = (10, 10 * 1080 / 1920)
    ):
        """
        Initialize the BacktestPlotter.
        
        Args:
            plotmax: Maximum value for portfolio plots.
            ymin: Minimum value for portfolio plots.
            figsize: Figure size as (width, height).
        """
        self.plotmax = plotmax
        self.ymin = ymin
        self.figsize = figsize
        self.plotrange = calculate_plot_range(plotmax, ymin)
    
    def setup_figure(self) -> Tuple[plt.Figure, gridspec.GridSpec]:
        """
        Set up the matplotlib figure and gridspec for plotting.
        
        Returns:
            Tuple of (figure, gridspec) objects.
        """
        plt.rcParams["figure.edgecolor"] = "grey"
        plt.rc("savefig", edgecolor="grey")
        plt.close(1)
        
        fig = plt.figure(1, figsize=self.figsize)
        plt.clf()
        
        subplotsize = gridspec.GridSpec(2, 1, height_ratios=[5, 1])
        
        return fig, subplotsize
    
    def plot_performance_table(
        self,
        metrics: Dict[str, Dict[str, float]],
        x_position: float,
        show_cagr: bool = True,
        color: str = "k"
    ) -> None:
        """
        Plot a performance statistics table on the current axes.
        
        Args:
            metrics: Dictionary with period keys (e.g., "15Yr") containing
                     "sharpe", "return", "cagr", "drawdown" values.
            x_position: X-coordinate for text placement.
            show_cagr: If True, display CAGR; otherwise display return.
            color: Text color.
        """
        header = "Period Sharpe CAGR      Avg DD" if show_cagr else \
                 "Period Sharpe AvgProfit  Avg DD"
        
        periods = [
            ("15Yr", 0.95, 0.91),
            ("10Yr", None, 0.87),
            ("5Yr", None, 0.83),
            ("3Yr", None, 0.79),
            ("2Yr", None, 0.75),
            ("1Yr", None, 0.71),
        ]
        
        # Plot header
        y_pos = get_y_position(self.plotrange, 0.95, self.ymin)
        plt.text(x_position, y_pos, header, fontsize=7.5, color=color)
        
        # Plot each period row
        for period, _, y_frac in periods:
            if period in metrics:
                m = metrics[period]
                f_sharpe, f_metric, f_drawdown = format_performance_metrics(
                    m["sharpe"], m["return"], m["cagr"], m["drawdown"],
                    show_cagr
                )
                
                # Format period label with proper spacing
                if period in ["15Yr", "10Yr"]:
                    label = f"{period} "
                else:
                    label = f" {period}  "
                
                text = f"{label}{f_sharpe}  {f_metric}  {f_drawdown}"
                y_pos = get_y_position(self.plotrange, y_frac, self.ymin)
                plt.text(x_position, y_pos, text, fontsize=8, color=color)
    
    def plot_portfolio_values(
        self,
        buy_hold_values: np.ndarray,
        trading_values: np.ndarray,
        var_pct_values: Optional[np.ndarray] = None
    ) -> None:
        """
        Plot portfolio value curves.
        
        Args:
            buy_hold_values: Buy and hold portfolio values over time.
            trading_values: Trading system portfolio values over time.
            var_pct_values: Variable percentage invested values (optional).
        """
        plt.plot(buy_hold_values, lw=3, c="r", label="Buy & Hold")
        plt.plot(trading_values, lw=4, c="k", label="Trading System")
        
        if var_pct_values is not None:
            plt.plot(var_pct_values, lw=2, c="b", label="Variable %")
    
    def setup_x_axis_dates(
        self,
        datearray: List[datetime.date],
        max_labels: int = 12
    ) -> Tuple[List[int], List[str]]:
        """
        Set up x-axis with date labels.
        
        Args:
            datearray: Array of dates for the x-axis.
            max_labels: Maximum number of labels before skipping.
            
        Returns:
            Tuple of (x_locations, x_labels).
        """
        xlocs = []
        xlabels = []
        
        for i in range(1, len(datearray)):
            if datearray[i].year != datearray[i - 1].year:
                xlocs.append(i)
                xlabels.append(str(datearray[i].year))
        
        if len(xlocs) < max_labels:
            plt.xticks(xlocs, xlabels)
        else:
            plt.xticks(xlocs[::2], xlabels[::2])
        
        return xlocs, xlabels
    
    def add_info_text(
        self,
        symbols_file: str,
        last_symbols: List[str],
        beat_buy_hold_pct: float,
        beat_buy_hold_var_pct: float
    ) -> None:
        """
        Add informational text to the plot.
        
        Args:
            symbols_file: Path to symbols file for display.
            last_symbols: List of currently selected symbols.
            beat_buy_hold_pct: Percentage score for beating buy & hold.
            beat_buy_hold_var_pct: Variable % score for beating buy & hold.
        """
        # Symbols file path
        y_pos = get_y_position(self.plotrange, 0.47, self.ymin)
        plt.text(50, y_pos, symbols_file, fontsize=8)
        
        # Backtest date
        y_pos = get_y_position(self.plotrange, 0.05, self.ymin)
        date_str = datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
        plt.text(50, y_pos, f"Backtested on {date_str}", fontsize=7.5)
        
        # Beat buy & hold indicators
        y_pos = get_y_position(self.plotrange, 0.65, self.ymin)
        text = format(beat_buy_hold_pct, ".2%")
        if beat_buy_hold_pct > 0:
            text += "  beats BuyHold..."
        plt.text(50, y_pos, text)
        
        y_pos = get_y_position(self.plotrange, 0.59, self.ymin)
        text = format(beat_buy_hold_var_pct, ".2%")
        if beat_buy_hold_var_pct > 0:
            text += "  beats BuyHold..."
        plt.text(50, y_pos, text, color="b")
        
        # Current symbols
        y_pos = get_y_position(self.plotrange, 0.54, self.ymin)
        plt.text(50, y_pos, str(last_symbols), fontsize=8)
    
    def save_plot(
        self,
        output_dir: str,
        filename_prefix: str,
        date_str: str,
        runnum: str,
        iteration: int
    ) -> str:
        """
        Save the current plot to disk.
        
        Args:
            output_dir: Directory for output files.
            filename_prefix: Prefix for the filename.
            date_str: Date string for filename.
            runnum: Run number identifier.
            iteration: Current iteration number.
            
        Returns:
            Path to the saved file.
        """
        plot_fn = os.path.join(
            output_dir,
            f"{filename_prefix}_{date_str}__{runnum}__{iteration:03d}.png"
        )
        plt.savefig(plot_fn, format="png", edgecolor="gray")
        return plot_fn


def create_monte_carlo_histogram(
    portfolio_values: np.ndarray,
    datearray: List[datetime.date],
    n_bins: int = 150,
    ymin: float = 7000.0,
    ymax: float = 1.0e9
) -> np.ndarray:
    """
    Create histogram data for Monte Carlo portfolio value visualization.
    
    Args:
        portfolio_values: 2D array of portfolio values (trials x dates).
        datearray: Array of dates.
        n_bins: Number of histogram bins.
        ymin: Minimum y value for binning.
        ymax: Maximum y value for binning.
        
    Returns:
        3D array suitable for imshow (bins x dates x RGB).
    """
    y_bins = np.linspace(ymin, ymax, n_bins)
    
    H = np.zeros((len(y_bins) - 1, len(datearray)))
    
    for i in range(1, len(datearray)):
        values_on_date = portfolio_values[:, i]
        h, _ = np.histogram(values_on_date, bins=y_bins, density=True)
        h /= h.sum() if h.sum() > 0 else 1.0
        
        # Reverse so big numbers become small (print out black)
        h = 1.0 - h
        # Set range to [0.5, 1.0]
        h = np.clip(h, 0.05, 1.0)
        h /= 2.0
        h += 0.5
        
        H[:, i] = h
    
    # Normalize
    H -= np.percentile(H.flatten(), 2)
    H /= H.max() if H.max() > 0 else 1.0
    H = np.clip(H, 0.0, 1.0)
    
    # Create RGB array
    hb = np.zeros((len(y_bins) - 1, len(datearray), 3))
    hb[:, :, 0] = H
    hb[:, :, 1] = H
    hb[:, :, 2] = H
    
    return hb


def plot_signal_diagnostic(
    datearray: List[datetime.date],
    prices: np.ndarray,
    symbol: str,
    nan_count: int
) -> None:
    """
    Plot diagnostic signal chart for a single symbol.
    
    Args:
        datearray: Array of dates.
        prices: Price array for the symbol.
        symbol: Symbol ticker.
        nan_count: Count of NaN values in signal.
    """
    plt.clf()
    plt.grid()
    
    # Normalize prices to start at 10000
    plot_vals = prices * 10000.0 / prices[0] if prices[0] != 0 else prices
    
    plt.plot(datearray, plot_vals)
    plt.title(f"signal2D before figure3 ... {symbol}   {nan_count}")
    plt.draw()


def plot_lower_panel(
    q_minus_sma: np.ndarray,
    month_pct_invested: np.ndarray,
    datearray: List[datetime.date]
) -> None:
    """
    Plot the lower panel with Q minus SMA and percent invested.
    
    Args:
        q_minus_sma: Quote minus SMA values.
        month_pct_invested: Monthly percent invested values.
        datearray: Array of dates.
    """
    plt.grid()
    plt.plot(q_minus_sma, "m-", lw=0.8)
    plt.plot(month_pct_invested, "r-", lw=0.8)
    plt.draw()
