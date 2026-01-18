"""
Diagnostic plot for a stock symbol.

This script creates a two-panel plot:
1. Upper subplot: Stock price over time with percentile channel bands.
2. Lower subplot: Rolling Sharpe ratio used for weight determination.

The plot helps diagnose data quality issues like repeated/infilled price data.

Usage:
    uv run python tests/plot_jef_diagnostic.py [SYMBOL]

Examples:
    uv run python tests/plot_jef_diagnostic.py JEF
    uv run python tests/plot_jef_diagnostic.py AAPL
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean
from typing import Tuple, Optional
from numpy.lib.stride_tricks import sliding_window_view

# Add project root to path.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


#############################################################################
# Configuration Constants
#############################################################################
DEFAULT_SYMBOL = "JEF"
TRADING_DAYS_PER_YEAR = 252
SHARPE_LOOKBACK_PERIOD = 252  # Days for rolling Sharpe calculation
SHARPE_THRESHOLD = 10.0  # Max absolute Sharpe value considered reliable


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with symbol attribute.
    """
    parser = argparse.ArgumentParser(
        description="Generate diagnostic plot for a stock symbol."
    )
    parser.add_argument(
        "symbol",
        nargs="?",
        default=DEFAULT_SYMBOL,
        help=f"Stock symbol to analyze (default: {DEFAULT_SYMBOL})"
    )
    return parser.parse_args()


def load_data() -> Tuple[np.ndarray, list, list, str]:
    """
    Load stock data from HDF5 file.

    Returns
    -------
    adjClose : np.ndarray
        Adjusted close prices [n_stocks, n_days].
    symbols : list
        List of stock symbols.
    datearray : list
        List of dates.
    json_fn : str
        Path to JSON configuration file.
    """
    from functions.UpdateSymbols_inHDF5 import loadQuotes_fromHDF
    from functions.GetParams import get_json_params

    json_fn = "/Users/donaldpg/pyTAAA_data/sp500_pine/pytaaa_sp500_pine.json"
    params = get_json_params(json_fn, verbose=False)
    symbols_file = params["symbols_file"]

    print(f" ... Loading data from: {symbols_file}")
    adjClose, symbols, datearray, _, _ = loadQuotes_fromHDF(
        symbols_file, json_fn
    )

    print(f" ... Loaded {len(symbols)} symbols")
    print(f" ... Date range: {datearray[0]} to {datearray[-1]}")
    print(f" ... adjClose shape: {adjClose.shape}")

    return adjClose, symbols, datearray, json_fn


def clean_data(adjClose: np.ndarray) -> np.ndarray:
    """
    Clean price data by interpolating and filling missing values.

    Parameters
    ----------
    adjClose : np.ndarray
        Raw adjusted close prices.

    Returns
    -------
    np.ndarray
        Cleaned adjusted close prices.
    """
    from functions.TAfunctions import interpolate, cleantobeginning, cleantoend

    adjClose_clean = adjClose.copy()
    for ii in range(adjClose_clean.shape[0]):
        adjClose_clean[ii, :] = interpolate(adjClose_clean[ii, :])
        adjClose_clean[ii, :] = cleantobeginning(adjClose_clean[ii, :])
        adjClose_clean[ii, :] = cleantoend(adjClose_clean[ii, :])

    print(f" ... Cleaned data, NaN count: {np.sum(np.isnan(adjClose_clean))}")

    return adjClose_clean


def compute_percentile_channels(
    adjClose: np.ndarray,
    ma1: int = 150,
    ma2: int = 20,
    ma2offset: int = 10,
    low_pct: float = 20.0,
    hi_pct: float = 80.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute percentile channels for price data.

    Parameters
    ----------
    adjClose : np.ndarray
        Adjusted close prices [n_stocks, n_days].
    ma1 : int
        Long moving average period.
    ma2 : int
        Short moving average period.
    ma2offset : int
        Offset increment for channel calculation.
    low_pct : float
        Lower percentile threshold.
    hi_pct : float
        Upper percentile threshold.

    Returns
    -------
    lowChannel : np.ndarray
        Lower percentile channel values.
    hiChannel : np.ndarray
        Upper percentile channel values.
    """
    from functions.TAfunctions import percentileChannel_2D

    print(f" ... Computing percentile channels:")
    print(f"     MA1={ma1}, MA2={ma2}, MA2offset={ma2offset}")
    print(f"     lowPct={low_pct}, hiPct={hi_pct}")

    lowChannel, hiChannel = percentileChannel_2D(
        adjClose, ma2, ma1 + 0.01, ma2offset, low_pct, hi_pct,
        verbose=False
    )

    return lowChannel, hiChannel


def rolling_sharpe_numpy(
    prices: np.ndarray,
    window: int = 252,
    risk_free_rate: float = 0.0,
    trading_days: int = 252
) -> np.ndarray:
    """
    Compute rolling Sharpe ratios using pure NumPy (vectorized).

    Parameters
    ----------
    prices : np.ndarray
        1D array of daily historical stock prices.
    window : int
        Rolling window size in days.
    risk_free_rate : float
        Daily risk-free rate (default 0).
    trading_days : int
        Number of trading days per year (default 252).

    Returns
    -------
    sharpe : np.ndarray
        Rolling Sharpe ratio values (length = len(prices) - window).
    """
    prices = np.asarray(prices)

    # Compute daily returns.
    returns = np.diff(prices) / prices[:-1]

    # Handle NaN and inf values in returns.
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    # Excess returns.
    excess_returns = returns - risk_free_rate

    # Create rolling windows (shape: [n_windows, window]).
    windows = sliding_window_view(excess_returns, window_shape=window)

    # Vectorized mean and std across axis=1.
    mean_returns = windows.mean(axis=1)
    std_returns = windows.std(axis=1, ddof=1)  # Sample std.

    # Avoid division by zero.
    std_returns = np.where(std_returns == 0, np.nan, std_returns)

    # Sharpe ratio (annualized).
    sharpe = (mean_returns / std_returns) * np.sqrt(trading_days)

    # Replace NaN with 0.
    sharpe = np.nan_to_num(sharpe, nan=0.0)

    return sharpe


def compute_rolling_sharpe_vectorized(
    adjClose: np.ndarray,
    lookback: int = SHARPE_LOOKBACK_PERIOD
) -> np.ndarray:
    """
    Compute rolling Sharpe ratio using vectorized NumPy method.

    Parameters
    ----------
    adjClose : np.ndarray
        Adjusted close prices [n_stocks, n_days].
    lookback : int
        Number of days for rolling window.

    Returns
    -------
    np.ndarray
        Rolling Sharpe ratios [n_stocks, n_days].
    """
    print(f" ... Computing rolling Sharpe ratio - VECTORIZED METHOD "
          f"(lookback={lookback} days)")

    n_stocks, n_days = adjClose.shape

    # Initialize output array.
    sharpe_full = np.zeros((n_stocks, n_days), dtype=float)

    # Compute for each stock.
    for i in range(n_stocks):
        prices = adjClose[i, :]

        # Skip if all prices are the same (no variance).
        if np.all(prices == prices[0]):
            continue

        # Compute rolling Sharpe using vectorized function.
        sharpe_values = rolling_sharpe_numpy(
            prices, window=lookback, trading_days=TRADING_DAYS_PER_YEAR
        )

        # Place results in correct positions.
        start_idx = lookback
        end_idx = start_idx + len(sharpe_values)
        if end_idx <= n_days:
            sharpe_full[i, start_idx:end_idx] = sharpe_values

    return sharpe_full


def compute_valid_data_mask(adjClose_raw: np.ndarray) -> np.ndarray:
    """
    Create a mask identifying valid (non-repeated) price data points.

    A point is considered valid (mask=1) when BOTH conditions are met:
    1. Price differs from at least one neighbor (not flat).
    2. Gainloss differs from at least one neighbor (not constant slope).

    A point is invalid (mask=0) when:
    - Price matches BOTH previous and following values (flat section), OR
    - Gainloss matches BOTH previous and following values (constant slope).

    This filters out both flat/infilled data AND linear interpolation.

    Parameters
    ----------
    adjClose_raw : np.ndarray
        Raw adjusted close prices [n_stocks, n_days] or [n_days].

    Returns
    -------
    np.ndarray
        Binary mask array (same shape as input), 1=valid, 0=invalid.
    """
    # Handle both 1D and 2D arrays.
    if adjClose_raw.ndim == 1:
        prices = adjClose_raw.reshape(1, -1)
    else:
        prices = adjClose_raw

    n_stocks, n_days = prices.shape

    # Compute daily gainloss (returns as ratio).
    gainloss = np.ones_like(prices, dtype=float)
    gainloss[:, 1:] = prices[:, 1:] / prices[:, :-1]
    gainloss[np.isnan(gainloss)] = 1.0

    # Initialize mask as ones (assume valid).
    mask = np.ones_like(prices, dtype=float)

    # For interior points (indices 1 to n_days-2).
    for i in range(1, n_days - 1):
        # Check for FLAT sections: price same as both neighbors.
        price_same_prev = np.abs(prices[:, i] - prices[:, i - 1]) < 1e-6
        price_same_next = np.abs(prices[:, i] - prices[:, i + 1]) < 1e-6
        is_flat = price_same_prev & price_same_next

        # Check for CONSTANT SLOPE: gainloss same as both neighbors.
        gl_same_prev = np.abs(gainloss[:, i] - gainloss[:, i - 1]) < 1e-8
        gl_same_next = np.abs(gainloss[:, i] - gainloss[:, i + 1]) < 1e-8
        is_constant_slope = gl_same_prev & gl_same_next

        # Invalid if flat OR constant slope.
        is_invalid = is_flat | is_constant_slope
        mask[:, i] = (~is_invalid).astype(float)

    # Handle boundary points.
    if n_days >= 3:
        # First point.
        price_same = np.abs(prices[:, 0] - prices[:, 1]) < 1e-6
        gl_same = np.abs(gainloss[:, 1] - gainloss[:, 2]) < 1e-8
        is_invalid_first = price_same | gl_same
        mask[:, 0] = (~is_invalid_first).astype(float)

        # Last point.
        price_same = np.abs(prices[:, -1] - prices[:, -2]) < 1e-6
        gl_same = np.abs(gainloss[:, -1] - gainloss[:, -2]) < 1e-8
        is_invalid_last = price_same | gl_same
        mask[:, -1] = (~is_invalid_last).astype(float)

    # Return in same shape as input.
    if adjClose_raw.ndim == 1:
        return mask.flatten()
    return mask


def find_symbol_index(symbols: list, target_symbol: str) -> Optional[int]:
    """
    Find the index of a symbol in the symbols list.

    Parameters
    ----------
    symbols : list
        List of stock symbols.
    target_symbol : str
        Symbol to find.

    Returns
    -------
    int or None
        Index of symbol, or None if not found.
    """
    try:
        return symbols.index(target_symbol)
    except ValueError:
        print(f" ... ERROR: Symbol '{target_symbol}' not found in list")
        return None


def create_diagnostic_plot(
    symbol: str,
    dates: list,
    prices: np.ndarray,
    prices_raw: np.ndarray,
    mask: np.ndarray,
    low_channel: np.ndarray,
    hi_channel: np.ndarray,
    sharpe_vectorized: np.ndarray,
    output_dir: str
) -> str:
    """
    Create a two-panel diagnostic plot for a symbol.

    Parameters
    ----------
    symbol : str
        Stock symbol.
    dates : list
        List of dates.
    prices : np.ndarray
        Cleaned price time series.
    prices_raw : np.ndarray
        Raw price time series (may contain NaN values).
    mask : np.ndarray
        Binary mask for valid data points (1=valid, 0=invalid).
    low_channel : np.ndarray
        Lower percentile channel.
    hi_channel : np.ndarray
        Upper percentile channel.
    sharpe_vectorized : np.ndarray
        Rolling Sharpe ratio time series (vectorized method).
    output_dir : str
        Directory to save the plot.

    Returns
    -------
    str
        Path to saved plot file.
    """
    print(f" ... Creating diagnostic plot for {symbol}")

    # Create figure with two subplots.
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 10), height_ratios=[3, 1],
        sharex=True
    )

    # Convert dates to numeric for plotting.
    x_vals = np.arange(len(dates))

    #########################################################################
    # Upper subplot: Price with percentile channels.
    #########################################################################
    ax1.set_title(
        f"{symbol} - Price with Percentile Channels (Red = Valid Data)",
        fontsize=14, fontweight="bold"
    )

    # Prepare raw prices for plotting (replace NaN with 0.0).
    prices_raw_plot = prices_raw.copy()
    prices_raw_plot[np.isnan(prices_raw_plot)] = 0.0

    # Create masked price array (mask * prices_raw).
    masked_prices = mask * prices_raw_plot
    masked_prices[mask == 0] = np.nan

    # Plot cleaned price (blue line).
    ax1.plot(
        x_vals, prices, "b-", linewidth=1.0,
        label="Cleaned Price", alpha=0.7
    )

    # Plot masked valid data points (red line).
    ax1.plot(
        x_vals, masked_prices, "r-", linewidth=1.5,
        label="Valid Raw Data (mask=1)", alpha=0.9
    )

    # Plot percentile channels (thin yellow lines).
    ax1.plot(
        x_vals, low_channel, "y-", linewidth=0.8,
        label="Lower Channel (20th pct)", alpha=0.8
    )
    ax1.plot(
        x_vals, hi_channel, "y-", linewidth=0.8,
        label="Upper Channel (80th pct)", alpha=0.8
    )

    # Fill between channels for visibility.
    ax1.fill_between(
        x_vals, low_channel, hi_channel,
        color="yellow", alpha=0.15, label="Channel Range"
    )

    ax1.set_ylabel("Price ($)", fontsize=12)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Add price and mask statistics annotation.
    valid_count = np.sum(mask == 1)
    total_count = len(mask)
    valid_pct = valid_count / total_count * 100

    price_stats = (
        f"Price Stats:\n"
        f"  Min: ${prices.min():.2f}\n"
        f"  Max: ${prices.max():.2f}\n"
        f"  Last: ${prices[-1]:.2f}\n"
        f"  Unique values: {len(np.unique(np.round(prices, 2)))}\n"
        f"\nMask Stats:\n"
        f"  Valid points: {valid_count}/{total_count}\n"
        f"  Valid ratio: {valid_pct:.1f}%"
    )
    ax1.text(
        0.02, 0.98, price_stats,
        transform=ax1.transAxes, fontsize=9,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    #########################################################################
    # Lower subplot: Rolling Sharpe ratio (vectorized method only).
    #########################################################################
    ax2.set_title(
        f"{symbol} - Rolling Sharpe Ratio ({SHARPE_LOOKBACK_PERIOD}-day) "
        f"- Vectorized Method",
        fontsize=12, fontweight="bold"
    )

    # Plot vectorized method Sharpe ratio (black line).
    ax2.plot(
        x_vals, sharpe_vectorized, "k-", linewidth=1.0, alpha=0.8,
        label="Vectorized Method"
    )

    # Fill positive/negative regions.
    ax2.fill_between(
        x_vals, 0, sharpe_vectorized,
        where=(sharpe_vectorized >= 0), color="green", alpha=0.3,
        label="Positive Sharpe"
    )
    ax2.fill_between(
        x_vals, 0, sharpe_vectorized,
        where=(sharpe_vectorized < 0), color="red", alpha=0.3,
        label="Negative Sharpe"
    )

    # Add horizontal lines for reference.
    ax2.axhline(y=0, color="black", linewidth=0.5, linestyle="-")
    ax2.axhline(y=1.0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax2.axhline(y=-1.0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

    ax2.set_ylabel("Sharpe Ratio", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Add Sharpe statistics annotation.
    valid_sharpe_vec = sharpe_vectorized[sharpe_vectorized != 0]
    if len(valid_sharpe_vec) > 0:
        sharpe_stats = (
            f"Sharpe Stats:\n"
            f"  Min: {valid_sharpe_vec.min():.2f}\n"
            f"  Max: {valid_sharpe_vec.max():.2f}\n"
            f"  Mean: {valid_sharpe_vec.mean():.2f}\n"
            f"  Last: {sharpe_vectorized[-1]:.2f}"
        )
    else:
        sharpe_stats = "Sharpe Stats:\n  No valid data"

    ax2.text(
        0.02, 0.98, sharpe_stats,
        transform=ax2.transAxes, fontsize=8,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    #########################################################################
    # Set x-axis labels with dates.
    #########################################################################
    n_labels = 10
    step = max(1, len(dates) // n_labels)
    tick_positions = x_vals[::step]
    tick_labels = [str(dates[i]) for i in range(0, len(dates), step)]

    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    # Adjust layout.
    plt.tight_layout()

    # Save plot.
    output_path = os.path.join(output_dir, f"plot_{symbol}_diagnostic.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f" ... Plot saved to: {output_path}")

    plt.close(fig)

    return output_path


def analyze_repeated_values(prices: np.ndarray, symbol: str) -> None:
    """
    Analyze and report repeated values in price data.

    Parameters
    ----------
    prices : np.ndarray
        Price time series.
    symbol : str
        Stock symbol for reporting.
    """
    print(f"\n ... Repeated values analysis for {symbol}:")

    # Round to 2 decimal places for comparison.
    rounded = np.round(prices, 2)
    unique_values = np.unique(rounded)
    n_total = len(prices)
    n_unique = len(unique_values)
    repeated_ratio = 1.0 - (n_unique / n_total)

    print(f"     Total data points: {n_total}")
    print(f"     Unique prices (rounded): {n_unique}")
    print(f"     Repeated ratio: {repeated_ratio:.1%}")

    # Find longest streak of repeated values.
    max_streak = 1
    current_streak = 1
    for i in range(1, len(rounded)):
        if rounded[i] == rounded[i - 1]:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 1

    print(f"     Longest streak of same price: {max_streak} days")

    # Count daily returns of exactly 0.
    returns = prices[1:] / prices[:-1] - 1.0
    zero_returns = np.sum(np.abs(returns) < 1e-8)
    print(
        f"     Days with zero return: {zero_returns} "
        f"({zero_returns/len(returns):.1%})"
    )


def main() -> None:
    """Main function to generate the diagnostic plot."""
    # Parse command-line arguments.
    args = parse_arguments()
    symbol_to_plot = args.symbol.upper()

    print("\n" + "=" * 70)
    print(f"{symbol_to_plot} DIAGNOSTIC PLOT GENERATOR")
    print("=" * 70)

    # Load and clean data.
    adjClose_raw, symbols, datearray, json_fn = load_data()
    adjClose = clean_data(adjClose_raw)

    # Find symbol index.
    symbol_idx = find_symbol_index(symbols, symbol_to_plot)
    if symbol_idx is None:
        print(f" ... Available symbols (first 20): {symbols[:20]}")
        return

    print(f" ... Found {symbol_to_plot} at index {symbol_idx}")

    # Extract symbol data.
    symbol_prices = adjClose[symbol_idx, :]
    symbol_prices_raw = adjClose_raw[symbol_idx, :]

    #########################################################################
    # Verify if raw and cleaned data are the same.
    #########################################################################
    print("\n" + "-" * 70)
    print(f"RAW vs CLEANED DATA COMPARISON FOR {symbol_to_plot}")
    print("-" * 70)

    nan_count_raw = np.sum(np.isnan(symbol_prices_raw))
    nan_count_clean = np.sum(np.isnan(symbol_prices))
    print(f" ... NaN values in raw data: {nan_count_raw}")
    print(f" ... NaN values in cleaned data: {nan_count_clean}")

    valid_mask_comparison = ~np.isnan(symbol_prices_raw)
    if np.sum(valid_mask_comparison) > 0:
        raw_valid = symbol_prices_raw[valid_mask_comparison]
        clean_valid = symbol_prices[valid_mask_comparison]

        exactly_equal = np.allclose(raw_valid, clean_valid, rtol=1e-10)
        print(f" ... Raw and cleaned values exactly equal: {exactly_equal}")

        diff = np.abs(raw_valid - clean_valid)
        print(f" ... Max absolute difference: {np.max(diff):.10f}")
        print(f" ... Mean absolute difference: {np.mean(diff):.10f}")

        tolerance = 1e-6
        mismatches = np.sum(diff > tolerance)
        print(f" ... Values differing by more than {tolerance}: {mismatches}")

    all_equal = np.allclose(
        symbol_prices_raw, symbol_prices,
        rtol=1e-10, equal_nan=True
    )
    print(f" ... Entire arrays equal (with NaN handling): {all_equal}")

    # Show sample values.
    print(f"\n ... Sample values (first 10 dates):")
    print(f"     {'Date':<12} {'Raw':>12} {'Cleaned':>12} {'Diff':>12}")
    print(f"     {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for i in range(min(10, len(datearray))):
        raw_val = symbol_prices_raw[i]
        clean_val = symbol_prices[i]
        if np.isnan(raw_val):
            diff_str = "NaN"
            raw_str = "NaN"
        else:
            diff_str = f"{clean_val - raw_val:.6f}"
            raw_str = f"{raw_val:.4f}"
        print(f"     {str(datearray[i]):<12} {raw_str:>12} "
              f"{clean_val:>12.4f} {diff_str:>12}")

    print(f"\n ... Sample values (last 10 dates):")
    print(f"     {'Date':<12} {'Raw':>12} {'Cleaned':>12} {'Diff':>12}")
    print(f"     {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for i in range(max(0, len(datearray) - 10), len(datearray)):
        raw_val = symbol_prices_raw[i]
        clean_val = symbol_prices[i]
        if np.isnan(raw_val):
            diff_str = "NaN"
            raw_str = "NaN"
        else:
            diff_str = f"{clean_val - raw_val:.6f}"
            raw_str = f"{raw_val:.4f}"
        print(f"     {str(datearray[i]):<12} {raw_str:>12} "
              f"{clean_val:>12.4f} {diff_str:>12}")

    # Analyze repeated values.
    analyze_repeated_values(symbol_prices, symbol_to_plot)

    #########################################################################
    # Compute valid data mask for raw prices.
    #########################################################################
    print("\n" + "-" * 70)
    print("COMPUTING VALID DATA MASK")
    print("-" * 70)
    symbol_mask = compute_valid_data_mask(symbol_prices_raw)
    valid_count = np.sum(symbol_mask == 1)
    total_count = len(symbol_mask)
    print(f" ... Valid data points (mask=1): {valid_count}/{total_count} "
          f"({valid_count/total_count*100:.1f}%)")

    # Compute percentile channels.
    low_channel, hi_channel = compute_percentile_channels(adjClose)
    symbol_low_channel = low_channel[symbol_idx, :]
    symbol_hi_channel = hi_channel[symbol_idx, :]

    #########################################################################
    # Compute rolling Sharpe ratio using VECTORIZED method only.
    #########################################################################
    print("\n" + "-" * 70)
    print("TIMING: Vectorized Method")
    print("-" * 70)
    start_time_vec = time.time()
    sharpe_vec = compute_rolling_sharpe_vectorized(adjClose)
    elapsed_vec = time.time() - start_time_vec
    print(f" ... Vectorized method elapsed time: {elapsed_vec:.3f} seconds")

    symbol_sharpe_vec = sharpe_vec[symbol_idx, :]

    #########################################################################
    # Apply mask to Sharpe ratio: set to 0.0 where mask==0.
    #########################################################################
    print("\n" + "-" * 70)
    print("APPLYING MASK TO SHARPE RATIO")
    print("-" * 70)

    nonzero_before = np.sum(symbol_sharpe_vec != 0)
    print(f" ... Non-zero Sharpe values before masking: {nonzero_before}")

    symbol_sharpe_vec_masked = symbol_sharpe_vec.copy()
    symbol_sharpe_vec_masked[symbol_mask == 0] = 0.0

    nonzero_after_mask = np.sum(symbol_sharpe_vec_masked != 0)
    print(f" ... Non-zero Sharpe values after data mask: {nonzero_after_mask}")
    print(f" ... Sharpe values zeroed by data mask: "
          f"{nonzero_before - nonzero_after_mask}")

    #########################################################################
    # Also set Sharpe to 0.0 when abs(Sharpe) > threshold (unreliable).
    #########################################################################
    extreme_sharpe_mask = np.abs(symbol_sharpe_vec_masked) > SHARPE_THRESHOLD
    extreme_count = np.sum(extreme_sharpe_mask)
    print(f" ... Sharpe values with |Sharpe| > {SHARPE_THRESHOLD}: "
          f"{extreme_count}")

    symbol_sharpe_vec_masked[extreme_sharpe_mask] = 0.0
    symbol_mask[extreme_sharpe_mask] = 0.0

    nonzero_after = np.sum(symbol_sharpe_vec_masked != 0)
    print(f" ... Non-zero Sharpe values after all filters: {nonzero_after}")
    print(f" ... Total Sharpe values zeroed: {nonzero_before - nonzero_after}")

    # Create output directory (tests folder).
    output_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\n ... Output directory: {output_dir}")

    # Create the diagnostic plot with masked Sharpe ratio.
    plot_path = create_diagnostic_plot(
        symbol=symbol_to_plot,
        dates=datearray,
        prices=symbol_prices,
        prices_raw=symbol_prices_raw,
        mask=symbol_mask,
        low_channel=symbol_low_channel,
        hi_channel=symbol_hi_channel,
        sharpe_vectorized=symbol_sharpe_vec_masked,
        output_dir=output_dir
    )

    print("\n" + "=" * 70)
    print("COMPLETE")
    print(f"Plot saved to: {plot_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
