"""Oracle signal generation from perfect knowledge of price extrema.

This module implements the core "what-if" scenario logic: given perfect
knowledge of when stock prices will hit local lows and highs within
centered windows, generate trading signals with optional delays.
"""

import logging
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter

logger = logging.getLogger(__name__)


def _compute_extrema_masks(
    adjClose: np.ndarray,
    window_half_width: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute local low/high masks for centered windows."""
    _, num_dates = adjClose.shape
    k = window_half_width

    window_size = (1, (2 * k) + 1)
    valid_mask = np.isfinite(adjClose)
    valid_fraction = uniform_filter(
        valid_mask.astype(float),
        size=window_size,
        mode="constant",
        cval=0.0
    )
    valid_window = valid_fraction >= (1.0 - 1e-6)

    adj_for_min = np.where(valid_mask, adjClose, np.inf)
    adj_for_max = np.where(valid_mask, adjClose, -np.inf)

    window_min = minimum_filter(adj_for_min, size=window_size, mode="nearest")
    window_max = maximum_filter(adj_for_max, size=window_size, mode="nearest")

    low_mask = (adjClose == window_min) & valid_window
    high_mask = (adjClose == window_max) & valid_window
    high_mask &= ~low_mask

    within_edges = np.zeros(num_dates, dtype=bool)
    within_edges[k:num_dates - k] = True
    low_mask[:, ~within_edges] = False
    high_mask[:, ~within_edges] = False

    return low_mask, high_mask


def compute_centered_extrema_masks(
    adjClose: np.ndarray,
    window_half_width: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Public wrapper for centered-window low/high mask detection."""
    return _compute_extrema_masks(adjClose, window_half_width)


def _generate_slope_signal2D_from_masks(
    adjClose: np.ndarray,
    low_mask: np.ndarray,
    high_mask: np.ndarray
) -> np.ndarray:
    """Build signal from interpolated extrema and slope sign."""
    num_stocks, num_dates = adjClose.shape
    signal2D = np.zeros((num_stocks, num_dates), dtype=np.float32)

    extrema_mask = low_mask | high_mask
    extrema_prices = np.where(extrema_mask, adjClose, np.nan)
    all_indices = np.arange(num_dates)

    for stock_idx in range(num_stocks):
        finite_idx = np.flatnonzero(np.isfinite(extrema_prices[stock_idx]))
        if finite_idx.size < 2:
            continue

        finite_prices = extrema_prices[stock_idx, finite_idx]
        interpolated = np.interp(all_indices, finite_idx, finite_prices)
        slope = np.gradient(interpolated)
        signal2D[stock_idx] = np.where(slope >= 0.0, 1.0, 0.0)

        start_idx = int(finite_idx[0])
        stop_idx = int(finite_idx[-1])
        signal2D[stock_idx, :start_idx] = 0.0
        signal2D[stock_idx, stop_idx + 1:] = 0.0

    return signal2D


def detect_centered_extrema(
    adjClose: np.ndarray,
    window_half_width: int,
    datearray: List[date],
    symbols: List[str]
) -> Dict[str, List[Tuple[date, float, str, int]]]:
    """Detect local extrema using centered windows.
    
    For each date i, examines window [i-k, i+k] to identify if date i is a
    local minimum or maximum within that window. This represents "oracle
    knowledge" of future price movements.
    
    Args:
        adjClose: Price array (stocks × dates)
        window_half_width: Half-width k of centered window (total window = 2k+1)
        datearray: List of trading dates
        symbols: List of stock symbols
        
    Returns:
        Dictionary mapping symbol to list of extrema tuples:
        (extrema_date, price, type, window_half_width)
        where type is 'low' or 'high'
        
    Notes:
        - Edge dates (first k and last k) are excluded since centered windows
          would extend beyond available data
        - A date is a local low if its price equals the minimum in its window
        - A date is a local high if its price equals the maximum in its window
        - Consecutive extrema of the same type are deduplicated (keep first)
    """
    num_stocks, num_dates = adjClose.shape
    k = window_half_width
    
    # Dictionary to store extrema for each symbol
    extrema_dict = {}
    
    logger.info(f"Detecting extrema with window half-width={k} (total window={2*k+1} days)")
    logger.info(f"Valid date range: [{k}, {num_dates-k-1}] ({num_dates - 2*k} usable dates)")
    
    low_mask, high_mask = _compute_extrema_masks(adjClose, k)

    high_rows, high_cols = np.where(high_mask)
    low_rows, low_cols = np.where(low_mask)

    extrema_by_stock: Dict[int, List[Tuple[int, float, str, int]]] = {
        idx: [] for idx in range(num_stocks)
    }

    for row, col in zip(high_rows, high_cols, strict=False):
        time_idx = col
        extrema_by_stock[row].append(
            (time_idx, float(adjClose[row, time_idx]), "high", k)
        )

    for row, col in zip(low_rows, low_cols, strict=False):
        time_idx = col
        extrema_by_stock[row].append(
            (time_idx, float(adjClose[row, time_idx]), "low", k)
        )

    for stock_idx, symbol in enumerate(symbols):
        extrema_list = sorted(extrema_by_stock[stock_idx], key=lambda x: x[0])

        deduplicated = []
        prev_type = None
        for time_idx, price, extrema_type, window in extrema_list:
            if extrema_type != prev_type:
                deduplicated.append(
                    (datearray[time_idx], price, extrema_type, window)
                )
                prev_type = extrema_type

        extrema_dict[symbol] = deduplicated
    
    # Log summary statistics
    total_extrema = sum(len(v) for v in extrema_dict.values())
    avg_extrema_per_symbol = total_extrema / len(symbols) if symbols else 0
    logger.info(f"Detected {total_extrema} total extrema across {len(symbols)} symbols")
    logger.info(f"Average extrema per symbol: {avg_extrema_per_symbol:.1f}")
    
    return extrema_dict


def generate_oracle_signal2D(
    extrema_dict: Dict[str, List[Tuple[date, float, str, int]]],
    symbols: List[str],
    datearray: List[date],
    adjClose_shape: Tuple[int, int]
) -> np.ndarray:
    """Generate binary trading signals from extrema knowledge.
    
    Signal logic:
    - Signal = 1.0 during "bullish segments" (from local low to next local high)
    - Signal = 0.0 otherwise (cash position)
    - Each low→high segment is non-overlapping
    
    Args:
        extrema_dict: Output from detect_centered_extrema()
        symbols: List of stock symbols
        datearray: List of trading dates
        adjClose_shape: Shape of output signal array (stocks × dates)
        
    Returns:
        signal2D: Binary signal array (stocks × dates)
        - 1.0 = hold stock during low→high segment
        - 0.0 = hold cash otherwise
        
    Notes:
        - Signal transitions: 0→1 at low date, 1→0 at high date
        - Segments do not overlap (each high terminates previous segment)
        - Edge dates (first/last window_half_width) are 0.0
    """
    num_stocks, num_dates = adjClose_shape
    signal2D = np.zeros(adjClose_shape, dtype=np.float32)

    date_to_idx = {d: i for i, d in enumerate(datearray)}

    logger.info(
        "Generating oracle signals for %d symbols × %d dates",
        num_stocks,
        num_dates
    )

    total_segments = 0

    for stock_idx, symbol in enumerate(symbols):
        extrema_list = extrema_dict.get(symbol, [])

        if not extrema_list:
            continue

        i = 0
        while i < len(extrema_list):
            extrema_date, _, extrema_type, _ = extrema_list[i]
            if extrema_type == "low":
                low_idx = date_to_idx[extrema_date]
                high_idx = None

                for j in range(i + 1, len(extrema_list)):
                    next_date, _, next_type, _ = extrema_list[j]
                    if next_type == "high":
                        high_idx = date_to_idx[next_date]
                        break

                if high_idx is not None:
                    signal2D[stock_idx, low_idx:high_idx] = 1.0
                    total_segments += 1
                    i = j + 1
                else:
                    i += 1
            else:
                i += 1

    signal_active = (signal2D == 1.0).sum()
    total_cells = signal2D.size
    coverage_pct = 100.0 * signal_active / total_cells if total_cells > 0 else 0.0

    logger.info("Generated %d low→high segments", total_segments)
    logger.info(
        "Signal coverage: %d/%d cells (%.1f%%)",
        signal_active,
        total_cells,
        coverage_pct
    )

    return signal2D




def apply_delay(
    signal2D: np.ndarray,
    days_delay: int,
    datearray: List[date]
) -> np.ndarray:
    """Apply time delay to oracle signals.
    
    Simulates the effect of receiving perfect information with a delay.
    For delay d, the signal at date t reflects knowledge available at date t-d.
    
    Args:
        signal2D: Binary signal array (stocks × dates)
        days_delay: Number of trading days to delay signal availability
        datearray: List of trading dates (for logging)
        
    Returns:
        signal2D_delayed: Time-shifted signal array
        - First d days are 0.0 (no signal available)
        - signal_delayed[:, j] = signal[:, j-d] for j >= d
        
    Notes:
        - Delay of 0 returns signal unchanged
        - Delay shifts signal to the RIGHT (later in time)
        - This represents information lag in real trading
    """
    if days_delay == 0:
        logger.debug("No delay applied (days_delay=0)")
        return signal2D.copy()
    
    num_stocks, num_dates = signal2D.shape
    signal_delayed = np.zeros_like(signal2D)
    
    # Shift signal to the right by days_delay
    # signal_delayed[:, j] = signal[:, j - days_delay] for j >= days_delay
    if days_delay < num_dates:
        signal_delayed[:, days_delay:] = signal2D[:, :-days_delay]
    
    logger.debug(
        "Applied %d-day delay: first %d dates have zero signal",
        days_delay,
        days_delay
    )
    
    return signal_delayed


def generate_scenario_signals(
    adjClose: np.ndarray,
    symbols: List[str],
    datearray: List[date],
    extrema_windows: List[int],
    delays: List[int]
) -> Dict[Tuple[int, int], np.ndarray]:
    """Generate all signal scenarios for parameter sweep.
    
    Convenience function that generates signals for all combinations of
    extrema detection windows and information delays.
    
    Args:
        adjClose: Price array (stocks × dates)
        symbols: List of stock symbols
        datearray: List of trading dates
        extrema_windows: List of window half-widths to test
        delays: List of delay values (in trading days) to test
        
    Returns:
        Dictionary mapping (window, delay) -> signal2D array
        
    Example:
        scenarios = generate_scenario_signals(
            adjClose, symbols, datearray, [25, 50], [0, 5]
        )
        # Returns 4 scenarios: (25,0), (25,5), (50,0), (50,5)
    """
    scenarios = {}
    
    logger.info(
        "Generating %d signal scenarios (%d windows × %d delays)",
        len(extrema_windows) * len(delays),
        len(extrema_windows),
        len(delays)
    )
    logger.info("Extrema windows: %s", extrema_windows)
    logger.info("Delays: %s", delays)
    
    for window in extrema_windows:
        logger.info("Window %d: computing centered extrema masks", window)
        low_mask, high_mask = _compute_extrema_masks(adjClose, window)

        extrema_points = int(low_mask.sum() + high_mask.sum())
        logger.info(
            "Window %d: building slope signal from %d extrema points",
            window,
            extrema_points
        )
        signal_base = _generate_slope_signal2D_from_masks(
            adjClose,
            low_mask,
            high_mask
        )
        
        # Apply each delay to create scenarios
        for delay in delays:
            signal_delayed = apply_delay(signal_base, delay, datearray)
            scenarios[(window, delay)] = signal_delayed
            logger.debug(
                "Scenario (window=%d, delay=%d): shape=%s",
                window,
                delay,
                signal_delayed.shape
            )
    
    return scenarios
