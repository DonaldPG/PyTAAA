"""Oracle signal generation from perfect knowledge of price extrema.

This module implements the core "what-if" scenario logic: given perfect
knowledge of when stock prices will hit local lows and highs within
centered windows, generate trading signals with optional delays.
"""

import logging
import numpy as np
from datetime import date
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


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
    
    for stock_idx, symbol in enumerate(symbols):
        prices = adjClose[stock_idx, :]
        extrema_list = []
        
        # Scan valid date range (excluding edges)
        for i in range(k, num_dates - k):
            # Extract centered window
            window = prices[i-k:i+k+1]
            
            # Skip if window contains NaN
            if np.isnan(window).any():
                continue
            
            current_price = prices[i]
            window_min = np.min(window)
            window_max = np.max(window)
            
            # Check if current date is a local extremum
            is_low = (current_price == window_min)
            is_high = (current_price == window_max)
            
            # Record extremum (prefer 'low' if both - should be rare)
            if is_low:
                extrema_list.append((datearray[i], current_price, 'low', k))
            elif is_high:
                extrema_list.append((datearray[i], current_price, 'high', k))
        
        # Deduplicate consecutive extrema of same type (keep first occurrence)
        deduplicated = []
        prev_type = None
        for extremum in extrema_list:
            extrema_date, price, extrema_type, window = extremum
            if extrema_type != prev_type:
                deduplicated.append(extremum)
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
    
    # Create date-to-index mapping for fast lookup
    date_to_idx = {d: i for i, d in enumerate(datearray)}
    
    logger.info(f"Generating oracle signals for {num_stocks} symbols × {num_dates} dates")
    
    total_segments = 0
    
    for stock_idx, symbol in enumerate(symbols):
        extrema_list = extrema_dict.get(symbol, [])
        
        if not extrema_list:
            # No extrema detected for this symbol - signal stays 0.0
            continue
        
        # Process extrema in chronological order to create low→high segments
        i = 0
        while i < len(extrema_list):
            extrema_date, price, extrema_type, window = extrema_list[i]
            
            if extrema_type == 'low':
                # Look for next high to define segment
                low_idx = date_to_idx[extrema_date]
                high_idx = None
                
                for j in range(i + 1, len(extrema_list)):
                    next_date, next_price, next_type, _ = extrema_list[j]
                    if next_type == 'high':
                        high_idx = date_to_idx[next_date]
                        break
                
                if high_idx is not None:
                    # Set signal to 1.0 for segment [low_idx, high_idx)
                    # Note: signal turns ON at low, turns OFF at high
                    signal2D[stock_idx, low_idx:high_idx] = 1.0
                    total_segments += 1
                    i = j + 1  # Skip to next extremum after the high
                else:
                    # No matching high found - skip this low
                    i += 1
            else:
                # High without preceding low - skip
                i += 1
    
    # Calculate coverage statistics
    signal_active = (signal2D == 1.0).sum()
    total_cells = signal2D.size
    coverage_pct = 100.0 * signal_active / total_cells if total_cells > 0 else 0.0
    
    logger.info(f"Generated {total_segments} low→high segments")
    logger.info(f"Signal coverage: {signal_active}/{total_cells} cells ({coverage_pct:.1f}%)")
    
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
        logger.info("No delay applied (days_delay=0)")
        return signal2D.copy()
    
    num_stocks, num_dates = signal2D.shape
    signal_delayed = np.zeros_like(signal2D)
    
    # Shift signal to the right by days_delay
    # signal_delayed[:, j] = signal[:, j - days_delay] for j >= days_delay
    if days_delay < num_dates:
        signal_delayed[:, days_delay:] = signal2D[:, :-days_delay]
    
    logger.info(f"Applied {days_delay}-day delay: first {days_delay} dates have zero signal")
    
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
    
    logger.info(f"Generating {len(extrema_windows) * len(delays)} signal scenarios")
    logger.info(f"Extrema windows: {extrema_windows}")
    logger.info(f"Delays: {delays}")
    
    for window in extrema_windows:
        # Detect extrema for this window size
        extrema_dict = detect_centered_extrema(adjClose, window, datearray, symbols)
        
        # Generate base signal (no delay)
        signal_base = generate_oracle_signal2D(
            extrema_dict, symbols, datearray, adjClose.shape
        )
        
        # Apply each delay to create scenarios
        for delay in delays:
            signal_delayed = apply_delay(signal_base, delay, datearray)
            scenarios[(window, delay)] = signal_delayed
            logger.info(f"Scenario (window={window}, delay={delay}): shape={signal_delayed.shape}")
    
    return scenarios
