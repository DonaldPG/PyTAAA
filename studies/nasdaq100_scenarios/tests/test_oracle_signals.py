"""Unit tests for oracle_signals module."""

import time
from datetime import date, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pytest

from studies.nasdaq100_scenarios.oracle_signals import (
    detect_centered_extrema,
    generate_oracle_signal2D,
    apply_delay,
    generate_scenario_signals
)


class TestDetectCenteredExtrema:
    """Tests for centered-window extrema detection."""
    
    def test_sine_wave_extrema(self):
        """Detect peaks and troughs in synthetic sine wave."""
        # Create sine wave with known period
        num_dates = 200
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(num_dates)]
        
        # Single stock: sine wave with period ~40 days
        t = np.linspace(0, 5 * np.pi, num_dates)
        prices = 100 + 10 * np.sin(t)
        adjClose = prices.reshape(1, -1)
        symbols = ['SINE']
        
        # Detect with window of 10 days
        extrema_dict = detect_centered_extrema(adjClose, 10, datearray, symbols)
        
        assert 'SINE' in extrema_dict
        extrema_list = extrema_dict['SINE']
        
        # Should detect multiple extrema
        assert len(extrema_list) > 0
        
        # Extrema should alternate between low and high
        types = [e[2] for e in extrema_list]
        for i in range(len(types) - 1):
            assert types[i] != types[i+1], "Consecutive extrema should alternate type"
    
    def test_edge_dates_excluded(self):
        """First and last k dates should have no extrema."""
        num_dates = 100
        k = 20
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(num_dates)]
        
        # Monotonic increasing prices (high at end)
        prices = np.linspace(90, 110, num_dates)
        adjClose = prices.reshape(1, -1)
        symbols = ['MONO']
        
        extrema_dict = detect_centered_extrema(adjClose, k, datearray, symbols)
        extrema_list = extrema_dict['MONO']
        
        # Extract extrema dates
        extrema_indices = [datearray.index(e[0]) for e in extrema_list]
        
        # All extrema should be in range [k, num_dates-k-1]
        for idx in extrema_indices:
            assert k <= idx < num_dates - k, f"Extremum at index {idx} outside valid range"
    
    def test_flat_prices_no_extrema(self):
        """Constant prices should produce no meaningful extrema."""
        num_dates = 100
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(num_dates)]
        
        # Flat prices
        prices = np.full(num_dates, 100.0)
        adjClose = prices.reshape(1, -1)
        symbols = ['FLAT']
        
        extrema_dict = detect_centered_extrema(adjClose, 10, datearray, symbols)
        
        # Flat prices might detect every point as both low and high
        # The deduplication should reduce this significantly
        # We mostly care that it doesn't crash
        assert 'FLAT' in extrema_dict
    
    def test_nan_handling(self):
        """Windows with NaN should be skipped."""
        num_dates = 100
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(num_dates)]
        
        # Prices with NaN gap in middle
        prices = np.linspace(90, 110, num_dates)
        prices[45:55] = np.nan
        adjClose = prices.reshape(1, -1)
        symbols = ['GAPPY']
        
        extrema_dict = detect_centered_extrema(adjClose, 10, datearray, symbols)
        extrema_list = extrema_dict['GAPPY']
        
        # Extrema dates should not include NaN region
        extrema_indices = [datearray.index(e[0]) for e in extrema_list]
        for idx in extrema_indices:
            # Check window around extremum doesn't overlap NaN region
            window_start = max(0, idx - 10)
            window_end = min(num_dates, idx + 11)
            assert not np.isnan(prices[window_start:window_end]).any()
    
    def test_multiple_stocks(self):
        """Handle multiple stocks independently."""
        num_dates = 100
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(num_dates)]
        
        # Three stocks with different patterns
        t = np.linspace(0, 4 * np.pi, num_dates)
        stock1 = 100 + 10 * np.sin(t)  # Sine
        stock2 = 100 + 10 * np.cos(t)  # Cosine (90° phase shift)
        stock3 = np.linspace(90, 110, num_dates)  # Linear
        
        adjClose = np.vstack([stock1, stock2, stock3])
        symbols = ['SIN', 'COS', 'LIN']
        
        extrema_dict = detect_centered_extrema(adjClose, 10, datearray, symbols)
        
        assert len(extrema_dict) == 3
        assert all(sym in extrema_dict for sym in symbols)
        
        # Sin and Cos should have similar number of extrema
        assert len(extrema_dict['SIN']) > 0
        assert len(extrema_dict['COS']) > 0

    def test_matches_reference_implementation(self):
        """Vectorized extrema detection should match reference loop."""
        num_dates = 30
        k = 2
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(num_dates)]

        stock1 = np.array(
            [0, 1, 2, 1, 0, 1, 2, 1, 0, 1,
             2, 1, 0, 1, 2, 1, 0, 1, 2, 1,
             0, 1, 2, 1, 0, 1, 2, 1, 0, 1],
            dtype=float
        )
        stock2 = stock1 + 0.5
        stock2[12:14] = np.nan

        adjClose = np.vstack([stock1, stock2])
        symbols = ["A", "B"]

        expected = _reference_detect_centered_extrema(
            adjClose, k, datearray, symbols
        )
        actual = detect_centered_extrema(adjClose, k, datearray, symbols)

        assert actual == expected


def _reference_detect_centered_extrema(
    adjClose: np.ndarray,
    window_half_width: int,
    datearray: list,
    symbols: list
) -> Dict[str, List[Tuple[date, float, str, int]]]:
    num_stocks, num_dates = adjClose.shape
    k = window_half_width
    extrema_dict: Dict[str, List[Tuple[date, float, str, int]]] = {}

    for stock_idx, symbol in enumerate(symbols):
        prices = adjClose[stock_idx, :]
        extrema_list = []

        for i in range(k, num_dates - k):
            window = prices[i - k:i + k + 1]
            if np.isnan(window).any():
                continue

            current_price = prices[i]
            window_min = np.min(window)
            window_max = np.max(window)

            if current_price == window_min:
                extrema_list.append((datearray[i], current_price, "low", k))
            elif current_price == window_max:
                extrema_list.append((datearray[i], current_price, "high", k))

        deduplicated = []
        prev_type = None
        for extremum in extrema_list:
            extrema_date, price, extrema_type, window = extremum
            if extrema_type != prev_type:
                deduplicated.append(extremum)
                prev_type = extrema_type

        extrema_dict[symbol] = deduplicated

    return extrema_dict


class TestGenerateOracleSignal2D:
    """Tests for binary signal generation from extrema."""
    
    def test_signal_transitions(self):
        """Signal should be 1.0 from low to high, 0.0 otherwise."""
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(100)]
        symbols = ['TEST']
        
        # Simple extrema: low at day 10, high at day 30, low at day 50, high at day 70
        extrema_dict = {
            'TEST': [
                (datearray[10], 95.0, 'low', 5),
                (datearray[30], 105.0, 'high', 5),
                (datearray[50], 90.0, 'low', 5),
                (datearray[70], 110.0, 'high', 5),
            ]
        }
        
        signal2D = generate_oracle_signal2D(extrema_dict, symbols, datearray, (1, 100))
        signal = signal2D[0, :]
        
        # Check segment 1: [10, 30) should be 1.0
        assert np.all(signal[10:30] == 1.0)
        
        # Check segment 2: [50, 70) should be 1.0
        assert np.all(signal[50:70] == 1.0)
        
        # Check gaps: before 10, [30, 50), after 70 should be 0.0
        assert np.all(signal[0:10] == 0.0)
        assert np.all(signal[30:50] == 0.0)
        assert np.all(signal[70:] == 0.0)
    
    def test_no_extrema_zero_signal(self):
        """Stock with no extrema should have all-zero signal."""
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(100)]
        symbols = ['NONE']
        
        extrema_dict = {'NONE': []}
        
        signal2D = generate_oracle_signal2D(extrema_dict, symbols, datearray, (1, 100))
        
        assert np.all(signal2D == 0.0)
    
    def test_orphan_low_no_signal(self):
        """Low without subsequent high should produce no signal."""
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(100)]
        symbols = ['ORPHAN']
        
        # Low at day 10, then another low at day 30 (no high between)
        extrema_dict = {
            'ORPHAN': [
                (datearray[10], 95.0, 'low', 5),
                (datearray[30], 90.0, 'low', 5),
            ]
        }
        
        signal2D = generate_oracle_signal2D(extrema_dict, symbols, datearray, (1, 100))
        
        assert np.all(signal2D == 0.0)
    
    def test_orphan_high_no_signal(self):
        """High without preceding low should be skipped."""
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(100)]
        symbols = ['ORPHAN']
        
        # High at day 10, low at day 30, high at day 50
        extrema_dict = {
            'ORPHAN': [
                (datearray[10], 105.0, 'high', 5),
                (datearray[30], 95.0, 'low', 5),
                (datearray[50], 105.0, 'high', 5),
            ]
        }
        
        signal2D = generate_oracle_signal2D(extrema_dict, symbols, datearray, (1, 100))
        signal = signal2D[0, :]
        
        # Only segment [30, 50) should be active
        assert np.all(signal[30:50] == 1.0)
        assert np.all(signal[0:30] == 0.0)
        assert np.all(signal[50:] == 0.0)
    
    def test_multiple_stocks_independent(self):
        """Signals for multiple stocks should be independent."""
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(100)]
        symbols = ['A', 'B']
        
        extrema_dict = {
            'A': [
                (datearray[10], 95.0, 'low', 5),
                (datearray[30], 105.0, 'high', 5),
            ],
            'B': [
                (datearray[40], 90.0, 'low', 5),
                (datearray[60], 110.0, 'high', 5),
            ]
        }
        
        signal2D = generate_oracle_signal2D(extrema_dict, symbols, datearray, (2, 100))
        
        # Stock A: [10, 30) active
        assert np.all(signal2D[0, 10:30] == 1.0)
        assert np.all(signal2D[0, 40:60] == 0.0)
        
        # Stock B: [40, 60) active
        assert np.all(signal2D[1, 40:60] == 1.0)
        assert np.all(signal2D[1, 10:30] == 0.0)

    def test_matches_reference_segments(self):
        """Vectorized signal should match low→high segment logic."""
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(50)]
        symbols = ["A", "B"]

        extrema_dict = {
            "A": [
                (datearray[10], 95.0, "low", 5),
                (datearray[20], 105.0, "high", 5),
                (datearray[30], 90.0, "low", 5),
                (datearray[40], 110.0, "high", 5),
            ],
            "B": [
                (datearray[5], 90.0, "low", 5),
                (datearray[15], 100.0, "high", 5),
            ],
        }

        expected = _reference_generate_oracle_signal2D(
            extrema_dict, symbols, datearray, (2, 50)
        )
        actual = generate_oracle_signal2D(
            extrema_dict, symbols, datearray, (2, 50)
        )

        np.testing.assert_array_equal(actual, expected)


def _reference_generate_oracle_signal2D(
    extrema_dict: Dict[str, List[Tuple[date, float, str, int]]],
    symbols: List[str],
    datearray: List[date],
    adjClose_shape: Tuple[int, int]
) -> np.ndarray:
    num_stocks, num_dates = adjClose_shape
    signal2D = np.zeros(adjClose_shape, dtype=np.float32)
    date_to_idx = {d: i for i, d in enumerate(datearray)}

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
                    i = j + 1
                else:
                    i += 1
            else:
                i += 1

    return signal2D


class TestApplyDelay:
    """Tests for time delay operator."""
    
    def test_zero_delay_unchanged(self):
        """Delay of 0 should return copy of signal."""
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(50)]
        signal = np.random.rand(5, 50)
        
        signal_delayed = apply_delay(signal, 0, datearray)
        
        np.testing.assert_array_equal(signal_delayed, signal)
    
    def test_delay_shifts_right(self):
        """Delay should shift signal to the right (later in time)."""
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(50)]
        
        # Create signal with distinct pattern
        signal = np.zeros((1, 50))
        signal[0, 10:20] = 1.0
        
        delay = 5
        signal_delayed = apply_delay(signal, delay, datearray)
        
        # Original segment [10, 20) should now be at [15, 25)
        assert np.all(signal_delayed[0, 15:25] == 1.0)
        
        # First 5 days should be zero
        assert np.all(signal_delayed[0, 0:5] == 0.0)
        
        # Original position [10, 15) should be zero
        assert np.all(signal_delayed[0, 10:15] == 0.0)
    
    def test_delay_fills_with_zeros(self):
        """First d days should be zero after delay."""
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(50)]
        signal = np.ones((3, 50))
        
        delay = 10
        signal_delayed = apply_delay(signal, delay, datearray)
        
        # First 10 days should be zero
        assert np.all(signal_delayed[:, 0:10] == 0.0)
        
        # Remaining days should match original (shifted)
        np.testing.assert_array_equal(signal_delayed[:, 10:], signal[:, :-10])
    
    def test_large_delay_truncates(self):
        """Delay >= num_dates should produce all zeros."""
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(50)]
        signal = np.random.rand(2, 50)
        
        delay = 50  # Equal to num_dates
        signal_delayed = apply_delay(signal, delay, datearray)
        
        assert np.all(signal_delayed == 0.0)


class TestGenerateScenarioSignals:
    """Tests for scenario generation convenience function."""
    
    def test_generates_all_combinations(self):
        """Should generate signal for each (window, delay) combination."""
        num_dates = 200
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(num_dates)]
        
        # Sine wave
        t = np.linspace(0, 4 * np.pi, num_dates)
        prices = 100 + 10 * np.sin(t)
        adjClose = prices.reshape(1, -1)
        symbols = ['SINE']
        
        windows = [10, 20]
        delays = [0, 5, 10]
        
        scenarios = generate_scenario_signals(adjClose, symbols, datearray, windows, delays)
        
        # Should have 2 * 3 = 6 scenarios
        assert len(scenarios) == 6
        
        # Check all expected keys exist
        for window in windows:
            for delay in delays:
                assert (window, delay) in scenarios
                signal = scenarios[(window, delay)]
                assert signal.shape == adjClose.shape
    
    def test_delay_zero_uses_base_signal(self):
        """Delay=0 scenarios should be identical to base signal."""
        num_dates = 200
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(num_dates)]
        
        t = np.linspace(0, 4 * np.pi, num_dates)
        prices = 100 + 10 * np.sin(t)
        adjClose = prices.reshape(1, -1)
        symbols = ['SINE']
        
        windows = [15]
        delays = [0, 5]
        
        scenarios = generate_scenario_signals(adjClose, symbols, datearray, windows, delays)
        
        signal_no_delay = scenarios[(15, 0)]
        signal_with_delay = scenarios[(15, 5)]
        
        # They should be different (delay shifts signal)
        assert not np.array_equal(signal_no_delay, signal_with_delay)
        
        # No-delay signal should have more 1.0s (not lost to leading zeros)
        assert signal_no_delay.sum() >= signal_with_delay.sum()


# Integration test
def test_extrema_detection_performance():
    """Integration test: extrema detection on realistic data volume."""
    num_stocks = 100
    num_dates = 500
    datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(num_dates)]
    
    # Generate random walk prices
    np.random.seed(42)
    returns = np.random.randn(num_stocks, num_dates) * 0.02
    prices = 100 * np.exp(np.cumsum(returns, axis=1))
    symbols = [f"SYM{i:03d}" for i in range(num_stocks)]
    
    # Time the extrema detection
    start = time.time()
    extrema_dict = detect_centered_extrema(prices, 25, datearray, symbols)
    elapsed = time.time() - start
    
    # Should complete in reasonable time
    assert elapsed < 10.0, f"Extrema detection took {elapsed:.2f}s (expected <10s)"
    
    # Should detect extrema for most symbols
    non_empty = sum(1 for v in extrema_dict.values() if len(v) > 0)
    assert non_empty > 0.8 * num_stocks, "Most symbols should have detected extrema"
    
    print(f"\nPerformance test: {num_stocks} stocks × {num_dates} dates in {elapsed:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
