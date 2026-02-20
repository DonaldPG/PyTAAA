"""
test_jef_signal_zeroed.py

Phase 3 test: verify apply_rolling_window_filter() correctly zeros signals
for a stock with artificially linear-infilled prices, while leaving signals
intact for a stock with realistic noisy prices.

This test exercises the filter in isolation with fully synthetic data and
should pass both before and after the GetParams.py fix (it does not test
the integration path).

The filter trigger: window_gainloss_std < 0.001
Linear-trend prices produce std << 0.001 in every rolling window.
Noisy stock prices produce std ~ 0.01, well above the threshold.
"""

import numpy as np
import pytest
from functions.rolling_window_filter import apply_rolling_window_filter


#############################################################################
# Synthetic data helpers
#############################################################################

def make_linear_prices(n_days: int, p0: float = 20.0, slope: float = 0.01) -> np.ndarray:
    """
    Return prices increasing by `slope` per day starting at `p0`.

    The daily gain/loss ratios are all ≈ 1 + slope/p0, so the std of
    gain/loss across any rolling window is <<< 0.001 (the filter threshold).
    """
    return p0 + np.arange(n_days, dtype=float) * slope


def make_noisy_prices(n_days: int, p0: float = 100.0, daily_vol: float = 0.012,
                      seed: int = 42) -> np.ndarray:
    """
    Return prices following a random walk with 1.2% daily volatility.

    The std of gain/loss across any rolling window is ≈ daily_vol >> 0.001.
    """
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0, daily_vol, n_days)
    log_returns[0] = 0.0  # Start at p0.
    prices = p0 * np.exp(np.cumsum(log_returns))
    return prices


#############################################################################
# Fixtures
#############################################################################

@pytest.fixture
def synthetic_data():
    """
    Two-stock synthetic dataset.

    Stock 0 = JEF  : perfectly linear infilled prices → filter zeroes signals.
    Stock 1 = AAPL : noisy realistic prices → filter leaves signals intact.
    """
    n_days = 200
    window_size = 50

    jef_prices = make_linear_prices(n_days)
    aapl_prices = make_noisy_prices(n_days)

    adjClose = np.vstack([jef_prices, aapl_prices])  # Shape (2, 200).
    signal2D = np.ones_like(adjClose, dtype=float)    # All signals on.

    symbols = ["JEF", "AAPL"]
    return adjClose, signal2D, symbols, window_size


#############################################################################
# Tests
#############################################################################

class TestRollingFilterZerosLinearPrices:
    """Filter isolation tests using fully synthetic data."""

    def test_filter_zeros_jef_after_window_warmup(self, synthetic_data):
        """
        After the rolling window filter runs, every JEF signal at
        date_idx >= window_size - 1 must be 0.

        The filter only starts at date_idx == window_size - 1 (it needs a
        full window). Dates before that are left unchanged.
        """
        adjClose, signal2D, symbols, window_size = synthetic_data
        result = apply_rolling_window_filter(
            adjClose, signal2D, window_size, symbols=symbols
        )

        jef_idx = symbols.index("JEF")
        filter_start = window_size - 1  # First date where filter applies.

        # Dates within filter range must be zeroed.
        filtered_signals = result[jef_idx, filter_start:]
        assert np.all(filtered_signals == 0.0), (
            f"Expected all JEF signals from date {filter_start} onward to be "
            f"0.0, but got non-zero values at indices: "
            f"{np.where(filtered_signals != 0.0)[0] + filter_start}"
        )

    def test_filter_preserves_jef_before_window_warmup(self, synthetic_data):
        """
        JEF signals BEFORE the filter's warmup period (date_idx < window_size - 1)
        must be left at their original value of 1.0. The filter has no opinion
        about dates for which a full window is not yet available.
        """
        adjClose, signal2D, symbols, window_size = synthetic_data
        result = apply_rolling_window_filter(
            adjClose, signal2D, window_size, symbols=symbols
        )

        jef_idx = symbols.index("JEF")
        filter_start = window_size - 1

        pre_filter_signals = result[jef_idx, :filter_start]
        assert np.all(pre_filter_signals == 1.0), (
            f"Expected JEF signals before date {filter_start} to remain 1.0, "
            f"but got {pre_filter_signals}"
        )

    def test_filter_preserves_noisy_stock_signals(self, synthetic_data):
        """
        A stock with realistic daily price volatility (std(gainloss) >> 0.001)
        must have all its signals left intact.
        """
        adjClose, signal2D, symbols, window_size = synthetic_data
        result = apply_rolling_window_filter(
            adjClose, signal2D, window_size, symbols=symbols
        )

        aapl_idx = symbols.index("AAPL")
        all_signals = result[aapl_idx, :]
        zero_dates = np.where(all_signals == 0.0)[0]
        assert len(zero_dates) == 0, (
            f"Expected no AAPL signals to be filtered, but got zeros at "
            f"dates {zero_dates}. Check that noisy prices have std(gainloss) >> 0.001."
        )

    def test_filter_does_not_mutate_input_signal(self, synthetic_data):
        """
        apply_rolling_window_filter must return a copy, not modify signal2D
        in-place. Input signal2D must remain all 1.0 after the call.
        """
        adjClose, signal2D, symbols, window_size = synthetic_data
        signal2D_before = signal2D.copy()
        _ = apply_rolling_window_filter(
            adjClose, signal2D, window_size, symbols=symbols
        )
        assert np.array_equal(signal2D, signal2D_before), (
            "apply_rolling_window_filter modified the input signal2D in-place. "
            "It must return a copy."
        )

    def test_linear_prices_gainloss_std_below_threshold(self):
        """
        Sanity check: confirm the synthetic linear prices actually produce
        window_gainloss_std < 0.001 (the filter threshold).

        This validates the test assumption, not the production code.
        """
        n_days = 200
        window_size = 50
        prices = make_linear_prices(n_days)

        # Compute gainloss the same way the filter does.
        gainloss = np.ones(n_days, dtype=float)
        gainloss[1:] = prices[1:] / prices[:-1]

        # Check every window that the filter would examine.
        for date_idx in range(window_size - 1, n_days):
            start_idx = max(0, date_idx - window_size + 1)
            window_gl = gainloss[start_idx:date_idx + 1]
            std = float(np.std(window_gl))
            assert std < 0.001, (
                f"Expected gainloss std < 0.001 for linear prices at "
                f"date_idx={date_idx}, but got {std:.6f}"
            )

    def test_noisy_prices_gainloss_std_above_threshold(self):
        """
        Sanity check: confirm the synthetic noisy prices actually produce
        window_gainloss_std > 0.001 for a representative window.

        This validates the test assumption, not the production code.
        """
        n_days = 200
        window_size = 50
        prices = make_noisy_prices(n_days)

        gainloss = np.ones(n_days, dtype=float)
        gainloss[1:] = prices[1:] / prices[:-1]

        # All windows from filter start onward must exceed threshold.
        for date_idx in range(window_size - 1, n_days):
            start_idx = max(0, date_idx - window_size + 1)
            window_gl = gainloss[start_idx:date_idx + 1]
            std = float(np.std(window_gl))
            assert std >= 0.001, (
                f"Expected gainloss std >= 0.001 for noisy prices at "
                f"date_idx={date_idx}, but got {std:.6f}. "
                "Increase daily_vol in make_noisy_prices()."
            )
