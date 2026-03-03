"""Phase G: Unit tests for computeSignal2D.

All tests use synthetic numpy arrays — no HDF5, no network, no disk I/O.

The function signatures tested are:

    from functions.ta.signal_generation import computeSignal2D
    # Also importable as:
    from functions.TAfunctions import computeSignal2D

    computeSignal2D(
        adjClose: NDArray,       # shape (n_stocks, n_dates)
        gainloss: NDArray,       # shape (n_stocks, n_dates)
        params: dict,
    ) -> NDArray | tuple[NDArray, NDArray, NDArray]
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sma_params(*, MA1: int = 50, MA2: int = 10) -> dict:
    """Return a minimal params dict for the 'SMAs' method."""
    return {
        "MA1": MA1,
        "MA2": MA2,
        "MA2offset": 5,
        "MA2factor": 1.0,
        "uptrendSignalMethod": "SMAs",
        "narrowDays": [6.0, 40.2],
        "mediumDays": [25.2, 38.3],
        "wideDays": [75.2, 512.3],
        "lowPct": 17,
        "hiPct": 84,
        "minperiod": 4,
        "maxperiod": 12,
        "incperiod": 3,
    }


def _gainloss(adjClose: np.ndarray) -> np.ndarray:
    """Compute daily gain/loss ratios matching the production formula."""
    gl = np.ones_like(adjClose)
    gl[:, 1:] = adjClose[:, 1:] / adjClose[:, :-1]
    return gl


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_uptrend_gets_signal_1():
    """A stock clearly above its long moving average receives signal 1.

    Price rises 1% per day from 100 → ~738 over 200 days.
    After the MA1=50 warm-up, the price is well above the 50-day SMA,
    so the final signal should be 1.
    """
    from functions.ta.signal_generation import computeSignal2D

    n_days = 200
    MA1 = 50
    # Single strongly uptrending stock.
    price = 100.0 * (1.01 ** np.arange(n_days))
    adjClose = price[np.newaxis, :]          # shape (1, n_days)
    gainloss = _gainloss(adjClose)

    params = _sma_params(MA1=MA1, MA2=10)
    signal2D = computeSignal2D(adjClose, gainloss, params)

    # Final day's signal for the uptrending stock must be 1.
    assert signal2D[0, -1] == 1.0, (
        f"Expected signal 1 for strongly uptrending stock; got {signal2D[0, -1]}"
    )


def test_downtrend_gets_signal_0():
    """A stock clearly below its long moving average receives signal 0.

    Price falls 1% per day from 200 → ~27 over 200 days.
    After the MA1=50 warm-up, the price is well below the 50-day SMA,
    so the final signal should be 0.
    """
    from functions.ta.signal_generation import computeSignal2D

    n_days = 200
    MA1 = 50
    # Single strongly downtrending stock.
    price = 200.0 * (0.99 ** np.arange(n_days))
    adjClose = price[np.newaxis, :]
    gainloss = _gainloss(adjClose)

    params = _sma_params(MA1=MA1, MA2=10)
    signal2D = computeSignal2D(adjClose, gainloss, params)

    # Only the last day matters for clarity — signal must be 0.
    assert signal2D[0, -1] == 0.0, (
        f"Expected signal 0 for strongly downtrending stock; got {signal2D[0, -1]}"
    )


def test_output_is_binary():
    """SMAs method must return only 0s and 1s — no intermediate values."""
    from functions.ta.signal_generation import computeSignal2D

    rng = np.random.default_rng(99)
    n_stocks, n_days = 5, 300
    adjClose = np.cumprod(
        1.0 + rng.normal(0.0005, 0.015, (n_stocks, n_days)), axis=1
    ) * 100.0
    gainloss = _gainloss(adjClose)

    params = _sma_params(MA1=100, MA2=20)
    signal2D = computeSignal2D(adjClose, gainloss, params)

    unique_vals = set(np.unique(signal2D))
    assert unique_vals <= {0.0, 1.0}, (
        f"signal2D contains non-binary values: {unique_vals - {0.0, 1.0}}"
    )


def test_nan_on_last_date_sets_signal_zero():
    """A NaN price on the final date must force the last signal to 0."""
    from functions.ta.signal_generation import computeSignal2D

    n_days = 200
    # Rising stock that would normally get signal=1.
    price = 100.0 * (1.01 ** np.arange(n_days))
    price[-1] = np.nan                        # Simulate delisted stock.
    adjClose = price[np.newaxis, :]           # shape (1, n_days)
    gainloss = _gainloss(adjClose)
    gainloss[np.isnan(gainloss)] = 1.0        # Clear NaN in gainloss.

    params = _sma_params(MA1=50, MA2=10)
    signal2D = computeSignal2D(adjClose, gainloss, params)

    assert signal2D[0, -1] == 0.0, (
        "Expected final signal=0 for stock with NaN last price; "
        f"got {signal2D[0, -1]}"
    )


def test_hma_method_returns_correct_shape():
    """HMAs method must return a 2D array with the same shape as adjClose."""
    from functions.ta.signal_generation import computeSignal2D

    rng = np.random.default_rng(5)
    n_stocks, n_days = 3, 150
    adjClose = np.cumprod(
        1.0 + rng.normal(0.0008, 0.012, (n_stocks, n_days)), axis=1
    ) * 50.0
    gainloss = _gainloss(adjClose)

    params = _sma_params(MA1=40, MA2=8)
    params["uptrendSignalMethod"] = "HMAs"

    signal2D = computeSignal2D(adjClose, gainloss, params)

    assert signal2D.shape == (n_stocks, n_days), (
        f"Expected shape {(n_stocks, n_days)}, got {signal2D.shape}"
    )
    # Values must still be binary.
    assert np.all((signal2D == 0.0) | (signal2D == 1.0))


def test_multiple_stocks_independent_signals():
    """Each stock's signal is based on its own price history.

    Stock A trends up (signal=1), Stock B trends down (signal=0).
    Their signals on the same day should differ.
    """
    from functions.ta.signal_generation import computeSignal2D

    n_days = 200
    MA1 = 50
    price_up = 100.0 * (1.015 ** np.arange(n_days))   # Strong uptrend.
    price_dn = 200.0 * (0.985 ** np.arange(n_days))   # Strong downtrend.
    adjClose = np.stack([price_up, price_dn])          # shape (2, n_days)
    gainloss = _gainloss(adjClose)

    params = _sma_params(MA1=MA1, MA2=10)
    signal2D = computeSignal2D(adjClose, gainloss, params)

    assert signal2D[0, -1] == 1.0, "Uptrending stock A should have signal=1."
    assert signal2D[1, -1] == 0.0, "Downtrending stock B should have signal=0."
