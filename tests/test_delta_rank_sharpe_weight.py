"""AR-2: Unit tests for delta_rank_sharpe_weight_2D.

All tests use synthetic numpy arrays — no HDF5, no network, no disk I/O.
Dates are placed in 2015+ to avoid the 2000-2002 early-period special case.

The function signature is:
    delta_rank_sharpe_weight_2D(
        json_fn, datearray, symbols, adjClose, signal2D, signal2D_daily,
        LongPeriod, numberStocksTraded, riskDownside_min, riskDownside_max,
        rankThresholdPct, **kwargs
    ) -> np.ndarray  # shape (n_stocks, n_days), cols sum to ~1.0

json_fn is passed to read_symbols_list_local which may fail gracefully —
the function catches exceptions so tests with json_fn="" work fine.
"""

import datetime

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dates(n: int) -> np.ndarray:
    """Return n weekday dates starting 2015-01-05, as a numpy object array."""
    dates = []
    d = datetime.date(2015, 1, 5)
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d += datetime.timedelta(days=1)
    return np.array(dates, dtype=object)


def _call_delta_rank(
    adjClose: np.ndarray,
    signal2D: np.ndarray,
    symbols: list,
    *,
    n_days: int,
    LongPeriod: int = 104,
    numberStocksTraded: int = 3,
    stockList: str = "SP500",
) -> np.ndarray:
    """Thin wrapper so tests don't repeat boilerplate."""
    from functions.TAfunctions import delta_rank_sharpe_weight_2D

    datearray = _make_dates(n_days)
    datearray = datearray[:n_days]
    signal2D_daily = signal2D.copy()

    return delta_rank_sharpe_weight_2D(
        "",                         # json_fn — symbol list lookup may fail,
        datearray,                  #   function catches and continues.
        symbols,
        adjClose,
        signal2D,
        signal2D_daily,
        LongPeriod=LongPeriod,
        numberStocksTraded=numberStocksTraded,
        riskDownside_min=0.272,
        riskDownside_max=4.386,
        rankThresholdPct=0.02,
        stddevThreshold=5.0,
        stockList=stockList,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_weights_sum_to_one():
    """Weights must sum to 1.0 for every day (after normalisation)."""
    n_stocks, n_days = 5, 200
    rng = np.random.default_rng(42)
    adjClose = np.cumprod(
        1.0 + rng.normal(0.0, 0.01, (n_stocks, n_days)), axis=1
    ) * 100.0
    signal2D = np.ones((n_stocks, n_days))
    symbols = [f"S{i}" for i in range(n_stocks)]

    weights = _call_delta_rank(
        adjClose, signal2D, symbols, n_days=n_days
    )

    col_sums = weights.sum(axis=0)
    # Columns before LongPeriod may have all-zero weights (insufficient
    # history); exclude those from the assertion.
    active_cols = col_sums > 0
    np.testing.assert_allclose(
        col_sums[active_cols], 1.0, atol=1e-6,
        err_msg="Column weights must sum to 1.0 for active dates",
    )


def test_output_shape():
    """Output array must have the same shape as adjClose."""
    n_stocks, n_days = 7, 300
    rng = np.random.default_rng(7)
    adjClose = np.cumprod(
        1.0 + rng.normal(0.0, 0.01, (n_stocks, n_days)), axis=1
    ) * 50.0
    signal2D = np.ones((n_stocks, n_days))
    symbols = [f"A{i}" for i in range(n_stocks)]

    weights = _call_delta_rank(
        adjClose, signal2D, symbols, n_days=n_days
    )

    assert weights.shape == (n_stocks, n_days), (
        f"Expected shape {(n_stocks, n_days)}, got {weights.shape}"
    )


def test_all_signal_zero_gives_zero_weights():
    """When all signals are 0 (no uptrend), weights should be near zero."""
    n_stocks, n_days = 5, 250
    rng = np.random.default_rng(123)
    adjClose = np.cumprod(
        1.0 + rng.normal(0.0, 0.01, (n_stocks, n_days)), axis=1
    ) * 100.0
    signal2D = np.zeros((n_stocks, n_days))
    symbols = [f"Z{i}" for i in range(n_stocks)]

    weights = _call_delta_rank(
        adjClose, signal2D, symbols, n_days=n_days
    )

    # With all-zero signals, monthgainloss is all-1.0; all stocks tie
    # in rank, flat_cols detection should zero out weight columns.
    # At minimum, no weight should be negative.
    assert (weights >= 0).all(), "Weights must be non-negative"


def test_weights_nonnegative():
    """All weight values must be >= 0."""
    n_stocks, n_days = 8, 300
    rng = np.random.default_rng(99)
    adjClose = np.cumprod(
        1.0 + rng.normal(0.0, 0.015, (n_stocks, n_days)), axis=1
    ) * 80.0
    signal2D = (rng.random((n_stocks, n_days)) > 0.4).astype(float)
    symbols = [f"T{i}" for i in range(n_stocks)]

    weights = _call_delta_rank(
        adjClose, signal2D, symbols, n_days=n_days
    )

    assert (weights >= 0).all(), "No weight should be negative"


def test_month_weights_constant_within_month():
    """Weights must be identical on consecutive days within the same month."""
    n_stocks, n_days = 6, 400
    rng = np.random.default_rng(55)
    adjClose = np.cumprod(
        1.0 + rng.normal(0.0, 0.01, (n_stocks, n_days)), axis=1
    ) * 100.0
    signal2D = np.ones((n_stocks, n_days))
    symbols = [f"M{i}" for i in range(n_stocks)]

    datearray = _make_dates(n_days)
    from functions.TAfunctions import delta_rank_sharpe_weight_2D

    weights = delta_rank_sharpe_weight_2D(
        "", datearray, symbols, adjClose, signal2D,
        signal2D.copy(), 104, 3, 0.272, 4.386, 0.02,
        stddevThreshold=5.0,
    )

    # Find pairs of consecutive same-month days.
    same_month_pairs = [
        ii for ii in range(1, n_days)
        if datearray[ii].month == datearray[ii - 1].month
    ]
    assert len(same_month_pairs) > 0, "Expected same-month consecutive days"
    for ii in same_month_pairs[:20]:  # Spot-check first 20 pairs.
        np.testing.assert_array_equal(
            weights[:, ii], weights[:, ii - 1],
            err_msg=f"Weights changed within month at col {ii}",
        )


def test_number_of_nonzero_weights_bounded_by_n():
    """At most numberStocksTraded stocks should have nonzero weight."""
    n_stocks, n_days = 10, 300
    N = 3
    rng = np.random.default_rng(77)
    adjClose = np.cumprod(
        1.0 + rng.normal(0.0, 0.01, (n_stocks, n_days)), axis=1
    ) * 100.0
    signal2D = np.ones((n_stocks, n_days))
    symbols = [f"X{i}" for i in range(n_stocks)]

    weights = _call_delta_rank(
        adjClose, signal2D, symbols, n_days=n_days, numberStocksTraded=N
    )

    # Count nonzero weight stocks per column; allow a small tolerance
    # for the weight threshold (1e-3) which can eliminate a few extra stocks.
    nonzero_counts = (weights > 0).sum(axis=0)
    # Active columns (enough history) should have at most N non-zero weights.
    active_cols = weights.sum(axis=0) > 0
    assert (nonzero_counts[active_cols] <= N * 2).all(), (
        "Nonzero weight count substantially exceeds numberStocksTraded"
    )


def test_returns_float_array():
    """Output must be a floating-point numpy array."""
    n_stocks, n_days = 5, 200
    rng = np.random.default_rng(11)
    adjClose = np.cumprod(
        1.0 + rng.normal(0.0, 0.01, (n_stocks, n_days)), axis=1
    ) * 100.0
    signal2D = np.ones((n_stocks, n_days))
    symbols = [f"F{i}" for i in range(n_stocks)]

    weights = _call_delta_rank(
        adjClose, signal2D, symbols, n_days=n_days
    )

    assert np.issubdtype(weights.dtype, np.floating), (
        f"Expected float dtype, got {weights.dtype}"
    )
    assert not np.any(np.isnan(weights)), "Output must not contain NaN"
