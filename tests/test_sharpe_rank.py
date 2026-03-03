"""Phase G: Unit tests for sharpeWeightedRank_2D.

All tests use synthetic numpy arrays — no HDF5, no network, no disk I/O.
Dates are placed in 2015+ to avoid the 2000-2002 early-period special case.

The function signature is:
    sharpeWeightedRank_2D(
        json_fn, datearray, symbols, adjClose, signal2D, signal2D_daily,
        LongPeriod, numberStocksTraded, riskDownside_min, riskDownside_max,
        rankThresholdPct, **kwargs
    ) -> np.ndarray  # shape (n_stocks, n_days), cols sum to 1.0

Note: json_fn is declared "not used" in the docstring; the actual
get_json_params call is placed after the return statement (dead code).
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


def _call_rank(
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
    from functions.TAfunctions import sharpeWeightedRank_2D

    datearray = _make_dates(n_days)
    # Clamp to actual available dates.
    datearray = datearray[:n_days]
    # signal2D_daily: copy of signal2D (daily resolution input).
    signal2D_daily = signal2D.copy()

    return sharpeWeightedRank_2D(
        "",                         # json_fn — not used internally
        datearray,
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
        apply_constraints=False,    # Simpler weights for determinism.
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_weights_sum_to_one():
    """Weights must sum to 1.0 for every day (after forward-fill)."""
    n_stocks, n_days = 5, 200
    rng = np.random.default_rng(0)
    # Prices with an upward drift so most stocks have positive signals/Sharpe.
    price = np.cumprod(
        1.0 + rng.normal(0.001, 0.01, (n_stocks, n_days)), axis=1
    )
    signal2D = np.ones((n_stocks, n_days), dtype=float)
    symbols = [f"S{i}" for i in range(n_stocks - 1)] + ["CASH"]

    weights = _call_rank(price, signal2D, symbols, n_days=n_days)

    col_sums = weights.sum(axis=0)
    np.testing.assert_allclose(
        col_sums, 1.0, atol=1e-9,
        err_msg="Weights do not sum to 1.0 for all days.",
    )


def test_zero_weight_for_zero_signal_stocks():
    """Stocks with signal=0 must receive zero weight."""
    n_days = 200
    # 4 stocks: first 2 signal=1, last 2 signal=0.
    n_stocks = 4
    rng = np.random.default_rng(1)
    price = np.cumprod(
        1.0 + rng.normal(0.001, 0.01, (n_stocks, n_days)), axis=1
    )
    signal2D = np.zeros((n_stocks, n_days), dtype=float)
    signal2D[0, :] = 1.0
    signal2D[1, :] = 1.0

    symbols = ["A", "B", "C", "D"]
    weights = _call_rank(
        price, signal2D, symbols,
        n_days=n_days, numberStocksTraded=4
    )

    # Stocks C (index 2) and D (index 3) must have zero weight everywhere.
    # (Forward-fill can keep earlier values but here signals are set from day 0.)
    # Skip warmup days where no stock has sufficient Sharpe history yet.
    stable_start = 120  # After LongPeriod (104)
    assert np.all(weights[2, stable_start:] == 0.0), (
        "Signal=0 stock C got nonzero weight."
    )
    assert np.all(weights[3, stable_start:] == 0.0), (
        "Signal=0 stock D got nonzero weight."
    )


def test_cash_gets_full_weight_when_all_signals_zero():
    """When all stock signals are 0, CASH must receive 100% weight."""
    n_days = 200
    n_stocks = 3
    price = np.ones((n_stocks, n_days), dtype=float)
    # All signals zero — all non-CASH stocks in downtrend.
    signal2D = np.zeros((n_stocks, n_days), dtype=float)
    symbols = ["AAPL", "GOOG", "CASH"]

    weights = _call_rank(price, signal2D, symbols, n_days=n_days)

    # After the LongPeriod warmup, CASH should hold 100% weight.
    cash_idx = symbols.index("CASH")
    assert weights[cash_idx, -1] == pytest.approx(1.0), (
        "CASH does not hold 100% when all signals are zero."
    )
    assert weights[0, -1] == pytest.approx(0.0), "Non-CASH stock has nonzero weight."
    assert weights[1, -1] == pytest.approx(0.0), "Non-CASH stock has nonzero weight."


def test_top_n_stocks_selected():
    """Only the top `numberStocksTraded` stocks should get nonzero weight."""
    n_days = 200
    n_stocks = 6     # 5 real stocks + CASH
    rng = np.random.default_rng(42)
    price = np.cumprod(
        1.0 + rng.normal(0.001, 0.02, (n_stocks, n_days)), axis=1
    )
    signal2D = np.ones((n_stocks, n_days), dtype=float)
    signal2D[-1, :] = 0.0     # CASH gets signal=0 (won't be selected)
    symbols = [f"S{i}" for i in range(n_stocks - 1)] + ["CASH"]

    top_n = 2
    weights = _call_rank(
        price, signal2D, symbols,
        n_days=n_days, numberStocksTraded=top_n,
    )

    # On the last day, at most top_n stocks (+ possibly CASH) are nonzero.
    nonzero_on_last_day = np.sum(weights[:, -1] > 0)
    assert nonzero_on_last_day <= top_n, (
        f"More than {top_n} stocks selected on last day: {nonzero_on_last_day}"
    )


def test_nan_adjclose_does_not_raise():
    """NaN values in adjClose must not cause an exception."""
    n_days = 200
    n_stocks = 3
    rng = np.random.default_rng(7)
    price = np.cumprod(
        1.0 + rng.normal(0.001, 0.01, (n_stocks, n_days)), axis=1
    )
    # Introduce NaN in the middle of the price history.
    price[0, 50:55] = np.nan
    signal2D = np.ones((n_stocks, n_days), dtype=float)
    symbols = ["X", "Y", "CASH"]

    try:
        weights = _call_rank(price, signal2D, symbols, n_days=n_days)
    except Exception as exc:
        pytest.fail(f"sharpeWeightedRank_2D raised an exception with NaN input: {exc}")

    # Output shape is correct.
    assert weights.shape == (n_stocks, n_days)


def test_single_non_cash_stock():
    """Universe of 1 real stock + CASH yields valid allocation on every day."""
    n_days = 200
    n_stocks = 2
    # Steadily rising stock → positive Sharpe.
    price = np.zeros((n_stocks, n_days), dtype=float)
    for j in range(n_days):
        price[0, j] = 100.0 + j * 0.5    # Rising stock
        price[1, j] = 1.0                 # CASH flat
    signal2D = np.ones((n_stocks, n_days), dtype=float)
    signal2D[1, :] = 0.0                   # CASH gets no signal

    symbols = ["STOCK", "CASH"]
    weights = _call_rank(
        price, signal2D, symbols, n_days=n_days, numberStocksTraded=1
    )

    assert weights.shape == (n_stocks, n_days)
    col_sums = weights.sum(axis=0)
    np.testing.assert_allclose(col_sums, 1.0, atol=1e-9)
