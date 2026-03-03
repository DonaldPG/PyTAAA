"""Phase G: Unit tests for computeDailyBacktest.

All tests use synthetic numpy arrays. File I/O (output .params file) is
satisfied by redirecting get_performance_store() to a tmp_path via mock.
HDF5 / network calls are blocked by mocking newHighsAndLows().

The function signature is:
    computeDailyBacktest(
        json_fn, datearray, symbols, adjClose,
        numberStocksTraded=7, trade_cost=7.95, monthsToHold=4,
        LongPeriod=104, MA1=207, MA2=26, ..., uptrendSignalMethod=str
    ) -> None   # writes pyTAAAweb_backtestPortfolioValue.params
"""
import datetime
import os
import unittest.mock as mock

import numpy as np
import pytest

##############################################################################
# Pre-load computeDailyBacktest at collection time so scipy is fully in
# sys.modules before any test in test_phaseE_imports.py can trigger module-
# cache manipulation that leaves scipy submodules in a corrupted state.
##############################################################################
from functions.dailyBacktest import computeDailyBacktest  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Minimum n_days to avoid IndexError on fixed slices like PortfolioValue[-2520].
N_DAYS = 2600


def _make_business_dates(n: int, start: datetime.date) -> np.ndarray:
    """Return n weekday-only dates as a numpy object array."""
    dates: list[datetime.date] = []
    d = start
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d += datetime.timedelta(days=1)
    return np.array(dates, dtype=object)


def _minimal_json_params() -> dict:
    """Minimal dict returned by the mocked get_json_params."""
    return {
        "stockList": "SP500",
        "enable_rolling_filter": False,
    }


@pytest.fixture()
def business_dates() -> np.ndarray:
    """2600 weekday dates starting 2005-01-03 (avoids 2000-2002 early period)."""
    return _make_business_dates(N_DAYS, datetime.date(2005, 1, 3))


@pytest.fixture()
def declining_universe(business_dates: np.ndarray):
    """2-stock universe: AAPL declining strongly, CASH flat.

    Expected result: signal2D[AAPL]=0, so all weight goes to CASH.
    """
    n_days = len(business_dates)
    # AAPL declines 0.15% per day (signal will be 0 after warmup).
    price_aapl = 200.0 * (0.9985 ** np.arange(n_days))
    # CASH is always 1.0.
    price_cash = np.ones(n_days)
    adjClose = np.stack([price_aapl, price_cash])   # shape (2, n_days)
    symbols = ["AAPL", "CASH"]
    return symbols, adjClose


@pytest.fixture()
def rising_universe(business_dates: np.ndarray):
    """2-stock universe: AAPL rising strongly, CASH flat.

    Expected result: signal2D[AAPL]=1, so AAPL receives nonzero weight.
    """
    n_days = len(business_dates)
    # AAPL rises 0.15% per day (price well above long SMA after warmup).
    price_aapl = 50.0 * (1.0015 ** np.arange(n_days))
    price_cash = np.ones(n_days)
    adjClose = np.stack([price_aapl, price_cash])
    symbols = ["AAPL", "CASH"]
    return symbols, adjClose


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_runs_without_error_declining_stock(
    tmp_path, business_dates, declining_universe
):
    """computeDailyBacktest must complete without raising on declining-stock data.

    Mocks out all disk I/O so the test stays hermetic.
    """
    symbols, adjClose = declining_universe
    zeros = np.zeros(N_DAYS)

    with (
        mock.patch(
            "functions.dailyBacktest.get_json_params",
            return_value=_minimal_json_params(),
        ),
        mock.patch(
            "functions.dailyBacktest.get_performance_store",
            return_value=str(tmp_path),
        ),
        mock.patch(
            "functions.dailyBacktest.newHighsAndLows",
            return_value=(zeros, zeros, None),
        ),
    ):
        computeDailyBacktest(
            "dummy.json",
            business_dates,
            symbols,
            adjClose,
            numberStocksTraded=1,
            LongPeriod=50,
            MA1=50,
            MA2=10,
            MA2offset=5,
            sma2factor=1.0,
            monthsToHold=1,
            uptrendSignalMethod="SMAs",
        )


def test_output_file_written(
    tmp_path, business_dates, rising_universe
):
    """The portfolio-value params file must be created after a successful run."""
    symbols, adjClose = rising_universe
    zeros = np.zeros(N_DAYS)

    with (
        mock.patch(
            "functions.dailyBacktest.get_json_params",
            return_value=_minimal_json_params(),
        ),
        mock.patch(
            "functions.dailyBacktest.get_performance_store",
            return_value=str(tmp_path),
        ),
        mock.patch(
            "functions.dailyBacktest.newHighsAndLows",
            return_value=(zeros, zeros, None),
        ),
    ):
        computeDailyBacktest(
            "dummy.json",
            business_dates,
            symbols,
            adjClose,
            numberStocksTraded=1,
            LongPeriod=50,
            MA1=50,
            MA2=10,
            MA2offset=5,
            sma2factor=1.0,
            monthsToHold=1,
            uptrendSignalMethod="SMAs",
        )

    output_file = tmp_path / "pyTAAAweb_backtestPortfolioValue.params"
    assert output_file.exists(), (
        "computeDailyBacktest did not create pyTAAAweb_backtestPortfolioValue.params"
    )
    content = output_file.read_text()
    lines = [ln for ln in content.splitlines() if ln.strip()]
    assert len(lines) == N_DAYS, (
        f"Expected {N_DAYS} data rows in output file; got {len(lines)}"
    )


def test_output_file_has_numeric_columns(
    tmp_path, business_dates, declining_universe
):
    """Each output row must have at least 2 parseable float columns.

    Columns are: date B&H_value portfolio_value [highs] [lows]
    The full-success path writes 5 columns; the fallback writes 3.
    The test simply verifies numeric parsability of the value columns.
    """
    symbols, adjClose = declining_universe
    zeros = np.zeros(N_DAYS)

    with (
        mock.patch(
            "functions.dailyBacktest.get_json_params",
            return_value=_minimal_json_params(),
        ),
        mock.patch(
            "functions.dailyBacktest.get_performance_store",
            return_value=str(tmp_path),
        ),
        mock.patch(
            "functions.dailyBacktest.newHighsAndLows",
            return_value=(zeros, zeros, None),
        ),
    ):
        computeDailyBacktest(
            "dummy.json",
            business_dates,
            symbols,
            adjClose,
            numberStocksTraded=1,
            LongPeriod=50,
            MA1=50,
            MA2=10,
            MA2offset=5,
            sma2factor=1.0,
            monthsToHold=1,
            uptrendSignalMethod="SMAs",
        )

    content = (tmp_path / "pyTAAAweb_backtestPortfolioValue.params").read_text()
    for line in content.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        # First token is the date string; remaining tokens are floats.
        assert len(parts) >= 3, f"Expected ≥3 tokens per line, got: {line!r}"
        for token in parts[1:]:
            try:
                float(token)
            except ValueError:
                pytest.fail(
                    f"Non-numeric value in output file column: {token!r} "
                    f"(line: {line!r})"
                )
