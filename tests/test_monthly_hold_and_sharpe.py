import datetime
import numpy as np

from functions import dailyBacktest
from functions.TAfunctions import sharpeWeightedRank_2D


def make_datearray(start_date, days):
    return [start_date + datetime.timedelta(days=i) for i in range(days)]


def old_monthly_hold(signal2D, datearray, monthsToHold=1):
    # replicate previous logic: forward-fill except on rebalance dates
    arr = signal2D.copy()
    n_days = arr.shape[1]
    for jj in range(1, n_days):
        is_rebalance = (
            (datearray[jj].month != datearray[jj-1].month) and
            ((datearray[jj].month - 1) % monthsToHold == 0)
        )
        if not is_rebalance:
            arr[:, jj] = arr[:, jj - 1]
    return arr


def new_monthly_hold(signal2D_daily, datearray, monthsToHold=1):
    # replicate new logic introduced in computeDailyBacktest
    n_days = signal2D_daily.shape[1]
    arr = np.zeros_like(signal2D_daily)
    last_month_signals = signal2D_daily[:, 0].copy()
    arr[:, 0] = last_month_signals
    for jj in range(1, n_days):
        is_rebalance = (
            (datearray[jj].month != datearray[jj-1].month) and
            ((datearray[jj].month - 1) % monthsToHold == 0)
        )
        if is_rebalance:
            last_month_signals = signal2D_daily[:, jj].copy()
        arr[:, jj] = last_month_signals
    return arr


def test_monthly_hold_difference():
    # Create 2 stocks (JEF at index 0) across days crossing a month boundary
    start = datetime.date(2014, 12, 25)
    days = 40
    datearray = make_datearray(start, days)
    n_stocks = 2
    n_days = days

    # signal2D_pre is what computeSignal2D produced (e.g., all ones)
    signal2D_pre = np.ones((n_stocks, n_days), dtype=float)

    # signal2D_daily is the filtered daily signals (rolling filter applied)
    signal2D_daily = signal2D_pre.copy()
    # Simulate rolling filter zeroing JEF (index 0) on the rebalance date (first of month)
    # Find index where month changes to Jan 2015
    rebalance_idx = None
    for i in range(1, n_days):
        if datearray[i].month != datearray[i-1].month:
            rebalance_idx = i
            break
    assert rebalance_idx is not None

    # Zero JEF on the rebalance date in the filtered daily signals
    signal2D_daily[0, rebalance_idx] = 0.0

    # Old behavior used the (possibly unfiltered) signal2D_pre to determine month picks
    old_result = old_monthly_hold(signal2D_pre.copy(), datearray, monthsToHold=1)

    # New behavior uses filtered daily signals at rebalance then forward-fills
    new_result = new_monthly_hold(signal2D_daily.copy(), datearray, monthsToHold=1)

    # Assert new_result has JEF == 0 for the whole month following rebalance
    # and old_result has JEF == 1 (different behavior)
    # Identify range for that month (from rebalance_idx until month changes again)
    end_idx = n_days
    for i in range(rebalance_idx + 1, n_days):
        if datearray[i].month != datearray[rebalance_idx].month:
            end_idx = i
            break

    assert np.all(new_result[0, rebalance_idx:end_idx] == 0.0)
    assert np.all(old_result[0, rebalance_idx:end_idx] == 1.0)


def test_sharpe_weighted_rank_runs():
    # Small synthetic price series for 3 stocks over 60 days
    start = datetime.date(2020, 1, 1)
    days = 60
    datearray = make_datearray(start, days)
    n_stocks = 3
    n_days = days
    np.random.seed(0)

    # Simple random walk prices (ensure positive)
    adjClose = np.cumprod(1 + 0.001 * np.random.randn(n_stocks, n_days), axis=1) * 10.0

    symbols = ["JEF", "AAA", "CASH"]

    # signal2D: pretend all are in uptrend except JEF is zero on a particular day
    signal2D = np.ones((n_stocks, n_days), dtype=float)
    signal2D_daily = signal2D.copy()
    # Zero JEF on day 30 in filtered daily signals
    signal2D_daily[0, 30] = 0.0

    # Use an existing JSON config in repo (non-critical for this test)
    json_fn = "pytaaa_sp500_pine_montecarlo.json"

    # Call sharpeWeightedRank_2D and ensure it returns valid weights
    weights = sharpeWeightedRank_2D(
        json_fn, datearray, symbols, adjClose, signal2D, signal2D_daily,
        LongPeriod=20, numberStocksTraded=2, riskDownside_min=0.1,
        riskDownside_max=2.0, rankThresholdPct=0.1, stddevThreshold=4.0,
        stockList="SP500"
    )

    assert weights.shape == (n_stocks, n_days)
    # Each column should sum to ~1.0 (allowing for tiny numerical diffs)
    col_sums = weights.sum(axis=0)
    assert np.allclose(col_sums[col_sums > 0], 1.0, atol=1e-6)
