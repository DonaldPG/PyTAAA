"""
test_sharpe_rank_respects_zeros.py

Phase 5 test: verify sharpeWeightedRank_2D() assigns zero weight to
stocks whose signal2D entry is zero, and does NOT spread equal weight to
all stocks as a fallback (the secondary bug fixed in functions/TAfunctions.py).

Two scenarios are covered:

1. all_signals_zero — every stock has signal=0 for all dates (simulates the
   rolling window filter zeroing every row). After fix: CASH gets 100%, all
   other stocks including JEF get 0%.

2. filtered_stock_zero, active_stock_nonzero — JEF signals are zero, AAPL
   signals are 1. After fix: JEF weight stays 0 for all dates, AAPL weight
   may be > 0 for dates where Sharpe is computable.
"""

import datetime
import numpy as np
import pytest
from functions.TAfunctions import sharpeWeightedRank_2D


#############################################################################
# Helpers
#############################################################################

def _make_datearray(start_year: int, start_month: int, n_days: int):
    """Return a list of consecutive calendar dates."""
    start = datetime.date(start_year, start_month, 1)
    return [start + datetime.timedelta(days=i) for i in range(n_days)]


def _noisy_prices(n_days: int, p0: float = 100.0, drift: float = 0.0005,
                  vol: float = 0.012, seed: int = 1) -> np.ndarray:
    """Random-walk prices with positive drift (decent Sharpe ratio)."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(drift, vol, n_days)
    log_returns[0] = 0.0
    return p0 * np.exp(np.cumsum(log_returns))


def _linear_prices(n_days: int, p0: float = 20.0, slope: float = 0.01) -> np.ndarray:
    """Perfectly linear prices (extremely low gainloss std)."""
    return p0 + np.arange(n_days, dtype=float) * slope


#############################################################################
# Test 1 — all signals zero → no stock should get weight; CASH gets 100%
#############################################################################

class TestAllSignalsZero:
    """When the rolling filter has zeroed every signal, no stock should receive
    a positive weight in the non-early-period fallback. CASH absorbs all capital."""

    @pytest.fixture
    def setup(self):
        n_days = 80
        long_period = 20
        symbols = ["JEF", "AAPL", "CASH"]
        # 2010 is well outside the 2000-2002 early period.
        datearray = _make_datearray(2010, 1, n_days)

        adjClose = np.vstack([
            _linear_prices(n_days),      # JEF
            _noisy_prices(n_days),       # AAPL
            np.ones(n_days),             # CASH
        ])

        # All signals zero — simulates rolling filter eliminating everything.
        signal2D = np.zeros((3, n_days), dtype=float)
        signal2D_daily = np.zeros((3, n_days), dtype=float)

        weights = sharpeWeightedRank_2D(
            json_fn="dummy_not_used",
            datearray=datearray,
            symbols=symbols,
            adjClose=adjClose,
            signal2D=signal2D,
            signal2D_daily=signal2D_daily,
            LongPeriod=long_period,
            numberStocksTraded=2,
            riskDownside_min=0.0,
            riskDownside_max=100.0,
            rankThresholdPct=0.0,
            stddevThreshold=100.0,
            makeQCPlots=False,
            is_backtest=True,
            stockList="SP500",
        )
        return weights, symbols, n_days

    def test_jef_weight_is_zero_when_all_signals_zero(self, setup):
        """JEF must have zero weight on every date when all signals are zero."""
        weights, symbols, n_days = setup
        jef_idx = symbols.index("JEF")
        jef_weights = weights[jef_idx, :]
        bad_dates = np.where(jef_weights != 0.0)[0]
        assert len(bad_dates) == 0, (
            f"JEF has non-zero weight on {len(bad_dates)} dates when all signals "
            f"are zero (secondary bug: equal-weight fallback re-enabling JEF). "
            f"First bad index: {bad_dates[0]}, weight: {jef_weights[bad_dates[0]]:.4f}"
        )

    def test_cash_weight_is_one_when_all_signals_zero(self, setup):
        """CASH must absorb 100% of capital on dates where all signals are zero."""
        weights, symbols, n_days = setup
        cash_idx = symbols.index("CASH")
        cash_weights = weights[cash_idx, :]
        # All dates post-warmup (once forward-fill is consistent) should be 1.0.
        # Allow the first few dates (0..long_period-1) to be handled by warmup.
        post_warmup = cash_weights[20:]  # After day 20.
        bad_dates = np.where(np.abs(post_warmup - 1.0) > 1e-6)[0]
        assert len(bad_dates) == 0, (
            f"CASH weight is not 1.0 on {len(bad_dates)} post-warmup dates when "
            f"all signals are zero. Expected 100% CASH. "
            f"First bad idx (offset by 20): {bad_dates[0]}, weight: {post_warmup[bad_dates[0]]:.4f}"
        )


#############################################################################
# Test 2 — filtered stock has zero signal, active stock has non-zero signal
#############################################################################

class TestFilteredStockZeroSignal:
    """When the rolling filter has zeroed JEF's signal but left AAPL's intact,
    JEF must receive zero portfolio weight for all dates."""

    @pytest.fixture
    def setup(self):
        n_days = 80
        long_period = 20
        symbols = ["JEF", "AAPL", "CASH"]
        datearray = _make_datearray(2010, 3, n_days)

        adjClose = np.vstack([
            _linear_prices(n_days),      # JEF — infilled, low vol
            _noisy_prices(n_days),       # AAPL — realistic prices
            np.ones(n_days),             # CASH
        ])

        # JEF: zero signal (filter excluded it).
        # AAPL: signal = 1 from day long_period onward (post-warmup).
        # CASH: signal = 0 (passive).
        signal2D = np.zeros((3, n_days), dtype=float)
        signal2D[1, long_period:] = 1.0  # AAPL uptrend from day 20.

        signal2D_daily = signal2D.copy()

        weights = sharpeWeightedRank_2D(
            json_fn="dummy_not_used",
            datearray=datearray,
            symbols=symbols,
            adjClose=adjClose,
            signal2D=signal2D,
            signal2D_daily=signal2D_daily,
            LongPeriod=long_period,
            numberStocksTraded=1,
            riskDownside_min=0.0,
            riskDownside_max=100.0,
            rankThresholdPct=0.0,
            stddevThreshold=100.0,
            makeQCPlots=False,
            is_backtest=True,
            stockList="SP500",
        )
        return weights, symbols, n_days, long_period

    def test_jef_weight_is_zero_when_signal_is_zero(self, setup):
        """JEF has zero signal everywhere; its weight must be 0 on every date."""
        weights, symbols, n_days, _ = setup
        jef_idx = symbols.index("JEF")
        jef_weights = weights[jef_idx, :]
        bad_dates = np.where(jef_weights != 0.0)[0]
        assert len(bad_dates) == 0, (
            f"JEF has non-zero weight on {len(bad_dates)} dates despite having "
            f"signal=0 (secondary bug: equal-weight fallback re-enabling JEF). "
            f"First bad index: {bad_dates[0]}, weight: {jef_weights[bad_dates[0]]:.4f}"
        )

    def test_aapl_has_nonzero_weight_post_warmup(self, setup):
        """AAPL should receive positive weight on at least some post-warmup dates."""
        weights, symbols, n_days, long_period = setup
        aapl_idx = symbols.index("AAPL")
        post_warmup_weights = weights[aapl_idx, long_period + 5:]
        nonzero_count = np.sum(post_warmup_weights > 0)
        assert nonzero_count > 0, (
            "AAPL should have positive weight on some post-warmup dates when "
            f"it is the only stock with a positive signal. Got zero weight on all "
            f"{len(post_warmup_weights)} post-warmup dates."
        )

    def test_weight_sum_is_valid(self, setup):
        """Total portfolio weight per date must be 0 or 1 (no partial allocation)."""
        weights, symbols, n_days, long_period = setup
        for j in range(long_period, n_days):
            col_sum = float(np.sum(weights[:, j]))
            assert abs(col_sum - 1.0) < 1e-6 or col_sum == 0.0, (
                f"Weight sum at date index {j} is {col_sum:.6f}; expected 0 or 1."
            )
