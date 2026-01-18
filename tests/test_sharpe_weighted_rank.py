"""
Tests for sharpeWeightedRank_2D function.

These tests verify that stock selection is based on Sharpe ratio ranking,
not alphabetical order.
"""

import numpy as np
import pytest
import sys
import os

# Add project root to path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions.TAfunctions import sharpeWeightedRank_2D


def create_synthetic_price_data(
    symbols: list,
    n_days: int = 500,
    sharpe_values: dict = None,
    base_price: float = 100.0,
    seed: int = 42
) -> np.ndarray:
    """
    Create synthetic price data with controlled Sharpe ratios.

    Parameters
    ----------
    symbols : list
        List of stock symbols.
    n_days : int
        Number of trading days.
    sharpe_values : dict
        Dictionary mapping symbol to target Sharpe ratio.
        Higher Sharpe = better performance.
    base_price : float
        Starting price for all stocks.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        2D array of prices [n_stocks, n_days].
    """
    np.random.seed(seed)
    n_stocks = len(symbols)
    prices = np.zeros((n_stocks, n_days), dtype=float)

    for i, sym in enumerate(symbols):
        # Get target Sharpe (default 0 = random walk).
        target_sharpe = sharpe_values.get(sym, 0.0) if sharpe_values else 0.0

        # Convert Sharpe to daily drift and volatility.
        # Sharpe = (annualized_return) / (annualized_volatility)
        # Assuming 15% annual volatility as baseline.
        annual_vol = 0.15
        daily_vol = annual_vol / np.sqrt(252)

        # Target annual return = Sharpe * annual_vol.
        annual_return = target_sharpe * annual_vol
        daily_drift = annual_return / 252

        # Generate daily returns.
        daily_returns = np.random.normal(daily_drift, daily_vol, n_days)
        daily_returns[0] = 0  # First day has no return.

        # Convert to prices.
        prices[i, 0] = base_price
        for j in range(1, n_days):
            prices[i, j] = prices[i, j-1] * (1 + daily_returns[j])

    return prices


class TestSharpeWeightedRank2D:
    """Test suite for sharpeWeightedRank_2D function."""

    def test_selects_by_sharpe_not_alphabetically(self):
        """
        Verify that stock selection is based on Sharpe ratio, not
        alphabetical order.

        This is the key test - if stocks are selected alphabetically,
        we'd see AAA, BBB, CCC selected. But with Sharpe-based selection,
        we should see ZZZ (highest Sharpe) selected first.
        """
        # Create symbols in alphabetical order.
        symbols = ["AAA", "BBB", "CCC", "DDD", "ZZZ"]

        # Assign Sharpe ratios - ZZZ has highest, AAA has lowest.
        sharpe_values = {
            "AAA": -1.0,  # Worst performer.
            "BBB": 0.0,   # Random walk.
            "CCC": 0.5,   # Moderate.
            "DDD": 1.0,   # Good.
            "ZZZ": 2.0,   # Best performer.
        }

        n_days = 500
        adjClose = create_synthetic_price_data(
            symbols, n_days, sharpe_values, seed=42
        )

        # Create signal2D - all stocks in uptrend.
        signal2D = np.ones((len(symbols), n_days), dtype=float)
        signal2D_daily = signal2D.copy()

        # Create datearray.
        import datetime
        start_date = datetime.date(2020, 1, 1)
        datearray = np.array([
            start_date + datetime.timedelta(days=i) for i in range(n_days)
        ])

        # Call sharpeWeightedRank_2D with backtest signature.
        monthgainlossweight = sharpeWeightedRank_2D(
            json_fn="dummy.json",
            datearray=datearray,
            symbols=symbols,
            adjClose=adjClose,
            signal2D=signal2D,
            signal2D_daily=signal2D_daily,
            LongPeriod=252,
            numberStocksTraded=2,  # Select top 2 stocks.
            riskDownside_min=0.0,
            riskDownside_max=1.0,
            rankThresholdPct=0.5,
            stddevThreshold=5.0,
            makeQCPlots=False,
            verbose=True
        )

        # Check the last day's weights.
        weights_last_day = monthgainlossweight[:, -1]

        print("\n" + "=" * 60)
        print("TEST: Verifying selection is by Sharpe, not alphabetical")
        print("=" * 60)
        print(f"Symbols: {symbols}")
        print(f"Target Sharpe values: {sharpe_values}")
        print(f"Final weights: {dict(zip(symbols, weights_last_day))}")

        # ZZZ (highest Sharpe) should have weight > 0.
        zzz_idx = symbols.index("ZZZ")
        assert weights_last_day[zzz_idx] > 0, \
            f"ZZZ (highest Sharpe) should be selected, but weight={weights_last_day[zzz_idx]}"

        # AAA (lowest Sharpe) should NOT be selected.
        aaa_idx = symbols.index("AAA")
        assert weights_last_day[aaa_idx] == 0, \
            f"AAA (lowest Sharpe) should NOT be selected, but weight={weights_last_day[aaa_idx]}"

        # BBB (2nd lowest Sharpe) should also NOT be selected.
        bbb_idx = symbols.index("BBB")
        assert weights_last_day[bbb_idx] == 0, \
            f"BBB (2nd lowest Sharpe) should NOT be selected, but weight={weights_last_day[bbb_idx]}"

        # The key test: if alphabetical, we'd have AAA and BBB selected.
        # But we have ZZZ selected, proving it's by Sharpe not alphabetical.
        print("✓ PASSED: Selection is by Sharpe ratio, not alphabetical!")
        print("  - ZZZ (best Sharpe) is selected")
        print("  - AAA and BBB (worst Sharpe) are NOT selected")

    def test_weights_sum_to_one(self):
        """Verify that weights sum to 1.0 for each day."""
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
        n_days = 400

        # All stocks with similar Sharpe.
        sharpe_values = {sym: 1.0 for sym in symbols}
        adjClose = create_synthetic_price_data(
            symbols, n_days, sharpe_values, seed=123
        )

        signal2D = np.ones((len(symbols), n_days), dtype=float)
        signal2D_daily = signal2D.copy()

        import datetime
        start_date = datetime.date(2020, 1, 1)
        datearray = np.array([
            start_date + datetime.timedelta(days=i) for i in range(n_days)
        ])

        monthgainlossweight = sharpeWeightedRank_2D(
            json_fn="dummy.json",
            datearray=datearray,
            symbols=symbols,
            adjClose=adjClose,
            signal2D=signal2D,
            signal2D_daily=signal2D_daily,
            LongPeriod=252,
            numberStocksTraded=3,
            riskDownside_min=0.0,
            riskDownside_max=1.0,
            rankThresholdPct=0.5,
            stddevThreshold=5.0,
            makeQCPlots=False,
            verbose=False
        )

        # Check weights sum to 1.0 for days after warmup period.
        for j in range(260, n_days):
            weight_sum = monthgainlossweight[:, j].sum()
            if weight_sum > 0:  # Only check days with allocations.
                assert abs(weight_sum - 1.0) < 0.01, \
                    f"Day {j}: weights sum to {weight_sum}, expected 1.0"

        print("✓ PASSED: Weights sum to 1.0 for all days!")

    def test_signal_filter_excludes_downtrend_stocks(self):
        """Verify that stocks with signal2D=0 are excluded."""
        symbols = ["GOOD", "BAD"]
        n_days = 400

        # GOOD has high Sharpe, BAD has even higher.
        sharpe_values = {"GOOD": 1.0, "BAD": 2.0}
        adjClose = create_synthetic_price_data(
            symbols, n_days, sharpe_values, seed=456
        )

        # But BAD has signal2D=0 (downtrend).
        signal2D = np.ones((len(symbols), n_days), dtype=float)
        signal2D[1, :] = 0  # BAD is in downtrend.
        signal2D_daily = signal2D.copy()

        import datetime
        start_date = datetime.date(2020, 1, 1)
        datearray = np.array([
            start_date + datetime.timedelta(days=i) for i in range(n_days)
        ])

        monthgainlossweight = sharpeWeightedRank_2D(
            json_fn="dummy.json",
            datearray=datearray,
            symbols=symbols,
            adjClose=adjClose,
            signal2D=signal2D,
            signal2D_daily=signal2D_daily,
            LongPeriod=252,
            numberStocksTraded=2,
            riskDownside_min=0.0,
            riskDownside_max=1.0,
            rankThresholdPct=0.5,
            stddevThreshold=5.0,
            makeQCPlots=False,
            verbose=False
        )

        # BAD should have 0 weight despite higher Sharpe.
        bad_idx = symbols.index("BAD")
        weights_last = monthgainlossweight[:, -1]

        assert weights_last[bad_idx] == 0, \
            f"BAD (downtrend) should have 0 weight, but has {weights_last[bad_idx]}"

        # GOOD should have all the weight.
        good_idx = symbols.index("GOOD")
        assert weights_last[good_idx] > 0, \
            f"GOOD (uptrend) should be selected"

        print("✓ PASSED: Downtrend stocks are correctly excluded!")

    def test_infilled_data_excluded(self):
        """Verify that stocks with repeated/infilled prices are excluded."""
        symbols = ["NORMAL", "INFILLED"]
        n_days = 400

        # Create normal data.
        sharpe_values = {"NORMAL": 0.5, "INFILLED": 2.0}
        adjClose = create_synthetic_price_data(
            symbols, n_days, sharpe_values, seed=789
        )

        # Make INFILLED have constant prices (simulating stale data).
        adjClose[1, :] = 100.0  # All same price.

        signal2D = np.ones((len(symbols), n_days), dtype=float)
        signal2D_daily = signal2D.copy()

        import datetime
        start_date = datetime.date(2020, 1, 1)
        datearray = np.array([
            start_date + datetime.timedelta(days=i) for i in range(n_days)
        ])

        monthgainlossweight = sharpeWeightedRank_2D(
            json_fn="dummy.json",
            datearray=datearray,
            symbols=symbols,
            adjClose=adjClose,
            signal2D=signal2D,
            signal2D_daily=signal2D_daily,
            LongPeriod=252,
            numberStocksTraded=2,
            riskDownside_min=0.0,
            riskDownside_max=1.0,
            rankThresholdPct=0.5,
            stddevThreshold=5.0,
            makeQCPlots=False,
            verbose=False
        )

        # INFILLED should be excluded due to repeated values.
        infilled_idx = symbols.index("INFILLED")
        weights_last = monthgainlossweight[:, -1]

        assert weights_last[infilled_idx] == 0, \
            f"INFILLED (stale data) should have 0 weight, but has {weights_last[infilled_idx]}"

        print("✓ PASSED: Infilled/stale data stocks are excluded!")


if __name__ == "__main__":
    # Run tests.
    test = TestSharpeWeightedRank2D()

    print("\n" + "=" * 70)
    print("Running sharpeWeightedRank_2D tests")
    print("=" * 70)

    test.test_selects_by_sharpe_not_alphabetically()
    test.test_weights_sum_to_one()
    test.test_signal_filter_excludes_downtrend_stocks()
    test.test_infilled_data_excluded()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
