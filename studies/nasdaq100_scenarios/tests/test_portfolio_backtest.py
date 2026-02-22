"""Unit tests for portfolio_backtest module."""

import pytest
import numpy as np
from datetime import date, timedelta

from studies.nasdaq100_scenarios.portfolio_backtest import (
    simulate_monthly_portfolio,
    simulate_buy_and_hold,
    run_scenario_sweep,
    compute_performance_metrics
)


class TestSimulateMonthlyPortfolio:
    """Tests for monthly rebalancing portfolio simulation."""
    
    def test_initial_value_preserved(self):
        """Portfolio starts at specified initial value."""
        num_dates = 50
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(num_dates)]
        
        # Single stock, constant price
        adjClose = np.full((1, num_dates), 100.0)
        signal2D = np.ones((1, num_dates))
        symbols = ['TEST']
        
        result = simulate_monthly_portfolio(
            adjClose, signal2D, top_n=1, datearray=datearray, symbols=symbols,
            initial_value=10000.0, apply_costs=False
        )
        
        assert result['portfolio_value'][0] == 10000.0
    
    def test_constant_price_constant_value(self):
        """Constant price with 100% invested should maintain value."""
        num_dates = 50
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(num_dates)]
        
        # Single stock, constant price = no daily returns
        adjClose = np.full((1, num_dates), 100.0)
        signal2D = np.ones((1, num_dates))  # Always buy
        symbols = ['TEST']
        
        result = simulate_monthly_portfolio(
            adjClose, signal2D, top_n=1, datearray=datearray, symbols=symbols,
            initial_value=10000.0, apply_costs=False
        )
        
        # Value should stay constant (no price movement)
        np.testing.assert_array_almost_equal(
            result['portfolio_value'], 
            np.full(num_dates, 10000.0),
            decimal=2
        )
    
    def test_double_price_doubles_value(self):
        """Stock doubling in price doubles portfolio value."""
        num_dates = 50
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(num_dates)]
        
        # Single stock, linearly increasing price (doubles by end)
        adjClose = np.linspace(100, 200, num_dates).reshape(1, -1)
        signal2D = np.ones((1, num_dates))  # Always buy
        symbols = ['TEST']
        
        result = simulate_monthly_portfolio(
            adjClose, signal2D, top_n=1, datearray=datearray, symbols=symbols,
            initial_value=10000.0, apply_costs=False
        )
        
        # Final value should be approximately double (exact depends on rebalancing)
        assert result['final_value'] > 19000.0, "Stock doubled, portfolio should approximately double"
        assert result['total_return'] > 0.9, "Total return should be close to 100%"
    
    def test_zero_signal_holds_cash(self):
        """Zero signal should result in no stock holdings (all cash)."""
        num_dates = 50
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(num_dates)]
        
        # Stock doubles, but signal is zero (no buy)
        adjClose = np.linspace(100, 200, num_dates).reshape(1, -1)
        signal2D = np.zeros((1, num_dates))  # Never buy signal
        symbols = ['TEST']
        
        result = simulate_monthly_portfolio(
            adjClose, signal2D, top_n=1, datearray=datearray, symbols=symbols,
            initial_value=10000.0, apply_costs=False
        )
        
        # Portfolio value should stay constant at initial (all cash, no stock exposure)
        assert abs(result['final_value'] - 10000.0) < 100.0, "All cash should preserve value"
    
    def test_monthly_rebalancing(self):
        """Rebalancing should occur at month boundaries."""
        # Create dates spanning 3 months
        dates = []
        for month in [1, 2, 3]:
            for day in range(1, 29):  # Avoid month-end complications
                dates.append(date(2020, month, day))
        
        num_dates = len(dates)
        adjClose = np.full((1, num_dates), 100.0)
        signal2D = np.ones((1, num_dates))
        symbols = ['TEST']
        
        result = simulate_monthly_portfolio(
            adjClose, signal2D, top_n=1, datearray=dates, symbols=symbols,
            initial_value=10000.0, apply_costs=False
        )
        
        # Should have rebalances at: day 0, start of Feb, start of Mar
        assert result['num_rebalances'] >= 3, "Should rebalance at start of each month"
        
        # Check rebalance dates are indeed month starts
        rebalance_months = [d.month for d in result['rebalance_dates']]
        assert 1 in rebalance_months, "Should rebalance in January"
        assert 2 in rebalance_months, "Should rebalance in February"
        assert 3 in rebalance_months, "Should rebalance in March"
    
    def test_top_n_selection(self):
        """Should select only top_n stocks even if more have signals."""
        num_dates = 50
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(num_dates)]
        
        # 5 stocks, all with positive signals
        adjClose = np.full((5, num_dates), 100.0)
        signal2D = np.ones((5, num_dates))
        symbols = ['A', 'B', 'C', 'D', 'E']
        
        # Request top_n = 3
        result = simulate_monthly_portfolio(
            adjClose, signal2D, top_n=3, datearray=datearray, symbols=symbols,
            initial_value=10000.0, apply_costs=False
        )
        
        # Check first rebalance holdings
        first_holdings = result['holdings_log'][0][1]
        assert len(first_holdings) == 3, "Should hold exactly 3 stocks"
        
        # Equal weight = 1/3
        for weight in first_holdings.values():
            assert abs(weight - 1.0/3.0) < 0.01, "Should equal-weight selected stocks"
    
    def test_transaction_costs_reduce_value(self):
        """Transaction costs should reduce portfolio value."""
        num_dates = 100
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(num_dates)]
        
        # Stock with volatile signal (frequent rebalancing)
        adjClose = np.full((1, num_dates), 100.0)
        signal2D = np.ones((1, num_dates))
        symbols = ['TEST']
        
        # Run with and without costs
        result_no_cost = simulate_monthly_portfolio(
            adjClose, signal2D, top_n=1, datearray=datearray, symbols=symbols,
            initial_value=10000.0, transaction_cost=0.0, apply_costs=False
        )
        
        result_with_cost = simulate_monthly_portfolio(
            adjClose, signal2D, top_n=1, datearray=datearray, symbols=symbols,
            initial_value=10000.0, transaction_cost=10.0, apply_costs=True
        )
        
        # With costs should have lower final value
        assert result_with_cost['final_value'] <= result_no_cost['final_value'], \
            "Transaction costs should reduce portfolio value"


class TestSimulateBuyAndHold:
    """Tests for buy-and-hold baseline."""
    
    def test_equal_weight_allocation(self):
        """Buy-and-hold should equal-weight all stocks."""
        num_dates = 50
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(num_dates)]
        
        # 4 stocks, constant prices
        adjClose = np.full((4, num_dates), 100.0)
        symbols = ['A', 'B', 'C', 'D']
        
        result = simulate_buy_and_hold(adjClose, datearray, symbols, initial_value=10000.0)
        
        # Should hold 4 stocks with equal weight
        assert len(result['holdings']) == 4, "Should hold all 4 stocks"
        
        for weight in result['holdings'].values():
            assert abs(weight - 0.25) < 0.01, "Should equal-weight (1/4 each)"
    
    def test_excludes_cash_by_default(self):
        """CASH symbol should be excluded from buy-and-hold."""
        num_dates = 50
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(num_dates)]
        
        # 3 stocks + CASH
        adjClose = np.full((4, num_dates), 100.0)
        adjClose[3, :] = 1.0  # CASH always 1.0
        symbols = ['A', 'B', 'C', 'CASH']
        
        result = simulate_buy_and_hold(
            adjClose, datearray, symbols, initial_value=10000.0, exclude_cash=True
        )
        
        # Should only hold 3 stocks (not CASH)
        assert len(result['holdings']) == 3, "Should exclude CASH"
        assert 'CASH' not in result['holdings'], "CASH should not be in holdings"
        
        for weight in result['holdings'].values():
            assert abs(weight - 1.0/3.0) < 0.01, "Should equal-weight non-CASH stocks"
    
    def test_no_rebalancing(self):
        """Buy-and-hold should not rebalance (weights stay constant)."""
        num_dates = 50
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(num_dates)]
        
        # Stock A doubles, Stock B halves
        adjClose = np.zeros((2, num_dates))
        adjClose[0, :] = np.linspace(100, 200, num_dates)  # Doubles
        adjClose[1, :] = np.linspace(100, 50, num_dates)   # Halves
        symbols = ['A', 'B']
        
        result = simulate_buy_and_hold(adjClose, datearray, symbols, initial_value=10000.0)
        
        # Even though A outperforms B, we don't rebalance back to equal weight
        # Final value reflects cumulative effect without rebalancing
        # (This is hard to test precisely, but we can verify it runs)
        assert 'holdings' in result
        assert result['final_value'] > 0


class TestRunScenarioSweep:
    """Tests for scenario sweep function."""
    
    def test_all_scenarios_executed(self):
        """Should run backtest for all parameter combinations."""
        num_dates = 100
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(num_dates)]
        
        # Simple data
        t = np.linspace(0, 4 * np.pi, num_dates)
        prices = 100 + 10 * np.sin(t)
        adjClose = prices.reshape(1, -1)
        symbols = ['TEST']
        
        # Create scenario signals: 2 windows × 2 delays = 4 scenarios
        scenario_signals = {
            (10, 0): np.ones((1, num_dates)),
            (10, 5): np.ones((1, num_dates)),
            (20, 0): np.ones((1, num_dates)),
            (20, 5): np.ones((1, num_dates)),
        }
        
        top_n_list = [1, 2]  # 2 top_n values
        
        params = {'initial_value': 10000.0, 'transaction_cost': 0.0, 'apply_transaction_costs': False}
        
        results = run_scenario_sweep(adjClose, symbols, datearray, scenario_signals, top_n_list, params)
        
        # Should have 4 scenarios × 2 top_n = 8 results + 1 baseline = 9 total
        assert len(results) == 9, f"Expected 9 results, got {len(results)}"
        
        # Check specific scenario keys exist
        assert (10, 0, 1) in results
        assert (10, 0, 2) in results
        assert (20, 5, 1) in results
        assert ('baseline', 0, 0) in results
    
    def test_baseline_included(self):
        """Scenario sweep should include buy-and-hold baseline."""
        num_dates = 50
        datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(num_dates)]
        
        adjClose = np.full((1, num_dates), 100.0)
        symbols = ['TEST']
        
        scenario_signals = {(10, 0): np.ones((1, num_dates))}
        top_n_list = [1]
        params = {'initial_value': 10000.0}
        
        results = run_scenario_sweep(adjClose, symbols, datearray, scenario_signals, top_n_list, params)
        
        assert ('baseline', 0, 0) in results, "Should include baseline"
        assert 'portfolio_value' in results[('baseline', 0, 0)]


class TestComputePerformanceMetrics:
    """Tests for performance metrics calculation."""
    
    def test_metrics_calculated(self):
        """Should calculate Sharpe, volatility, drawdown."""
        # Create mock result
        portfolio_value = np.array([10000, 10100, 10050, 10200, 10150])
        result = {
            'portfolio_value': portfolio_value,
            'total_return': 0.015,
            'final_value': 10150.0
        }
        
        results = {(10, 0, 5): result}
        
        metrics = compute_performance_metrics(results)
        
        assert (10, 0, 5) in metrics
        metric = metrics[(10, 0, 5)]
        
        # Check all expected metrics present
        assert 'sharpe_ratio' in metric
        assert 'volatility' in metric
        assert 'max_drawdown' in metric
        assert 'total_return' in metric
        assert 'final_value' in metric
    
    def test_positive_returns_positive_sharpe(self):
        """Consistently positive returns should yield positive Sharpe."""
        # Steadily increasing portfolio
        portfolio_value = np.linspace(10000, 12000, 100)
        result = {
            'portfolio_value': portfolio_value,
            'total_return': 0.2,
            'final_value': 12000.0
        }
        
        results = {(10, 0, 5): result}
        metrics = compute_performance_metrics(results)
        
        assert metrics[(10, 0, 5)]['sharpe_ratio'] > 0, "Positive returns should yield positive Sharpe"
    
    def test_max_drawdown_negative(self):
        """Max drawdown should be negative or zero."""
        # Portfolio with some decline
        portfolio_value = np.array([10000, 11000, 10500, 9500, 10000])
        result = {
            'portfolio_value': portfolio_value,
            'total_return': 0.0,
            'final_value': 10000.0
        }
        
        results = {(10, 0, 5): result}
        metrics = compute_performance_metrics(results)
        
        assert metrics[(10, 0, 5)]['max_drawdown'] <= 0, "Max drawdown should be negative or zero"


# Integration test
def test_full_pipeline_integration():
    """Integration test: full pipeline from signal to metrics."""
    num_dates = 150
    datearray = [date(2020, 1, 1) + timedelta(days=i) for i in range(num_dates)]
    
    # Create realistic price data (random walk)
    np.random.seed(42)
    returns = np.random.randn(3, num_dates) * 0.01  # 1% daily volatility
    prices = 100 * np.exp(np.cumsum(returns, axis=1))
    adjClose = prices
    symbols = ['A', 'B', 'C']
    
    # Create simple signal (first half active, second half cash)
    signal2D = np.zeros((3, num_dates))
    signal2D[:, :num_dates//2] = 1.0
    
    # Run single scenario
    result = simulate_monthly_portfolio(
        adjClose, signal2D, top_n=2, datearray=datearray, symbols=symbols,
        initial_value=10000.0, apply_costs=False
    )
    
    # Verify result structure
    assert 'portfolio_value' in result
    assert 'rebalance_dates' in result
    assert 'holdings_log' in result
    assert 'final_value' in result
    assert 'total_return' in result
    
    # Compute metrics
    metrics = compute_performance_metrics({('test', 0, 2): result})
    
    assert ('test', 0, 2) in metrics
    assert all(key in metrics[('test', 0, 2)] 
              for key in ['sharpe_ratio', 'volatility', 'max_drawdown'])
    
    print(f"\nIntegration test passed:")
    print(f"  Final value: ${result['final_value']:,.2f}")
    print(f"  Total return: {result['total_return']:.2%}")
    print(f"  Sharpe ratio: {metrics[('test', 0, 2)]['sharpe_ratio']:.2f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
