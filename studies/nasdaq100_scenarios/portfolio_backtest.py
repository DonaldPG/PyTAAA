"""Portfolio backtesting with monthly rebalancing.

Implements simplified backtesting logic similar to PyTAAA's dailyBacktest,
but specialized for the oracle signal study.
"""

import logging
import numpy as np
from datetime import date
from typing import Dict, List, Tuple, Optional

DateLike = date | np.datetime64

logger = logging.getLogger(__name__)


def simulate_monthly_portfolio(
    adjClose: np.ndarray,
    signal2D: np.ndarray,
    top_n: int,
    datearray: List[DateLike],
    symbols: List[str],
    initial_value: float = 10000.0,
    transaction_cost: float = 0.0,
    apply_costs: bool = False,
    ranking_method: Optional[str] = None,
    window_half_width: int = 10,
    delay_days: int = 0,
    interpolated_series: Optional[np.ndarray] = None
) -> Dict:
    """Simulate monthly-rebalanced portfolio driven by oracle signals.
    
    Strategy:
    - On first day of each month (rebalance date):
      - Select top N stocks where signal2D == 1.0 (oracle says "buy")
      - Rank stocks according to ranking_method
            - If fewer than N stocks have signal, keep remaining weight in cash
            - Equal-weight selected stocks: w = 1/num_selected
    - Between rebalances: hold positions, value changes with daily returns
    - Optional: deduct transaction costs on rebalance
    
    Args:
        adjClose: Price array (stocks × dates)
        signal2D: Binary oracle signal (stocks × dates), 1.0 = buy, 0.0 = cash
        top_n: Number of stocks to hold (portfolio concentration)
        datearray: List of trading dates
        symbols: List of stock symbols
        initial_value: Starting portfolio value ($)
        transaction_cost: Cost per position change ($)
        apply_costs: Whether to deduct transaction costs
        ranking_method: Stock selection method:
            - None or 'equal': Select first N stocks with signal (no ranking)
            - 'oracle': Rank by forward monthly return (oracle knowledge)
            - 'slope': Rank by extrema-based slope at rebalance date
        window_half_width: Window size for extrema detection (slope ranking only)
        delay_days: Days to look back for slope computation (slope ranking only)
        
    Returns:
        Dictionary containing:
        - portfolio_value: 1D array of daily portfolio values
        - rebalance_dates: List of dates when rebalancing occurred
        - holdings_log: List of (date, holdings_dict) tuples
        - final_value: Portfolio value on last day
        - total_return: (final_value / initial_value) - 1
        - num_rebalances: Count of rebalancing events
    """
    num_stocks, num_dates = adjClose.shape
    
    # Initialize portfolio state
    portfolio_value = np.zeros(num_dates)
    portfolio_value[0] = initial_value
    
    # Current holdings: weights array (sums to 1.0)
    current_weights = np.zeros(num_stocks)
    
    # Tracking data
    rebalance_dates = []
    holdings_log = []
    num_rebalances = 0
    
    # Compute daily returns (gainloss)
    gainloss = np.zeros_like(adjClose)
    gainloss[:, 0] = 1.0
    for j in range(1, num_dates):
        # gainloss[i,j] = adjClose[i,j] / adjClose[i,j-1]
        # Handle NaN and zero prices
        with np.errstate(divide='ignore', invalid='ignore'):
            gainloss[:, j] = adjClose[:, j] / adjClose[:, j-1]
        gainloss[np.isnan(gainloss[:, j]), j] = 1.0  # No change if NaN
        gainloss[np.isinf(gainloss[:, j]), j] = 1.0  # No change if inf
    
    logger.info(f"Simulating monthly portfolio: top_n={top_n}, costs={apply_costs}, ranking={ranking_method or 'none'}")
    logger.info(f"Date range: {datearray[0]} to {datearray[-1]} ({num_dates} days)")
    
    # Import pandas for datetime handling
    import pandas as pd
    
    if ranking_method == "slope" and interpolated_series is None:
        interpolated_series = compute_extrema_interpolated_series(
            adjClose,
            datearray,
            window_half_width,
        )

    # Main simulation loop
    for j in range(num_dates):
        # Check if this is a rebalance date (first day of new month)
        is_rebalance = False
        if j == 0:
            # Always rebalance on first day
            is_rebalance = True
        else:
            # Convert to pandas Timestamp for consistent .month access
            curr_date = pd.Timestamp(datearray[j])
            prev_date = pd.Timestamp(datearray[j-1])
            if curr_date.month != prev_date.month:
                # First day of new month
                is_rebalance = True
        
        rebalance_cost_ratio = 1.0

        if is_rebalance:
            # Select top N stocks where signal == 1.0
            signal_today = signal2D[:, j]
            
            # Determine which stocks to select based on ranking method
            if ranking_method == 'oracle':
                # Rank by forward monthly return (oracle knowledge)
                forward_returns = compute_forward_monthly_return(adjClose, datearray, j)
                selected_indices = rank_by_forward_return(forward_returns, signal_today, top_n)
            elif ranking_method == 'slope':
                # Rank by extrema-based slope
                selected_indices = rank_by_extrema_slope(
                    adjClose, datearray, signal_today, j, top_n,
                    window_half_width, delay_days,
                    interpolated_series=interpolated_series
                )
            else:
                # No ranking - select first N candidates with positive signal
                active_stocks = (signal_today > 0.5)  # Binary threshold
                candidate_indices = np.where(active_stocks)[0]
                
                # Select up to top_n stocks from candidates
                num_selected = min(len(candidate_indices), top_n)
                selected_indices = candidate_indices[:num_selected] if num_selected > 0 else np.array([], dtype=int)
            
            if len(selected_indices) > 0:
                # Equal-weight selected stocks
                new_weights = np.zeros(num_stocks)
                new_weights[selected_indices] = 1.0 / len(selected_indices)
            else:
                # No stocks with positive signal - hold all cash
                new_weights = np.zeros(num_stocks)
            
            # Calculate transaction costs if weights changed
            if j > 0:
                weight_changes = np.abs(new_weights - current_weights)
                num_changed = np.sum(weight_changes > 1e-6)
                
                if num_changed > 0 and apply_costs:
                    total_cost = num_changed * transaction_cost
                    rebalance_cost_ratio = max(
                        0.0,
                        1.0 - total_cost / portfolio_value[j-1],
                    )
            
            # Update weights
            current_weights = new_weights
            rebalance_dates.append(datearray[j])
            num_rebalances += 1
            
            # Log holdings
            holdings = {symbols[i]: current_weights[i] 
                       for i in range(num_stocks) if current_weights[i] > 1e-6}
            holdings_log.append((datearray[j], holdings))
        
        # Compute today's portfolio value
        if j == 0:
            portfolio_value[j] = initial_value
        else:
            # value[j] = value[j-1] * sum(weight[i] * gainloss[i,j])
            daily_return = np.dot(current_weights, gainloss[:, j])
            
            # If all weights are zero (100% cash), maintain value
            if np.sum(current_weights) < 1e-6:
                daily_return = 1.0
            
            portfolio_value[j] = (
                portfolio_value[j-1] * rebalance_cost_ratio * daily_return
            )
    
    # Calculate summary statistics
    final_value = portfolio_value[-1]
    total_return = (final_value / initial_value) - 1.0
    
    logger.info(f"Backtest complete: {num_rebalances} rebalances")
    logger.info(f"Final value: ${final_value:,.2f} (return: {total_return:.2%})")
    
    return {
        'portfolio_value': portfolio_value,
        'rebalance_dates': rebalance_dates,
        'holdings_log': holdings_log,
        'final_value': final_value,
        'total_return': total_return,
        'num_rebalances': num_rebalances
    }


def simulate_buy_and_hold(
    adjClose: np.ndarray,
    datearray: List[DateLike],
    symbols: List[str],
    initial_value: float = 10000.0,
    exclude_cash: bool = True,
    tradable_mask: Optional[np.ndarray] = None
) -> Dict:
    """Simulate buy-and-hold baseline strategy.
    
    Equal-weight all stocks at start, hold until end without rebalancing.
    Provides baseline for comparison with active oracle strategies.
    
    Args:
        adjClose: Price array (stocks × dates)
        datearray: List of trading dates
        symbols: List of stock symbols
        initial_value: Starting portfolio value ($)
        exclude_cash: If True, exclude 'CASH' symbol from holdings
        tradable_mask: Optional boolean mask (stocks × dates) for inclusion
        
    Returns:
        Dictionary containing:
        - portfolio_value: 1D array of daily portfolio values
        - final_value: Portfolio value on last day
        - total_return: (final_value / initial_value) - 1
        - holdings: Dict of symbol -> weight
    """
    num_stocks, num_dates = adjClose.shape
    
    # Determine which stocks to include
    eligible_mask = np.ones(num_stocks, dtype=bool)
    if tradable_mask is not None:
        eligible_mask = tradable_mask.all(axis=1)

    if exclude_cash:
        for i, sym in enumerate(symbols):
            if sym.upper() == 'CASH':
                eligible_mask[i] = False
                break

    stock_indices = np.where(eligible_mask)[0].tolist()
    
    num_holdings = len(stock_indices)
    
    # Equal-weight holdings
    weights = np.zeros(num_stocks)
    if num_holdings > 0:
        for idx in stock_indices:
            weights[idx] = 1.0 / num_holdings
    
    # Compute daily returns
    gainloss = np.zeros_like(adjClose)
    gainloss[:, 0] = 1.0
    for j in range(1, num_dates):
        with np.errstate(divide='ignore', invalid='ignore'):
            gainloss[:, j] = adjClose[:, j] / adjClose[:, j-1]
        gainloss[np.isnan(gainloss[:, j]), j] = 1.0
        gainloss[np.isinf(gainloss[:, j]), j] = 1.0
    
    # Simulate portfolio (no rebalancing - weights stay constant)
    portfolio_value = np.zeros(num_dates)
    portfolio_value[0] = initial_value
    
    for j in range(1, num_dates):
        daily_return = np.dot(weights, gainloss[:, j])
        portfolio_value[j] = portfolio_value[j-1] * daily_return
    
    final_value = portfolio_value[-1]
    total_return = (final_value / initial_value) - 1.0
    
    holdings = {symbols[i]: weights[i] 
                for i in range(num_stocks) if weights[i] > 1e-6}
    
    logger.info(f"Buy-and-hold baseline: {num_holdings} stocks held")
    logger.info(f"Final value: ${final_value:,.2f} (return: {total_return:.2%})")
    
    return {
        'portfolio_value': portfolio_value,
        'final_value': final_value,
        'total_return': total_return,
        'holdings': holdings
    }


def run_scenario_sweep(
    adjClose: np.ndarray,
    symbols: List[str],
    datearray: List[DateLike],
    scenario_signals: Dict[Tuple[int, int], np.ndarray],
    top_n_list: List[int],
    params: Dict,
    ranking_method: Optional[str] = None,
    window_half_width: int = 10,
    delay_days: int = 0
) -> Dict[Tuple[int, int, int], Dict]:
    """Run backtest for all scenario combinations.
    
    Iterates over all (window, delay, top_n) combinations and runs
    monthly portfolio simulation for each.
    
    Args:
        adjClose: Price array (stocks × dates)
        symbols: List of stock symbols
        datearray: List of trading dates
        scenario_signals: Dict from generate_scenario_signals()
                         keys: (window, delay) -> signal2D
        top_n_list: List of portfolio concentration values to test
        params: Parameter dict with 'initial_value', 'transaction_cost', etc.
        ranking_method: Stock selection method for portfolio simulation
        window_half_width: Window size for extrema detection (slope ranking)
        delay_days: Days to look back for slope computation (slope ranking)
        
    Returns:
        Dictionary mapping (window, delay, top_n) -> backtest result dict
        
    Example:
        results = run_scenario_sweep(adjClose, symbols, datearray,
                                     scenarios, [5, 6, 7, 8], params)
        # Returns 80 results for 4 windows × 5 delays × 4 top_n values
    """
    results = {}
    
    initial_value = params.get('initial_value', 10000.0)
    transaction_cost = params.get('transaction_cost', 0.0)
    apply_costs = params.get('apply_transaction_costs', False)
    
    total_scenarios = len(scenario_signals) * len(top_n_list)
    logger.info(f"Running {total_scenarios} backtest scenarios")
    
    scenario_count = 0
    for (window, delay), signal2D in scenario_signals.items():
        for top_n in top_n_list:
            scenario_count += 1
            
            logger.info(f"Scenario {scenario_count}/{total_scenarios}: "
                       f"window={window}, delay={delay}, top_n={top_n}")
            
            result = simulate_monthly_portfolio(
                adjClose=adjClose,
                signal2D=signal2D,
                top_n=top_n,
                datearray=datearray,
                symbols=symbols,
                initial_value=initial_value,
                transaction_cost=transaction_cost,
                apply_costs=apply_costs,
                ranking_method=ranking_method,
                window_half_width=window_half_width,
                delay_days=delay_days
            )
            
            # Store result with key
            results[(window, delay, top_n)] = result
    
    # Also run buy-and-hold baseline
    logger.info("Running buy-and-hold baseline")
    baseline = simulate_buy_and_hold(adjClose, datearray, symbols, initial_value)
    results[('baseline', 0, 0)] = baseline
    
    logger.info(f"Scenario sweep complete: {len(results)} results")
    
    return results


def compute_performance_metrics(results: Dict[Tuple, Dict]) -> Dict[Tuple, Dict]:
    """Calculate additional performance metrics from backtest results.
    
    Computes Sharpe ratio, max drawdown, and other statistics for each scenario.
    
    Args:
        results: Output from run_scenario_sweep()
        
    Returns:
        Dictionary mapping scenario key -> metrics dict
        Adds keys: 'sharpe_ratio', 'max_drawdown', 'volatility', etc.
    """
    metrics = {}
    
    for key, result in results.items():
        portfolio_value = result['portfolio_value']
        
        # Calculate daily returns
        daily_returns = np.diff(portfolio_value) / portfolio_value[:-1]
        
        # Remove NaN/inf
        valid_returns = daily_returns[np.isfinite(daily_returns)]
        
        if len(valid_returns) > 0:
            # Sharpe ratio (annualized, assuming 252 trading days)
            mean_return = np.mean(valid_returns)
            std_return = np.std(valid_returns)
            sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0.0
            
            # Volatility (annualized)
            volatility = std_return * np.sqrt(252)
            
            # Max drawdown
            cummax = np.maximum.accumulate(portfolio_value)
            drawdown = (portfolio_value - cummax) / cummax
            max_drawdown = np.min(drawdown)
        else:
            sharpe = 0.0
            volatility = 0.0
            max_drawdown = 0.0
        
        metrics[key] = {
            'sharpe_ratio': sharpe,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'total_return': result.get('total_return', 0.0),
            'final_value': result.get('final_value', 0.0)
        }
    
    return metrics


def compute_forward_monthly_return(
    adjClose: np.ndarray,
    datearray: List[DateLike],
    rebalance_idx: int
) -> np.ndarray:
    """Compute forward return from rebalance date to month end.
    
    This is "oracle knowledge" - knowing which stocks will perform best
    over the next month. Used for testing whether ranking matters.
    
    Args:
        adjClose: Price array (stocks × dates)
        datearray: List of trading dates (can be date, datetime64, or Timestamp)
        rebalance_idx: Index of rebalance date
        
    Returns:
        Array of forward returns for each stock (length = num_stocks)
        Returns NaN for stocks with invalid prices
        
    Notes:
        - Forward return = (price_at_month_end / price_at_rebalance) - 1
        - Month end is last trading day of the same month
        - This is unknowable at rebalance time (oracle/lookahead)
    """
    num_stocks, num_dates = adjClose.shape
    
    # Convert to pandas Timestamp for consistent .month/.year access
    import pandas as pd
    rebalance_date = pd.Timestamp(datearray[rebalance_idx])
    rebalance_month = rebalance_date.month
    rebalance_year = rebalance_date.year
    
    # Find last trading day of this month
    month_end_idx = rebalance_idx
    for j in range(rebalance_idx + 1, num_dates):
        j_date = pd.Timestamp(datearray[j])
        if j_date.month != rebalance_month or j_date.year != rebalance_year:
            # Found first day of next month
            month_end_idx = j - 1
            break
        month_end_idx = j  # Update to last date in month
    
    # Compute forward returns
    forward_returns = np.full(num_stocks, np.nan)
    
    prices_start = adjClose[:, rebalance_idx]
    prices_end = adjClose[:, month_end_idx]
    
    # Calculate returns where both prices are valid
    valid_mask = (~np.isnan(prices_start)) & (~np.isnan(prices_end)) & (prices_start > 0)
    forward_returns[valid_mask] = (prices_end[valid_mask] / prices_start[valid_mask]) - 1.0
    
    return forward_returns


def rank_by_forward_return(
    forward_returns: np.ndarray,
    signal2D_today: np.ndarray,
    top_n: int
) -> np.ndarray:
    """Rank stocks by forward return, filtered by signal.
    
    Selects top N stocks from those with positive signals, ranked by
    their forward return (oracle knowledge).
    
    Args:
        forward_returns: Array of forward returns for each stock
        signal2D_today: Signal values for today (1.0 = buy, 0.0 = cash)
        top_n: Number of stocks to select
        
    Returns:
        Array of selected stock indices (length <= top_n)
        
    Notes:
        - Only considers stocks where signal2D_today > 0.5
        - Ranks by forward return descending (best performers first)
        - Returns fewer than top_n if insufficient candidates
    """
    # Filter to stocks with positive signals
    active_mask = signal2D_today > 0.5
    
    # Also exclude stocks with NaN forward returns
    valid_mask = active_mask & (~np.isnan(forward_returns))
    
    # Get indices of valid candidates
    candidate_indices = np.where(valid_mask)[0]
    
    if len(candidate_indices) == 0:
        return np.array([], dtype=int)
    
    # Get forward returns for candidates
    candidate_returns = forward_returns[candidate_indices]
    
    # Sort candidates by forward return (descending)
    sorted_order = np.argsort(candidate_returns)[::-1]  # Descending order
    
    # Select top N
    num_selected = min(len(sorted_order), top_n)
    selected_sorted_indices = sorted_order[:num_selected]
    
    # Map back to original stock indices
    selected_indices = candidate_indices[selected_sorted_indices]
    
    return selected_indices


def compute_extrema_interpolated_series(
    adjClose: np.ndarray,
    datearray: List[DateLike],
    window_half_width: int
) -> np.ndarray:
    """Build interpolated time series from detected extrema.
    
    For each stock:
    1. Detect local highs and lows using centered windows
    2. Create simplified series with only extrema values
    3. Linearly interpolate between extrema for all other dates
    
    Args:
        adjClose: Price array (stocks × dates)
        datearray: List of trading dates
        window_half_width: Half-width for extrema detection
        
    Returns:
        Interpolated price array (stocks × dates) same shape as adjClose
        
    Notes:
        - Uses centered window extrema detection
        - Extrema points retain their actual prices
        - Non-extrema points are linearly interpolated
        - Edge regions (first/last k days) use original prices
    """
    from studies.nasdaq100_scenarios.oracle_signals import (
        compute_centered_extrema_masks,
    )
    
    num_stocks, num_dates = adjClose.shape
    low_mask, high_mask = compute_centered_extrema_masks(
        adjClose,
        window_half_width,
    )

    extrema_mask = low_mask | high_mask
    extrema_prices = np.where(extrema_mask, adjClose, np.nan)

    interpolated = np.array(adjClose, copy=True)
    all_indices = np.arange(num_dates)

    for stock_idx in range(num_stocks):
        finite_indices = np.flatnonzero(np.isfinite(extrema_prices[stock_idx]))
        if finite_indices.size == 0:
            continue
        if finite_indices.size == 1:
            continue

        finite_prices = extrema_prices[stock_idx, finite_indices]
        stock_interp = np.interp(all_indices, finite_indices, finite_prices)

        left_edge = int(finite_indices[0])
        right_edge = int(finite_indices[-1])

        stock_interp[:left_edge] = adjClose[stock_idx, :left_edge]
        stock_interp[right_edge + 1:] = adjClose[stock_idx, right_edge + 1:]

        interpolated[stock_idx] = stock_interp
    
    return interpolated


def compute_extrema_slopes(
    interpolated_series: np.ndarray,
    datearray: List[DateLike],
    rebalance_idx: int,
    delay_days: int = 0
) -> np.ndarray:
    """Compute instantaneous slopes at rebalance date from interpolated extrema series.
    
    Args:
        interpolated_series: Interpolated price array from compute_extrema_interpolated_series
        datearray: List of trading dates
        rebalance_idx: Index of rebalance date
        delay_days: Number of days to look back (0 = use current day)
        
    Returns:
        Array of slopes for each stock (length = num_stocks)
        Slope computed as (price[i] - price[i-1]) / 1 day
        Returns NaN for invalid computations
        
    Notes:
        - Slope is instantaneous: computed using adjacent days
        - If delay_days > 0, uses slope from earlier date
        - Returns NaN if lookback date is out of bounds
    """
    num_stocks, num_dates = interpolated_series.shape
    
    # Adjust for delay
    effective_idx = rebalance_idx - delay_days
    
    # Check bounds
    if effective_idx < 1 or effective_idx >= num_dates:
        # Out of bounds - return NaNs
        return np.full(num_stocks, np.nan)
    
    # Compute slope as (price[t] - price[t-1]) / 1
    price_today = interpolated_series[:, effective_idx]
    price_yesterday = interpolated_series[:, effective_idx - 1]
    slopes = price_today - price_yesterday

    invalid = np.isnan(price_today) | np.isnan(price_yesterday)
    slopes = slopes.astype(float, copy=False)
    slopes[invalid] = np.nan
    
    return slopes


def rank_by_extrema_slope(
    adjClose: np.ndarray,
    datearray: List[DateLike],
    signal2D_today: np.ndarray,
    rebalance_idx: int,
    top_n: int,
    window_half_width: int,
    delay_days: int = 0,
    interpolated_series: Optional[np.ndarray] = None
) -> np.ndarray:
    """Rank stocks by extrema-based slope and select top N.
    
    Strategy:
    1. Build interpolated time series from detected extrema
    2. Compute instantaneous slope at rebalance date (with optional delay)
    3. Rank stocks by slope (highest positive slopes first)
    4. Select top N stocks with positive signals
    
    Args:
        adjClose: Price array (stocks × dates)
        datearray: List of trading dates
        signal2D_today: Signal values for today (1.0 = buy, 0.0 = cash)
        rebalance_idx: Index of rebalance date
        top_n: Number of stocks to select
        window_half_width: Half-width for extrema detection
        delay_days: Days to look back for slope computation (0 = no delay)
        
    Returns:
        Array of selected stock indices (length <= top_n)
        
    Notes:
        - Only considers stocks where signal2D_today > 0.5
        - Ranks by slope descending (steepest upward slopes first)
        - Returns fewer than top_n if insufficient candidates
        - Negative slopes still eligible if they're the best available
    """
    # Build interpolated series from extrema
    if interpolated_series is None:
        interpolated_series = compute_extrema_interpolated_series(
            adjClose,
            datearray,
            window_half_width,
        )
    
    # Compute slopes at rebalance date (with delay)
    slopes = compute_extrema_slopes(
        interpolated_series,
        datearray,
        rebalance_idx,
        delay_days,
    )
    
    # Filter to stocks with positive signals
    active_mask = signal2D_today > 0.5
    
    # Also exclude stocks with NaN slopes
    valid_mask = active_mask & (~np.isnan(slopes))
    
    # Get indices of valid candidates
    candidate_indices = np.where(valid_mask)[0]
    
    if len(candidate_indices) == 0:
        return np.array([], dtype=int)
    
    # Get slopes for candidates
    candidate_slopes = slopes[candidate_indices]
    
    # Sort candidates by slope (descending - highest slopes first)
    sorted_order = np.argsort(candidate_slopes)[::-1]
    
    # Select top N
    num_selected = min(len(sorted_order), top_n)
    selected_sorted_indices = sorted_order[:num_selected]
    
    # Map back to original stock indices
    selected_indices = candidate_indices[selected_sorted_indices]
    
    return selected_indices
