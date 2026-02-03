"""Monte Carlo backtesting system with read-only data access."""

import os
import time
import math
import signal
import sys
import threading
from datetime import datetime, date
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
import logging
import json
import random
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from functions.PortfolioMetrics import (
    calculate_period_metrics, 
    analyze_model_switching_effectiveness
)

# Global interrupt flag using threading.Event for thread safety
_INTERRUPT_EVENT = threading.Event()

# Set up global signal handler at module level
def _global_signal_handler(signum, frame):
    """Global signal handler for immediate interrupt detection."""
    print("\n\n*** INTERRUPT SIGNAL RECEIVED! ***")
    print("Stopping after current iteration...")
    _INTERRUPT_EVENT.set()

# Install the signal handler immediately when module is imported
signal.signal(signal.SIGINT, _global_signal_handler)

def check_interrupt():
    """Check if interrupt was requested."""
    if _INTERRUPT_EVENT.is_set():
        raise KeyboardInterrupt("User requested interrupt")

def reset_interrupt():
    """Reset the interrupt flag."""
    _INTERRUPT_EVENT.clear()

class PerformanceMetrics(NamedTuple):
    """Performance metrics for portfolio evaluation."""
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    avg_drawdown: float
    annualized_return: float

try:
    import numba
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)

# Define optimized functions outside the class
@jit(nopython=True) if HAS_NUMBA else lambda x: x
def compute_rolling_max(arr: np.ndarray) -> np.ndarray:
    """Compute rolling maximum of an array."""
    result = np.zeros_like(arr)
    curr_max = arr[0]
    result[0] = curr_max
    for i in range(1, len(arr)):
        curr_max = max(curr_max, arr[i])
        result[i] = curr_max
    return result

@jit(nopython=True) if HAS_NUMBA else lambda x: x
def compute_metrics_fast(portfolio_values: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Optimized computation of performance metrics."""
    if len(portfolio_values) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Pre-allocate arrays
    n = len(portfolio_values)
    
    # Calculate annualized return
    start_price = portfolio_values[0]
    end_price = portfolio_values[-1]
    years = len(portfolio_values) / 252
    annualized_return = (end_price / start_price) ** (1 / years) - 1
    
    # Sharpe ratio
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    mean_return = np.mean(daily_returns)
    std_dev_return = np.std(daily_returns)
    std_dev_return *= np.sqrt(252) # annualize
    # risk_free_rate = 0.02 / 252  # Convert annualized rate to daily
    risk_free_rate = 0.0
    sharpe = (annualized_return - risk_free_rate) / std_dev_return if std_dev_return > 0 else 0.0
    # sharpe = (avg_daily_return * np.sqrt(252) / volatility) if volatility > 0 else 0.0
    
    # Sortino ratio
    downside_returns = daily_returns[daily_returns < risk_free_rate]
    downside_deviation = np.sqrt(np.mean(downside_returns**2)) if len(downside_returns) > 0 else 0
    downside_deviation *= np.sqrt(252) # annualize
    sortino = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 1.0
    
    # Drawdown calculations
    cumulative_returns = np.zeros(len(daily_returns) + 1)
    cumulative_returns[0] = 1.0
    for i in range(len(daily_returns)):
        cumulative_returns[i + 1] = cumulative_returns[i] * (1 + daily_returns[i])
    
    rolling_max = compute_rolling_max(cumulative_returns)
    drawdowns = np.zeros_like(cumulative_returns)
    
    for i in range(len(cumulative_returns)):
        if rolling_max[i] > 0:
            drawdowns[i] = ((cumulative_returns[i] - rolling_max[i]) / rolling_max[i])
    
    max_dd = np.min(drawdowns)
    avg_dd = np.mean(drawdowns)
    
    # Return only 5 values: sharpe, sortino, max_dd, avg_dd, annualized_return
    return sharpe, sortino, max_dd, avg_dd, annualized_return

def compute_daily_metrics(portfolio_values: np.ndarray) -> PerformanceMetrics:
    """Compute comprehensive performance metrics for a portfolio using optimized calculations."""
    metrics = compute_metrics_fast(portfolio_values)
    return PerformanceMetrics(*metrics)

@numba.jit(nopython=True)
def rank_models_fast(metric_values: np.ndarray) -> np.ndarray:
    """Optimized ranking calculation using Numba."""
    n_metrics, n_models = metric_values.shape
    ranks = np.zeros((n_metrics, n_models))
    
    for i in range(n_metrics):
        # Create sorting indices
        temp = np.zeros(n_models)
        for j in range(n_models):
            temp[j] = metric_values[i, j]
        
        # Simple ranking algorithm
        for j in range(n_models):
            rank = 0
            for k in range(n_models):
                if temp[k] > temp[j]:
                    rank += 1
            ranks[i, j] = rank
    
    return ranks

def rank_models(metrics_list: List[PerformanceMetrics]) -> np.ndarray:
    """Rank models based on their performance metrics using optimized calculations."""
    n_models = len(metrics_list)
    metric_arrays = np.zeros((5, n_models))  # Changed from 7 to 5 metrics
    
    for i, metrics in enumerate(metrics_list):
        metric_arrays[:, i] = [
            metrics.sharpe_ratio,
            metrics.sortino_ratio,
            metrics.max_drawdown,  # Negative since higher is better (less negative)
            metrics.avg_drawdown,  # Negative since higher is better (less negative)
            metrics.annualized_return
        ]
    
    return rank_models_fast(metric_arrays)

@numba.jit(nopython=True)
def _compute_bin_indices(lookbacks_arr, bin_edges):
    """Convert lookback values to bin indices using Numba."""
    indices = np.zeros(len(lookbacks_arr), dtype=np.int64)
    for i in range(len(lookbacks_arr)):
        # Manual binary search since searchsorted isn't supported in nopython mode
        idx = 0
        for j in range(len(bin_edges) - 1):
            if lookbacks_arr[i] >= bin_edges[j]:
                idx = j
        indices[i] = idx
    return indices

@numba.jit(nopython=True)
def _get_random_indices_from_probs(probs_flat):
    """Get random indices based on provided probabilities using Numba."""
    cumsum = np.zeros_like(probs_flat)
    total = 0.0
    for i in range(len(probs_flat)):
        total += probs_flat[i]
        cumsum[i] = total
    
    rand_val = np.random.random()
    idx = 0
    for i in range(len(cumsum)):
        if rand_val <= cumsum[i]:
            idx = i
            break
    return idx

@numba.jit(nopython=True)
def _add_noise_to_lookbacks(lookbacks, bin_width, noise_scale=1.0,
                           min_val=0.0, max_val=1000.0):
    """Add random noise to lookback values within bin width using Numba."""
    noisy = np.zeros_like(lookbacks)
    for i in range(len(lookbacks)):
        noise = np.random.uniform(-bin_width/2 * noise_scale, bin_width/2 * noise_scale)
        val = lookbacks[i] + noise
        noisy[i] = min(max(val, min_val), max_val)
    return noisy

class MonteCarloBacktest:
    # Define normalization parameters as class constants
    # most recent set of values
    CENTRAL_VALUES = {
        'annual_return': 0.46,
        'sharpe_ratio': 1.50,
        'sortino_ratio': 1.475,
        'max_drawdown': -0.53,
        'avg_drawdown': -0.105
    }
    STD_VALUES = {
        'annual_return': 0.040,
        'sharpe_ratio': 0.16,
        'sortino_ratio': 0.030,
        'max_drawdown': 0.053,
        'avg_drawdown': 0.011
    }

    # # revert to 1a set of values
    # CENTRAL_VALUES = {
    #     'annual_return': .4537,
    #     'sharpe_ratio': 1.44,
    #     'sortino_ratio': 1.42,
    #     'max_drawdown': -0.58,
    #     'avg_drawdown': -0.115
    # }
    # STD_VALUES = {
    #     'annual_return': .074,
    #     'sharpe_ratio': 0.05,   # Fixed: Changed from 0.005 to 0.05 (matches compute_performance_metrics)
    #     'sortino_ratio': 0.21,
    #     'max_drawdown': .069,
    #     'avg_drawdown': .019
    # }
    # # revert to 2nd set of values
    # CENTRAL_VALUES = {
    #     'annual_return': .4537,
    #     'sharpe_ratio': 1.44,
    #     'sortino_ratio': 1.42,
    #     'max_drawdown': -0.58,
    #     'avg_drawdown': -0.115
    # }
    # STD_VALUES = {
    #     'annual_return': .074,
    #     'sharpe_ratio': 0.17,   # Fixed: Changed from 0.005 to 0.05 (matches compute_performance_metrics)
    #     'sortino_ratio': 0.21,
    #     'max_drawdown': .069,
    #     'avg_drawdown': .019
    # }

    # # revert to 3rd set of values
    # CENTRAL_VALUES = {
    #     'annual_return': .4145,
    #     'sharpe_ratio': 1.365,
    #     'sortino_ratio': 1.31,
    #     'max_drawdown': -0.556,
    #     'avg_drawdown': -0.120
    # }    
    # STD_VALUES = {
    #     'annual_return': .044,
    #     'sharpe_ratio': 0.135,   # Fixed: Changed from 0.005 to 0.05 (matches compute_performance_metrics)
    #     'sortino_ratio': 0.146,
    #     'max_drawdown': .052,
    #     'avg_drawdown': .016
    # }

    # focus mostly on sortino ratio
    CENTRAL_VALUES = {
        'annual_return': 0.46,
        'sharpe_ratio': 1.50,
        'sortino_ratio': 1.475,
        'max_drawdown': -0.53,
        'avg_drawdown': -0.105
    }
    STD_VALUES = {
        'annual_return': 0.040,
        'sharpe_ratio': 0.16,
        'sortino_ratio': 0.010,
        'max_drawdown': 0.053,
        'avg_drawdown': 0.011
    }

    # randomly generated values (1st pass)
    # CENTRAL_VALUES = {
    #     'annual_return': np.random.choice([0.415, 0.425, 0.435, 0.445, 0.455, 0.465]),
    #     'sharpe_ratio': np.random.choice([1.35, 1.45, 1.55]),
    #     'sortino_ratio': np.random.choice([1.30, 1.35, 1.40, 1.45]),
    #     'max_drawdown': np.random.choice([-0.58, -0.56, -0.54, -0.52]),
    #     'avg_drawdown': np.random.choice([-0.125, -0.120, -0.115, -0.110, -0.105])
    # }
    # STD_VALUES = {
    #     'annual_return': np.random.choice([0.020, 0.030, 0.040, 0.045, 0.050, 0.060]),
    #     'sharpe_ratio': np.random.choice([0.135, 0.150, 0.165, 0.180, 0.200]),
    #     'sortino_ratio': np.random.choice([0.120, 0.140, 0.160, 0.180]),
    #     'max_drawdown': np.random.choice([0.05, 0.06, 0.07]),
    #     'avg_drawdown': np.random.choice([0.010, 0.013, 0.016, 0.019])
    # }
    # randomly generated values (2nd pass)
    CENTRAL_VALUES = {
        'annual_return': np.random.choice([0.425, 0.435, 0.445, 0.455]),
        'sharpe_ratio': np.random.choice([1.35, 1.45]),
        'sortino_ratio': np.random.choice([1.30, 1.35, 1.40, 1.45]),
        'max_drawdown': np.random.choice([-0.58, -0.56, -0.54]),
        'avg_drawdown': np.random.choice([-0.125, -0.120, -0.115, -0.110, -0.105])
    }
    STD_VALUES = {
        'annual_return': np.random.choice([0.020, 0.030, 0.040, 0.045, 0.050, 0.060]),
        'sharpe_ratio': np.random.choice([0.135, 0.150, 0.165, 0.180, 0.200]),
        'sortino_ratio': np.random.choice([0.120, 0.140, 0.160]),
        'max_drawdown': np.random.choice([0.05, 0.06, 0.07]),
        'avg_drawdown': np.random.choice([0.010, 0.013, 0.016, 0.019])
    }
    def __init__(
        self,
        model_paths: Dict[str, str],
        iterations: int = 50000,
        min_iterations_for_exploit: int = 50,
        trading_frequency: str = "monthly",
        max_iterations: int = 50000,
        min_lookback: int = 20,
        max_lookback: int = 300,
        n_lookbacks: int = 3,
        search_mode: str = "explore-exploit",
        verbose: bool = False,
        json_config: Optional[Dict] = None
    ) -> None:
        """Initialize Monte Carlo backtesting framework with permutation-invariant tracking.
        
        Args:
            model_paths: Dictionary mapping model names to data file paths
            iterations: Number of Monte Carlo iterations to run
            min_iterations_for_exploit: Minimum iterations before exploitation starts
            trading_frequency: Frequency of trading decisions ("monthly" or "daily")
            max_iterations: Maximum iterations (legacy parameter, same as iterations)
            min_lookback: Minimum lookback period in days
            max_lookback: Maximum lookback period in days
            n_lookbacks: Number of lookback periods to use
            search_mode: Search strategy - "explore-exploit" (default), "explore", or "exploit"
            verbose: Whether to show detailed normalized score breakdown
            json_config: Optional JSON configuration dictionary
        """
        self.iterations = iterations
        self.min_iterations_for_exploit = min_iterations_for_exploit
        self.trading_frequency = trading_frequency
        self.model_paths = model_paths
        self.min_lookback = min_lookback
        self.max_lookback = max_lookback
        self.n_lookbacks = n_lookbacks
        self.search_mode = search_mode
        self.verbose = verbose  # Store verbose setting
        
        # Store JSON configuration for focus period blending
        self.json_config = json_config or {}
        self.metric_blending_config = self.json_config.get('metric_blending', {})
        self.focus_period_enabled = self.metric_blending_config.get('enabled', False)
        
        # Check if we have two focus periods or just one
        self.has_two_focus_periods = ('focus_period_1_start' in self.metric_blending_config and 
                                       'focus_period_2_start' in self.metric_blending_config)
        
        if self.focus_period_enabled:
            if self.has_two_focus_periods:
                print(f"Focus period blending enabled with TWO focus periods:")
                print(f"  Focus Period 1: {self.metric_blending_config.get('focus_period_1_start')} to {self.metric_blending_config.get('focus_period_1_end')}")
                print(f"  Focus Period 2: {self.metric_blending_config.get('focus_period_2_start')} to {self.metric_blending_config.get('focus_period_2_end')}")
                print(f"Blending weights: Full period {self.metric_blending_config.get('full_period_weight', 1.0):.1f}, Focus periods {self.metric_blending_config.get('focus_period_weight', 0.0):.1f}")
            else:
                print(f"Focus period blending enabled with ONE focus period:")
                print(f"  Focus Period: {self.metric_blending_config.get('focus_period_start')} to {self.metric_blending_config.get('focus_period_end')}")
                print(f"Blending weights: Full period {self.metric_blending_config.get('full_period_weight', 1.0):.1f}, Focus period {self.metric_blending_config.get('focus_period_weight', 0.0):.1f}")
        
        # Maps canonical tuple -> index in tracking arrays
        self.combination_indices: Dict[Tuple[int, ...], int] = {}
        
        # Performance tracking arrays (using lists for dynamic sizing)
        self.canonical_performance_scores: List[float] = []
        self.canonical_visit_counts: List[int] = []
        
        # Set up logging
        self.logger = logging.getLogger("MonteCarloBacktest")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler for exploitation rate logging
        fh = logging.FileHandler("monte_carlo_exploitation.log")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # Initialize storage
        self.dates: List[date] = []
        self.portfolio_histories: Dict[str, pd.Series] = {}
        
        # Load historical data immediately
        try:
            self._load_historical_data()
        except Exception as e:
            logger.error(f"Failed to load historical data: {str(e)}")
            raise ValueError(f"Failed to initialize Monte Carlo backtesting: {str(e)}")

        self.csv_filename = "abacus_best_performers.csv"

        # Initialize additional tracking for equivalent lookback combinations
        self.equivalent_combinations_found = 0
        self.logger.info(f"Initialized MonteCarloBacktest with permutation-invariant tracking (search_mode: {search_mode})")
        
    def _parse_line(self, line: str, is_backtested: bool) -> Tuple[Optional[date], Optional[float]]:
        """Parse a line from portfolio value file.

        Args:
            line: The line to parse
            is_backtested: If True, format is "YYYY-MM-DD VALUE" (backtested data),
                         if False, format is "cumu_value: YYYY-MM-DD HH:MM:SS.SSSSSS VALUE1 VALUE2 VALUE3" (actual data)
        
        Returns:
            Tuple of (date, value) or (None, None) if line can't be parsed
        """
        try:
            parts = line.strip().split()
            if is_backtested:
                if len(parts) >= 2:
                    date_val = datetime.strptime(parts[0], "%Y-%m-%d").date()
                    value = float(parts[2])
                    return date_val, value
            else:  # actual data format
                if len(parts) >= 6 and parts[0] == "cumu_value:":
                    date_val = datetime.strptime(parts[1], "%Y-%m-%d").date()
                    value = float(parts[3])  # Use the first value after date
                    return date_val, value
            return None, None
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing line: {str(e)}\nLine content: {line.strip()}")
            return None, None

    def _load_historical_data(self) -> None:
        """Load historical portfolio values from data files in read-only mode."""
        portfolio_histories = {}
        all_dates = set()
        
        print("Loading historical portfolio data...")
        
        # Read data from each model file
        for model, path in self.model_paths.items():
            if model != "cash" and os.path.exists(path):
                data = []
                # Determine format based on filename
                is_backtested = "pyTAAAweb_backtestPortfolioValue.params" in path
                print(f"\nProcessing {model} data from {path}")
                print(f"Using {'backtested' if is_backtested else 'actual'} portfolio values")
                
                try:
                    with open(path, 'r') as f:  # Open in read-only mode
                        for line_num, line in enumerate(f, 1):
                            date_val, value = self._parse_line(line, is_backtested)
                            if date_val and value:
                                data.append((date_val, value))
                                all_dates.add(date_val)
                                
                                if line_num % 1000 == 0:
                                    print(f"Processed {line_num} lines...", end="\r")
                                    
                    if data:
                        print(f"\nLoaded {len(data)} data points for {model}")
                        # Convert to pandas Series with date index
                        df = pd.DataFrame(data, columns=['date', 'value'])
                        # Handle duplicate dates by keeping the last value for each date
                        df = df.groupby('date')['value'].last()
                        
                        # Find first valid value for backfilling
                        first_valid_value = df.iloc[0]
                        
                        # Create series with unified date range starting from min date
                        min_date = min(all_dates)
                        if df.index[0] > min_date:
                            # Add the first valid value at the start date
                            df.loc[min_date] = first_valid_value
                            
                        # Sort index to ensure proper forward fill
                        df = df.sort_index()
                        portfolio_histories[model] = df
                        
                        logger.info(
                            f"Successfully loaded {len(portfolio_histories[model])} values "
                            f"for {model}"
                        )
                    else:
                        logger.warning(f"No valid data found in {path}")
                        
                except IOError as e:
                    logger.error(f"Error reading file {path}: {str(e)}")
                    continue
        
        if not portfolio_histories:
            raise ValueError("No valid data found in any input files")
            
        # Convert dates to sorted list
        self.dates = sorted(all_dates)
        print(f"\nTotal unique dates across all models: {len(self.dates)}")
        print(f"Date range: {min(self.dates)} to {max(self.dates)}")
        
        # Create unified date index
        date_index = pd.DatetimeIndex(self.dates)
        
        # Ensure all models have values for all dates through forward-filling
        print("\nAligning data across models...")
        for model in portfolio_histories:
            # Reindex and forward fill, ensuring no duplicate dates
            portfolio_histories[model] = portfolio_histories[model].reindex(
                date_index
            ).ffill()  # Use ffill() instead of fillna(method='ffill')
            
            # Backward fill any remaining NaN values at the start
            portfolio_histories[model] = portfolio_histories[model].bfill()  # Use bfill() instead of fillna(method='bfill')
            
            print(f"Aligned {model} data starting at ${portfolio_histories[model].iloc[0]:.2f}")
        
        # Add cash model with constant value
        portfolio_histories["cash"] = pd.Series(10000.0, index=date_index)
        
        self.portfolio_histories = portfolio_histories
        print("\nData loading complete!")

    def run(self) -> List[str]:
        """Run Monte Carlo backtesting with actual portfolio values."""
        if not self.dates:
            raise ValueError("No historical data loaded")
            
        # Reset interrupt flag
        reset_interrupt()
            
        n_dates = len(self.dates)
        top_models = []
        best_performance = float('-inf')
        start_time = time.time()
        
        print(f"\nRunning {self.iterations} Monte Carlo iterations...")
        print(f"Using {len(self.portfolio_histories)} models over {n_dates} trading days")
        print("Press Ctrl+C to stop early with current best result\n")
        
        try:
            # Run Monte Carlo iterations
            for i in range(self.iterations):
                # Check for interrupt at the start of each iteration
                check_interrupt()
                    
                if i % 10 == 0:  # Progress update every 10 iterations
                    elapsed = time.time() - start_time
                    if i > 0:
                        est_total = elapsed * self.iterations / i
                        remaining = est_total - elapsed
                        remaining_min = int(remaining // 60)
                        remaining_sec = int(remaining % 60)
                        print(f"Running iteration {i+1}/{self.iterations}... "
                              f"Est. remaining: {remaining_min}m {remaining_sec}s", end="\r")
                
                portfolio_value = 10000.0
                current_model = "cash"
                current_lookbacks = []  # Track lookbacks for this iteration
                portfolio_values = np.zeros(n_dates)
                portfolio_values[0] = portfolio_value
                
                # Track model selections and parameters for this iteration
                current_selections = {}
                current_params = {"lookbacks": [], "models": []}

                # Generate lookbacks once at the start of iteration using new permutation-invariant system
                current_lookbacks = self._generate_diverse_lookbacks(self.n_lookbacks, iteration=i)
                current_params["lookbacks"] = current_lookbacks
                
                # Simulate trading through all dates
                for t in range(1, n_dates):
                    # Check for interrupt every 25 dates for maximum responsiveness
                    if t % 25 == 0:
                        check_interrupt()
                    
                    date = pd.Timestamp(self.dates[t])
                    prev_date = pd.Timestamp(self.dates[t-1])
                    
                    # Check if we should trade
                    if self._should_trade(self.dates[t]):
                        # Pass current lookbacks to model selection
                        current_model, _ = self._select_best_model(t, lookbacks=current_lookbacks)
                        current_selections[date.strftime('%Y-%m-%d')] = current_model
                        current_params["models"].append(current_model)
                    
                    # Update portfolio value
                    if current_model == "cash":
                        portfolio_values[t] = portfolio_values[t-1]
                    else:
                        model_return = (
                            self.portfolio_histories[current_model][date] /
                            self.portfolio_histories[current_model][prev_date]
                        )
                        portfolio_values[t] = portfolio_values[t-1] * model_return
                
                # Calculate performance metrics for this iteration
                # Use consistent portfolio calculation method for both optimization and display
                consistent_portfolio = self._calculate_model_switching_portfolio(current_lookbacks)
                metrics_dict = self.compute_performance_metrics(consistent_portfolio)
                
                # Compute focus period metrics and blended score
                focus_metrics = self.compute_focus_period_metrics(consistent_portfolio)
                if focus_metrics:
                    # Add focus period metrics to main metrics dict
                    metrics_dict.update(focus_metrics)
                
                blended_score = self.compute_blended_score(
                    focus_metrics.get('focus_1_normalized_score') if focus_metrics else None,
                    focus_metrics.get('focus_2_normalized_score') if focus_metrics else None
                )
                metrics_dict['blended_score'] = blended_score
                
                # Use blended score for Monte Carlo optimization
                current_performance = blended_score
                
                # Update permutation-invariant tracking arrays
                self._update_tracking_arrays(current_lookbacks, current_performance)
                
                # Track best performing portfolio
                if current_performance > best_performance:
                    best_performance = current_performance
                    self.best_portfolio_value = consistent_portfolio.copy()  # Use consistent calculation
                    self.best_params = current_params.copy()
                    self.best_model_selections = current_selections.copy()

                    # Create unique PNG in repo `pngs/` and save plot for this NEW BEST
                    try:
                        repo_root = os.path.dirname(os.path.dirname(__file__))
                        pngs_dir = os.path.join(repo_root, 'pngs')
                        os.makedirs(pngs_dir, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        lb_list = sorted(current_lookbacks) if current_lookbacks else []
                        lb_str = '-'.join(str(x) for x in lb_list) if lb_list else 'none'
                        score_val = metrics_dict.get('normalized_score', 0.0)
                        score_str = f"{score_val:.3f}".replace('.', '_')
                        png_name = f"monte_carlo_best_{timestamp}_lb{lb_str}_{score_str}.png"
                        png_path = os.path.join(pngs_dir, png_name)

                        # Save per-new-best plot with unique name
                        try:
                            self.create_monte_carlo_plot(consistent_portfolio, metrics_dict, save_path=png_path)
                        except Exception as e:
                            logger.warning(f"Failed to create per-best PNG {png_path}: {e}")
                            png_path = ''
                    except Exception as e:
                        logger.warning(f"Failed to prepare pngs directory: {e}")
                        png_path = ''

                    # Display performance metrics using the same consistent calculation and log PNG filename
                    # Format as ./pngs/filename.png for consistency
                    png_rel_path = f"./pngs/{png_name}" if png_path else ''
                    self._print_best_parameters(metrics_dict, png_filename=png_rel_path)

                # Record model choice
                top_models.append(current_model)
                
                # Add small delay every 100 iterations to prevent system overload
                if i % 100 == 0:
                    time.sleep(0.01)
                    
                    # Log current state of exploration/exploitation with new permutation-invariant system
                    if i > 0:
                        stats = self.get_permutation_statistics()
                        self.logger.info(
                            f"Permutation-invariant statistics after {i} iterations:\n"
                            f"  Total canonical combinations explored: {stats['total_canonical_combinations']}\n"
                            f"  Total visits: {stats['total_visits']}\n"
                            f"  Average visits per combination: {stats['average_visits_per_combination']:.2f}\n"
                            f"  Efficiency improvement factor: {stats['efficiency_improvement_factor']}"
                        )
                
        except KeyboardInterrupt:
            print("\n\nStopped early by user")
            print(f"Completed {i+1} iterations")
            if not top_models:
                return []
        
        total_time = time.time() - start_time
        print(f"\nMonte Carlo simulation completed in {total_time/60:.1f} minutes")
        print(f"Best normalized score: {best_performance:.2f}")
        
        # Print final permutation-invariant statistics
        final_stats = self.get_permutation_statistics()
        print(f"\nPermutation-Invariant Tracking Statistics:")
        print(f"  Total canonical combinations explored: {final_stats['total_canonical_combinations']}")
        print(f"  Theoretical efficiency improvement: {final_stats['efficiency_improvement_factor']}")
        if final_stats['best_performing_combination']:
            best_combo = final_stats['best_performing_combination']
            print(f"  Best performing lookbacks: {best_combo['lookbacks']} (score: {best_combo['score']:.4f})")
        
        return top_models
    
    def _should_trade(self, date_val: date) -> bool:
        """Determine if we should trade on given date."""
        if not hasattr(self, 'last_trade_date'):
            self.last_trade_date = None
            return True
            
        if self.last_trade_date is None:
            self.last_trade_date = date_val
            return True
            
        if self.trading_frequency == "monthly":
            should_trade = (
                date_val.year != self.last_trade_date.year or
                date_val.month != self.last_trade_date.month
            )
        else:  # daily
            should_trade = date_val != self.last_trade_date
            
        if should_trade:
            self.last_trade_date = date_val
            
        return should_trade
    
    def _calculate_exploitation_rate(self, iteration: int) -> float:
        """Calculate exploitation rate that increases from 5% to 75% based on iteration progress.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Float between 0.05 and 0.75 representing exploitation probability
        """
        if iteration <= self.min_iterations_for_exploit:
            rate = 0.05  # Start with 5% exploitation
        else:
            # Calculate progress after min_iterations_for_exploit
            progress = (iteration - self.min_iterations_for_exploit) / (self.iterations - self.min_iterations_for_exploit)
            progress = min(1.0, progress)  # Cap at 1.0
            
            # Linear interpolation from 5% to 75%
            rate = 0.05 + (0.75 - 0.05) * progress
        
        # Log exploitation rate every 100 iterations
        if iteration % 100 == 0:
            self.logger.info(f"Iteration {iteration}: Exploitation rate = {rate:.1%}")
            
        return rate

    def _generate_diverse_lookbacks(self, n_lookbacks: int, base_lookback: Optional[int] = None, 
                               iteration: Optional[int] = None) -> List[int]:
        """Generate lookback periods using configurable exploration/exploitation strategy.
        
        Args:
            n_lookbacks: Number of lookback periods to generate
            base_lookback: Optional base lookback period for testing
            iteration: Current iteration number for exploration/exploitation decision
            
        Returns:
            List of lookback periods within min_lookback and max_lookback bounds
        """
        if base_lookback is not None:
            # If testing with base_lookback, use original testing logic
            width = (self.max_lookback - self.min_lookback) // 6
            ranges = [
                (max(self.min_lookback, base_lookback - width), 
                 min(base_lookback + width, self.max_lookback)),
                (max(self.min_lookback, base_lookback), 
                 min(base_lookback + width, self.max_lookback)),
                (max(self.min_lookback, base_lookback + width), 
                 self.max_lookback)
            ]
            ranges = [(start, end) for start, end in ranges if start < end]
            lookbacks = []
            for range_min, range_max in ranges[:n_lookbacks]:
                lookbacks.append(random.randint(range_min, range_max))
            return sorted(lookbacks)
            
        # Generate unique lookbacks with retry mechanism
        max_attempts = 100  # Prevent infinite loops
        attempt = 0
        
        while attempt < max_attempts:
            # Determine search strategy based on search_mode
            if self.search_mode == "explore":
                # Pure exploration: always use random lookbacks
                lookbacks = self._get_random_lookbacks_exploration()
            elif self.search_mode == "exploit":
                # Pure exploitation: always use UCB1 if data available
                if self.canonical_visit_counts:
                    lookbacks = self._get_random_lookbacks_exploitation()
                else:
                    # Fallback to exploration if no data yet
                    lookbacks = self._get_random_lookbacks_exploration()
            else:  # "explore-exploit" (default)
                # Dynamic strategy: use original logic
                if iteration is not None and iteration > self.min_iterations_for_exploit:
                    exploit_rate = self._calculate_exploitation_rate(iteration)
                    if random.random() < exploit_rate:
                        # Use exploitation strategy
                        lookbacks = self._get_random_lookbacks_exploitation()
                    else:
                        # Use exploration strategy
                        lookbacks = self._get_random_lookbacks_exploration()
                else:
                    # Use exploration strategy
                    lookbacks = self._get_random_lookbacks_exploration()
            
            # Always check for duplicates to avoid revisiting combinations
            canonical = self._get_canonical_lookbacks(lookbacks)
            if canonical not in self.combination_indices:
                # This is a new combination, return it
                return lookbacks
            
            attempt += 1
            if attempt % 10 == 0:
                self.logger.debug(f"Attempt {attempt}: Regenerating lookbacks due to duplicate {canonical}")
        
        # If we've tried many times and still getting duplicates, 
        # fall back to returning the last generated combination
        # This can happen when we've explored most of the space
        self.logger.warning(f"Could not generate unique lookbacks after {max_attempts} attempts, "
                           f"using duplicate combination {canonical}")
        return lookbacks

    def _select_best_model(self, current_idx: int, 
                          lookbacks: Optional[List[int]] = None,
                          iteration: Optional[int] = None) -> Tuple[str, List[int]]:
        """Select best performing model based on multiple performance metrics and lookbacks.
        
        Args:
            current_idx: Current index in the backtest
            lookbacks: List of lookback periods to use (should be passed from iteration)
            iteration: Current iteration number (not used when lookbacks are passed)
            
        Returns:
            Tuple of (best_model_name, list_of_lookbacks_used)
        """
        # Get configuration for model selection
        config_path = os.path.join(os.path.dirname(self.model_paths[next(iter(self.model_paths))]), 
                                 'pytaaa_model_switching_params.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        metric_weights = config['model_selection']['performance_metrics']
        
        # Use passed lookbacks or generate new ones if not provided
        # This fallback should only be used in testing scenarios
        if lookbacks is None:
            n_lookbacks = config['model_selection']['n_lookbacks']
            lookbacks = self._generate_diverse_lookbacks(n_lookbacks, iteration=iteration)
            logger.warning("No lookbacks provided to _select_best_model, generating new ones")
        
        # FIXED: Force consistent alphabetical ordering of models
        models = sorted(list(self.portfolio_histories.keys()))
        all_ranks = np.zeros((len(models), 5 * len(lookbacks))) # 5 metrics × n_lookbacks
        
        # Calculate metrics for each lookback period
        for i, lookback_period in enumerate(lookbacks):
            start_idx = max(0, current_idx - lookback_period)
            start_date = pd.Timestamp(self.dates[start_idx])
            end_date = pd.Timestamp(self.dates[current_idx - 1])
            
            # Calculate metrics for each model
            metrics_list = []
            for model in models:
                if model == "cash":
                    # Create synthetic constant portfolio for cash
                    portfolio_values = np.ones(lookback_period) * 10000.0
                else:
                    portfolio_values = self.portfolio_histories[model][start_date:end_date].values
                
                metrics = compute_daily_metrics(portfolio_values)
                metrics_list.append(metrics)
            
            # Get ranks for this lookback period
            period_ranks = rank_models(metrics_list)
            
            # Store ranks with metric weights applied
            weights = np.array([
                metric_weights.get('sharpe_ratio_weight', 1.0),
                metric_weights.get('sortino_ratio_weight', 1.0),
                metric_weights.get('max_drawdown_weight', 1.0),
                metric_weights.get('avg_drawdown_weight', 1.0),
                metric_weights.get('annualized_return_weight', 1.0)
            ])
            
            start_col = i * 5  # Changed from 7 to 5
            all_ranks[:, start_col:start_col + 5] = period_ranks.T * weights[:, np.newaxis].T
        
        # Calculate average rank across all metrics and lookbacks
        avg_ranks = np.mean(all_ranks, axis=1)
        
        # Select model with best (lowest) average rank
        best_model_idx = np.argmin(avg_ranks)
        return models[best_model_idx], lookbacks

    def compute_performance_metrics(self, portfolio_values: np.ndarray) -> Dict[str, float]:
        """Compute summary performance metrics for displaying results.
        
        Args:
            portfolio_values: Array of portfolio values over time
            
        Returns:
            Dictionary containing performance metrics including normalized_score and period metrics
        """
        # Check if this is a cash portfolio (constant values)
        is_cash_portfolio = len(np.unique(portfolio_values)) == 1
        
        metrics = compute_daily_metrics(portfolio_values)
        
        # Calculate basic metrics
        basic_metrics = {
            'final_value': portfolio_values[-1],
            'annual_return': ((portfolio_values[-1] / portfolio_values[0]) ** (252 / len(portfolio_values)) - 1),
            'sharpe_ratio': metrics.sharpe_ratio,
            'sortino_ratio': metrics.sortino_ratio,
            'max_drawdown': metrics.max_drawdown,
            'avg_drawdown': metrics.avg_drawdown
        }
        
        # For cash portfolios, set normalized_score to 0.0 (neutral baseline)
        if is_cash_portfolio:
            basic_metrics['normalized_score'] = 0.0
        else:
            # Define normalization parameters (excluding final_value from normalized score)
            # Updated with more realistic standard deviations based on typical portfolio variations
            central_values = self.CENTRAL_VALUES
            std_values = self.STD_VALUES
            
            # Calculate normalized metrics (excluding final_value)
            normalized_metrics = {}
            for metric_name in ['annual_return', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'avg_drawdown']:
                raw_value = basic_metrics[metric_name]
                central = central_values[metric_name]
                std = std_values[metric_name]
                normalized_metrics[metric_name] = (raw_value - central) / std
            
            # Calculate normalized average score (excluding final_value)
            normalized_score = sum(normalized_metrics.values()) / len(normalized_metrics)
            basic_metrics['normalized_score'] = normalized_score

        # Add period metrics and model-switching effectiveness analysis
        try:
            # Only calculate period metrics if we have valid portfolio data
            if len(portfolio_values) > 1 and len(self.dates) == len(portfolio_values):
                period_metrics = calculate_period_metrics(portfolio_values, self.dates)
                basic_metrics['period_metrics'] = period_metrics
                
                # Calculate model-switching effectiveness if we have best params
                if hasattr(self, 'best_params') and self.best_params.get('lookbacks'):
                    effectiveness_analysis = analyze_model_switching_effectiveness(
                        self, self.best_params['lookbacks']
                    )
                    basic_metrics.update({
                        'model_effectiveness': effectiveness_analysis,
                        'outperformance_percentage': effectiveness_analysis.get('sharpe_outperformance_pct', 0.0)
                    })
                else:
                    # Use current portfolio values for effectiveness analysis if no best_params yet
                    effectiveness_analysis = analyze_model_switching_effectiveness(
                        self, [50, 150, 250]  # Default lookbacks
                    )
                    basic_metrics.update({
                        'model_effectiveness': effectiveness_analysis,
                        'outperformance_percentage': effectiveness_analysis.get('sharpe_outperformance_pct', 0.0)
                    })
            else:
                # Set default values if portfolio data is invalid
                basic_metrics['period_metrics'] = {}
                basic_metrics['outperformance_percentage'] = 0.0
                
        except Exception as e:
            logger.warning(f"Error calculating period metrics: {e}")
            basic_metrics['period_metrics'] = {}
            basic_metrics['outperformance_percentage'] = 0.0
        
        return basic_metrics

    def compute_focus_period_metrics(self, portfolio_values: np.ndarray) -> Optional[Dict[str, float]]:
        """Compute performance metrics for TWO focus periods.
        
        Args:
            portfolio_values: Array of portfolio values over time
            
        Returns:
            Dictionary containing both focus period metrics, or None if periods not available
        """
        if not self.focus_period_enabled:
            return None
        
        #############################################################################
        # Helper function to extract metrics for a single focus period
        #############################################################################
        def compute_single_period_metrics(start_date_str: str, end_date_str: str, 
                                          prefix: str) -> Optional[Dict[str, float]]:
            """Compute metrics for a single focus period."""
            try:
                start_date = pd.to_datetime(start_date_str).date()
                end_date = pd.to_datetime(end_date_str).date()
                
                # Find indices for focus period in our data
                start_idx = None
                end_idx = None
                
                for i, date_val in enumerate(self.dates):
                    if isinstance(date_val, pd.Timestamp):
                        date_val = date_val.date()
                    
                    if start_idx is None and date_val >= start_date:
                        start_idx = i
                    if date_val <= end_date:
                        end_idx = i
                
                # Check if we have sufficient data for the focus period
                if start_idx is None or end_idx is None or start_idx >= end_idx:
                    return None
                    
                # Extract focus period portfolio values
                focus_period_values = portfolio_values[start_idx:end_idx + 1]
                
                if len(focus_period_values) < 2:
                    return None
                
                #############################################################################
                # Normalize focus period to start at $10,000
                #############################################################################
                initial_value = focus_period_values[0]
                normalized_focus_values = focus_period_values * (10000.0 / initial_value)
                
                # Check if this is a cash portfolio (constant values)
                is_cash_portfolio = len(np.unique(normalized_focus_values)) == 1
                
                # Calculate focus period metrics using normalized values
                metrics = compute_daily_metrics(normalized_focus_values)
                
                period_metrics = {
                    f'{prefix}_final_value': normalized_focus_values[-1],
                    f'{prefix}_annual_return': ((normalized_focus_values[-1] / normalized_focus_values[0]) ** (252 / len(normalized_focus_values)) - 1),
                    f'{prefix}_sharpe_ratio': metrics.sharpe_ratio,
                    f'{prefix}_sortino_ratio': metrics.sortino_ratio,
                    f'{prefix}_max_drawdown': metrics.max_drawdown,
                    f'{prefix}_avg_drawdown': metrics.avg_drawdown
                }
                
                # Calculate normalized score for focus period
                if is_cash_portfolio:
                    period_metrics[f'{prefix}_normalized_score'] = 0.0
                else:
                    # Use same normalization parameters as full period
                    central_values = self.CENTRAL_VALUES
                    std_values = self.STD_VALUES
                    
                    normalized_metrics = {}
                    for metric_name in ['annual_return', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'avg_drawdown']:
                        period_metric_name = f'{prefix}_{metric_name}'
                        raw_value = period_metrics[period_metric_name]
                        central = central_values[metric_name]
                        std = std_values[metric_name]
                        normalized_metrics[metric_name] = (raw_value - central) / std
                    
                    period_metrics[f'{prefix}_normalized_score'] = sum(normalized_metrics.values()) / len(normalized_metrics)
                
                return period_metrics
                
            except Exception as e:
                logger.warning(f"Error computing {prefix} metrics: {e}")
                return None

        #############################################################################
        # Compute metrics for both focus periods
        #############################################################################
        all_metrics = {}
        
        if self.has_two_focus_periods:
            # Get configuration for both focus periods
            fp1_start = self.metric_blending_config.get('focus_period_1_start', '2010-01-01')
            fp1_end = self.metric_blending_config.get('focus_period_1_end', '2015-01-01')
            fp2_start = self.metric_blending_config.get('focus_period_2_start', '2015-01-01')
            fp2_end = self.metric_blending_config.get('focus_period_2_end', '2020-01-01')
            
            # Compute metrics for focus period 1
            fp1_metrics = compute_single_period_metrics(fp1_start, fp1_end, 'focus_1')
            if fp1_metrics:
                all_metrics.update(fp1_metrics)
            
            # Compute metrics for focus period 2
            fp2_metrics = compute_single_period_metrics(fp2_start, fp2_end, 'focus_2')
            if fp2_metrics:
                all_metrics.update(fp2_metrics)
            
            # Only return if we have both periods
            if fp1_metrics and fp2_metrics:
                return all_metrics
            else:
                return None
        else:
            # Legacy single focus period support
            fp_start = self.metric_blending_config.get('focus_period_start', '2003-01-01')
            fp_end = self.metric_blending_config.get('focus_period_end', '2009-12-31')
            
            fp_metrics = compute_single_period_metrics(fp_start, fp_end, 'focus')
            if fp_metrics:
                return fp_metrics
            else:
                return None

    def compute_blended_score(self, focus_1_score: Optional[float] = None, 
                             focus_2_score: Optional[float] = None) -> float:
        """Compute blended score combining TWO focus period metrics.
        
        Args:
            focus_1_score: Normalized score for focus period 1 (can be None)
            focus_2_score: Normalized score for focus period 2 (can be None)
            
        Returns:
            Blended score weighted according to configuration
        """
        if not self.focus_period_enabled:
            # If focus periods are disabled, return 0.0 as neutral baseline
            return 0.0
        
        # Check if we have two focus periods
        if self.has_two_focus_periods:
            # Need both scores to compute blended score
            if focus_1_score is None or focus_2_score is None:
                # If either score is missing, return 0.0
                return 0.0
            
            # Get weights for each focus period (should sum to 1.0 typically)
            fp1_weight = self.metric_blending_config.get('focus_period_1_weight', 0.5)
            fp2_weight = self.metric_blending_config.get('focus_period_2_weight', 0.5)
            
            # Normalize weights to sum to 1.0
            total_weight = fp1_weight + fp2_weight
            if total_weight <= 0:
                return 0.0
                
            normalized_fp1_weight = fp1_weight / total_weight
            normalized_fp2_weight = fp2_weight / total_weight
            
            blended_score = (normalized_fp1_weight * focus_1_score + 
                           normalized_fp2_weight * focus_2_score)
            
            return blended_score
        else:
            # Legacy single focus period support
            if focus_1_score is None:
                return 0.0
            
            # With only one focus period, just return its score
            return focus_1_score

    def create_monte_carlo_plot(self, portfolio_values: np.ndarray, 
                               metrics: dict, save_path: Optional[str] = None,
                               custom_text: Optional[str] = None) -> None:
        """Create Monte Carlo optimization performance plot with model rankings.
        
        This unified function creates plots showing:
        - Portfolio performance comparison (upper subplot)
        - Model selection timeline (lower subplot)
        - Model rankings for current date and first weekday of month
        
        Args:
            portfolio_values: Array of portfolio values over time (ignored, recalculated)
            metrics: Dictionary of performance metrics
            save_path: Optional path to save the plot (defaults to 'monte_carlo_best_performance.png')
            custom_text: Optional custom text for upper subplot (if None, uses default text)
        """
        from calendar import monthrange
        from datetime import date as date_type  # Import with alias to avoid conflicts
        
        if save_path is None:
            save_path = 'monte_carlo_best_performance.png'
            
        #############################################################################
        # Calculate dynamic model-switching portfolio values
        #############################################################################
        # Use the best lookbacks to calculate the proper model-switching portfolio
        lookbacks = self.best_params.get('lookbacks', [50, 150, 250])
        model_switching_portfolio = self._calculate_model_switching_portfolio(lookbacks)
        
        # Recalculate metrics based on the model-switching portfolio
        model_switching_metrics = self.compute_performance_metrics(model_switching_portfolio)
            
        #############################################################################
        # Create figure with subplot layout
        #############################################################################
        fig = plt.figure(figsize=(12, 7))
        gs = plt.GridSpec(6, 1)
        
        # Main portfolio performance plot (83% of space)
        ax1 = fig.add_subplot(gs[0:5, 0])
        
        #############################################################################
        # Define explicit model-to-color mapping
        #############################################################################
        model_to_color = {
            'sp500_hma': 'c',      # cyan
            'sp500_pine': 'm',     # magenta (new model)
            'naz100_pine': 'b',    # blue (royal blue)
            'naz100_hma': 'r',     # red
            'naz100_pi': 'g',      # green
            'cash': 'k'            # black
        }
        
        date_index = pd.DatetimeIndex(self.dates)
        
        # FIXED: Use sorted items to ensure consistent ordering
        sorted_portfolio_items = sorted(self.portfolio_histories.items(), key=lambda x: x[0])
        
        # Normalize all series to start at 10000 and plot (excluding cash from upper plot)
        for (model, values) in sorted_portfolio_items:
            if model != "cash":
                start_value = values.iloc[0]
                normalized_values = values * (10000.0 / start_value)
                ax1.plot(date_index, normalized_values,
                        color=model_to_color.get(model, 'gray'), alpha=0.5, 
                        label=f"{model}", linewidth=1)
        
        #############################################################################
        # Plot the model-switching portfolio (the optimized portfolio)
        #############################################################################
        ax1.plot(date_index, model_switching_portfolio,
                color='black', alpha=0.9, linewidth=2.5,
                label='Model-Switching Portfolio', linestyle='-')
        
        #############################################################################
        # Configure main plot styling
        #############################################################################
        ax1.set_yscale('log')
        
        # Configure grid with both major and minor lines
        ax1.grid(True, which='major', alpha=0.4, linewidth=0.8)
        ax1.grid(True, which='minor', alpha=0.2, linewidth=0.5)
        ax1.minorticks_on()
        
        # Configure date formatting for upper subplot - 5 year major, 1 year minor
        ax1.xaxis.set_major_locator(mdates.YearLocator(5))  # Every 5 years
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Just year
        ax1.xaxis.set_minor_locator(mdates.YearLocator(1))  # Every year
        
        ax1.set_title('Portfolio Performance: Monte Carlo Optimization vs Base Models', 
                     pad=20, fontsize=14)
        ax1.set_xlabel('')  # Remove x-label from top plot
        ax1.set_ylabel('Portfolio Value ($)', fontsize=10)  # Reduced from 12 to 10
        
        # Format axes with reduced font sizes
        ax1.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'${int(x):,}')
        )
        ax1.tick_params(axis='y', which='major', labelsize=8)  # Reduced from 10 to 8
        ax1.tick_params(axis='y', which='minor', labelsize=6)  # Reduced from 8 to 6
        
        # Set x-axis labels to be horizontal (no rotation)
        ax1.tick_params(axis='x', which='major', labelsize=10, rotation=0)
        
        #############################################################################
        # Generate text content for upper subplot
        #############################################################################
        if custom_text is not None:
            # Use provided custom text
            full_text = custom_text
        else:
            # Generate default text with current date/time and lookback parameters
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            _lb_list = sorted(lookbacks) if isinstance(lookbacks, list) else lookbacks
            lookback_text = f"Best parameters: lookbacks={_lb_list} days"
            
            # Update text to show model-switching portfolio metrics
            text_str = (f"{current_time}\n{lookback_text}\n"
                       f"Model-switching portfolio:\n"
                       f"  Final Value: ${model_switching_metrics['final_value']:,.0f}\n"
                       f"  Annual Return: {model_switching_metrics['annual_return']*100:.1f}%\n"
                       f"  Normalized Score: {model_switching_metrics['normalized_score']:.3f}")
            
            #############################################################################
            # Add model ranking information for current date and first weekday of month
            #############################################################################
            try:
                # Get current date (August 2, 2025)
                current_date = date_type(2025, 8, 2)
                current_date_str = current_date.strftime("%Y-%m-%d")
                
                # Find first weekday of current month (August 2025)
                year, month = 2025, 8
                first_weekday = None
                for day in range(1, monthrange(year, month)[1] + 1):
                    test_date = date_type(year, month, day)
                    if test_date.weekday() < 5:  # Monday=0, Friday=4
                        first_weekday = test_date
                        break
                
                if first_weekday:
                    first_weekday_str = first_weekday.strftime("%Y-%m-%d")
                else:
                    first_weekday_str = "No weekday found"
                
                # Get model rankings for both dates if they exist in our data
                ranking_text = ""
                
                # Find closest available date to current date
                available_dates = [d for d in self.dates]
                if available_dates:
                    # Find the most recent date <= current_date
                    closest_current = None
                    for d in reversed(available_dates):
                        if d <= current_date:
                            closest_current = d
                            break
                    
                    # Find closest date to first weekday of month
                    closest_first_weekday = None
                    if first_weekday:
                        for d in reversed(available_dates):
                            if d <= first_weekday:
                                closest_first_weekday = d
                                break
                    
                    # Generate ranking text for current date
                    if closest_current and len(self.dates) > 1:
                        current_idx = self.dates.index(closest_current)
                        if current_idx > 0:
                            try:
                                # Get model rankings using current best lookbacks
                                models = sorted(list(self.portfolio_histories.keys()))
                                lookback_period = max(lookbacks) if lookbacks else 60
                                start_idx = max(0, current_idx - lookback_period)
                                start_date = pd.Timestamp(self.dates[start_idx])
                                end_date = pd.Timestamp(self.dates[current_idx - 1])
                                
                                # Calculate metrics and ranks
                                metrics_list = []
                                for model in models:
                                    if model == "cash":
                                        portfolio_vals = np.ones(lookback_period) * 10000.0
                                    else:
                                        portfolio_vals = self.portfolio_histories[model][start_date:end_date].values
                                        if len(portfolio_vals) == 0:
                                            portfolio_vals = np.ones(lookback_period) * 10000.0
                                    
                                    model_metrics = compute_daily_metrics(portfolio_vals)
                                    metrics_list.append(model_metrics)
                                
                                # Rank models by normalized score (primary metric)
                                normalized_scores = [self.compute_performance_metrics(portfolio_vals)['normalized_score'] 
                                                   for portfolio_vals in [np.ones(lookback_period) * 10000.0 if model == "cash" 
                                                                        else self.portfolio_histories[model][start_date:end_date].values 
                                                                        for model in models]]
                                model_ranking = sorted(zip(models, normalized_scores), key=lambda x: x[1], reverse=True)
                                
                                ranking_text += f"\n\nModel ranks on {closest_current.strftime('%Y-%m-%d')}:\n"
                                for i, (model, score) in enumerate(model_ranking, 1):
                                    ranking_text += f"{i:>1}. {model:<13} {score:>7.3f}\n"
                                    
                            except Exception as e:
                                ranking_text += f"\n\nModel ranks on {closest_current.strftime('%Y-%m-%d')}: [Error calculating]\n"
                    
                    # Generate ranking text for first weekday of month (if different from current)
                    if (closest_first_weekday and closest_first_weekday != closest_current and 
                        len(self.dates) > 1):
                        first_weekday_idx = self.dates.index(closest_first_weekday)
                        if first_weekday_idx > 0:
                            try:
                                # Similar ranking calculation for first weekday
                                models = sorted(list(self.portfolio_histories.keys()))
                                lookback_period = max(lookbacks) if lookbacks else 60
                                start_idx = max(0, first_weekday_idx - lookback_period)
                                start_date = pd.Timestamp(self.dates[start_idx])
                                end_date = pd.Timestamp(self.dates[first_weekday_idx - 1])
                                
                                metrics_list = []
                                for model in models:
                                    if model == "cash":
                                        portfolio_vals = np.ones(lookback_period) * 10000.0
                                    else:
                                        portfolio_vals = self.portfolio_histories[model][start_date:end_date].values
                                        if len(portfolio_vals) == 0:
                                            portfolio_vals = np.ones(lookback_period) * 10000.0
                                    
                                    model_metrics = compute_daily_metrics(portfolio_vals)
                                    metrics_list.append(model_metrics)
                                
                                normalized_scores = [self.compute_performance_metrics(portfolio_vals)['normalized_score'] 
                                                   for portfolio_vals in [np.ones(lookback_period) * 10000.0 if model == "cash" 
                                                                        else self.portfolio_histories[model][start_date:end_date].values 
                                                                        for model in models]]
                                model_ranking = sorted(zip(models, normalized_scores), key=lambda x: x[1], reverse=True)
                                
                                ranking_text += f"\n\nModel ranks on {closest_first_weekday.strftime('%Y-%m-%d')}:\n"
                                for i, (model, score) in enumerate(model_ranking, 1):
                                    ranking_text += f"{i:>1}. {model:<13} {score:>7.3f}\n"
                                    
                            except Exception as e:
                                ranking_text += f"\n\nModel ranks on {closest_first_weekday.strftime('%Y-%m-%d')}: [Error calculating]\n"
                
                # Combine all text
                full_text = text_str + ranking_text
                
            except Exception as e:
                # Fallback to original text if ranking calculation fails
                full_text = text_str + f"\n[Model ranking unavailable: {str(e)}]"
        
        # Position text in upper left with monospace font for proper table alignment
        ax1.text(0.02, 0.95, full_text,
                transform=ax1.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                fontsize=7, verticalalignment='top',  # Reduced from 8 to 7
                fontfamily='monospace')  # Use monospace font for table alignment
        
        # Add legend to main plot
        ax1.legend(loc='lower right', fontsize=8)
        
        #############################################################################
        # Create model selection subplot (17% of space) showing actual switching
        #############################################################################
        ax2 = fig.add_subplot(gs[5, 0])
        
        # Create numeric mapping for all models
        unique_models = sorted(list(self.portfolio_histories.keys()))
        model_to_num = {model: i for i, model in enumerate(unique_models)}
        
        # Calculate model selections over time using the same logic as portfolio calculation
        current_model = "cash"
        model_selections = []
        self.last_trade_date = None  # Reset for this calculation
        
        for t, date_val in enumerate(self.dates):
            # Check if we should trade (monthly rebalancing)
            if self._should_trade(date_val):
                # Select best model for this date using specified lookbacks
                if t >= max(lookbacks):  # Ensure we have enough history
                    current_model, _ = self._select_best_model(t, lookbacks=lookbacks)
                else:
                    current_model = "cash"  # Default to cash if insufficient history
            
            model_selections.append(current_model)
        
        # Convert to numeric values and plot
        numeric_selections = [model_to_num.get(model, -1) for model in model_selections]
        
        for model in unique_models:
            mask = [x == model_to_num[model] for x in numeric_selections]
            if any(mask):
                ax2.scatter(date_index[mask], [model_to_num[model]] * sum(mask),
                      color=model_to_color.get(model, 'gray'), alpha=0.7, s=20,
                      label=f"{model} periods")
                ax2.plot(date_index[mask], [model_to_num[model]] * sum(mask),
                    color=model_to_color.get(model, 'gray'), alpha=0.3, linewidth=1)
        
        #############################################################################
        # Configure lower subplot to match upper subplot exactly
        #############################################################################
        # Add major and minor grids to match upper subplot
        ax2.grid(True, which='major', alpha=0.4, linewidth=0.8)
        ax2.grid(True, which='minor', alpha=0.2, linewidth=0.5)
        ax2.minorticks_on()
        
        # Configure date formatting to match upper subplot - 5 year major, 1 year minor
        ax2.xaxis.set_major_locator(mdates.YearLocator(5))  # Every 5 years
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Just year
        ax2.xaxis.set_minor_locator(mdates.YearLocator(1))  # Every year
        
        # Set labels with reduced font sizes
        ax2.set_xlabel('Date', fontsize=10)  # Reduced from 12 to 10
        ax2.set_ylabel('Selected Model', fontsize=10)  # Reduced from 12 to 10
        
        # Reduce y-axis label font sizes
        ax2.tick_params(axis='y', which='major', labelsize=10)  # Reduced from 12 to 10
        ax2.tick_params(axis='x', which='major', labelsize=10, rotation=0)  # Horizontal year labels
        
        # Add legend
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=7)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _calculate_model_switching_portfolio(self, lookbacks: Optional[List[int]] = None) -> np.ndarray:
        """Calculate portfolio values using dynamic model switching.
        
        This method simulates the model-switching strategy across all dates,
        selecting the best model each month based on the specified lookbacks.
        
        Args:
            lookbacks: Lookback periods to use for model selection. 
                      If None, uses best_params lookbacks.
        
        Returns:
            Array of portfolio values over time using model switching
        """
        if lookbacks is None:
            lookbacks = self.best_params.get('lookbacks', [50, 150, 250])
        
        if not lookbacks:
            # Fallback to a reasonable default
            lookbacks = [50, 150, 250]
        
        n_dates = len(self.dates)
        portfolio_values = np.zeros(n_dates)
        portfolio_values[0] = 10000.0
        current_model = "cash"
        
        # Reset trading state for this calculation
        self.last_trade_date = None
        
        # Simulate trading through all dates
        for t in range(1, n_dates):
            date_val = self.dates[t]
            prev_date_val = self.dates[t-1]
            
            # Convert date objects to pandas Timestamps for indexing
            date = pd.Timestamp(date_val)
            prev_date = pd.Timestamp(prev_date_val)
            
            # Check if we should trade (monthly rebalancing)
            if self._should_trade(date_val):
                # Select best model for this date using specified lookbacks
                if t >= max(lookbacks):  # Ensure we have enough history
                    current_model, _ = self._select_best_model(t, lookbacks=lookbacks)
                else:
                    current_model = "cash"  # Default to cash if insufficient history
            
            # Update portfolio value based on current model
            if current_model == "cash":
                portfolio_values[t] = portfolio_values[t-1]
            else:
                try:
                    model_return = (
                        self.portfolio_histories[current_model][date] /
                        self.portfolio_histories[current_model][prev_date]
                    )
                    portfolio_values[t] = portfolio_values[t-1] * model_return
                except (KeyError, ZeroDivisionError):
                    # Fallback to cash if model data is unavailable
                    portfolio_values[t] = portfolio_values[t-1]
        
        return portfolio_values

    def save_state(self, filename: str = "monte_carlo_state.pkl") -> None:
        """Save current Monte Carlo state to pickle file.
        
        Args:
            filename: Path to save the state file
        """
        try:
            state_data = {
                'timestamp': datetime.now().isoformat(),
                'combination_indices': self.combination_indices,
                'canonical_performance_scores': self.canonical_performance_scores,
                'canonical_visit_counts': self.canonical_visit_counts,
                'equivalent_combinations_found': self.equivalent_combinations_found,
                'min_lookback': self.min_lookback,
                'max_lookback': self.max_lookback,
                'n_lookbacks': self.n_lookbacks,
                'search_mode': self.search_mode
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(state_data, f)
                
            self.logger.info(f"State saved: {len(self.combination_indices)} combinations, "
                           f"{sum(self.canonical_visit_counts)} total visits")
        except Exception as e:
            self.logger.error(f"Failed to save state: {str(e)}")
            raise

    def load_state(self, filename: str = "monte_carlo_state.pkl") -> bool:
        """Load Monte Carlo state from pickle file.
        
        Args:
            filename: Path to load the state file from
            
        Returns:
            True if state was loaded successfully, False otherwise
        """
        if not os.path.exists(filename):
            print(f"No previous state file found at {filename}")
            return False
            
        try:
            with open(filename, 'rb') as f:
                state_data = pickle.load(f)
            
            # Verify compatibility with current settings
            if (state_data.get('min_lookback') != self.min_lookback or
                state_data.get('max_lookback') != self.max_lookback or
                state_data.get('n_lookbacks') != self.n_lookbacks):
                print(
                    f"State file parameters don't match current settings. "
                    f"Saved: min={state_data.get('min_lookback')}, "
                    f"max={state_data.get('max_lookback')}, "
                    f"n={state_data.get('n_lookbacks')}. "
                    f"Current: min={self.min_lookback}, "
                    f"max={self.max_lookback}, n={self.n_lookbacks}. "
                    f"Starting fresh."
                )
                return False
            
            # Restore state
            self.combination_indices = state_data.get('combination_indices', {})
            self.canonical_performance_scores = state_data.get('canonical_performance_scores', [])
            self.canonical_visit_counts = state_data.get('canonical_visit_counts', [])
            self.equivalent_combinations_found = state_data.get('equivalent_combinations_found', 0)
            
            # Print restoration summary
            saved_time = state_data.get('timestamp', 'unknown')
            print(f"Monte Carlo state loaded from {filename}")
            print(f"  Saved: {saved_time}")
            print(f"  Restored: {len(self.combination_indices)} combinations")
            print(f"  Total visits: {sum(self.canonical_visit_counts)}")
            
            if self.canonical_performance_scores:
                best_score = max(self.canonical_performance_scores)
                print(f"  Best previous score: {best_score:.4f}")
            
            self.logger.info(f"State loaded: {len(self.combination_indices)} combinations, "
                           f"{sum(self.canonical_visit_counts)} total visits")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load state: {str(e)}")
            print(f"Error loading state from {filename}: {str(e)}")
            print("Starting with fresh state")
            return False

    def clear_state(self) -> None:
        """Clear all stored state data."""
        self.combination_indices.clear()
        self.canonical_performance_scores.clear()
        self.canonical_visit_counts.clear()
        self.equivalent_combinations_found = 0
        self.logger.info("State cleared")
    
    def _update_tracking_arrays(self, lookbacks: List[int], performance: float) -> None:
        """Update tracking arrays for permutation-invariant exploration/exploitation.
        
        Args:
            lookbacks: List of lookback periods used in this iteration
            performance: Performance score achieved with these lookbacks
        """
        # Convert to canonical (sorted) form
        canonical = self._get_canonical_lookbacks(lookbacks)
        
        if canonical in self.combination_indices:
            # Update existing combination
            idx = self.combination_indices[canonical]
            self.canonical_visit_counts[idx] += 1
            # Update performance with exponential moving average
            alpha = 0.1  # Learning rate
            self.canonical_performance_scores[idx] = (
                (1 - alpha) * self.canonical_performance_scores[idx] + 
                alpha * performance
            )
        else:
            # Add new combination
            idx = len(self.canonical_performance_scores)
            self.combination_indices[canonical] = idx
            self.canonical_performance_scores.append(performance)
            self.canonical_visit_counts.append(1)

    def _get_canonical_lookbacks(self, lookbacks: List[int]) -> Tuple[int, ...]:
        """Convert lookbacks to canonical (sorted) tuple form."""
        return tuple(sorted(lookbacks))

    def _get_random_lookbacks_exploration(self) -> List[int]:
        """Generate random lookbacks for exploration."""
        lookbacks = []
        for _ in range(self.n_lookbacks):
            lookback = random.randint(self.min_lookback, self.max_lookback)
            lookbacks.append(lookback)
        return sorted(lookbacks)

    def _get_random_lookbacks_exploitation(self) -> List[int]:
        """Generate lookbacks using Upper Confidence Bound (UCB1) for exploitation."""
        if not self.canonical_performance_scores:
            return self._get_random_lookbacks_exploration()
        
        # Calculate UCB1 scores for all combinations
        total_visits = sum(self.canonical_visit_counts)
        ucb_scores = []
        
        for i, (score, visits) in enumerate(zip(self.canonical_performance_scores, self.canonical_visit_counts)):
            if visits == 0:
                ucb_score = float('inf')
            else:
                exploration_bonus = math.sqrt(2 * math.log(total_visits) / visits)
                ucb_score = score + exploration_bonus
            ucb_scores.append(ucb_score)
        
        # Select combination with highest UCB1 score
        best_idx = np.argmax(ucb_scores)
        canonical_combo = list(self.combination_indices.keys())[best_idx]
        
        # Convert back to list and add some noise
        lookbacks = list(canonical_combo)
        
        # Add small amount of noise to avoid exact repetition
        noise_scale = 0.1
        for i in range(len(lookbacks)):
            noise = random.uniform(-noise_scale * lookbacks[i], noise_scale * lookbacks[i])
            lookbacks[i] = int(max(self.min_lookback, min(self.max_lookback, lookbacks[i] + noise)))
        
        return sorted(lookbacks)

    def _log_best_parameters_to_csv(self, metrics: Dict[str, float], png_filename: Optional[str] = None) -> None:
        """Log best performing parameters to CSV file with full period and both focus periods.
        
        Args:
            metrics: Dictionary containing performance metrics for all periods
        """
        import csv
        import os
        from datetime import datetime
        
        #############################################################################
        # Setup CSV file path and check if it exists
        #############################################################################
        csv_filename = "abacus_best_performers.csv"
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                csv_filename)
        file_exists = os.path.exists(csv_path)
        
        #############################################################################
        # Extract lookback periods and sort them
        #############################################################################
        lookbacks = self.best_params.get('lookbacks', [])
        sorted_lookbacks = sorted(lookbacks)
        
        # Pad lookbacks to ensure we have exactly 3 values
        while len(sorted_lookbacks) < 3:
            sorted_lookbacks.append(0)
        
        #############################################################################
        # Get Monte Carlo normalization parameters
        #############################################################################
        central_values = self.CENTRAL_VALUES
        std_values = self.STD_VALUES
        
        #############################################################################
        # Get performance metric weights from JSON config
        #############################################################################
        if self.json_config and 'model_selection' in self.json_config:
            metric_weights = self.json_config['model_selection'].get(
                'performance_metrics', {})
            sharpe_weight = metric_weights.get('sharpe_ratio_weight', 1.0)
            sortino_weight = metric_weights.get('sortino_ratio_weight', 1.0)
            max_dd_weight = metric_weights.get('max_drawdown_weight', 1.0)
            avg_dd_weight = metric_weights.get('avg_drawdown_weight', 1.0)
            annual_ret_weight = metric_weights.get(
                'annualized_return_weight', 1.0)
        else:
            sharpe_weight = 1.0
            sortino_weight = 1.0
            max_dd_weight = 1.0
            avg_dd_weight = 1.0
            annual_ret_weight = 1.0
        
        #############################################################################
        # Recalculate model effectiveness for consistency
        #############################################################################
        from functions.PortfolioMetrics import (
            analyze_model_switching_effectiveness)
        
        if hasattr(self, 'best_params') and self.best_params.get('lookbacks'):
            fresh_effectiveness = analyze_model_switching_effectiveness(
                self, self.best_params['lookbacks'])
        else:
            fresh_effectiveness = analyze_model_switching_effectiveness(
                self, lookbacks)
        
        model_effectiveness = fresh_effectiveness
        
        #############################################################################
        # Extract full period metrics (always available)
        #############################################################################
        full_period_metrics = {
            'Full Period Final Value': f"${metrics['final_value']:,.0f}",
            'Full Period Annual Return': f"{metrics['annual_return']*100:.2f}%",
            'Full Period Sharpe Ratio': f"{metrics['sharpe_ratio']:.5f}",
            'Full Period Sortino Ratio': f"{metrics['sortino_ratio']:.3f}",
            'Full Period Max Drawdown': f"{metrics['max_drawdown']*100:.2f}%",
            'Full Period Avg Drawdown': f"{metrics['avg_drawdown']*100:.2f}%",
            'Full Period Normalized Score': f"{metrics['normalized_score']:.3f}",
            'Full Period Start Date': str(self.dates[0]) if self.dates else '',
            'Full Period End Date': str(self.dates[-1]) if self.dates else '',
        }
        
        #############################################################################
        # Extract focus period metrics based on configuration
        #############################################################################
        focus_1_metrics = {}
        focus_2_metrics = {}
        focus_tracking_params = {}
        
        if self.has_two_focus_periods and 'focus_1_annual_return' in metrics:
            # TWO focus periods available
            focus_1_metrics = {
                'Focus Period 1 Final Value': 
                    f"${metrics['focus_1_final_value']:,.0f}",
                'Focus Period 1 Annual Return': 
                    f"{metrics['focus_1_annual_return']*100:.2f}%",
                'Focus Period 1 Sharpe Ratio': 
                    f"{metrics['focus_1_sharpe_ratio']:.5f}",
                'Focus Period 1 Sortino Ratio': 
                    f"{metrics['focus_1_sortino_ratio']:.3f}",
                'Focus Period 1 Max Drawdown': 
                    f"{metrics['focus_1_max_drawdown']*100:.2f}%",
                'Focus Period 1 Avg Drawdown': 
                    f"{metrics['focus_1_avg_drawdown']*100:.2f}%",
                'Focus Period 1 Normalized Score': 
                    f"{metrics['focus_1_normalized_score']:.3f}",
                'Focus Period 1 Start Date': 
                    self.metric_blending_config.get('focus_period_1_start', ''),
                'Focus Period 1 End Date': 
                    self.metric_blending_config.get('focus_period_1_end', ''),
            }
            
            focus_2_metrics = {
                'Focus Period 2 Final Value': 
                    f"${metrics['focus_2_final_value']:,.0f}",
                'Focus Period 2 Annual Return': 
                    f"{metrics['focus_2_annual_return']*100:.2f}%",
                'Focus Period 2 Sharpe Ratio': 
                    f"{metrics['focus_2_sharpe_ratio']:.5f}",
                'Focus Period 2 Sortino Ratio': 
                    f"{metrics['focus_2_sortino_ratio']:.3f}",
                'Focus Period 2 Max Drawdown': 
                    f"{metrics['focus_2_max_drawdown']*100:.2f}%",
                'Focus Period 2 Avg Drawdown': 
                    f"{metrics['focus_2_avg_drawdown']*100:.2f}%",
                'Focus Period 2 Normalized Score': 
                    f"{metrics['focus_2_normalized_score']:.3f}",
                'Focus Period 2 Start Date': 
                    self.metric_blending_config.get('focus_period_2_start', ''),
                'Focus Period 2 End Date': 
                    self.metric_blending_config.get('focus_period_2_end', ''),
            }
            
            # Calculate focus period tracking parameters
            fp1_weight = self.metric_blending_config.get(
                'focus_period_1_weight', 0.5)
            fp2_weight = self.metric_blending_config.get(
                'focus_period_2_weight', 0.5)
            
            # Calculate duration in years for each focus period
            fp1_start = pd.to_datetime(
                self.metric_blending_config.get('focus_period_1_start')
            ).date()
            fp1_end = pd.to_datetime(
                self.metric_blending_config.get('focus_period_1_end')
            ).date()
            fp1_duration_years = (fp1_end - fp1_start).days / 365.25
            
            fp2_start = pd.to_datetime(
                self.metric_blending_config.get('focus_period_2_start')
            ).date()
            fp2_end = pd.to_datetime(
                self.metric_blending_config.get('focus_period_2_end')
            ).date()
            fp2_duration_years = (fp2_end - fp2_start).days / 365.25
            
            # Calculate overlap percentage between periods
            overlap_start = max(fp1_start, fp2_start)
            overlap_end = min(fp1_end, fp2_end)
            if overlap_start < overlap_end:
                overlap_days = (overlap_end - overlap_start).days
                total_days = (fp1_end - fp1_start).days + (
                    fp2_end - fp2_start).days
                overlap_pct = (overlap_days / total_days) * 100 if total_days > 0 else 0
            else:
                overlap_pct = 0.0
            
            focus_tracking_params = {
                'Focus Period 1 Weight': fp1_weight,
                'Focus Period 2 Weight': fp2_weight,
                'Focus Period 1 Duration Years': f"{fp1_duration_years:.2f}",
                'Focus Period 2 Duration Years': f"{fp2_duration_years:.2f}",
                'Focus Period Year Min': self.metric_blending_config.get(
                    'fp_year_min', ''),
                'Focus Period Year Max': self.metric_blending_config.get(
                    'fp_year_max', ''),
                'Focus Periods Overlap Percentage': f"{overlap_pct:.1f}%",
            }
            
        elif self.focus_period_enabled and 'focus_annual_return' in metrics:
            # Legacy single focus period support
            focus_1_metrics = {
                'Focus Period 1 Final Value': 
                    f"${metrics['focus_final_value']:,.0f}",
                'Focus Period 1 Annual Return': 
                    f"{metrics['focus_annual_return']*100:.2f}%",
                'Focus Period 1 Sharpe Ratio': 
                    f"{metrics['focus_sharpe_ratio']:.5f}",
                'Focus Period 1 Sortino Ratio': 
                    f"{metrics['focus_sortino_ratio']:.3f}",
                'Focus Period 1 Max Drawdown': 
                    f"{metrics['focus_max_drawdown']*100:.2f}%",
                'Focus Period 1 Avg Drawdown': 
                    f"{metrics['focus_avg_drawdown']*100:.2f}%",
                'Focus Period 1 Normalized Score': 
                    f"{metrics['focus_normalized_score']:.3f}",
                'Focus Period 1 Start Date': 
                    self.metric_blending_config.get('focus_period_start', ''),
                'Focus Period 1 End Date': 
                    self.metric_blending_config.get('focus_period_end', ''),
            }
            
            # Empty focus period 2 metrics for legacy mode
            focus_2_metrics = {
                'Focus Period 2 Final Value': '',
                'Focus Period 2 Annual Return': '',
                'Focus Period 2 Sharpe Ratio': '',
                'Focus Period 2 Sortino Ratio': '',
                'Focus Period 2 Max Drawdown': '',
                'Focus Period 2 Avg Drawdown': '',
                'Focus Period 2 Normalized Score': '',
                'Focus Period 2 Start Date': '',
                'Focus Period 2 End Date': '',
            }
            
            # Minimal tracking params for legacy mode
            focus_tracking_params = {
                'Focus Period 1 Weight': 1.0,
                'Focus Period 2 Weight': 0.0,
                'Focus Period 1 Duration Years': '',
                'Focus Period 2 Duration Years': '',
                'Focus Period Year Min': '',
                'Focus Period Year Max': '',
                'Focus Periods Overlap Percentage': '',
            }
        else:
            # No focus periods - empty columns
            focus_1_metrics = {
                'Focus Period 1 Final Value': '',
                'Focus Period 1 Annual Return': '',
                'Focus Period 1 Sharpe Ratio': '',
                'Focus Period 1 Sortino Ratio': '',
                'Focus Period 1 Max Drawdown': '',
                'Focus Period 1 Avg Drawdown': '',
                'Focus Period 1 Normalized Score': '',
                'Focus Period 1 Start Date': '',
                'Focus Period 1 End Date': '',
            }
            
            focus_2_metrics = {
                'Focus Period 2 Final Value': '',
                'Focus Period 2 Annual Return': '',
                'Focus Period 2 Sharpe Ratio': '',
                'Focus Period 2 Sortino Ratio': '',
                'Focus Period 2 Max Drawdown': '',
                'Focus Period 2 Avg Drawdown': '',
                'Focus Period 2 Normalized Score': '',
                'Focus Period 2 Start Date': '',
                'Focus Period 2 End Date': '',
            }
            
            focus_tracking_params = {
                'Focus Period 1 Weight': '',
                'Focus Period 2 Weight': '',
                'Focus Period 1 Duration Years': '',
                'Focus Period 2 Duration Years': '',
                'Focus Period Year Min': '',
                'Focus Period Year Max': '',
                'Focus Periods Overlap Percentage': '',
            }
        
        #############################################################################
        # Build complete row data dictionary with proper column ordering
        #############################################################################
        current_time = datetime.now()
        row_data = {
            # Date/Time columns
            'Date': current_time.strftime('%Y-%m-%d'),
            'Time': current_time.strftime('%H:%M:%S'),
            
            # Full period metrics (9 columns)
            **full_period_metrics,
            
            # Focus period 1 metrics (9 columns)
            **focus_1_metrics,
            
            # Focus period 2 metrics (9 columns)
            **focus_2_metrics,
            
            # Blended score
            'Blended Score': f"{metrics.get('blended_score', metrics['normalized_score']):.3f}",
            
            # Model effectiveness metrics (3 columns)
            'Sharpe Outperformance Percentage': f"{model_effectiveness.get('sharpe_outperformance_pct', 0.0):.1f}%",
            'Sortino Outperformance Percentage': f"{model_effectiveness.get('sortino_outperformance_pct', 0.0):.1f}%",
            'Average Rank': f"{model_effectiveness.get('average_rank', 6.0):.2f}",
            
            # Lookback periods (3 columns)
            'Lookback Period 1': sorted_lookbacks[0],
            'Lookback Period 2': sorted_lookbacks[1],
            'Lookback Period 3': sorted_lookbacks[2],
            
            # Performance metric weights (5 columns)
            'Sharpe Ratio Weight': sharpe_weight,
            'Sortino Ratio Weight': sortino_weight,
            'Max Drawdown Weight': max_dd_weight,
            'Avg Drawdown Weight': avg_dd_weight,
            'Annualized Return Weight': annual_ret_weight,
            
            # Monte Carlo normalization parameters (10 columns)
            'Central Annual Return': central_values['annual_return'],
            'Central Sharpe Ratio': central_values['sharpe_ratio'],
            'Central Sortino Ratio': central_values['sortino_ratio'],
            'Central Max Drawdown': central_values['max_drawdown'],
            'Central Avg Drawdown': central_values['avg_drawdown'],
            'Std Annual Return': std_values['annual_return'],
            'Std Sharpe Ratio': std_values['sharpe_ratio'],
            'Std Sortino Ratio': std_values['sortino_ratio'],
            'Std Max Drawdown': std_values['max_drawdown'],
            'Std Avg Drawdown': std_values['avg_drawdown'],
            
            # Focus period tracking parameters (7 columns)
            **focus_tracking_params,
            
            # Metric blending enabled flag (1 column)
            'Metric Blending Enabled': self.metric_blending_config.get(
                'enabled', False),
        }
        
        #############################################################################
        # Write to CSV file
        #############################################################################
        # Add PNG filename column if provided
        if png_filename:
            row_data['PNG Filename'] = png_filename
        else:
            row_data['PNG Filename'] = ''

        try:
            with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=row_data.keys())

                # Write header if file is new
                if not file_exists:
                    writer.writeheader()

                # Write data row
                writer.writerow(row_data)

            print(f"Best parameters logged to: {csv_filename}")

        except Exception as e:
            logger.error(f"Failed to write to CSV file {csv_path}: {str(e)}")
            print(f"Warning: Could not log to CSV file: {str(e)}")

    def debug_normalized_score(self, metrics: Dict[str, float]) -> None:
        """Debug function to show individual normalized values for each statistic.
        
        Args:
            metrics: Dictionary containing performance metrics
        """
        # Define normalization parameters (same as in compute_performance_metrics)
        # central_values = {
        #     'annual_return': 52.5,
        #     'sharpe_ratio': 0.0,
        #     'sortino_ratio': 0.0,
        #     'max_drawdown': -60.3,
        #     'avg_drawdown': -10.7
        # }
        
        # std_values = {
        #     'annual_return': 3.48,
        #     'sharpe_ratio': 1.0,   # Fixed: Changed from 0.005 to 0.05 (matches compute_performance_metrics)
        #     'sortino_ratio': 1.0,
        #     'max_drawdown': 6.98,
        #     'avg_drawdown': 1.42
        # }
        # central_values = {
        #     'annual_return': .4537,
        #     'sharpe_ratio': 1.44,
        #     'sortino_ratio': 1.42,
        #     'max_drawdown': -0.58,
        #     'avg_drawdown': -0.115
        # }
        
        # std_values = {
        #     'annual_return': .074,
        #     'sharpe_ratio': 0.17,   # Fixed: Changed from 0.005 to 0.05 (matches compute_performance_metrics)
        #     'sortino_ratio': 0.21,
        #     'max_drawdown': .069,
        #     'avg_drawdown': .019
        # }
        # central_values = {
        #     'annual_return': .4145,
        #     'sharpe_ratio': 1.365,
        #     'sortino_ratio': 1.31,
        #     'max_drawdown': -0.556,
        #     'avg_drawdown': -0.120
        # }
        
        # std_values = {
        #     'annual_return': .044,
        #     'sharpe_ratio': 0.135,   # Fixed: Changed from 0.005 to 0.05 (matches compute_performance_metrics)
        #     'sortino_ratio': 0.146,
        #     'max_drawdown': .052,
        #     'avg_drawdown': .016
        # }
        central_values = self.CENTRAL_VALUES
        std_values = self.STD_VALUES
        print(f"\n" + "="*70)
        print(f"NORMALIZED SCORE BREAKDOWN")
        print(f"="*70)
        print(f"Raw Values and Individual Normalized Scores:")
        print(f"")
        
        total_normalized = 0.0
        count = 0
        
        for metric_name in ['annual_return', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'avg_drawdown']:
            raw_value = metrics[metric_name]
            central = central_values[metric_name]
            std = std_values[metric_name]
            normalized = (raw_value - central) / std
            total_normalized += normalized
            count += 1
            
            print(f"{metric_name.replace('_', ' ').title():<15}: "
                  f"Raw = {raw_value:>8.2f}, "
                  f"Central = {central:>8.2f}, "
                  f"Std = {std:>6.2f}, "
                  f"Normalized = {normalized:>8.3f}")
        
        calculated_avg = total_normalized / count
        
        print(f"")
        print(f"Sum of normalized values: {total_normalized:.3f}")
        print(f"Count of metrics: {count}")
        print(f"Calculated average: {calculated_avg:.3f}")
        print(f"Reported normalized score: {metrics['normalized_score']:.3f}")
        print(f"Difference: {abs(calculated_avg - metrics['normalized_score']):.6f}")
        print(f"="*70)

    def _print_best_parameters(self, metrics: Dict[str, float], png_filename: Optional[str] = None) -> None:
        """Print information about the best performing parameters found so far."""
        lookbacks = self.best_params.get('lookbacks', [])
        print(f"\n" + "="*90)
        print(f"NEW BEST PERFORMANCE FOUND!")
        print(f"="*90)
        
        # Check if TWO focus period metrics are available
        has_two_focus_periods = (self.has_two_focus_periods and 
                                 'focus_1_annual_return' in metrics and 
                                 'focus_2_annual_return' in metrics)
        
        # Check if ONE focus period metrics are available (legacy)
        has_one_focus_period = (self.focus_period_enabled and 
                               'focus_annual_return' in metrics)
        
        if has_two_focus_periods:
            # Print THREE columns: Full Period | Focus Period 1 | Focus Period 2
            print(f"{'Metric':<20} {'Full Period':>20} {'Focus Period 1':>20} {'Focus Period 2':>20}")
            print(f"{'-'*20} {'-'*20} {'-'*20} {'-'*20}")
            print(f"{'Final Value:':<20} ${metrics['final_value']:>19,.0f} ${metrics['focus_1_final_value']:>19,.0f} ${metrics['focus_2_final_value']:>19,.0f}")
            print(f"{'Annual Return:':<20} {metrics['annual_return']*100:>19.2f}% {metrics['focus_1_annual_return']*100:>19.2f}% {metrics['focus_2_annual_return']*100:>19.2f}%")
            print(f"{'Sharpe Ratio:':<20} {metrics['sharpe_ratio']:>20.5f} {metrics['focus_1_sharpe_ratio']:>20.5f} {metrics['focus_2_sharpe_ratio']:>20.5f}")
            print(f"{'Sortino Ratio:':<20} {metrics['sortino_ratio']:>20.3f} {metrics['focus_1_sortino_ratio']:>20.3f} {metrics['focus_2_sortino_ratio']:>20.3f}")
            print(f"{'Max Drawdown:':<20} {metrics['max_drawdown']*100:>19.1f}% {metrics['focus_1_max_drawdown']*100:>19.1f}% {metrics['focus_2_max_drawdown']*100:>19.1f}%")
            print(f"{'Avg Drawdown:':<20} {metrics['avg_drawdown']*100:>19.1f}% {metrics['focus_1_avg_drawdown']*100:>19.1f}% {metrics['focus_2_avg_drawdown']*100:>19.1f}%")
            print(f"{'Normalized Score:':<20} {metrics['normalized_score']:>20.3f} {metrics['focus_1_normalized_score']:>20.3f} {metrics['focus_2_normalized_score']:>20.3f}")
            print(f"{'Blended Score:':<20} {'(not used)':>20} {metrics['blended_score']:>41.3f}")
            
            # Print period date ranges
            full_start = str(self.dates[0]) if self.dates else 'N/A'
            full_end = str(self.dates[-1]) if self.dates else 'N/A'
            fp1_start = self.metric_blending_config.get('focus_period_1_start', 'N/A')
            fp1_end = self.metric_blending_config.get('focus_period_1_end', 'N/A')
            fp2_start = self.metric_blending_config.get('focus_period_2_start', 'N/A')
            fp2_end = self.metric_blending_config.get('focus_period_2_end', 'N/A')
            
            print(f"\n{'Period':<20} {'Start Date':>20} {'End Date':>20}")
            print(f"{'-'*20} {'-'*20} {'-'*20}")
            print(f"{'Full Period:':<20} {full_start:>20} {full_end:>20}")
            print(f"{'Focus Period 1:':<20} {fp1_start:>20} {fp1_end:>20}")
            print(f"{'Focus Period 2:':<20} {fp2_start:>20} {fp2_end:>20}")
            
        elif has_one_focus_period:
            # Legacy single focus period display (kept for backward compatibility)
            print(f"{'Metric':<20} {'Full Period':>20} {'Focus Period':>20}")
            print(f"{'-'*20} {'-'*20} {'-'*20}")
            print(f"{'Final Value:':<20} ${metrics['final_value']:>19,.0f} ${metrics['focus_final_value']:>19,.0f}")
            print(f"{'Annual Return:':<20} {metrics['annual_return']*100:>19.2f}% {metrics['focus_annual_return']*100:>19.2f}%")
            print(f"{'Sharpe Ratio:':<20} {metrics['sharpe_ratio']:>20.5f} {metrics['focus_sharpe_ratio']:>20.5f}")
            print(f"{'Sortino Ratio:':<20} {metrics['sortino_ratio']:>20.3f} {metrics['focus_sortino_ratio']:>20.3f}")
            print(f"{'Max Drawdown:':<20} {metrics['max_drawdown']*100:>19.1f}% {metrics['focus_max_drawdown']*100:>19.1f}%")
            print(f"{'Avg Drawdown:':<20} {metrics['avg_drawdown']*100:>19.1f}% {metrics['focus_avg_drawdown']*100:>19.1f}%")
            print(f"{'Normalized Score:':<20} {metrics['normalized_score']:>20.3f} {metrics['focus_normalized_score']:>20.3f}")
            print(f"{'Blended Score:':<20} {metrics['blended_score']:>20.3f}")
        else:
            # No focus periods available - show only full period
            print(f"{'Metric':<20} {'Full Period':>20}")
            print(f"{'-'*20} {'-'*20}")
            print(f"{'Final Value:':<20} ${metrics['final_value']:>19,.0f}")
            print(f"{'Annual Return:':<20} {metrics['annual_return']*100:>19.2f}%")
            print(f"{'Sharpe Ratio:':<20} {metrics['sharpe_ratio']:>20.5f}")
            print(f"{'Sortino Ratio:':<20} {metrics['sortino_ratio']:>20.3f}")
            print(f"{'Max Drawdown:':<20} {metrics['max_drawdown']*100:>19.1f}%")
            print(f"{'Avg Drawdown:':<20} {metrics['avg_drawdown']*100:>19.1f}%")
            print(f"Blended Score: {metrics['blended_score']:.3f}")
        
        print(f"\nLookback periods: {sorted(lookbacks)} days")
        print(f"="*90)
        
        # Only show debug breakdown if verbose mode is enabled
        if self.verbose:
            self.debug_normalized_score(metrics)
        
        # Log to CSV file (include PNG filename if provided)
        self._log_best_parameters_to_csv(metrics, png_filename=png_filename)

    def plot_performance(self, portfolio_values: np.ndarray, metrics: Dict[str, float]) -> None:
        """Create a plot of the current best performance."""
        # Use the unified plotting function
        self.create_monte_carlo_plot(portfolio_values, metrics)

    def get_permutation_statistics(self) -> Dict[str, Any]:
        """Get statistics about the permutation-invariant exploration/exploitation."""
        if not self.canonical_performance_scores:
            return {
                'total_canonical_combinations': 0,
                'total_visits': 0,
                'average_visits_per_combination': 0.0,
                'efficiency_improvement_factor': 1.0,
                'best_performing_combination': None
            }
        
        total_combinations = len(self.canonical_performance_scores)
        total_visits = sum(self.canonical_visit_counts)
        avg_visits = total_visits / total_combinations if total_combinations > 0 else 0.0
        
        # Calculate theoretical efficiency improvement
        # This is the factor by which we've reduced the search space through permutation invariance
        total_possible_permutations = math.factorial(self.n_lookbacks) if self.n_lookbacks <= 10 else float('inf')
        efficiency_factor = total_possible_permutations / max(1, total_combinations)
        
        # Find best performing combination
        best_combination = None
        if self.canonical_performance_scores:
            best_idx = np.argmax(self.canonical_performance_scores)
            best_lookbacks = list(self.combination_indices.keys())[best_idx]
            best_combination = {
                'lookbacks': list(best_lookbacks),
                'score': self.canonical_performance_scores[best_idx],
                'visits': self.canonical_visit_counts[best_idx]
            }
        print(f"best_combination: {best_combination}")
        return {
            'total_canonical_combinations': total_combinations,
            'total_visits': total_visits,
            'average_visits_per_combination': avg_visits,
            'efficiency_improvement_factor': efficiency_factor,
            'best_performing_combination': best_combination
        }

    def create_monthly_performance_analysis(self, save_path: Optional[str] = None) -> None:
        """Create monthly performance analysis plot with normalized scores.
        
        This function analyzes performance metrics on the first day of every month
        and creates a dual subplot visualization showing:
        - Upper subplot: Portfolio values over time for all models
        - Lower subplot: Monthly normalized scores for non-cash models
        
        Args:
            save_path: Optional path to save the plot (defaults to 'monthly_performance_analysis.png')
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import matplotlib.ticker as mticker
        import pandas as pd
        from calendar import monthrange
        from datetime import date as date_type
        
        if save_path is None:
            save_path = 'monthly_performance_analysis.png'
            
        print("Generating monthly performance analysis...")
        
        #############################################################################
        # Find first day of each month in the date range
        #############################################################################
        monthly_dates = []
        monthly_indices = []
        
        # Convert dates to proper format for analysis
        date_objects = []
        for d in self.dates:
            if isinstance(d, date_type):
                date_objects.append(d)
            else:
                date_objects.append(pd.to_datetime(d).date())
        
        # Find first trading day of each month
        current_month = None
        for i, date_obj in enumerate(date_objects):
            if current_month != (date_obj.year, date_obj.month):
                current_month = (date_obj.year, date_obj.month)
                monthly_dates.append(date_obj)
                monthly_indices.append(i)
        
        print(f"Found {len(monthly_dates)} monthly analysis points")
        
        #############################################################################
        # Compute performance metrics for each model on monthly dates
        #############################################################################
        models = sorted(list(self.portfolio_histories.keys()))
        non_cash_models = [m for m in models if m != "cash"]
        
        # Initialize arrays for normalized scores
        monthly_scores = {model: [] for model in non_cash_models}
        monthly_date_objects = []
        
        # Use 60-day lookback for monthly analysis
        lookback_period = 60
        
        for date_obj, date_idx in zip(monthly_dates, monthly_indices):
            # Skip if insufficient history
            if date_idx < lookback_period:
                continue
            
            monthly_date_objects.append(date_obj)
            
            # Calculate metrics for each non-cash model
            for model in non_cash_models:
                start_idx = max(0, date_idx - lookback_period)
                start_date = pd.Timestamp(self.dates[start_idx])
                end_date = pd.Timestamp(self.dates[date_idx - 1])
                
                # Get portfolio values for the lookback period
                portfolio_vals = self.portfolio_histories[model][start_date:end_date].values
                
                if len(portfolio_vals) == 0:
                    # Fallback to constant values if no data
                    portfolio_vals = np.ones(lookback_period) * 10000.0
                
                # Compute performance metrics
                metrics = self.compute_performance_metrics(portfolio_vals)
                monthly_scores[model].append(metrics['normalized_score'])
        
        print(f"Computed metrics for {len(monthly_date_objects)} monthly points")
        
        #############################################################################
        # Create figure with subplot layout matching create_monte_carlo_plot
        #############################################################################
        fig = plt.figure(figsize=(12, 8))
        gs = plt.GridSpec(6, 1)
        
        # Upper subplot for portfolio values (75% of space)
        ax1 = fig.add_subplot(gs[0:4, 0])
        
        # Lower subplot for normalized scores (25% of space)
        ax2 = fig.add_subplot(gs[4:6, 0])
        
        #############################################################################
        # Plot portfolio values (upper subplot)
        #############################################################################
        colors = ['b', 'r', 'g', 'c', 'm']
        model_to_color = {}
        date_index = pd.DatetimeIndex(self.dates)
        
        # FIXED: Use sorted items to ensure consistent ordering with rest of codebase
        sorted_portfolio_items = sorted(self.portfolio_histories.items(), key=lambda x: x[0])
        
        # Plot historical values for each model
        for (model, values), color in zip(sorted_portfolio_items, colors):
            if model != "cash":
                model_to_color[model] = color
                start_value = values.iloc[0]
                normalized_values = values * (10000.0 / start_value)
                ax1.plot(date_index, normalized_values,
                        color=color, alpha=0.7, label=f"{model}",
                        linewidth=1.5)
        
        # Configure upper subplot styling
        ax1.set_yscale('log')
        ax1.grid(True, which='major', alpha=0.4, linewidth=0.8)
        ax1.grid(True, which='minor', alpha=0.2, linewidth=0.5)
        ax1.minorticks_on()
        
        # Configure date formatting - 5 year major, 1 year minor
        ax1.xaxis.set_major_locator(mdates.YearLocator(5))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
        
        ax1.set_title('Monthly Performance Analysis: Portfolio Values and Normalized Scores', 
                     pad=20, fontsize=14)
        ax1.set_xlabel('')  # Remove x-label from upper plot
        ax1.set_ylabel('Portfolio Value ($)', fontsize=10)
        
        # Format y-axis
        ax1.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'${int(x):,}')
        )
        ax1.tick_params(axis='y', which='major', labelsize=8)
        ax1.tick_params(axis='y', which='minor', labelsize=6)
        ax1.tick_params(axis='x', which='major', labelsize=10, rotation=0)
        
        # Add legend
        ax1.legend(loc='lower right', fontsize=8)
        
        #############################################################################
        # Plot normalized scores (lower subplot)
        #############################################################################
        monthly_pd_dates = pd.DatetimeIndex(monthly_date_objects)
        
        # Plot normalized scores for each non-cash model
        for model in non_cash_models:
            if model in model_to_color and len(monthly_scores[model]) > 0:
                ax2.plot(monthly_pd_dates, monthly_scores[model],
                        color=model_to_color[model], marker='o', markersize=3,
                        linewidth=1.5, alpha=0.8, label=f"{model}")
        
        # Configure lower subplot styling to match upper subplot
        ax2.grid(True, which='major', alpha=0.4, linewidth=0.8)
        ax2.grid(True, which='minor', alpha=0.2, linewidth=0.5)
        ax2.minorticks_on()
        
        # Configure date formatting to match upper subplot
        ax2.xaxis.set_major_locator(mdates.YearLocator(5))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax2.xaxis.set_minor_locator(mdates.YearLocator(1))
        
        # Set labels
        ax2.set_xlabel('Date', fontsize=10)
        ax2.set_ylabel('Normalized Score', fontsize=10)
        
        # Configure tick labels
        ax2.tick_params(axis='y', which='major', labelsize=8)
        ax2.tick_params(axis='y', which='minor', labelsize=6)
        ax2.tick_params(axis='x', which='major', labelsize=10, rotation=0)
        
        # Add horizontal line at zero for reference
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
        
        # Add legend
        ax2.legend(loc='upper right', fontsize=7)
        
        #############################################################################
        # Add summary text overlay
        #############################################################################
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        summary_text = (f"{current_time}\n"
                       f"Monthly Performance Analysis\n"
                       f"Lookback period: {lookback_period} days\n"
                       f"Analysis points: {len(monthly_date_objects)} months\n"
                       f"Models analyzed: {len(non_cash_models)} non-cash models")
        
        # Position text in upper left of upper subplot
        ax1.text(0.02, 0.95, summary_text,
                transform=ax1.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                fontsize=7, verticalalignment='top',
                fontfamily='monospace')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Monthly performance analysis plot saved to: {save_path}")
        
        # Return summary statistics
        return {
            'monthly_dates': monthly_date_objects,
            'monthly_scores': monthly_scores,
            'analysis_points': len(monthly_date_objects),
            'models_analyzed': non_cash_models
        }
