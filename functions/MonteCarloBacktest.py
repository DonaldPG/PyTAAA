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
    gain_loss: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    avg_drawdown: float
    daily_return: float
    volatility: float

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
def compute_metrics_fast(portfolio_values: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
    """Optimized computation of performance metrics."""
    if len(portfolio_values) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Pre-allocate arrays
    n = len(portfolio_values)
    daily_returns = np.zeros(n - 1)
    
    # Calculate daily returns
    for i in range(n - 1):
        if portfolio_values[i] > 0:
            daily_returns[i] = (portfolio_values[i + 1] / portfolio_values[i]) - 1
    
    # Basic metrics
    gain_loss = ((portfolio_values[-1] / portfolio_values[0]) - 1) * 100 if portfolio_values[0] > 0 else 0.0
    avg_daily_return = np.mean(daily_returns) * 100
    volatility = np.std(daily_returns) * np.sqrt(252) * 100
    
    # Sharpe ratio
    sharpe = (avg_daily_return * np.sqrt(252) / volatility) if volatility > 0 else 0.0
    
    # Sortino ratio
    downside_returns = np.zeros(0)  # Initialize empty array
    for ret in daily_returns:
        if ret < 0:
            downside_returns = np.append(downside_returns, ret)
            
    downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 1
    sortino = (avg_daily_return * np.sqrt(252) / downside_vol) if downside_vol > 0 else 0.0
    
    # Drawdown calculations
    cumulative_returns = np.zeros(len(daily_returns) + 1)
    cumulative_returns[0] = 1.0
    for i in range(len(daily_returns)):
        cumulative_returns[i + 1] = cumulative_returns[i] * (1 + daily_returns[i])
    
    rolling_max = compute_rolling_max(cumulative_returns)
    drawdowns = np.zeros_like(cumulative_returns)
    
    for i in range(len(cumulative_returns)):
        if rolling_max[i] > 0:
            drawdowns[i] = ((cumulative_returns[i] - rolling_max[i]) / rolling_max[i]) * 100
    
    max_dd = np.min(drawdowns)
    avg_dd = np.mean(drawdowns)
    
    return gain_loss, sharpe, sortino, max_dd, avg_dd, avg_daily_return, volatility

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
    metric_arrays = np.zeros((7, n_models))
    
    for i, metrics in enumerate(metrics_list):
        metric_arrays[:, i] = [
            metrics.gain_loss,
            metrics.sharpe_ratio,
            metrics.sortino_ratio,
            -metrics.max_drawdown,  # Negative since higher is better
            -metrics.avg_drawdown,  # Negative since higher is better
            metrics.daily_return,
            -metrics.volatility  # Negative since lower is better
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
        search_mode: str = "explore-exploit"
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
        """
        self.iterations = iterations
        self.min_iterations_for_exploit = min_iterations_for_exploit
        self.trading_frequency = trading_frequency
        self.model_paths = model_paths
        self.min_lookback = min_lookback
        self.max_lookback = max_lookback
        self.n_lookbacks = n_lookbacks
        self.search_mode = search_mode
        
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
                metrics = compute_daily_metrics(portfolio_values)
                current_performance = metrics.sharpe_ratio
                
                # Update permutation-invariant tracking arrays
                self._update_tracking_arrays(current_lookbacks, current_performance)
                
                # Track best performing portfolio
                if current_performance > best_performance:
                    best_performance = current_performance
                    self.best_portfolio_value = portfolio_values.copy()
                    self.best_params = current_params.copy()
                    self.best_model_selections = current_selections.copy()

                    # Display performance metrics
                    metrics_dict = self.compute_performance_metrics(portfolio_values)
                    self._print_best_parameters(metrics_dict)

                    # Plot performance of the best portfolio
                    self.plot_performance(portfolio_values, metrics_dict)

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
        print(f"Best Sharpe ratio: {best_performance:.2f}")
        
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
        
        # Initialize rank storage
        models = list(self.portfolio_histories.keys())
        all_ranks = np.zeros((len(models), 7 * len(lookbacks))) # 7 metrics × n_lookbacks
        
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
                metric_weights['gain_loss_weight'],
                metric_weights['sharpe_ratio_weight'],
                metric_weights['sortino_ratio_weight'],
                metric_weights['max_drawdown_weight'],
                metric_weights['avg_drawdown_weight'],
                metric_weights['daily_return_weight'],
                metric_weights['volatility_weight']
            ])
            
            start_col = i * 7
            all_ranks[:, start_col:start_col + 7] = period_ranks.T * weights[:, np.newaxis].T
        
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
            Dictionary containing performance metrics
        """
        metrics = compute_daily_metrics(portfolio_values)
        
        return {
            'final_value': portfolio_values[-1],
            'annual_return': ((portfolio_values[-1] / portfolio_values[0]) ** (252 / len(portfolio_values)) - 1) * 100,
            'sharpe_ratio': metrics.sharpe_ratio,
            'sortino_ratio': metrics.sortino_ratio,
            'max_drawdown': metrics.max_drawdown,
            'avg_drawdown': metrics.avg_drawdown
        }
    
    def create_monte_carlo_plot(self, portfolio_values: np.ndarray, 
                               metrics: dict, save_path: Optional[str] = None) -> None:
        """Create Monte Carlo optimization performance plot with model rankings.
        
        This unified function creates plots showing:
        - Portfolio performance comparison (upper subplot)
        - Model selection timeline (lower subplot)
        - Model rankings for current date and first weekday of month
        
        Args:
            portfolio_values: Array of portfolio values over time
            metrics: Dictionary of performance metrics
            save_path: Optional path to save the plot (defaults to 'monte_carlo_best_performance.png')
        """
        from calendar import monthrange
        
        if save_path is None:
            save_path = 'monte_carlo_best_performance.png'
            
        #############################################################################
        # Create figure with subplot layout
        #############################################################################
        fig = plt.figure(figsize=(12, 7))
        gs = plt.GridSpec(6, 1)
        
        # Main portfolio performance plot (83% of space)
        ax1 = fig.add_subplot(gs[0:5, 0])
        
        # Plot historical values for each model
        colors = ['b', 'r', 'g', 'c', 'm', 'k']
        model_to_color = {}
        date_index = pd.DatetimeIndex(self.dates)
        
        # Normalize all series to start at 10000 and plot
        for (model, values), color in zip(self.portfolio_histories.items(), colors):
            if model != "cash":
                model_to_color[model] = color
                start_value = values.iloc[0]
                normalized_values = values * (10000.0 / start_value)
                ax1.plot(date_index, normalized_values,
                        color=color, alpha=0.5, label=f"{model}",
                        linewidth=1)
        
        model_to_color["cash"] = 'k'
        
        # Plot Monte Carlo best portfolio
        ax1.plot(date_index, portfolio_values, 
                'k-', linewidth=2, label='Monte Carlo Best')
        
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
        # Add current date/time and lookback parameters
        #############################################################################
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        lookbacks = self.best_params.get('lookbacks', [])
        _lb_list = sorted(lookbacks) if isinstance(lookbacks, list) else lookbacks
        lookback_text = f"Best parameters: lookbacks={_lb_list} days"
        text_str = f"{current_time}\n{lookback_text}"
        
        #############################################################################
        # Add model ranking information for current date and first weekday of month
        #############################################################################
        try:
            # Get current date (August 2, 2025)
            current_date = date(2025, 8, 2)
            current_date_str = current_date.strftime("%Y-%m-%d")
            
            # Find first weekday of current month (August 2025)
            year, month = 2025, 8
            first_weekday = None
            for day in range(1, monthrange(year, month)[1] + 1):
                test_date = date(year, month, day)
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
                            models = list(self.portfolio_histories.keys())
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
                            
                            # Rank models by Sharpe ratio (primary metric)
                            sharpe_scores = [m.sharpe_ratio for m in metrics_list]
                            model_ranking = sorted(zip(models, sharpe_scores), key=lambda x: x[1], reverse=True)
                            
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
                            models = list(self.portfolio_histories.keys())
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
                            
                            sharpe_scores = [m.sharpe_ratio for m in metrics_list]
                            model_ranking = sorted(zip(models, sharpe_scores), key=lambda x: x[1], reverse=True)
                            
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
                fontsize=8, verticalalignment='top',  # Reduced from 9 to 8
                fontfamily='monospace')  # Use monospace font for table alignment
        
        # Add legend to main plot
        ax1.legend(loc='lower right', fontsize=8)
        
        #############################################################################
        # Create model selection subplot (17% of space)
        #############################################################################
        ax2 = fig.add_subplot(gs[5, 0])
        
        # Create numeric mapping for all models
        unique_models = sorted(list(self.portfolio_histories.keys()))
        model_to_num = {model: i for i, model in enumerate(unique_models)}
        
        # Get model selections over time
        current_model = None
        model_selections = []
        for date_val in self.dates:
            if self._should_trade(date_val):
                current_model = self.best_model_selections.get(
                    pd.Timestamp(date_val).strftime('%Y-%m-%d'), current_model
                )
            model_selections.append(current_model if current_model is not None else "cash")
        
        # Convert to numeric values and plot
        numeric_selections = [model_to_num.get(model, -1) for model in model_selections]
        
        for model in unique_models:
            mask = [x == model_to_num[model] for x in numeric_selections]
            if any(mask):
                ax2.scatter(date_index[mask], [model_to_num[model]] * sum(mask),
                          color=model_to_color[model], alpha=0.7, s=20,
                          label=f"{model} periods")
                ax2.plot(date_index[mask], [model_to_num[model]] * sum(mask),
                        color=model_to_color[model], alpha=0.3, linewidth=1)
        
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

    def plot_performance(self, portfolio_values: np.ndarray, metrics: dict) -> None:
        """Plot portfolio performance metrics and save visualization.
        
        Args:
            portfolio_values: Array of portfolio values over time
            metrics: Dictionary of performance metrics
        """
        # Use the unified plotting function
        self.create_monte_carlo_plot(portfolio_values, metrics)

    def _write_best_performance_to_csv(self, metrics: Dict[str, float]) -> None:
        """Write best performance metrics to CSV file.
        
        Args:
            metrics: Dictionary containing performance metrics
        """
        import csv
        from datetime import datetime
        import os

        # Prepare data row
        current_time = datetime.now()
        lookbacks = self.best_params.get('lookbacks', [])
        sorted_lookbacks = sorted(lookbacks) if isinstance(lookbacks, list) else lookbacks

        # Define headers if file doesn't exist
        headers = [
            'date', 'time', 'final_value', 'annual_return', 'sharpe_ratio',
            'sortino_ratio', 'max_drawdown', 'avg_drawdown'
        ] + [f'lookback_{i+1}' for i in range(len(sorted_lookbacks))]

        # Prepare row data
        row_data = {
            'date': current_time.strftime('%Y-%m-%d'),
            'time': current_time.strftime('%H:%M:%S'),
            'final_value': metrics['final_value'],
            'annual_return': metrics['annual_return'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'sortino_ratio': metrics['sortino_ratio'],
            'max_drawdown': metrics['max_drawdown'],
            'avg_drawdown': metrics['avg_drawdown']
        }
        
        # Add lookback values
        for i, lb in enumerate(sorted_lookbacks):
            row_data[f'lookback_{i+1}'] = lb

        # Write to CSV
        file_exists = os.path.exists(self.csv_filename)
        with open(self.csv_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)

    def _print_best_parameters(self, metrics: Dict[str, float]) -> None:
        """Print best parameters with sorted lookback list and write to CSV.
        
        Args:
            metrics: Dictionary containing performance metrics
        """
        print("\n\n\nNew best portfolio return found. Performance Metrics:")
        print(f"  Final Portfolio Value: ${int(metrics['final_value']):,}")
        print(f"  Average Annual Return: {metrics['annual_return']:.1f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"  Maximum Drawdown: {metrics['max_drawdown']:.1f}%")
        print(f"  Average Drawdown: {metrics['avg_drawdown']:.1f}%")
        
        # Get lookbacks and ensure it's a list before sorting
        lookbacks = self.best_params.get('lookbacks', [])
        sorted_lookbacks = sorted(lookbacks) if isinstance(lookbacks, list) else lookbacks
        print(f"  Best parameters: lookbacks={sorted_lookbacks} days")

        # Write metrics to CSV
        self._write_best_performance_to_csv(metrics)

    def _get_canonical_lookbacks(self, lookbacks: List[int]) -> Tuple[int, ...]:
        """Convert lookbacks to canonical (sorted) form.
        
        Args:
            lookbacks: List of lookback periods
            
        Returns:
            Tuple of sorted lookback periods
        """
        return tuple(sorted(lookbacks))
        
    def _get_combination_index(self, canonical: Tuple[int, ...]) -> int:
        """Get or create index for a canonical combination.
        
        Args:
            canonical: Tuple of sorted lookback periods
            
        Returns:
            Index for tracking arrays
        """
        if canonical not in self.combination_indices:
            idx = len(self.combination_indices)
            self.combination_indices[canonical] = idx
            self.canonical_performance_scores.append(0.0)
            self.canonical_visit_counts.append(0)
            self.logger.info(f"New canonical combination: {canonical} -> index {idx}")
        return self.combination_indices[canonical]
    
    def _update_tracking_arrays(self, lookbacks: List[int], score: float) -> None:
        """Update performance tracking for a lookback combination.
        
        Args:
            lookbacks: List of lookback periods (any order)
            score: Performance score to record
        """
        canonical = self._get_canonical_lookbacks(lookbacks)
        idx = self._get_combination_index(canonical)
        
        # Update tracking arrays
        self.canonical_visit_counts[idx] += 1
        old_score = self.canonical_performance_scores[idx]
        n = self.canonical_visit_counts[idx]
        
        # Running average update
        self.canonical_performance_scores[idx] = old_score + (score - old_score) / n
        
        self.logger.debug(f"Updated canonical {canonical}: score={score:.4f}, "
                         f"avg_score={self.canonical_performance_scores[idx]:.4f}, visits={n}")
        
    def _get_random_lookbacks_exploration(self) -> List[int]:
        """Generate random lookbacks for exploration strategy.
        
        Returns:
            List of random lookback periods
        """
        lookbacks = []
        for _ in range(self.n_lookbacks):
            lookback = random.randint(self.min_lookback, self.max_lookback)
            lookbacks.append(lookback)
        
        # Return sorted to maintain canonical form
        return sorted(lookbacks)
    
    def _get_random_lookbacks_exploitation(self) -> List[int]:
        """Generate lookbacks using exploitation strategy (UCB1).
        
        Returns:
            List of lookback periods based on UCB1 selection
        """
        if not self.canonical_visit_counts:
            # Fallback to exploration if no data yet
            return self._get_random_lookbacks_exploration()
            
        total_visits = sum(self.canonical_visit_counts)
        if total_visits == 0:
            return self._get_random_lookbacks_exploration()
            
        # Calculate UCB1 scores for each canonical combination
        ucb_scores = []
        exploration_weight = 1.0  # Can be made configurable
        
        for canonical, idx in self.combination_indices.items():
            exploitation = self.canonical_performance_scores[idx]
            exploration = exploration_weight * math.sqrt(
                2 * math.log(total_visits) / max(1, self.canonical_visit_counts[idx])
            )
            ucb_score = exploitation + exploration
            ucb_scores.append((ucb_score, canonical))
            
        # Select best combination
        if ucb_scores:
            _, best_canonical = max(ucb_scores)
            self.logger.debug(f"UCB1 selected canonical combination: {best_canonical}")
            return list(best_canonical)
        else:
            return self._get_random_lookbacks_exploration()

    def _calculate_theoretical_combinations(self) -> int:
        """Calculate theoretical number of unique combinations possible.
        
        For n_lookbacks lookback periods chosen from min_lookback to max_lookback,
        this calculates the number of unique sorted combinations possible.
        
        Returns:
            Number of theoretical combinations
        """
        # Range of possible lookback values
        range_size = self.max_lookback - self.min_lookback + 1
        
        # Calculate combinations with replacement (since lookbacks can be repeated)
        # Formula: C(n + r - 1, r) where n = range_size, r = n_lookbacks
        # This is equivalent to choosing n_lookbacks items from range_size options with replacement
        from math import comb
        try:
            theoretical = comb(range_size + self.n_lookbacks - 1, self.n_lookbacks)
        except (ValueError, OverflowError):
            # Fallback for very large numbers or edge cases
            theoretical = range_size ** self.n_lookbacks
            
        self.logger.debug(f"Theoretical combinations: {theoretical} "
                         f"(range_size={range_size}, n_lookbacks={self.n_lookbacks})")
        return theoretical

    def select_lookbacks(self, exploration_weight: float = 1.0) -> List[int]:
        """Select next lookback combination using UCB1 algorithm.
        
        Args:
            exploration_weight: Weight for exploration term in UCB1
            
        Returns:
            List of lookback periods to try next
        """
        total_visits = sum(self.canonical_visit_counts) if self.canonical_visit_counts else 0
        if total_visits == 0:
            # Start with evenly spaced lookbacks
            lookbacks = [
                self.min_lookback + 
                i * (self.max_lookback - self.min_lookback) // (self.n_lookbacks - 1)
                for i in range(self.n_lookbacks)
            ]
            return sorted(lookbacks)
            
        # Calculate UCB scores for each canonical combination
        ucb_scores = []
        for canonical, idx in self.combination_indices.items():
            exploitation = self.canonical_performance_scores[idx]
            exploration = exploration_weight * math.sqrt(
                2 * math.log(total_visits) / max(1, self.canonical_visit_counts[idx])
            )
            ucb_scores.append((exploitation + exploration, canonical))
            
        # Select best combination
        if ucb_scores:
            _, best_canonical = max(ucb_scores)
            return list(best_canonical)
        else:
            return self._get_random_lookbacks_exploration()

    def get_permutation_statistics(self) -> Dict[str, Any]:
        """Get statistics about permutation-invariant tracking efficiency.
        
        Returns:
            Dictionary with tracking statistics
        """
        total_combinations = len(self.combination_indices)
        total_visits = sum(self.canonical_visit_counts) if self.canonical_visit_counts else 0
        
        if total_combinations == 0:
            return {
                'total_canonical_combinations': 0,
                'total_visits': 0,
                'average_visits_per_combination': 0.0,
                'most_visited_combination': None,
                'best_performing_combination': None,
                'efficiency_improvement': 'N/A - no data yet'
            }
        
        # Find most visited and best performing combinations
        max_visits_idx = np.argmax(self.canonical_visit_counts)
        best_score_idx = np.argmax(self.canonical_performance_scores)
        
        most_visited_combination = None
        best_performing_combination = None
        
        for canonical, idx in self.combination_indices.items():
            if idx == max_visits_idx:
                most_visited_combination = {
                    'lookbacks': canonical,
                    'visits': self.canonical_visit_counts[idx]
                }
            if idx == best_score_idx:
                best_performing_combination = {
                    'lookbacks': canonical,
                    'score': self.canonical_performance_scores[idx],
                    'visits': self.canonical_visit_counts[idx]
                }
        
        # Calculate efficiency improvement factor
        efficiency_improvement_factor = (
            self._calculate_theoretical_combinations() / total_combinations
        ) if total_combinations > 0 else float('inf')
        
        return {
            'total_canonical_combinations': total_combinations,
            'total_visits': total_visits,
            'average_visits_per_combination': total_visits / total_combinations if total_combinations > 0 else 0.0,
            'most_visited_combination': most_visited_combination,
            'best_performing_combination': best_performing_combination,
            'efficiency_improvement_factor': efficiency_improvement_factor
        }

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
