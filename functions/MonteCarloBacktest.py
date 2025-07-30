"""Monte Carlo backtesting system with read-only data access."""

import os
import time
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
from dataclasses import dataclass, field

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
        min_lookback: int = 10,
        max_lookback: int = 252,
        n_bins: int = 20  # Number of bins for lookback values
    ) -> None:
        """Initialize Monte Carlo backtesting framework with read-only data access."""
        self.iterations = iterations
        self.min_iterations_for_exploit = min_iterations_for_exploit
        self.trading_frequency = trading_frequency
        self.model_paths = model_paths
        self.min_lookback = min_lookback
        self.max_lookback = max_lookback
        
        # Initialize binning for lookback periods
        self.bin_edges = np.linspace(min_lookback, max_lookback, n_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        
        # Initialize tracking arrays
        self.visit_counts = np.zeros([n_bins] * 3)  # 3D array for 3 lookback periods
        self.performance_scores = np.zeros_like(self.visit_counts)
        
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

                # Simulate trading through all dates
                for t in range(1, n_dates):
                    date = pd.Timestamp(self.dates[t])
                    prev_date = pd.Timestamp(self.dates[t-1])
                    
                    # Check if we should trade
                    if self._should_trade(self.dates[t]):
                        # Pass current iteration number for exploration/exploitation decision
                        current_model, lookbacks = self._select_best_model(t, iteration=i)
                        current_lookbacks = lookbacks  # Save lookbacks for this iteration
                        current_selections[date.strftime('%Y-%m-%d')] = current_model
                        current_params["lookbacks"] = lookbacks
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
                current_performance = metrics.sharpe_ratio  # Use Sharpe ratio as performance metric
                
                # Update n-dimensional tracking arrays
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
                    
                    # Log current state of exploration/exploitation
                    if i > 0:
                        total_visits = np.sum(self.visit_counts)
                        max_visits = np.max(self.visit_counts)
                        min_visits = np.min(self.visit_counts)
                        avg_visits = np.mean(self.visit_counts)
                        self.logger.info(
                            f"Visit statistics after {i} iterations:\n"
                            f"  Total visits: {total_visits}\n"
                            f"  Max visits to a combination: {max_visits}\n"
                            f"  Min visits to a combination: {min_visits}\n"
                            f"  Average visits per combination: {avg_visits:.2f}"
                        )
                
        except KeyboardInterrupt:
            print("\n\nStopped early by user")
            print(f"Completed {i+1} iterations")
            if not top_models:
                return []
        
        total_time = time.time() - start_time
        print(f"\nMonte Carlo simulation completed in {total_time/60:.1f} minutes")
        print(f"Best Sharpe ratio: {best_performance:.2f}")
        
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
        """Generate lookback periods using n-dimensional binned exploration/exploitation.
        
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
            
        # Check if we should exploit
        if iteration is not None and iteration > self.min_iterations_for_exploit:
            exploit_rate = self._calculate_exploitation_rate(iteration)
            if random.random() < exploit_rate:
                # Use exploitation strategy
                return self._get_random_lookbacks_exploitation()
                
        # Use exploration strategy
        return self._get_random_lookbacks_exploration()

    def _select_best_model(self, current_idx: int, lookback: Optional[int] = None, 
                          iteration: Optional[int] = None) -> Tuple[str, List[int]]:
        """Select best performing model based on multiple performance metrics and lookbacks.
        
        Args:
            current_idx: Current index in the backtest
            lookback: Optional override for lookback period (used for testing)
            iteration: Current iteration number for exploration/exploitation decision
            
        Returns:
            Tuple of (best_model_name, list_of_lookbacks_used)
        """
        # Get configuration for model selection
        config_path = os.path.join(os.path.dirname(self.model_paths[next(iter(self.model_paths))]), 
                                 'pytaaa_model_switching_params.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        n_lookbacks = config['model_selection']['n_lookbacks']
        metric_weights = config['model_selection']['performance_metrics']
        
        # Generate diverse lookback periods, passing current iteration
        lookbacks = self._generate_diverse_lookbacks(n_lookbacks, lookback, iteration)
        
        # Initialize rank storage
        models = list(self.portfolio_histories.keys())
        all_ranks = np.zeros((len(models), 7 * n_lookbacks))  # 7 metrics × n_lookbacks
        
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
    
    def plot_performance(self, portfolio_values: np.ndarray, metrics: dict) -> None:
        """Plot portfolio performance metrics and save visualization.
        
        Args:
            portfolio_values: Array of portfolio values over time
            metrics: Dictionary of performance metrics
        """
        # Create figure with subplot layout - adjusted for laptop screen
        fig = plt.figure(figsize=(12, 7))  # Width: 12 inches, Height: 7 inches for 16:9 aspect ratio
        gs = plt.GridSpec(6, 1)  # 6 rows total to allow for 83/17 split
        
        # Main portfolio performance plot
        ax1 = fig.add_subplot(gs[0:5, 0])  # Takes up first 5 rows (83%)
        
        # Plot historical values for each model
        colors = ['b', 'r', 'g', 'c', 'm', 'k']  # Added black for cash
        model_to_color = {}  # Map models to their colors
        date_index = pd.DatetimeIndex(self.dates)
        
        # Normalize all series to start at 10000
        for (model, values), color in zip(self.portfolio_histories.items(), colors):
            if model != "cash":
                model_to_color[model] = color
                # Normalize to start at 10000
                start_value = values.iloc[0]
                normalized_values = values * (10000.0 / start_value)
                ax1.plot(date_index, normalized_values,
                        color=color, alpha=0.5, label=f"{model}",
                        linewidth=1)
        
        model_to_color["cash"] = 'k'  # Add cash to color mapping

        # Plot Monte Carlo best portfolio (already starts at 10000)
        ax1.plot(date_index, portfolio_values, 
                'k-', linewidth=2, label='Monte Carlo Best')
        
        # Configure main plot
        ax1.set_yscale('log')
        
        # Configure grid with both major and minor lines
        ax1.grid(True, which='major', alpha=0.4, linewidth=0.8)
        ax1.grid(True, which='minor', alpha=0.2, linewidth=0.5)
        ax1.minorticks_on()
        
        # Add current date/time and lookback values to plot
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        lookbacks = self.best_params.get('lookbacks', [])
        _lb_list = sorted(lookbacks) if isinstance(lookbacks, list) else lookbacks
        lookback_text = f"Best parameters: lookbacks={_lb_list} days"
        text_str = f"{current_time}\n{lookback_text}"
        
        # Position text in upper left, slightly below the title
        ax1.text(0.02, 0.95, text_str,
                transform=ax1.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                fontsize=9, verticalalignment='top')
        
        ax1.set_title('Portfolio Performance: Historical Models vs Monte Carlo', 
                     pad=20, fontsize=14)
        ax1.set_xlabel('')  # Remove x-label from top plot
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        
        # Format axes
        ax1.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'${int(x):,}')
        )
        
        # Add legend to main plot with smaller font in lower right corner
        ax1.legend(loc='lower right', fontsize=8)
        
        # Create model selection subplot
        ax2 = fig.add_subplot(gs[5, 0])  # Takes up last row (17%)
        
        # Create numeric mapping for all models including cash
        unique_models = sorted(list(self.portfolio_histories.keys()))
        model_to_num = {model: i for i, model in enumerate(unique_models)}
        
        # Get model selections over time
        current_model = None
        model_selections = []
        for date in self.dates:
            if self._should_trade(date):
                # Find closest model selection for this date
                current_model = self.best_model_selections.get(
                    pd.Timestamp(date).strftime('%Y-%m-%d'), current_model
                )
            model_selections.append(current_model if current_model is not None else "cash")
            
        # Convert model selections to numeric values
        numeric_selections = [model_to_num.get(model, -1) for model in model_selections]
        
        # Plot model selections with corresponding colors
        for model in unique_models:
            mask = [x == model_to_num[model] for x in numeric_selections]
            if any(mask):  # Only plot if model was selected
                ax2.scatter(date_index[mask], [model_to_num[model]] * sum(mask),
                          color=model_to_color[model], alpha=0.7, s=20,
                          label=f"{model} periods")
                
                # Draw lines connecting points for the same model
                ax2.plot(date_index[mask], [model_to_num[model]] * sum(mask),
                        color=model_to_color[model], alpha=0.3, linewidth=1)
        
        # Configure model selection subplot
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Selected Model', fontsize=12)
        
        # Set tick locations and labels for all models
        ax2.set_yticks(list(range(len(unique_models))))
        ax2.set_yticklabels(unique_models)

        # Rotate and align the tick labels so they look better
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Add legend to model selection subplot with smaller font
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=7)
        
        # Adjust layout to prevent overlapping
        plt.tight_layout()
        
        # Save plot
        plt.savefig('monte_carlo_best_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _print_best_parameters(self, metrics: Dict[str, float]) -> None:
        """Print best parameters with sorted lookback list.
        
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
    
    def _get_bin_indices(self, lookbacks):
        """Convert lookback values to bin indices."""
        lookbacks_arr = np.array(lookbacks, dtype=np.float64)
        indices = _compute_bin_indices(lookbacks_arr, self.bin_edges)
        return indices.tolist()
    
    def _get_random_lookbacks_exploration(self):
        """Generate random lookbacks favoring less visited combinations."""
        # Invert and normalize visit counts to get probabilities
        probs = 1 / (self.visit_counts + 1)  # Add 1 to avoid division by zero
        probs = probs / np.sum(probs)
        
        # Get random index
        flat_probs = probs.ravel()
        idx = np.random.choice(len(flat_probs), p=flat_probs)
        
        # Convert flat index back to n-dimensional indices
        nd_indices = np.unravel_index(idx, self.visit_counts.shape)
        
        # Get lookback values from bin centers
        lookbacks = [self.bin_centers[i] for i in nd_indices]
        
        # Add noise within bins
        bin_width = self.bin_edges[1] - self.bin_edges[0]
        noisy_lookbacks = []
        for lb in lookbacks:
            noise = np.random.uniform(-bin_width/2, bin_width/2)
            val = lb + noise
            val = min(max(val, self.min_lookback), self.max_lookback)
            noisy_lookbacks.append(int(val))
        
        return noisy_lookbacks
    
    def _get_random_lookbacks_exploitation(self):
        """Generate random lookbacks based on past performance."""
        # Normalize performance scores to probabilities
        scores = self.performance_scores - np.min(self.performance_scores)
        if np.max(scores) > 0:
            probs = scores / np.sum(scores)
        else:
            probs = np.ones_like(scores) / scores.size
        
        # Get random index
        flat_probs = probs.ravel()
        idx = np.random.choice(len(flat_probs), p=flat_probs)
        
        # Convert flat index back to n-dimensional indices
        nd_indices = np.unravel_index(idx, self.performance_scores.shape)
        
        # Get lookback values from bin centers
        lookbacks = [self.bin_centers[i] for i in nd_indices]
        
        # Add small noise within bins (using half bin width)
        bin_width = self.bin_edges[1] - self.bin_edges[0]
        noisy_lookbacks = []
        for lb in lookbacks:
            noise = np.random.uniform(-bin_width/4, bin_width/4)  # Using half the noise
            val = lb + noise
            val = min(max(val, self.min_lookback), self.max_lookback)
            noisy_lookbacks.append(int(val))
        
        return noisy_lookbacks
    
    def _update_tracking_arrays(self, lookbacks: List[int], performance: float) -> None:
        """Update visit counts and performance scores for a combination of lookbacks."""
        indices = self._get_bin_indices(lookbacks)
        self.visit_counts[tuple(indices)] += 1
        
        # Update performance score if better than current best
        curr_score = self.performance_scores[tuple(indices)]
        if performance > curr_score:
            self.performance_scores[tuple(indices)] = performance