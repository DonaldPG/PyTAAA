"""
Monte Carlo simulation functions for backtest optimization.

This module contains functions for running Monte Carlo simulations
to optimize trading strategy parameters.
"""

import json
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import gmean

from src.backtest.config import TradingConstants


def random_triangle(
    low: float = 0.0,
    mid: float = 0.5,
    high: float = 1.0,
    size: int = 1
) -> float | np.ndarray:
    """
    Generate random values using averaged uniform and triangular distributions.
    
    This produces values that are more concentrated around the middle
    than a pure uniform distribution, but less peaked than triangular.
    
    Args:
        low: Minimum value.
        mid: Mode (peak) of the triangular distribution.
        high: Maximum value.
        size: Number of values to generate.
        
    Returns:
        Single float if size=1, otherwise numpy array.
    """
    uni = np.random.uniform(low, high, size)
    tri = np.random.triangular(low, mid, high, size)
    
    if size == 1:
        return ((uni + tri) / 2.0)[0]
    else:
        return (uni + tri) / 2.0


def create_temporary_json(
    base_json_fn: str,
    realization_params: Dict[str, Any],
    iter_num: int
) -> str:
    """
    Create a temporary JSON file for a single Monte Carlo realization.
    
    Args:
        base_json_fn: Path to base JSON configuration file.
        realization_params: Dictionary with parameters for this realization.
        iter_num: Iteration number for unique temp file naming.
        
    Returns:
        Path to temporary JSON file.
        
    Raises:
        IOError: If unable to write temporary file.
    """
    # Load base parameters.
    try:
        with open(base_json_fn, "r") as f:
            base_params = json.load(f)
    except Exception as e:
        print(f" ... Warning: Could not load base JSON file: {e}")
        # Create minimal base structure if file doesn't exist or is corrupt.
        base_params = {
            "Email": {"To": "", "From": "", "PW": "", "IPaddress": ""},
            "Text": {"phoneEmail": "", "send_texts": False},
            "FTP": {
                "hostname": "", "remoteIP": "", "username": "",
                "password": "", "remotepath": ""
            },
            "Stock Server": {"quote_download_server": ""},
            "Setup": {"runtime": "15 days", "pausetime": "24 hours"},
            "Valuation": {}
        }
    
    # Update with realization-specific parameters.
    updated_params = base_params.copy()
    
    # Ensure the Valuation section exists.
    if "Valuation" not in updated_params:
        updated_params["Valuation"] = {}
    
    # Update Valuation section with our parameters.
    updated_params["Valuation"].update(realization_params)
    
    # Create temporary file.
    temp_dir = os.path.dirname(base_json_fn)
    temp_json_fn = os.path.join(temp_dir, f"temp_realization_{iter_num}.json")
    
    # Write temporary JSON file.
    try:
        with open(temp_json_fn, "w") as f:
            json.dump(updated_params, f, indent=2)
        print(f" ... Successfully created temp JSON: {temp_json_fn}")
    except Exception as e:
        print(f" ... Error writing temp JSON file: {e}")
        raise
        
    return temp_json_fn


def cleanup_temporary_json(temp_json_fn: str) -> None:
    """
    Clean up temporary JSON file.
    
    Args:
        temp_json_fn: Path to temporary JSON file to remove.
    """
    try:
        if os.path.exists(temp_json_fn):
            os.remove(temp_json_fn)
            print(f"Cleaned up temporary file: {temp_json_fn}")
    except Exception as e:
        print(f"Warning: Could not remove temporary file {temp_json_fn}: {e}")


class MonteCarloBacktest:
    """
    Monte Carlo backtest simulation manager.
    
    Handles parameter generation, simulation execution, and results
    collection for Monte Carlo optimization of trading strategies.
    """
    
    def __init__(
        self,
        base_json_fn: str,
        n_trials: int = 250,
        hold_months: Optional[List[int]] = None
    ):
        """
        Initialize the Monte Carlo backtest manager.
        
        Args:
            base_json_fn: Path to base JSON configuration file.
            n_trials: Number of Monte Carlo trials to run.
            hold_months: List of possible holding periods in months.
        """
        self.base_json_fn = base_json_fn
        self.n_trials = n_trials
        self.hold_months = hold_months or [1, 2, 3, 4, 6, 12]
        
        # Results storage.
        self.results: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_sharpe: float = -np.inf
    
    def generate_random_params(self, iteration: int) -> Dict[str, Any]:
        """
        Generate random parameters for a Monte Carlo trial.
        
        Args:
            iteration: Current iteration number.
            
        Returns:
            Dictionary of randomly generated parameters.
        """
        from random import choice
        
        # Use triangular distributions for better parameter exploration.
        params = {
            "numberStocksTraded": choice([5, 6, 6, 7, 7, 8, 8]),
            "monthsToHold": choice([1, 1, 1, 1, 1, 1, 1, 1, 1, 2]),
            "LongPeriod": int(random_triangle(low=190, mid=370, high=550)),
            "stddevThreshold": random_triangle(low=5.0, mid=7.50, high=10.0),
            "MA1": int(random_triangle(low=75, mid=151, high=300)),
            "MA2": int(random_triangle(low=10, mid=20, high=50)),
            "sma2factor": random_triangle(low=1.65, mid=2.5, high=2.75),
            "rankThresholdPct": random_triangle(low=0.14, mid=0.20, high=0.26),
            "riskDownside_min": random_triangle(low=0.50, mid=0.70, high=0.90),
            "riskDownside_max": random_triangle(low=8.0, mid=10.0, high=13.0),
            "sma_filt_val": random_triangle(low=0.010, mid=0.015, high=0.0225),
            "lowPct": np.random.uniform(10.0, 30.0),
            "hiPct": np.random.uniform(70.0, 90.0),
            "uptrendSignalMethod": "percentileChannels",
        }
        
        # Calculate MA2offset based on MA1 and MA2.
        ma1 = params["MA1"]
        ma2 = params["MA2"]
        params["MA2offset"] = int(
            random_triangle(
                low=(ma1 - ma2) // 20,
                mid=(ma1 - ma2) // 15,
                high=(ma1 - ma2) // 10
            )
        )
        params["MA3"] = ma2 + params["MA2offset"]
        
        # Ensure valid parameter ranges.
        params["MA2"] = max(params["MA2"], 3)
        params["MA1"] = max(params["MA1"], params["MA2"] + 1)
        
        return params
    
    def generate_variation_params(
        self,
        base_params: Dict[str, Any],
        param_to_vary: int
    ) -> Dict[str, Any]:
        """
        Generate parameter variation from a base set.
        
        Args:
            base_params: Base parameter dictionary to vary.
            param_to_vary: Index of parameter to vary (0-11).
            
        Returns:
            Dictionary with one parameter varied from base.
        """
        from random import choice
        
        params = base_params.copy()
        
        if param_to_vary == 0:
            params["numberStocksTraded"] += choice([-1, 0, 1])
        elif param_to_vary == 1:
            for _ in range(15):
                temp = choice(self.hold_months)
                if temp != params["monthsToHold"]:
                    params["monthsToHold"] = temp
                    break
        elif param_to_vary == 2:
            lp = params["LongPeriod"]
            params["LongPeriod"] = int(
                lp * np.around(np.random.uniform(-0.01 * lp, 0.01 * lp))
            )
        elif param_to_vary == 3:
            ma1 = params["MA1"]
            params["MA1"] = int(
                ma1 * np.around(np.random.uniform(-0.01 * ma1, 0.01 * ma1))
            )
        elif param_to_vary == 4:
            ma2 = params["MA2"]
            params["MA2"] = int(
                ma2 * np.around(np.random.uniform(-0.01 * ma2, 0.01 * ma2))
            )
        elif param_to_vary == 5:
            params["MA2offset"] = choice([1, 2, 3])
        elif param_to_vary == 6:
            sf = params["sma2factor"]
            params["sma2factor"] = sf * np.around(
                np.random.uniform(-0.01 * sf, 0.01 * sf), -3
            )
        elif param_to_vary == 7:
            rtp = params["rankThresholdPct"]
            params["rankThresholdPct"] = rtp * np.around(
                np.random.uniform(-0.01 * rtp, 0.01 * rtp), -2
            )
        elif param_to_vary == 8:
            rdmin = params["riskDownside_min"]
            params["riskDownside_min"] = rdmin * np.around(
                np.random.uniform(-0.01 * rdmin, 0.01 * rdmin), -3
            )
        elif param_to_vary == 9:
            rdmax = params["riskDownside_max"]
            params["riskDownside_max"] = rdmax * np.around(
                np.random.uniform(-0.01 * rdmax, 0.01 * rdmax), -3
            )
        elif param_to_vary == 10:
            params["stddevThreshold"] *= np.random.uniform(0.8, 1.2)
        elif param_to_vary == 11:
            params["sma_filt_val"] *= np.random.uniform(0.8, 1.2)
        
        return params
    
    def update_best_result(
        self,
        params: Dict[str, Any],
        sharpe: float
    ) -> None:
        """
        Update best result if current result is better.
        
        Args:
            params: Parameters that produced this result.
            sharpe: Sharpe ratio achieved.
        """
        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            self.best_params = params.copy()
            print(f" ... New best Sharpe: {sharpe:.3f}")


def calculate_sharpe_ratio(
    daily_gains: np.ndarray,
    trading_days: int = TradingConstants.TRADING_DAYS_PER_YEAR
) -> float:
    """
    Calculate annualized Sharpe ratio from daily gains.
    
    Args:
        daily_gains: Array of daily gain ratios (e.g., 1.01 for 1% gain).
        trading_days: Number of trading days per year.
        
    Returns:
        Annualized Sharpe ratio.
    """
    try:
        annual_return = gmean(daily_gains) ** trading_days - 1.0
        annual_volatility = np.std(daily_gains) * np.sqrt(trading_days)
        
        if annual_volatility > 0:
            return annual_return / annual_volatility
        else:
            return 0.0
    except Exception as e:
        print(f" ... Error calculating Sharpe ratio: {e}")
        return 0.0


def calculate_period_metrics(
    portfolio_value: np.ndarray,
    trading_days: int = TradingConstants.TRADING_DAYS_PER_YEAR
) -> Dict[str, Dict[str, float]]:
    """
    Calculate performance metrics for multiple time periods.
    
    Args:
        portfolio_value: Array of portfolio values over time.
        trading_days: Number of trading days per year.
        
    Returns:
        Dictionary with metrics for each period (15Y, 10Y, 5Y, 3Y, 2Y, 1Y).
    """
    daily_gains = portfolio_value[1:] / portfolio_value[:-1]
    
    # Define periods in trading days.
    periods = {
        "15Yr": min(3780, len(daily_gains)),
        "10Yr": 2520,
        "5Yr": 1260,
        "3Yr": 756,
        "2Yr": 504,
        "1Yr": 252,
    }
    
    metrics = {}
    
    for period_name, days in periods.items():
        if days > len(daily_gains):
            continue
            
        period_gains = daily_gains[-days:]
        
        # Sharpe ratio.
        sharpe = calculate_sharpe_ratio(period_gains, trading_days)
        
        # Return (annualized).
        total_return = portfolio_value[-1] / portfolio_value[-days]
        years = days / trading_days
        annual_return = total_return ** (1.0 / years) if years > 0 else 1.0
        
        # CAGR.
        cagr = (portfolio_value[-1] / portfolio_value[-days]) ** (
            trading_days / days
        ) - 1.0
        
        metrics[period_name] = {
            "sharpe": sharpe,
            "return": annual_return,
            "cagr": cagr,
            "days": days,
        }
    
    return metrics


def calculate_drawdown_metrics(
    portfolio_value: np.ndarray
) -> Dict[str, float]:
    """
    Calculate average drawdown for multiple time periods.
    
    Args:
        portfolio_value: Array of portfolio values over time.
        
    Returns:
        Dictionary with average drawdown for each period.
    """
    # Calculate running maximum.
    max_value = np.zeros_like(portfolio_value)
    max_value[0] = portfolio_value[0]
    for i in range(1, len(portfolio_value)):
        max_value[i] = max(max_value[i - 1], portfolio_value[i])
    
    # Calculate drawdown.
    drawdown = portfolio_value / max_value - 1.0
    
    # Define periods.
    periods = {
        "15Yr": min(3780, len(drawdown)),
        "10Yr": 2520,
        "5Yr": 1260,
        "3Yr": 756,
        "2Yr": 504,
        "1Yr": 252,
    }
    
    metrics = {}
    
    for period_name, days in periods.items():
        if days > len(drawdown):
            continue
        metrics[period_name] = float(np.mean(drawdown[-days:]))
    
    return metrics


def beat_buy_hold_test(
    strategy_metrics: Dict[str, Dict[str, float]],
    buyhold_metrics: Dict[str, Dict[str, float]]
) -> float:
    """
    Calculate weighted score for beating buy & hold strategy.
    
    Args:
        strategy_metrics: Performance metrics for trading strategy.
        buyhold_metrics: Performance metrics for buy & hold.
        
    Returns:
        Weighted score (positive means strategy beats buy & hold).
    """
    weights = {
        "15Yr": 1.0 / 15.0,
        "10Yr": 1.0 / 10.0,
        "5Yr": 1.0 / 5.0,
        "3Yr": 1.0 / 3.0,
        "2Yr": 1.0 / 2.0,
        "1Yr": 1.0,
    }
    
    total_weight = sum(weights.values())
    score = 0.0
    
    for period, weight in weights.items():
        if period in strategy_metrics and period in buyhold_metrics:
            strat_sharpe = strategy_metrics[period]["sharpe"]
            bh_sharpe = buyhold_metrics[period]["sharpe"]
            score += (strat_sharpe - bh_sharpe) * weight
    
    return score / total_weight


def beat_buy_hold_test2(
    strategy_metrics: Dict[str, Dict[str, float]],
    buyhold_metrics: Dict[str, Dict[str, float]],
    strategy_drawdown: Dict[str, float],
    buyhold_drawdown: Dict[str, float]
) -> float:
    """
    Calculate comprehensive score for beating buy & hold strategy.
    
    This test considers returns, positive returns, and drawdowns
    with time-weighted scoring (recent periods weighted more).
    
    Args:
        strategy_metrics: Performance metrics for trading strategy.
        buyhold_metrics: Performance metrics for buy & hold.
        strategy_drawdown: Drawdown metrics for trading strategy.
        buyhold_drawdown: Drawdown metrics for buy & hold.
        
    Returns:
        Score as ratio from 0 to 1 (higher is better).
    """
    period_weights = {
        "15Yr": 1.0,
        "10Yr": 1.0,
        "5Yr": 1.0,
        "3Yr": 1.5,
        "2Yr": 2.0,
        "1Yr": 2.5,
    }
    
    score = 0.0
    
    for period, weight in period_weights.items():
        # Return comparison.
        if period in strategy_metrics and period in buyhold_metrics:
            if strategy_metrics[period]["return"] > buyhold_metrics[period]["return"]:
                score += weight
            
            # Positive return bonus.
            if strategy_metrics[period]["return"] > 0:
                score += weight
        
        # Drawdown comparison (less negative is better).
        if period in strategy_drawdown and period in buyhold_drawdown:
            if strategy_drawdown[period] > buyhold_drawdown[period]:
                score += weight
    
    # Normalize to 0-1 range.
    max_score = sum(period_weights.values()) * 3  # 3 tests per period
    return score / max_score
