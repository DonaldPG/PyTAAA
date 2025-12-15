"""
Portfolio period metrics calculation module.

This module provides functions to calculate Sharpe and Sortino ratios
for multiple time periods, as well as model-switching effectiveness
analysis comparing dynamic switching vs individual trading methods.
"""

import numpy as np
import pandas as pd
from datetime import date
from typing import List, Dict, Optional, Tuple, Any
from scipy.stats import linregress
import logging

# Get module-specific logger
logger = logging.getLogger(__name__)


# Period mapping following existing patterns from dailyBacktest_pctLong.py
PERIOD_DAYS_MAPPING = {
    "3M": 63,     # ~3 months
    "6M": 126,    # ~6 months  
    "1Y": 252,    # ~1 year
    "3Y": 756,    # ~3 years
    "5Y": 1260,   # ~5 years
    "10Y": 2520,  # ~10 years
    "20Y": 5040,  # ~20 years
    "MAX": None   # Full dataset
}


def calculate_sharpe_sortino_ratios(portfolio_values: np.ndarray, 
                                  risk_free_rate: float = 0.0) -> Tuple[float, float]:
    """
    Calculate Sharpe and Sortino ratios for portfolio values.
    
    Args:
        portfolio_values: Array of portfolio values over time
        risk_free_rate: Risk-free rate (default: 0.0)
        
    Returns:
        Tuple of (sharpe_ratio, sortino_ratio)
    """
    if len(portfolio_values) < 2:
        return 0.0, 0.0
        
    try:
        # Calculate daily returns
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Calculate annualized return
        mean_return = np.mean(daily_returns)
        annualized_return = (1 + mean_return) ** 252 - 1
        
        # Calculate standard deviation of returns (annualized)
        std_dev_return = np.std(daily_returns) * np.sqrt(252)
        
        # Sharpe ratio
        sharpe = (annualized_return - risk_free_rate) / std_dev_return if std_dev_return > 0 else 0.0
        
        # Sortino ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < risk_free_rate / 252]
        downside_deviation = np.sqrt(np.mean(downside_returns**2)) if len(downside_returns) > 0 else 0
        downside_deviation *= np.sqrt(252)  # Annualize
        
        sortino = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0.0
        
        return sharpe, sortino
        
    except Exception as e:
        logger.warning(f"Error calculating Sharpe/Sortino ratios: {e}")
        return 0.0, 0.0


def calculate_period_metrics(portfolio_values: np.ndarray, 
                           dates: List[date],
                           periods: List[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Calculate Sharpe and Sortino ratios for specified periods.
    
    Args:
        portfolio_values: Array of portfolio values over time
        dates: List of dates corresponding to portfolio values
        periods: List of period strings (default: all standard periods)
        
    Returns:
        Dict with structure: {
            "3M": {"sharpe_ratio": 1.5, "sortino_ratio": 1.8},
            "6M": {"sharpe_ratio": 1.6, "sortino_ratio": 1.9},
            ...
        }
    """
    if periods is None:
        periods = list(PERIOD_DAYS_MAPPING.keys())
    
    # Improved validation
    if (portfolio_values is None or len(portfolio_values) < 2 or 
        dates is None or len(dates) != len(portfolio_values)):
        logger.warning(f"Invalid input: portfolio_values length={len(portfolio_values) if portfolio_values is not None else 'None'}, dates length={len(dates) if dates is not None else 'None'}")
        return {period: {"sharpe_ratio": 0.0, "sortino_ratio": 0.0} 
                for period in periods}
    
    # Check for valid portfolio values
    if np.any(np.isnan(portfolio_values)) or np.any(np.isinf(portfolio_values)):
        logger.warning("Portfolio values contain NaN or inf values")
        return {period: {"sharpe_ratio": 0.0, "sortino_ratio": 0.0} 
                for period in periods}
    
    results = {}
    
    for period in periods:
        try:
            if period == "MAX":
                # Use full dataset
                period_values = portfolio_values
            else:
                period_days = PERIOD_DAYS_MAPPING.get(period)
                if period_days is None or period_days >= len(portfolio_values):
                    # Use full dataset if period is longer than available data
                    period_values = portfolio_values
                else:
                    # Use last N days
                    period_values = portfolio_values[-period_days:]
            
            if len(period_values) < 2:
                results[period] = {"sharpe_ratio": 0.0, "sortino_ratio": 0.0}
                continue
            
            # Calculate Sharpe and Sortino ratios
            sharpe, sortino = calculate_sharpe_sortino_ratios(period_values)
            
            results[period] = {
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino
            }
            
        except Exception as e:
            logger.warning(f"Error calculating metrics for period {period}: {e}")
            results[period] = {"sharpe_ratio": 0.0, "sortino_ratio": 0.0}
    
    return results


def calculate_equal_weight_baseline(monte_carlo) -> np.ndarray:
    """
    Calculate equal-weight buy-and-hold portfolio of 4 trading methods.
    
    Args:
        monte_carlo: MonteCarloBacktest instance with portfolio_histories
        
    Returns:
        np.ndarray: Portfolio values over time with 25% allocation to each method
    """
    try:
        # Get the 4 base trading methods (exclude cash)
        base_methods = [model for model in monte_carlo.portfolio_histories.keys() 
                       if model != "cash"]
        
        if len(base_methods) < 4:
            logger.warning(f"Expected 4 base methods, found {len(base_methods)}")
            # Pad with cash if needed
            while len(base_methods) < 4:
                base_methods.append("cash")
        
        # Get date range from monte_carlo
        date_index = pd.DatetimeIndex(monte_carlo.dates)
        
        # Initialize equal-weight portfolio with starting value of 10000
        equal_weight_portfolio = np.zeros(len(date_index))
        equal_weight_portfolio[0] = 10000.0
        
        # Calculate equal-weight performance (25% in each method)
        weight_per_method = 0.25
        
        for i in range(1, len(date_index)):
            daily_return = 0.0
            
            for method in base_methods[:4]:  # Only use first 4 methods
                if method == "cash":
                    method_return = 0.0  # Cash has no return
                else:
                    try:
                        prev_value = monte_carlo.portfolio_histories[method].iloc[i-1]
                        curr_value = monte_carlo.portfolio_histories[method].iloc[i]
                        method_return = (curr_value / prev_value) - 1 if prev_value > 0 else 0.0
                    except (IndexError, KeyError):
                        method_return = 0.0
                
                daily_return += weight_per_method * method_return
            
            equal_weight_portfolio[i] = equal_weight_portfolio[i-1] * (1 + daily_return)
        
        return equal_weight_portfolio
        
    except Exception as e:
        logger.error(f"Error calculating equal-weight baseline: {e}")
        # Return constant portfolio as fallback
        return np.ones(len(monte_carlo.dates)) * 10000.0


def calculate_period_outperformance(model_switching_portfolio: np.ndarray,
                                  equal_weight_portfolio: np.ndarray,
                                  dates: List[date],
                                  period_length_days: int = 252) -> Dict[str, float]:
    """
    Calculate percentage of periods where model-switching beats equal-weight.
    
    Args:
        model_switching_portfolio: Model-switching portfolio values
        equal_weight_portfolio: Equal-weight baseline portfolio values
        dates: List of dates
        period_length_days: Rolling period length in days (default: 1 year)
        
    Returns:
        Dict with outperformance statistics
    """
    try:
        if len(model_switching_portfolio) != len(equal_weight_portfolio):
            logger.warning("Portfolio lengths don't match for outperformance calculation")
            return {"period_outperformance_pct": 0.0, "avg_excess_return": 0.0}
        
        if len(model_switching_portfolio) < period_length_days:
            # Use full period if data is shorter than specified period
            period_length_days = len(model_switching_portfolio)
        
        outperformance_count = 0
        total_periods = 0
        excess_returns = []
        
        # Calculate rolling period outperformance
        for i in range(period_length_days, len(model_switching_portfolio)):
            start_idx = i - period_length_days
            
            # Calculate returns for this period
            ms_start = model_switching_portfolio[start_idx]
            ms_end = model_switching_portfolio[i]
            ew_start = equal_weight_portfolio[start_idx]
            ew_end = equal_weight_portfolio[i]
            
            if ms_start > 0 and ew_start > 0:
                ms_return = (ms_end / ms_start) - 1
                ew_return = (ew_end / ew_start) - 1
                
                excess_return = ms_return - ew_return
                excess_returns.append(excess_return)
                
                if ms_return > ew_return:
                    outperformance_count += 1
                
                total_periods += 1
        
        if total_periods == 0:
            return {"period_outperformance_pct": 0.0, "avg_excess_return": 0.0}
        
        outperformance_pct = (outperformance_count / total_periods) * 100
        avg_excess_return = np.mean(excess_returns) if excess_returns else 0.0
        
        return {
            "period_outperformance_pct": outperformance_pct,
            "avg_excess_return": avg_excess_return,
            "total_periods_analyzed": total_periods
        }
        
    except Exception as e:
        logger.error(f"Error calculating period outperformance: {e}")
        return {"period_outperformance_pct": 0.0, "avg_excess_return": 0.0}


def calculate_outperformance_percentage(model_switching_metrics: Dict[str, Dict[str, float]],
                                      all_methods_metrics: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, float]:
    """
    Calculate percentage of individual period-method comparisons where model-switching outperforms base methods.
    
    Args:
        model_switching_metrics: Period metrics for model-switching portfolio
        all_methods_metrics: Metrics for all methods
        
    Returns:
        Percentage of individual comparisons where model-switching wins
    """
    try:
        periods = list(PERIOD_DAYS_MAPPING.keys())
        base_methods = ["naz100_pine", "naz100_hma", "naz100_pi", "sp500_hma"]
        
        sharpe_wins = 0
        sortino_wins = 0
        total_sharpe_comparisons = 0
        total_sortino_comparisons = 0
        
        for period in periods:
            # Get model-switching metrics for this period
            ms_data = model_switching_metrics.get(period, {})
            ms_sharpe = ms_data.get("sharpe_ratio", 0.0)
            ms_sortino = ms_data.get("sortino_ratio", 0.0)
            
            # Compare against each base method individually
            for method in base_methods:
                method_data = all_methods_metrics.get(method, {}).get(period, {})
                method_sharpe = method_data.get("sharpe_ratio", 0.0)
                method_sortino = method_data.get("sortino_ratio", 0.0)
                
                # Count individual Sharpe wins
                if ms_sharpe > method_sharpe:
                    sharpe_wins += 1
                total_sharpe_comparisons += 1
                
                # Count individual Sortino wins
                if ms_sortino > method_sortino:
                    sortino_wins += 1
                total_sortino_comparisons += 1
        
        # Calculate percentages
        sharpe_outperformance_pct = (sharpe_wins / total_sharpe_comparisons * 100) if total_sharpe_comparisons > 0 else 0.0
        sortino_outperformance_pct = (sortino_wins / total_sortino_comparisons * 100) if total_sortino_comparisons > 0 else 0.0
        
        return {
            "sharpe_outperformance_pct": sharpe_outperformance_pct,
            "sortino_outperformance_pct": sortino_outperformance_pct,
            "sharpe_wins": sharpe_wins,
            "total_sharpe_comparisons": total_sharpe_comparisons,
            "sortino_wins": sortino_wins,
            "total_sortino_comparisons": total_sortino_comparisons
        }
        
    except Exception as e:
        logger.error(f"Error calculating outperformance percentage: {e}")
        return {
            "sharpe_outperformance_pct": 0.0,
            "sortino_outperformance_pct": 0.0,
            "sharpe_wins": 0,
            "total_sharpe_comparisons": 0,
            "sortino_wins": 0,
            "total_sortino_comparisons": 0
        }


def calculate_all_methods_metrics(monte_carlo, 
                                dates: List[date],
                                periods: List[str] = None) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Calculate period metrics for all 4 trading methods plus model-switching.
    
    Args:
        monte_carlo: MonteCarloBacktest instance
        dates: List of dates
        periods: List of period strings (default: all standard periods)
        
    Returns:
        Dict with structure: {
            "naz100_pine": {"3M": {"sharpe_ratio": 1.5, "sortino_ratio": 1.8}, ...},
            "model_switching": {"3M": {"sharpe_ratio": 1.6, "sortino_ratio": 1.9}, ...},
            ...
        }
    """
    if periods is None:
        periods = list(PERIOD_DAYS_MAPPING.keys())
    
    results = {}
    
    try:
        # Calculate metrics for each base trading method
        base_methods = [model for model in monte_carlo.portfolio_histories.keys() 
                       if model != "cash"]
        
        for method in base_methods:
            try:
                # Normalize portfolio to start at 10000
                method_values = monte_carlo.portfolio_histories[method].values
                if len(method_values) > 0 and method_values[0] > 0:
                    normalized_values = method_values * (10000.0 / method_values[0])
                else:
                    normalized_values = np.ones(len(method_values)) * 10000.0
                
                results[method] = calculate_period_metrics(
                    normalized_values, dates, periods
                )
                
            except Exception as e:
                logger.warning(f"Error calculating metrics for method {method}: {e}")
                results[method] = {period: {"sharpe_ratio": 0.0, "sortino_ratio": 0.0} 
                                 for period in periods}
        
        # Add CASH method with zero ratios
        results["cash"] = {period: {"sharpe_ratio": 0.0, "sortino_ratio": 0.0} 
                          for period in periods}
        
        return results
        
    except Exception as e:
        logger.error(f"Error calculating all methods metrics: {e}")
        return {}


def calculate_ranking_effectiveness(model_switching_metrics: Dict[str, Dict[str, float]],
                                  all_methods_metrics: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, float]:
    """
    Calculate ranking-based effectiveness metrics.
    
    Args:
        model_switching_metrics: Period metrics for model-switching portfolio
        all_methods_metrics: Metrics for all methods including model-switching
        
    Returns:
        Dict with ranking statistics
    """
    try:
        periods = list(PERIOD_DAYS_MAPPING.keys())
        all_ranks = []
        
        # Methods to include in ranking (model-switching will be added separately)
        base_methods = ["naz100_pine", "naz100_hma", "naz100_pi", "sp500_hma", "cash"]
        
        for period in periods:
            # Get model-switching metrics for this period
            ms_sharpe = model_switching_metrics.get(period, {}).get("sharpe_ratio", 0.0)
            ms_sortino = model_switching_metrics.get(period, {}).get("sortino_ratio", 0.0)
            
            # Collect metrics for all methods for this period
            sharpe_values = [ms_sharpe]  # Start with model-switching
            sortino_values = [ms_sortino]
            
            for method in base_methods:
                method_data = all_methods_metrics.get(method, {}).get(period, {})
                sharpe_values.append(method_data.get("sharpe_ratio", 0.0))
                sortino_values.append(method_data.get("sortino_ratio", 0.0))
            
            # Calculate ranks using inverted ranking (5=best, 0=worst) for "bigger is better"
            # For Sharpe ratios: higher is better
            sharpe_array = np.array(sharpe_values)
            sharpe_ranks = (-sharpe_array).argsort().argsort()  # 0-based descending ranks
            ms_sharpe_rank = 5 - sharpe_ranks[0]  # Invert to make 5=best, 0=worst
            
            # For Sortino ratios: higher is better  
            sortino_array = np.array(sortino_values)
            sortino_ranks = (-sortino_array).argsort().argsort()  # 0-based descending ranks
            ms_sortino_rank = 5 - sortino_ranks[0]  # Invert to make 5=best, 0=worst
            
            all_ranks.extend([ms_sharpe_rank, ms_sortino_rank])
        
        # Calculate average rank across all periods and metrics
        average_rank = np.mean(all_ranks) if all_ranks else 0.0  # Worst case default (inverted)
        
        return {
            "average_rank": average_rank,
            "total_rankings": len(all_ranks)
        }
        
    except Exception as e:
        logger.error(f"Error calculating ranking effectiveness: {e}")
        return {"average_rank": 0.0, "total_rankings": 0}


def analyze_model_switching_effectiveness(monte_carlo,
                                        lookbacks: List[int]) -> Dict[str, Any]:
    """
    Complete analysis comparing model-switching vs individual trading methods.
    
    Args:
        monte_carlo: MonteCarloBacktest instance
        lookbacks: List of lookback periods used for model switching
        
    Returns:
        Dict containing:
        - period_metrics: Period metrics for model-switching portfolio
        - individual_method_metrics: Metrics for all base methods
        - sharpe_outperformance_pct: % of individual comparisons where model-switching beats base methods (Sharpe)
        - sortino_outperformance_pct: % of individual comparisons where model-switching beats base methods (Sortino)
        - average_rank: Average ranking across all periods and metrics
    """
    try:
        # Calculate model-switching portfolio using the best lookbacks
        model_switching_portfolio = monte_carlo._calculate_model_switching_portfolio(lookbacks)
        
        # Get standard periods for analysis
        periods = ["3M", "6M", "1Y", "3Y", "5Y", "10Y", "20Y", "MAX"]
        
        # Calculate period metrics for model-switching portfolio
        model_switching_metrics = calculate_period_metrics(
            model_switching_portfolio, monte_carlo.dates, periods
        )
        
        # Calculate metrics for all individual methods
        all_methods_metrics = calculate_all_methods_metrics(
            monte_carlo, monte_carlo.dates, periods
        )
        
        # Calculate outperformance percentages (new method)
        outperformance_stats = calculate_outperformance_percentage(
            model_switching_metrics, all_methods_metrics
        )
        
        # Calculate ranking effectiveness
        ranking_stats = calculate_ranking_effectiveness(
            model_switching_metrics, all_methods_metrics
        )
        
        return {
            "period_metrics": model_switching_metrics,
            "individual_method_metrics": all_methods_metrics,
            "sharpe_outperformance_pct": outperformance_stats["sharpe_outperformance_pct"],
            "sortino_outperformance_pct": outperformance_stats["sortino_outperformance_pct"],
            "sharpe_wins": outperformance_stats["sharpe_wins"],
            "total_sharpe_comparisons": outperformance_stats["total_sharpe_comparisons"],
            "sortino_wins": outperformance_stats["sortino_wins"],
            "total_sortino_comparisons": outperformance_stats["total_sortino_comparisons"],
            "average_rank": ranking_stats["average_rank"],
            "total_rankings": ranking_stats["total_rankings"]
        }
        
    except Exception as e:
        logger.error(f"Error in model switching effectiveness analysis: {e}")
        return {
            "period_metrics": {},
            "individual_method_metrics": {},
            "sharpe_outperformance_pct": 0.0,
            "sortino_outperformance_pct": 0.0,
            "sharpe_wins": 0,
            "total_sharpe_comparisons": 0,
            "sortino_wins": 0,
            "total_sortino_comparisons": 0,
            "average_rank": 6.0,
            "total_rankings": 0
        }


def create_comparison_dataframes(model_effectiveness: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert model effectiveness data into comparison DataFrames for Sharpe and Sortino ratios.
    
    Args:
        model_effectiveness: Dictionary from analyze_model_switching_effectiveness()
        
    Returns:
        Tuple of (sharpe_df, sortino_df) where each DataFrame has:
        - Index: Time periods (3M, 6M, 1Y, etc.)
        - Columns: Trading methods (model_switching, naz100_pine, naz100_hma, naz100_pi, sp500_hma, cash)
    """
    try:
        # Get model-switching metrics
        ms_metrics = model_effectiveness.get('period_metrics', {})
        
        # Get individual method metrics  
        individual_metrics = model_effectiveness.get('individual_method_metrics', {})
        
        # Define periods and methods
        periods = ["3M", "6M", "1Y", "3Y", "5Y", "10Y", "20Y", "MAX"]
        methods = ["model_switching", "naz100_pine", "naz100_hma", "naz100_pi", "sp500_hma", "cash"]
        
        # Initialize arrays for Sharpe and Sortino ratios
        sharpe_data = np.zeros((len(periods), len(methods)))
        sortino_data = np.zeros((len(periods), len(methods)))
        
        # Fill data for each period
        for period_idx, period in enumerate(periods):
            # Model-switching data
            ms_period_data = ms_metrics.get(period, {"sharpe_ratio": 0.0, "sortino_ratio": 0.0})
            sharpe_data[period_idx, 0] = ms_period_data.get("sharpe_ratio", 0.0)
            sortino_data[period_idx, 0] = ms_period_data.get("sortino_ratio", 0.0)
            
            # Individual methods data
            for method_idx, method in enumerate(methods[1:], 1):  # Skip model_switching
                method_data = individual_metrics.get(method, {}).get(period, {"sharpe_ratio": 0.0, "sortino_ratio": 0.0})
                sharpe_data[period_idx, method_idx] = method_data.get("sharpe_ratio", 0.0)
                sortino_data[period_idx, method_idx] = method_data.get("sortino_ratio", 0.0)
        
        # Create DataFrames
        sharpe_df = pd.DataFrame(
            sharpe_data,
            index=periods,
            columns=methods
        )
        
        sortino_df = pd.DataFrame(
            sortino_data,
            index=periods,
            columns=methods
        )
        
        return sharpe_df, sortino_df
        
    except Exception as e:
        logger.error(f"Error creating comparison DataFrames: {e}")
        # Return empty DataFrames with proper structure
        periods = ["3M", "6M", "1Y", "3Y", "5Y", "10Y", "20Y", "MAX"]
        methods = ["model_switching", "naz100_pine", "naz100_hma", "naz100_pi", "sp500_hma", "cash"]
        
        empty_df = pd.DataFrame(
            np.zeros((len(periods), len(methods))),
            index=periods,
            columns=methods
        )
        
        return empty_df.copy(), empty_df.copy()


def analyze_method_performance(sharpe_df: pd.DataFrame, sortino_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze performance comparison between model-switching and individual methods.
    
    Args:
        sharpe_df: DataFrame with Sharpe ratios for all methods and periods
        sortino_df: DataFrame with Sortino ratios for all methods and periods
        
    Returns:
        Dictionary with comparison statistics
    """
    try:
        methods = sharpe_df.columns.tolist()
        periods = sharpe_df.index.tolist()
        base_methods = [m for m in methods if m not in ["model_switching", "cash"]]
        
        results = {
            "sharpe_comparison": {},
            "sortino_comparison": {},
            "combined_analysis": {}
        }
        
        # Sharpe ratio analysis
        for method in base_methods:
            ms_wins = (sharpe_df["model_switching"] > sharpe_df[method]).sum()
            total_periods = len(periods)
            win_rate = (ms_wins / total_periods) * 100
            
            results["sharpe_comparison"][method] = {
                "periods_outperformed": ms_wins,
                "total_periods": total_periods,
                "win_rate_pct": win_rate,
                "avg_difference": (sharpe_df["model_switching"] - sharpe_df[method]).mean()
            }
        
        # Sortino ratio analysis  
        for method in base_methods:
            ms_wins = (sortino_df["model_switching"] > sortino_df[method]).sum()
            total_periods = len(periods)
            win_rate = (ms_wins / total_periods) * 100
            
            results["sortino_comparison"][method] = {
                "periods_outperformed": ms_wins,
                "total_periods": total_periods,
                "win_rate_pct": win_rate,
                "avg_difference": (sortino_df["model_switching"] - sortino_df[method]).mean()
            }
        
        # Combined analysis - periods where model-switching beats ALL base methods
        sharpe_beats_all = sharpe_df["model_switching"] > sharpe_df[base_methods].max(axis=1)
        sortino_beats_all = sortino_df["model_switching"] > sortino_df[base_methods].max(axis=1)
        combined_beats_all = sharpe_beats_all & sortino_beats_all
        
        results["combined_analysis"] = {
            "periods_beat_all_sharpe": sharpe_beats_all.sum(),
            "periods_beat_all_sortino": sortino_beats_all.sum(), 
            "periods_beat_all_combined": combined_beats_all.sum(),
            "total_periods": len(periods),
            "combined_dominance_pct": (combined_beats_all.sum() / len(periods)) * 100
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing method performance: {e}")
        return {}