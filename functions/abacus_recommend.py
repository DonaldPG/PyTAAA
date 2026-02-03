#!/usr/bin/env python3

"""Recommendation engine for Abacus model-switching trading system.

This module provides utilities for generating model recommendations based on
the model-switching methodology. It includes date helpers, recommendation
logic, and display functions for manual trading decisions.
"""

import os
import pickle
import pandas as pd
import numpy as np
from datetime import date as date_type, datetime
from typing import List, Tuple, Optional, Dict, Any
import logging
import json

from functions.logger_config import get_logger

# Get module-specific logger
logger = get_logger(__name__, log_file='abacus_recommend.log')


class ConfigurationHelper:
    """Helper for loading and validating configuration."""
    
    @staticmethod
    def load_best_lookbacks_from_state(state_file: str = "monte_carlo_state.pkl") -> Optional[List[int]]:
        """Load best lookback parameters from saved Monte Carlo state.
        
        Args:
            state_file: Path to saved state pickle file
            
        Returns:
            List of lookback periods from saved state, or None if unavailable
        """
        if not os.path.exists(state_file):
            logger.warning(f"No saved state file found: {state_file}")
            return None
        
        try:
            with open(state_file, 'rb') as f:
                state = pickle.load(f)
            
            # Extract best lookbacks from saved state
            best_params = state.get('best_params', {})
            lookbacks = best_params.get('lookbacks', None)
            
            if lookbacks and isinstance(lookbacks, list):
                logger.info(f"Loaded best lookbacks from saved state: {sorted(lookbacks)}")
                return sorted(lookbacks)
            else:
                logger.warning("No valid lookbacks found in saved state")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load saved state: {str(e)}")
            return None
    
    @staticmethod
    def get_recommendation_lookbacks(
        lookbacks_arg: Optional[str], 
        config: Dict[str, Any]
    ) -> List[int]:
        """Get lookbacks from user input, saved state, or config defaults.
        
        Args:
            lookbacks_arg: User-provided lookback argument
            config: Configuration dictionary
            
        Returns:
            List of lookback periods to use
        """
        if lookbacks_arg == "use-saved":
            # Try to load from monte_carlo_state.pkl
            saved_lookbacks = ConfigurationHelper.load_best_lookbacks_from_state()
            if saved_lookbacks:
                return saved_lookbacks
            else:
                print("Warning: Could not load saved parameters, using config defaults")
                return config['recommendation_mode']['default_lookbacks']
                
        elif lookbacks_arg:
            # Parse user-provided lookbacks like "25,50,100"
            try:
                lookbacks = [int(x.strip()) for x in lookbacks_arg.split(',')]
                if not lookbacks:
                    raise ValueError("Empty lookbacks list")
                return sorted(lookbacks)
            except ValueError as e:
                raise ValueError(f"Invalid lookbacks format: {e}")
        else:
            # Use defaults from JSON config
            return config['recommendation_mode']['default_lookbacks']
    
    @staticmethod
    def ensure_config_defaults(config: Dict[str, Any]) -> None:
        """Ensure configuration has required sections with sensible defaults.
        
        Args:
            config: Configuration dictionary to validate and populate
        """
        # Ensure model_selection section exists
        if 'model_selection' not in config:
            config['model_selection'] = {}

        # Ensure performance_metrics exists
        if 'performance_metrics' not in config['model_selection']:
            logger.warning("Missing 'performance_metrics' in JSON; using defaults")
            config['model_selection']['performance_metrics'] = {
                'sharpe_ratio_weight': 1.0,
                'sortino_ratio_weight': 1.0,
                'max_drawdown_weight': 1.0,
                'avg_drawdown_weight': 1.0,
                'annualized_return_weight': 1.0
            }
        
        # Validate required performance metric weights
        required_weights = [
            'sharpe_ratio_weight', 'sortino_ratio_weight',
            'max_drawdown_weight', 'avg_drawdown_weight',
            'annualized_return_weight'
        ]
        performance_metrics = config['model_selection']['performance_metrics']
        missing_weights = [w for w in required_weights if w not in performance_metrics]
        if missing_weights:
            raise ValueError(
                f"Missing required performance metric weights: {', '.join(missing_weights)}"
            )

        # Ensure metric_blending section exists
        if 'metric_blending' not in config:
            logger.warning("Missing 'metric_blending' in JSON; using defaults")
            config['metric_blending'] = {
                'enabled': False,
                'full_period_weight': 1.0,
                'focus_period_weight': 0.0
            }
        
        # Ensure recommendation_mode section exists
        if 'recommendation_mode' not in config:
            config['recommendation_mode'] = {
                'default_lookbacks': [50, 150, 250],
                'output_format': 'both',
                'generate_plot': True,
                'show_model_ranks': True
            }


class RecommendationDisplay:
    """Display utilities for recommendation output and parameter summaries."""
    
    def __init__(self, monte_carlo):
        """Initialize display utility with Monte Carlo instance.
        
        Args:
            monte_carlo: MonteCarloBacktest instance with portfolio data
        """
        self.monte_carlo = monte_carlo
    
    def display_parameters_summary(
        self, 
        lookbacks: List[int],
        model_switching_portfolio: np.ndarray
    ) -> None:
        """Display detailed parameter information to stdout.
        
        Args:
            lookbacks: List of lookback periods used
            model_switching_portfolio: Portfolio values array
        """
        print("\n" + "="*70)
        print("PARAMETERS SUMMARY")
        print("="*70)
        
        # Display lookback periods
        print(f"\nLookback Periods:")
        print(f"  Values: {sorted(lookbacks)} days")
        print(f"  Count: {len(lookbacks)} periods")
        
        # Display combined normalization parameters table
        print(f"\nNormalization Parameters:")
        print(f"  {'Metric':<18} {'Central Value':<14} {'Std Deviation':<14}")
        print(f"  {'-'*18} {'-'*14} {'-'*14}")
        
        central_values = self.monte_carlo.CENTRAL_VALUES
        std_values = self.monte_carlo.STD_VALUES
        
        for metric in central_values.keys():
            central_val = central_values[metric]
            std_val = std_values[metric]
            
            metric_name = metric.replace('_', ' ').title()
            
            if metric in ['annual_return', 'max_drawdown', 'avg_drawdown']:
                print(f"  {metric_name:<18} {central_val:>13.1%} {std_val:>13.1%}")
            else:
                print(f"  {metric_name:<18} {central_val:>13.3f} {std_val:>13.3f}")
        
        # Display final portfolio value and key metrics
        print(f"\nFinal Portfolio Results:")
        final_value = model_switching_portfolio[-1]
        initial_value = model_switching_portfolio[0]
        
        print(f"  Initial Value        : ${initial_value:>12,.2f}")
        print(f"  Final Value          : ${final_value:>12,.2f}")
        
        # Calculate and display annualized return
        years = len(model_switching_portfolio) / 252  # 252 trading days per year
        if years > 0:
            annualized_return = ((final_value / initial_value) ** (1 / years)) - 1
            print(f"  Annualized Return    : {annualized_return:>11.1%}")
            print(f"  Time Period          : {years:>11.1f} years")
        
        from functions.PortfolioMetrics import analyze_model_switching_effectiveness

        # Add effectiveness analysis
        effectiveness = analyze_model_switching_effectiveness(
            self.monte_carlo, lookbacks
        )
        
        print(f"\nModel-Switching Effectiveness:")
        print(f"  Outperformance Rate  : {effectiveness['sharpe_outperformance_pct']:>11.1f}%")
        print(f"  vs Equal-Weight Base : {effectiveness.get('avg_excess_return', 0.0)*100:>11.1f}% excess annual return")
        
        print("="*70)


class PlotGenerator:
    """Generate recommendation plots with model-switching portfolio data."""
    
    def __init__(self, monte_carlo, config: Dict[str, Any]):
        """Initialize plot generator.
        
        Args:
            monte_carlo: MonteCarloBacktest instance
            config: Configuration dictionary
        """
        self.monte_carlo = monte_carlo
        self.config = config
    
    def generate_recommendation_plot(
        self,
        lookbacks: List[int],
        dates: List[date_type],
        target_date: date_type,
        first_weekday: Optional[date_type],
        output_path: str
    ) -> bool:
        """Generate and save recommendation plot.
        
        Args:
            lookbacks: Lookback periods used
            dates: Recommendation dates
            target_date: Target date
            first_weekday: First weekday of month (if different)
            output_path: Path to save plot
            
        Returns:
            True if plot generated successfully, False otherwise
        """
        try:
            if not self.monte_carlo.portfolio_histories:
                print("No portfolio data available for plotting")
                return False
            
            # Calculate model-switching portfolio
            model_switching_portfolio = self.monte_carlo._calculate_model_switching_portfolio(lookbacks)
            sample_metrics = self.monte_carlo.compute_performance_metrics(model_switching_portfolio)
            
            # Set best_params for plotting if not already set
            if not hasattr(self.monte_carlo, 'best_params') or not self.monte_carlo.best_params:
                self.monte_carlo.best_params = {'lookbacks': lookbacks}
            
            # Initialize best_model_selections for plotting
            if not hasattr(self.monte_carlo, 'best_model_selections'):
                self.monte_carlo.best_model_selections = {}
            
            # Generate custom text for plot
            recommender = ModelRecommender(self.monte_carlo, lookbacks)
            recommendation_text = recommender.generate_recommendation_text(
                dates, target_date, first_weekday
            )
            
            # Create plot
            self.monte_carlo.create_monte_carlo_plot(
                model_switching_portfolio,
                sample_metrics,
                output_path,
                custom_text=recommendation_text
            )
            return True
            
        except Exception as e:
            logger.warning(f"Could not generate plot: {str(e)}")
            print(f"Plot generation skipped: {str(e)}")
            return False



class DateHelper:
    """Utility class for date operations in recommendation system."""

    @staticmethod
    def get_first_weekday_of_month(target_date: date_type) -> Optional[date_type]:
        """Find the first weekday (Monday-Friday) of the given month.
        
        Args:
            target_date: Date to find the first weekday for its month
            
        Returns:
            First weekday of the month, or None if no weekday found
        """
        year, month = target_date.year, target_date.month
        
        for day in range(1, 32):  # Max days in any month
            try:
                test_date = date_type(year, month, day)
                if test_date.weekday() < 5:  # Monday=0 to Friday=4
                    return test_date
            except ValueError:
                break  # Invalid date (month ended)
        
        return None

    @staticmethod
    def get_recommendation_dates(target_date_str: Optional[str]) -> Tuple[List[date_type], date_type, Optional[date_type]]:
        """Get dates for recommendation: target date and first weekday of current month.
        
        Args:
            target_date_str: Optional date string in YYYY-MM-DD format
            
        Returns:
            Tuple of (dates_list, target_date, first_weekday)
        """
        dates = []
        
        # Handle target date (default to today)
        if target_date_str:
            target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
        else:
            target_date = date_type.today()
        dates.append(target_date)
        
        # Find first weekday of target date's month
        first_weekday = DateHelper.get_first_weekday_of_month(target_date)
        
        # Add first weekday if different from target date
        if first_weekday and first_weekday != target_date:
            dates.append(first_weekday)
        
        return dates, target_date, first_weekday

    @staticmethod
    def find_closest_trading_date(target_date: date_type, available_dates: List[date_type]) -> Tuple[Optional[date_type], int]:
        """Find the closest trading date to the target date.
        
        Args:
            target_date: The date to find a match for
            available_dates: List of available trading dates
            
        Returns:
            Tuple of (closest_date, days_difference)
            Returns (None, 0) if no dates available
        """
        if not available_dates:
            return None, 0
        
        closest_date = None
        min_diff = float('inf')
        
        for available_date in available_dates:
            diff = abs((available_date - target_date).days)
            if diff < min_diff:
                min_diff = diff
                closest_date = available_date
        
        return closest_date, int(min_diff)


class ModelRecommender:
    """Generates model recommendations based on historical performance."""

    def __init__(self, monte_carlo, lookbacks: List[int]):
        """Initialize ModelRecommender.
        
        Args:
            monte_carlo: MonteCarloBacktest instance with loaded portfolio data
            lookbacks: List of lookback periods to use for model selection
        """
        self.monte_carlo = monte_carlo
        self.lookbacks = lookbacks
        self.available_dates = [d for d in monte_carlo.dates if isinstance(d, date_type)]

    def get_recommendation_for_date(self, target_date: date_type) -> Tuple[Optional[str], List[Tuple[str, float]], int]:
        """Get model recommendation for a specific date.
        
        Args:
            target_date: Date to generate recommendation for
            
        Returns:
            Tuple of (best_model, model_rankings, days_difference)
            where model_rankings is [(model_name, normalized_score), ...]
            Returns (None, [], days_diff) if insufficient data
        """
        # Find closest available trading date
        closest_date, min_diff = DateHelper.find_closest_trading_date(
            target_date, self.available_dates
        )
        
        if closest_date is None:
            return None, [], 0
        
        try:
            # Find the index for this date
            date_idx = self.monte_carlo.dates.index(closest_date)
            
            if date_idx <= max(self.lookbacks):
                # Insufficient historical data
                return None, [], min_diff
            
            # Generate model recommendation for this date
            best_model, used_lookbacks = self.monte_carlo._select_best_model(
                date_idx, lookbacks=self.lookbacks
            )
            
            # Get model rankings
            model_rankings = self.rank_models_at_date(date_idx, used_lookbacks)
            
            return best_model, model_rankings, min_diff
            
        except Exception as e:
            logger.error(f"Error generating recommendation for {target_date}: {str(e)}")
            return None, [], min_diff

    def rank_models_at_date(self, date_idx: int, used_lookbacks: List[int]) -> List[Tuple[str, float]]:
        """Rank all models at a specific date index.
        
        Args:
            date_idx: Index into monte_carlo.dates
            used_lookbacks: Lookback periods used for this ranking
            
        Returns:
            Sorted list of (model_name, normalized_score) tuples, best first
        """
        # Force consistent alphabetical ordering of models
        models = sorted(list(self.monte_carlo.portfolio_histories.keys()))
        lookback_period = max(used_lookbacks)
        start_idx = max(0, date_idx - lookback_period)
        start_date = pd.Timestamp(self.monte_carlo.dates[start_idx])
        end_date = pd.Timestamp(self.monte_carlo.dates[date_idx - 1])
        
        # Calculate normalized scores for ranking
        normalized_scores = []
        for model in models:
            if model == "cash":
                portfolio_vals = np.ones(lookback_period) * 10000.0
            else:
                portfolio_vals = self.monte_carlo.portfolio_histories[model][start_date:end_date].values
                if len(portfolio_vals) == 0:
                    portfolio_vals = np.ones(lookback_period) * 10000.0
            
            normalized_score = self.monte_carlo.compute_performance_metrics(portfolio_vals)['normalized_score']
            normalized_scores.append(normalized_score)
        
        # Return sorted by score descending
        return sorted(zip(models, normalized_scores), key=lambda x: x[1], reverse=True)

    def display_recommendations(
        self,
        dates: List[date_type],
        target_date: date_type,
        first_weekday: Optional[date_type]
    ) -> str:
        """Display recommendations to console and return plot text.
        
        Args:
            dates: List of dates to generate recommendations for
            target_date: The main target date
            first_weekday: First weekday of the month (if different)
            
        Returns:
            Formatted recommendation text for plot display
        """
        print("\n" + "="*60)
        print("MODEL RECOMMENDATION RESULTS")
        print("="*60)
        
        print(f"\nRecommendation Parameters:")
        print(f"  Lookback periods: {self.lookbacks} days")
        print(f"  Target date: {target_date.strftime('%Y-%m-%d (%A)')}")
        if first_weekday and first_weekday != target_date:
            print(f"  First weekday of month: {first_weekday.strftime('%Y-%m-%d (%A)')}")
        
        # Process each date
        for recommendation_date in dates:
            print(f"\n{'-'*40}")
            print(f"Recommendation for {recommendation_date.strftime('%Y-%m-%d')}:")
            print(f"{'-'*40}")
            
            best_model, rankings, min_diff = self.get_recommendation_for_date(recommendation_date)
            
            if best_model is None:
                if not rankings:
                    print("No data available for this date")
                else:
                    print(f"  Insufficient data (need at least {max(self.lookbacks)} days)")
                continue
            
            if min_diff > 0:
                closest_date, _ = DateHelper.find_closest_trading_date(
                    recommendation_date, self.available_dates
                )
                print(f"  Using closest available date: {closest_date.strftime('%Y-%m-%d')} ({min_diff} days difference)")
            
            print(f"  Best model: {best_model}")
            # --- Print combined Multi-Rank and Normalized Score table ---
            try:
                date_idx = self.monte_carlo.dates.index(recommendation_date)
            except ValueError:
                closest_date, _ = DateHelper.find_closest_trading_date(recommendation_date, self.available_dates)
                date_idx = self.monte_carlo.dates.index(closest_date)
            lookbacks = self.lookbacks
            config_path = os.path.join(os.path.dirname(self.monte_carlo.model_paths[next(iter(self.monte_carlo.model_paths))]), 'pytaaa_model_switching_params.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            metric_weights = config['model_selection']['performance_metrics']
            models = sorted(list(self.monte_carlo.portfolio_histories.keys()))
            all_ranks = np.zeros((len(models), 5 * len(lookbacks)))
            for i, lookback_period in enumerate(lookbacks):
                start_idx = max(0, date_idx - lookback_period)
                start_date = pd.Timestamp(self.monte_carlo.dates[start_idx])
                end_date = pd.Timestamp(self.monte_carlo.dates[date_idx - 1])
                metrics_list = []
                for model in models:
                    if model == "cash":
                        portfolio_values = np.ones(lookback_period) * 10000.0
                    else:
                        portfolio_values = self.monte_carlo.portfolio_histories[model][start_date:end_date].values
                    metrics = self.monte_carlo.compute_performance_metrics(portfolio_values)
                    metrics_list.append([
                        metrics['sharpe_ratio'],
                        metrics['sortino_ratio'],
                        metrics['max_drawdown'],
                        metrics['avg_drawdown'],
                        metrics['annual_return']
                    ])
                metric_array = np.array(metrics_list).T
                period_ranks = np.zeros_like(metric_array)
                for m in range(5):
                    order = np.argsort(-metric_array[m])
                    ranks = np.empty_like(order)
                    ranks[order] = np.arange(len(order))
                    period_ranks[m] = ranks
                weights = np.array([
                    metric_weights.get('sharpe_ratio_weight', 1.0),
                    metric_weights.get('sortino_ratio_weight', 1.0),
                    metric_weights.get('max_drawdown_weight', 1.0),
                    metric_weights.get('avg_drawdown_weight', 1.0),
                    metric_weights.get('annualized_return_weight', 1.0)
                ])
                weighted_ranks = period_ranks * weights[:, np.newaxis]
                all_ranks[:, i*5:(i+1)*5] = weighted_ranks.T
            avg_ranks = np.mean(all_ranks, axis=1)
            # Get normalized scores in model order
            norm_score_dict = dict(rankings)
            print("Model     Multi  Normalized")
            print("rankings  Rank   Score")
            # Sort by multi-rank (lowest is best), then by normalized score
            sorted_models = sorted(models, key=lambda m: (avg_ranks[models.index(m)], -norm_score_dict.get(m, 0)))
            for idx, model in enumerate(sorted_models, 1):
                norm_score = norm_score_dict.get(model, 0)
                multi_rank = int(round(avg_ranks[models.index(model)]))
                # Format model name to full width (no truncation)
                print(f"  {idx}. {model:<15} {multi_rank:>3} {norm_score:>8.2f}")
        print(f"{'='*60}")
        print(f"Generated recommendations for manual portfolio updates")
        print(f"Based on model-switching trading system methodology")
        
        # Generate plot text
        return self.generate_recommendation_text(dates, target_date, first_weekday)

    def generate_recommendation_text(
        self,
        dates: List[date_type],
        target_date: date_type,
        first_weekday: Optional[date_type]
    ) -> str:
        """Generate formatted recommendation text for plot display.
        
        Args:
            dates: List of dates to generate recommendations for
            target_date: The main target date
            first_weekday: First weekday of the month (if different)
            
        Returns:
            Formatted text for plot overlay
        """
        # Calculate model-switching portfolio metrics
        model_switching_portfolio = self.monte_carlo._calculate_model_switching_portfolio(self.lookbacks)
        sample_metrics = self.monte_carlo.compute_performance_metrics(model_switching_portfolio)
        
        # Build text for plot
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        plot_text = f"{current_time}\n"
        plot_text += f"Model Recommendation Analysis\n"
        plot_text += f"Lookback periods: {sorted(self.lookbacks)} days\n"
        plot_text += f"Target date: {target_date.strftime('%Y-%m-%d (%A)')}\n"
        
        if first_weekday and first_weekday != target_date:
            plot_text += f"First weekday: {first_weekday.strftime('%Y-%m-%d (%A)')}\n"
        
        plot_text += f"\nModel-switching portfolio:\n"
        plot_text += f"  Final Value: ${sample_metrics['final_value']:,.0f}\n"
        plot_text += f"  Annual Return: {sample_metrics['annual_return']:.1f}%\n"
        plot_text += f"  Sharpe Ratio: {sample_metrics['sharpe_ratio']:.2f}\n"
        plot_text += f"  Normalized Score: {sample_metrics['normalized_score']:.3f}\n"
        
        # Add recommendations for each date
        for recommendation_date in dates:
            best_model, rankings, min_diff = self.get_recommendation_for_date(recommendation_date)
            
            if best_model is None:
                if not rankings:  # No data available
                    plot_text += f"\nRecommendation for {recommendation_date.strftime('%Y-%m-%d')}: No data\n"
                else:  # Insufficient historical data
                    plot_text += f"\nRecommendation for {recommendation_date.strftime('%Y-%m-%d')}: Insufficient data\n"
            else:
                plot_text += f"\nRecommendation for {recommendation_date.strftime('%Y-%m-%d')}:\n"
                if min_diff > 0:
                    closest_date, _ = DateHelper.find_closest_trading_date(recommendation_date, self.available_dates)
                    plot_text += f"  (Using {closest_date.strftime('%Y-%m-%d')}, {min_diff}d diff)\n"
                plot_text += f"  Best model: {best_model}\n"
                
                # Compute multi-ranks for this date
                try:
                    date_idx = self.monte_carlo.dates.index(recommendation_date)
                except ValueError:
                    closest_date, _ = DateHelper.find_closest_trading_date(recommendation_date, self.available_dates)
                    date_idx = self.monte_carlo.dates.index(closest_date)
                lookbacks = self.lookbacks
                config_path = os.path.join(os.path.dirname(self.monte_carlo.model_paths[next(iter(self.monte_carlo.model_paths))]), 'pytaaa_model_switching_params.json')
                with open(config_path, 'r') as f:
                    config = json.load(f)
                metric_weights = config['model_selection']['performance_metrics']
                models = sorted(list(self.monte_carlo.portfolio_histories.keys()))
                all_ranks = np.zeros((len(models), 5 * len(lookbacks)))
                for i, lookback_period in enumerate(lookbacks):
                    start_idx = max(0, date_idx - lookback_period)
                    start_date = pd.Timestamp(self.monte_carlo.dates[start_idx])
                    end_date = pd.Timestamp(self.monte_carlo.dates[date_idx - 1])
                    metrics_list = []
                    for model in models:
                        if model == "cash":
                            portfolio_values = np.ones(lookback_period) * 10000.0
                        else:
                            portfolio_values = self.monte_carlo.portfolio_histories[model][start_date:end_date].values
                        metrics = self.monte_carlo.compute_performance_metrics(portfolio_values)
                        metrics_list.append([
                            metrics['sharpe_ratio'],
                            metrics['sortino_ratio'],
                            metrics['max_drawdown'],
                            metrics['avg_drawdown'],
                            metrics['annual_return']
                        ])
                    metric_array = np.array(metrics_list).T
                    period_ranks = np.zeros_like(metric_array)
                    for m in range(5):
                        order = np.argsort(-metric_array[m])
                        ranks = np.empty_like(order)
                        ranks[order] = np.arange(len(order))
                        period_ranks[m] = ranks
                    weights = np.array([
                        metric_weights.get('sharpe_ratio_weight', 1.0),
                        metric_weights.get('sortino_ratio_weight', 1.0),
                        metric_weights.get('max_drawdown_weight', 1.0),
                        metric_weights.get('avg_drawdown_weight', 1.0),
                        metric_weights.get('annualized_return_weight', 1.0)
                    ])
                    weighted_ranks = period_ranks * weights[:, np.newaxis]
                    all_ranks[:, i*5:(i+1)*5] = weighted_ranks.T
                avg_ranks = np.mean(all_ranks, axis=1)
                # Get normalized scores in model order
                norm_score_dict = dict(rankings)
                
                plot_text += f"  Model        Multi  Normalized\n"
                plot_text += f"  Rankings     Rank   Score\n"
                sorted_models = sorted(models, key=lambda m: (-norm_score_dict.get(m, 0), avg_ranks[models.index(m)]))
                for idx, model in enumerate(sorted_models, 1):
                    norm_score = norm_score_dict.get(model, 0)
                    multi_rank = int(round(avg_ranks[models.index(model)]))
                    plot_text += f"  {idx}. {(model[:11]):<11}{multi_rank:>3} {norm_score:>7.2f}\n"
        
        return plot_text

