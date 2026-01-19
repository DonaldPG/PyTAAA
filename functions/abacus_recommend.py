#!/usr/bin/env python3

"""Recommendation engine for Abacus model-switching trading system.

This module provides utilities for generating model recommendations based on
the model-switching methodology. It includes date helpers, recommendation
logic, and display functions for manual trading decisions.
"""

import pandas as pd
import numpy as np
from datetime import date as date_type, datetime
from typing import List, Tuple, Optional, Dict, Any
import logging

from functions.logger_config import get_logger

# Get module-specific logger
logger = get_logger(__name__, log_file='abacus_recommend.log')


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
                
                plot_text += f"  Model ranks (Normalized Score):\n"
                for i, (model, score) in enumerate(rankings, 1):
                    plot_text += f"    {i}. {model:<12} {score:>6.3f}\n"
        
        return plot_text

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
            print(f"  Model rankings (Normalized Score):")
            for i, (model, score) in enumerate(rankings, 1):
                print(f"    {i}. {model:<12} {score:>6.3f}")
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Generated recommendations for manual portfolio updates")
        print(f"Based on model-switching trading system methodology")
        
        # Generate plot text
        return self.generate_recommendation_text(dates, target_date, first_weekday)

