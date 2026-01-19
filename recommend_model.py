#!/usr/bin/env python3

"""Model recommendation system for PyTAAA trading strategy.

This module generates model recommendations for manual trading decisions
based on the model-switching methodology. It provides recommendations
for both the current date and the first weekday of the current month.
"""

import os
import json
import click
import pickle
import pandas as pd
import numpy as np
from datetime import date as date_type, datetime
from typing import List, Tuple, Optional, Dict, Any
import logging

from functions.MonteCarloBacktest import MonteCarloBacktest
from functions.logger_config import get_logger
from functions.abacus_recommend import DateHelper, ModelRecommender, RecommendationDisplay
from functions.abacus_backtest import BacktestDataLoader

# Get module-specific logger
logger = get_logger(__name__, log_file='recommend_model.log')


def load_best_params_from_saved_state() -> Optional[List[int]]:
    """Load best parameters from saved Monte Carlo state.
    
    Returns:
        List of lookback periods from saved state, or None if unavailable
    """
    state_file = "monte_carlo_state.pkl"
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


def get_recommendation_lookbacks(lookbacks_arg: Optional[str], config: Dict[str, Any]) -> List[int]:
    """Get lookbacks from user input, saved state, or config defaults.
    
    Args:
        lookbacks_arg: User-provided lookback argument
        config: Configuration dictionary
        
    Returns:
        List of lookback periods to use
    """
    if lookbacks_arg == "use-saved":
        # Try to load from monte_carlo_state.pkl
        saved_lookbacks = load_best_params_from_saved_state()
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
            raise click.BadParameter(f"Invalid lookbacks format: {e}")
    else:
        # Use defaults from JSON config
        return config['recommendation_mode']['default_lookbacks']


# Removed: generate_recommendation_output() - replaced with ModelRecommender class
# Removed: display_parameters_info() - replaced with RecommendationDisplay class


@click.command()
@click.option('--date', default=None, 
              help='Specific date for recommendation (YYYY-MM-DD), defaults to today')
@click.option('--lookbacks', default=None, 
              help='Lookback periods: comma-separated values like "25,50,100" or "use-saved" for Monte Carlo results')
@click.option('--json', 'json_config_path', default=None,
              help='Path to JSON configuration file for centralized settings')
def main(date: Optional[str], lookbacks: Optional[str], json_config_path: Optional[str]) -> None:
    """Generate model recommendation for specified date or today.
    
    This tool generates model recommendations for manual trading decisions
    based on the model-switching methodology. It always shows recommendations
    for both the target date and the first weekday of the current month.
    """
    
    try:
        logger.info("Starting model recommendation generation")
        print("Starting model recommendation generation...")
        
        # Determine configuration source and load config
        if json_config_path:
            print(f"Using JSON configuration: {json_config_path}")
            config_path = json_config_path
            # Use JSON configuration functions for centralized values
            from functions.GetParams import get_web_output_dir, get_central_std_values
            web_output_dir = get_web_output_dir(json_config_path)
            normalization_values = get_central_std_values(json_config_path)
        else:
            # Use legacy configuration
            config_path = 'pytaaa_model_switching_params.json'
            web_output_dir = None
            normalization_values = None
            
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Ensure model_selection section exists and provide sensible defaults
        if 'model_selection' not in config:
            config['model_selection'] = {}

        # Ensure performance_metrics exists; if missing, fall back to defaults
        if 'performance_metrics' not in config['model_selection']:
            logger.warning("Missing 'performance_metrics' in JSON; using defaults")
            config['model_selection']['performance_metrics'] = {
                'sharpe_ratio_weight': 1.0,
                'sortino_ratio_weight': 1.0,
                'max_drawdown_weight': 1.0,
                'avg_drawdown_weight': 1.0,
                'annualized_return_weight': 1.0
            }

        # Validate required performance metric weights (after defaults applied)
        required_weights = [
            'sharpe_ratio_weight',
            'sortino_ratio_weight',
            'max_drawdown_weight',
            'avg_drawdown_weight',
            'annualized_return_weight'
        ]
        performance_metrics = config['model_selection']['performance_metrics']
        missing_weights = [w for w in required_weights if w not in performance_metrics]
        if missing_weights:
            raise ValueError(f"Missing required performance metric weights in JSON configuration: {', '.join(missing_weights)}")

        # Ensure metric_blending section exists; if missing, provide defaults
        if 'metric_blending' not in config:
            logger.warning("Missing 'metric_blending' in JSON; using defaults")
            config['metric_blending'] = {
                'enabled': False,
                'full_period_weight': 1.0,
                'focus_period_weight': 0.0
            }
        metric_blending = config['metric_blending']
        
        # Ensure recommendation_mode section exists
        if 'recommendation_mode' not in config:
            config['recommendation_mode'] = {
                'default_lookbacks': [50, 150, 250],
                'output_format': 'both',
                'generate_plot': True,
                'show_model_ranks': True
            }
        
        # Get recommendation parameters
        dates, target_date, first_weekday = DateHelper.get_recommendation_dates(date)
        recommendation_lookbacks = get_recommendation_lookbacks(lookbacks, config)
        
        print(f"Target date: {target_date.strftime('%Y-%m-%d')}")
        if first_weekday and first_weekday != target_date:
            print(f"First weekday of month: {first_weekday.strftime('%Y-%m-%d')}")
        print(f"Using lookback periods: {recommendation_lookbacks}")
        
        # Load monte carlo settings
        monte_carlo_config = config.get('monte_carlo', {})
        data_format = monte_carlo_config.get('data_format', 'actual')
        
        # Configure model paths using BacktestDataLoader
        loader = BacktestDataLoader(config)
        model_choices = loader.build_model_paths(data_format, json_config_path)
        model_choices = loader.validate_model_paths(model_choices)
        
        print(f"\nInitializing model recommendation system...")
        print(f"Using {'actual' if data_format == 'actual' else 'backtested'} portfolio values")
        
        # Initialize Monte Carlo backtesting in recommendation mode
        monte_carlo = MonteCarloBacktest(
            model_paths=model_choices,
            iterations=1,  # Single iteration for recommendation
            trading_frequency=monte_carlo_config.get('trading_frequency', 'monthly'),
            min_lookback=monte_carlo_config.get('min_lookback', 10),
            max_lookback=monte_carlo_config.get('max_lookback', 400),
            search_mode='exploit',  # Use exploitation for recommendations
            json_config=config
        )
        
        # Apply JSON normalization values if available by updating class constants
        if normalization_values:
            monte_carlo.CENTRAL_VALUES = normalization_values['central_values']
            monte_carlo.STD_VALUES = normalization_values['std_values']
            print(f"Applied JSON normalization values to Monte Carlo instance")
        
        # Load previous state if available for context
        state_file = "monte_carlo_state.pkl"
        if os.path.exists(state_file):
            print(f"Loading previous Monte Carlo state from {state_file}...")
            monte_carlo.load_state(state_file)
        
        # Generate recommendations using ModelRecommender
        recommender = ModelRecommender(monte_carlo, recommendation_lookbacks)
        plot_text = recommender.display_recommendations(
            dates, target_date, first_weekday
        )
        
        # Calculate model-switching portfolio for parameters display
        model_switching_portfolio = monte_carlo._calculate_model_switching_portfolio(recommendation_lookbacks)
        
        # Display detailed parameter information using RecommendationDisplay
        display = RecommendationDisplay(monte_carlo)
        display.display_parameters_summary(recommendation_lookbacks, model_switching_portfolio)
        
        # Generate plot if requested
        if config['recommendation_mode'].get('generate_plot', True):
            print(f"\nGenerating recommendation plot...")
            
            # Determine output path - use JSON web_output_dir if available
            if web_output_dir:
                os.makedirs(web_output_dir, exist_ok=True)
                output_path = os.path.join(web_output_dir, "recommendation_plot.png")
                print(f"Saving plot to: {output_path}")
            else:
                output_path = "recommendation_plot.png"
                print(f"Saving plot to: {output_path}")
            
            # For recommendation mode, create a plot using existing portfolio data
            try:
                # Use the best performing model or cash as baseline for plotting
                if monte_carlo.portfolio_histories:
                    # Calculate model-switching portfolio using recommendation lookbacks
                    model_switching_portfolio = monte_carlo._calculate_model_switching_portfolio(recommendation_lookbacks)
                    sample_metrics = monte_carlo.compute_performance_metrics(model_switching_portfolio)
                    
                    # Set best_params for plotting if not already set
                    if not hasattr(monte_carlo, 'best_params') or not monte_carlo.best_params:
                        monte_carlo.best_params = {'lookbacks': recommendation_lookbacks}
                    
                    # Initialize best_model_selections for plotting
                    if not hasattr(monte_carlo, 'best_model_selections'):
                        monte_carlo.best_model_selections = {}
                    
                    # Generate custom text for recommendation plot using ModelRecommender
                    recommender = ModelRecommender(monte_carlo, recommendation_lookbacks)
                    recommendation_text = recommender.generate_recommendation_text(
                        dates, target_date, first_weekday
                    )
                    
                    monte_carlo.create_monte_carlo_plot(
                        model_switching_portfolio,
                        sample_metrics, 
                        output_path,
                        custom_text=recommendation_text
                    )
                else:
                    print("No portfolio data available for plotting")
            except Exception as e:
                logger.warning(f"Could not generate plot: {str(e)}")
                print(f"Plot generation skipped: {str(e)}")
        
        logger.info("Model recommendation generation completed successfully")
        print("Model recommendation generation completed successfully")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        print(f"Error: {str(e)}")
        print("Please ensure all required files are present.")
        
    except Exception as e:
        logger.error(f"Model recommendation failed: {str(e)}", exc_info=True)
        print(f"Error generating recommendations: {str(e)}")
        raise


if __name__ == "__main__":
    main()