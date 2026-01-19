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
from functions.abacus_recommend import (
    DateHelper, ModelRecommender, RecommendationDisplay, 
    ConfigurationHelper, PlotGenerator
)
from functions.abacus_backtest import BacktestDataLoader

# Get module-specific logger
logger = get_logger(__name__, log_file='recommend_model.log')


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
        
        # Ensure config has required sections with defaults
        ConfigurationHelper.ensure_config_defaults(config)
        
        # Get recommendation parameters
        dates, target_date, first_weekday = DateHelper.get_recommendation_dates(date)
        recommendation_lookbacks = ConfigurationHelper.get_recommendation_lookbacks(
            lookbacks, config
        )
        
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
            
            # Determine output path
            if web_output_dir:
                os.makedirs(web_output_dir, exist_ok=True)
                output_path = os.path.join(web_output_dir, "recommendation_plot.png")
            else:
                output_path = "recommendation_plot.png"
            print(f"Saving plot to: {output_path}")
            
            # Generate plot
            plot_gen = PlotGenerator(monte_carlo, config)
            plot_gen.generate_recommendation_plot(
                recommendation_lookbacks, dates, target_date, first_weekday, output_path
            )
        
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