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

# Get module-specific logger
logger = get_logger(__name__, log_file='recommend_model.log')


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
    first_weekday = get_first_weekday_of_month(target_date)
    
    # Add first weekday if different from target date
    if first_weekday and first_weekday != target_date:
        dates.append(first_weekday)
    
    return dates, target_date, first_weekday


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


def generate_recommendation_output(
        monte_carlo: MonteCarloBacktest, 
        dates: List[date_type], 
        lookbacks: List[int],
        target_date: date_type,
        first_weekday: Optional[date_type]
    ) -> str:
    """Generate and display model recommendations for specified dates.
    
    Args:
        monte_carlo: Monte Carlo backtesting instance
        dates: List of dates to generate recommendations for
        lookbacks: Lookback periods to use
        target_date: The main target date
        first_weekday: First weekday of the month (if different)
        
    Returns:
        Formatted recommendation text for plot display
    """
    print("\n" + "="*60)
    print("MODEL RECOMMENDATION RESULTS")
    print("="*60)
    
    print(f"\nRecommendation Parameters:")
    print(f"  Lookback periods: {lookbacks} days")
    print(f"  Target date: {target_date.strftime('%Y-%m-%d (%A)')}")
    if first_weekday and first_weekday != target_date:
        print(f"  First weekday of month: {first_weekday.strftime('%Y-%m-%d (%A)')}")
    
    # Find closest available dates in data
    available_dates = [d for d in monte_carlo.dates if isinstance(d, date_type)]
    
    # Build text for plot (matching plot format)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    plot_text = f"{current_time}\n"
    plot_text += f"Model Recommendation Analysis\n"
    plot_text += f"Lookback periods: {sorted(lookbacks)} days\n"
    plot_text += f"Target date: {target_date.strftime('%Y-%m-%d (%A)')}\n"
    
    if first_weekday and first_weekday != target_date:
        plot_text += f"First weekday: {first_weekday.strftime('%Y-%m-%d (%A)')}\n"
    
    # Calculate model-switching portfolio metrics
    model_switching_portfolio = monte_carlo._calculate_model_switching_portfolio(lookbacks)
    sample_metrics = monte_carlo.compute_performance_metrics(model_switching_portfolio)
    
    plot_text += f"\nModel-switching portfolio:\n"
    plot_text += f"  Final Value: ${sample_metrics['final_value']:,.0f}\n"
    plot_text += f"  Annual Return: {sample_metrics['annual_return']:.1f}%\n"
    plot_text += f"  Sharpe Ratio: {sample_metrics['sharpe_ratio']:.2f}\n"
    plot_text += f"  Normalized Score: {sample_metrics['normalized_score']:.3f}\n"
    
    for recommendation_date in dates:
        print(f"\n{'-'*40}")
        print(f"Recommendation for {recommendation_date.strftime('%Y-%m-%d')}:")
        print(f"{'-'*40}")
        
        # Find closest available date
        closest_date = None
        min_diff = float('inf')
        
        for available_date in available_dates:
            diff = abs((available_date - recommendation_date).days)
            if diff < min_diff:
                min_diff = diff
                closest_date = available_date
        
        if closest_date is None:
            print("No data available for this date")
            plot_text += f"\nRecommendation for {recommendation_date.strftime('%Y-%m-%d')}: No data\n"
            continue
            
        if min_diff > 0:
            print(f"  Using closest available date: {closest_date.strftime('%Y-%m-%d')} ({min_diff} days difference)")
        
        try:
            # Find the index for this date
            date_idx = monte_carlo.dates.index(closest_date)
            
            if date_idx > max(lookbacks):
                # Generate model recommendation for this date
                best_model, used_lookbacks = monte_carlo._select_best_model(
                    date_idx, lookbacks=lookbacks
                )
                
                print(f"  Best model: {best_model}")
                
                # Add to plot text (matching plot format)
                plot_text += f"\nRecommendation for {recommendation_date.strftime('%Y-%m-%d')}:\n"
                if min_diff > 0:
                    plot_text += f"  (Using {closest_date.strftime('%Y-%m-%d')}, {min_diff}d diff)\n"
                plot_text += f"  Best model: {best_model}\n"
                
                # Get model rankings for detailed display
                # FIXED: Force consistent alphabetical ordering of models
                models = sorted(list(monte_carlo.portfolio_histories.keys()))
                lookback_period = max(used_lookbacks)
                start_idx = max(0, date_idx - lookback_period)
                start_date = pd.Timestamp(monte_carlo.dates[start_idx])
                end_date = pd.Timestamp(monte_carlo.dates[date_idx - 1])
                
                # Calculate normalized scores for ranking
                normalized_scores = []
                for model in models:
                    if model == "cash":
                        portfolio_vals = np.ones(lookback_period) * 10000.0
                    else:
                        portfolio_vals = monte_carlo.portfolio_histories[model][start_date:end_date].values
                        if len(portfolio_vals) == 0:
                            portfolio_vals = np.ones(lookback_period) * 10000.0
                    
                    normalized_score = monte_carlo.compute_performance_metrics(portfolio_vals)['normalized_score']
                    normalized_scores.append(normalized_score)
                
                model_ranking = sorted(zip(models, normalized_scores), key=lambda x: x[1], reverse=True)
                
                print(f"  Model rankings (Normalized Score):")
                plot_text += f"  Model ranks (Normalized Score):\n"
                for i, (model, score) in enumerate(model_ranking, 1):
                    print(f"    {i}. {model:<12} {score:>6.3f}")
                    plot_text += f"    {i}. {model:<12} {score:>6.3f}\n"
                
            else:
                print(f"  Insufficient data (need at least {max(lookbacks)} days)")
                plot_text += f"  Insufficient data\n"
                
        except Exception as e:
            logger.error(f"Error generating recommendation for {recommendation_date}: {str(e)}")
            print(f"  Error: {str(e)}")
            plot_text += f"  Error calculating\n"
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Generated recommendations for manual portfolio updates")
    print(f"Based on model-switching trading system methodology")
    
    return plot_text


def display_parameters_info(monte_carlo: MonteCarloBacktest, 
                          lookbacks: List[int],
                          model_switching_portfolio: np.ndarray) -> None:
    """Display detailed parameter information to stdout.
    
    Args:
        monte_carlo: Monte Carlo backtesting instance
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
    
    central_values = monte_carlo.CENTRAL_VALUES
    std_values = monte_carlo.STD_VALUES
    
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
    effectiveness = analyze_model_switching_effectiveness(monte_carlo, lookbacks)
    
    print(f"\nModel-Switching Effectiveness:")
    print(f"  Outperformance Rate  : {effectiveness['sharpe_outperformance_pct']:>11.1f}%")
    print(f"  vs Equal-Weight Base : {effectiveness.get('avg_excess_return', 0.0)*100:>11.1f}% excess annual return")
    
    print("="*70)


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
        dates, target_date, first_weekday = get_recommendation_dates(date)
        recommendation_lookbacks = get_recommendation_lookbacks(lookbacks, config)
        
        print(f"Target date: {target_date.strftime('%Y-%m-%d')}")
        if first_weekday and first_weekday != target_date:
            print(f"First weekday of month: {first_weekday.strftime('%Y-%m-%d')}")
        print(f"Using lookback periods: {recommendation_lookbacks}")
        
        # Load monte carlo settings
        monte_carlo_config = config.get('monte_carlo', {})
        data_format = monte_carlo_config.get('data_format', 'actual')
        data_files = monte_carlo_config.get('data_files', {
            'actual': 'PyTAAA_status.params',
            'backtested': 'pyTAAAweb_backtestPortfolioValue.params'
        })
        
        # Configure model paths - use JSON config if available
        if json_config_path and 'models' in config:
            # Use JSON configuration for model paths
            models_config = config['models']
            base_folder = models_config.get('base_folder', '/Users/donaldpg/pyTAAA_data')
            model_choices = {}
            
            for model_name, path_template in models_config.get('model_choices', {}).items():
                if path_template == "":  # Cash model
                    model_choices[model_name] = ""
                else:
                    # Replace placeholders in path template
                    data_file = data_files[data_format]
                    model_path = path_template.format(
                        base_folder=base_folder,
                        data_file=data_file
                    )
                    model_choices[model_name] = model_path
        else:
            # Use legacy hard-coded paths
            base_folder = "/Users/donaldpg/pyTAAA_data"
            model_choices = {
                "cash": "",
                "naz100_pine": f"{base_folder}/naz100_pine/data_store/{data_files[data_format]}",
                "naz100_hma": f"{base_folder}/naz100_hma/data_store/{data_files[data_format]}",
                "naz100_pi": f"{base_folder}/naz100_pi/data_store/{data_files[data_format]}",
                "sp500_hma": f"{base_folder}/sp500_hma/data_store/{data_files[data_format]}",
                "sp500_pine": f"{base_folder}/sp500_pine/data_store/{data_files[data_format]}",
            }

        # Validate resolved model paths similar to run_monte_carlo: log warnings for
        # missing data files but keep the mapping so downstream code can handle them.
        validated_model_choices = {}
        for mname, mtemplate in model_choices.items():
            if not mtemplate:
                validated_model_choices[mname] = ""
                continue

            resolved_path = os.path.expanduser(mtemplate)
            resolved_path = os.path.abspath(resolved_path)

            if not os.path.exists(resolved_path):
                logger.warning(f"Model data file not found for {mname}: {resolved_path}. Keeping mapping; MonteCarloBacktest will handle missing data.")
                print(f"WARNING: Model data file not found for {mname}: {resolved_path}. Keeping mapping; MonteCarloBacktest will handle missing data.")

            validated_model_choices[mname] = resolved_path

        model_choices = validated_model_choices
        
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
        
        # Generate recommendations
        plot_text = generate_recommendation_output(
            monte_carlo, dates, recommendation_lookbacks, 
            target_date, first_weekday
        )
        
        # Calculate model-switching portfolio for parameters display
        model_switching_portfolio = monte_carlo._calculate_model_switching_portfolio(recommendation_lookbacks)
        
        # Display detailed parameter information
        display_parameters_info(monte_carlo, recommendation_lookbacks, model_switching_portfolio)
        
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
                    
                    # Generate custom text for recommendation plot
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                    recommendation_text = (
                        f"{current_time}\n"
                        f"Model Recommendation Analysis\n"
                        f"Lookback periods: {sorted(recommendation_lookbacks)} days\n"
                        f"Target date: {target_date.strftime('%Y-%m-%d (%A)')}\n"
                    )
                    
                    if first_weekday and first_weekday != target_date:
                        recommendation_text += f"First weekday: {first_weekday.strftime('%Y-%m-%d (%A)')}\n"
                    
                    recommendation_text += (
                        f"\nModel-switching portfolio:\n"
                        f"  Final Value: ${sample_metrics['final_value']:,.0f}\n"
                        f"  Annual Return: {sample_metrics['annual_return']:.1f}%\n"
                        f"  Sharpe Ratio: {sample_metrics['sharpe_ratio']:.2f}\n"
                        f"  Normalized Score: {sample_metrics['normalized_score']:.3f}\n"
                    )
                    
                    # Add recommendation results from console output
                    available_dates = [d for d in monte_carlo.dates if isinstance(d, date_type)]
                    
                    for recommendation_date in dates:
                        # Find closest available date
                        closest_date = None
                        min_diff = float('inf')
                        
                        for available_date in available_dates:
                            diff = abs((available_date - recommendation_date).days)
                            if diff < min_diff:
                                min_diff = diff
                                closest_date = available_date
                        
                        if closest_date is not None:
                            try:
                                # Find the index for this date
                                date_idx = monte_carlo.dates.index(closest_date)
                                
                                if date_idx > max(recommendation_lookbacks):
                                    # Generate model recommendation for this date
                                    best_model, used_lookbacks = monte_carlo._select_best_model(
                                        date_idx, lookbacks=recommendation_lookbacks
                                    )
                                    
                                    recommendation_text += f"\nRecommendation for {recommendation_date.strftime('%Y-%m-%d')}:\n"
                                    if min_diff > 0:
                                        recommendation_text += f"  (Using {closest_date.strftime('%Y-%m-%d')}, {min_diff}d diff)\n"
                                    recommendation_text += f"  Best model: {best_model}\n"
                                    
                                    # Get model rankings for this date
                                    # FIXED: Force consistent alphabetical ordering of models
                                    models = sorted(list(monte_carlo.portfolio_histories.keys()))
                                    lookback_period = max(used_lookbacks)
                                    start_idx = max(0, date_idx - lookback_period)
                                    start_date = pd.Timestamp(monte_carlo.dates[start_idx])
                                    end_date = pd.Timestamp(monte_carlo.dates[date_idx - 1])
                                    
                                    # Calculate normalized scores for ranking
                                    normalized_scores = []
                                    for model in models:
                                        if model == "cash":
                                            portfolio_vals = np.ones(lookback_period) * 10000.0
                                        else:
                                            portfolio_vals = monte_carlo.portfolio_histories[model][start_date:end_date].values
                                            if len(portfolio_vals) == 0:
                                                portfolio_vals = np.ones(lookback_period) * 10000.0
                                        
                                        normalized_score = monte_carlo.compute_performance_metrics(portfolio_vals)['normalized_score']
                                        normalized_scores.append(normalized_score)
                                    
                                    model_ranking = sorted(zip(models, normalized_scores), key=lambda x: x[1], reverse=True)
                                    
                                    recommendation_text += f"  Model ranks (Normalized Score):\n"
                                    for i, (model, score) in enumerate(model_ranking, 1):
                                        recommendation_text += f"    {i}. {model:<12} {score:>6.3f}\n"
                                        
                            except Exception as e:
                                recommendation_text += f"\nRecommendation for {recommendation_date.strftime('%Y-%m-%d')}: Error\n"
                    
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