#!/usr/bin/env python3

"""Monte Carlo backtesting runner for PyTAAA.

This module runs Monte Carlo simulations using actual portfolio values from 
PyTAAA_status.params files and/or backtested portfolio values from 
pyTAAAweb_backtestPortfolioValue.params files to evaluate portfolio performance.
"""

import os
import time
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
import json
import click

from functions.MonteCarloBacktest import MonteCarloBacktest
from functions.logger_config import get_logger

# Get module-specific logger
logger = get_logger(__name__, log_file='monte_carlo_run.log')


def print_progress(i: int, total: int, start_time: float) -> None:
    """Print progress with estimated time remaining."""
    elapsed = time.time() - start_time
    if i > 0:
        est_total = elapsed * total / i
        remaining = est_total - elapsed
        remaining_min = int(remaining // 60)
        remaining_sec = int(remaining % 60)
        print(f"Running iteration {i+1}/{total}... "
              f"Est. remaining: {remaining_min}m {remaining_sec}s", end="\r")


@click.command()
@click.option(
    '--search', 
    type=click.Choice(['explore-exploit', 'explore', 'exploit']), 
    default='explore-exploit',
    help='Search strategy: explore-exploit (default, dynamic), explore (pure exploration), exploit (pure exploitation)'
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Show detailed normalized score breakdown for each new best performance'
)
@click.option(
    '--json', 'json_config_path', 
    default=None,
    help='Path to JSON configuration file for centralized settings'
)
@click.option(
    '--randomize',
    is_flag=True,
    default=False,
    help='Use randomized values for CENTRAL_VALUES and STD_VALUES (default: False)'
)
def main(search: str, verbose: bool, json_config_path: Optional[str], randomize: bool) -> None:
    """Run Monte Carlo backtesting with actual or backtested portfolio values.
    
    Args:
        search: Search strategy to use for parameter exploration
        verbose: Whether to show detailed normalized score breakdown
        json_config_path: Path to JSON configuration file
        randomize: Whether to use randomized normalization values
    """
    
    try:
        logger.info("Starting Monte Carlo backtesting process")
        logger.info(f"Search strategy: {search}")
        print("Starting Monte Carlo backtesting process")
        print(f"Search strategy: {search}")
        
        # Determine configuration source and load config
        if json_config_path:
            print(f"Using JSON configuration: {json_config_path}")
            config_path = json_config_path
            # Use JSON configuration functions for centralized values
            from functions.GetParams import get_web_output_dir, get_central_std_values
            try:
                web_output_dir = get_web_output_dir(json_config_path)
            except (KeyError, FileNotFoundError):
                print("Warning: Could not load web output directory from JSON, using default")
                web_output_dir = None
            try:
                normalization_values = get_central_std_values(json_config_path)
            except (KeyError, FileNotFoundError):
                print("Warning: Could not load normalization values from JSON, using defaults")
                normalization_values = None
        else:
            # Use legacy configuration
            config_path = 'pytaaa_model_switching_params.json'
            web_output_dir = None
            normalization_values = None
        
        # Load monte carlo settings from config
        with open(config_path, 'r') as f:
            config = json.load(f)
        monte_carlo_config = config.get('monte_carlo', {})
        iterations = monte_carlo_config.get('max_iterations', 50000)
        min_iterations_for_exploit = monte_carlo_config.get('min_iterations_for_exploit', 50)
        trading_frequency = monte_carlo_config.get('trading_frequency', 'monthly')
        min_lookback = monte_carlo_config.get('min_lookback', 10)
        max_lookback = monte_carlo_config.get('max_lookback', 252)
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
            }

        print("\nInitializing Monte Carlo backtesting...")
        print(f"Using {'actual' if data_format == 'actual' else 'backtested'} portfolio values")
        print(f"Reading from: {data_files[data_format]}")
        start_time = time.time()
        
        # Initialize Monte Carlo backtesting with config settings and search mode
        monte_carlo = MonteCarloBacktest(
            model_paths=model_choices,
            iterations=iterations,
            min_iterations_for_exploit=min_iterations_for_exploit,
            trading_frequency=trading_frequency,
            min_lookback=min_lookback,
            max_lookback=max_lookback,
            search_mode=search,
            verbose=verbose,
            json_config=config
        )
        
        # Apply JSON normalization values if available by updating class constants
        if not randomize and normalization_values:
            monte_carlo.CENTRAL_VALUES = normalization_values['central_values']
            monte_carlo.STD_VALUES = normalization_values['std_values']
            print(f"Applied JSON normalization values to Monte Carlo instance")
        else:
            monte_carlo.CENTRAL_VALUES = {
                'annual_return': np.random.choice([0.425, 0.435, 0.445, 0.455, 0.465]),
                'sharpe_ratio': np.random.choice([1.35, 1.45]),
                'sortino_ratio': np.random.choice([1.30, 1.35, 1.40, 1.45]),
                'max_drawdown': np.random.choice([-0.58, -0.56, -0.54]),
                'avg_drawdown': np.random.choice([-0.125, -0.120, -0.115, -0.110, -0.105])
            }
            monte_carlo.STD_VALUES = {
                'annual_return': np.random.choice([0.020/3, 0.020, 0.030, 0.040, 0.045, 0.050, 0.060]),
                'sharpe_ratio': np.random.choice([0.135/3,0.135, 0.150, 0.165, 0.180, 0.200]),
                'sortino_ratio': np.random.choice([0.120/3, 0.120, 0.140, 0.160]),
                'max_drawdown': np.random.choice([0.05/3, 0.05, 0.06, 0.07]) * 4.5,
                'avg_drawdown': np.random.choice([0.010/3, 0.010, 0.013, 0.016, 0.019]) * 4.5
            }


        # Load previous state if available
        state_file = "monte_carlo_state.pkl"
        print(f"\nChecking for previous state in {state_file}...")
        monte_carlo.load_state(state_file)
        
        logger.info(f"Running Monte Carlo simulations with {iterations} iterations...")
        print(f"\nStarting Monte Carlo simulations with {iterations} iterations...")
        print("Note: This may take about 5 minutes to complete...")
        print("Press Ctrl+C to stop early with current best result\n")
        
        try:
            top_models = monte_carlo.run()
            print(f"\nMonte Carlo simulation completed in {(time.time() - start_time)/60:.1f} minutes")
            
            # Save state after successful completion
            print(f"\nSaving exploration/exploitation state to {state_file}...")
            monte_carlo.save_state(state_file)
            
        except KeyboardInterrupt:
            print("\n\nStopped early by user")
            # Save state even if interrupted
            print(f"Saving current state to {state_file}...")
            try:
                monte_carlo.save_state(state_file)
                print("State saved successfully")
            except Exception as e:
                print(f"Failed to save state: {e}")
            return
        
        # Create and save the final plot using the unified plotting function
        print("\nCreating final performance plot...")
        
        # Create timestamp for filename (nearest minute)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        
        # Determine output path - use JSON web_output_dir if available
        if web_output_dir:
            # Create pngs subdirectory in web output directory
            pngs_dir = os.path.join(web_output_dir, "pngs")
            os.makedirs(pngs_dir, exist_ok=True)
            output_path = os.path.join(pngs_dir, f"monte_carlo_best_performance_{timestamp}.png")
            print(f"Using JSON web output directory: {web_output_dir}")
        else:
            # Create pngs subdirectory in current directory for legacy path
            pngs_dir = "pngs"
            os.makedirs(pngs_dir, exist_ok=True)
            output_path = os.path.join(pngs_dir, f"monte_carlo_best_performance_{timestamp}.png")

    

        # Use consistent portfolio calculation for both stdout and plot
        if monte_carlo.best_portfolio_value is not None:
            # Recalculate using best parameters for consistency with plot
            best_lookbacks = monte_carlo.best_params.get('lookbacks', [50, 150, 250])
            consistent_portfolio = monte_carlo._calculate_model_switching_portfolio(best_lookbacks)
            final_metrics = monte_carlo.compute_performance_metrics(consistent_portfolio)
            monte_carlo.create_monte_carlo_plot(consistent_portfolio, final_metrics, output_path)
        else:
            print("Warning: No best portfolio found to plot")
        
        # Print final statistics
        print("\nMonte Carlo Backtesting Results:")
        print("-" * 40)
        print(f"Total iterations: {iterations}")
        print(f"Analyzed {len(monte_carlo.dates)} trading days")
        print(f"Date range: {monte_carlo.dates[0]} to {monte_carlo.dates[-1]}")

        # Print best parameters
        print("\nBest performing configuration:")
        lookbacks = monte_carlo.best_params.get('lookbacks', [])
        if isinstance(lookbacks, list) and lookbacks:
            print(f"Lookback periods: {sorted(lookbacks)} days")
        else:
            print("No lookback periods found")

        print("\nModel Selection Analysis:")
        print("-" * 40)
        model_counts = {}
        total_selections = 0

        # Count total selections first
        for date, model in monte_carlo.best_model_selections.items():
            model_counts[model] = model_counts.get(model, 0) + 1
            total_selections += 1

        # Print model selection counts and percentages
        for i, (model, count) in enumerate(
            sorted(model_counts.items(), 
                  key=lambda x: x[1], 
                  reverse=True), 1
        ):
            percentage = (count / total_selections) * 100
            print(f"{i}. {model}: {count} selections ({percentage:.1f}%)")

        print("\nPortfolio Statistics:")
        print("-" * 40)
        initial_value = monte_carlo.best_portfolio_value[0]
        final_value = monte_carlo.best_portfolio_value[-1]
        
        # Calculate annualized return instead of total return
        num_days = len(monte_carlo.best_portfolio_value)
        years = num_days / 252  # 252 trading days per year
        annualized_return = ((final_value / initial_value) ** (1 / years) - 1) * 100
        
        print(f"Initial Value: ${initial_value:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Average Annualized Gain: {annualized_return:.1f}%")
        print("-" * 40)
        print(f"Performance plot saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Monte Carlo backtesting failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()