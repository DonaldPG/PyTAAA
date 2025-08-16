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
def main(search: str, verbose: bool) -> None:
    """Run Monte Carlo backtesting with actual or backtested portfolio values.
    
    Args:
        search: Search strategy to use for parameter exploration
        verbose: Whether to show detailed normalized score breakdown
    """
    
    try:
        logger.info("Starting Monte Carlo backtesting process")
        logger.info(f"Search strategy: {search}")
        print("Starting Monte Carlo backtesting process")
        print(f"Search strategy: {search}")
        
        # Load monte carlo settings from config
        with open('pytaaa_model_switching_params.json', 'r') as f:
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
        
        # Configure model paths with correct data_store locations
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
            verbose=verbose
        )
        
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
        output_path = "assets/model_switching_portfolio_performance.png"
        os.makedirs("assets", exist_ok=True)
        if os.path.exists(output_path):
            os.remove(output_path)
        
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