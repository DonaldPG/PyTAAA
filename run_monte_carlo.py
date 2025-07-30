#!/usr/bin/env python3

"""Monte Carlo backtesting runner for PyTAAA.

This module runs Monte Carlo simulations using actual portfolio values from 
PyTAAA_status.params files and/or backtested portfolio values from 
pyTAAAweb_backtestPortfolioValue.params files to evaluate portfolio performance.
"""

import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
import json

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

def main() -> None:
    """Run Monte Carlo backtesting with actual or backtested portfolio values."""
    try:
        logger.info("Starting Monte Carlo backtesting process")
        print("Starting Monte Carlo backtesting process")
        
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
        
        # Initialize Monte Carlo backtesting with config settings 
        monte_carlo = MonteCarloBacktest(
            model_paths=model_choices,
            iterations=iterations,
            min_iterations_for_exploit=min_iterations_for_exploit,
            trading_frequency=trading_frequency,
            min_lookback=min_lookback,
            max_lookback=max_lookback
        )
        
        logger.info(f"Running Monte Carlo simulations with {iterations} iterations...")
        print(f"\nStarting Monte Carlo simulations with {iterations} iterations...")
        print("Note: This may take about 5 minutes to complete...")
        print("Press Ctrl+C to stop early with current best result\n")
        
        try:
            top_models = monte_carlo.run()
            print(f"\nMonte Carlo simulation completed in {(time.time() - start_time)/60:.1f} minutes")
        except KeyboardInterrupt:
            print("\n\nStopped early by user")
            return
        
        # Create figure with subplot layout - adjusted for laptop screen
        fig = plt.figure(figsize=(12, 7))  # Width: 12 inches, Height: 7 inches for 16:9 aspect ratio
        gs = plt.GridSpec(6, 1)  # 6 rows total to allow for 83/17 split (5/6 rows and 1/6 row)
        
        # Main portfolio performance plot
        ax1 = fig.add_subplot(gs[0:5, 0])  # Takes up first 5 rows (83%)
        
        # Plot historical values for each model
        colors = ['b', 'r', 'g', 'c', 'm', 'k']  # Added black for cash
        model_to_color = {}  # Map models to their colors
        date_index = pd.DatetimeIndex([pd.Timestamp(d) for d in monte_carlo.dates])
        
        print("\nNormalizing and plotting portfolio values...")
        
        # Plot Monte Carlo best portfolio
        if monte_carlo.best_portfolio_value is not None:
            ax1.plot(date_index, monte_carlo.best_portfolio_value, 
                    'k-', linewidth=2, label='Monte Carlo Best')

        # Normalize all series to start at 10000 and plot after best portfolio
        for (model, values), color in zip(monte_carlo.portfolio_histories.items(), colors):
            if model != "cash":
                model_to_color[model] = color
                # Normalize to start at 10000
                start_value = values.iloc[0]
                normalized_values = values * (10000.0 / start_value)
                print(f"{model}: Original start ${start_value:.2f}, normalized to $10,000")
                ax1.plot(date_index, normalized_values,
                        color=color, alpha=0.5, label=f"{model}",
                        linewidth=1)
        
        model_to_color["cash"] = 'k'  # Add cash to color mapping

        # Configure main plot
        ax1.set_yscale('log')
        
        # Configure grid with both major and minor lines
        ax1.grid(True, which='major', alpha=0.4, linewidth=0.8)  # Thicker major grid
        ax1.grid(True, which='minor', alpha=0.2, linewidth=0.5)  # Thinner minor grid
        ax1.minorticks_on()  # Enable minor ticks
        
        ax1.set_title('Portfolio Performance: Monte Carlo Optimization vs Base Models', 
                     pad=20, fontsize=14)
        ax1.set_xlabel('')  # Remove x-label from top plot
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        
        # Format axes
        ax1.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'${int(x):,}')
        )
        
        # Add current date/time and lookback values to plot
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        lookbacks = monte_carlo.best_params.get('lookbacks', [])
        _lb_list = sorted(lookbacks) if isinstance(lookbacks, list) else lookbacks
        lookback_text = f"Best parameters: lookbacks={_lb_list} days"
        text_str = f"{current_time}\n{lookback_text}"
        
        # Position text in upper left, slightly below the title
        ax1.text(0.02, 0.95, text_str,
                transform=ax1.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                fontsize=9, verticalalignment='top')
        
        # Move legend to lower right
        ax1.legend(loc='lower right', fontsize=8)

        # Create model selection subplot
        ax2 = fig.add_subplot(gs[5, 0])  # Takes up last row (17%)
        
        # Create numeric mapping for all models including cash
        unique_models = sorted(list(monte_carlo.portfolio_histories.keys()))
        model_to_num = {model: i for i, model in enumerate(unique_models)}
        
        # Get model selections over time (assuming monthly trading)
        current_model = None
        model_selections = []
        for date in monte_carlo.dates:
            if monte_carlo._should_trade(date):
                # Find closest model selection for this date
                if hasattr(monte_carlo, 'best_model_selections'):
                    current_model = monte_carlo.best_model_selections.get(
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
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=7)  # Reduced from 8 to 7
        
        # Adjust layout to prevent overlapping
        plt.tight_layout()
        
        # Save final plot
        output_path = "assets/model_switching_portfolio_performance.png"
        os.makedirs("assets", exist_ok=True)
        if os.path.exists(output_path):
            os.remove(output_path)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
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
        total_return = ((final_value / initial_value) - 1) * 100
        print(f"Initial Value: ${initial_value:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.1f}%")
        print("-" * 40)
        print(f"Performance plot saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Monte Carlo backtesting failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()