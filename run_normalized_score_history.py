#!/usr/bin/env python3

"""Normalized score history plotter for PyTAAA models.

This module creates a single combined plot showing portfolio values in the upper 
subplot and normalized score history for all models in the lower subplot.
The colors for each curve match between the upper and lower subplots.
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
logger = get_logger(__name__, log_file='normalized_score_history.log')


def create_combined_normalized_score_plot(
    model_paths: Dict[str, str], 
    output_path: str,
    data_format: str = 'actual'
) -> Dict[str, any]:
    """Create a combined plot showing all models' portfolio values and normalized scores.
    
    Args:
        model_paths: Dictionary mapping model names to data file paths
        output_path: Path to save the combined plot
        data_format: Data format being used ('actual' or 'backtested')
        
    Returns:
        Dictionary with plot statistics
    """
    print("Initializing combined analysis...")
    
    #############################################################################
    # Load configuration to get lookback parameters
    #############################################################################
    with open('pytaaa_model_switching_params.json', 'r') as f:
        config = json.load(f)
    
    # Get lookback configuration from the same source as run_monte_carlo.py
    recommendation_config = config.get('recommendation_mode', {})
    default_lookbacks = recommendation_config.get('default_lookbacks', [96, 108, 184])
    
    # Try to load best lookbacks from Monte Carlo state if available
    lookbacks_to_use = default_lookbacks
    state_file = "monte_carlo_state.pkl"
    
    if os.path.exists(state_file):
        try:
            import pickle
            with open(state_file, 'rb') as f:
                state_data = pickle.load(f)
            
            # Get best performing combination from state
            if state_data.get('canonical_performance_scores'):
                best_idx = max(range(len(state_data['canonical_performance_scores'])), 
                             key=lambda i: state_data['canonical_performance_scores'][i])
                best_combination = list(state_data['combination_indices'].keys())[best_idx]
                lookbacks_to_use = list(best_combination)
                print(f"Using best performing lookbacks from Monte Carlo state: {lookbacks_to_use}")
            else:
                print(f"Using default lookbacks from config: {lookbacks_to_use}")
        except Exception as e:
            print(f"Could not load Monte Carlo state, using default lookbacks: {lookbacks_to_use}")
    else:
        print(f"No Monte Carlo state found, using default lookbacks: {lookbacks_to_use}")
    
    # Initialize MonteCarloBacktest with minimal iterations since we're just using for data loading
    monte_carlo = MonteCarloBacktest(
        model_paths=model_paths,
        iterations=1,
        trading_frequency="monthly",
        min_lookback=10,
        max_lookback=252
    )
    
    print(f"Loaded data for {len(monte_carlo.portfolio_histories)} models")
    print(f"Date range: {monte_carlo.dates[0]} to {monte_carlo.dates[-1]}")
    print(f"Total data points: {len(monte_carlo.dates)}")
    
    #############################################################################
    # Find first trading day of each month for analysis
    #############################################################################
    monthly_dates = []
    monthly_indices = []
    
    # Convert dates to proper format for analysis
    date_objects = []
    for d in monte_carlo.dates:
        if hasattr(d, 'year'):
            date_objects.append(d)
        else:
            date_objects.append(pd.to_datetime(d).date())
    
    # Find first trading day of each month
    current_month = None
    for i, date_obj in enumerate(date_objects):
        if current_month != (date_obj.year, date_obj.month):
            current_month = (date_obj.year, date_obj.month)
            monthly_dates.append(date_obj)
            monthly_indices.append(i)
    
    print(f"Found {len(monthly_dates)} monthly analysis points")
    
    #############################################################################
    # Compute normalized scores for all models on monthly dates using same method as Monte Carlo
    #############################################################################
    # FIXED: Force consistent alphabetical ordering of models
    models = sorted(list(monte_carlo.portfolio_histories.keys()))
    non_cash_models = [m for m in models if m != "cash"]
    
    # Initialize arrays for normalized scores
    monthly_scores = {model: [] for model in non_cash_models}
    monthly_date_objects = []
    
    # Use the same lookback approach as Monte Carlo model selection
    print(f"Computing monthly normalized scores using lookbacks: {lookbacks_to_use}")
    for date_obj, date_idx in zip(monthly_dates, monthly_indices):
        # Skip if insufficient history (need enough for the maximum lookback)
        max_lookback = max(lookbacks_to_use)
        if date_idx < max_lookback:
            continue
        
        monthly_date_objects.append(date_obj)
        
        # Calculate metrics for each non-cash model using the same multi-lookback approach
        # as used in _select_best_model
        for model in non_cash_models:
            # Calculate performance across all lookback periods (same as Monte Carlo)
            model_scores = []
            
            for lookback_period in lookbacks_to_use:
                start_idx = max(0, date_idx - lookback_period)
                start_date = pd.Timestamp(monte_carlo.dates[start_idx])
                end_date = pd.Timestamp(monte_carlo.dates[date_idx - 1])
                
                # Get portfolio values for the lookback period
                portfolio_vals = monte_carlo.portfolio_histories[model][start_date:end_date].values
                
                if len(portfolio_vals) == 0:
                    # Fallback to constant values if no data
                    portfolio_vals = np.ones(lookback_period) * 10000.0
                
                # Compute performance metrics for this lookback period
                metrics = monte_carlo.compute_performance_metrics(portfolio_vals)
                model_scores.append(metrics['normalized_score'])
            
            # Average the normalized scores across all lookback periods
            # This matches the approach used in Monte Carlo model selection
            avg_normalized_score = np.mean(model_scores)
            monthly_scores[model].append(avg_normalized_score)
    
    print(f"Computed metrics for {len(monthly_date_objects)} monthly points")
    
    #############################################################################
    # Create figure with subplot layout
    #############################################################################
    fig = plt.figure(figsize=(12, 8))
    gs = plt.GridSpec(6, 1)
    
    # Upper subplot for portfolio values (67% of space)
    ax1 = fig.add_subplot(gs[0:4, 0])
    
    # Lower subplot for normalized scores (33% of space)
    ax2 = fig.add_subplot(gs[4:6, 0])
    
    #############################################################################
    # Plot portfolio values (upper subplot) and establish color mapping
    #############################################################################
    colors = ['b', 'r', 'g', 'c', 'm']
    model_to_color = {}
    date_index = pd.DatetimeIndex(monte_carlo.dates)
    
    # Plot historical values for each model
    for i, (model, values) in enumerate(monte_carlo.portfolio_histories.items()):
        if model != "cash":
            color = colors[i % len(colors)]
            model_to_color[model] = color
            start_value = values.iloc[0]
            normalized_values = values * (10000.0 / start_value)
            ax1.plot(date_index, normalized_values,
                    color=color, alpha=0.7, label=f"{model}",
                    linewidth=1.5)
    
    # Configure upper subplot styling
    ax1.set_yscale('log')
    ax1.grid(True, which='major', alpha=0.4, linewidth=0.8)
    ax1.grid(True, which='minor', alpha=0.2, linewidth=0.5)
    ax1.minorticks_on()
    
    # Configure date formatting - 5 year major, 1 year minor
    ax1.xaxis.set_major_locator(mdates.YearLocator(5))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
    
    ax1.set_title('Portfolio Performance and Monthly Normalized Scores', 
                 pad=20, fontsize=14)
    ax1.set_xlabel('')  # Remove x-label from upper plot
    ax1.set_ylabel('Portfolio Value ($)', fontsize=10)
    
    # Format y-axis
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'${int(x):,}')
    )
    ax1.tick_params(axis='y', which='major', labelsize=8)
    ax1.tick_params(axis='y', which='minor', labelsize=6)
    ax1.tick_params(axis='x', which='major', labelsize=10, rotation=0)
    
    # Add legend
    ax1.legend(loc='lower right', fontsize=8)
    
    #############################################################################
    # Plot normalized scores (lower subplot) with matching colors
    #############################################################################
    monthly_pd_dates = pd.DatetimeIndex(monthly_date_objects)
    
    # Plot normalized scores for each non-cash model using matching colors
    # Use thin lines without markers for cleaner appearance
    for model in non_cash_models:
        if model in model_to_color and len(monthly_scores[model]) > 0:
            ax2.plot(monthly_pd_dates, monthly_scores[model],
                    color=model_to_color[model], 
                    linewidth=1.0, alpha=0.8, label=f"{model}")
    
    # Configure lower subplot styling to match upper subplot
    ax2.grid(True, which='major', alpha=0.4, linewidth=0.8)
    ax2.grid(True, which='minor', alpha=0.2, linewidth=0.5)
    ax2.minorticks_on()
    
    # Configure date formatting to match upper subplot
    ax2.xaxis.set_major_locator(mdates.YearLocator(5))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_minor_locator(mdates.YearLocator(1))
    
    # Set labels
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Normalized Score', fontsize=10)
    
    # Set y-axis range for normalized scores
    ax2.set_ylim(-25, 50)
    
    # Configure tick labels
    ax2.tick_params(axis='y', which='major', labelsize=8)
    ax2.tick_params(axis='y', which='minor', labelsize=6)
    ax2.tick_params(axis='x', which='major', labelsize=10, rotation=0)
    
    # Add horizontal line at zero for reference
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    
    # Add legend
    ax2.legend(loc='upper right', fontsize=7)
    
    #############################################################################
    # Add summary text overlay
    #############################################################################
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    summary_text = (f"{current_time}\n"
                   f"Combined Normalized Score Analysis\n"
                   f"Data format: {data_format}\n"
                   f"Lookback periods: {lookbacks_to_use}\n"
                   f"Analysis points: {len(monthly_date_objects)} months\n"
                   f"Models analyzed: {len(non_cash_models)} models")
    
    # Position text in upper left of upper subplot
    ax1.text(0.02, 0.95, summary_text,
            transform=ax1.transAxes,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
            fontsize=7, verticalalignment='top',
            fontfamily='monospace')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined plot saved to: {output_path}")
    
    # Return summary statistics
    return {
        'monthly_dates': monthly_date_objects,
        'monthly_scores': monthly_scores,
        'analysis_points': len(monthly_date_objects),
        'models_analyzed': non_cash_models,
        'model_colors': model_to_color,
        'lookbacks_used': lookbacks_to_use
    }


@click.command()
@click.option(
    '--data-format',
    type=click.Choice(['actual', 'backtested']),
    default='actual',
    help='Data format to use: actual (PyTAAA_status.params) or backtested (pyTAAAweb_backtestPortfolioValue.params)'
)
@click.option(
    '--output-dir',
    default='assets',
    help='Output directory for plot files (default: assets)'
)
def main(data_format: str, output_dir: str) -> None:
    """Generate combined normalized score history plot for all PyTAAA models.
    
    Args:
        data_format: Whether to use actual or backtested portfolio values
        output_dir: Directory to save the plot files
    """
    
    try:
        logger.info("Starting combined normalized score history plot generation")
        logger.info(f"Data format: {data_format}")
        print("Starting combined normalized score history plot generation")
        print(f"Data format: {data_format}")
        
        # Configure data files based on format
        data_files = {
            'actual': 'PyTAAA_status.params',
            'backtested': 'pyTAAAweb_backtestPortfolioValue.params'
        }
        
        # Configure model paths with correct data_store locations
        base_folder = "/Users/donaldpg/pyTAAA_data"
        model_paths = {
            "cash": "",
            "naz100_pine": f"{base_folder}/naz100_pine/data_store/{data_files[data_format]}",
            "naz100_hma": f"{base_folder}/naz100_hma/data_store/{data_files[data_format]}",
            "naz100_pi": f"{base_folder}/naz100_pi/data_store/{data_files[data_format]}",
            "sp500_hma": f"{base_folder}/sp500_hma/data_store/{data_files[data_format]}"
        }

        print(f"\nUsing {'actual' if data_format == 'actual' else 'backtested'} portfolio values")
        print(f"Reading from: {data_files[data_format]}")
        print(f"Output directory: {output_dir}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if all data files exist
        missing_files = []
        for model_key, path in model_paths.items():
            if model_key != "cash" and not os.path.exists(path):
                missing_files.append(f"{model_key}: {path}")
        
        if missing_files:
            print(f"\nWarning: Missing data files:")
            for missing in missing_files:
                print(f"  {missing}")
            print("\nProceeding with available models...")
        
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Creating Combined Normalized Score History Plot")
        print(f"{'='*60}")
        
        # Generate the combined plot
        output_filename = f"combined_normalized_score_history_{data_format}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Generating combined normalized score plot...")
        
        # Create the combined plot
        plot_stats = create_combined_normalized_score_plot(
            model_paths=model_paths,
            output_path=output_path,
            data_format=data_format
        )
        
        elapsed_time = time.time() - start_time
        print(f"✓ Successfully created combined plot in {elapsed_time:.1f} seconds")
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Data format used: {data_format}")
        print(f"Plot saved to: {output_path}")
        print(f"Analysis points: {plot_stats['analysis_points']} months")
        print(f"Models analyzed: {len(plot_stats['models_analyzed'])}")
        
        print(f"\nModel Colors:")
        for model, color in plot_stats['model_colors'].items():
            print(f"  {model}: {color}")
        
        print(f"\nPlot Structure:")
        print(f"  Upper subplot: Portfolio values over time (log scale)")
        print(f"  Lower subplot: Monthly normalized scores for all models")
        print(f"  Colors: Matching between upper and lower subplots")
        print(f"  Analysis: 60-day lookback period for performance metrics")
        
        logger.info("Combined normalized score history plot generation completed")
        
    except Exception as e:
        logger.error(f"Combined normalized score plot generation failed: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()