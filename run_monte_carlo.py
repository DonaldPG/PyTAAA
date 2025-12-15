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
@click.option(
    '--fp-duration',
    type=int,
    default=None,
    help='Focus period duration in years (default: 5 or from JSON config)'
)
@click.option(
    '--fp-year-min',
    type=int,
    default=None,
    help='Minimum year for focus period start (default: 1995)'
)
@click.option(
    '--fp-year-max',
    type=int,
    default=None,
    help='Maximum year for focus period start (default: 2021)'
)
def main(
    search: str, 
    verbose: bool, 
    json_config_path: Optional[str], 
    randomize: bool,
    fp_duration: Optional[int],
    fp_year_min: Optional[int],
    fp_year_max: Optional[int]
) -> None:
    """Run Monte Carlo backtesting with actual or backtested portfolio values.
    
    Args:
        search: Search strategy to use for parameter exploration
        verbose: Whether to show detailed normalized score breakdown
        json_config_path: Path to JSON configuration file
        randomize: Whether to use randomized normalization values
        fp_duration: Focus period duration in years
        fp_year_min: Minimum year for focus period start
        fp_year_max: Maximum year for focus period start
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
        
        #############################################################################
        # Handle focus period parameter overrides
        #############################################################################
        
        # Set defaults for focus period parameters
        fp_duration_years = fp_duration if fp_duration is not None else 5
        fp_year_min_val = fp_year_min if fp_year_min is not None else 1995
        fp_year_max_val = fp_year_max if fp_year_max is not None else 2021
        
        # Validate year range
        if fp_year_min_val > fp_year_max_val:
            raise ValueError(f"Minimum year ({fp_year_min_val}) cannot be greater than maximum year ({fp_year_max_val})")
        
        # Generate random focus period start years with overlap constraint
        max_attempts = 100
        overlap_threshold = 0.4
        
        for attempt in range(max_attempts):
            # Randomly select two start years
            year1 = np.random.randint(fp_year_min_val, fp_year_max_val + 1)
            year2 = np.random.randint(fp_year_min_val, fp_year_max_val + 1)
            
            # Ensure ascending chronological order: FP1 (earlier) before FP2 (later)
            fp1_start_year = min(year1, year2)
            fp2_start_year = max(year1, year2)
            
            # Calculate end years
            fp1_end_year = fp1_start_year + fp_duration_years
            fp2_end_year = fp2_start_year + fp_duration_years
            
            # Calculate overlap
            overlap_start = max(fp1_start_year, fp2_start_year)
            overlap_end = min(fp1_end_year, fp2_end_year)
            overlap_years = max(0, overlap_end - overlap_start)
            overlap_percentage = overlap_years / fp_duration_years
            
            # Check if overlap is acceptable
            if overlap_percentage <= overlap_threshold:
                break
        else:
            # If we couldn't find valid periods after max_attempts, use them anyway but warn
            logger.warning(f"Could not find focus periods with <{overlap_threshold*100}% overlap after {max_attempts} attempts")
            print(f"Warning: Focus periods overlap by {overlap_percentage*100:.1f}% (exceeds {overlap_threshold*100}% threshold)")
        
        # Override config with generated focus periods if CLI parameters were provided
        if fp_duration is not None or fp_year_min is not None or fp_year_max is not None:
            # Update metric_blending config with new focus periods
            if 'metric_blending' not in config:
                config['metric_blending'] = {}
            
            config['metric_blending']['focus_period_1_start'] = f"{fp1_start_year}-01-01"
            config['metric_blending']['focus_period_1_end'] = f"{fp1_end_year}-01-01"
            config['metric_blending']['focus_period_2_start'] = f"{fp2_start_year}-01-01"
            config['metric_blending']['focus_period_2_end'] = f"{fp2_end_year}-01-01"
            
            print(f"\nFocus Period Configuration (overriding JSON):")
            print(f"  Duration: {fp_duration_years} years")
            print(f"  Year range: {fp_year_min_val} to {fp_year_max_val}")
            print(f"  Focus Period 1: {fp1_start_year} to {fp1_end_year}")
            print(f"  Focus Period 2: {fp2_start_year} to {fp2_end_year}")
            print(f"  Overlap: {overlap_years} years ({overlap_percentage*100:.1f}%)")
            logger.info(f"Focus periods: FP1={fp1_start_year}-{fp1_end_year}, FP2={fp2_start_year}-{fp2_end_year}, overlap={overlap_percentage*100:.1f}%")
        
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

        print("\nModel-Switching Effectiveness Analysis:")
        print("-" * 50)
        effectiveness = final_metrics.get('model_effectiveness', {})
        sharpe_outperformance_pct = effectiveness.get('sharpe_outperformance_pct', 0.0)
        sortino_outperformance_pct = effectiveness.get('sortino_outperformance_pct', 0.0)
        average_rank = effectiveness.get('average_rank', 6.0)
        
        print(f"Sharpe ratio outperformance: {sharpe_outperformance_pct:.1f}% ({effectiveness.get('sharpe_wins', 0)}/{effectiveness.get('total_sharpe_comparisons', 32)} comparisons)")
        print(f"Sortino ratio outperformance: {sortino_outperformance_pct:.1f}% ({effectiveness.get('sortino_wins', 0)}/{effectiveness.get('total_sortino_comparisons', 32)} comparisons)")
        print(f"Average rank across all periods and metrics (Sharpe + Sortino): {average_rank:.2f} (5=best, 0=worst)")

        # Create comparison DataFrames for detailed analysis
        from functions.PortfolioMetrics import create_comparison_dataframes, analyze_method_performance
        
        sharpe_df, sortino_df = create_comparison_dataframes(effectiveness)
        performance_analysis = analyze_method_performance(sharpe_df, sortino_df)

        print("\n" + "="*80)
        print("DETAILED COMPARISON DATAFRAMES")
        print("="*80)
        
        print("\nSharpe Ratio Comparison:")
        print("-" * 40)
        print(sharpe_df.round(3))
        
        print("\nSortino Ratio Comparison:")
        print("-" * 40)
        print(sortino_df.round(3))
        
        print("\n" + "="*80)
        print("PERFORMANCE ANALYSIS SUMMARY")
        print("="*80)
        
        # Individual method comparisons
        print("\nSharpe Ratio Head-to-Head Results:")
        for method, stats in performance_analysis.get("sharpe_comparison", {}).items():
            print(f"  vs {method:>12}: {stats['periods_outperformed']}/{stats['total_periods']} periods ({stats['win_rate_pct']:>5.1f}%), avg diff: {stats['avg_difference']:>+6.3f}")
        
        print("\nSortino Ratio Head-to-Head Results:")
        for method, stats in performance_analysis.get("sortino_comparison", {}).items():
            print(f"  vs {method:>12}: {stats['periods_outperformed']}/{stats['total_periods']} periods ({stats['win_rate_pct']:>5.1f}%), avg diff: {stats['avg_difference']:>+6.3f}")
        
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