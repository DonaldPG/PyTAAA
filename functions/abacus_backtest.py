#!/usr/bin/env python3

"""Backtest data management for Abacus model-switching trading system.

This module provides utilities for loading, validating, and managing backtest
data for the model-switching methodology. It includes data loading, model path
configuration, and portfolio generation capabilities.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from functions.logger_config import get_logger

# Get module-specific logger
logger = get_logger(__name__, log_file='abacus_backtest.log')


class BacktestDataLoader:
    """Load and validate backtest data files for model-switching analysis."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize data loader with optional configuration.
        
        Args:
            config: Optional JSON configuration dictionary
        """
        self.config = config or {}
    
    def build_model_paths(
        self, 
        data_format: str = 'backtested',
        json_config_path: Optional[str] = None
    ) -> Dict[str, str]:
        """Build model paths dictionary from configuration.
        
        Args:
            data_format: Either 'actual' or 'backtested'
            json_config_path: Path to JSON config file (if available)
            
        Returns:
            Dictionary mapping model names to file paths
        """
        # Determine data file format
        data_files = {
            'actual': 'PyTAAA_status.params',
            'backtested': 'pyTAAAweb_backtestPortfolioValue.params'
        }
        
        # Configure model paths - use JSON config if available
        if json_config_path and 'models' in self.config:
            # Use JSON configuration for model paths
            models_config = self.config['models']
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

        return model_choices
    
    def validate_model_paths(self, model_paths: Dict[str, str]) -> Dict[str, str]:
        """Validate that model data files exist.
        
        Logs warnings for missing files but keeps mappings so downstream
        code (MonteCarloBacktest) can handle them appropriately.
        
        Args:
            model_paths: Dictionary mapping model names to file paths
            
        Returns:
            Validated dictionary (same as input, with resolved paths)
        """
        validated_model_choices = {}
        
        for mname, mtemplate in model_paths.items():
            if not mtemplate:  # Cash model has empty path
                validated_model_choices[mname] = ""
                continue

            resolved_path = os.path.expanduser(mtemplate)
            resolved_path = os.path.abspath(resolved_path)

            if not os.path.exists(resolved_path):
                logger.warning(
                    f"Model data file not found for {mname}: {resolved_path}. "
                    f"Keeping mapping; MonteCarloBacktest will handle missing data."
                )
                print(
                    f"WARNING: Model data file not found for {mname}: {resolved_path}. "
                    f"Keeping mapping; MonteCarloBacktest will handle missing data."
                )

            validated_model_choices[mname] = resolved_path

        return validated_model_choices


def write_abacus_backtest_portfolio_values(
    json_config_path: str,
    lookbacks: Optional[List[int]] = None
) -> bool:
    """Write abacus model-switching portfolio values to backtest params file.
    
    This function generates the abacus model-switching portfolio values
    and writes them to column 3 of the pyTAAAweb_backtestPortfolioValue.params
    file, replacing existing values while preserving dates and other columns.
    Also adds model name as 6th column.
    
    Args:
        json_config_path: Path to the JSON configuration file (must be abacus config)
        lookbacks: Optional lookback periods for model selection.
                  If None, will try to load from saved state or use config defaults.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Verify this is an abacus configuration
        if "abacus" not in json_config_path.lower():
            logger.warning(f"Skipping: Not an abacus configuration: {json_config_path}")
            return False
        
        logger.info(f"Generating abacus backtest portfolio values for {json_config_path}")
        print(f"\nGenerating abacus backtest portfolio values...")
        
        # Load configuration
        import json
        with open(json_config_path, 'r') as f:
            config = json.load(f)
        
        # Get lookbacks if not provided
        if lookbacks is None:
            from functions.abacus_recommend import ConfigurationHelper
            ConfigurationHelper.ensure_config_defaults(config)
            lookbacks = ConfigurationHelper.get_recommendation_lookbacks(
                "use-saved", config
            )
        
        print(f"Using lookback periods: {lookbacks}")
        
        # Suppress numba's verbose debug logging
        import logging
        numba_logger = logging.getLogger('numba')
        original_numba_level = numba_logger.level
        numba_logger.setLevel(logging.WARNING)
        
        # Initialize Monte Carlo system
        # Note: Import MonteCarloBacktest with care - numba initialization can conflict
        # with custom print functions in __main__. Temporarily restore original print.
        import builtins
        original_print = builtins.print
        try:
            # Restore original print during import to avoid numba conflicts
            if hasattr(builtins, '__original_print__'):
                builtins.print = builtins.__original_print__
            from functions.MonteCarloBacktest import MonteCarloBacktest
        finally:
            # Restore whatever print was being used
            builtins.print = original_print
            # Keep numba logging suppressed (don't restore original level)
        
        # Get data format from config
        monte_carlo_config = config.get('monte_carlo', {})
        data_format = 'backtested'  # Always use backtested for this operation
        
        # Configure model paths
        loader = BacktestDataLoader(config)
        model_choices = loader.build_model_paths(data_format, json_config_path)
        model_choices = loader.validate_model_paths(model_choices)
        
        print(f"Initializing Monte Carlo backtest system...")
        
        # Initialize Monte Carlo instance
        monte_carlo = MonteCarloBacktest(
            model_paths=model_choices,
            iterations=1,
            trading_frequency=monte_carlo_config.get('trading_frequency', 'monthly'),
            min_lookback=monte_carlo_config.get('min_lookback', 10),
            max_lookback=monte_carlo_config.get('max_lookback', 400),
            search_mode='exploit',
            json_config=config
        )
        
        # Force date range to use longest available model (Nasdaq starts 1991, SP500 starts 2000)
        # Find the model with the most dates (earliest start date)
        max_dates = 0
        longest_model = None
        for model_name in monte_carlo.portfolio_histories.keys():
            if model_name != 'cash':
                model_dates = len(monte_carlo.portfolio_histories[model_name])
                if model_dates > max_dates:
                    max_dates = model_dates
                    longest_model = model_name
        
        if longest_model:
            # Use the date range from the longest model
            longest_dates = monte_carlo.portfolio_histories[longest_model].index.tolist()
            monte_carlo.dates = [d.date() if hasattr(d, 'date') else d for d in longest_dates]
            print(f"Using date range from {longest_model}: {len(monte_carlo.dates)} dates from {monte_carlo.dates[0]} to {monte_carlo.dates[-1]}")
        
        # Apply normalization values if available
        from functions.GetParams import get_central_std_values
        try:
            normalization_values = get_central_std_values(json_config_path)
            monte_carlo.CENTRAL_VALUES = normalization_values['central_values']
            monte_carlo.STD_VALUES = normalization_values['std_values']
            print(f"Applied JSON normalization values")
        except Exception as e:
            logger.warning(f"Could not load normalization values: {e}")
        
        # Calculate model-switching portfolio with model selections
        print(f"Calculating model-switching portfolio...")
        model_switching_portfolio, model_selections = _calculate_model_switching_portfolio_with_selections(
            monte_carlo, lookbacks
        )
        
        # Get dates from monte_carlo
        dates = monte_carlo.dates
        
        # Determine output file path
        from functions.GetParams import get_json_params
        params = get_json_params(json_config_path)
        data_folder = params.get('data_folder', '/Users/donaldpg/pyTAAA_data')
        folder_name = params.get('folder_name', 'naz100_sp500_abacus')
        output_file = os.path.join(
            data_folder, folder_name, 'data_store',
            'pyTAAAweb_backtestPortfolioValue.params'
        )
        
        # Read existing file
        if not os.path.exists(output_file):
            logger.error(f"Output file does not exist: {output_file}")
            print(f"Error: Output file does not exist: {output_file}")
            return False
        
        print(f"Reading existing backtest file: {output_file}")
        
        # Read existing data
        existing_dates = []
        existing_col2 = []
        existing_col3 = []
        existing_col4 = []
        existing_col5 = []
        existing_col6 = []  # Model names (may not exist in older files)
        
        with open(output_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    existing_dates.append(parts[0])
                    existing_col2.append(parts[1])
                    existing_col3.append(parts[2])
                    existing_col4.append(parts[3])
                    existing_col5.append(parts[4])
                    # Handle optional 6th column
                    if len(parts) >= 6:
                        existing_col6.append(parts[5])
                    else:
                        existing_col6.append("UNKNOWN")
        
        # Create mapping from date to abacus portfolio value and model name
        abacus_values = {}
        abacus_models = {}
        abacus_dates_set = set()
        for i, date_val in enumerate(dates):
            from datetime import datetime
            if isinstance(date_val, datetime):
                date_str = date_val.strftime('%Y-%m-%d')
            else:
                date_str = str(date_val)
            abacus_values[date_str] = model_switching_portfolio[i]
            abacus_models[date_str] = model_selections[i]
            abacus_dates_set.add(date_str)
        
        # Find dates in abacus that are missing from existing file (dates before existing start)
        from datetime import datetime
        existing_dates_set = set(existing_dates)
        missing_dates = []
        if existing_dates:
            first_existing_date = datetime.strptime(existing_dates[0], '%Y-%m-%d').date()
            for date_str in sorted(abacus_dates_set):
                date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                if date_obj < first_existing_date and date_str not in existing_dates_set:
                    missing_dates.append(date_str)
        
        # Prepend missing dates (from 1991-2000 if they exist)
        updated_lines = []
        updates_count = 0
        prepended_count = 0
        
        # Add missing dates at the beginning
        for date_str in sorted(missing_dates):
            if date_str in abacus_values:
                # Create new line with abacus value in col 3 and model name in col 6
                # Use placeholder values for other columns per user specification
                new_col2 = "10000.0"  # Col 1: Buy-hold value (10000 if no data)
                new_col3 = f"{abacus_values[date_str]:.2f}"  # Col 2: Abacus portfolio value
                new_col4 = "-1000"  # Col 3: Placeholder (-1000 when padding)
                new_col5 = "-100000.0"  # Col 4: Placeholder (-100000 when padding)
                # Col 5: Model name - use "CASH" (uppercase) when value is at initial 10000
                # or when model is "cash", otherwise use actual model name
                model_name = abacus_models[date_str]
                if model_name.lower() == "cash" or abacus_values[date_str] == 10000.0:
                    new_col6 = "CASH"
                else:
                    new_col6 = model_name
                updated_lines.append(
                    f"{date_str} {new_col2} {new_col3} "
                    f"{new_col4} {new_col5} {new_col6}\n"
                )
                prepended_count += 1
        
        # Update existing dates
        for i in range(len(existing_dates)):
            date_str = existing_dates[i]
            
            if date_str in abacus_values:
                # Replace column 3 with abacus value and column 6 with model name
                new_col3 = f"{abacus_values[date_str]:.2f}"
                new_col6 = abacus_models[date_str]
                updated_lines.append(
                    f"{existing_dates[i]} {existing_col2[i]} {new_col3} "
                    f"{existing_col4[i]} {existing_col5[i]} {new_col6}\n"
                )
                updates_count += 1
            else:
                # Keep original line (preserve existing col6 if it exists)
                updated_lines.append(
                    f"{existing_dates[i]} {existing_col2[i]} {existing_col3[i]} "
                    f"{existing_col4[i]} {existing_col5[i]} {existing_col6[i]}\n"
                )
        
        # Write updated data back to file
        print(f"Writing updated values to {output_file}")
        with open(output_file, 'w') as f:
            f.writelines(updated_lines)
        
        total_dates = len(updated_lines)
        print(f"Successfully prepended {prepended_count} dates and updated {updates_count} existing dates")
        print(f"Total dates in file: {total_dates} (from {updated_lines[0].split()[0]} to {updated_lines[-1].split()[0]})")
        logger.info(f"Prepended {prepended_count} dates, updated {updates_count} dates in {output_file}")
        logger.info(f"Total dates: {total_dates}")

        
        # Compute metrics for the abacus portfolio
        metrics = monte_carlo.compute_performance_metrics(model_switching_portfolio)
        print(f"\nAbacus Model-Switching Portfolio:")
        print(f"  Final Value: ${metrics['final_value']:,.0f}")
        print(f"  Annual Return: {metrics['annual_return']:.1f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Normalized Score: {metrics['normalized_score']:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to write abacus backtest portfolio values: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        return False


def _calculate_model_switching_portfolio_with_selections(
    monte_carlo, lookbacks: List[int]
) -> tuple:
    """Calculate portfolio values and track model selections.
    
    This is similar to MonteCarloBacktest._calculate_model_switching_portfolio
    but also returns which model was selected at each date.
    
    Args:
        monte_carlo: MonteCarloBacktest instance
        lookbacks: Lookback periods to use for model selection
        
    Returns:
        Tuple of (portfolio_values, model_selections)
        where model_selections[i] is the model name used at date i
    """
    n_dates = len(monte_carlo.dates)
    portfolio_values = np.zeros(n_dates)
    portfolio_values[0] = 10000.0
    model_selections = ["cash"] * n_dates  # Track which model is selected
    current_model = "cash"
    
    # Reset trading state
    monte_carlo.last_trade_date = None
    
    # Simulate trading through all dates
    for t in range(1, n_dates):
        date_val = monte_carlo.dates[t]
        prev_date_val = monte_carlo.dates[t-1]
        
        # Convert date objects to pandas Timestamps for indexing
        date = pd.Timestamp(date_val)
        prev_date = pd.Timestamp(prev_date_val)
        
        # Check if we should trade (monthly rebalancing)
        if monte_carlo._should_trade(date_val):
            # Select best model for this date using specified lookbacks
            if t >= max(lookbacks):  # Ensure we have enough history
                current_model, _ = monte_carlo._select_best_model(t, lookbacks=lookbacks)
            else:
                current_model = "cash"  # Default to cash if insufficient history
        
        # Record which model is being used
        model_selections[t] = current_model
        
        # Update portfolio value based on current model
        if current_model == "cash":
            portfolio_values[t] = portfolio_values[t-1]
        else:
            try:
                model_return = (
                    monte_carlo.portfolio_histories[current_model][date] /
                    monte_carlo.portfolio_histories[current_model][prev_date]
                )
                portfolio_values[t] = portfolio_values[t-1] * model_return
            except (KeyError, ZeroDivisionError):
                # Fallback to cash if model data is unavailable
                portfolio_values[t] = portfolio_values[t-1]
                model_selections[t] = "cash"
    
    return portfolio_values, model_selections
