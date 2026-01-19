#!/usr/bin/env python3

"""Backtest data management for Abacus model-switching trading system.

This module provides utilities for loading, validating, and managing backtest
data for the model-switching methodology. It includes data loading, model path
configuration, and portfolio generation capabilities.
"""

import os
import pandas as pd
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
