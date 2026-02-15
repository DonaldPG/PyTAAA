#!/usr/bin/env python3
"""
Temporary script to run portfolio backtest and generate debug weight files.
This script runs the backtest with specified JSON config to debug portfolio differences.
"""

import sys
import os
from functions.GetParams import get_json_params, get_symbols_file
from functions.PortfolioPerformanceCalcs import PortfolioPerformanceCalcs

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_debug_weights.py <json_file_path>")
        sys.exit(1)
    
    json_fn = sys.argv[1]
    
    if not os.path.exists(json_fn):
        print(f"Error: JSON file not found: {json_fn}")
        sys.exit(1)
    
    print(f"Running portfolio backtest with: {json_fn}")
    print("-" * 80)
    
    # Load params from JSON
    params = get_json_params(json_fn, verbose=True)
    symbols_file = get_symbols_file(json_fn)
    
    # Split into directory and file
    symbol_directory, symbol_file = os.path.split(symbols_file)
    
    print(f"Symbol directory: {symbol_directory}")
    print(f"Symbol file: {symbol_file}")
    print("-" * 80)
    
    # Run the backtest (this will generate the debug CSV)
    last_date, symbols, weights, prices = PortfolioPerformanceCalcs(
        symbol_directory, symbol_file, params, json_fn
    )
    
    print("-" * 80)
    print("✓ Backtest complete")
    print(f"  Last date: {last_date}")
    print(f"  Number of symbols: {len(symbols)}")
    print("-" * 80)

if __name__ == "__main__":
    main()
