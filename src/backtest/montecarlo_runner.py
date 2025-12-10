"""
Monte Carlo backtest runner for parameter optimization.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
from tqdm import tqdm

from src.backtest.montecarlo import MonteCarloBacktest
from src.backtest.dailyBacktest_pctLong import dailyBacktest_pctLong


def run_montecarlo(
    n_trials: int,
    base_json_fn: str,
    output_csv: str,
    plot_individual: bool = False,
    hold_months: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """
    Run Monte Carlo backtesting trials.
    
    Args:
        n_trials: Number of trials to run.
        base_json_fn: Path to base JSON configuration file.
        output_csv: Path to output CSV file.
        plot_individual: Whether to generate individual plots for each trial.
        hold_months: List of possible holding periods.
        
    Returns:
        List of results dictionaries.
    """
    # Initialize Monte Carlo backtest manager
    mc_backtest = MonteCarloBacktest(
        base_json_fn=base_json_fn,
        n_trials=n_trials,
        hold_months=hold_months
    )
    
    results = []
    
    # Load base params
    with open(base_json_fn, 'r') as f:
        full_config = json.load(f)
        base_params = full_config.get('Valuation', full_config)
    
    # Run trials with progress bar
    for trial in tqdm(range(n_trials), desc="Running Monte Carlo trials"):
        # Generate random parameters for this trial
        trial_params = mc_backtest.generate_random_params(trial)
        
        # Merge with base params
        params = {**base_params, **trial_params}
        
        # Run backtest
        try:
            result = dailyBacktest_pctLong(params, base_json_fn, return_results=True, plot=plot_individual)
            result['trial'] = trial
            results.append(result)
        except Exception as e:
            print(f"Error in trial {trial}: {e}")
            continue
    
    # Write results to CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
    
    return results


if __name__ == "__main__":
    # Example usage
    results = run_montecarlo(
        n_trials=10,
        base_json_fn="path/to/base.json",
        output_csv="montecarlo_results.csv",
        plot_individual=False
    )
    print(f"Completed {len(results)} trials")