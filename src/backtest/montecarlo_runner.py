"""
Monte Carlo backtest runner for parameter optimization.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
from tqdm import tqdm

from src.backtest.montecarlo import MonteCarloBacktest, create_temporary_json
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

    # add run identifier using current date and time as "run_" + YYYYMMDD_HHMM
    from datetime import datetime
    run_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M")

    # Define CSV fieldnames
    fieldnames = [
        "run","trial",
        "Number stocks","monthsToHold","LongPeriod","MA1","MA2","MA3",
        "volatility min","volatility max",
        "stddevThreshold","sma2factor","rank Threshold (%)","sma_filt_val",
        "Portfolio Final Value","Portfolio std","Portfolio Sharpe",
        "begin date for recent performance",
        "Portfolio Ann Gain - recent","Portfolio Sharpe - recent",
        "B&H Ann Gain - recent","B&H Sharpe - recent",
        "Sharpe 15 Yr","Sharpe 10 Yr","Sharpe 5 Yr","Sharpe 3 Yr","Sharpe 2 Yr","Sharpe 1 Yr",
        "Avg Return 15 Yr","Avg Return 10 Yr","Avg Return 5 Yr","Avg Return 3 Yr","Avg Return 2 Yr","Avg Return 1 Yr",
        "CAGR 15 Yr","CAGR 10 Yr","CAGR 5 Yr","CAGR 3 Yr","CAGR 2 Yr","CAGR 1 Yr",
        "Avg Drawdown 15 Yr","Avg Drawdown 10 Yr","Avg Drawdown 5 Yr","Avg Drawdown 3 Yr","Avg Drawdown 2 Yr","Avg Drawdown 1 Yr",
        "beatBuyHoldTest","beatBuyHoldTest2"
    ]

    # ### TODO: ------------------------------ remove this code block later - start
    # print(
    #     "\n\n\n ... inside run_montecarlo"
    #     "\n   . Base parameters for Monte Carlo trials:"
    # )
    # for key, value in base_params.items():
    #     print(f"   {key}: {value}")
    # print("\n\n\n")
    # ### TODO: ------------------------------ remove this code block later - end

    # Run trials with progress bar
    for trial in tqdm(range(n_trials), desc="Running Monte Carlo trials"):
        # Generate random parameters for this trial
        trial_params = mc_backtest.generate_random_params(
            trial,
            base_params.get('uptrendSignalMethod', 'percentileChannels')
        )
        trial_params['trial'] = trial
        trial_params['run_id'] = run_id
        
        # Merge with base params
        params = {**base_params, **trial_params}
        temp_trial_json = create_temporary_json(
            base_json_fn,
            params,
            trial
        )

        ### TODO: ------------------------------ remove this code block later - start
        print(
            "\n   . Base parameters for Monte Carlo trial:"
        )
        for key, value in trial_params.items():
            print(f"   {key}: trial_params {value}, base_params {base_params.get(key)}")
        print("\n\n\n")
        ### TODO: ------------------------------ remove this code block later - end 
       
        # Run backtest
        try:
            result = dailyBacktest_pctLong(
                params, temp_trial_json, return_results=True, plot=plot_individual
            )
            result['trial'] = trial
            results.append(result)
            
            # Append result to CSV
            mode = 'w' if trial == 0 else 'a'
            with open(output_csv, mode, newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if trial == 0:
                    writer.writeheader()
                writer.writerow(result)
        except Exception as e:
            print(f"Error in trial {trial}: {e}")
            continue
    
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