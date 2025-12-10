import click
import json
import os
import tempfile
from src.backtest.montecarlo_runner import run_monte_carlo_backtest

@click.command()
@click.option('--n_trials', default=250, help='Number of Monte Carlo trials')
@click.option('--config', 'config_file', required=True, type=click.Path(exists=True), help='Path to JSON config file with fixed parameters')
def main(n_trials, config_file):
    run_monte_carlo_backtest(config_file, n_trials)

if __name__ == '__main__':
    main()