import click
import json
import os
import tempfile
from src.backtest.montecarlo_runner import run_montecarlo

@click.command()
@click.option('--n_trials', default=250, help='Number of Monte Carlo trials')
@click.option('--config', 'config_file', required=True, type=click.Path(exists=True), help='Path to JSON config file with fixed parameters')
@click.option('--output_csv', default='montecarlo_results.csv', help='Path to output CSV file')
@click.option('--plot_individual', is_flag=True, help='Generate individual plots for each trial')
def main(n_trials, config_file, output_csv, plot_individual):
    run_montecarlo(
        n_trials=n_trials,
        base_json_fn=config_file,
        output_csv=output_csv,
        plot_individual=plot_individual
    )

if __name__ == '__main__':
    main()