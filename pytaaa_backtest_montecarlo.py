"""Monte Carlo backtest CLI tool for PyTAAA trading strategies.

This module provides a JSON-driven command-line interface for running
Monte Carlo backtests across different trading models (sp500_pine,
naz100_hma, etc.). It replaces hardcoded paths with dynamic configuration.

Usage:
    uv run python pytaaa_backtest_montecarlo.py --json config.json
    uv run python pytaaa_backtest_montecarlo.py --json config.json --trials 3

Configuration:
    Requires JSON file with 'Valuation' section containing:
    - symbols_file: Path to stock symbols
    - performance_store: Output directory base
    - webpage: Model identifier extraction
    - backtest_monte_carlo_trials: Number of trials (default: 250)

Outputs:
    - CSV: {model_id}_montecarlo_{date}_{runnum}.csv
    - JSON: {model_id}_optimized_{date}.json

Related:
    - PyTAAA_backtest_sp500_pine_refactored.py: Original implementation
"""

import matplotlib
matplotlib.use("Agg")

import os
import sys
import datetime
import click
from functions.GetParams import get_json_params
from functions.backtesting.config_helpers import (
    extract_model_identifier,
    setup_output_paths,
    generate_output_filename,
)
from functions.backtesting.monte_carlo_runner import run_monte_carlo_backtest
from functions.logger_config import get_logger

logger = get_logger(__name__, log_file="pytaaa_backtest_montecarlo.log")

EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_CONFIG_ERROR = 2
EXIT_DATA_NOT_FOUND = 3


@click.command()
@click.option(
    "--json", "json_fn",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    help="Path to JSON configuration file with Monte Carlo parameters",
)
@click.option(
    "--trials", "trials_override",
    type=int,
    default=None,
    help="Number of Monte Carlo trials (overrides JSON config)",
)
def main(json_fn: str, trials_override: int | None) -> None:
    """Execute Monte Carlo backtest with JSON configuration.

    Examples:
        uv run python pytaaa_backtest_montecarlo.py --json config.json
        uv run python pytaaa_backtest_montecarlo.py --json config.json --trials 3
    """
    try:
        print("=" * 80)
        print("PyTAAA Monte Carlo Backtest")
        print("=" * 80)
        print(f"Configuration: {json_fn}")

        # Load configuration
        params = get_json_params(json_fn, verbose=True)
        
        # Note: performance_store and webpage are NOT in params dict
        # (they're retrieved separately via get_performance_store/get_webpage_store)
        # Validate only that symbols_file exists
        if "symbols_file" not in params:
            raise KeyError("Missing required configuration key: 'symbols_file'")

        # Get trial count (CLI override or JSON or default)
        n_trials = trials_override or params.get(
            "backtest_monte_carlo_trials", 250
        )
        print(f"Monte Carlo trials: {n_trials}")

        # Set up output paths
        model_id, output_dir, perf_store = setup_output_paths(json_fn)
        print(f"Model identifier: {model_id}")
        print(f"Output directory: {output_dir}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate output filenames
        today = datetime.date.today()
        date_str = f"{today.year}-{today.month}-{today.day}"

        # Determine runnum from symbol file
        symbols_file = params.get("symbols_file", "")
        basename = os.path.basename(symbols_file)
        runnum_map = {
            "symbols.txt": "run2501a",
            "Naz100_Symbols.txt": "run250b",
            "biglist.txt": "run2503",
            "ProvidentFundSymbols.txt": "run2504",
            "sp500_symbols.txt": "run2505",
            "cmg_symbols.txt": "run2507",
            "SP500_Symbols.txt": "run2506",
        }
        runnum = runnum_map.get(basename, "run2508d")

        outfilename = os.path.join(
            output_dir,
            f"{model_id}_montecarlo_{date_str}_{runnum}.csv"
        )

        output_paths = {
            "model_id": model_id,
            "outfiledir": output_dir,
            "outfilename": outfilename,
            "date_str": date_str,
            "runnum": runnum,
        }

        logger.info(
            "Starting Monte Carlo backtest: model=%s, trials=%d",
            model_id, n_trials
        )

        # Run Monte Carlo backtest
        results = run_monte_carlo_backtest(json_fn, n_trials, output_paths)

        print("=" * 80)
        print("Monte Carlo backtest completed successfully")
        print(
            f"Best Sharpe: {results['best_sharpe']:.4f}"
            f" (trial #{results['best_trial']})"
        )
        print(f"Output: {outfilename}")
        print("=" * 80)

        sys.exit(EXIT_SUCCESS)

    except FileNotFoundError as exc:
        click.echo(f"Error: File not found - {exc}", err=True)
        sys.exit(EXIT_DATA_NOT_FOUND)
    except KeyError as exc:
        click.echo(f"Error: Missing configuration key - {exc}", err=True)
        sys.exit(EXIT_CONFIG_ERROR)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(EXIT_ERROR)


if __name__ == "__main__":
    main()
