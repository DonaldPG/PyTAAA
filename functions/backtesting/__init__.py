"""Modular backtesting package for PyTAAA trading strategies.

This package provides a clean, modular interface for running Monte Carlo
backtests across different trading models. It separates configuration,
parameter generation, output writing, and backtest orchestration into
distinct submodules.

Submodules:
    config_helpers: Path setup, model ID extraction, and config validation.
    parameter_exploration: Random parameter generation for Monte Carlo trials.
    output_writers: CSV and JSON output utilities.
    monte_carlo_runner: High-level Monte Carlo orchestration.

Example:
    >>> from functions.backtesting.config_helpers import (
    ...     extract_model_identifier, setup_output_paths,
    ... )
    >>> from functions.backtesting.parameter_exploration import (
    ...     generate_random_parameters,
    ... )
    >>> from functions.backtesting.output_writers import (
    ...     write_csv_header, append_csv_row,
    ... )
    >>> from functions.backtesting.monte_carlo_runner import (
    ...     run_monte_carlo_backtest,
    ... )
"""

from functions.backtesting.config_helpers import (
    extract_model_identifier,
    setup_output_paths,
    generate_output_filename,
    validate_configuration,
)
from functions.backtesting.parameter_exploration import (
    random_triangle,
    generate_random_parameters,
)
from functions.backtesting.output_writers import (
    get_csv_header,
    format_csv_row,
    write_csv_header,
    append_csv_row,
    export_optimized_parameters,
)

__all__ = [
    "extract_model_identifier",
    "setup_output_paths",
    "generate_output_filename",
    "validate_configuration",
    "random_triangle",
    "generate_random_parameters",
    "get_csv_header",
    "format_csv_row",
    "write_csv_header",
    "append_csv_row",
    "export_optimized_parameters",
]
