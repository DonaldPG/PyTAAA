"""Configuration helpers for the backtesting package.

This module provides utility functions for extracting model identifiers
from filesystem paths, setting up output directories, generating output
filenames, and validating JSON configuration dictionaries.
"""

import os
from typing import Tuple

from functions.GetParams import get_webpage_store, get_performance_store
from functions.logger_config import get_logger

logger = get_logger(__name__, log_file="pytaaa_backtest_montecarlo.log")

#############################################################################
# Required and optional configuration keys
#############################################################################

_REQUIRED_KEYS = ("symbols_file", "performance_store", "webpage")
_DEFAULT_TRIALS = 250


def extract_model_identifier(webpage_path: str) -> str:
    """Extract the model identifier from a webpage path.

    The model identifier is the second-to-last component of the path.
    For example, ``/path/to/sp500_pine/webpage`` → ``sp500_pine``.

    Args:
        webpage_path: Filesystem path ending in ``webpage``.

    Returns:
        Model identifier string (e.g. ``sp500_pine``, ``naz100_hma``).

    Raises:
        ValueError: If the path has fewer than two components or does not
            end with ``webpage``.

    Example:
        >>> extract_model_identifier("/data/sp500_pine/webpage")
        'sp500_pine'
    """
    parts = webpage_path.rstrip("/").split("/")
    if len(parts) < 2:
        raise ValueError(
            f"Path too short to extract model identifier: {webpage_path!r}"
        )
    if parts[-1] != "webpage":
        raise ValueError(
            f"Path does not end with 'webpage': {webpage_path!r}"
        )
    return parts[-2]


def setup_output_paths(json_fn: str) -> Tuple[str, str, str]:
    """Set up output paths for a Monte Carlo backtest run.

    Derives the model identifier from the ``webpage`` config key,
    constructs the output directory as ``<model_base>/pytaaa_backtest/``
    (one level up from performance_store), and creates that directory 
    if it does not already exist.

    Args:
        json_fn: Path to the JSON configuration file.

    Returns:
        Tuple of (model_id, output_dir, perf_store):
            - model_id: Model identifier string (e.g. ``sp500_pine``).
            - output_dir: Absolute path to the PNG/CSV output directory.
            - perf_store: Absolute path to the performance store directory.

    Example:
        >>> model_id, output_dir, perf_store = setup_output_paths(
        ...     "pytaaa_sp500_pine.json"
        ... )
    """
    webpage_path = get_webpage_store(json_fn)
    perf_store = get_performance_store(json_fn)

    model_id = extract_model_identifier(webpage_path)
    # Go up one level from perf_store (removes /data_store) and add /pytaaa_backtest
    model_base = os.path.dirname(perf_store)
    output_dir = os.path.join(model_base, "pytaaa_backtest")

    os.makedirs(output_dir, exist_ok=True)
    logger.debug(
        "Output paths: model=%s, output_dir=%s", model_id, output_dir
    )

    return model_id, output_dir, perf_store


def generate_output_filename(
    model_id: str,
    file_type: str,
    date_str: str,
    suffix: str = "",
) -> str:
    """Generate a standardised output filename (no directory path).

    Args:
        model_id: Model identifier string (e.g. ``sp500_pine``).
        file_type: File type tag (e.g. ``montecarlo``, ``optimized``).
        date_str: Date string (e.g. ``2025-6-1``).
        suffix: Optional extra suffix appended after date_str.

    Returns:
        Filename string without directory path, e.g.
        ``sp500_pine_montecarlo_2025-6-1_run2501a``.

    Example:
        >>> generate_output_filename(
        ...     "sp500_pine", "montecarlo", "2025-6-1", "run2501a"
        ... )
        'sp500_pine_montecarlo_2025-6-1_run2501a'
    """
    parts = [model_id, file_type, date_str]
    if suffix:
        parts.append(suffix)
    return "_".join(parts)


def validate_configuration(params: dict) -> dict:
    """Validate required keys and apply defaults to a configuration dict.

    Checks that all required keys (``symbols_file``, ``performance_store``,
    ``webpage``) are present.  Sets ``backtest_monte_carlo_trials`` to
    ``250`` if not already supplied.

    Args:
        params: Configuration dictionary (typically loaded from JSON).

    Returns:
        The same dictionary with defaults applied.

    Raises:
        KeyError: If any required key is missing.

    Example:
        >>> validated = validate_configuration({"symbols_file": "...",
        ...     "performance_store": "...", "webpage": "..."})
        >>> validated["backtest_monte_carlo_trials"]
        250
    """
    for key in _REQUIRED_KEYS:
        if key not in params:
            raise KeyError(
                f"Missing required configuration key: {key!r}"
            )

    params.setdefault("backtest_monte_carlo_trials", _DEFAULT_TRIALS)
    logger.debug("Configuration validated successfully.")
    return params
