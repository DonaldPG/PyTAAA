"""Raw file I/O for PyTAAA configuration and status files.

This module handles only disk operations — opening INI files, parsing
the ``PyTAAA_status.params`` line format, and appending new status
entries. It has no dependency on the JSON config cache or on any
business-logic helpers (``TAfunctions``, ``config_cache``).

Functions:
    from_config_file: Parse an INI-style config file into ConfigParser
    parse_pytaaa_status: Extract dates and values from status .params file
    _write_status_line: Append one cumulative-value line to status file
"""

import os
import configparser
import datetime
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def from_config_file(config_filename: str) -> configparser.ConfigParser:
    """Load configuration from an INI-style config file.

    Args:
        config_filename: Path to the INI configuration file.

    Returns:
        Populated ConfigParser instance.

    Note:
        Legacy function. New code should use
        ``config_accessors.get_json_params()``.
    """
    with open(config_filename, "r") as fid:
        config = configparser.ConfigParser(strict=False)
        config.read_file(fid)
    return config


def parse_pytaaa_status(file_path: str) -> Tuple[list, list]:
    """Parse ``PyTAAA_status.params`` to extract dates and portfolio values.

    Each non-blank line in the file is expected to have the format::

        cumu_value: <datetime> <portfolio_value> <signal> <traded_value>

    where ``parts[0]`` is the label ``cumu_value:``, ``parts[1]`` is
    the date (``YYYY-MM-DD``), ``parts[2]`` is the time, and
    ``parts[3]`` is the portfolio value.

    Args:
        file_path: Absolute path to the ``PyTAAA_status.params`` file.

    Returns:
        Tuple of two lists ``(dates, portfolio_values)`` where
        ``dates`` contains the raw date strings and
        ``portfolio_values`` contains ``float`` values.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
    """
    dates: list = []
    portfolio_values: list = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue  # Skip short or blank lines.
            try:
                dates.append(parts[1])
                portfolio_values.append(float(parts[3]))
            except (ValueError, IndexError) as err:
                logger.warning(
                    "Skipping malformed status line: %s — %s",
                    line.strip(), err,
                )

    return dates, portfolio_values


def _write_status_line(
    status_filename: str,
    cumu_status: float,
    last_signal: int,
    traded_value: float,
) -> None:
    """Append one cumulative-value record to the status params file.

    Writes a single line in the format used by ``PyTAAA_status.params``::

        cumu_value: <ISO-datetime> <cumu_status> <last_signal> <traded_value>

    Args:
        status_filename: Absolute path to the ``PyTAAA_status.params``
            file to append to.
        cumu_status: Current portfolio cumulative value.
        last_signal: Most-recent trading signal (1 = long, 0 = cash).
        traded_value: Signal-timed portfolio value at the last signal
            date (as returned by ``compute_long_hold_signal``).
    """
    with open(status_filename, "a") as fh:
        fh.write(
            "cumu_value: "
            + str(datetime.datetime.now()) + " "
            + str(cumu_status) + " "
            + str(last_signal) + " "
            + str(traded_value) + "\n"
        )
