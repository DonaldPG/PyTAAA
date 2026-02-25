"""Background Monte Carlo backtest runner for PyTAAA.

Minimal CLI wrapper around :func:`functions.dailyBacktest_pctLong.dailyBacktest_pctLong`
designed to be launched as a fire-and-forget detached subprocess so that
the main program can return immediately.

Usage::

    python -m functions.background_montecarlo_runner --json-file /path/to/config.json

The script forces the ``Agg`` Matplotlib backend so it can run safely in
headless environments (no display).
"""

import argparse
import datetime
import logging
import os
import sys
from typing import Optional, List

import matplotlib
matplotlib.use("Agg")

# Configure module-level logger.
logger = logging.getLogger(__name__)


##############################################################################
# CLI helpers
##############################################################################

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed :class:`argparse.Namespace` with attribute ``json_file``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Fire-and-forget background Monte Carlo backtest runner for PyTAAA."
        )
    )
    parser.add_argument(
        "--json-file",
        required=True,
        help="Path to the JSON configuration file.",
    )
    return parser.parse_args(argv)


##############################################################################
# Main entry point
##############################################################################

def main(json_file: str) -> None:
    """Run the Monte Carlo backtest and log progress.

    Calls :func:`functions.dailyBacktest_pctLong.dailyBacktest_pctLong`
    with *json_file* and prints timestamped start/finish messages.

    Args:
        json_file: Path to the JSON configuration file passed through to
            ``dailyBacktest_pctLong``.

    Returns:
        None
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    start_time = datetime.datetime.now()
    print(
        f"[{start_time:%H:%M:%S}] Starting Monte Carlo backtest "
        f"(json_file={json_file})"
    )

    try:
        from functions.dailyBacktest_pctLong import (  # noqa: PLC0415
            dailyBacktest_pctLong,
        )
        dailyBacktest_pctLong(json_file, verbose=True)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[{datetime.datetime.now():%H:%M:%S}] ERROR during Monte Carlo "
            f"backtest: {exc}"
        )
        logger.exception("Monte Carlo backtest failed")
        sys.exit(1)

    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    print(
        f"[{end_time:%H:%M:%S}] Monte Carlo backtest complete "
        f"(elapsed {elapsed})"
    )


##############################################################################
# Script entry point
##############################################################################

if __name__ == "__main__":
    args = _parse_args()
    main(args.json_file)
