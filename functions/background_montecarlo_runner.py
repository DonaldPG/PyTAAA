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
import subprocess
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
# Process management helpers (moved from MakeValuePlot.py, Item 11)
##############################################################################

def _kill_existing_montecarlo_processes(json_fn: str) -> None:
    """Kill any existing background Monte Carlo processes for the same config.
    
    Searches for running processes matching the background_montecarlo_runner
    pattern with the same JSON file argument and terminates them.
    
    Args:
        json_fn: Path to the JSON configuration file to match.
    
    Returns:
        None
    """
    try:
        # Get list of all processes
        ps_output = subprocess.check_output(
            ["ps", "aux"],
            text=True,
            stderr=subprocess.DEVNULL
        )
        
        # Normalize the json_fn path for comparison
        json_fn_normalized = os.path.abspath(json_fn)
        
        # Find matching processes
        killed_count = 0
        for line in ps_output.splitlines():
            # Look for background_montecarlo_runner processes
            if "background_montecarlo_runner" in line and "--json-file" in line:
                # Extract PID (second column in ps aux output)
                parts = line.split()
                if len(parts) < 2:
                    continue
                pid_str = parts[1]
                
                # Check if this process is using the same JSON file
                if json_fn in line or json_fn_normalized in line:
                    try:
                        pid = int(pid_str)
                        # Don't kill our own process
                        if pid != os.getpid():
                            os.kill(pid, 15)  # SIGTERM
                            killed_count += 1
                            print(f" [async] Killed existing Monte Carlo process (PID {pid})")
                    except (ValueError, ProcessLookupError, PermissionError) as e:
                        # Process may have already exited or permission denied
                        pass
        
        if killed_count == 0:
            print(" [async] No existing Monte Carlo processes found for this config")
    
    except Exception as e:
        # Don't fail the spawn if process detection fails
        print(f" [async] Warning: Could not check for existing processes: {e}")


def _spawn_background_montecarlo(json_fn: str, web_dir: str) -> None:
    """Spawn a detached background process for Monte Carlo backtest generation.

    Launches ``functions/background_montecarlo_runner.py`` as a fully
    detached subprocess (new session, stdout/stderr redirected to a log
    file).  Returns immediately; the caller does not wait for the backtest
    to complete.
    
    Before spawning, checks for and terminates any existing background
    Monte Carlo processes for the same configuration to prevent duplicate
    computations.

    Args:
        json_fn: Path to the JSON configuration file.
        web_dir: Directory where the log file will be written.

    Returns:
        None

    Side Effects:
        - Kills any existing background Monte Carlo processes for this config.
        - Spawns a detached background process.
        - Writes subprocess stdout/stderr to ``montecarlo_backtest.log``
          in *web_dir* (file is recreated, not appended).
    """
    # Kill any existing Monte Carlo processes for this config
    _kill_existing_montecarlo_processes(json_fn)
    
    project_root = os.path.dirname(os.path.dirname(__file__))

    env = os.environ.copy()
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = (
            f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
        )
    else:
        env["PYTHONPATH"] = project_root

    cmd = [
        sys.executable,
        "-m",
        "functions.background_montecarlo_runner",
        "--json-file",
        json_fn,
    ]

    # Ensure the web directory exists
    os.makedirs(web_dir, exist_ok=True)
    
    log_file = os.path.join(web_dir, "montecarlo_backtest.log")
    with open(log_file, "w") as log_fh:
        log_fh.write(
            f"[{datetime.datetime.now().isoformat()}] "
            f"Spawning background Monte Carlo backtest\n"
        )

    with open(log_file, "a") as log_fh:
        subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=log_fh,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )

    print(" [async] Background Monte Carlo backtest started.")
    print(f"   Log: {log_file}\n\n\n")




##############################################################################
# Script entry point
##############################################################################

if __name__ == "__main__":
    args = _parse_args()
    main(args.json_file)
