"""Background plot generator for PyTAAA.

Standalone CLI worker that loads serialized plot data from a pickle file
and generates portfolio PNG plots in parallel using ProcessPoolExecutor.
This script is designed to be launched as a fire-and-forget background
process so that the main program can return immediately.

Usage:
    python -m functions.background_plot_generator \\
        --data-file /tmp/plot_data.pkl \\
        --max-workers 2

Phase 1 of async plot generation implementation.
"""

import argparse
import datetime
import logging
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Configure module-level logger; callers may configure the root logger.
logger = logging.getLogger(__name__)


##############################################################################
# Worker helper types
##############################################################################

# Type alias for the data bundle passed to each worker function.
_PlotBundle = Dict[str, Any]


##############################################################################
# Data loading
##############################################################################

def load_plot_data(data_file: str) -> Dict[str, Any]:
    """Deserialize plot data from a pickle file.

    The pickle file is expected to contain the dictionary that was written
    by :func:`functions.output_generators._spawn_background_plot_generation`.

    Args:
        data_file: Absolute path to the pickle file.

    Returns:
        Dictionary with keys: ``adjClose``, ``symbols``, ``datearray``,
        ``signal2D``, ``signal2D_daily``, ``params``, ``output_dir``,
        ``lowChannel`` (optional), ``hiChannel`` (optional).

    Raises:
        FileNotFoundError: If *data_file* does not exist.
        ValueError: If the loaded object is not a dictionary.
    """
    if not os.path.isfile(data_file):
        raise FileNotFoundError(
            f"Plot data file not found: {data_file}"
        )
    with open(data_file, "rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected dict in pickle file, got {type(data).__name__}"
        )
    return data


##############################################################################
# Single-symbol worker functions (run inside worker processes)
##############################################################################

def generate_single_full_history_plot(bundle: _PlotBundle) -> str:
    """Generate the full-history PNG for a single symbol.

    Designed to be called from a :class:`ProcessPoolExecutor` worker.
    Each call is fully self-contained so that Matplotlib state is
    isolated per process.

    Args:
        bundle: Dictionary containing:
            - ``i``: integer index into ``symbols`` / ``adjClose``
            - ``symbol``: ticker string
            - ``adjClose_row``: 1-D array of adjusted close prices
            - ``datearray``: list/array of :class:`datetime.datetime` objects
            - ``signal2D_row``: 1-D uptrend signal array
            - ``quotes_despike_row``: despiked price array (1-D)
            - ``uptrendSignalMethod``: string
            - ``lowChannel_row``: optional 1-D array (may be ``None``)
            - ``hiChannel_row``: optional 1-D array (may be ``None``)
            - ``output_dir``: target directory
            - ``today_str``: formatted timestamp string

    Returns:
        Human-readable status string (success or error message).
    """
    # Import inside worker to avoid issues with forked processes
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    symbol = bundle["symbol"]
    plotfilepath = os.path.join(bundle["output_dir"], f"0_{symbol}.png")

    # Skip if less than 20 hours old
    if os.path.isfile(plotfilepath):
        mtime = datetime.datetime.fromtimestamp(
            os.path.getmtime(plotfilepath)
        )
        modified_hours = (
            (datetime.datetime.now() - mtime).days * 24
            + (datetime.datetime.now() - mtime).seconds / 3600
        )
        if modified_hours < 20.0:
            return f"SKIP {symbol}: full-history plot is recent"

    adjClose_row = bundle["adjClose_row"]
    datearray = bundle["datearray"]
    signal2D_row = bundle["signal2D_row"]
    quotes_despike_row = bundle["quotes_despike_row"]
    uptrendSignalMethod = bundle["uptrendSignalMethod"]
    lowChannel_row = bundle.get("lowChannel_row")
    hiChannel_row = bundle.get("hiChannel_row")
    today_str = bundle["today_str"]

    try:
        plt.clf()
        plt.grid(True)
        plt.plot(datearray, adjClose_row)
        plt.plot(datearray, signal2D_row * adjClose_row[-1], lw=0.2)

        number_nans = np.sum(np.isnan(quotes_despike_row))
        if number_nans == 0:
            plt.plot(datearray, quotes_despike_row)

        if (
            uptrendSignalMethod == "percentileChannels"
            and lowChannel_row is not None
        ):
            plt.plot(datearray, lowChannel_row, "m-")
            plt.plot(datearray, hiChannel_row, "m-")

        plot_text = str(adjClose_row[-7:])
        plt.text(datearray[50], 0, plot_text)

        # Add annotation with most recent date
        x_range = datearray[-1] - datearray[0]
        text_x = datearray[0] + datetime.timedelta(x_range.days / 20.0)
        adj_no_nans = adjClose_row[~np.isnan(adjClose_row)]
        text_y = (
            (np.max(adj_no_nans) - np.min(adj_no_nans)) * 0.085
            + np.min(adj_no_nans)
        )

        plt.text(
            text_x,
            text_y,
            f"most recent value from {datearray[-1]}\n"
            f"plotted at {today_str}\n"
            f"value = {adjClose_row[-1]}",
            fontsize=8,
        )
        plt.title(f"{symbol}")
        plt.yscale("log")

        plt.savefig(plotfilepath, format="png")
        plt.close("all")
        return f"OK {symbol}: full-history plot saved"
    except Exception as exc:  # noqa: BLE001
        plt.close("all")
        return f"ERROR {symbol}: full-history plot failed – {exc}"


def generate_single_recent_plot(bundle: _PlotBundle) -> str:
    """Generate the recent-history PNG for a single symbol.

    Designed to be called from a :class:`ProcessPoolExecutor` worker.

    Args:
        bundle: Dictionary containing:
            - ``symbol``: ticker string
            - ``adjClose_row``: 1-D array of adjusted close prices
            - ``datearray``: list/array of datetime objects
            - ``signal2D_row``: 1-D monthly uptrend signal array
            - ``signal2D_daily_row``: 1-D daily uptrend signal array
            - ``quotes_despike_row``: despiked price array (1-D)
            - ``firstdate_index``: integer — index of first date for plot
            - ``uptrendSignalMethod``: string
            - ``lowChannel_row``: optional 1-D array
            - ``hiChannel_row``: optional 1-D array
            - ``lowerTrend``: trend channel lower bound
            - ``upperTrend``: trend channel upper bound
            - ``NoGapLowerTrend``: no-gap trend channel lower bound
            - ``NoGapUpperTrend``: no-gap trend channel upper bound
            - ``params``: dict with ``numdaysinfit``, ``numdaysinfit2``,
              ``offset`` keys
            - ``output_dir``: target directory
            - ``today_str``: formatted timestamp string

    Returns:
        Human-readable status string.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    symbol = bundle["symbol"]
    plotfilepath = os.path.join(
        bundle["output_dir"], f"0_recent_{symbol}.png"
    )

    # Skip if less than 20 hours old
    if os.path.isfile(plotfilepath):
        mtime = datetime.datetime.fromtimestamp(
            os.path.getmtime(plotfilepath)
        )
        modified_hours = (
            (datetime.datetime.now() - mtime).days * 24
            + (datetime.datetime.now() - mtime).seconds / 3600
        )
        if modified_hours < 20.0:
            return f"SKIP {symbol}: recent plot is recent"

    adjClose_row = bundle["adjClose_row"]
    datearray = bundle["datearray"]
    signal2D_row = bundle["signal2D_row"]
    signal2D_daily_row = bundle["signal2D_daily_row"]
    quotes_despike_row = bundle["quotes_despike_row"]
    firstdate_index = bundle["firstdate_index"]
    uptrendSignalMethod = bundle["uptrendSignalMethod"]
    lowChannel_row = bundle.get("lowChannel_row")
    hiChannel_row = bundle.get("hiChannel_row")
    lowerTrend = bundle["lowerTrend"]
    upperTrend = bundle["upperTrend"]
    NoGapLowerTrend = bundle["NoGapLowerTrend"]
    NoGapUpperTrend = bundle["NoGapUpperTrend"]
    params = bundle["params"]
    today_str = bundle["today_str"]

    try:
        plt.figure(10)
        plt.clf()
        plt.grid(True)

        plt.plot(
            datearray[firstdate_index:],
            signal2D_row[firstdate_index:] * adjClose_row[-1],
            lw=0.25,
            alpha=0.6,
        )
        plt.plot(
            datearray[firstdate_index:],
            signal2D_daily_row[firstdate_index:] * adjClose_row[-1],
            lw=0.25,
            alpha=0.6,
        )
        plt.plot(
            datearray[firstdate_index:],
            quotes_despike_row[firstdate_index:],
            lw=0.15,
        )

        adj_no_nans = adjClose_row[~np.isnan(adjClose_row)]
        ymax = np.around(np.max(adj_no_nans[firstdate_index:]) * 1.1)

        if (
            uptrendSignalMethod == "percentileChannels"
            and lowChannel_row is not None
        ):
            ymin = np.around(
                np.min(lowChannel_row[firstdate_index:]) * 0.85
            )
        else:
            ymin = np.around(
                np.min(adjClose_row[firstdate_index:]) * 0.90
            )

        plt.ylim((ymin, ymax))
        xmin = datearray[firstdate_index]
        xmax = datearray[-1] + datetime.timedelta(10)
        plt.xlim((xmin, xmax))

        if (
            uptrendSignalMethod == "percentileChannels"
            and lowChannel_row is not None
        ):
            plt.plot(
                datearray[firstdate_index:], lowChannel_row[firstdate_index:], "m-"
            )
            plt.plot(
                datearray[firstdate_index:], hiChannel_row[firstdate_index:], "m-"
            )

        # Plot trend channels
        relativedates = list(
            range(
                -params["numdaysinfit"] - params["offset"],
                -params["offset"] + 1,
            )
        )
        plt.plot(
            np.array(datearray)[relativedates], upperTrend, "y-", lw=0.5
        )
        plt.plot(
            np.array(datearray)[relativedates], lowerTrend, "y-", lw=0.5
        )
        plt.plot(
            [datearray[-1]],
            [(upperTrend[-1] + lowerTrend[-1]) / 2.0],
            "y.",
            ms=10,
            alpha=0.6,
        )

        plt.plot(
            np.array(datearray)[-params["numdaysinfit2"]:],
            NoGapUpperTrend,
            ls="-",
            c=(0, 0, 1),
            lw=1.0,
        )
        plt.plot(
            np.array(datearray)[-params["numdaysinfit2"]:],
            NoGapLowerTrend,
            ls="-",
            c=(0, 0, 1),
            lw=1.0,
        )
        plt.plot(
            [datearray[-1]],
            [(NoGapUpperTrend[-1] + NoGapLowerTrend[-1]) / 2.0],
            ".",
            c=(0, 0, 1),
            ms=10,
            alpha=0.6,
        )

        plt.plot(
            datearray[firstdate_index:],
            adjClose_row[firstdate_index:],
            "k-",
            lw=0.5,
        )

        plot_text = str(adjClose_row[-7:])
        plt.text(datearray[firstdate_index + 10], ymin, plot_text, fontsize=10)

        # Add annotation with most recent date
        x_range = datearray[-1] - datearray[firstdate_index]
        text_x = datearray[firstdate_index] + datetime.timedelta(
            x_range.days / 20.0
        )
        text_y = (ymax - ymin) * 0.085 + ymin

        plt.text(
            text_x,
            text_y,
            f"most recent value from {datearray[-1]}\n"
            f"plotted at {today_str}\n"
            f"value = {adjClose_row[-1]}",
            fontsize=8,
        )
        plt.title(f"{symbol}")

        plt.tick_params(axis="both", which="major", labelsize=8)
        plt.tick_params(axis="both", which="minor", labelsize=6)

        plt.savefig(plotfilepath, format="png")
        plt.close("all")
        return f"OK {symbol}: recent plot saved"
    except Exception as exc:  # noqa: BLE001
        plt.close("all")
        return f"ERROR {symbol}: recent plot failed – {exc}"


##############################################################################
# Orchestration
##############################################################################

def _build_full_history_bundles(data: Dict[str, Any]) -> List[_PlotBundle]:
    """Build per-symbol bundles for full-history plot generation.

    Args:
        data: Deserialized plot data dictionary (from pickle).

    Returns:
        List of bundles, one per symbol.
    """
    from functions.TAfunctions import despike_2D  # noqa: PLC0415

    adjClose = data["adjClose"]
    symbols = data["symbols"]
    datearray = data["datearray"]
    signal2D = data["signal2D"]
    params = data["params"]
    output_dir = data["output_dir"]
    lowChannel = data.get("lowChannel")
    hiChannel = data.get("hiChannel")
    today_str = datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")

    LongPeriod = params["LongPeriod"]
    stddevThreshold = float(params["stddevThreshold"])
    uptrendSignalMethod = params["uptrendSignalMethod"]

    bundles = []
    for i, symbol in enumerate(symbols):
        row = adjClose[i, :].copy()
        despike_in = row.reshape(1, len(row))
        despike_out = despike_2D(
            despike_in, LongPeriod, stddevThreshold=stddevThreshold
        )
        bundle: _PlotBundle = {
            "i": i,
            "symbol": symbol,
            "adjClose_row": row,
            "datearray": list(datearray),
            "signal2D_row": signal2D[i, :].copy(),
            "quotes_despike_row": despike_out[0, :].copy(),
            "uptrendSignalMethod": uptrendSignalMethod,
            "lowChannel_row": (
                lowChannel[i, :].copy() if lowChannel is not None else None
            ),
            "hiChannel_row": (
                hiChannel[i, :].copy() if hiChannel is not None else None
            ),
            "output_dir": output_dir,
            "today_str": today_str,
        }
        bundles.append(bundle)
    return bundles


def _build_recent_bundles(
    data: Dict[str, Any], firstdate_index: int
) -> List[_PlotBundle]:
    """Build per-symbol bundles for recent-history plot generation.

    Args:
        data: Deserialized plot data dictionary (from pickle).
        firstdate_index: Index of the first date to include in recent plots.

    Returns:
        List of bundles, one per symbol.
    """
    from functions.TAfunctions import (  # noqa: PLC0415
        despike_2D,
        recentTrendAndMidTrendChannelFitWithAndWithoutGap,
    )

    adjClose = data["adjClose"]
    symbols = data["symbols"]
    datearray = data["datearray"]
    signal2D = data["signal2D"]
    signal2D_daily = data["signal2D_daily"]
    params = data["params"]
    output_dir = data["output_dir"]
    lowChannel = data.get("lowChannel")
    hiChannel = data.get("hiChannel")
    today_str = datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")

    LongPeriod = params["LongPeriod"]
    stddevThreshold = float(params["stddevThreshold"])
    uptrendSignalMethod = params["uptrendSignalMethod"]

    bundles = []
    for i, symbol in enumerate(symbols):
        row = adjClose[i, :].copy()
        despike_in = row.reshape(1, len(row))
        despike_out = despike_2D(
            despike_in, LongPeriod, stddevThreshold=stddevThreshold
        )
        # Re-scale despiked quotes to match raw at firstdate_index
        despike_out *= row[firstdate_index] / despike_out[0, firstdate_index]

        try:
            lowerTrend, upperTrend, NoGapLowerTrend, NoGapUpperTrend = (
                recentTrendAndMidTrendChannelFitWithAndWithoutGap(
                    row,
                    minperiod=params["minperiod"],
                    maxperiod=params["maxperiod"],
                    incperiod=params["incperiod"],
                    numdaysinfit=params["numdaysinfit"],
                    numdaysinfit2=params["numdaysinfit2"],
                    offset=params["offset"],
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Trend fit failed for %s: %s – skipping recent plot", symbol, exc
            )
            continue

        bundle: _PlotBundle = {
            "symbol": symbol,
            "adjClose_row": row,
            "datearray": list(datearray),
            "signal2D_row": signal2D[i, :].copy(),
            "signal2D_daily_row": signal2D_daily[i, :].copy(),
            "quotes_despike_row": despike_out[0, :].copy(),
            "firstdate_index": firstdate_index,
            "uptrendSignalMethod": uptrendSignalMethod,
            "lowChannel_row": (
                lowChannel[i, :].copy() if lowChannel is not None else None
            ),
            "hiChannel_row": (
                hiChannel[i, :].copy() if hiChannel is not None else None
            ),
            "lowerTrend": lowerTrend,
            "upperTrend": upperTrend,
            "NoGapLowerTrend": NoGapLowerTrend,
            "NoGapUpperTrend": NoGapUpperTrend,
            "params": params,
            "output_dir": output_dir,
            "today_str": today_str,
        }
        bundles.append(bundle)
    return bundles


def main(data_file: str, max_workers: int = 2) -> None:
    """Load plot data and generate plots in parallel.

    This is the entry-point called by the background process.  It
    deletes the temporary pickle file after loading to avoid leaving
    stale files on disk.

    Args:
        data_file: Path to the pickle file written by the main process.
        max_workers: Maximum number of parallel worker processes.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print(
        f"[background_plot_generator] Starting at "
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(f"[background_plot_generator] data_file = {data_file}")
    print(f"[background_plot_generator] max_workers = {max_workers}")

    # Load and immediately delete the pickle file
    try:
        data = load_plot_data(data_file)
    except Exception as exc:  # noqa: BLE001
        print(f"[background_plot_generator] FATAL: could not load data – {exc}")
        sys.exit(1)
    finally:
        try:
            os.remove(data_file)
        except OSError:
            pass

    adjClose = data["adjClose"]
    symbols = data["symbols"]
    datearray = data["datearray"]

    # Determine firstdate_index for 2013+ recent plots
    firstdate_index = 0
    for ii in range(len(datearray)):
        if (
            datearray[ii].year > datearray[ii - 1].year
            and datearray[ii].year == 2013
        ):
            firstdate_index = ii
            break

    print(
        f"[background_plot_generator] Generating plots for "
        f"{len(symbols)} symbols with {max_workers} workers"
    )

    ##########################################################################
    # 1. Full-history plots
    ##########################################################################
    full_bundles = _build_full_history_bundles(data)
    completed_full = 0
    errors_full = 0
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(generate_single_full_history_plot, b): b["symbol"]
            for b in full_bundles
        }
        for future in as_completed(futures):
            result_msg = future.result()
            print(f"[background_plot_generator] {result_msg}")
            if result_msg.startswith("ERROR"):
                errors_full += 1
            else:
                completed_full += 1

    print(
        f"[background_plot_generator] Full-history plots: "
        f"{completed_full} done, {errors_full} errors"
    )

    ##########################################################################
    # 2. Recent-history plots
    ##########################################################################
    recent_bundles = _build_recent_bundles(data, firstdate_index)
    completed_recent = 0
    errors_recent = 0
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(generate_single_recent_plot, b): b["symbol"]
            for b in recent_bundles
        }
        for future in as_completed(futures):
            result_msg = future.result()
            print(f"[background_plot_generator] {result_msg}")
            if result_msg.startswith("ERROR"):
                errors_recent += 1
            else:
                completed_recent += 1

    print(
        f"[background_plot_generator] Recent plots: "
        f"{completed_recent} done, {errors_recent} errors"
    )
    print(
        f"[background_plot_generator] Finished at "
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


##############################################################################
# CLI entry point
##############################################################################

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description="Generate portfolio PNG plots in parallel background workers."
    )
    parser.add_argument(
        "--data-file",
        required=True,
        help="Path to the pickle file containing serialized plot data.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Maximum number of parallel worker processes (default: 2).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    main(args.data_file, args.max_workers)
