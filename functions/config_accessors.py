"""Typed configuration accessors for PyTAAA.

All functions in this module accept a ``json_fn`` path and return a
typed value extracted from the JSON configuration cache.  They do not
perform raw file parsing (that lives in ``config_loader``); instead they
read from ``functions.config_cache`` and optionally consult helper
modules (``configparser`` for legacy ``.params`` files,
``TAfunctions`` for signal computation).

Functions:
    get_json_params: Build the main params dict from a JSON config file
    get_json_ftp_params: Extract FTP connection parameters
    get_symbols_file: Resolve the path to the symbols list file
    get_performance_store: Resolve the performance-history directory
    get_webpage_store: Resolve the webpage output directory
    get_web_output_dir: Resolve the web output directory
    get_central_std_values: Extract normalization central and std values
    get_holdings: Load portfolio holdings from ``.params`` files
    get_json_status: Read the last cumulative portfolio value
    get_status: Alias for get_json_status (deprecated)
    compute_long_hold_signal: Compute MA-channel signal on status history
    put_status: Append current portfolio value to the status file
    GetIP: Return the machine's external IP address
    GetEdition: Return a string describing the running platform
"""

import os
import configparser
import json
import datetime
import numpy as np
from typing import Dict, Optional, Tuple

from functions.logger_config import get_logger
from functions.config_cache import config_cache
from functions.config_loader import _write_status_line

logger = get_logger(__name__, log_file="GetParams.log")


# ---------------------------------------------------------------------------
# Path accessors
# ---------------------------------------------------------------------------

def get_symbols_file(json_fn: str) -> str:
    """Get path to the file containing the list of stock symbols.

    Reads the JSON configuration to determine which symbol list to use
    (Naz100 or SP500) and constructs the full path to the symbols file.

    Args:
        json_fn: Path to the JSON configuration file.

    Returns:
        Full path to the symbols file
        (e.g. ``"symbols/Naz100_Symbols.txt"``).

    Example:
        >>> symbols_file = get_symbols_file("config/pytaaa_naz100_pine.json")
        >>> print(symbols_file)
        '/path/to/symbols/Naz100_Symbols.txt'
    """
    params = get_json_params(json_fn)
    stockList = params["stockList"]

    if "symbols_file" in params:
        return params["symbols_file"]

    top_dir = os.path.split(json_fn)[0]
    symbol_directory = os.path.join(top_dir, "symbols")
    if stockList == "Naz100":
        symbol_file = "Naz100_Symbols.txt"
    elif stockList == "SP500":
        symbol_file = "SP500_Symbols.txt"
    else:
        symbol_file = f"{stockList}_Symbols.txt"
    return os.path.join(symbol_directory, symbol_file)


def get_performance_store(json_fn: str) -> str:
    """Get the directory where performance history files are stored.

    Performance history files (``*.params``) contain backtest results,
    portfolio allocations, and trading history for each model.

    Args:
        json_fn: Path to the JSON configuration file.

    Returns:
        Path to the ``performance_store`` directory from the config.

    Raises:
        FileNotFoundError: If ``json_fn`` does not exist.
        KeyError: If the ``Valuation`` section or
            ``performance_store`` key is missing.

    Example:
        >>> store = get_performance_store("config/pytaaa_sp500_pine.json")
        >>> print(store)
        '/Users/user/pyTAAA_data_static/sp500_pine/data_store'
    """
    config = config_cache.get(json_fn)
    valuation_section = config.get("Valuation")
    return valuation_section["performance_store"]


def get_hdf_store(json_fn: str) -> Optional[str]:
    """Return the explicit HDF5 file path from config, or None.

    When the ``Valuation`` section of the JSON config contains an
    ``hdf_store`` key, its value is returned as the absolute path to
    the HDF5 quotes file.  Callers should fall back to the default
    path-construction logic when this function returns ``None``.

    Args:
        json_fn: Path to the JSON configuration file.

    Returns:
        Absolute path to the HDF5 file if ``hdf_store`` is present in
        the config, otherwise ``None``.

    Example:
        >>> path = get_hdf_store("config/pytaaa_naz100_pine_nans.json")
        >>> print(path)
        '/Users/user/pyTAAA_data/naz100_pine/data_store/Naz100_Symbols_nans.hdf5'
    """
    config = config_cache.get(json_fn)
    valuation_section = config.get("Valuation", {})
    return valuation_section.get("hdf_store") or None


def get_webpage_store(json_fn: str) -> str:
    """Get the directory where updated webpage files are created.

    Args:
        json_fn: Path to the JSON configuration file.

    Returns:
        Path to the webpage output directory from the config.

    Raises:
        FileNotFoundError: If ``json_fn`` does not exist.
        KeyError: If the ``Valuation`` section or ``webpage`` key is
            missing.

    Example:
        >>> webpage = get_webpage_store("config/pytaaa_naz100_hma.json")
        >>> print(webpage)
        '/Users/user/pyTAAA_data_static/naz100_hma/webpage'
    """
    config = config_cache.get(json_fn)
    valuation_section = config.get("Valuation")
    return valuation_section["webpage"]


def get_web_output_dir(json_fn: str) -> str:
    """Get the web output directory from the JSON configuration.

    Args:
        json_fn: Path to the JSON configuration file.

    Returns:
        Web output directory path string.

    Raises:
        FileNotFoundError: If ``json_fn`` does not exist.
        KeyError: If ``web_output_dir`` key is missing.
        json.JSONDecodeError: If the JSON file is malformed.
    """
    config = config_cache.get(json_fn)
    if "web_output_dir" not in config:
        raise KeyError(
            "'web_output_dir' key not found in JSON configuration"
        )
    return config["web_output_dir"]


# ---------------------------------------------------------------------------
# Section accessors
# ---------------------------------------------------------------------------

def get_central_std_values(json_fn: str) -> Dict[str, Dict[str, float]]:
    """Get normalization values from the JSON configuration.

    Args:
        json_fn: Path to the JSON configuration file.

    Returns:
        Dict with keys ``"central_values"`` and ``"std_values"``, each
        mapping metric names to ``float`` normalization constants.

    Raises:
        FileNotFoundError: If ``json_fn`` does not exist.
        KeyError: If required normalization keys are missing.
        json.JSONDecodeError: If the JSON file is malformed.
    """
    config = config_cache.get(json_fn)

    model_selection = config.get("model_selection")
    if model_selection is None:
        raise KeyError(
            "'model_selection' section not found in JSON configuration"
        )

    normalization = model_selection.get("normalization")
    if normalization is None:
        raise KeyError(
            "'normalization' section not found in model_selection"
        )

    central_values = normalization.get("central_values")
    std_values = normalization.get("std_values")

    if central_values is None:
        raise KeyError(
            "'central_values' not found in normalization section"
        )
    if std_values is None:
        raise KeyError(
            "'std_values' not found in normalization section"
        )

    return {"central_values": central_values, "std_values": std_values}


def get_json_ftp_params(
    json_fn: str, verbose: bool = False
) -> Dict[str, str]:
    """Extract FTP connection parameters from the JSON configuration.

    Args:
        json_fn: Path to the JSON configuration file.
        verbose: If ``True``, log the FTP section contents at INFO level.

    Returns:
        Dict with keys ``ftpHostname``, ``ftpUsername``, ``ftpPassword``,
        ``remotepath``, ``remoteIP``.
    """
    config = config_cache.get(json_fn)
    ftp_section = config.get("FTP")

    if verbose:
        logger.info("FTP Section: %s", ftp_section)

    ftpparams: Dict[str, str] = {
        "ftpHostname": str(ftp_section["hostname"]),
        "ftpUsername": str(ftp_section["username"]),
        "ftpPassword": str(ftp_section["password"]),
        "remotepath":  str(ftp_section["remotepath"]),
        "remoteIP":    str(ftp_section["remoteIP"]),
    }
    return ftpparams


# ---------------------------------------------------------------------------
# Holdings and status readers
# ---------------------------------------------------------------------------

def get_holdings(json_fn: str) -> Dict:
    """Load the current portfolio holdings from ``.params`` files.

    Reads ``PyTAAA_holdings.params`` and ``PyTAAA_ranks.params`` from
    the performance store directory defined in the JSON config.

    Args:
        json_fn: Path to the JSON configuration file.

    Returns:
        Dict with keys ``"stocks"``, ``"shares"``, ``"buyprice"``,
        ``"cumulativecashin"``, and ``"ranks"`` — all as string lists.
    """
    holdings: Dict = {}

    p_store = get_performance_store(json_fn)
    config_filename = os.path.join(p_store, "PyTAAA_holdings.params")

    config = configparser.ConfigParser(strict=False)
    with open(config_filename, "r") as configfile:
        config.read_file(configfile)

    holdings["stocks"] = config.get("Holdings", "stocks").split()
    holdings["shares"] = config.get("Holdings", "shares").split()
    holdings["buyprice"] = config.get("Holdings", "buyprice").split()
    holdings["cumulativecashin"] = (
        config.get("Holdings", "cumulativecashin").split()
    )

    config_filename = os.path.join(p_store, "PyTAAA_ranks.params")
    with open(config_filename, "r") as configfile:
        config.read_file(configfile)
    symbols = config.get("Ranks", "symbols").split()
    ranks = config.get("Ranks", "ranks").split()

    holdings_ranks = []
    for holding in holdings["stocks"]:
        for j, symbol in enumerate(symbols):
            if symbol == holding:
                holdings_ranks.append(ranks[j])
                break
    holdings["ranks"] = holdings_ranks
    return holdings


def get_json_status(json_fn: str) -> str:
    """Read the last cumulative portfolio value from the status file.

    Args:
        json_fn: Path to the JSON configuration file.

    Returns:
        Most-recent cumulative portfolio value as a string.
    """
    json_folder = get_performance_store(json_fn)
    status_filename = os.path.join(json_folder, "PyTAAA_status.params")

    config = configparser.ConfigParser(strict=False)
    with open(status_filename, "r") as configfile:
        config.read_file(configfile)

    return config.get("Status", "cumu_value").split()[-3]


def get_status(json_fn: str) -> str:
    """Alias for ``get_json_status``; kept for backward compatibility.

    .. deprecated::
        Use :func:`get_json_status` directly.

    Args:
        json_fn: Path to the JSON configuration file.

    Returns:
        Most-recent cumulative portfolio value as a string.
    """
    p_store = get_performance_store(json_fn)
    status_filename = os.path.join(p_store, "PyTAAA_status.params")

    config = configparser.ConfigParser(strict=False)
    with open(status_filename, "r") as configfile:
        config.read_file(configfile)

    return config.get("Status", "cumu_value").split()[-3]


# ---------------------------------------------------------------------------
# Main params builder
# ---------------------------------------------------------------------------

def get_json_params(json_fn: str, verbose: bool = False) -> Dict:
    """Build the main trading parameters dictionary from a JSON config.

    Reads all sections of the configuration file, applies type
    coercions, resolves defaults for optional keys, and returns a flat
    ``params`` dict consumed by every pipeline module.

    Args:
        json_fn: Path to the JSON configuration file.
        verbose: If ``True``, log the full JSON content at INFO level.

    Returns:
        Dict containing all trading parameters.  Required keys include:
        ``fromaddr``, ``toaddrs``, ``PW``, ``runtime``, ``pausetime``,
        ``quote_server``, ``numberStocksTraded``, ``trade_cost``,
        ``monthsToHold``, ``LongPeriod``, ``stddevThreshold``,
        ``MA1``, ``MA2``, ``MA3``, ``rankThresholdPct``,
        ``uptrendSignalMethod``, ``stockList``, ``symbols_file``.
    """
    config = config_cache.get(json_fn)

    if verbose:
        logger.info("JSON config:\n%s", json.dumps(config, indent=4))

    runtime = config.get("Setup")["runtime"]
    pausetime = config.get("Setup")["pausetime"]

    if len(runtime) == 1:
        runtime.join("days")
    if len(pausetime) == 1:
        pausetime.join("hours")

    def _time_factor(unit: str) -> float:
        return {
            "seconds": 1,
            "minutes": 60,
            "hours": 3_600,
            "days": 86_400,
            "months": 86_400 * 30.4,
            "years": 86_400 * 365.25,
        }.get(unit, 86_400)  # Default: days.

    rt_parts = runtime.split(" ")
    max_uptime = int(rt_parts[0]) * _time_factor(rt_parts[1])

    pt_parts = pausetime.split(" ")
    seconds_between_runs = int(pt_parts[0]) * _time_factor(pt_parts[1])

    send_texts_raw = str(
        config.get("Text_from_email")["send_texts"]
    ).lower()

    valuation_section = config.get("Valuation")
    params: Dict = {
        "fromaddr":             str(config.get("Email")["From"]),
        "toaddrs":              str(config.get("Email")["To"]),
        "toSMS":                config.get("Text_from_email")["phoneEmail"],
        "send_texts":           send_texts_raw == "true",
        "PW":                   str(config.get("Email")["PW"]),
        "runtime":              max_uptime,
        "pausetime":            seconds_between_runs,
        "quote_server":         config.get("stock_server")[
                                    "quote_download_server"
                                ],
        "numberStocksTraded":   int(valuation_section["numberStocksTraded"]),
        "trade_cost":           float(valuation_section["trade_cost"]),
        "monthsToHold":         int(valuation_section["monthsToHold"]),
        "LongPeriod":           int(valuation_section["LongPeriod"]),
        "stddevThreshold":      float(valuation_section["stddevThreshold"]),
        "MA1":                  int(valuation_section["MA1"]),
        "MA2":                  int(valuation_section["MA2"]),
        "MA3":                  int(valuation_section["MA3"]),
        "rankThresholdPct":     float(valuation_section["rankThresholdPct"]),
        "riskDownside_min":     float(valuation_section["riskDownside_min"]),
        "riskDownside_max":     float(valuation_section["riskDownside_max"]),
        "narrowDays": [
            float(valuation_section["narrowDays_min"]),
            float(valuation_section["narrowDays_max"]),
        ],
        "mediumDays": [
            float(valuation_section["mediumDays_min"]),
            float(valuation_section["mediumDays_max"]),
        ],
        "wideDays": [
            float(valuation_section["wideDays_min"]),
            float(valuation_section["wideDays_max"]),
        ],
        "uptrendSignalMethod":  valuation_section["uptrendSignalMethod"],
        "lowPct":               valuation_section["lowPct"],
        "hiPct":                valuation_section["hiPct"],
        "minperiod":            int(valuation_section.get("minperiod", 10)),
        "maxperiod":            int(valuation_section.get("maxperiod", 100)),
        "incperiod":            int(valuation_section.get("incperiod", 10)),
        "numdaysinfit":         int(
                                    valuation_section.get("numdaysinfit", 100)
                                ),
        "numdaysinfit2":        int(
                                    valuation_section.get("numdaysinfit2", 200)
                                ),
        "offset":               int(valuation_section.get("offset", 0)),
        "stockList":            valuation_section["stockList"],
        "symbols_file":         valuation_section["symbols_file"],
        # Rolling window data-quality filter.
        "enable_rolling_filter": bool(
            valuation_section.get("enable_rolling_filter", False)
        ),
        "window_size":          int(
                                    valuation_section.get("window_size", 50)
                                ),
        # Background plot generation.
        "async_plot_generation": bool(
            valuation_section.get("async_plot_generation", True)
        ),
        "plot_generation_workers": int(
            valuation_section.get("plot_generation_workers", 2)
        ),
    }

    # Derived convenience key.
    params["MA2offset"] = params["MA3"] - params["MA2"]
    params["MA2factor"] = float(valuation_section["sma2factor"])

    # Recent plot start date (default: Jan 1 four years ago).
    if "recent_plot_start_date" in valuation_section:
        date_str = valuation_section["recent_plot_start_date"]
        params["recent_plot_start_date"] = datetime.datetime.strptime(
            date_str, "%Y-%m-%d"
        )
    else:
        current_year = datetime.datetime.now().year
        params["recent_plot_start_date"] = datetime.datetime(
            current_year - 4, 1, 1
        )

    return params


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def compute_long_hold_signal(
    json_fn: str,
) -> Tuple[list, np.ndarray, list, np.ndarray]:
    """Compute a long/hold signal based on MA of the system portfolio value.

    Reads historical portfolio values from ``PyTAAA_status.params``,
    deduplicates and sorts by date, then computes a mid-channel moving
    average signal using ``dpgchannel`` and ``SMA`` from ``TAfunctions``.

    Args:
        json_fn: Path to the JSON configuration file.

    Returns:
        Tuple of ``(sorted_dates, traded_values, sorted_values,
        last_signal)`` where:

        - ``sorted_dates`` — list of datetime objects, oldest first.
        - ``traded_values`` — ``np.ndarray`` shape ``(n_days,)`` of
          signal-timed cumulative portfolio values.
        - ``sorted_values`` — list of raw portfolio values, same order.
        - ``last_signal`` — ``np.ndarray`` shape ``(n_days,)`` of
          integer signals (``1`` = long, ``0`` = cash).
    """
    from functions.TAfunctions import dpgchannel, SMA

    def _uniqueify2lists(
        seq: list, seq2: list
    ) -> Tuple[list, list]:
        """Return order-preserving unique pairs (uniqueness from seq)."""
        seen: Dict = {}
        result: list = []
        result2: list = []
        for i, item in enumerate(seq):
            if item in seen:
                continue
            seen[item] = 1
            result.append(item)
            result2.append(seq2[i])
        return result, result2

    json_folder = get_performance_store(json_fn)
    filepath = os.path.join(json_folder, "PyTAAA_status.params")

    date: list = []
    value: list = []
    with open(filepath, "r") as fh:
        lines = fh.read().split("\n")
        for line in lines:
            parts = (line.split("\r")[0]).split(" ")
            if len(parts) >= 4:
                date.append(
                    datetime.datetime.strptime(parts[1], "%Y-%m-%d")
                )
                value.append(float(parts[3]))

    value_arr = np.array(value, dtype=float)

    # Deduplicate and sort (most recent last).
    sorted_dates, sorted_values = _uniqueify2lists(
        date[::-1], value_arr[::-1]
    )
    sorted_dates = sorted_dates[::-1]
    sorted_values = sorted_values[::-1]

    minchannel, maxchannel = dpgchannel(sorted_values, 5, 18, 4)
    midchannel = (minchannel + maxchannel) / 2.0
    MA_midchannel = SMA(midchannel, 5)

    signal = np.ones_like(sorted_values) * 11_000.0
    signal[MA_midchannel > midchannel] = 10_001
    signal[0] = 11_000.0

    gainloss = np.array(sorted_values)[1:] / np.array(sorted_values)[:-1]
    gainloss = np.hstack(((1.0,), gainloss)) - 1.0
    last_signal = (signal / 11_000.0).astype(int)
    gainloss *= last_signal.astype(float)
    gainloss += 1
    gainloss[0] = sorted_values[0]
    traded_values = np.cumprod(gainloss)

    return sorted_dates, traded_values, sorted_values, last_signal


def put_status(cumu_status: float, json_fn: str) -> None:
    """Append the current portfolio value to the status tracking file.

    Prints the previous and updated portfolio values for operator
    visibility, then appends a new record to ``PyTAAA_status.params``
    only if the value or signal has changed since the last write.

    Args:
        cumu_status: Current cumulative portfolio value.
        json_fn: Path to the JSON configuration file.
    """
    p_store = get_performance_store(json_fn)
    status_filename = os.path.join(p_store, "PyTAAA_status.params")

    with open(status_filename, "r") as fh:
        lines = fh.read()
    last_line = lines.split("\n")[-2]
    old_cumu_status = last_line.split(" ")[-3]
    old_cumu_signal = last_line.split(" ")[-2]

    _, traded_values, _, last_signal = compute_long_hold_signal(json_fn)

    def _fmt(v: float) -> str:
        """Format a portfolio value with commas and 2 decimal places."""
        try:
            return f"{float(v):>14,.2f}"
        except (ValueError, TypeError):
            return str(v)

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"  Previous portfolio value        : {_fmt(old_cumu_status)}")
    print(
        f"  Updated portfolio value         : {_fmt(cumu_status)}"
        f"  [{ts}]"
    )

    if (
        str(cumu_status) != str(old_cumu_status)
        or str(last_signal[-1]) != str(old_cumu_signal)
    ):
        _write_status_line(
            status_filename,
            cumu_status,
            int(last_signal[-1]),
            float(traded_values[-1]),
        )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def GetIP() -> str:
    """Return the machine's current external IP address.

    Makes an HTTP request to ``canyouseeme.org`` and parses the response
    for an IPv4 address.

    Returns:
        External IP address as a dotted-decimal string.

    Note:
        This function makes a live network call and is not unit-testable
        in isolation.
    """
    import urllib.request
    import re as _re

    with urllib.request.urlopen("http://www.canyouseeme.org/") as fh:
        html_doc = fh.read().decode("utf-8")

    m = _re.search(
        r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\."
        r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\."
        r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\."
        r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)",
        html_doc,
    )
    return m.group(0)


def GetEdition() -> str:
    """Return a string describing the platform this code is running on.

    Returns one of: ``'pi'``, ``'Windows32'``, ``'Windows64'``,
    ``'MacOS'``, or ``'none'``.

    Returns:
        Platform edition identifier string.
    """
    import platform

    arch = platform.uname()[4]
    if "armv6l" in arch:
        return "pi"
    if "x86" in arch:
        return "Windows32"
    if "AMD64" in arch:
        return "Windows64"
    if "arm64" in arch:
        return "MacOS"
    return "none"
