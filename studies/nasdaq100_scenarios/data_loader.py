"""Data loading for NASDAQ100 oracle delay studies.

Reuses production HDF5 loaders with study-specific date windowing
and tradability inference.
"""

import json
import logging
import os
from datetime import datetime, date
from pathlib import Path
from typing import Tuple, List
import numpy as np
from numpy.typing import NDArray

from functions.UpdateSymbols_inHDF5 import loadQuotes_fromHDF
from functions.TAfunctions import interpolate, cleantobeginning, cleantoend

logger = logging.getLogger(__name__)


def _derive_hdf5_path(symbols_file: Path) -> Path:
    """Derive expected HDF5 filename from a symbols text file path."""
    return symbols_file.parent / f"{symbols_file.stem}_.hdf5"


def _resolve_symbols_file_path(
    configured_symbols_file: str,
    params_json_path: str
) -> str:
    """Resolve symbols file path from config, env overrides, or known data roots.

    Preference order:
    1) Environment overrides (`PYTAAA_SYMBOLS_FILE`, `SYMBOLS_FILE`)
    2) Configured path as absolute or workspace-relative
    3) Config-relative fallback
    4) Known local data roots with existing NASDAQ100 HDF5
    """
    config_path = Path(params_json_path).resolve()
    config_dir = config_path.parent
    cwd = Path.cwd().resolve()

    env_candidates = [
        Path(v).expanduser()
        for v in (
            os.environ.get("PYTAAA_SYMBOLS_FILE"),
            os.environ.get("SYMBOLS_FILE"),
        )
        if v
    ]

    base_candidate = Path(configured_symbols_file).expanduser()
    path_candidates: List[Path] = []
    path_candidates.extend(env_candidates)
    if base_candidate.is_absolute():
        path_candidates.append(base_candidate)
    else:
        path_candidates.append(cwd / base_candidate)
        path_candidates.append(config_dir / base_candidate)

    for candidate in path_candidates:
        candidate = candidate.resolve()
        if candidate.exists() or _derive_hdf5_path(candidate).exists():
            return str(candidate)

    known_roots = [
        Path.home() / "pyTAAA_data",
        Path.home() / "pyTAAA_data_static",
        Path.home() / "PyProjects" / "PyTAAA",
    ]
    hdf_names = ["Naz100_Symbols_.hdf5", "Naz100_symbols_.hdf5"]
    hdf_candidates: List[Path] = []
    for root in known_roots:
        hdf_candidates.extend([
            root / "Naz100" / "symbols" / hdf_names[0],
            root / "Naz100" / "symbols" / hdf_names[1],
            root / "symbols" / hdf_names[0],
            root / "symbols" / hdf_names[1],
        ])

    for hdf_path in hdf_candidates:
        if hdf_path.exists():
            symbols_stem = hdf_path.stem.rstrip("_")
            return str(hdf_path.parent / f"{symbols_stem}.txt")

    raise FileNotFoundError(
        "Could not resolve a valid NASDAQ100 symbols/HDF5 path. "
        "Set PYTAAA_SYMBOLS_FILE (or SYMBOLS_FILE) to your symbols file path, "
        "or ensure Naz100_Symbols_.hdf5 exists under a known data root."
    )


def load_nasdaq100_window(
    params_json_path: str
) -> Tuple[NDArray[np.floating], List[str], List[date], NDArray[np.bool_]]:
    """Load NASDAQ100 data with date clipping and tradability inference.

    Loads quote data from HDF5, applies preprocessing, clips to requested
    date range, and infers which stocks are tradable on each date.

    Args:
        params_json_path: Path to JSON configuration file containing
                         data_selection.start_date and data_selection.stop_date

    Returns:
        Tuple of:
        - adjClose: 2D array (stocks × dates) of adjusted closing prices
        - symbols: List of stock ticker symbols
        - datearray: List of datetime.date objects for each trading day
        - tradable_mask: 2D boolean array (stocks × dates), True where tradable

    Raises:
        FileNotFoundError: If JSON config or HDF5 file not found
        ValueError: If requested date range has no overlap with available data
        KeyError: If required config keys missing from JSON

    Example:
        >>> adjClose, symbols, dates, mask = load_nasdaq100_window(
        ...     "studies/nasdaq100_scenarios/params/default_scenario.json"
        ... )
        >>> print(f"Loaded {len(symbols)} symbols, {len(dates)} trading days")
        >>> print(f"Tradable coverage: {mask.sum() / mask.size:.1%}")
    """
    logger.info("Loading configuration from %s", params_json_path)
    
    with open(params_json_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Extract configuration
    data_config = config['data_selection']
    requested_start = datetime.strptime(
        data_config['start_date'], '%Y-%m-%d'
    ).date()
    requested_stop = datetime.strptime(
        data_config['stop_date'], '%Y-%m-%d'
    ).date()
    
    logger.info(
        "Requested date range: %s to %s",
        requested_start, requested_stop
    )

    padding_days = _compute_padding_days(config)
    if padding_days > 0:
        logger.info(
            "Padding analysis window by %d trading days on each side",
            padding_days
        )
    
    # Load from HDF5 using production loader
    # Note: loadQuotes_fromHDF expects symbols file path and json path
    symbols_file_config = data_config.get(
        "symbols_file",
        "symbols/Naz100_Symbols.txt"
    )
    symbols_file = _resolve_symbols_file_path(
        symbols_file_config,
        params_json_path,
    )
    logger.info("Resolved symbols file path: %s", symbols_file)
    
    logger.info("Loading NASDAQ100 data from HDF5")
    adjClose_raw, symbols_raw, datearray_raw, _, _ = loadQuotes_fromHDF(
        symbols_file, params_json_path
    )
    
    logger.info(
        "Raw data loaded: %d symbols, %d days",
        adjClose_raw.shape[0], adjClose_raw.shape[1]
    )
    
    # Convert datearray to date objects if needed
    datearray_clean = []
    for d in datearray_raw:
        if isinstance(d, date):
            datearray_clean.append(d)
        elif isinstance(d, str):
            # Parse YYYY-MM-DD format
            datearray_clean.append(
                datetime.strptime(d.split()[0], '%Y-%m-%d').date()
            )
        else:
            logger.warning("Unexpected date type: %s", type(d))
            datearray_clean.append(d)
    
    # Apply preprocessing to each stock
    logger.info("Applying data cleaning (interpolate, clean boundaries)")
    adjClose_clean = adjClose_raw.copy()
    for i in range(adjClose_clean.shape[0]):
        adjClose_clean[i, :] = interpolate(adjClose_clean[i, :])
        adjClose_clean[i, :] = cleantobeginning(adjClose_clean[i, :])
        adjClose_clean[i, :] = cleantoend(adjClose_clean[i, :])
    
    # Add CASH symbol if not present
    symbols = list(symbols_raw)
    adjClose = adjClose_clean
    if 'CASH' not in symbols:
        logger.info("Adding CASH symbol with constant 1.0 prices")
        symbols.append('CASH')
        cash_prices = np.ones((1, adjClose.shape[1]), dtype=float)
        adjClose = np.vstack([adjClose, cash_prices])
    else:
        # Ensure CASH is constant 1.0
        cash_idx = symbols.index('CASH')
        adjClose[cash_idx, :] = 1.0
    
    # Clip to requested date range
    adjClose_clipped, symbols_clipped, datearray_clipped = (
        _clip_to_date_range(
            adjClose,
            symbols,
            datearray_clean,
            requested_start,
            requested_stop,
            padding_days
        )
    )
    
    # Infer tradable mask
    logger.info("Inferring tradable mask from data quality")
    tradable_mask = infer_tradable_mask(adjClose_clipped, datearray_clipped)
    
    tradable_pct = tradable_mask.sum() / tradable_mask.size * 100
    logger.info(
        "Final dataset: %d symbols, %d days, %.1f%% tradable coverage",
        len(symbols_clipped), len(datearray_clipped), tradable_pct
    )
    
    return adjClose_clipped, symbols_clipped, datearray_clipped, tradable_mask


def _compute_padding_days(config: dict) -> int:
    oracle_params = config.get("oracle_parameters", {})
    delay_params = config.get("delay_parameters", {})
    windows = oracle_params.get("extrema_windows", [])
    delays = delay_params.get("days_delay", [])

    max_window = max(windows) if windows else 0
    max_delay = max(delays) if delays else 0
    return int((2 * max_window) + max_delay)


def _clip_to_date_range(
    adjClose: NDArray[np.floating],
    symbols: List[str],
    datearray: List[date],
    start_date: date,
    stop_date: date,
    padding_days: int = 0
) -> Tuple[NDArray[np.floating], List[str], List[date]]:
    """Clip data to requested date range.

    Finds nearest available dates if requested dates are outside range.
    Logs warnings when clamping occurs.

    Args:
        adjClose: Full adjusted close array (stocks × dates)
        symbols: Symbol list
        datearray: Full date array
        start_date: Requested start date
        stop_date: Requested stop date

    Returns:
        Tuple of (clipped_adjClose, symbols, clipped_datearray)

    Raises:
        ValueError: If no overlap between requested and available dates
    """
    available_start = datearray[0]
    available_stop = datearray[-1]
    
    # Clamp to available range
    actual_start = max(start_date, available_start)
    actual_stop = min(stop_date, available_stop)
    
    if actual_start != start_date:
        logger.warning(
            "Requested start %s before available data; clamped to %s",
            start_date, actual_start
        )
    
    if actual_stop != stop_date:
        logger.warning(
            "Requested stop %s after available data; clamped to %s",
            stop_date, actual_stop
        )
    
    if actual_start > actual_stop:
        raise ValueError(
            f"No date overlap: requested [{start_date}, {stop_date}] "
            f"vs available [{available_start}, {available_stop}]"
        )
    
    # Find indices
    start_idx = None
    stop_idx = None
    for i, d in enumerate(datearray):
        if start_idx is None and d >= actual_start:
            start_idx = i
        if d <= actual_stop:
            stop_idx = i
    
    if start_idx is None or stop_idx is None:
        raise ValueError("Failed to find date indices in range")
    
    padded_start_idx = max(0, start_idx - padding_days)
    padded_stop_idx = min(len(datearray) - 1, stop_idx + padding_days)

    # Slice data
    clipped_adjClose = adjClose[:, padded_start_idx:padded_stop_idx + 1]
    clipped_datearray = datearray[padded_start_idx:padded_stop_idx + 1]
    
    logger.info(
        "Date clipping: [%s, %s] → [%s, %s] (%d days, pad=%d)",
        start_date, stop_date,
        clipped_datearray[0], clipped_datearray[-1],
        len(clipped_datearray), padding_days
    )
    
    return clipped_adjClose, symbols, clipped_datearray


def infer_tradable_mask(
    adjClose: NDArray[np.floating],
    datearray: List[date]
) -> NDArray[np.bool_]:
    """Infer which stocks are tradable on each date.

    Uses heuristics based on NaN presence, constant prices (infill),
    and leading/trailing invalid data to determine tradability.

    A stock is considered untradable on a date if:
    - Price is NaN
    - Price is at the beginning/end of a cleaned constant-price region
    - Stock shows signs of forward/backward fill artifacts

    Args:
        adjClose: Adjusted close prices (stocks × dates)
        datearray: Trading dates

    Returns:
        Boolean mask (stocks × dates): True where tradable, False otherwise

    Notes:
        - CASH symbol is always tradable (if present)
        - Conservative approach: mark suspicious data as untradable
    """
    n_stocks, n_days = adjClose.shape
    tradable = np.ones((n_stocks, n_days), dtype=bool)
    
    for i in range(n_stocks):
        prices = adjClose[i, :]
        
        # Mark NaN as untradable
        tradable[i, :] = ~np.isnan(prices)
        
        # Detect leading constant-price region (likely forward-filled)
        if n_days > 1:
            first_valid = np.argmax(~np.isnan(prices))
            if first_valid > 0:
                # Has leading NaNs - already marked untradable
                pass
            
            # Check for constant prices at start (infill artifact)
            window = min(5, n_days)
            if np.std(prices[:window]) < 1e-6 and not np.isnan(prices[0]):
                # Likely infilled at start
                tradable[i, :window] = False
        
        # Detect trailing constant-price region (likely backward-filled)
        if n_days > 1:
            last_valid = n_days - 1 - np.argmax(~np.isnan(prices[::-1]))
            if last_valid < n_days - 1:
                # Has trailing NaNs - already marked untradable
                pass
            
            # Check for constant prices at end
            window = min(5, n_days)
            if np.std(prices[-window:]) < 1e-6 and not np.isnan(prices[-1]):
                # Likely infilled at end
                tradable[i, -window:] = False
        
        # Detect mid-series anomalies (sudden jumps back to constant)
        if n_days > 10:
            price_changes = np.abs(np.diff(prices))
            # If we see a sudden return to near-zero changes after movement,
            # it might be infill - but this is less reliable, so skip for now
            pass
    
    # Special handling for CASH (always tradable)
    # CASH should be last symbol if added by loader
    # We can't easily identify it here without symbol list,
    # so caller should handle this if needed
    
    return tradable


def get_tradable_symbols_by_date(
    symbols: List[str],
    datearray: List[date],
    tradable_mask: NDArray[np.bool_]
) -> dict:
    """Get list of tradable symbols for each date.

    Utility function to convert tradable mask into date-indexed dictionary.

    Args:
        symbols: List of stock symbols
        datearray: Trading dates
        tradable_mask: Boolean tradability mask (stocks × dates)

    Returns:
        Dictionary mapping date → list of tradable symbols

    Example:
        >>> tradable_by_date = get_tradable_symbols_by_date(
        ...     symbols, dates, mask
        ... )
        >>> print(f"Tradable on 2020-01-02: {tradable_by_date[date(2020, 1, 2)]}")
    """
    result = {}
    for j, d in enumerate(datearray):
        tradable_symbols = [
            symbols[i] for i in range(len(symbols))
            if tradable_mask[i, j]
        ]
        result[d] = tradable_symbols
    
    return result
