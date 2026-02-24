"""
Stock selection runner — Phase 1 scaffold (NOT YET FUNCTIONAL).

This module was written as infrastructure scaffolding during Phase 1
planning.  It is NOT used by the current look-ahead bias study or
pytest suite.

The actual study uses run_lookahead_study.run_selection_pipeline(),
which calls computeSignal2D + sharpeWeightedRank_2D directly on
in-memory numpy arrays.

get_ranked_stocks_for_date() below is a PLACEHOLDER.  It returns
symbols[:7] with equal weights, not actual ranked stock selections.
Any code calling this function will not get meaningful results.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from functions.dailyBacktest_pctLong import dailyBacktest_pctLong
from functions.GetParams import get_json_params, get_symbols_file
from functions.UpdateSymbols_inHDF5 import loadQuotes_fromHDF


def get_ranked_stocks_for_date(
    hdf5_path: str,
    json_params_path: str,
    as_of_date: str
):
    """
    PLACEHOLDER — does not return real ranked stocks.

    This function was written as Phase 1 scaffolding.  It has never been
    connected to the production pipeline.  It returns the first 7 symbols
    from the HDF5 file with equal weights, regardless of actual rankings.

    The working implementation is run_lookahead_study.run_selection_pipeline(),
    which runs computeSignal2D + sharpeWeightedRank_2D on in-memory arrays
    and is what the study script and pytest suite actually use.

    Args:
        hdf5_path: Path to HDF5 file (loaded but rank result is not used)
        json_params_path: Path to JSON config file
        as_of_date: Date string "YYYY-MM-DD" (validated but not used for
                    actual ranking)

    Returns:
        Tuple of (symbols[:7], [1/7, ...]) — a hardcoded placeholder,
        NOT the real ranked selections.

    Raises:
        ValueError: If as_of_date is not found in datearray
    """
    
    # Load parameters
    params = get_json_params(json_params_path)
    symbols_file = get_symbols_file(json_params_path)
    
    # Load quotes from the (possibly patched) HDF5
    adjClose, symbols, datearray, _, _ = loadQuotes_fromHDF(
        symbols_file,
        json_params_path
    )
    
    # Optionally override the HDF5 path in params (if needed for custom loading)
    # For now, we assume the standard loading path is used.
    
    # Run the daily backtest
    # (This is a simplified call; actual integration may need adjustment)
    print(f"[selection_runner] Running backtest for {as_of_date} "
          f"using {hdf5_path}")
    print(f"[selection_runner] Symbols: {symbols[:5]}... (total {len(symbols)})")
    print(f"[selection_runner] Date range: {datearray[0]} to {datearray[-1]}")
    
    # Find the index of as_of_date
    import pandas as pd
    datearray_dt = pd.to_datetime(datearray)
    as_of_dt = pd.to_datetime(as_of_date)
    
    matching_indices = [i for i, d in enumerate(datearray_dt) if d == as_of_dt]
    if not matching_indices:
        raise ValueError(f"Date {as_of_date} not found in datearray")
    
    date_idx = matching_indices[0]
    
    # PLACEHOLDER: returns first 7 symbols with equal weights.
    # This is NOT connected to the production ranking pipeline.
    ranked_symbols = symbols[:7]
    ranked_weights = [1.0 / 7.0] * 7
    
    print(f"[selection_runner] Top-{len(ranked_symbols)} stocks for {as_of_date}: "
          f"{ranked_symbols}")
    
    return ranked_symbols, ranked_weights
