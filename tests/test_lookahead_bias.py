"""
Look-Ahead Bias Tests (pytest)

Tests the PyTAAA stock selection pipeline for look-ahead bias using
production HDF5 price data loaded directly from disk.  No test HDF5
files are created or copied — price patching is performed entirely in
memory.

STRATEGY
--------
For each model (naz100_hma, naz100_pine, naz100_pi):
  1. Load the full production adjClose array from the model's HDF5.
  2. Select a CUTOFF date \u2248200 trading days before the last date.
  3. Build a PATCHED copy of adjClose where prices AFTER the cutoff
     are altered dramatically (top-half performers stepped down 40 %,
     bottom-half performers stepped up 40 %).
  4. Run the full signal+rank pipeline on both the original and the
     patched adjClose arrays.
  5. Compare stock selections at the CUTOFF date:
       PASS \u2192 identical selections (no look-ahead bias)
       FAIL \u2192 different selections (look-ahead bias detected)

Tests are skipped automatically when the production JSON / HDF5 files
are not present (e.g., in CI environments without real data).
"""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from functions.GetParams import get_symbols_file
from functions.data_loaders import load_quotes_for_analysis
from studies.lookahead_bias.run_lookahead_study import (
    _params_from_json,
    _load_production_adjclose,
    _patch_adjclose,
    run_selection_pipeline,
)

# =====================================================================
# Configuration — default production JSON paths
# =====================================================================

_DEFAULT_JSON = {
    "naz100_hma": (
        "/Users/donaldpg/pyTAAA_data/naz100_hma/pytaaa_naz100_hma.json"
    ),
    "naz100_pine": (
        "/Users/donaldpg/pyTAAA_data/naz100_pine/pytaaa_naz100_pine.json"
    ),
    "naz100_pi": (
        "/Users/donaldpg/pyTAAA_data/naz100_pi/pytaaa_naz100_pi.json"
    ),
}

# Days before the end of the HDF5 data to use as the cutoff.
# ~200 trading days gives ≥8 complete post-cutoff calendar months.
_CUTOFF_DAYS_FROM_END = 200

# Slice window around the cutoff — mirrors the study script constants.
# 600 pre-cutoff days covers all rolling windows; 200 post-cutoff days
# give enough divergence to observe without running the full dataset.
_PRE_DAYS  = 600
_POST_DAYS = 200


# =====================================================================
# Helper
# =====================================================================

def _run_both_pipelines(json_fn: str):
    """
    Load production HDF5 data, slice to a narrow window around the
    cutoff, build original and patched adjClose arrays in memory, run
    the full selection pipeline on both, and return the results needed
    for look-ahead bias comparison.

    Slicing to _PRE_DAYS + _POST_DAYS (≈800 days) instead of the full
    8 000+ day history gives the same result at the cutoff date while
    running ~10× faster.

    Returns:
        weights_orig  : (n_stocks, slice_len) weight array, original
        weights_patch : (n_stocks, slice_len) weight array, patched
        cut_sl        : int index of the cutoff date within the slice
        datearray_sl  : list of date objects for the slice
        symbols       : list of ticker strings
    """
    params = _params_from_json(json_fn)

    adjClose_orig, symbols, datearray = _load_production_adjclose(json_fn)

    n_days = adjClose_orig.shape[1]
    cutoff_idx = n_days - _CUTOFF_DAYS_FROM_END

    # Slice to a narrow window — same logic as the study script.
    start_idx    = max(0, cutoff_idx - _PRE_DAYS)
    end_idx      = min(n_days, cutoff_idx + _POST_DAYS + 1)
    adjClose_sl  = adjClose_orig[:, start_idx:end_idx]
    datearray_sl = datearray[start_idx:end_idx]
    cut_sl       = cutoff_idx - start_idx   # Cutoff index within slice

    adjClose_patch = _patch_adjclose(
        adjClose_sl,
        symbols,
        cutoff_idx=cut_sl,
        step_down_factor=0.60,
        step_up_factor=1.40,
    )

    # Sanity: pre-cutoff prices must be byte-identical
    assert np.array_equal(
        adjClose_sl[:, : cut_sl + 1],
        adjClose_patch[:, : cut_sl + 1],
    ), "BUG: _patch_adjclose altered pre-cutoff prices"

    weights_orig = run_selection_pipeline(
        adjClose_sl, symbols, datearray_sl, params, json_fn
    )
    weights_patch = run_selection_pipeline(
        adjClose_patch, symbols, datearray_sl, params, json_fn
    )

    return weights_orig, weights_patch, cut_sl, datearray_sl, symbols


# =====================================================================
# Tests
# =====================================================================


@pytest.mark.parametrize("model,json_fn", list(_DEFAULT_JSON.items()))
def test_selection_consistency_across_models(model, json_fn):
    """
    Core look-ahead bias test.

    Selections at the CUTOFF date must be identical whether prices
    after the cutoff are original or dramatically altered.  Any
    difference proves the pipeline reads future prices.

    Skipped automatically when production JSON / HDF5 files are
    absent (e.g., CI environments without real data).
    """
    if not Path(json_fn).exists():
        pytest.skip(f"Production JSON not found: {json_fn}")

    try:
        (weights_orig, weights_patch,
         cutoff_idx, datearray, symbols) = _run_both_pipelines(json_fn)
    except FileNotFoundError as exc:
        pytest.skip(f"Production HDF5 data not found for {model}: {exc}")

    # Extract sets of selected stocks at the cutoff index
    orig_sel = {
        symbols[i] for i in range(len(symbols))
        if weights_orig[i, cutoff_idx] > 1e-6
    }
    patch_sel = {
        symbols[i] for i in range(len(symbols))
        if weights_patch[i, cutoff_idx] > 1e-6
    }

    assert orig_sel == patch_sel, (
        f"LOOK-AHEAD BIAS DETECTED in {model} at "
        f"cutoff={datearray[cutoff_idx]}:\n"
        f"  original selections : {sorted(orig_sel)}\n"
        f"  patched  selections : {sorted(patch_sel)}"
    )


def test_patch_is_identity_before_cutoff():
    """
    Verify _patch_adjclose leaves pre-cutoff prices unchanged for all
    tickers, regardless of which half they fall into.
    """
    rng = np.random.default_rng(0)
    n_stocks, n_days = 20, 500
    cutoff_idx = 300
    adjClose = rng.uniform(50, 200, size=(n_stocks, n_days))
    symbols = [f"T{i:02d}" for i in range(n_stocks)]

    patched = _patch_adjclose(adjClose, symbols, cutoff_idx)

    assert np.array_equal(
        adjClose[:, : cutoff_idx + 1],
        patched[:, : cutoff_idx + 1],
    ), "Pre-cutoff prices were modified"

    # Post-cutoff prices should differ for at least some stocks
    assert not np.array_equal(
        adjClose[:, cutoff_idx + 1:],
        patched[:, cutoff_idx + 1:],
    ), "Post-cutoff prices were not modified"


def test_hdf5_utils_imports():
    """Verify hdf5_utils module imports correctly."""
    from studies.lookahead_bias.hdf5_utils import (
        copy_hdf5, patch_hdf5_prices
    )
    assert callable(copy_hdf5)
    assert callable(patch_hdf5_prices)


def test_patch_strategies_imports():
    """Verify patch_strategies module imports correctly."""
    from studies.lookahead_bias.patch_strategies import (
        step_down, step_up, linear_down, linear_up
    )
    assert callable(step_down)
    assert callable(step_up)
    assert callable(linear_down)
    assert callable(linear_up)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
