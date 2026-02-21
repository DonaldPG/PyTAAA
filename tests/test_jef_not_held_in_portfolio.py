"""
test_jef_not_held_in_portfolio.py

Phase 1 test: assert that get_json_params() exposes enable_rolling_filter
and window_size from the JSON Valuation section.

Root cause: get_json_params() never read these keys, so the rolling window
filter was always skipped in the pytaaa_main.py code path
(dailyBacktest.py defaulted enable_rolling_filter to False).

This test MUST FAIL before the fix to GetParams.py and PASS after.
"""

import os
import pytest
from functions.GetParams import get_json_params


#############################################################################
# Path to the workspace-safe dev copy of the production JSON.
# Production JSON at /Users/donaldpg/pytaaa_data/sp500_pine/pytaaa_sp500_pine.json
# must never be modified.
#############################################################################
DEV_JSON = os.path.join(
    os.path.dirname(__file__), "..", "pytaaa_sp500_pine_dev.json"
)


def test_dev_json_exists():
    """Confirm the dev JSON copy is present before running filter tests."""
    assert os.path.isfile(DEV_JSON), (
        f"Dev JSON not found at {DEV_JSON}. "
        "Run: cp /Users/donaldpg/pytaaa_data/sp500_pine/pytaaa_sp500_pine.json "
        "pytaaa_sp500_pine_dev.json"
    )


def test_get_json_params_exposes_enable_rolling_filter():
    """
    get_json_params must return enable_rolling_filter from the Valuation
    section of the JSON. Without this key the rolling window filter is
    unconditionally skipped in dailyBacktest.py, allowing artificially
    infilled prices (e.g. JEF 2015-2018) to produce portfolio selections.

    Fails before the 2-line fix to functions/GetParams.py.
    Passes after.
    """
    params = get_json_params(DEV_JSON)
    assert params is not None, "get_json_params returned None"
    assert "enable_rolling_filter" in params, (
        "get_json_params must include 'enable_rolling_filter' in returned "
        "params dict. Add: params['enable_rolling_filter'] = bool("
        "valuation_section.get('enable_rolling_filter', False)) before "
        "'return params' in functions/GetParams.py"
    )
    assert params["enable_rolling_filter"] is True, (
        f"Expected enable_rolling_filter=True (from dev JSON Valuation section), "
        f"got {params['enable_rolling_filter']!r}"
    )


def test_get_json_params_exposes_window_size():
    """
    get_json_params must return window_size from the Valuation section.
    The dev JSON Valuation section has window_size=50.

    Fails before the 2-line fix to functions/GetParams.py.
    Passes after.
    """
    params = get_json_params(DEV_JSON)
    assert params is not None, "get_json_params returned None"
    assert "window_size" in params, (
        "get_json_params must include 'window_size' in returned params dict. "
        "Add: params['window_size'] = int(valuation_section.get('window_size', 50)) "
        "before 'return params' in functions/GetParams.py"
    )
    assert params["window_size"] == 50, (
        f"Expected window_size=50, got {params['window_size']!r}"
    )
