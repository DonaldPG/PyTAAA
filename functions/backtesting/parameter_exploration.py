"""Random parameter generation for Monte Carlo backtesting.

This module provides functions for sampling trading parameters from
statistical distributions.  Three trial modes are used:

- **Exploration** (first half of trials): draw all parameters from
  broad triangular distributions bounded by per-scenario ranges.
- **JSON + one varied** (second half minus last): start from the JSON
  config values and perturb exactly one parameter per trial.
- **JSON exact** (final trial only): reproduce the JSON config values
  unchanged, giving a reproducible single-point baseline.

Per-scenario triangular ranges are stored in ``_RANGES``.  Run
``scripts/extract_montecarlo_ranges.py`` after xlsx optimisation files
are ready to update them, then paste the printed dict back here.
"""

import os
import random as stdlib_random
from random import choice
from typing import Union

import numpy as np
from numpy import random

from functions.logger_config import get_logger

logger = get_logger(__name__, log_file="pytaaa_backtest_montecarlo.log")

#############################################################################
# Per-scenario triangular sampling ranges
#
# Each sub-dict maps parameter names → (low, mid, high) triples for
# random_triangle().  Initialised from the original hard-coded exploration
# literals.  Update by running scripts/extract_montecarlo_ranges.py.
#############################################################################

_RANGES: dict[str, dict[str, tuple]] = {
    "naz100_pine": {
        "LongPeriod":       (150,   300,    600),
        "stddevThreshold":  (2.0,   8.0,    20.0),
        "MA1":              (75,    151,    300),
        "MA2":              (7,     22,     45),
        "MA2offset":        (2,     7,      12),
        "sma2factor":       (1.65,  2.5,    2.75),
        "rankThresholdPct": (0.14,  0.20,   0.26),
        "riskDownside_min": (0.50,  0.70,   0.90),
        "riskDownside_max": (8.0,   10.0,   13.0),
        "sma_filt_val":     (0.010, 0.015,  0.0225),
        "lowPct":           (10.0,  20.0,   30.0),
        "hiPct":            (70.0,  80.0,   90.0),
    },
    "naz100_hma": {
        "LongPeriod":       (150,   300,    600),
        "stddevThreshold":  (2.0,   8.0,    20.0),
        "MA1":              (75,    151,    300),
        "MA2":              (7,     22,     45),
        "MA2offset":        (2,     7,      12),
        "sma2factor":       (1.65,  2.5,    2.75),
        "rankThresholdPct": (0.14,  0.20,   0.26),
        "riskDownside_min": (0.50,  0.70,   0.90),
        "riskDownside_max": (8.0,   10.0,   13.0),
        "sma_filt_val":     (0.010, 0.015,  0.0225),
        "lowPct":           (10.0,  20.0,   30.0),
        "hiPct":            (70.0,  80.0,   90.0),
    },
    "naz100_pi": {
        "LongPeriod":       (150,   300,    600),
        "stddevThreshold":  (2.0,   8.0,    20.0),
        "MA1":              (75,    151,    300),
        "MA2":              (7,     22,     45),
        "MA2offset":        (2,     7,      12),
        "sma2factor":       (1.65,  2.5,    2.75),
        "rankThresholdPct": (0.14,  0.20,   0.26),
        "riskDownside_min": (0.50,  0.70,   0.90),
        "riskDownside_max": (8.0,   10.0,   13.0),
        "sma_filt_val":     (0.010, 0.015,  0.0225),
        "lowPct":           (10.0,  20.0,   30.0),
        "hiPct":            (70.0,  80.0,   90.0),
    },
    "sp500_pine": {
        "LongPeriod":       (150,   300,    600),
        "stddevThreshold":  (2.0,   8.0,    20.0),
        "MA1":              (75,    151,    300),
        "MA2":              (7,     22,     45),
        "MA2offset":        (2,     7,      12),
        "sma2factor":       (1.65,  2.5,    2.75),
        "rankThresholdPct": (0.14,  0.20,   0.26),
        "riskDownside_min": (0.50,  0.70,   0.90),
        "riskDownside_max": (8.0,   10.0,   13.0),
        "sma_filt_val":     (0.010, 0.015,  0.0225),
        "lowPct":           (10.0,  20.0,   30.0),
        "hiPct":            (70.0,  80.0,   90.0),
    },
    "sp500_hma": {
        "LongPeriod":       (150,   300,    600),
        "stddevThreshold":  (2.0,   8.0,    20.0),
        "MA1":              (75,    151,    300),
        "MA2":              (7,     22,     45),
        "MA2offset":        (2,     7,      12),
        "sma2factor":       (1.65,  2.5,    2.75),
        "rankThresholdPct": (0.14,  0.20,   0.26),
        "riskDownside_min": (0.50,  0.70,   0.90),
        "riskDownside_max": (8.0,   10.0,   13.0),
        "sma_filt_val":     (0.010, 0.015,  0.0225),
        "lowPct":           (10.0,  20.0,   30.0),
        "hiPct":            (70.0,  80.0,   90.0),
    },
}

#############################################################################
# JSON parameter keys to copy verbatim into each trial dict.
# MA3 is excluded because it is always derived from MA2 + MA2offset.
#############################################################################

_JSON_PARAM_KEYS: tuple = (
    "numberStocksTraded",
    "monthsToHold",
    "LongPeriod",
    "stddevThreshold",
    "MA1",
    "MA2",
    "MA2offset",
    "sma2factor",
    "rankThresholdPct",
    "riskDownside_min",
    "riskDownside_max",
    "sma_filt_val",
    "lowPct",
    "hiPct",
)


def random_triangle(
    low: float,
    mid: float,
    high: float,
    size: int = 1,
) -> Union[float, list]:
    """Sample from a triangular distribution.

    Args:
        low: Lower limit of the distribution.
        mid: Mode (peak) of the distribution.
        high: Upper limit of the distribution.
        size: Number of samples to draw.

    Returns:
        A single float when ``size == 1``, otherwise a list of floats.

    Example:
        >>> val = random_triangle(100.0, 200.0, 300.0)
        >>> 100.0 <= val <= 300.0
        True
    """
    samples = [random.triangular(low, mid, high) for _ in range(size)]
    return samples[0] if size == 1 else samples


def _generate_weight_constraints() -> dict:
    """Return randomly sampled weight constraint parameters.

    Returns:
        Dict with keys max_weight_factor, min_weight_factor,
        absolute_max_weight, apply_constraints.
    """
    return {
        "max_weight_factor": random_triangle(2.0, 3.0, 5.0),
        "min_weight_factor": random_triangle(0.1, 0.3, 0.5),
        "absolute_max_weight": random_triangle(0.7, 0.9, 1.0),
        "apply_constraints": True,
    }


def _apply_ma_constraints(params: dict) -> dict:
    """Enforce minimum values and MA3 derivation on MA parameters.

    Enforces:
    - MA2 >= 3
    - MA1 >= MA2 + 1
    - MA3 = MA2 + MA2offset

    Args:
        params: Parameter dict containing MA1, MA2, MA2offset.

    Returns:
        The same dict with enforced MA values.
    """
    params["MA2"] = max(int(params["MA2"]), 3)
    params["MA1"] = max(int(params["MA1"]), params["MA2"] + 1)
    params["MA3"] = int(params["MA2"]) + int(params["MA2offset"])
    return params


def _derive_scenario(params: dict) -> str:
    """Derive the scenario key from JSON config params.

    Maps the combination of ``symbols_file`` basename and
    ``uptrendSignalMethod`` to one of the five known scenario keys.

    Args:
        params: Flattened JSON config dict containing at least
            ``symbols_file`` and ``uptrendSignalMethod``.

    Returns:
        One of: ``naz100_pine``, ``naz100_hma``, ``naz100_pi``,
        ``sp500_pine``, ``sp500_hma``.  Falls back to
        ``naz100_pine`` with a warning when unrecognised.
    """
    sf = os.path.basename(
        params.get("symbols_file", "")
    ).lower()
    usm = params.get("uptrendSignalMethod", "percentileChannels")

    is_naz = "naz100" in sf or "naz_100" in sf
    is_sp500 = "sp500" in sf or "biglist" in sf

    if is_naz:
        if usm == "HMAs":
            return "naz100_hma"
        if usm == "SMAs":
            return "naz100_pi"
        return "naz100_pine"

    if is_sp500:
        if usm == "HMAs":
            return "sp500_hma"
        return "sp500_pine"

    logger.warning(
        "Could not derive scenario from symbols_file=%r "
        "uptrendSignalMethod=%r; defaulting to naz100_pine.",
        params.get("symbols_file", ""),
        usm,
    )
    return "naz100_pine"


def _get_ranges(params: dict) -> dict:
    """Return the triangular-range sub-dict for the current scenario.

    Args:
        params: Flattened JSON config dict.

    Returns:
        Dict mapping parameter names to (low, mid, high) tuples.
    """
    return _RANGES[_derive_scenario(params)]


def _generate_json_exact_params(params: dict, usm: str) -> dict:
    """Build a trial dict using JSON config values exactly.

    No randomisation is applied to the trading parameters.  Weight
    constraints are still randomly sampled because they are not stored
    in the JSON config.

    Args:
        params: Flattened JSON config dict.
        usm: ``uptrendSignalMethod`` value.

    Returns:
        Trial parameter dict with all required keys.
    """
    p = {k: params[k] for k in _JSON_PARAM_KEYS if k in params}
    p["uptrendSignalMethod"] = usm
    p["paramNumberToVary"] = -999
    p.update(_generate_weight_constraints())
    p = _apply_ma_constraints(p)
    return p


def _generate_json_plus_one_param(
    params: dict,
    iter_num: int,
    usm: str,
    ranges: dict,
) -> dict:
    """Build a trial dict from JSON values with one parameter randomised.

    Copies all JSON config values, then randomly selects one parameter
    (``paramNumberToVary``) and replaces it with a triangular-sampled
    value drawn from ``ranges``.

    Args:
        params: Flattened JSON config dict.
        iter_num: Current iteration index (used for logging only).
        usm: ``uptrendSignalMethod`` value.
        ranges: Per-scenario (low, mid, high) triples from ``_get_ranges``.

    Returns:
        Trial parameter dict with all required keys.
    """
    p = {k: params[k] for k in _JSON_PARAM_KEYS if k in params}
    paramNumberToVary = stdlib_random.randint(0, 12)
    # Replace exactly one parameter with a range-sampled value.
    if paramNumberToVary == 0:
        p["numberStocksTraded"] = stdlib_random.randint(3, 12)
    elif paramNumberToVary == 1:
        p["LongPeriod"] = int(random_triangle(*ranges["LongPeriod"]))
    elif paramNumberToVary == 2:
        p["stddevThreshold"] = random_triangle(*ranges["stddevThreshold"])
    elif paramNumberToVary == 3:
        p["MA1"] = int(random_triangle(*ranges["MA1"]))
    elif paramNumberToVary == 4:
        p["MA2"] = int(random_triangle(*ranges["MA2"]))
    elif paramNumberToVary == 5:
        p["MA2offset"] = int(random_triangle(*ranges["MA2offset"]))
    elif paramNumberToVary == 6:
        p["sma2factor"] = random_triangle(*ranges["sma2factor"])
    elif paramNumberToVary == 7:
        p["rankThresholdPct"] = random_triangle(
            *ranges["rankThresholdPct"]
        )
    elif paramNumberToVary == 8:
        p["riskDownside_min"] = random_triangle(
            *ranges["riskDownside_min"]
        )
    elif paramNumberToVary == 9:
        p["riskDownside_max"] = random_triangle(
            *ranges["riskDownside_max"]
        )
    elif paramNumberToVary == 10:
        p["sma_filt_val"] = random_triangle(*ranges["sma_filt_val"])
    elif paramNumberToVary == 11:
        p["lowPct"] = random_triangle(*ranges["lowPct"])
    elif paramNumberToVary == 12:
        p["hiPct"] = random_triangle(*ranges["hiPct"])
    p["uptrendSignalMethod"] = usm
    p["paramNumberToVary"] = paramNumberToVary
    p.update(_generate_weight_constraints())
    p = _apply_ma_constraints(p)
    logger.debug(
        "Trial %d: JSON+one phase, varying param %d",
        iter_num, paramNumberToVary,
    )
    return p


def _generate_exploration_params(
    hold_months: list,
    usm: str,
    ranges: dict,
) -> dict:
    """Build a trial dict by sampling all parameters from ranges.

    All numeric trading parameters are drawn independently from
    triangular distributions.  ``monthsToHold`` is sampled uniformly
    from ``hold_months`` and ``numberStocksTraded`` uses randint.

    Args:
        hold_months: List of valid holding-period values (months).
        usm: ``uptrendSignalMethod`` value.
        ranges: Per-scenario (low, mid, high) triples from ``_get_ranges``.

    Returns:
        Trial parameter dict with all required keys.
    """
    MA2 = int(random_triangle(*ranges["MA2"]))
    MA2offset = int(random_triangle(*ranges["MA2offset"]))
    p = {
        "monthsToHold": choice(hold_months),
        "numberStocksTraded": stdlib_random.randint(3, 12),
        "LongPeriod": int(random_triangle(*ranges["LongPeriod"])),
        "stddevThreshold": random_triangle(*ranges["stddevThreshold"]),
        "MA1": int(random_triangle(*ranges["MA1"])),
        "MA2": MA2,
        "MA3": MA2 + MA2offset,
        "MA2offset": MA2offset,
        "sma2factor": random_triangle(*ranges["sma2factor"]),
        "rankThresholdPct": random_triangle(*ranges["rankThresholdPct"]),
        "riskDownside_min": random_triangle(*ranges["riskDownside_min"]),
        "riskDownside_max": random_triangle(*ranges["riskDownside_max"]),
        "sma_filt_val": random_triangle(*ranges["sma_filt_val"]),
        "lowPct": random_triangle(*ranges["lowPct"]),
        "hiPct": random_triangle(*ranges["hiPct"]),
        "uptrendSignalMethod": usm,
        "paramNumberToVary": -999,
    }
    p.update(_generate_weight_constraints())
    p = _apply_ma_constraints(p)
    return p


def generate_random_parameters(
    hold_months: list,
    iter_num: int,
    total_trials: int,
    params: dict = None,
) -> dict:
    """Generate random parameters for a single Monte Carlo trial.

    Dispatches to one of three modes based on ``iter_num``:

    1. **JSON exact** (``iter_num == total_trials - 1``, when
       ``total_trials > 1``): copy JSON config values unchanged.  Gives
       a reproducible single-point baseline at the end of every run.
    2. **Exploration** (first 90 % of non-final trials): draw all
       parameters from broad triangular distributions.
    3. **JSON + one varied** (remaining ~10 % of non-final trials):
       start from JSON config values, replace exactly one randomly
       selected parameter with a range-sampled value.

    With ``total_trials == 1`` the single trial uses exploration mode
    so that random sampling is always exercised.

    Args:
        hold_months: List of valid holding-period values (months) to
            sample from.
        iter_num: Current iteration index (0-based).
        total_trials: Total number of Monte Carlo trials.
        params: Flattened JSON config dict.  Must contain
            ``symbols_file`` and ``uptrendSignalMethod`` for correct
            scenario selection.
    """
    usm = (params or {}).get("uptrendSignalMethod", "percentileChannels")
    swm = (params or {}).get(
        "stockWeightMethod", "delta_rank_sharpe_weight"
    )
    ranges = _get_ranges(params or {})
    # Boundary between exploration (first 90 %) and JSON+one (last 10 %)
    # of non-final trials.  max(1, ...) guarantees at least one exploration
    # trial even when total_trials is very small.
    mid = max(1, round(0.9 * (total_trials - 1)))

    # Final trial: JSON values verbatim (skip when only one trial).
    if total_trials > 1 and iter_num == total_trials - 1:
        logger.debug("Trial %d: JSON-exact phase", iter_num)
        result = _generate_json_exact_params(params or {}, usm)
        result["stockWeightMethod"] = swm
        return result

    # First half: broad exploration across the full parameter space.
    if iter_num < mid:
        logger.debug("Trial %d: exploration phase", iter_num)
        result = _generate_exploration_params(hold_months, usm, ranges)
        result["stockWeightMethod"] = swm
        return result

    # Second half (minus last): JSON baseline with one param varied.
    result = _generate_json_plus_one_param(
        params or {}, iter_num, usm, ranges
    )
    result["stockWeightMethod"] = swm
    return result
