"""Random parameter generation for Monte Carlo backtesting.

This module provides functions for sampling trading parameters from
statistical distributions.  Two exploration strategies are used:

- **Exploration phase** (early trials): broad triangular distributions to
  discover promising regions of the parameter space.
- **Base phase** (later trials): start from known-good pyTAAA defaults and
  perturb a single parameter per trial.
- **Linux edition** (final trial): use the Pi Linux edition defaults.
"""

import random as stdlib_random
from random import choice
from typing import Union

import numpy as np
from numpy import random

from functions.logger_config import get_logger

logger = get_logger(__name__, log_file="pytaaa_backtest_montecarlo.log")

#############################################################################
# Base and Linux edition parameter defaults
#############################################################################

_BASE_DEFAULTS: dict = {
    "numberStocksTraded": 6,
    "monthsToHold": 1,
    "LongPeriod": 412,
    "stddevThreshold": 8.495,
    "MA1": 264,
    "MA2": 22,
    "MA3": 26,
    "MA2offset": 4,
    "sma2factor": 3.495,
    "rankThresholdPct": 0.3210,
    "riskDownside_min": 0.855876,
    "riskDownside_max": 16.9086,
    "sma_filt_val": 0.02988,
}

_LINUX_DEFAULTS: dict = {
    "numberStocksTraded": 7,
    "monthsToHold": 1,
    "LongPeriod": 455,
    "stddevThreshold": 6.12,
    "MA1": 197,
    "MA2": 19,
    "MA3": 21,
    "MA2offset": 2,
    "sma2factor": 1.46,
    "rankThresholdPct": 0.132,
    "riskDownside_min": 0.5,
    "riskDownside_max": 7.4,
    "sma_filt_val": 0.02988,
}


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


def generate_random_parameters(
    hold_months: list,
    iter_num: int,
    total_trials: int,
    runs_fraction: int = 4,
) -> dict:
    """Generate random parameters for a single Monte Carlo trial.

    Three phases are used based on ``iter_num``:

    1. **Exploration phase** (``iter_num < total_trials / runs_fraction``):
       Draw all parameters from broad triangular distributions.
    2. **Base phase** (``iter_num >= total_trials / runs_fraction``):
       Start from pyTAAA default parameters and vary one randomly
       selected parameter slightly.
    3. **Linux edition phase** (``iter_num == total_trials - 1``):
       Use the Pi Linux edition fixed defaults without variation.

    Args:
        hold_months: List of valid holding-period values (months) to
            sample from.
        iter_num: Current iteration index (0-based).
        total_trials: Total number of Monte Carlo trials.
        runs_fraction: Denominator that determines the boundary between
            exploration and base phases.  Default is 4 (25 % exploration).

    Returns:
        Dictionary with all parameters required by
        ``run_single_monte_carlo_realization``.  Keys:
        monthsToHold, numberStocksTraded, LongPeriod, stddevThreshold,
        MA1, MA2, MA3, MA2offset, sma2factor, rankThresholdPct,
        riskDownside_min, riskDownside_max, lowPct, hiPct,
        uptrendSignalMethod, sma_filt_val, max_weight_factor,
        min_weight_factor, absolute_max_weight, apply_constraints,
        paramNumberToVary.

    Example:
        >>> params = generate_random_parameters([1, 2, 3], 0, 100)
        >>> "LongPeriod" in params
        True
    """
    #############################################################################
    # Common random parameters shared across all phases
    #############################################################################
    lowPct = random.uniform(10.0, 30.0)
    hiPct = random.uniform(70.0, 90.0)
    weight_constraints = _generate_weight_constraints()

    paramNumberToVary = -999

    #############################################################################
    # Linux edition: fixed defaults, final trial only
    #############################################################################
    if iter_num == total_trials - 1:
        p = dict(_LINUX_DEFAULTS)
        p["monthsToHold"] = 1
        p["lowPct"] = lowPct
        p["hiPct"] = hiPct
        p["uptrendSignalMethod"] = "percentileChannels"
        p.update(weight_constraints)
        p["paramNumberToVary"] = paramNumberToVary
        p = _apply_ma_constraints(p)
        logger.debug("Trial %d: Linux edition defaults", iter_num)
        return p

    #############################################################################
    # Base phase: start from known-good defaults, vary one parameter
    #############################################################################
    if iter_num >= total_trials / runs_fraction:
        p = dict(_BASE_DEFAULTS)
        p["monthsToHold"] = 1

        paramNumberToVary = choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        p["paramNumberToVary"] = paramNumberToVary

        if paramNumberToVary == 0:
            p["numberStocksTraded"] += choice([-1, 0, 1])
        elif paramNumberToVary == 1:
            for _ in range(15):
                candidate = choice(hold_months)
                if candidate != p["monthsToHold"]:
                    p["monthsToHold"] = candidate
                    break
        elif paramNumberToVary == 2:
            delta = random.uniform(
                -0.01 * p["LongPeriod"], 0.01 * p["LongPeriod"]
            )
            p["LongPeriod"] = int(p["LongPeriod"] + delta)
        elif paramNumberToVary == 3:
            delta = random.uniform(-0.01 * p["MA1"], 0.01 * p["MA1"])
            p["MA1"] = int(p["MA1"] + delta)
        elif paramNumberToVary == 4:
            delta = random.uniform(-0.01 * p["MA2"], 0.01 * p["MA2"])
            p["MA2"] = int(p["MA2"] + delta)
        elif paramNumberToVary == 5:
            p["MA2offset"] = choice([1, 2, 3])
        elif paramNumberToVary == 6:
            delta = random.uniform(
                -0.01 * p["sma2factor"], 0.01 * p["sma2factor"]
            )
            p["sma2factor"] = round(p["sma2factor"] + delta, 3)
        elif paramNumberToVary == 7:
            delta = random.uniform(
                -0.01 * p["rankThresholdPct"],
                0.01 * p["rankThresholdPct"],
            )
            p["rankThresholdPct"] = round(p["rankThresholdPct"] + delta, 2)
        elif paramNumberToVary == 8:
            delta = random.uniform(
                -0.01 * p["riskDownside_min"],
                0.01 * p["riskDownside_min"],
            )
            p["riskDownside_min"] = round(p["riskDownside_min"] + delta, 3)
        elif paramNumberToVary == 9:
            delta = random.uniform(
                -0.01 * p["riskDownside_max"],
                0.01 * p["riskDownside_max"],
            )
            p["riskDownside_max"] = round(p["riskDownside_max"] + delta, 3)
        elif paramNumberToVary == 10:
            p["stddevThreshold"] *= random.uniform(0.8, 1.2)
        elif paramNumberToVary == 11:
            p["sma_filt_val"] *= random.uniform(0.8, 1.2)

        p["lowPct"] = lowPct
        p["hiPct"] = hiPct
        p["uptrendSignalMethod"] = "percentileChannels"
        p.update(weight_constraints)
        p = _apply_ma_constraints(p)
        logger.debug(
            "Trial %d: base phase, varying param %d",
            iter_num, paramNumberToVary,
        )
        return p

    #############################################################################
    # Exploration phase: broad triangular sampling
    #############################################################################
    MA1 = int(random_triangle(low=75, mid=151, high=300))
    MA2 = int(random_triangle(low=10, mid=20, high=50))
    # MA2offset is scaled to ~5-10% of the MA1-MA2 gap, keeping the two
    # moving averages meaningfully apart while allowing natural variation.
    MA2offset = int(
        random_triangle(
            low=max(1, (MA1 - MA2) // 20),
            mid=max(1, (MA1 - MA2) // 15),
            high=max(1, (MA1 - MA2) // 10),
        )
    )

    p = {
        "numberStocksTraded": choice([5, 6, 6, 7, 7, 8, 8]),
        "monthsToHold": choice([1, 1, 1, 1, 1, 1, 1, 1, 1, 2]),
        "LongPeriod": int(random_triangle(low=190, mid=370, high=550)),
        "stddevThreshold": random_triangle(low=5.0, mid=7.50, high=10.0),
        "MA1": MA1,
        "MA2": MA2,
        "MA2offset": MA2offset,
        "sma2factor": random_triangle(low=1.65, mid=2.5, high=2.75),
        "rankThresholdPct": random_triangle(low=0.14, mid=0.20, high=0.26),
        "riskDownside_min": random_triangle(low=0.50, mid=0.70, high=0.90),
        "riskDownside_max": random_triangle(low=8.0, mid=10.0, high=13.0),
        "sma_filt_val": random_triangle(low=0.010, mid=0.015, high=0.0225),
        "lowPct": lowPct,
        "hiPct": hiPct,
        "uptrendSignalMethod": "percentileChannels",
        "paramNumberToVary": paramNumberToVary,
    }
    p.update(weight_constraints)
    p = _apply_ma_constraints(p)
    logger.debug("Trial %d: exploration phase", iter_num)
    return p
