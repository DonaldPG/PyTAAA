"""
Price perturbation strategy callables for look-ahead bias testing.

Each function is a factory that returns a transform callable.  The
callable takes a pd.Series of prices and returns a modified Series.

IMPORTANT: These functions operate on whatever Series is passed to them.
They do NOT enforce pre-cutoff protection themselves.  The caller is
responsible for passing only post-cutoff price rows.  The study's current
in-memory approach (_patch_adjclose in run_lookahead_study.py) handles
slicing and does not use these callables — they exist for use with the
legacy hdf5_utils.patch_hdf5_prices() interface.
"""

import pandas as pd
import numpy as np


def step_down(magnitude: float):
    """
    Create a callable that applies a downward step to post-cutoff prices.

    After the cutoff date, all prices are multiplied by (1 - magnitude).
    For example, magnitude=0.30 yields a 30% price reduction.

    Args:
        magnitude: Fraction to reduce prices (0 < magnitude < 1)

    Returns:
        Callable that applies the downward step to post-cutoff values
    """
    def transform(series: pd.Series) -> pd.Series:
        return series * (1.0 - magnitude)
    
    return transform


def step_up(magnitude: float):
    """
    Create a callable that applies an upward step to post-cutoff prices.

    After the cutoff date, all prices are multiplied by (1 + magnitude).
    For example, magnitude=0.30 yields a 30% price increase.

    Args:
        magnitude: Fraction to increase prices (0 < magnitude < 1)

    Returns:
        Callable that applies the upward step to post-cutoff values
    """
    def transform(series: pd.Series) -> pd.Series:
        return series * (1.0 + magnitude)
    
    return transform


def linear_down(slope: float):
    """
    Create a callable that applies a downward linear gradient to
    post-cutoff prices.

    Prices decrease linearly over time at the specified slope
    (fraction of price per trading day).

    Args:
        slope: Daily price reduction rate (e.g., 0.001 = 0.1% per day)

    Returns:
        Callable that applies the linear downward gradient
    """
    def transform(series: pd.Series) -> pd.Series:
        # Create a linear decay from 1.0 down
        n_periods = len(series)
        # Linearly interpolate from 1.0 to (1.0 - slope * n_periods)
        factors = np.linspace(1.0, max(0.5, 1.0 - slope * n_periods), n_periods)
        return series * factors
    
    return transform


def linear_up(slope: float):
    """
    Create a callable that applies an upward linear gradient to
    post-cutoff prices.

    Prices increase linearly over time at the specified slope
    (fraction of price per trading day).

    Args:
        slope: Daily price increase rate (e.g., 0.001 = 0.1% per day)

    Returns:
        Callable that applies the linear upward gradient
    """
    def transform(series: pd.Series) -> pd.Series:
        # Create a linear growth from 1.0 up
        n_periods = len(series)
        growth_factor = np.linspace(1.0, 1.0 + slope * n_periods, n_periods)
        return series * growth_factor
    
    return transform
