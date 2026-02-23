"""
Price perturbation strategies for look-ahead bias testing.

Each callable takes a pd.Series of prices and returns a modified Series.
Pre-cutoff values (dates <= cutoff_date) are never modified; only
post-cutoff values are altered.
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
