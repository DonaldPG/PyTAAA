"""
Noise utilities for synthetic CAGR backtest data generation.

This module provides OpenSimplex-based noise generation calibrated to
real market statistics from historical price data.
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from opensimplex import OpenSimplex


class NoiseCalibratorFromHDF5:
    """
    Calibrate noise parameters from real HDF5 price data.
    
    Computes recommended OpenSimplex noise amplitude and frequency
    based on statistics of historical daily returns and extrema spacing.
    """
    
    def __init__(self, hdf5_path: str, key: str = None):
        """
        Args:
            hdf5_path: Path to HDF5 file with price data
            key: HDF5 key (table name); if None, uses first available key
        """
        self.hdf5_path = hdf5_path
        
        # Load price data
        store = pd.HDFStore(hdf5_path, mode='r')
        if key is None:
            key = store.keys()[0]
        
        self.df = store.get(key)
        store.close()
        
        print(f"[NoiseCalibratorFromHDF5] Loaded {len(self.df.columns)} symbols, "
              f"{len(self.df)} time periods")
        
        # Compute statistics
        self._compute_statistics()
    
    def _compute_statistics(self):
        """Compute calibration statistics."""
        # Daily returns
        daily_returns = self.df.pct_change().dropna()
        self.mean_daily_return = daily_returns.values.mean()
        self.std_daily_return = daily_returns.values.std()
        
        # Inter-extrema spacing (days between local min/max)
        spacings = []
        for col in self.df.columns[:10]:  # Sample first 10 columns
            prices = self.df[col].dropna().values
            if len(prices) > 5:
                locs_min = argrelextrema(prices, np.less, order=5)[0]
                locs_max = argrelextrema(prices, np.greater, order=5)[0]
                locs = np.sort(np.concatenate([locs_min, locs_max]))
                if len(locs) > 1:
                    spacing = np.diff(locs)
                    spacings.extend(spacing)
        
        if spacings:
            self.mean_spacing = np.mean(spacings)
            self.std_spacing = np.std(spacings)
        else:
            self.mean_spacing = 10.0
            self.std_spacing = 3.0
        
        print(f"[NoiseCalibratorFromHDF5] Mean daily return: {self.mean_daily_return:.4f}")
        print(f"[NoiseCalibratorFromHDF5] Std daily return: {self.std_daily_return:.4f}")
        print(f"[NoiseCalibratorFromHDF5] Mean inter-extrema spacing: {self.mean_spacing:.1f} days")
        print(f"[NoiseCalibratorFromHDF5] Std inter-extrema spacing: {self.std_spacing:.1f} days")
    
    @property
    def recommended_amplitude(self) -> float:
        """Recommended amplitude for OpenSimplex noise."""
        # Scale to match observed daily volatility
        return self.std_daily_return * 0.5
    
    @property
    def recommended_frequency(self) -> float:
        """Recommended frequency for OpenSimplex noise."""
        # Lower frequency = smoother noise
        # Use inverse of mean spacing to set oscillation frequency
        return 1.0 / max(self.mean_spacing, 5.0)


def opensimplex_noise_1d(n: int, frequency: float, amplitude: float, 
                          seed: int = 42) -> np.ndarray:
    """
    Generate 1D OpenSimplex noise.
    
    Args:
        n: Number of samples
        frequency: Oscillation frequency (typically 0.01 to 0.1)
        amplitude: Noise amplitude (scale factor)
        seed: Random seed for reproducibility
    
    Returns:
        Array of shape (n,) with noise values centered near 0
    """
    np.random.seed(seed)
    
    # Create OpenSimplex generator
    tmp = OpenSimplex(seed=seed)
    
    # Generate noise along a 1D slice
    noise = np.zeros(n)
    for i in range(n):
        # Map indices to continuous coordinates
        x = i * frequency
        # Use 2D noise with fixed y for pseudo-1D behavior
        noise[i] = tmp.noise2(x, 0.0)
    
    # Normalize to [-1, 1] and scale by amplitude
    noise = (noise / 1.0) * amplitude  # OpenSimplex returns [-1, 1]
    
    return noise


def create_synthetic_price_series(
    num_days: int,
    cagr: float,
    start_price: float = 100.0,
    noise_amplitude: float = 0.01,
    noise_frequency: float = 0.02,
    seed: int = 42
) -> np.ndarray:
    """
    Create a synthetic price series with known CAGR and noise.
    
    Args:
        num_days: Number of trading days
        cagr: Compound annual growth rate (e.g., 0.20 for 20% CAGR)
        start_price: Starting price
        noise_amplitude: Amplitude of OpenSimplex noise
        noise_frequency: Frequency of OpenSimplex noise
        seed: Random seed
    
    Returns:
        Array of daily prices
    """
    # Geometric trend: P(t) = P0 * exp(CAGR/252 * t)
    daily_rate = cagr / 252.0
    trend = start_price * np.exp(daily_rate * np.arange(num_days))
    
    # Add OpenSimplex noise
    noise = opensimplex_noise_1d(num_days, noise_frequency, noise_amplitude, seed)
    
    # Combine: price = trend * (1 + noise)
    prices = trend * (1.0 + noise)
    
    # Ensure no prices go negative
    prices = np.maximum(prices, start_price * 0.1)
    
    return prices
