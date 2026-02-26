"""
Create mock HDF5 with real structure (dates, tickers) but synthetic prices.

This allows running actual PyTAAA backtest code on known data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from opensimplex import OpenSimplex

# Real NAZ100 HDF5 as template
REAL_HDF5 = Path("/Users/donaldpg/pyTAAA_data/Naz100/symbols/Naz100_Symbols_.hdf5")
MOCK_HDF5 = Path(__file__).parent / "data" / "mock_naz100_for_backtest.hdf5"

print(f"Loading real HDF5 structure from: {REAL_HDF5}")
store = pd.HDFStore(str(REAL_HDF5), mode='r')
real_df = store.get('/Naz100_Symbols')
store.close()

print(f"Real HDF5 shape: {real_df.shape}")
print(f"Date range: {real_df.index[0]} to {real_df.index[-1]}")
print(f"Tickers: {list(real_df.columns[:10])}")

# Create synthetic data with same structure
dates = real_df.index
tickers = real_df.columns
n_dates = len(dates)
n_tickers = len(tickers)

print(f"\nGenerating synthetic prices: {n_dates} dates × {n_tickers} tickers")

# Initialize prices
prices = np.zeros((n_dates, n_tickers), dtype=float)

# For each ticker, generate synthetic price series with trend
for j, ticker in enumerate(tickers):
    # Starting price
    start_price = 100.0 + (j % 50) * 2  # Vary initial prices
    
    # Assign deterministic CAGR based on ticker index (cyclical)
    # This creates diversity in performance
    cagr_values = np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24])
    cagr = cagr_values[j % len(cagr_values)]
    
    # Generate price series with trend + noise
    opensimplex = OpenSimplex(seed=j)  # Deterministic per ticker
    
    for i in range(n_dates):
        # Geometric trend
        t_years = i / 252.0  # Approximate trading days per year
        trend = start_price * np.exp(cagr * t_years)
        
        # Oscillation noise (OpenSimplex)
        noise_val = opensimplex.noise2(i / 100.0, j / 50.0)
        # Scale noise to ±5%
        noise_mult = 1 + (noise_val * 0.05)
        
        prices[i, j] = trend * noise_mult
    
    if (j + 1) % 20 == 0:
        print(f"  Generated {j + 1}/{n_tickers} ticker series")

# Create DataFrame with same structure as real
mock_df = pd.DataFrame(prices, index=dates, columns=tickers)

# Ensure no NaN or inf
mock_df = mock_df.fillna(100.0)
mock_df = mock_df.replace([np.inf, -np.inf], 100.0)

print(f"Mock data shape: {mock_df.shape}")
print(f"Mock data range: ${mock_df.min().min():.2f} to ${mock_df.max().max():.2f}")

# Save to HDF5
MOCK_HDF5.parent.mkdir(parents=True, exist_ok=True)
store = pd.HDFStore(str(MOCK_HDF5), mode='w')
store.put('/Naz100_Symbols', mock_df, format='table')
store.close()

print(f"\nSaved mock HDF5: {MOCK_HDF5}")
print(f"File size: {MOCK_HDF5.stat().st_size / 1024 / 1024:.1f} MB")

# Verify
store = pd.HDFStore(str(MOCK_HDF5), mode='r')
verify_df = store.get('/Naz100_Symbols')
print(f"\nVerified mock HDF5:")
print(f"  Shape: {verify_df.shape}")
print(f"  Date range: {verify_df.index[0]} to {verify_df.index[-1]}")
print(f"  First ticker stats: {verify_df.iloc[:, 0].describe()}")
store.close()
