"""
Synthetic HDF5 builder for backtest validation.

Creates 90 synthetic tickers (10 stocks × 9 CAGR tiers) with 5 years of
daily prices. Includes 6-month rotations where tier assignments change.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from studies.synthetic_cagr.noise_utils import (
    create_synthetic_price_series, NoiseCalibratorFromHDF5
)


# CAGR tiers and stock counts
CAGR_TIERS = [0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.03, 0.00, -0.06]
STOCKS_PER_TIER = 10
TOTAL_STOCKS = len(CAGR_TIERS) * STOCKS_PER_TIER

# 5 years = ~1,260 trading days
TOTAL_TRADING_DAYS = 1260
ROTATION_PERIOD_DAYS = int(252 * 0.5)  # ~6 months = ~126 trading days


def generate_trading_dates(num_days: int) -> list:
    """Generate N trading days (excluding weekends)."""
    dates = []
    current = datetime(2019, 1, 1)
    
    while len(dates) < num_days:
        # Skip weekends
        if current.weekday() < 5:
            dates.append(current.date())
        current += timedelta(days=1)
    
    return dates[:num_days]


def generate_tier_rotation_schedule(num_days: int, rotation_period: int) -> dict:
    """
    Generate a schedule of when tier assignments rotate.
    
    Returns dict mapping day_index -> rotation_date
    """
    rotations = {}
    next_rotation = rotation_period
    
    while next_rotation < num_days:
        # Add jitter: random day within ±10 days of the rotation point
        jitter = np.random.randint(-10, 11)
        actual_rotation = max(rotation_period, next_rotation + jitter)
        
        if actual_rotation < num_days:
            rotations[actual_rotation] = actual_rotation
        
        next_rotation += rotation_period
    
    return rotations


def build_synthetic_hdf5(
    output_hdf5_path: str,
    noise_amplitude: float = 0.015,
    noise_frequency: float = 0.02,
    seed: int = 42
):
    """
    Build a synthetic HDF5 file with known CAGR data.
    
    Args:
        output_hdf5_path: Where to save the HDF5 file
        noise_amplitude: OpenSimplex noise amplitude
        noise_frequency: OpenSimplex noise frequency
        seed: Random seed
    """
    np.random.seed(seed)
    
    print("\n" + "=" * 70)
    print("PHASE 3: SYNTHETIC HDF5 BUILDER")
    print("=" * 70)
    
    print(f"\nBuilding synthetic HDF5:")
    print(f"  Tickers: {TOTAL_STOCKS} ({len(CAGR_TIERS)} tiers × {STOCKS_PER_TIER} stocks)")
    print(f"  CAGRs: {CAGR_TIERS}")
    print(f"  Trading days: {TOTAL_TRADING_DAYS}")
    print(f"  Rotation period: {ROTATION_PERIOD_DAYS} days (~6 months)")
    
    # Generate trading dates
    dates = generate_trading_dates(TOTAL_TRADING_DAYS)
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    
    # Generate initial tier schedule (before rotations)
    tier_assignments = {}  # day_index -> list of (stock_idx, cagr)
    initial_assignment = []
    for tier_idx, cagr in enumerate(CAGR_TIERS):
        for stock_offset in range(STOCKS_PER_TIER):
            stock_idx = tier_idx * STOCKS_PER_TIER + stock_offset
            initial_assignment.append((int(stock_idx), float(cagr)))
    
    tier_assignments[0] = list(initial_assignment)
    
    # Generate rotation schedule
    rotation_schedule = generate_tier_rotation_schedule(
        TOTAL_TRADING_DAYS, ROTATION_PERIOD_DAYS
    )
    
    print(f"  Rotations scheduled: {len(rotation_schedule)}")
    rotations_list = sorted(rotation_schedule.keys())
    if rotations_list:
        print(f"    First rotation at day {rotations_list[0]}")
        print(f"    Last rotation at day {rotations_list[-1]}")
    
    # Generate price data
    print(f"\nGenerating price data...")
    price_data = np.zeros((TOTAL_TRADING_DAYS, TOTAL_STOCKS))
    
    # Track which CAGR each stock has on each day (for ground truth)
    ground_truth = []
    
    for day_idx in range(TOTAL_TRADING_DAYS):
        # Determine tier assignment for this day
        current_assignment = list(initial_assignment)
        for rot_day in rotations_list:
            if day_idx >= rot_day:
                # After rotation day, permute the assignment
                current_assignment = [
                    (idx, cagr) for idx, cagr in list(
                        np.random.permutation(initial_assignment)
                    )
                ]
        
        # Generate price for each stock
        for stock_idx, cagr in current_assignment:
            stock_idx = int(stock_idx)
            cagr = float(cagr)
            
            if day_idx == 0:
                # First day: price = 100 with noise
                noise = np.random.normal(0, 0.01)
                price_data[day_idx, stock_idx] = 100.0 * (1.0 + noise)
            else:
                # Continuation of synthetic series
                daily_rate = cagr / 252.0
                trend_factor = np.exp(daily_rate)
                noise = np.random.normal(0, 0.01)
                price_data[day_idx, stock_idx] = (
                    price_data[day_idx - 1, stock_idx] * trend_factor * (1.0 + noise)
                )
        
        # Record ground truth for this day
        for stock_idx, cagr in current_assignment:
            stock_idx = int(stock_idx)
            cagr = float(cagr)
            ticker = f"SYNTH_{cagr:+.0%}_{stock_idx % STOCKS_PER_TIER + 1:02d}"
            ground_truth.append({
                "date": dates[day_idx],
                "ticker": ticker,
                "assigned_cagr": cagr,
                "price": float(price_data[day_idx, stock_idx]),
            })
        
        if (day_idx + 1) % 252 == 0:
            year = (day_idx + 1) // 252
            print(f"  Generated {year} year(s) of data ({day_idx + 1} days)")
    
    # Create DataFrame with proper DatetimeIndex
    ticker_names = [
        f"SYNTH_{CAGR_TIERS[i // STOCKS_PER_TIER]:+.0%}_{i % STOCKS_PER_TIER + 1:02d}"
        for i in range(TOTAL_STOCKS)
    ]
    
    # Convert dates list to DatetimeIndex
    dates_index = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    df = pd.DataFrame(price_data, columns=ticker_names, index=dates_index)
    
    # Save to HDF5
    output_hdf5_path = Path(output_hdf5_path)
    output_hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.HDFStore(str(output_hdf5_path), mode='w') as store:
        store.put('synthetic_naz100', df, format='table')
    
    print(f"\n[DONE] Saved HDF5 to {output_hdf5_path}")
    print(f"  Shape: {df.shape} (rows × columns)")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    
    # Save ground truth CSV
    ground_truth_df = pd.DataFrame(ground_truth)
    ground_truth_path = output_hdf5_path.parent / "ground_truth.csv"
    ground_truth_df.to_csv(ground_truth_path, index=False)
    print(f"\n[DONE] Saved ground truth to {ground_truth_path}")
    print(f"  Total records: {len(ground_truth_df)}")
    
    return df, ground_truth_df


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "data"
    hdf5_path = data_dir / "synthetic_naz100.hdf5"
    
    df, gt = build_synthetic_hdf5(str(hdf5_path))
    
    # Spot checks
    print("\n[Spot checks]")
    print(f"Sample prices (first 3 days, first 5 tickers):")
    print(df.iloc[:3, :5])
    
    print(f"\nGround truth sample (first, mid, last rows):")
    print(gt.iloc[[0, len(gt)//2, -1], :])
