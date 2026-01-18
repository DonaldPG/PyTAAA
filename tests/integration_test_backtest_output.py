"""
Integration test for full backtest pipeline with new columns.

This test runs a minimal backtest and verifies the output format.
"""
import pytest
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_full_backtest_with_new_columns():
    """Run full backtest and verify output format."""
    # Setup
    json_fn = "/Users/donaldpg/pyTAAA_data/sp500_pine/pytaaa_sp500_pine.json"
    
    if not os.path.exists(json_fn):
        pytest.skip("Config file not found - adjust path for your system")
    
    # Import here to avoid issues if module not available
    try:
        from src.backtest.dailyBacktest_pctLong import dailyBacktest_pctLong
    except ImportError:
        pytest.skip("dailyBacktest_pctLong module not available")
    
    print("\n" + "="*70)
    print("Running minimal backtest with 3 Monte Carlo trials...")
    print("="*70)
    
    # Run backtest (with small number of trials for speed)
    try:
        result = dailyBacktest_pctLong(
            json_fn=json_fn,
            randomtrials=3,  # Minimal trials for testing
            holdMonths=[1],
            verbose=True
        )
    except Exception as e:
        pytest.fail(f"Backtest execution failed: {e}")
    
    # Determine params file location
    from src.backtest.functions.GetParams import get_performance_store
    p_store = get_performance_store(json_fn)
    params_file = os.path.join(p_store, "pyTAAAweb_backtestPortfolioValue.params")
    
    # Verify file exists
    assert os.path.exists(params_file), \
        f"Params file should exist at: {params_file}"
    
    print(f"\n✓ Params file found: {params_file}")
    
    # Load and verify structure
    with open(params_file, 'r') as f:
        lines = f.readlines()
    
    assert len(lines) > 0, "File should have content"
    print(f"✓ File has {len(lines)} lines")
    
    # Check first and last lines
    first_cols = lines[0].strip().split()
    last_cols = lines[-1].strip().split()
    
    assert len(first_cols) == 5, \
        f"First line should have 5 columns, got {len(first_cols)}: {lines[0][:100]}"
    assert len(last_cols) == 5, \
        f"Last line should have 5 columns, got {len(last_cols)}: {lines[-1][:100]}"
    
    print(f"✓ First line: {' '.join(first_cols)}")
    print(f"✓ Last line: {' '.join(last_cols)}")
    
    # Parse data
    data = []
    for line in lines:
        cols = line.strip().split()
        if len(cols) == 5:
            date, buyhold, traded, highs, lows = cols
            data.append({
                'date': date,
                'buyhold': float(buyhold),
                'traded': float(traded),
                'highs': int(float(highs)),
                'lows': int(float(lows))
            })
    
    # Validate data integrity
    assert len(data) > 252, f"Should have at least 1 year of data, got {len(data)} days"
    
    print(f"\n✓ Data integrity checks:")
    print(f"  - Total rows: {len(data)}")
    
    # Check sample of rows
    errors = []
    for i, row in enumerate(data):
        try:
            assert row['buyhold'] > 0, f"Buy-hold value should be positive"
            assert row['traded'] > 0, f"Traded value should be positive"
            assert row['highs'] >= 0, f"New highs should be non-negative"
            assert row['lows'] >= 0, f"New lows should be non-negative"
        except AssertionError as e:
            errors.append(f"Line {i+1}: {e}")
            if len(errors) > 5:  # Only collect first 5 errors
                break
    
    if errors:
        pytest.fail("Data validation errors:\n" + "\n".join(errors))
    
    print(f"  - All portfolio values > 0")
    print(f"  - All highs/lows >= 0")
    
    # Check last few rows for reasonable values
    print(f"\n✓ Last 5 rows:")
    for row in data[-5:]:
        print(f"  {row['date']}: B&H={row['buyhold']:.2f}, "
              f"Traded={row['traded']:.2f}, "
              f"Highs={row['highs']}, Lows={row['lows']}")
    
    # Verify highs/lows are not all zeros
    total_highs = sum(row['highs'] for row in data)
    total_lows = sum(row['lows'] for row in data)
    
    assert total_highs > 0, "Total new highs should be > 0 across all days"
    assert total_lows > 0, "Total new lows should be > 0 across all days"
    
    print(f"\n✓ Summary statistics:")
    print(f"  - Total new highs across all days: {total_highs}")
    print(f"  - Total new lows across all days: {total_lows}")
    print(f"  - Average new highs per day: {total_highs/len(data):.2f}")
    print(f"  - Average new lows per day: {total_lows/len(data):.2f}")
    
    print("\n" + "="*70)
    print("✓✓✓ INTEGRATION TEST PASSED ✓✓✓")
    print("="*70)


if __name__ == "__main__":
    # Run test directly
    pytest.main([__file__, "-v", "-s"])
