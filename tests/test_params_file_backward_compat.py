"""
Test backward compatibility with existing 3-column readers.

Verifies that code expecting only 3 columns can still read the file.
"""
import pytest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_can_read_first_three_columns():
    """Verify old code can still read first 3 columns."""
    params_file = os.path.join(
        os.path.dirname(__file__), "..",
        "test_data", "pyTAAAweb_backtestPortfolioValue.params"
    )
    
    # Try alternate location if test_data doesn't exist
    if not os.path.exists(params_file):
        from src.backtest.functions.GetParams import get_json_params, get_performance_store
        json_fn = "/Users/donaldpg/pyTAAA_data/sp500_pine/pytaaa_sp500_pine.json"
        if os.path.exists(json_fn):
            p_store = get_performance_store(json_fn)
            params_file = os.path.join(p_store, "pyTAAAweb_backtestPortfolioValue.params")
    
    if not os.path.exists(params_file):
        pytest.skip("Params file not found - run backtest first")
    
    print(f"\nTesting backward compatibility with: {params_file}")
    
    # Simulate old reader code that only reads first 3 columns
    errors = []
    line_count = 0
    
    with open(params_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                cols = line.strip().split()
                
                # Old code only reads first 3 columns
                date = cols[0]
                buyhold = float(cols[1])
                traded = float(cols[2])
                # Ignore columns 3 and 4 (new highs/lows)
                
                # Should not raise errors
                assert len(date) > 0, "Date should not be empty"
                assert buyhold > 0, "Buy-hold value should be positive"
                assert traded > 0, "Traded value should be positive"
                
                line_count += 1
                
            except Exception as e:
                errors.append(f"Line {line_num}: {e}")
                if len(errors) > 5:  # Only collect first 5 errors
                    break
    
    if errors:
        pytest.fail("Backward compatibility issues:\n" + "\n".join(errors))
    
    print(f"✓ Successfully read {line_count} lines using 3-column format")
    print(f"✓ Backward compatibility maintained")


def test_numpy_loadtxt_first_three_columns():
    """Test that numpy.loadtxt can read first 3 columns."""
    params_file = os.path.join(
        os.path.dirname(__file__), "..",
        "test_data", "pyTAAAweb_backtestPortfolioValue.params"
    )
    
    # Try alternate location
    if not os.path.exists(params_file):
        from src.backtest.functions.GetParams import get_json_params, get_performance_store
        json_fn = "/Users/donaldpg/pyTAAA_data/sp500_pine/pytaaa_sp500_pine.json"
        if os.path.exists(json_fn):
            p_store = get_performance_store(json_fn)
            params_file = os.path.join(p_store, "pyTAAAweb_backtestPortfolioValue.params")
    
    if not os.path.exists(params_file):
        pytest.skip("Params file not found - run backtest first")
    
    import numpy as np
    
    # Old code might use numpy loadtxt with usecols
    try:
        data = np.loadtxt(
            params_file,
            usecols=(1, 2),  # Only load columns 1 and 2 (buyhold, traded)
            dtype=float
        )
        
        assert data.shape[0] > 0, "Should load some data"
        assert data.shape[1] == 2, "Should have 2 columns"
        assert np.all(data > 0), "All values should be positive"
        
        print(f"✓ numpy.loadtxt successfully loaded {data.shape[0]} rows")
        print(f"✓ First 5 rows:\n{data[:5]}")
        
    except Exception as e:
        pytest.fail(f"numpy.loadtxt failed: {e}")


def test_pandas_read_csv_first_three_columns():
    """Test that pandas can read first 3 columns."""
    params_file = os.path.join(
        os.path.dirname(__file__), "..",
        "test_data", "pyTAAAweb_backtestPortfolioValue.params"
    )
    
    # Try alternate location
    if not os.path.exists(params_file):
        from src.backtest.functions.GetParams import get_json_params, get_performance_store
        json_fn = "/Users/donaldpg/pyTAAA_data/sp500_pine/pytaaa_sp500_pine.json"
        if os.path.exists(json_fn):
            p_store = get_performance_store(json_fn)
            params_file = os.path.join(p_store, "pyTAAAweb_backtestPortfolioValue.params")
    
    if not os.path.exists(params_file):
        pytest.skip("Params file not found - run backtest first")
    
    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not available")
    
    # Old code might use pandas
    try:
        df = pd.read_csv(
            params_file,
            sep=r'\s+',
            names=['date', 'buyhold', 'traded', 'highs', 'lows'],
            usecols=['date', 'buyhold', 'traded']  # Only use first 3 columns
        )
        
        assert len(df) > 0, "Should load some data"
        assert list(df.columns) == ['date', 'buyhold', 'traded'], \
            "Should have correct column names"
        
        print(f"✓ pandas successfully loaded {len(df)} rows")
        print(f"✓ Columns: {list(df.columns)}")
        print(f"✓ First 5 rows:\n{df.head()}")
        
    except Exception as e:
        pytest.fail(f"pandas read failed: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
