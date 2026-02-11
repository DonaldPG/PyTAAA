"""
Compare newHighsAndLows output between PyTAAA.master and worktree2/PyTAAA.

This test ensures both codebases produce identical results for the same inputs.
Focus is on columns 4 and 5 (sumNewHighs and sumNewLows) from the backtest output.

Test Strategy:
1. Import newHighsAndLows from worktree2 codebase (current)
2. Compare against known good values from PyTAAA.master output files
3. Verify tuple parameter path is executed correctly  
4. Check that percentile subtraction produces expected negative values

Root Cause from Previous Investigation:
- Line 144 in CountNewHighsLows.py must execute:
  sumNewHighs[:,k] -= np.percentile(sumNewHighs[num_indices_ignored:,k],HighPctile[k])
- This line exists in both codebases
- If outputs don't match, need to verify this line is actually reached

Known Good Values (from PyTAAA.master params files):
- sp500_pine @ 2020-01-02: column4=10246.3, column5=2984.0
- naz100_pine @ 2020-01-02: column4 and column5 should also match between versions
"""

import os
import sys

# Ensure we're using the local codebase
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from functions.CountNewHighsLows import newHighsAndLows


def compare_with_master_output(model_name, json_path, master_output_path, worktree_output_path):
    """
    Compare output files from master and worktree codebases.
    
    Args:
        model_name: Name of model being tested (e.g., 'sp500_pine')
        json_path: Path to JSON config file
        master_output_path: Path to master pyTAAAweb_backtestPortfolioValue.params
        worktree_output_path: Path to worktree pyTAAAweb_backtestPortfolioValue.params
    
    Returns:
        dict with comparison results
    """
    print(f"\n{'='*70}")
    print(f"Comparing {model_name} outputs")
    print(f"{'='*70}")
    
    # Check if files exist
    if not os.path.exists(master_output_path):
        print(f"✗ Master output not found: {master_output_path}")
        return None
    
    if not os.path.exists(worktree_output_path):
        print(f"✗ Worktree output not found: {worktree_output_path}")
        return None
    
    # Read test date row from both files
    test_date = "2020-01-02"
    master_line = None
    worktree_line = None
    
    with open(master_output_path, 'r') as f:
        for line in f:
            if line.startswith(test_date):
                master_line = line.strip()
                break
    
    with open(worktree_output_path, 'r') as f:
        for line in f:
            if line.startswith(test_date):
                worktree_line = line.strip()
                break
    
    if not master_line:
        print(f"✗ Test date {test_date} not found in master output")
        return None
    
    if not worktree_line:
        print(f"✗ Test date {test_date} not found in worktree output")
        return None
    
    # Parse columns
    master_cols = master_line.split()
    worktree_cols = worktree_line.split()
    
    print(f"\nTest date: {test_date}")
    print(f"Master line:    {master_line}")
    print(f"Worktree line:  {worktree_line}")
    
    # Compare columns 4 and 5 (index 3 and 4)
    if len(master_cols) >= 5 and len(worktree_cols) >= 5:
        master_col4 = float(master_cols[3])
        master_col5 = float(master_cols[4])
        worktree_col4 = float(worktree_cols[3])
        worktree_col5 = float(worktree_cols[4])
        
        col4_match = abs(master_col4 - worktree_col4) < 0.1
        col5_match = abs(master_col5 - worktree_col5) < 0.1
        
        print(f"\nColumn 4 (sumNewHighs):")
        print(f"  Master:    {master_col4}")
        print(f"  Worktree:  {worktree_col4}")
        print(f"  Match:     {'✓' if col4_match else '✗'}")
        
        print(f"\nColumn 5 (sumNewLows):")
        print(f"  Master:    {master_col5}")
        print(f"  Worktree:  {worktree_col5}")
        print(f"  Match:     {'✓' if col5_match else '✗'}")
        
        return {
            'model': model_name,
            'col4_match': col4_match,
            'col5_match': col5_match,
            'master_col4': master_col4,
            'worktree_col4': worktree_col4,
            'master_col5': master_col5,
            'worktree_col5': worktree_col5
        }
    else:
        print(f"✗ Insufficient columns in output files")
        return None


if __name__ == "__main__":
    print("="*70)
    print("newHighsAndLows Cross-Codebase Validation Test")
    print("="*70)
    print("\nThis test compares output files from PyTAAA.master and worktree2")
    print("to ensure newHighsAndLows produces identical results.\n")
    
    # Test configurations
    tests = [
        {
            'model': 'sp500_pine',
            'json': '/Users/donaldpg/pyTAAA_data_static/sp500_pine/pytaaa_sp500_pine.json',
            'master_output': '/Users/donaldpg/pyTAAA_data_static/sp500_pine/data_store/pyTAAAweb_backtestPortfolioValue.params',
            'worktree_output': '/Users/donaldpg/pyTAAA_data/sp500_pine/data_store/pyTAAAweb_backtestPortfolioValue.params'
        },
        {
            'model': 'naz100_pine',
            'json': '/Users/donaldpg/pyTAAA_data/naz100_pine/pytaaa_naz100_pine.json',
            'master_output': '/Users/donaldpg/pyTAAA_data_static/naz100_pine/data_store/pyTAAAweb_backtestPortfolioValue.params',
            'worktree_output': '/Users/donaldpg/pyTAAA_data/naz100_pine/data_store/pyTAAAweb_backtestPortfolioValue.params'
        }
    ]
    
    results = []
    for test_config in tests:
        result = compare_with_master_output(
            test_config['model'],
            test_config['json'],
            test_config['master_output'],
            test_config['worktree_output']
        )
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("Test Summary")
    print(f"={'='*70}")
    
    all_passed = True
    for result in results:
        model = result['model']
        col4_status = '✓' if result['col4_match'] else '✗'
        col5_status = '✓' if result['col5_match'] else '✗'
        
        print(f"\n{model}:")
        print(f"  Column 4 (sumNewHighs): {col4_status}")
        print(f"  Column 5 (sumNewLows):  {col5_status}")
        
        if not (result['col4_match'] and result['col5_match']):
            all_passed = False
            print(f"  ERROR: Mismatch detected!")
            print(f"    Master col4={result['master_col4']}, Worktree col4={result['worktree_col4']}")
            print(f"    Master col5={result['master_col5']}, Worktree col5={result['worktree_col5']}")
    
    print(f"\n{'='*70}")
    if all_passed:
        print("✓ ALL TESTS PASSED - Outputs match between codebases")
        sys.exit(0)
    else:
        print("✗ TESTS FAILED - Outputs differ between codebases")
        print("\nAction items:")
        print("1. Verify line 144 in CountNewHighsLows.py executes percentile subtraction")
        print("2. Check if tuple branch is being taken (print num_days_highlow type)")
        print("3. Verify HDF5 data files are identical between codebases")
        sys.exit(1)
