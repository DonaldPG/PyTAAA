"""
LEGACY — Phase 2 evaluation script, SUPERSEDED.

This script reads a CSV written by experiment_future_prices.py.  That
script was never completed (it depends on the HDF5-file-patching approach
that was replaced by in-memory numpy patching in run_lookahead_study.py).

The CSV this script expects (experiment_output/lookahead_bias_results.csv)
has never been created, so this script will always exit with an error.

For actual look-ahead bias evaluation, run:
    PYTHONPATH=$(pwd) uv run python studies/lookahead_bias/run_lookahead_study.py
or:
    PYTHONPATH=$(pwd) uv run pytest tests/test_lookahead_bias.py -v
"""

import sys
from pathlib import Path
import csv

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def evaluate_lookahead_bias():
    """
    Read the experiment results CSV and report pass/fail.
    
    Pass criterion: selection_changed == False for all test date × model
    combinations.
    """
    results_file = Path(__file__).parent / "experiment_output" / "lookahead_bias_results.csv"
    
    if not results_file.exists():
        print(f"[ERROR] Results file not found: {results_file}")
        print("Have you run experiment_future_prices.py yet?")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("PHASE 2: LOOKAHEAD BIAS EVALUATION")
    print("=" * 70)
    
    # Read CSV
    results = []
    with open(results_file, "r") as f:
        reader = csv.DictReader(f)
        results = list(reader)
    
    print(f"\nRead {len(results)} test results")
    
    # Evaluate
    passed = []
    failed = []
    
    for row in results:
        date = row["date"]
        model = row["model"]
        selection_changed = row["selection_changed"].lower() == "true"
        rank_corr = float(row["rank_correlation"])
        
        status = "FAIL" if selection_changed else "PASS"
        
        if selection_changed:
            failed.append({
                "date": date,
                "model": model,
                "selection_changed": selection_changed,
                "rank_correlation": rank_corr,
            })
            print(f"[{status}] {date} {model:<20} "
                  f"selection_changed={selection_changed} "
                  f"corr={rank_corr:.3f}")
        else:
            passed.append({
                "date": date,
                "model": model,
                "selection_changed": selection_changed,
                "rank_correlation": rank_corr,
            })
            print(f"[{status}] {date} {model:<20} "
                  f"selection_changed={selection_changed} "
                  f"corr={rank_corr:.3f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Passed: {len(passed)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")
    
    if failed:
        print("\nFailed tests (selection changed with modified future prices):")
        for row in failed:
            print(f"  - {row['date']} {row['model']}: "
                  f"correlation={row['rank_correlation']:.3f}")
        print("\n[RESULT] ✗ FAIL - Some stocks changed when future prices were modified")
        return False
    else:
        print("\n[RESULT] ✓ PASS - All stocks remained the same despite future price changes")
        return True


if __name__ == "__main__":
    success = evaluate_lookahead_bias()
    sys.exit(0 if success else 1)
