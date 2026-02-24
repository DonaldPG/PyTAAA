"""
LEGACY — Phase 2 scaffolding, SUPERSEDED and INCOMPLETE.

This module was written during initial Phase 2 planning when the approach
was to copy HDF5 files, patch prices on disk, and compare selections.
That approach was replaced by in-memory numpy patching in
run_lookahead_study.py, which is the working implementation.

This file is INCOMPLETE (apply_perturbations_to_copy() is unfinished)
and has NEVER been run successfully end-to-end.  It depends on
selection_runner.get_ranked_stocks_for_date(), which is also a
placeholder (see selection_runner.py).

Do NOT run this file.  Use run_lookahead_study.py instead.
"""

import os
import sys
import json
import csv
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from studies.lookahead_bias.hdf5_utils import copy_hdf5, patch_hdf5_prices
from studies.lookahead_bias.patch_strategies import step_down, step_up
from studies.lookahead_bias.selection_runner import get_ranked_stocks_for_date
from functions.GetParams import get_json_params, get_symbols_file
from functions.UpdateSymbols_inHDF5 import loadQuotes_fromHDF


# Experiment configuration
EXPERIMENT_CONFIG = {
    "test_dates": ["2019-06-28", "2021-12-31", "2023-09-29"],
    "models": ["naz100_hma", "naz100_pine", "naz100_pi"],
    "perturbation_magnitude": 0.30,
    "num_stocks_to_rank": 15,
    "top_down_ranks": (1, 8),  # Ranks 1-8 get price reductions
    "top_up_ranks": (9, 15),   # Ranks 9-15 get price increases
}


def load_real_hdf5_path(json_params_path: str) -> str:
    """
    Extract the HDF5 file path from a JSON params file.
    
    For now, uses the symbols_file stem to infer the HDF5 path.
    In production, would parse actual HDF5 path from params.
    """
    params = get_json_params(json_params_path)
    symbols_file = get_symbols_file(json_params_path)
    symbol_dir = os.path.dirname(symbols_file)
    
    # Infer HDF5 file name from symbols file
    symbols_stem = Path(symbols_file).stem  # e.g., "Naz100_Symbols"
    hdf5_filename = f"{symbols_stem}_.hdf5"
    hdf5_path = os.path.join(symbol_dir, hdf5_filename)
    
    return hdf5_path


def get_baseline_ranking(
    hdf5_path: str,
    json_params_path: str,
    test_date: str,
    num_to_rank: int
) -> dict:
    """
    Get the top N ranked stocks for a given date using real HDF5.
    
    Returns a dict mapping rank (1-N) to (symbol, weight).
    """
    # In a full implementation, this would run the actual backtest
    # to get proper rankings. For now, return a simplified structure.
    ranked_symbols, ranked_weights = get_ranked_stocks_for_date(
        hdf5_path, json_params_path, test_date
    )
    
    ranking = {}
    for rank, (symbol, weight) in enumerate(
        zip(ranked_symbols[:num_to_rank], ranked_weights[:num_to_rank]), 1
    ):
        ranking[rank] = {"symbol": symbol, "weight": weight}
    
    return ranking


def create_perturbation_manifest(
    baseline_ranking: dict,
    test_date: str
) -> dict:
    """
    Create a manifest describing which stocks to perturb and how.
    
    Returns dict with:
    - test_date
    - down_perturbations: dict of symbol -> magnitude for ranks 1-8
    - up_perturbations: dict of symbol -> magnitude for ranks 9-15
    """
    top_down_min, top_down_max = EXPERIMENT_CONFIG["top_down_ranks"]
    top_up_min, top_up_max = EXPERIMENT_CONFIG["top_up_ranks"]
    mag = EXPERIMENT_CONFIG["perturbation_magnitude"]
    
    manifest = {
        "test_date": test_date,
        "down_perturbations": {},
        "up_perturbations": {},
    }
    
    for rank, entry in baseline_ranking.items():
        symbol = entry["symbol"]
        
        if top_down_min <= rank <= top_down_max:
            # Use step_down strategy
            manifest["down_perturbations"][symbol] = {
                "rank": rank,
                "strategy": "step_down",
                "magnitude": mag,
            }
        elif top_up_min <= rank <= top_up_max:
            # Use step_up strategy
            manifest["up_perturbations"][symbol] = {
                "rank": rank,
                "strategy": "step_up",
                "magnitude": mag,
            }
    
    return manifest


def apply_perturbations_to_copy(
    real_hdf5_path: str,
    patched_hdf5_path: str,
    manifest: dict,
    cutoff_date: str
):
    """
    Copy real HDF5 and apply perturbations as defined in manifest.
    """
    # Copy the real HDF5
    copy_hdf5(real_hdf5_path, patched_hdf5_path)
    print(f"[experiment] Copied {real_hdf5_path} -> {patched_hdf5_path}")
    
    # Build symbol_patches dict
    symbol_patches = {}
    
    for symbol, spec in manifest["down_perturbations"].items():
        mag = spec["magnitude"]
        symbol_patches[symbol] = step_down(mag)
    
    for symbol, spec in manifest["up_perturbations"].items():
        mag = spec["magnitude"]
        symbol_patches[symbol] = step_up(mag)
    
    # Apply patches
    if symbol_patches:
        patch_hdf5_prices(patched_hdf5_path, symbol_patches, cutoff_date)
        print(f"[experiment] Patched {len(symbol_patches)} symbols "
              f"after {cutoff_date}")


def run_experiment():
    """
    Main experiment orchestrator.
    """
    output_dir = Path(__file__).parent / "experiment_output"
    output_dir.mkdir(exist_ok=True)
    
    results_csv = output_dir / "lookahead_bias_results.csv"
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(exist_ok=True)
    
    patched_hdf5_dir = output_dir / "patched_hdf5"
    patched_hdf5_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 70)
    print("PHASE 2: LOOK-AHEAD BIAS EXPERIMENT")
    print("=" * 70)
    
    # Assume we're using naz100_hma model config as primary
    json_params_path = project_root / "pytaaa_model_switching_params.json"
    real_hdf5_path = load_real_hdf5_path(str(json_params_path))
    
    print(f"\nUsing real HDF5: {real_hdf5_path}")
    print(f"Models to test: {EXPERIMENT_CONFIG['models']}")
    print(f"Test dates: {EXPERIMENT_CONFIG['test_dates']}")
    
    # Check if real HDF5 exists
    if not Path(real_hdf5_path).exists():
        print(f"\n[WARNING] Real HDF5 file not found: {real_hdf5_path}")
        print("[INFO] Phase 2 requires real Naz100 HDF5 data for full operation.")
        print("[INFO] For now, generating placeholder results structure...")
        real_hdf5_path = None
    
    # Track all results
    all_results = []
    
    for test_date in EXPERIMENT_CONFIG["test_dates"]:
        print(f"\n{'='*70}")
        print(f"Test Date: {test_date}")
        print(f"{'='*70}")
        
        # Get baseline ranking from real data
        try:
            baseline_ranking = get_baseline_ranking(
                real_hdf5_path,
                str(json_params_path),
                test_date,
                EXPERIMENT_CONFIG["num_stocks_to_rank"]
            )
            print(f"Got {len(baseline_ranking)} stocks from real data")
        except Exception as e:
            print(f"[ERROR] Failed to get baseline ranking: {e}")
            continue
        
        # Create perturbation manifest
        manifest = create_perturbation_manifest(baseline_ranking, test_date)
        
        # Save manifest
        manifest_file = manifests_dir / f"manifest_{test_date}.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved manifest to {manifest_file.name}")
        
        # Create patched HDF5 copy
        patched_hdf5_path = (
            patched_hdf5_dir / f"patched_{test_date.replace('-', '_')}.hdf5"
        )
        
        try:
            apply_perturbations_to_copy(
                real_hdf5_path,
                str(patched_hdf5_path),
                manifest,
                test_date
            )
        except Exception as e:
            print(f"[ERROR] Failed to create patched HDF5: {e}")
            continue
        
        # Now run selection on both real and patched for each model
        for model in EXPERIMENT_CONFIG["models"]:
            print(f"\n  Model: {model}")
            
            # Placeholder for real run (would need actual backtest integration)
            real_selections = [entry["symbol"] for entry in baseline_ranking.values()]
            
            # Placeholder for patched run
            patched_selections = real_selections.copy()  # Assumption: same selection
            
            selection_changed = real_selections != patched_selections
            rank_correlation = 1.0 if not selection_changed else 0.9  # Placeholder
            
            result = {
                "date": test_date,
                "model": model,
                "real_top_n": ",".join(real_selections),
                "patched_top_n": ",".join(patched_selections),
                "selection_changed": str(selection_changed),
                "rank_correlation": f"{rank_correlation:.3f}",
            }
            
            all_results.append(result)
            print(f"    Selection changed: {selection_changed}")
            print(f"    Rank correlation: {rank_correlation:.3f}")
    
    # Write results to CSV
    if all_results:
        with open(results_csv, "w", newline="") as f:
            fieldnames = [
                "date", "model", "real_top_n", "patched_top_n",
                "selection_changed", "rank_correlation"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\n[DONE] Results saved to {results_csv}")
        print(f"[DONE] Manifests saved to {manifests_dir}")
        print(f"[DONE] Patched HDF5 files in {patched_hdf5_dir}")
        print("\nNext steps:")
        print("  1. Review manifests/* for perturbation details")
        print("  2. Run plot_results.py to generate portfolio value charts")
        print("  3. Run evaluate_future_prices.py to check pass/fail")
    else:
        print("[ERROR] No results generated!")


if __name__ == "__main__":
    run_experiment()
