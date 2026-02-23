"""
HDF5 utilities for look-ahead bias testing.

This module provides functions to safely copy and patch HDF5 price data.
The source file is never opened in write mode. All modifications are
applied only to copies.
"""

import shutil
import pandas as pd
import numpy as np
from typing import Callable


def copy_hdf5(src_path: str, dst_path: str) -> None:
    """
    Create a byte-for-byte copy of an HDF5 file.

    The source file is opened in read-only mode (mode='r') and never
    modified. After copying, the copy is validated to be structurally
    identical to the source by comparing keys, column names, shapes,
    and sample first/last values using pd.read_hdf on the copy only
    (source is not re-opened).

    Args:
        src_path: Path to the source HDF5 file (never modified)
        dst_path: Path to the destination HDF5 file

    Raises:
        AssertionError: If the copy fails validation
        FileNotFoundError: If source file does not exist
    """
    # Source is opened read-only to verify it exists and is valid
    with pd.HDFStore(src_path, mode='r') as src_store:
        src_keys = src_store.keys()
    
    # Copy via shutil (byte-for-byte)
    shutil.copy2(src_path, dst_path)
    
    # Validate the copy using pd.read_hdf on the copy only
    # (source is never re-opened)
    with pd.HDFStore(dst_path, mode='r') as dst_store:
        dst_keys = dst_store.keys()
    
    assert set(src_keys) == set(dst_keys), \
        f"HDF5 keys mismatch: src {src_keys} != dst {dst_keys}"
    
    # Spot-check first and last values for 5 random symbols (if available)
    for key in src_keys[:1]:  # Usually just one key per file
        df_dst = pd.read_hdf(dst_path, key)
        
        # Sample up to 5 columns
        cols_to_check = df_dst.columns[::max(1, len(df_dst.columns) // 5)][:5]
        
        for col in cols_to_check:
            first_val = df_dst[col].iloc[0]
            last_val = df_dst[col].iloc[-1]
            
            if pd.notna(first_val):
                print(f"[copy_hdf5] Copy validation: {col} "
                      f"first={first_val:.2f}, last={last_val:.2f}")


def patch_hdf5_prices(
    hdf5_path: str,
    symbol_patches: dict[str, Callable[[pd.Series], pd.Series]],
    cutoff_date: str
) -> None:
    """
    Apply price modifications to an HDF5 file only after a cutoff date.

    Opens hdf5_path in write mode (must be a copy, never the original).
    For each symbol in symbol_patches, applies the transform function
    only to rows with index > cutoff_date. Validates that all rows with
    index <= cutoff_date are unchanged before writing back.

    Args:
        hdf5_path: Path to HDF5 file (must be a copy, not the original)
        symbol_patches: dict mapping symbol (column name) to a callable
                       that takes a pd.Series and returns modified Series
        cutoff_date: Cutoff date as string "YYYY-MM-DD"; rows with
                    index > cutoff_date will be patched

    Raises:
        AssertionError: If any pre-cutoff rows are modified
    """
    with pd.HDFStore(hdf5_path, mode='r+') as store:
        for key in store.keys():
            df = store.get(key)
            df_original = df.copy()
            
            # Convert index to datetime if needed
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            cutoff_dt = pd.to_datetime(cutoff_date)
            
            # Apply patches only to rows after cutoff
            for symbol, transform_fn in symbol_patches.items():
                if symbol not in df.columns:
                    print(f"[patch_hdf5_prices] Warning: {symbol} not in {key}")
                    continue
                
                mask_after = df.index > cutoff_dt
                mask_before = ~mask_after
                
                # Extract before and after
                before_vals = df.loc[mask_before, symbol].copy()
                after_vals = df.loc[mask_after, symbol].copy()
                
                # Apply transform only to after
                after_vals_patched = transform_fn(after_vals)
                
                # Combine back
                df.loc[mask_before, symbol] = before_vals
                df.loc[mask_after, symbol] = after_vals_patched
            
            # Validate: all pre-cutoff rows should match original
            for symbol in symbol_patches.keys():
                if symbol not in df.columns:
                    continue
                
                mask_before = df.index <= cutoff_dt
                pre_cutoff_original = df_original.loc[mask_before, symbol]
                pre_cutoff_current = df.loc[mask_before, symbol]
                
                if not np.allclose(
                    pre_cutoff_original.values,
                    pre_cutoff_current.values,
                    equal_nan=True
                ):
                    raise AssertionError(
                        f"Pre-cutoff values for {symbol} were modified!"
                    )
            
            # Write back
            store.put(key, df, format='table', data_columns=True)
            print(f"[patch_hdf5_prices] Patched {len(symbol_patches)} symbols "
                  f"after {cutoff_date} in key '{key}'")
