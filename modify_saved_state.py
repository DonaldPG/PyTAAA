#!/usr/bin/env python3

"""Utility script to inspect and modify Monte Carlo saved state.

This script allows you to:
1. Inspect the current saved state and see all canonical combinations
2. Remove specific lookback combinations from the saved state
3. Clean up combinations containing specific lookback values
4. Reset the entire state if needed
"""

import pickle
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
import click


def load_state(filename: str = "monte_carlo_state.pkl") -> Dict[str, Any]:
    """Load the Monte Carlo state from pickle file.
    
    Args:
        filename: Path to the state file
        
    Returns:
        Dictionary containing the state data
    """
    if not os.path.exists(filename):
        print(f"State file {filename} does not exist.")
        return {}
    
    try:
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        print(f"Successfully loaded state from {filename}")
        return state
    except Exception as e:
        print(f"Error loading state: {str(e)}")
        return {}


def save_state(state: Dict[str, Any], filename: str = "monte_carlo_state.pkl", 
               backup: bool = True) -> None:
    """Save the Monte Carlo state to pickle file.
    
    Args:
        state: State dictionary to save
        filename: Path to save the state file
        backup: Whether to create a backup of the existing file
    """
    if backup and os.path.exists(filename):
        backup_name = f"{filename}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.rename(filename, backup_name)
        print(f"Created backup: {backup_name}")
    
    try:
        # Update timestamp
        state['timestamp'] = datetime.now().isoformat()
        
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        print(f"Successfully saved modified state to {filename}")
    except Exception as e:
        print(f"Error saving state: {str(e)}")


def inspect_state(state: Dict[str, Any]) -> None:
    """Inspect and display the current state contents.
    
    Args:
        state: State dictionary to inspect
    """
    if not state:
        print("No state data to inspect.")
        return
    
    print("\n" + "="*60)
    print("MONTE CARLO STATE INSPECTION")
    print("="*60)
    
    # Basic info
    print(f"Timestamp: {state.get('timestamp', 'Unknown')}")
    print(f"Min lookback: {state.get('min_lookback', 'Unknown')}")
    print(f"Max lookback: {state.get('max_lookback', 'Unknown')}")
    print(f"Number of lookbacks: {state.get('n_lookbacks', 'Unknown')}")
    
    # Combination data
    combination_indices = state.get('combination_indices', {})
    performance_scores = state.get('canonical_performance_scores', [])
    visit_counts = state.get('canonical_visit_counts', [])
    
    print(f"\nTotal canonical combinations: {len(combination_indices)}")
    print(f"Total visits: {sum(visit_counts) if visit_counts else 0}")
    
    if combination_indices and performance_scores and visit_counts:
        print("\nTop 10 performing combinations:")
        print("-" * 60)
        print("Rank | Lookbacks              | Score  | Visits")
        print("-" * 60)
        
        # Sort by performance score
        sorted_combos = []
        for canonical, idx in combination_indices.items():
            if idx < len(performance_scores) and idx < len(visit_counts):
                sorted_combos.append((
                    canonical,
                    performance_scores[idx],
                    visit_counts[idx]
                ))
        
        sorted_combos.sort(key=lambda x: x[1], reverse=True)
        
        for i, (canonical, score, visits) in enumerate(sorted_combos[:10], 1):
            lookbacks_str = str(list(canonical)).ljust(20)
            print(f"{i:4d} | {lookbacks_str} | {score:6.4f} | {visits:6d}")
    
    print("\n" + "="*60)


def remove_combinations_with_lookback(state: Dict[str, Any], 
                                    lookback_value: int) -> Dict[str, Any]:
    """Remove all combinations containing a specific lookback value.
    
    Args:
        state: State dictionary to modify
        lookback_value: Lookback value to remove from all combinations
        
    Returns:
        Modified state dictionary
    """
    if not state:
        print("No state data to modify.")
        return state
    
    combination_indices = state.get('combination_indices', {})
    performance_scores = state.get('canonical_performance_scores', [])
    visit_counts = state.get('canonical_visit_counts', [])
    
    if not combination_indices:
        print("No combinations found in state.")
        return state
    
    # Find combinations to remove
    combinations_to_remove = []
    for canonical in combination_indices.keys():
        if lookback_value in canonical:
            combinations_to_remove.append(canonical)
    
    print(f"Found {len(combinations_to_remove)} combinations containing lookback {lookback_value}")
    
    if not combinations_to_remove:
        print("No combinations found with that lookback value.")
        return state
    
    # Remove combinations and rebuild indices
    new_combination_indices = {}
    new_performance_scores = []
    new_visit_counts = []
    new_index = 0
    
    for canonical, old_index in combination_indices.items():
        if canonical not in combinations_to_remove:
            new_combination_indices[canonical] = new_index
            if old_index < len(performance_scores):
                new_performance_scores.append(performance_scores[old_index])
            if old_index < len(visit_counts):
                new_visit_counts.append(visit_counts[old_index])
            new_index += 1
    
    # Update state
    state['combination_indices'] = new_combination_indices
    state['canonical_performance_scores'] = new_performance_scores
    state['canonical_visit_counts'] = new_visit_counts
    
    print(f"Removed {len(combinations_to_remove)} combinations")
    print(f"Remaining combinations: {len(new_combination_indices)}")
    
    return state


def remove_specific_combination(state: Dict[str, Any], 
                               lookbacks: List[int]) -> Dict[str, Any]:
    """Remove a specific lookback combination from the state.
    
    Args:
        state: State dictionary to modify
        lookbacks: List of lookback values to remove (will be converted to canonical form)
        
    Returns:
        Modified state dictionary
    """
    if not state:
        print("No state data to modify.")
        return state
    
    # Convert to canonical form (sorted tuple)
    canonical = tuple(sorted(lookbacks))
    
    combination_indices = state.get('combination_indices', {})
    performance_scores = state.get('canonical_performance_scores', [])
    visit_counts = state.get('canonical_visit_counts', [])
    
    if canonical not in combination_indices:
        print(f"Combination {list(canonical)} not found in state.")
        return state
    
    # Get the index to remove
    index_to_remove = combination_indices[canonical]
    
    # Remove from combination_indices
    del combination_indices[canonical]
    
    # Remove from performance_scores and visit_counts
    if index_to_remove < len(performance_scores):
        performance_scores.pop(index_to_remove)
    if index_to_remove < len(visit_counts):
        visit_counts.pop(index_to_remove)
    
    # Rebuild indices for remaining combinations
    new_combination_indices = {}
    for canonical_key, old_index in combination_indices.items():
        if old_index > index_to_remove:
            new_combination_indices[canonical_key] = old_index - 1
        else:
            new_combination_indices[canonical_key] = old_index
    
    # Update state
    state['combination_indices'] = new_combination_indices
    state['canonical_performance_scores'] = performance_scores
    state['canonical_visit_counts'] = visit_counts
    
    print(f"Removed combination {list(canonical)}")
    print(f"Remaining combinations: {len(new_combination_indices)}")
    
    return state


@click.group()
def cli():
    """Monte Carlo state modification utility."""
    pass


@cli.command()
@click.option('--file', '-f', default='monte_carlo_state.pkl',
              help='State file to inspect')
def inspect(file: str):
    """Inspect the current saved state."""
    state = load_state(file)
    inspect_state(state)


@cli.command()
@click.argument('lookback_value', type=int)
@click.option('--file', '-f', default='monte_carlo_state.pkl',
              help='State file to modify')
@click.option('--no-backup', is_flag=True,
              help='Skip creating backup file')
def remove_lookback(lookback_value: int, file: str, no_backup: bool):
    """Remove all combinations containing a specific lookback value."""
    state = load_state(file)
    if not state:
        return
    
    print(f"\nBefore modification:")
    inspect_state(state)
    
    modified_state = remove_combinations_with_lookback(state, lookback_value)
    
    if click.confirm(f"\nProceed with removing all combinations containing lookback {lookback_value}?"):
        save_state(modified_state, file, backup=not no_backup)
        print(f"\nAfter modification:")
        inspect_state(modified_state)
    else:
        print("Operation cancelled.")


@cli.command()
@click.argument('lookbacks', nargs=-1, type=int, required=True)
@click.option('--file', '-f', default='monte_carlo_state.pkl',
              help='State file to modify')
@click.option('--no-backup', is_flag=True,
              help='Skip creating backup file')
def remove_combination(lookbacks: Tuple[int, ...], file: str, no_backup: bool):
    """Remove a specific lookback combination.
    
    Example: remove-combination 50 150 250
    """
    state = load_state(file)
    if not state:
        return
    
    lookback_list = list(lookbacks)
    canonical = tuple(sorted(lookback_list))
    
    print(f"\nRemoving combination: {lookback_list} (canonical: {list(canonical)})")
    print(f"Before modification:")
    inspect_state(state)
    
    modified_state = remove_specific_combination(state, lookback_list)
    
    if click.confirm(f"\nProceed with removing combination {lookback_list}?"):
        save_state(modified_state, file, backup=not no_backup)
        print(f"\nAfter modification:")
        inspect_state(modified_state)
    else:
        print("Operation cancelled.")


@cli.command()
@click.option('--file', '-f', default='monte_carlo_state.pkl',
              help='State file to reset')
@click.option('--no-backup', is_flag=True,
              help='Skip creating backup file')
@click.option('--no-confirm', is_flag=True,
              help='Skip confirmation prompt')
def reset(file: str, no_backup: bool, no_confirm: bool):
    """Reset the entire saved state (removes all combinations)."""
    if not os.path.exists(file):
        print(f"State file {file} does not exist.")
        return
    
    # Skip confirmation if --no-confirm flag is provided
    if no_confirm or click.confirm(f"This will completely reset {file}. Are you sure?"):
        if not no_backup:
            backup_name = f"{file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(file, backup_name)
            print(f"Created backup: {backup_name}")
        else:
            os.remove(file)
        print(f"Reset complete. {file} has been removed.")
    else:
        print("Operation cancelled.")


if __name__ == "__main__":
    cli()