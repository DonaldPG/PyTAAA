"""Script to remove duplicate function definitions from TAfunctions.py.

This script removes functions that have been extracted to functions/ta/ modules
and adds re-export statements instead.
"""

# Define the re-export block to add after imports
RE_EXPORT_BLOCK = '''
#############################################################################
# NOTE: Phase 5+ Refactoring - Functions re-exported from functions/ta/*
# 
# Common technical analysis functions have been extracted to modular subpackages
# and are re-exported here for backward compatibility.
#
# - functions/ta/utils.py: strip_accents, normcorrcoef, nanrms
# - functions/ta/data_cleaning.py: interpolate, cleantobeginning, cleantoend, etc.
# - functions/ta/moving_averages.py: SMA, SMA_2D, hma, MoveMax, MoveMin, etc.
# - functions/ta/channels.py: percentileChannel, dpgchannel (both 1D and 2D)
# - functions/ta/signal_generation.py: computeSignal2D
# - functions/ta/rolling_metrics.py: move_sharpe_2D, move_martin_2D, etc.
#
# New code should import directly from functions.ta.* modules for clarity.
#############################################################################

# Re-export functions from modular ta/ subpackage
from functions.ta.utils import (
    strip_accents,
    normcorrcoef,
    nanrms
)

from functions.ta.data_cleaning import (
    interpolate,
    cleantobeginning,
    cleantoend,
    clean_signal,
    cleanspikes,
    despike_2D
)

from functions.ta.moving_averages import (
    SMA,
    SMA_2D,
    SMS,
    hma,
    hma_pd,
    SMA_filtered_2D,
    MoveMax,
    MoveMax_2D,
    MoveMin
)

from functions.ta.channels import (
    percentileChannel,
    percentileChannel_2D,
    dpgchannel,
    dpgchannel_2D
)

from functions.ta.signal_generation import (
    computeSignal2D
)

from functions.ta.rolling_metrics import (
    move_sharpe_2D,
    move_martin_2D,
    move_informationRatio
)

#############################################################################
'''

# Functions to remove (these are duplicates now in functions/ta/)
FUNCTIONS_TO_REMOVE = [
    'strip_accents',
    'normcorrcoef',
    'nanrms',
    'interpolate',
    'cleantobeginning',
    'cleantoend',
    'clean_signal',
    'cleanspikes',
    'despike_2D',
    'SMA',
    'SMA_2D',
    'SMS',
    'hma',
    'hma_pd',
    'SMA_filtered_2D',
    'MoveMax',
    'MoveMax_2D',
    'MoveMin',
    'percentileChannel',
    'percentileChannel_2D',
    'dpgchannel',
    'dpgchannel_2D',
    'computeSignal2D',
    'move_sharpe_2D',
    'move_martin_2D',
    'move_informationRatio',
]

def find_function_ranges(lines):
    """Find start and end line numbers for each function definition.
    
    Returns:
        list: List of (function_name, start_line, end_line) tuples
    """
    func_list = []
    current_func = None
    func_start = None
    indent_level = None
    
    for i, line in enumerate(lines):
        # Check if this is a function definition
        if line.startswith('def '):
            # If we were tracking a previous function, save its end
            if current_func is not None:
                func_list.append((current_func, func_start, i - 1))
            
            # Start tracking new function
            func_name = line.split('(')[0].replace('def ', '').strip()
            current_func = func_name
            func_start = i
            indent_level = 0
        
        # Check if we're at top level again (new function or class)
        elif current_func and line and not line[0].isspace() and line.strip():
            if line.startswith('def ') or line.startswith('class ') or line.startswith('#---'):
                # Previous function ended
                func_list.append((current_func, func_start, i - 1))
                current_func = None
                func_start = None
    
    # Handle last function
    if current_func is not None:
        func_list.append((current_func, func_start, len(lines) - 1))
    
    return func_list

def main():
    """Main function to remove duplicates."""
    input_file = 'functions/TAfunctions.py'
    output_file = 'functions/TAfunctions.py.new'
    
    print(f"Reading {input_file}...")
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    print(f"Original file: {len(lines)} lines")
    
   # Find line where to insert re-exports (after matplotlib.use() call)
    insert_line = None
    for i, line in enumerate(lines):
        if "matplotlib.use('Agg')" in line:
            # Find the next blank line after this
            for j in range(i + 1, len(lines)):
                if lines[j].strip() == '':
                    insert_line = j + 1
                    break
            break
    
    if insert_line is None:
        print("ERROR: Could not find insertion point for re-exports")
        return 1
    
    print(f"Will insert re-exports at line {insert_line}")
    
    # Find all function definitions
    print("Finding function definitions...")
    func_list = find_function_ranges(lines)
    
    print(f"Found {len(func_list)} function definitions")
    
    # Identify duplicates to remove (may be multiple occurrences of same function)
    duplicates_found = []
    for func_name, start, end in func_list:
        if func_name in FUNCTIONS_TO_REMOVE:
            duplicates_found.append((func_name, start, end))
            print(f"  - {func_name}: lines {start+1} to {end+1}")
    
    # Sort by line number for processing in order
    duplicates_found.sort(key=lambda x: x[1])
    
    print(f"\nFound {len(duplicates_found)} duplicate function occurrences to remove")
    
    # Build new file content
    new_lines = []
    skip_until = -1
    
    for i, line in enumerate(lines):
        # If we're past the skip range and at insertion point, add re-exports
        if i == insert_line and skip_until < i:
            new_lines.append(RE_EXPORT_BLOCK)
       
        # Check if this line starts a function to remove
        should_skip = False
        for func_name, start, end in duplicates_found:
            if i == start:
                print(f"Removing {func_name} (lines {start+1}-{end+1})")
                skip_until = end
                should_skip = True
                break
        
        # Add line if we're not skipping
        if not should_skip and i > skip_until:
            new_lines.append(line)
    
    print(f"\nNew file: {len(new_lines)} lines")
    print(f"Removed {len(lines) - len(new_lines)} lines")
    
    # Write output
    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w') as f:
        f.writelines(new_lines)
    
    print(f"Done! Review {output_file} and then rename to {input_file}")
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
