#!/usr/bin/env python3
"""Wrap module-level execution code in if __name__ == '__main__' guard."""

# Read the file
with open('PyTAAA_backtest_sp500_pine_refactored.py', 'r') as f:
    lines = f.readlines()

# Find line with "# Determine number of Monte Carlo trials"
# This is around line 1410 (after our guard comment)
output_lines = []
found_marker = False
indent_depth = 0

for i, line in enumerate(lines, 1):
    if not found_marker:
        output_lines.append(line)
        # Look for the marker line
        if '# Determine number of Monte Carlo trials' in line and i > 1400:
            found_marker = True
            # Add the if __name__ == '__main__': guard before this line
            output_lines.pop()  # Remove the marker line temporarily
            output_lines.append("if __name__ == '__main__':\n")
            output_lines.append("    # Determine number of Monte Carlo trials. Allow CLI override with --trials/-t.\n")
            indent_depth = 4
    else:
        # Indent all remaining lines
        if line.strip():  # Non-empty line
            output_lines.append(' ' * indent_depth + line)
        else:  # Empty line
            output_lines.append(line)

# Write back
with open('PyTAAA_backtest_sp500_pine_refactored.py', 'w') as f:
    f.writelines(output_lines)
    
print(f"Wrapped {len(lines) - len([l for l in lines[:1410] if l])} lines in if __name__ == '__main__' guard")
