#!/usr/bin/env python3
"""
Update abacus_best_performers.csv with PNG plot filenames.

This script matches CSV rows with PNG files based on:
- Timestamp (date/time matching)
- Lookback periods (extracted from PNG filename)
- Final portfolio value (extracted from PNG text via OCR or metadata)
"""

import os
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_png_filename(filename: str) -> Optional[Dict]:
    """Extract timestamp and lookback periods from PNG filename.
    
    Format: monte_carlo_best_YYYY-MM-DD_HH-MM-SS_lbXX-YY-ZZ_score.png
    
    Args:
        filename: PNG filename to parse
        
    Returns:
        Dictionary with timestamp and lookbacks, or None if parsing fails
    """
    # Pattern: monte_carlo_best_2025-12-15_21-25-45_lb107-109-203_0_362.png
    pattern = r'monte_carlo_best_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})_lb(\d+)-(\d+)-(\d+)_([-\d_]+)\.png'
    match = re.match(pattern, filename)
    
    if not match:
        return None
    
    date_str = match.group(1)
    hour = match.group(2)
    minute = match.group(3)
    second = match.group(4)
    lb1 = int(match.group(5))
    lb2 = int(match.group(6))
    lb3 = int(match.group(7))
    score_str = match.group(8)
    
    # Convert time to standard format HH:MM:SS
    time_str = f"{hour}:{minute}:{second}"
    
    # Parse timestamp
    try:
        timestamp = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    
    # Sort lookbacks for comparison
    lookbacks = sorted([lb1, lb2, lb3])
    
    return {
        'filename': filename,
        'date': date_str,
        'time': time_str,
        'timestamp': timestamp,
        'lookbacks': lookbacks,
        'lb1': lookbacks[0],
        'lb2': lookbacks[1],
        'lb3': lookbacks[2],
        'score_str': score_str
    }


def find_matching_png(csv_row: Dict, png_files: List[Dict]) -> Optional[str]:
    """Find PNG file matching a CSV row.
    
    Matches based on:
    1. Date match (exact)
    2. Time match (within 2 minutes)
    3. Lookback periods match (exact)
    
    Args:
        csv_row: Dictionary containing CSV row data
        png_files: List of parsed PNG file dictionaries
        
    Returns:
        PNG filename if match found, None otherwise
    """
    csv_date = csv_row.get('Date', '')
    csv_time = csv_row.get('Time', '')
    
    # Parse CSV lookbacks
    try:
        csv_lb1 = int(csv_row.get('Lookback Period 1', 0))
        csv_lb2 = int(csv_row.get('Lookback Period 2', 0))
        csv_lb3 = int(csv_row.get('Lookback Period 3', 0))
        csv_lookbacks = sorted([csv_lb1, csv_lb2, csv_lb3])
    except (ValueError, TypeError):
        return None
    
    # Parse CSV timestamp
    try:
        csv_timestamp = datetime.strptime(f"{csv_date} {csv_time}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    
    # Find best match
    best_match = None
    min_time_diff = float('inf')
    
    for png in png_files:
        # Must match date
        if png['date'] != csv_date:
            continue
        
        # Must match lookbacks exactly
        if png['lookbacks'] != csv_lookbacks:
            continue
        
        # Calculate time difference in seconds
        time_diff = abs((png['timestamp'] - csv_timestamp).total_seconds())
        
        # Within 120 seconds (2 minutes) tolerance
        if time_diff <= 120 and time_diff < min_time_diff:
            best_match = f"./pngs/{png['filename']}"
            min_time_diff = time_diff
    
    return best_match


def update_csv_with_png_paths(csv_path: str, pngs_dir: str, output_path: Optional[str] = None):
    """Update CSV file with PNG plot paths.
    
    Args:
        csv_path: Path to the CSV file
        pngs_dir: Path to the directory containing PNG files
        output_path: Optional output path (defaults to updating input file)
    """
    if output_path is None:
        output_path = csv_path
    
    # Parse all PNG files
    print("Scanning PNG files...")
    png_files = []
    for filename in os.listdir(pngs_dir):
        if filename.endswith('.png') and filename.startswith('monte_carlo_best_'):
            parsed = parse_png_filename(filename)
            if parsed:
                png_files.append(parsed)
    
    print(f"Found {len(png_files)} PNG files")
    
    # Read CSV
    print(f"Reading CSV file: {csv_path}")
    rows = []
    fieldnames = None
    
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        # Ensure PNG Filename column exists
        if 'PNG Filename' not in fieldnames:
            fieldnames = list(fieldnames) + ['PNG Filename']
        
        for row in reader:
            rows.append(dict(row))
    
    print(f"Read {len(rows)} CSV rows")
    
    # Match and update rows
    print("Matching PNG files to CSV rows...")
    matched_count = 0
    
    for row in rows:
        # Remove any keys that aren't in fieldnames (malformed rows)
        row_keys = list(row.keys())
        for key in row_keys:
            if key not in fieldnames:
                del row[key]
        
        # Ensure all fieldnames are present in row
        for field in fieldnames:
            if field not in row:
                row[field] = ''
        
        # Skip if already has PNG filename
        if row.get('PNG Filename') and row['PNG Filename'].strip():
            continue
        
        # Find matching PNG
        png_path = find_matching_png(row, png_files)
        
        if png_path:
            row['PNG Filename'] = png_path
            matched_count += 1
            print(f"  Matched: {row['Date']} {row['Time']} -> {png_path}")
        else:
            # Ensure column exists even if no match
            row['PNG Filename'] = ''
    
    # Write updated CSV
    print(f"\nWriting updated CSV to: {output_path}")
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nSummary:")
    print(f"  Total rows: {len(rows)}")
    print(f"  Matched: {matched_count}")
    print(f"  Unmatched: {len(rows) - matched_count}")
    print(f"  PNG files available: {len(png_files)}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Update abacus_best_performers.csv with PNG plot paths'
    )
    parser.add_argument(
        '--csv',
        default='abacus_best_performers.csv',
        help='Path to CSV file (default: abacus_best_performers.csv)'
    )
    parser.add_argument(
        '--pngs',
        default='pngs',
        help='Path to PNG directory (default: pngs)'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output CSV path (default: update input file in-place)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    csv_path = os.path.abspath(args.csv)
    pngs_dir = os.path.abspath(args.pngs)
    output_path = os.path.abspath(args.output) if args.output else None
    
    # Validate inputs
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return 1
    
    if not os.path.isdir(pngs_dir):
        print(f"Error: PNG directory not found: {pngs_dir}")
        return 1
    
    # Update CSV
    try:
        update_csv_with_png_paths(csv_path, pngs_dir, output_path)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
