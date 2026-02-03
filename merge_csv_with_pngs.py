#!/usr/bin/env python3
"""
Merge CSV files and add PNG filenames.

This script merges abacus_best_performers_recreated.csv into
abacus_best_performers.csv, ensuring data format consistency and
adding PNG filenames based on timestamp and lookback period matching.
"""

import os
import csv
import re
from datetime import datetime
from typing import Dict, List, Optional


def parse_png_filename(filename: str) -> Optional[Dict]:
    """
    Parse PNG filename to extract date, time, and lookback periods.
    
    Format: monte_carlo_best_YYYY-MM-DD_HH-MM-SS_lbXX-YY-ZZ_score.png
    """
    pattern = r'monte_carlo_best_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})_lb(\d+)-(\d+)-(\d+)_.*\.png'
    match = re.match(pattern, filename)
    
    if not match:
        return None
    
    date_str, hour, minute, second, lb1, lb2, lb3 = match.groups()
    time_str = f"{hour}:{minute}:{second}"
    
    try:
        timestamp = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        lookbacks = sorted([int(lb1), int(lb2), int(lb3)])
        
        return {
            'filename': filename,
            'date': date_str,
            'time': time_str,
            'timestamp': timestamp,
            'lookbacks': lookbacks
        }
    except ValueError:
        return None


def scan_png_files(png_dir: str) -> List[Dict]:
    """Scan PNG directory and parse all filenames."""
    png_files = []
    
    for filename in os.listdir(png_dir):
        if filename.endswith('.png'):
            parsed = parse_png_filename(filename)
            if parsed:
                png_files.append(parsed)
    
    return png_files


def find_matching_png(csv_row: Dict, png_files: List[Dict]) -> Optional[str]:
    """
    Find PNG file matching CSV row by date, time, and lookback periods.
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


def clean_numeric_field(value: str) -> str:
    """
    Clean numeric fields by removing formatting characters.
    
    Examples:
        "$3,548,005,004" -> "3548005004"
        "44.11%" -> "44.11"
        "$7,557,054,686" -> "7557054686"
        "-52.31%" -> "-52.31"
    """
    if not value or not isinstance(value, str):
        return value
    
    # Remove dollar signs, commas, and percent signs
    cleaned = value.replace('$', '').replace(',', '').replace('%', '').strip()
    
    return cleaned


def normalize_row_format(row: Dict, fieldnames: List[str]) -> Dict:
    """
    Normalize row data format to match expected CSV format.
    Cleans numeric fields and ensures proper formatting.
    """
    # Fields that should have numeric formatting cleaned
    numeric_fields = {
        'Full Period Final Value',
        'Full Period Annual Return',
        'Full Period Sharpe Ratio',
        'Full Period Sortino Ratio',
        'Full Period Max Drawdown',
        'Full Period Avg Drawdown',
        'Full Period Normalized Score',
        'Focus Period 1 Final Value',
        'Focus Period 1 Annual Return',
        'Focus Period 1 Sharpe Ratio',
        'Focus Period 1 Sortino Ratio',
        'Focus Period 1 Max Drawdown',
        'Focus Period 1 Avg Drawdown',
        'Focus Period 1 Normalized Score',
        'Focus Period 2 Final Value',
        'Focus Period 2 Annual Return',
        'Focus Period 2 Sharpe Ratio',
        'Focus Period 2 Sortino Ratio',
        'Focus Period 2 Max Drawdown',
        'Focus Period 2 Avg Drawdown',
        'Focus Period 2 Normalized Score',
        'Blended Score',
        'Sharpe Outperformance Percentage',
        'Sortino Outperformance Percentage',
        'Central Annual Return',
        'Central Sharpe Ratio',
        'Central Sortino Ratio',
        'Central Max Drawdown',
        'Central Avg Drawdown',
        'Std Annual Return',
        'Std Sharpe Ratio',
        'Std Sortino Ratio',
        'Std Max Drawdown',
        'Std Avg Drawdown',
        'Focus Period 1 Weight',
        'Focus Period 2 Weight',
        'Focus Period 1 Duration Years',
        'Focus Period 2 Duration Years',
        'Focus Periods Overlap Percentage',
        'Average Rank'
    }
    
    normalized = {}
    for field in fieldnames:
        value = row.get(field, '')
        
        # Clean numeric fields
        if field in numeric_fields:
            value = clean_numeric_field(value)
        
        normalized[field] = value
    
    return normalized


def merge_csv_files(
    current_csv: str,
    recreated_csv: str,
    output_csv: str,
    png_dir: str
):
    """
    Merge CSV files, ensuring format consistency and adding PNG filenames.
    """
    print(f"Scanning PNG files in {png_dir}...")
    png_files = scan_png_files(png_dir)
    print(f"Found {len(png_files)} PNG files")
    
    # Read current CSV (may be corrupted/partial)
    print(f"\nReading current CSV: {current_csv}")
    current_rows = []
    current_fieldnames = None
    
    with open(current_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        current_fieldnames = reader.fieldnames
        for row in reader:
            current_rows.append(row)
    
    print(f"Current CSV has {len(current_rows)} rows")
    
    # Read recreated CSV
    print(f"\nReading recreated CSV: {recreated_csv}")
    recreated_rows = []
    recreated_fieldnames = None
    
    with open(recreated_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        recreated_fieldnames = reader.fieldnames
        for row in reader:
            recreated_rows.append(row)
    
    print(f"Recreated CSV has {len(recreated_rows)} rows")
    
    # Use current CSV fieldnames as the canonical format
    fieldnames = current_fieldnames
    
    # Create a set of existing entries (by date+time) to avoid duplicates
    existing_entries = set()
    merged_rows = []
    
    # Add current rows first
    print(f"\nAdding rows from current CSV...")
    for row in current_rows:
        key = (row.get('Date', ''), row.get('Time', ''))
        if key not in existing_entries and key != ('', ''):
            existing_entries.add(key)
            # Find PNG if missing
            if not row.get('PNG Filename') or not row['PNG Filename'].strip():
                png_path = find_matching_png(row, png_files)
                if png_path:
                    row['PNG Filename'] = png_path
            merged_rows.append(row)
    
    # Add recreated rows that don't exist in current
    print(f"Adding missing rows from recreated CSV...")
    added_count = 0
    
    for row in recreated_rows:
        key = (row.get('Date', ''), row.get('Time', ''))
        if key not in existing_entries and key != ('', ''):
            existing_entries.add(key)
            
            # Normalize and ensure all fieldnames exist
            normalized_row = normalize_row_format(row, fieldnames)
            
            # Find PNG filename
            png_path = find_matching_png(normalized_row, png_files)
            if png_path:
                normalized_row['PNG Filename'] = png_path
            
            merged_rows.append(normalized_row)
            added_count += 1
    
    print(f"Added {added_count} new rows from recreated CSV")
    
    # Sort by date and time
    print(f"\nSorting rows by date and time...")
    def get_sort_key(row):
        try:
            date_str = row.get('Date', '')
            time_str = row.get('Time', '')
            if date_str and time_str:
                return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        except:
            pass
        return datetime.min
    
    merged_rows.sort(key=get_sort_key)
    
    # Write merged CSV
    print(f"\nWriting merged CSV to: {output_csv}")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)
    
    # Calculate PNG matching statistics
    rows_with_png = sum(1 for row in merged_rows if row.get('PNG Filename', '').strip())
    
    print(f"\n{'='*60}")
    print("Merge Summary:")
    print(f"{'='*60}")
    print(f"Total merged rows: {len(merged_rows)}")
    print(f"Rows with PNG filenames: {rows_with_png}")
    print(f"Rows without PNG filenames: {len(merged_rows) - rows_with_png}")
    print(f"PNG files available: {len(png_files)}")
    print(f"{'='*60}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Merge CSV files and add PNG filenames'
    )
    parser.add_argument(
        '--current',
        default='abacus_best_performers.csv',
        help='Current CSV file (may be partial/corrupted)'
    )
    parser.add_argument(
        '--recreated',
        default='abacus_best_performers_recreated.csv',
        help='Recreated CSV file with complete data'
    )
    parser.add_argument(
        '--output',
        default='abacus_best_performers.csv',
        help='Output CSV file'
    )
    parser.add_argument(
        '--pngs',
        default='pngs',
        help='PNG directory'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    current_csv = os.path.abspath(args.current)
    recreated_csv = os.path.abspath(args.recreated)
    output_csv = os.path.abspath(args.output)
    png_dir = os.path.abspath(args.pngs)
    
    # Verify files exist
    if not os.path.exists(current_csv):
        print(f"Error: Current CSV not found: {current_csv}")
        return 1
    
    if not os.path.exists(recreated_csv):
        print(f"Error: Recreated CSV not found: {recreated_csv}")
        return 1
    
    if not os.path.isdir(png_dir):
        print(f"Error: PNG directory not found: {png_dir}")
        return 1
    
    # Perform merge
    merge_csv_files(current_csv, recreated_csv, output_csv, png_dir)
    
    return 0


if __name__ == '__main__':
    exit(main())
