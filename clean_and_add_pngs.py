#!/usr/bin/env python3
"""
Clean CSV formatting and add PNG filenames.

Takes a CSV file, cleans numeric formatting, and adds PNG filenames.
"""

import os
import csv
import re
from datetime import datetime
from typing import Dict, List, Optional


def clean_numeric_field(value: str) -> str:
    """
    Clean numeric fields by removing formatting characters.
    """
    if not value or not isinstance(value, str):
        return value
    
    # Remove dollar signs, commas, and percent signs
    cleaned = value.replace('$', '').replace(',', '').replace('%', '').strip()
    
    return cleaned


def parse_png_filename(filename: str) -> Optional[Dict]:
    """
    Parse PNG filename to extract date, time, and lookback periods.
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


def process_csv(input_csv: str, output_csv: str, png_dir: str):
    """
    Process CSV file: clean numeric formatting and add PNG filenames.
    """
    print(f"Scanning PNG files in {png_dir}...")
    png_files = scan_png_files(png_dir)
    print(f"Found {len(png_files)} PNG files\n")
    
    # Fields that should have numeric formatting cleaned
    numeric_fields = {
        'Full Period Final Value', 'Full Period Annual Return',
        'Full Period Sharpe Ratio', 'Full Period Sortino Ratio',
        'Full Period Max Drawdown', 'Full Period Avg Drawdown',
        'Full Period Normalized Score', 'Focus Period 1 Final Value',
        'Focus Period 1 Annual Return', 'Focus Period 1 Sharpe Ratio',
        'Focus Period 1 Sortino Ratio', 'Focus Period 1 Max Drawdown',
        'Focus Period 1 Avg Drawdown', 'Focus Period 1 Normalized Score',
        'Focus Period 2 Final Value', 'Focus Period 2 Annual Return',
        'Focus Period 2 Sharpe Ratio', 'Focus Period 2 Sortino Ratio',
        'Focus Period 2 Max Drawdown', 'Focus Period 2 Avg Drawdown',
        'Focus Period 2 Normalized Score', 'Blended Score',
        'Sharpe Outperformance Percentage', 'Sortino Outperformance Percentage',
        'Central Annual Return', 'Central Sharpe Ratio',
        'Central Sortino Ratio', 'Central Max Drawdown',
        'Central Avg Drawdown', 'Std Annual Return',
        'Std Sharpe Ratio', 'Std Sortino Ratio',
        'Std Max Drawdown', 'Std Avg Drawdown',
        'Focus Period 1 Weight', 'Focus Period 2 Weight',
        'Focus Period 1 Duration Years', 'Focus Period 2 Duration Years',
        'Focus Periods Overlap Percentage', 'Average Rank'
    }
    
    # Read input CSV
    print(f"Reading CSV: {input_csv}")
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    print(f"Read {len(rows)} rows")
    
    # Process rows
    print("\nProcessing rows...")
    matched_count = 0
    
    for row in rows:
        # Clean numeric fields
        for field in fieldnames:
            if field in numeric_fields and field in row:
                row[field] = clean_numeric_field(row[field])
        
        # Find PNG filename if not already set
        if not row.get('PNG Filename') or not row['PNG Filename'].strip():
            png_path = find_matching_png(row, png_files)
            if png_path:
                row['PNG Filename'] = png_path
                matched_count += 1
    
    # Write output CSV
    print(f"\nWriting cleaned CSV to: {output_csv}")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n{'='*60}")
    print("Processing Summary:")
    print(f"{'='*60}")
    print(f"Total rows: {len(rows)}")
    print(f"PNG files matched: {matched_count}")
    print(f"Rows without PNG: {len(rows) - matched_count}")
    print(f"{'='*60}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Clean CSV formatting and add PNG filenames'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input CSV file'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output CSV file'
    )
    parser.add_argument(
        '--pngs',
        default='pngs',
        help='PNG directory'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    input_csv = os.path.abspath(args.input)
    output_csv = os.path.abspath(args.output)
    png_dir = os.path.abspath(args.pngs)
    
    # Verify files exist
    if not os.path.exists(input_csv):
        print(f"Error: Input CSV not found: {input_csv}")
        return 1
    
    if not os.path.isdir(png_dir):
        print(f"Error: PNG directory not found: {png_dir}")
        return 1
    
    # Process CSV
    process_csv(input_csv, output_csv, png_dir)
    
    return 0


if __name__ == '__main__':
    exit(main())
