"""
Backtest Results Analysis Script

This script reads Monte Carlo backtest results from an Excel file,
sorts them by various performance metrics, and copies the corresponding
PNG plot files for the top-performing parameter sets to a designated folder.

The analysis focuses on risk-adjusted metrics to identify robust trading strategies.
"""

import pandas as pd
import os
import shutil

# Paths
excel_path = '/Users/donaldpg/pyTAAA_data/sp500_pine/pytaaa_sp500_pine_montecarlo.xlsx'
png_dir = '/Users/donaldpg/pyTAAA_data/sp500_pine/'
dest_dir = '/Users/donaldpg/pyTAAA_data/sp500_pine/backtest_best_params'

# Read Excel, header at row 10 (0-indexed, row 11 is header)
df = pd.read_excel(excel_path, header=10)

print("Columns:", df.columns.tolist())
print("First few rows:")
print(df.head())

# Assume the data is sorted best to worst, take top 25
top_25 = df.head(25)

# Assume there's a column 'run_id' or similar, or use index
# From the PNG names, it's like 0198.png, so probably 4-digit zero-padded
# Perhaps the index is the ID, starting from 0

# For now, assume the index corresponds to the 4-digit number
# But looking at PNGs, they start from 0198, so perhaps not.

# Perhaps there's a column with the ID.

# To be safe, let's assume the row index (after header) is the run number, but adjusted.

# Perhaps the PNG names have the run ID in the filename.

# Let's extract the IDs from top 25 rows.

# But I need to know which column has the ID.

# Perhaps it's the first column or 'Run' or something.

# For now, let's assume the index is the run ID, and PNG is f"{run_id:04d}.png"

# But from the list, they have timestamps, so multiple runs.

# The PNGs have different timestamps: 20251213_1713, 20251213_1720, 20251214_1233, 20251214_1243

# So different runs.

# Perhaps the Excel has a column with the run ID.

# To proceed, I need to see the columns.

# Since I can't, let's assume columns like 'Sharpe', 'Sortino', etc.

# Let's define rankings: sort by Sharpe descending, Sortino descending, etc.

rankings = ['Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'Total Return']  # assume these columns exist

all_top = set()

for rank in rankings:
    if rank in df.columns:
        sorted_df = df.sort_values(by=rank, ascending=False if 'Drawdown' not in rank else True)  # ascending for drawdown
        top = sorted_df.head(25)
        # Assume 'Run ID' column exists
        if 'Run ID' in df.columns:
            ids = top['Run ID'].tolist()
        else:
            # Assume index is ID
            ids = top.index.tolist()
        all_top.update(ids)

# Now, for each ID, find the PNG file.

# PNG pattern: PyTAAA_monteCarloBacktest_run_{timestamp}_{id:04d}.png

# But multiple timestamps, so need to find which one.

# Perhaps the latest or all.

# To simplify, perhaps the Excel has the timestamp or something.

# This is getting complicated.

# Perhaps the row number corresponds to the ID, starting from 0 for the first data row.

# Let's assume the data rows are indexed from 0, and PNG is f"PyTAAA_monteCarloBacktest_run_*_{i:04d}.png"

# But multiple timestamps.

# Perhaps all PNGs are from the same run, but different timestamps.

# Looking at the list, there are different timestamps, but same pattern.

# Perhaps the Excel has a 'Run' column with the timestamp or ID.

# To make it work, let's assume the 'Run ID' is the 4-digit number, and we need to find PNGs with that number.

# For each top ID, copy the PNG if exists.

# But since multiple, perhaps copy all matching.

# For simplicity, let's take the top 25 as is, assume the sheet is sorted by some metric, and assume the 'Run ID' column exists.

# Let's modify the code to print columns, and assume.

# Since I can't run it yet, let's write the code assuming columns.

# Let's assume columns include 'Run ID', 'Sharpe Ratio', etc.

# And 'Run ID' is the 4-digit string.

# Then, for each ranking, sort and get top 25 IDs, collect unique IDs, then for each ID, find PNG files containing that ID, copy them.

# Yes.

# Code:

import pandas as pd
import os
import shutil
import glob

excel_path = '/Users/donaldpg/pyTAAA_data/sp500_pine/pytaaa_sp500_pine_montecarlo.xlsx'
png_dir = '/Users/donaldpg/pyTAAA_data/sp500_pine/'
dest_dir = '/Users/donaldpg/pyTAAA_data/sp500_pine/backtest_best_params'

df = pd.read_excel(excel_path, header=10)

print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print(df.head())

# Define rankings (columns to sort by)
# To get exactly 25 top performers, sort by primary metric: sum ranks ascending (lowest sum ranks is best)
primary_metric = 'sum ranks'
ascending = True  # lower sum ranks is better

if primary_metric in df.columns:
    sorted_df = df.sort_values(by=primary_metric, ascending=ascending)
    top_25 = sorted_df.head(25)
    # Create run_id as timestamp_trial
    all_top_ids = set()
    for _, row in top_25.iterrows():
        timestamp = row['run'].split('_', 1)[1]  # e.g., 20251214_1243
        trial = f"{int(row['trial']):04d}"
        run_id = f"{timestamp}_{trial}"
        all_top_ids.add(run_id)
else:
    print(f"Primary metric '{primary_metric}' not found in columns.")
    all_top_ids = set()

print(f"Total top IDs: {len(all_top_ids)}")

# Now, copy PNGs
# Clear the destination folder first
if os.path.exists(dest_dir):
    for file in os.listdir(dest_dir):
        file_path = os.path.join(dest_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

for run_id in all_top_ids:
    png_name = f"PyTAAA_monteCarloBacktest_run_{run_id}.png"
    src_path = os.path.join(png_dir, png_name)
    if os.path.exists(src_path):
        shutil.copy(src_path, dest_dir)
        print(f"Copied {png_name} to {dest_dir}")
    else:
        print(f"PNG not found: {png_name}")

print("Done.")