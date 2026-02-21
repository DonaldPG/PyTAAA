#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_pytaaa_main_extract.sh [json_path] [timeout_sec]
# Example: ./run_pytaaa_main_extract.sh pytaaa_data/sp500_pine/pytaaa_sp500_pine.json 180

LOG_DIR=logs
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pytaaa_main_run.log"
EXTRACT_FILE="$LOG_DIR/pytaaa_main_rollingfilter.txt"
PID_FILE="$LOG_DIR/pytaaa_main_pid.txt"

JSON_PATH=${1:-pytaaa_data/sp500_pine/pytaaa_sp500_pine.json}
TIMEOUT_SEC=${2:-120}

echo "Running pytaaa_main.py for ${TIMEOUT_SEC}s, json=${JSON_PATH}"

# Prefer using the project's .venv if available
if [ -x ".venv/bin/uv" ]; then
	echo "Using .venv/bin/uv"
	CMD=(".venv/bin/uv" "run" "python" "pytaaa_main.py" "--json" "$JSON_PATH")
elif [ -x ".venv/bin/python" ]; then
	echo "Using .venv/bin/python"
	CMD=(".venv/bin/python" "pytaaa_main.py" "--json" "$JSON_PATH")
else
	echo "Using system 'uv' command"
	CMD=("uv" "run" "python" "pytaaa_main.py" "--json" "$JSON_PATH")
fi

# Use timeout to keep runs bounded; capture output to log file
# We allow the command to fail/exit non-zero and continue (|| true)
timeout "$TIMEOUT_SEC" "${CMD[@]}" > "$LOG_FILE" 2>&1 || true
uv run python pytaaa_main.py --json "$JSON_PATH" > "$LOG_FILE" 2>&1 || true

# Extract interesting lines related to the rolling-window filter
grep -E "RollingFilter|Zeroing|rolling_window_filter" "$LOG_FILE" > "$EXTRACT_FILE" || true

echo "Saved run log: $LOG_FILE"
echo "Saved rolling-filter extract: $EXTRACT_FILE"

# Print a short summary of counts
echo "Total run lines:" $(wc -l < "$LOG_FILE")
echo "Extracted lines:" $(wc -l < "$EXTRACT_FILE")

echo "Done."
