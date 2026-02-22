#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=$(pwd)

CONFIG="studies/nasdaq100_scenarios/params/minimal_scenario.json"
RESULTS_DIR="studies/nasdaq100_scenarios/results"
PLOTS_DIR="studies/nasdaq100_scenarios/plots"
STUDY_NAME="nasdaq100_oracle_delay_minimal"

HDF5_FILE=$(uv run python - <<'PY'
import json
import os

config_path = "studies/nasdaq100_scenarios/params/minimal_scenario.json"
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

symbols_file = config.get("data_selection", {}).get("symbols_file")
if not symbols_file:
    symbols_file = "symbols/Naz100_Symbols.txt"

folder, filename = os.path.split(symbols_file)
shortname, _ = os.path.splitext(filename)
hdf5_file = os.path.join(folder, f"{shortname}_.hdf5")
print(hdf5_file)
PY
)

if [[ ! -f "${HDF5_FILE}" ]]; then
  echo "Skipping integration test: missing ${HDF5_FILE}"
  exit 0
fi

uv run python studies/nasdaq100_scenarios/run_full_study.py \
  --config "${CONFIG}"

SUMMARY_JSON="${RESULTS_DIR}/summary_${STUDY_NAME}.json"
METRICS_JSON="${RESULTS_DIR}/metrics_${STUDY_NAME}.json"
PLOT_PNG="${PLOTS_DIR}/portfolio_histories_${STUDY_NAME}.png"

if [[ ! -s "${SUMMARY_JSON}" ]]; then
  echo "Missing or empty summary JSON: ${SUMMARY_JSON}"
  exit 1
fi

if [[ ! -s "${METRICS_JSON}" ]]; then
  echo "Missing or empty metrics JSON: ${METRICS_JSON}"
  exit 1
fi

if [[ ! -s "${PLOT_PNG}" ]]; then
  echo "Missing or empty plot: ${PLOT_PNG}"
  exit 1
fi

echo "Integration test passed."
