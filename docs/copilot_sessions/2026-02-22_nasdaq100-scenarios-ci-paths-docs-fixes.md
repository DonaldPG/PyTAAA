# Date and Context
- Date: 2026-02-22
- Context: Review-driven cleanup of the NASDAQ100 oracle delay study code and docs before merge, focused on CI portability and documentation correctness.

# Problem Statement
Several issues were identified in study assets:
- Hardcoded absolute paths in scenario JSON files.
- Integration shell test relying on path assumptions for `symbols_file`.
- README command using `python` to run a shell script.
- Plot documentation listing filenames that no longer match generated outputs.
- Potential headless CI plotting failures due to backend setup.
- Data loader config file opened without explicit UTF-8 encoding.
- Transaction-cost timing in portfolio simulation mutating prior-day values.
- Test `print` statements adding unnecessary output noise.
- Runtime failure when using relative `symbols_file` in study config if local
  workspace lacked `symbols/` data, causing fallback code to call production
  JSON parsing (`get_json_params`) and crash on missing `Email` fields.

# Solution Overview
Applied focused fixes in study-specific files to improve CI reliability, path portability, and doc accuracy, while preserving current study behavior and interfaces.

# Key Changes
- `studies/nasdaq100_scenarios/params/default_scenario.json`
  - Replaced absolute `symbols_file` path with relative `symbols/Naz100_Symbols.txt`.
- `studies/nasdaq100_scenarios/params/minimal_scenario.json`
  - Replaced absolute `symbols_file` path with relative `symbols/Naz100_Symbols.txt`.
- `studies/nasdaq100_scenarios/test_integration.sh`
  - Anchored execution to repository root via script-relative path resolution.
  - Added `SYMBOLS_FILE` environment-variable override.
  - Resolved relative `symbols_file` robustly before deriving HDF5 path.
- `studies/nasdaq100_scenarios/README.md`
  - Corrected integration test command from `uv run python ...test_integration.sh` to `bash ...test_integration.sh`.
- `docs/pytaaa-oracle-delay-studies.md`
  - Updated example plot filenames to match actual generated naming patterns with `<study_name>`, `<window>`, and `<top_n>` components.
- `studies/nasdaq100_scenarios/plotting.py`
  - Set `matplotlib` backend to `Agg` before importing `pyplot`.
- `studies/nasdaq100_scenarios/data_loader.py`
  - Added `encoding="utf-8"` when reading config JSON.
  - Added robust symbols/HDF5 path resolution that supports env overrides
    (`PYTAAA_SYMBOLS_FILE` / `SYMBOLS_FILE`), config-relative paths, and known
    local data roots, preventing accidental fallback into production JSON
    creation paths.
- `studies/nasdaq100_scenarios/portfolio_backtest.py`
  - Adjusted transaction-cost application to affect current rebalance-day valuation path rather than retroactively mutating `portfolio_value[j-1]`.
  - Updated date-related type hints to accept `datetime.date` and `numpy.datetime64` usage patterns present in tests.
  - Corrected strategy docstring text to reflect actual behavior (remaining weight is cash when fewer than `top_n` names are selected).
- `studies/nasdaq100_scenarios/oracle_signals.py`
  - Renamed unused `apply_delay` parameter `datearray` to `_datearray` for maintainability clarity.
- `studies/nasdaq100_scenarios/tests/test_data_loader.py`
  - Removed non-essential integration-test print statements.
- `studies/nasdaq100_scenarios/tests/test_portfolio_backtest.py`
  - Removed non-essential integration-test print statements.

# Technical Details
- Integration script now computes:
  - `SCRIPT_DIR` from `${BASH_SOURCE[0]}`
  - `REPO_ROOT` from `${SCRIPT_DIR}/../..`
  - `PYTHONPATH` set to `${REPO_ROOT}`
- `SYMBOLS_FILE` override precedence:
  1. Environment variable `SYMBOLS_FILE`
  2. `data_selection.symbols_file` from config
  3. Fallback `symbols/Naz100_Symbols.txt`
- Transaction-cost timing now uses a `rebalance_cost_ratio` multiplier applied in the same day’s value transition:
  - `portfolio_value[j] = portfolio_value[j-1] * rebalance_cost_ratio * daily_return`

# Testing
- Ran: `uv run python -m pytest -q studies/nasdaq100_scenarios/tests`
- Result: `71 passed, 1 skipped`
- Runtime dependency sanity check:
  - `matplotlib 3.10.0`
  - `scipy 1.15.1`
- Runtime loader verification:
  - `export PYTHONPATH="$PWD" && uv run python - <<'PY' ... load_nasdaq100_window(...)`
  - Result: loader resolved `/Users/donaldpg/pyTAAA_data/Naz100/symbols/Naz100_Symbols.txt`
    and successfully loaded `(215, 7418)` data matrix.

# Follow-up Items
- Consider standardizing logging style (`logger.info` `%s` interpolation vs f-strings) if a project-wide convention is desired.
- Consider clarifying transaction-cost semantics (`per changed position` vs `per buy/sell order`) in study docs to match intended methodology.
- Optional: reconcile the skipped data-loader integration test signature and assumptions for future manual validation.
