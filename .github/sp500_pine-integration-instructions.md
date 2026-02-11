---
title: Add sp500_pine model to Abacus - integration instructions
---

# Add `sp500_pine` model to Abacus (integration instructions)

## Overview

Add the new stock trading model `sp500_pine` (source JSON: `/Users/donaldpg/pytaaa_data/sp500_pine/pytaaa_sp500_pine.json`) so it is selectable in monthly re‑balancing, included in Monte Carlo backtests, and registered in project manifests and docs.

Note: This change adds a sixth stock trading model to the existing set of models currently registered in the project: ["cash", "naz100_pine", "naz100_hma", "naz100_pi", "sp500_hma"]. The goal is to register `sp500_pine` alongside these five, not to replace or rework them.

## Major steps

- [ ] Major Step 1 — Add model JSON and canonicalize path
  - [ ] Verify source JSON exists and is complete at `/Users/donaldpg/pytaaa_data/sp500_pine/pytaaa_sp500_pine.json`.
  - [ ] Decide whether to copy into the repo data folder or reference the external path.
  - [ ] If copying, add file under a canonical repo location (eg. `data/models/` or `pyTAAA_data/`) or create a gitignored symlink.
  - [ ] Validate JSON schema matches existing models (fields, names, params).
  - [ ] Add a minimal loader/check script to assert the model is readable.
  - [ ] Update any per-dataset manifest mapping (example: `/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json`) to include an entry for `sp500_pine` and create a backup.

- [ ] Major Step 2 — Integrate into monthly selection (`recommend_model.py`)
  - [ ] Inspect `recommend_model.py` to find where model IDs are discovered or enumerated.
  - [ ] Add the model identifier `sp500_pine` to the list/registry used for monthly selection.
  - [ ] Ensure the selection code can resolve the JSON path (absolute or repo-relative).
  - [ ] Add error handling and logging when the model cannot be loaded.
  - [ ] Add a unit test that exercises selection including `sp500_pine`.

- [ ] Major Step 3 — Monte Carlo & backtesting integration
  - [ ] Update the Python backtest runner invoked by `run_monte_carlo.sh` to accept and run `sp500_pine` (preferred over editing the shell wrapper).
    - [ ] Identify the Python entry point called by `run_monte_carlo.sh` (for example `run_monte_carlo.py`, or the function in `functions/MonteCarloBacktest.py`).
    - [ ] Add argument/option parsing to accept a `model_id` (e.g., `sp500_pine`) and a JSON path override.
    - [ ] Resolve the `sp500_pine` JSON path (repo-relative or absolute), load and validate the model JSON, and construct the model params object expected by the backtest engine.
    - [ ] Ensure the runner passes model params into the Monte Carlo engine, sets output filenames (include model name), and applies any model-specific backtest settings.
    - [ ] Add logging and clear error messages for missing/invalid model JSON and fallback behavior.
    - [ ] Add a small integration test or smoke-run that executes the runner for `sp500_pine` with a minimal iteration count.
  - [ ] Keep `run_monte_carlo.sh` as a thin wrapper: forward CLI args to the Python runner and document usage.

- [ ] Major Step 4 — Update combined params and other JSON manifests
  - [ ] Edit `abacus_combined_PyTAAA_status.params.json` to reference `sp500_pine` where applicable.
  - [ ] Create backups and bump version or timestamp fields in modified manifests.
  - [ ] Run code paths that load those manifests to detect schema mismatches early.
  - [ ] Add a short script to validate all project manifests for the new model.

- [ ] Major Step 5 — Docs, tests, and QA
  - [ ] Update `README.md` with a short section describing `sp500_pine` and how to run/backtest it.
  - [ ] Add/extend tests in `tests/` to cover model loading, selection, and a tiny backtest for `sp500_pine`.
  - [ ] Update any run scripts, CI configs, or `pyproject.toml` notes if new dependencies are needed.
  - [ ] Run an end‑to‑end smoke verification locally (selection → backtest → verify outputs).
  - [ ] Add helpful log messages and failure hints when `sp500_pine` fails to load.

## Recommendations / Additional checks

- Search the codebase for any canonical model lists or enumerations (models, names, IDs) and update them.
- Check `pytaaa_model_switching_params.json` and other project JSONs for enumerations of models; keep them consistent.
- Ensure path strategy is consistent across scheduler, backtests, and any webpage generation (eg. `functions/WriteWebPage_pi.py`).
- Consider adding a tiny CI job or GitHub Actions workflow that validates model JSON load + a one-iteration backtest to prevent regressions.
- Add clear logging when loading user model files so missing/invalid JSONs are obvious.

## Questions / Next steps

- Do you want me to copy the source JSON into the repo (recommended for reproducibility), or keep an external absolute path reference?
- If you want, I can create the required code patches and tests for `recommend_model.py`, `run_monte_carlo.sh`, and the manifest updates.

---
Generated: sp500_pine integration checklist
