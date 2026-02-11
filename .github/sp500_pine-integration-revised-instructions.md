---
title: sp500_pine integration - critique and revised plan
---

# Constructive critique and revised implementation plan

Summary: This document provides a constructive critique of the existing
`.github/sp500_pine-integration-instructions.md` plan, highlights what is
good, points out missing or ambiguous items, and delivers a self-contained,
actionable revised plan that a new agentic Copilot session can follow to
implement `sp500_pine` as a sixth stock-trading model.

**What is good about the existing plan**
- It enumerates concrete integration touchpoints (`recommend_model.py`,
  Monte Carlo runner, JSON manifests, docs, tests).
- It includes sensible validation and backup steps for manifest edits.
- It points out keeping `run_monte_carlo.sh` as a thin wrapper and
  focusing on the Python runner — correct direction.

**Main issues / omissions to fix**
- Missing precise file edits and exact JSON key to add (so an automated
  agent can make deterministic edits).
- No strict description of expected model JSON schema or required keys.
- Tests are suggested but no exact test locations, filenames, or commands.
- Lack of sample CLI commands to smoke-test the change.
- Ambiguous responsibilities between per-dataset manifest edits and
  central manifest updates — should be explicit and ordered.

--

**Revised Plan — goals (one line)**
Add `sp500_pine` as the sixth stock model (alongside
`["cash","naz100_pine","naz100_hma","naz100_pi","sp500_hma"]`),
register it in project manifests, allow `run_monte_carlo.py` to target it,
update `recommend_model.py` selection, and add tests/docs/backups.

Major steps (up to 5) — each with up to 5 substeps and explicit file edits.

 - [x] Major Step 1 — Add and verify model JSON (canonicalize path)
  - [x] Verify the source model JSON exists at `/Users/donaldpg/pytaaa_data/sp500_pine/pytaaa_sp500_pine.json` (development copy added to repo under `pyTAAA_data/sp500_pine/pytaaa_sp500_pine.json`).
  - [x] Create backup/canonical copy in repo at `pyTAAA_data/sp500_pine/pytaaa_sp500_pine.json` and ensure it's readable by `uv run python`.
  - [x] Validate minimal model JSON schema (required keys): `name`, `id` (== `sp500_pine`), `data_store` (optional path), and any model-specific params. Fail early if missing.
  - [ ] Update per-dataset manifest: add `sp500_pine` mapping to `/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json` (create a `.backup` first).
  - [x] Add a tiny loader script `scripts/validate_model_json.py` that loads a JSON path and asserts required keys.

 - [x] Major Step 2 — Register model in central manifest(s)
  - [x] Edit `abacus_combined_PyTAAA_status.params.json` at key `models.model_choices` and add the line (backup created):

    "sp500_pine": "{base_folder}/sp500_pine/data_store/{data_file}",

    (Place under the existing keys; maintain JSON formatting.)
  - [x] Create a backup of `abacus_combined_PyTAAA_status.params.json` as `*.backup` preserving timestamp. Backup created in repo: `abacus_combined_PyTAAA_status.params.json.backup`.
  - [ ] If you use `pytaaa_model_switching_params.json`, also add the same mapping there (or ensure it references `abacus_combined...` JSON consistently).
  - [ ] Run the `scripts/validate_model_json.py` for the new mapping to verify the resolved file exists.

 - [x] Major Step 3 — Update Python Monte Carlo runner (exact edits)
  - [x] File to edit: `run_monte_carlo.py` (no new CLI options). Do NOT change entrypoint options — keep `run_monte_carlo.sh` and `run_monte_carlo.py` usage exactly as-is. The Python runner now reads `models.model_choices` from the JSON and `sp500_pine` is discovered from the manifest.
  - [x] Validation logic: after `config = json.load(f)` the code loads `models.model_choices` when `--json` is provided. If a mapping entry results in a missing data file, the runner logs a warning and keeps the mapping so `MonteCarloBacktest` can handle missing data rather than crash.
  - [x] Ensure `MonteCarloBacktest` is invoked with `model_paths=model_choices` (existing call already supports this) and that outputs include the model name.
  - [x] Logging at `INFO` level shows resolved model paths and which models will be executed. `run_monte_carlo.sh` usage remains unchanged and compatible.

    ```bash
    ./run_monte_carlo.sh 30 explore-exploit --reset --randomize \
        --json=abacus_combined_PyTAAA_status.params.json \
        --fp-year-min=1995 --fp-year-max=2021 --fp-duration=5
    ```

 - [ ] Major Step 4 — Integrate into monthly selection (`recommend_model.py`) and other entry points (in-progress: recommend_model compatibility verified, plotting updated)
  - [x] Inspect `recommend_model.py` and find where model list or model IDs are enumerated. Replace any hard-coded list with dynamic reading from `abacus_combined_PyTAAA_status.params.json` (use `GetParams.get_models()` if available) or add `sp500_pine` consistently.
  - [ ] Add safe fallback: if manifest does not list `sp500_pine`, skip with logged warning instead of crashing.
  - [ ] Ensure any web page generation (`functions/WriteWebPage_pi.py`) and `scheduler.py` are tolerant of the new model (search code for explicit model lists).
  - [ ] Add a unit test `tests/test_sp500_pine_registration.py` that loads the central manifest and asserts `sp500_pine` is present and resolves to an accessible path.
  - [x] Update plotting and legend generation to include `sp500_pine` wherever plots compare base models and Monte Carlo best runs.
    - [x] Files to check/update: `functions/MonteCarloBacktest.py` (plot creation functions such as `create_monte_carlo_plot`), `functions/MakeValuePlot.py`, any daily-update plotters (e.g., `pytaaa_main.py`, `pytaaa_quotes_update.py`, or `functions/WriteWebPage_pi.py`).
    - [x] Ensure the top subplot legend includes `sp500_pine` with a distinct color and line style matching other models.
    - [x] Ensure the bottom subplot (selected-model timeline) maps the `sp500_pine` selections to the correct y-axis value and legend marker (match existing scheme: one discrete y-value per model).
    - [ ] Update any color/marker dictionaries or ordering arrays so `sp500_pine` appears consistently across plots and web output (partial: `MonteCarloBacktest` updated; other files remain).
    - [ ] Add a visual smoke test: run a short Monte Carlo or daily update that generates the plot and verify the legend shows `sp500_pine` and selection subplot includes its markers.
    - [ ] Files to check/update: `functions/MonteCarloBacktest.py` (plot creation functions such as `create_monte_carlo_plot`), `functions/MakeValuePlot.py`, any daily-update plotters (e.g., `pytaaa_main.py`, `pytaaa_quotes_update.py`, or `functions/WriteWebPage_pi.py`).
    - [ ] Ensure the top subplot legend includes `sp500_pine` with a distinct color and line style matching other models.
    - [ ] Ensure the bottom subplot (selected-model timeline) maps the `sp500_pine` selections to the correct y-axis value and legend marker (match existing scheme: one discrete y-value per model).
    - [ ] Update any color/marker dictionaries or ordering arrays so `sp500_pine` appears consistently across plots and web output.
    - [ ] Add a visual smoke test: run a short Monte Carlo or daily update that generates the plot and verify the legend shows `sp500_pine` and selection subplot includes its markers.

- [ ] Major Step 5 — Tests, docs, backups, smoke-run
  - [ ] Tests to add:
    - `tests/test_sp500_pine_json_schema.py` — validate required keys in `pytaaa_sp500_pine.json` using the loader script.
    - `tests/test_run_monte_carlo_model_flag.py` — run `run_monte_carlo.py` programmatically with `--model-id sp500_pine` and `iterations`/`max_iterations` lowered for a quick smoke test (mock or set `max_iterations` via a temp config override).
  - [ ] Update `README.md` with one paragraph: purpose of `sp500_pine`, how to run Monte Carlo for it, and example command:

    ```bash
    uv run python run_monte_carlo.py --model-id sp500_pine --json abacus_combined_PyTAAA_status.params.json --randomize --fp-duration=1
    ```

  - [ ] Create backups for any modified JSON files (append `.backup.YYYYMMDDHHMM`), commit changes with conventional commits (e.g., `feat(models): add sp500_pine mapping`), and push.
  - [ ] Smoke verification steps (commands to run locally):

    ```bash
    uv run python scripts/validate_model_json.py /Users/donaldpg/pytaaa_data/sp500_pine/pytaaa_sp500_pine.json
    uv run python run_monte_carlo.py --model-id sp500_pine --json abacus_combined_PyTAAA_status.params.json --randomize --fp-duration=1
    ```

  - [ ] If smoke test passes, run full test suite: `uv run python -m pytest -q`.

Implementation notes for a new agentic Copilot session (self-contained checklist)
- Files the agent must edit (exact paths):
  - `run_monte_carlo.py` — add `--model-id` and `--model-json-path` click options and resolution logic.
  - `abacus_combined_PyTAAA_status.params.json` — add `"sp500_pine": "{base_folder}/sp500_pine/data_store/{data_file}"` under `models.model_choices`.
  - `/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json` — add per-dataset mapping entry for `sp500_pine` (create backup).
  - `recommend_model.py` — ensure model enumeration reads manifests or includes `sp500_pine`.
  - `tests/` — add two test files described above.
  - `scripts/validate_model_json.py` — small utility to assert required keys.

- Required validation rules for model JSON (`pytaaa_sp500_pine.json`):
  - Top-level `id` (string) equals `sp500_pine`.
  - Optional `data_store` (string): path to folder or file where `pytaaa_status.params` or `pytaaaweb_backtestPortfolioValue.params` resides. If absent, the agent should assume `{parent_dir}/data_store/{data_file}`.
  - Basic `params` dictionary is acceptable (no strict schema beyond `id` required for first pass).

- Expected CLI usage examples (for smoke tests):
  - Quick validate: `uv run python scripts/validate_model_json.py /Users/donaldpg/pytaaa_data/sp500_pine/pytaaa_sp500_pine.json`
  - Smoke run: `uv run python run_monte_carlo.py --model-id sp500_pine --json abacus_combined_PyTAAA_status.params.json --randomize --fp-duration=1`

- Commit and review guidelines for the agent:
  - Create backups of any JSON before editing. Use `git add`/`git commit -m "feat(models): add sp500_pine mapping"`.
  - Keep changes minimal and isolated per file; add unit tests in `tests/` and run `pytest` locally.

Appendix — Minimal code snippets (drop-in)

1) Click options to add near other `@click.option` definitions in `run_monte_carlo.py`:

```python
@click.option('--model-id', type=str, default=None, help='Model id to run (e.g., sp500_pine)')
@click.option('--model-json-path', type=str, default=None, help='Direct path to the model JSON (overrides templates)')
```

2) Resolution logic (insert after `config = json.load(f)` and before building `model_choices`):

```python
if model_id:
    resolved_path = None
    # Prefer JSON templates when available
    if 'models' in config and model_id in config['models'].get('model_choices', {}):
        template = config['models']['model_choices'][model_id]
        base_folder = config['models'].get('base_folder', base_folder)
        data_file = data_files.get(data_format, 'pytaaaweb_backtestPortfolioValue.params')
        resolved_path = template.format(base_folder=base_folder, data_file=data_file)
    elif model_json_path:
        if not os.path.exists(model_json_path):
            raise FileNotFoundError(f"Model JSON not found: {model_json_path}")
        with open(model_json_path,'r') as mf:
            mjson = json.load(mf)
        # optionally read `data_store` key
        data_store = mjson.get('data_store')
        if data_store:
            resolved_path = os.path.join(data_store, data_files[data_format])
        else:
            resolved_path = os.path.join(os.path.dirname(model_json_path), 'data_store', data_files[data_format])
    else:
        raise ValueError('Provide --model-json-path or add model to JSON templates')
    model_choices = { model_id: resolved_path }
```

3) JSON mapping to add to `abacus_combined_PyTAAA_status.params.json` (exact location: under `models.model_choices`):

```json
  "sp500_pine": "{base_folder}/sp500_pine/data_store/{data_file}"
```

--

If you'd like, I can now:
- implement the `run_monte_carlo.py` changes and tests (small patch), or
- add the JSON mapping to `abacus_combined_PyTAAA_status.params.json` and update the per-dataset manifest (create backups first).

Choose which to do next and I'll proceed. 
