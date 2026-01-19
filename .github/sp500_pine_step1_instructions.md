Major Step 1 — sp500_pine model JSON placement and external manifest update

What I added in the repo:

- `scripts/validate_model_json.py` — small CLI validator for minimal model JSON keys.
- `pyTAAA_data/sp500_pine/pytaaa_sp500_pine.json` — canonical, local copy of the model JSON for development.

Why this matters:

- The original source model JSON path you specified is outside the repository
  (`/Users/donaldpg/pytaaa_data/...`). I cannot edit files outside the workspace.
- To continue, either copy the external JSON into the repo path above, or let
  the Monte Carlo runner use the absolute path.

Instructions to update external per-dataset manifest (manual steps):

1. Backup the external manifest (run on your machine):

```bash
cp /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json \
   /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json.backup.$(date +%Y%m%d%H%M)
```

2. Edit the manifest and add the following key under `models.model_choices` (maintain JSON syntax and commas):

```json
"sp500_pine": "{base_folder}/sp500_pine/data_store/{data_file}"
```

3. Run the validator against the canonical copy in this repo (or the original):

```bash
uv run python scripts/validate_model_json.py pyTAAA_data/sp500_pine/pytaaa_sp500_pine.json
```

If you want me to proceed with editing a manifest file inside the repository, point me to its path or place a copy under the workspace and I will backup and patch it automatically.
