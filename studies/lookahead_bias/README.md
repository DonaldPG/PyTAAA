# Look-Ahead Bias Testing for PyTAAA

## Purpose

This study empirically verifies that the PyTAAA stock selection pipeline
(`computeSignal2D` + `sharpeWeightedRank_2D`) does **not** exhibit look-ahead
bias — i.e. future price data does not influence which stocks are selected on
any past date.

## How It Works

1. **Load real HDF5 data** via the production `load_quotes_for_analysis()` path.
2. **Slice** to a narrow window: 600 trading days before the cutoff and 200
   after (≈800 total). This is sufficient for all rolling windows while
   running ~11× faster than the full dataset.
3. **Patch prices in memory**: build a copy of `adjClose` where post-cutoff
   prices are dramatically altered (top-half performers stepped down 40%,
   bottom-half stepped up 40%). No files are written.
4. **Run pipeline on both** original and patched `adjClose` arrays.
5. **Compare selections at the cutoff date**:
   - `[CUT ]` row identical → no look-ahead bias ✅
   - `[CUT ]` row differs → look-ahead bias detected ❌

## Files

- `run_lookahead_study.py` — Main CLI study script (human-readable output)
- `hdf5_utils.py` — HDF5 copy/patch utilities (original Phase 1 scaffolding)
- `patch_strategies.py` — Price perturbation callables (`step_down`,
  `step_up`, `linear_down`, `linear_up`)
- `selection_runner.py` — Thin wrapper for `get_ranked_stocks`
- `experiment_future_prices.py`, `plot_results.py`, `evaluate_future_prices.py`
  — Original Phase 2 scaffolding (superseded by `run_lookahead_study.py`)

## Running the Study

```bash
cd /path/to/PyTAAA

# Default: single cutoff 200 days from end of dataset
PYTHONPATH=$(pwd) uv run python studies/lookahead_bias/run_lookahead_study.py

# Three specific cutoff dates:
PYTHONPATH=$(pwd) uv run python studies/lookahead_bias/run_lookahead_study.py \
    --cutoff-date 2023-09-29 2024-03-29 2024-09-27
```

## Running the Pytest Regression Test

```bash
PYTHONPATH=$(pwd) uv run pytest tests/test_lookahead_bias.py -v
# Runtime: ~20 seconds for all 3 models
```

## Pass Criteria

- `[CUT ]` row shows identical ORIG and PATCH selections for all models and
  all tested cutoff dates.
- `*** DIFFER ***` may appear on `[POST]` rows (expected — patched prices
  diverge after the cutoff).
- pytest: `test_selection_consistency_across_models` passes for `naz100_hma`,
  `naz100_pine`, and `naz100_pi`.

## Results (2026-02-23)

All three models tested at multiple cutoff dates: **no look-ahead bias
detected**.  `test_selection_consistency_across_models` PASSES for all three
models.

## Phase 2 Implementation

See `../../plans/stock_selection_testing_plan.md` for full Phase 2 tasks.
