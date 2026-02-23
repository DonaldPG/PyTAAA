# Look-Ahead Bias Testing for PyTAAA

## Purpose

This test harness validates that future stock prices do not influence stock
selection decisions. If the same stocks are consistently selected when future
prices are modified, we can infer the selection logic is causally sound
(uses only past data).

## How It Works

1. **Create baseline**: Use real HDF5 data to determine which stocks the model
   selects for a test date.
2. **Patch future prices**: Create modified copies of the HDF5 file with
   significant price changes **only after** the test date.
3. **Rerun selection**: Extract stocks selected by the model using the
   patched data.
4. **Compare**: If selections are identical despite price changes, there is
   no look-ahead bias.

## Files

- `hdf5_utils.py` — Safe copying and patching of HDF5 data
- `patch_strategies.py` — Reusable price perturbation functions
  - `step_down(magnitude)` — sudden downward step
  - `step_up(magnitude)` — sudden upward step
  - `linear_down(slope)` — gradual downward trend
  - `linear_up(slope)` — gradual upward trend
- `selection_runner.py` — Extract ranked stocks for a given date
- `experiment_future_prices.py` — Main experiment harness (Phase 2)
- `plot_results.py` — Visualize portfolio performance (Phase 2)
- `evaluate_future_prices.py` — Report pass/fail results (Phase 2)

## Phase 1 Tests

All Phase 1 infrastructure tests pass:
- ✓ All modules import without errors
- ✓ All perturbation strategies work correctly
- ✓ Directory structure is complete

## Phase 2 Implementation

See `../../plans/stock_selection_testing_plan.md` for full Phase 2 tasks.
