# Experiment: Trading Lows-Highs with Delays - Implementation Plan

**Branch:** `experiment/trading-lows-highs-delays`  
**Date:** February 21, 2026  
**Status:** Planning  

## Overview

Research studies to justify the prize PyTAAA aims to capture through what-if scenarios demonstrating oracle trading with perfect/delayed information on price extrema. Uses real NASDAQ100 HDF5 data with user-specified date ranges and parameter sweeps.

## Core Concept

Generate signal2D directly from oracle knowledge of centered-window price extrema (lows/highs), then simulate portfolio histories under monthly rebalance with configurable information delays. Compare top-N selection strategies using unknowable forward returns vs delayed metrics to quantify the value of prediction accuracy.

---

## Implementation Phases

### Phase 1: Project Scaffolding and Configuration ✅

**Goal:** Create isolated study infrastructure with clear contracts

#### Tasks
- [x] Create directory structure under `studies/`
  - [x] `studies/nasdaq100_scenarios/`
  - [x] `studies/nasdaq100_scenarios/params/`
  - [x] `studies/nasdaq100_scenarios/results/`
  - [x] `studies/nasdaq100_scenarios/plots/`
  - [x] `studies/nasdaq100_scenarios/notes/`
- [x] Create JSON schema file: `studies/nasdaq100_scenarios/params/default_scenario.json`
  - [x] `start_date`, `stop_date` (string YYYY-MM-DD)
  - [x] `days_delay` (list: [0, 5, 10, 20, 40])
  - [x] `extrema_windows` (list of centered window half-widths: [25, 50, 100, 150])
  - [x] `top_n_list` (list: [5, 6, 7, 8])
  - [x] `enable_transaction_costs` (bool)
  - [x] `output_plots` (bool)
  - [x] `output_metrics` (bool)
- [x] Create `studies/nasdaq100_scenarios/README.md`
  - [x] Purpose and scope
  - [x] Dependencies on production code
  - [x] Expected outputs
  - [x] Interpretation guidelines

#### Code Review Checklist
- [x] JSON schema is well-documented with inline comments
- [x] Directory structure separates config, code, and outputs
- [x] README clearly distinguishes research from production code
- [x] No AI slop: remove placeholder text, ensure clarity

#### Tests
- [x] Verify JSON schema loads without errors
- [x] Verify all directories are created and .gitignore excludes results/plots

**Status:** Completed 2026-02-21  
**Commit:** `97cedfd` feat(studies): Phase 1 - project scaffolding for oracle delay studies

---

### Phase 2: Data Loading and Date Clipping ✅

**Goal:** Reuse HDF5 loaders with study-specific date windowing

**Commit:** `dbfdd34` feat(studies): Phase 2 - data loading with date clipping and tradability inference

#### Tasks
- [x] Create `studies/nasdaq100_scenarios/data_loader.py`
- [x] Implement `load_nasdaq100_window(params_json_path)` function
  - [x] Call `loadQuotes_fromHDF` from `functions/UpdateSymbols_inHDF5.py`
  - [x] Apply `load_quotes_for_analysis` preprocessing pattern from `functions/data_loaders.py`
  - [x] Clip `datearray` to nearest available dates for user `start_date`/`stop_date`
  - [x] Log warnings when requested dates are clamped
  - [x] Return `adjClose` (stocks × dates), `symbols`, `datearray`, `tradable_mask` (stocks × dates)
- [x] Implement `infer_tradable_mask(adjClose, datearray)` function
  - [x] Use NaN presence and infilled-data heuristics
  - [x] Return boolean mask: True where stock is tradable on that date
- [x] Add logging for date range, symbol count, and tradable coverage

#### Code Review Checklist
- [x] Function signatures use type hints
- [x] Date clipping logic handles edge cases (start > HDF range, stop < HDF range)
- [x] Logging is informative but concise
- [x] No duplicate logic from production loaders—reuse via imports
- [x] No AI slop: remove verbose comments, ensure DRY principles

#### Tests
- [x] Unit test: load with in-range dates returns expected shape
- [x] Unit test: load with out-of-range dates clamps correctly and logs warnings
- [x] Unit test: tradable mask excludes symbols with trailing NaNs
- [x] Integration test: full load completes in <5 seconds for 2-year window (skipped until HDF5 available)

---

### Phase 3: Oracle Extrema Detection ✅

**Goal:** Identify centered-window lows and highs for each stock

**Commit:** `2be8270` feat(studies): Phase 3 - oracle extrema detection and signal generation

#### Tasks
- [x] Create `studies/nasdaq100_scenarios/oracle_signals.py`
- [x] Implement `detect_centered_extrema(adjClose, window_half_width, datearray)` function
  - [x] For each stock and date, compute min/max over `[date-k, date+k]` window
  - [x] Store low/high metadata: date, price, window used
  - [x] Drop edge dates where centered window is incomplete
  - [x] Return dict: `{symbol: [(low_date, low_price, window), (high_date, high_price, window), ...]}`
- [x] Implement `generate_oracle_signal2D(extrema_dict, adjClose_shape, datearray)` function
  - [x] Binary signal: 1.0 when in low→high segment, 0.0 otherwise
  - [x] Segment defined: from low date to next high date
  - [x] Return `signal2D` (stocks × dates) matching adjClose shape
- [x] Add logging for extrema statistics (count per symbol, date coverage)
- [x] Bonus: Implemented `apply_delay()` and `generate_scenario_signals()` convenience functions

#### Code Review Checklist
- [x] Edge handling: first/last window_half_width days are correctly excluded
- [x] Signal segments do not overlap (each high terminates previous low→high)
- [x] Code is vectorized where possible (avoid nested loops on large arrays)
- [x] No AI slop: remove unnecessary intermediate variables, redundant comments

#### Tests
- [x] Unit test: detect extrema on synthetic sine wave returns correct peaks/troughs
- [x] Unit test: binary signal has expected transitions (0→1 at low, 1→0 at high)
- [x] Unit test: edge dates are 0.0 in signal2D
- [x] Integration test: extrema detection on 100 symbols × 500 days completes in <10 seconds (0.36s actual)

---

### Phase 4: Delay Operator and Scenario Generator

**Goal:** Shift signal availability by configurable delay

#### Tasks
- [ ] Add to `studies/nasdaq100_scenarios/oracle_signals.py`
- [ ] Implement `apply_delay(signal2D, days_delay, datearray)` function
  - [ ] For delay d > 0, shift signal: `signal_delayed[:, j] = signal2D[:, j-d]`
  - [ ] Fill first d days with 0.0 (no signal available)
  - [ ] Return delayed `signal2D_delayed`
- [ ] Implement `generate_scenario_grid(params)` function
  - [ ] Iterate over `days_delay` × `extrema_windows` × `top_n_list`
  - [ ] Return list of scenario dicts: `[{delay: 0, window: 25, top_n: 5, ...}, ...]`
- [ ] Add logging for scenario count and parameter ranges

#### Code Review Checklist
- [ ] Delay logic handles j-d < 0 correctly (fills with zeros)
- [ ] Scenario grid is flat list, not nested structure
- [ ] No AI slop: variable names are descriptive (avoid `temp`, `tmp`)

#### Tests
- [ ] Unit test: apply_delay(signal, 5, ...) shifts signal by 5 days
- [ ] Unit test: first 5 days of delayed signal are zeros
- [ ] Unit test: scenario grid with 4 delays, 4 windows, 4 top_n produces 64 scenarios

---

### Phase 5: Monthly Portfolio Simulator

**Goal:** Backtest with PyTAAA-style monthly rebalance

#### Tasks
- [ ] Create `studies/nasdaq100_scenarios/portfolio_backtest.py`
- [ ] Implement `simulate_monthly_portfolio(adjClose, signal2D, top_n, datearray, symbols, params)` function
  - [ ] Reuse month boundary logic from `functions/dailyBacktest.py` (`datearray[j].month != datearray[j-1].month`)
  - [ ] At rebalance date: select top N stocks where signal2D > 0
  - [ ] Equal-weight holdings: w = 1/N for selected stocks, 0 otherwise
  - [ ] Compute daily portfolio value: `value[j+1] = value[j] * sum(w[i] * gainloss[i,j])`
  - [ ] Optional transaction costs: deduct on rebalance
  - [ ] Return `portfolio_history` (1D array), `rebalance_dates`, `holdings_log`
- [ ] Implement `simulate_buy_and_hold(adjClose, datearray)` function
  - [ ] Equal-weight all stocks at start, hold until end
  - [ ] Return baseline `portfolio_history`
- [ ] Add logging for rebalance count, final value, turnover

#### Code Review Checklist
- [ ] Month boundary logic matches PyTAAA conventions exactly
- [ ] Equal-weighting is numerically stable (handles zero-signal cases)
- [ ] Transaction costs are applied consistently
- [ ] No AI slop: remove debug print statements, commented code

#### Tests
- [ ] Unit test: single rebalance at month boundary updates weights
- [ ] Unit test: buy-and-hold portfolio has constant weights
- [ ] Unit test: portfolio value >= 0 for all dates
- [ ] Integration test: simulate 2-year backtest completes in <15 seconds

---

### Phase 6: Top-N Oracle Ranking Study

**Goal:** Rank stocks by unknowable forward return with delays

#### Tasks
- [ ] Add to `studies/nasdaq100_scenarios/portfolio_backtest.py`
- [ ] Implement `compute_forward_monthly_return(adjClose, datearray, rebalance_date)` function
  - [ ] For given rebalance date, compute return to end of month
  - [ ] Return vector of returns for all stocks
- [ ] Implement `rank_by_forward_return(forward_returns, signal2D, date_idx)` function
  - [ ] Filter to stocks with signal2D > 0
  - [ ] Sort descending by forward return
  - [ ] Return indices of top N stocks
- [ ] Modify `simulate_monthly_portfolio` to accept `ranking_method` param
  - [ ] If `ranking_method == 'oracle'`, use forward returns
  - [ ] If `ranking_method == 'delayed_oracle'`, shift forward return availability by delay
  - [ ] Default to equal-weight all signal > 0 stocks (existing behavior)

#### Code Review Checklist
- [ ] Forward return calculation is point-in-time correct (no lookahead)
- [ ] Delayed ranking shifts availability, not the return values themselves
- [ ] Ranking handles ties consistently (use stable sort)
- [ ] No AI slop: consolidate similar ranking logic, avoid copy-paste

#### Tests
- [ ] Unit test: forward monthly return for Jan 1 includes all Jan trading days
- [ ] Unit test: delayed ranking with delay=5 uses forward return from 5 days prior
- [ ] Unit test: top-N selection returns exactly N stocks (or fewer if N > eligible count)

---

### Phase 7: Plotting and Output Generation

**Goal:** Multi-curve portfolio history plots and summary metrics

#### Tasks
- [ ] Create `studies/nasdaq100_scenarios/plotting.py`
- [ ] Implement `plot_portfolio_histories(scenario_results, output_path)` function
  - [ ] Reuse plotting conventions from `run_normalized_score_history.py`
  - [ ] One curve per scenario (color-coded by delay or window)
  - [ ] Include buy-and-hold baseline as reference line
  - [ ] X-axis: date, Y-axis: portfolio value (log scale optional)
  - [ ] Legend with delay/window/top_n labels
  - [ ] Save as PNG to `studies/nasdaq100_scenarios/plots/`
- [ ] Implement `generate_summary_json(scenario_results, output_path)` function
  - [ ] For each scenario: final value, CAGR, max drawdown, Sharpe ratio
  - [ ] Save as JSON to `studies/nasdaq100_scenarios/results/`
- [ ] Add parameter-sensitivity panel plots (3×3 grid: delay vs window vs top_n)

#### Code Review Checklist
- [ ] Plots are publication-quality (readable fonts, clear labels)
- [ ] JSON is human-readable (indented, sorted keys)
- [ ] Output paths use scenario identifiers (avoid generic `output.png`)
- [ ] No AI slop: remove unused imports, redundant plot calls

#### Tests
- [ ] Unit test: plot generation with 10 scenarios completes without error
- [ ] Unit test: summary JSON contains all expected keys
- [ ] Visual test: spot-check one plot matches expected curve shapes

---

### Phase 8: Documentation and Wiki Page

**Goal:** Explainer documentation for repository wiki

#### Tasks
- [ ] Create `docs/pytaaa-oracle-delay-studies.md`
- [ ] Write sections:
  - [ ] **Introduction**: What question does this answer?
  - [ ] **Methodology**: Oracle definition, delay operator, monthly rebalance
  - [ ] **Assumptions**: Survivorship bias, no slippage, simplified transaction costs
  - [ ] **Scenarios**: Table of all parameter combinations
  - [ ] **Results Interpretation**: What performance gaps mean
  - [ ] **Caveats**: Knowable vs unknowable distinction
  - [ ] **References**: Links to study code, parameter JSON, result files
- [ ] Add cross-links to study outputs in `studies/nasdaq100_scenarios/results/`
- [ ] Include example plots inline (use relative paths)

#### Code Review Checklist
- [ ] Documentation is concise but complete (avoid walls of text)
- [ ] Technical terms are defined on first use
- [ ] Links to code and outputs are correct (test in GitHub)
- [ ] No AI slop: remove marketing language, ensure scientific tone

#### Tests
- [ ] Manual test: all links in markdown resolve correctly
- [ ] Manual test: plots render correctly in GitHub preview

---

### Phase 9: Integration Testing and Final Cleanup

**Goal:** End-to-end validation and removal of temporary artifacts

#### Tasks
- [ ] Create `studies/nasdaq100_scenarios/run_full_study.py`
- [ ] Implement orchestrator that runs all phases in sequence
  - [ ] Load data
  - [ ] Detect extrema for all windows
  - [ ] Generate scenario grid
  - [ ] Run backtests for all scenarios
  - [ ] Generate plots and summary JSON
  - [ ] Log overall runtime
- [ ] Create integration test script: `studies/nasdaq100_scenarios/test_integration.sh`
  - [ ] Run study with minimal params (1 delay, 1 window, 1 top_n)
  - [ ] Verify outputs exist and are non-empty
- [ ] Clean up TODO list (see below)

#### Code Review Checklist
- [ ] Orchestrator has clear progress logging (phase start/end times)
- [ ] Error handling: early exit on data load failure, continue on plot errors
- [ ] No AI slop: remove hardcoded paths, leftover test code

#### Tests
- [ ] Integration test: full minimal study completes in <60 seconds
- [ ] Integration test: outputs match expected file count and structure
- [ ] Regression test: re-run with same params produces identical results

---

## TODO: Temporary Files and Cleanup Tracking

**Purpose:** Track code/files used during implementation that should not remain in production

### Implementation Artifacts (Remove Before Merge)
- [ ] `studies/nasdaq100_scenarios/scratch_test_loader.py` — ad-hoc data loading tests
- [ ] `studies/nasdaq100_scenarios/debug_extrema.ipynb` — Jupyter notebook for visual debugging
- [ ] `studies/nasdaq100_scenarios/results/test_run_*.json` — test outputs with dummy data
- [ ] `studies/nasdaq100_scenarios/plots/debug_*.png` — test plots during development
- [ ] Any `*.log` files in studies folder
- [ ] Any `*_temp.py`, `*_old.py`, `*_backup.py` files

### Development Data (Do Not Commit)
- [ ] `studies/nasdaq100_scenarios/params/local_test.json` — developer-specific test configs
- [ ] Any files in `studies/nasdaq100_scenarios/results/` if marked as draft outputs
- [ ] Any files in `studies/nasdaq100_scenarios/plots/` if marked as draft plots

### Final Cleanup Checklist (Before Merge Complete)
- [ ] All items in "Implementation Artifacts" section removed
- [ ] All `TODO` comments in code resolved or converted to issues
- [ ] No commented-out code blocks (except pedagogical examples in docs)
- [ ] No `print()` statements for debugging (only logging)
- [ ] No `import pdb; pdb.set_trace()` or similar debugger calls
- [ ] All test scripts have clear names and purpose (not `test123.py`)
- [ ] Git history does not contain sensitive data or large binary files
- [ ] All files in `studies/` have clear purpose (README documents structure)

---

## Verification and Acceptance Criteria

### Smoke Test
1. Clone repo and checkout `experiment/trading-lows-highs-delays` branch
2. Run: `export PYTHONPATH=$(pwd)`
3. Run: `uv run python studies/nasdaq100_scenarios/run_full_study.py --config studies/nasdaq100_scenarios/params/default_scenario.json`
4. Verify: plots generated in `studies/nasdaq100_scenarios/plots/`
5. Verify: summary JSON in `studies/nasdaq100_scenarios/results/`
6. Verify: no errors in console output

### Performance Targets
- [ ] Full study with 64 scenarios (4×4×4 grid) completes in <10 minutes
- [ ] Memory usage stays under 4GB for 5-year NASDAQ100 window
- [ ] No pandas SettingWithCopyWarning or similar runtime warnings

### Code Quality
- [ ] All functions have docstrings (numpy style)
- [ ] All modules pass `pylint` with score > 8.0
- [ ] No duplicate code blocks > 10 lines (refactor to shared functions)
- [ ] Variable names are descriptive (not `x`, `y`, `arr`)

### Documentation Quality
- [ ] README in `studies/nasdaq100_scenarios/` is complete
- [ ] Wiki page in `docs/` is publication-ready
- [ ] All plots have captions and axis labels
- [ ] JSON schema is documented with inline comments

---

## Code Review Protocol

After each phase completion:

1. **Self-Review**
   - Run all phase-specific tests
   - Read through code as if reviewing someone else's PR
   - Check for AI slop: generic comments, placeholder logic, verbose naming

2. **Constructive Critic Persona**
   - "Does this code do one thing well?"
   - "Can a new contributor understand this in 5 minutes?"
   - "Are there edge cases not covered by tests?"
   - "Is this the simplest implementation that works?"

3. **Cleanup Actions**
   - Remove comments that restate the code
   - Consolidate similar logic into helper functions
   - Replace magic numbers with named constants
   - Add type hints where missing

4. **Documentation Sync**
   - Ensure docstrings match current function signatures
   - Update README if phase changed directory structure
   - Add inline comments only for non-obvious *why* (not *what*)

---

## Success Metrics

### Research Outcomes
- [ ] Quantified performance gap: perfect oracle vs 20-day delay
- [ ] Identified optimal window size for extrema detection
- [ ] Demonstrated top-N sensitivity to ranking delay
- [ ] Generated 10+ publication-quality plots for wiki

### Code Quality Outcomes
- [ ] Zero duplication of production code (all via imports)
- [ ] All study code under `studies/` (no mixing with production)
- [ ] All tests pass with pytest
- [ ] No TODOs remain in merged code

### Knowledge Transfer Outcomes
- [ ] Wiki page is self-contained (no external dependencies to understand)
- [ ] Another developer can reproduce results from README alone
- [ ] Study design is defensible (clear assumptions, no obvious biases)

---

## Timeline Estimate

**Total Estimated Time:** 20-30 hours across 2-3 weeks

- Phase 1: 2 hours
- Phase 2: 3 hours
- Phase 3: 4 hours
- Phase 4: 2 hours
- Phase 5: 5 hours
- Phase 6: 3 hours
- Phase 7: 4 hours
- Phase 8: 3 hours
- Phase 9: 4 hours
- Code reviews and cleanup: 5 hours (distributed)

---

## Notes and Decisions

### Decision Log
1. **2026-02-21**: Use NASDAQ100 HDF5 (not synthetic data) for realism
2. **2026-02-21**: Clamp dates instead of failing (more user-friendly)
3. **2026-02-21**: Monthly rebalance only (daily deferred to future work)
4. **2026-02-21**: Binary signal (not slope strength) for clarity
5. **2026-02-21**: Forward monthly return as oracle ranking metric

### Open Questions
- Should we include Sharpe-based oracle ranking alternative?
- Should edge-excluded dates be marked in output or silently dropped?
- Should transaction costs scale with portfolio value or be flat per trade?

### Future Enhancements (Out of Scope for Phase 1)
- Daily rebalance mode for comparison
- Synthetic data generator for controlled experiments
- Parameter optimization via grid search
- Comparison to actual PyTAAA method performance
- Multi-universe support (SP500, Russell 2000)

---

## Contact and Support

For questions about this implementation plan:
- See existing patterns in `functions/` for PyTAAA conventions
- Refer to `.github/copilot-instructions.md` for code style
- Check `docs/ARCHITECTURE.md` for data structure details
